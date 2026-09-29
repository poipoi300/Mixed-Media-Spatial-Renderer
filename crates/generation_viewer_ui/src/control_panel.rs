//! The server-driven control panel pill.
//!
//! The viewer does not know what any control means. A server sends a
//! [`ControlPanel`] — a tree of widgets, a list of stat lines, a summary —
//! and this module renders it, collects values, and hands a submission back
//! to the loader, which sends every value to the server on each request.
//!
//! Two pieces of state are deliberately the viewer's own and never travel to
//! the server: the cutaway slice depth, which is a property of how the scene
//! is drawn rather than of the data, and which groups are collapsed, which is
//! pure display.
//!
//! **Widget lifetime.** The pill's shell (header, stats, slice-depth row, the
//! pooled dropdown menu) is spawned once and lives forever. The widgets
//! inside it are despawned and respawned when — and only when — the panel's
//! `revision` changes, which a server bumps only for a genuine structural
//! change. Everything that happens during normal use (values, labels, stats,
//! errors, collapsing a group) leaves the entities alone, so the interactive
//! path never despawns anything a click could be in flight to.

use std::collections::{BTreeMap, HashSet};

use bevy::prelude::*;
use bevy::ui::{FocusPolicy, RelativeCursorPosition};

use generation_api::{ControlPanel, ControlValue, ControlValues, ControlWidget};

use crate::{
    button_color, display_if, expanded_panel, header_button_color, pill_node, set_button_color,
    set_header_color, spawn_button_row, spawn_pill_button, spawn_text, spawn_title,
    truncate_ui_label, ViewerUiButton, ViewerUiPanel, ViewerUiText,
};

/// Option rows the dropdown menu can show at once.
pub(crate) const CONTROL_DROPDOWN_VISIBLE_OPTIONS: usize = 9;
const CONTROL_FILTER_MAX_CHARS: usize = 24;
const CONTROL_TEXT_MAX_CHARS: usize = 64;
/// Indent per group nesting level, in pixels.
const GROUP_INDENT: f32 = 10.0;
/// Slice depth granularity and cap, in coordinate cells (multiples of the
/// projection's coordinate spacing).
const CONTROL_SLICE_STEP_CELLS: f32 = 0.5;
const CONTROL_SLICE_MAX_CELLS: f32 = 24.0;
/// Stat lines the pill reserves room for, so the widgets below never move as
/// a server's numbers change.
const CONTROL_STAT_LINES: usize = 6;

/// What a rebuilt widget's button does when pressed. Carries the control id
/// rather than an index, so a rebuild cannot leave a stale row pointing at a
/// different control.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlAction {
    /// A button widget: submit, with this control named as the activated one.
    Activate,
    /// A select widget: open or close its option dropdown.
    ToggleDropdown,
    ToggleBool,
    SliderDrag,
    FocusText,
    ToggleGroup,
}

/// A button belonging to a server-described widget.
#[derive(Component, Debug, Clone)]
pub struct ControlWidgetButton {
    pub control_id: String,
    pub action: ControlAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlTextRole {
    /// The widget's name.
    Label,
    /// Its current value, as text.
    Value,
    /// A checkbox's mark.
    Mark,
    /// The server's secondary line.
    Detail,
}

#[derive(Component, Debug, Clone)]
pub struct ControlWidgetText {
    pub control_id: String,
    pub role: ControlTextRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlWidgetPanelRole {
    /// A group's children, hidden while the group is collapsed.
    GroupChildren,
    /// A slider's filled portion.
    SliderFill,
}

#[derive(Component, Debug, Clone)]
pub struct ControlWidgetPanel {
    pub control_id: String,
    pub role: ControlWidgetPanelRole,
}

/// Marks the node the server's widgets are spawned into.
#[derive(Component)]
pub struct ControlWidgetRoot;

/// What currently owns the keyboard and mouse wheel.
///
/// One enum rather than independent flags, because Escape, the wheel, and
/// WASD each have to respect every mode: with two booleans, typing into a
/// text field would still fly the camera.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum FocusOwner {
    #[default]
    None,
    Dropdown(String),
    TextField(String),
}

/// A submission waiting for the viewer's loader to turn into a request.
#[derive(Debug, Clone)]
pub struct PendingSubmit {
    pub values: ControlValues,
    /// The control the user interacted with, which is what tells a server a
    /// button was pressed rather than a value merely changed.
    pub activated: Option<String>,
}

#[derive(Resource)]
pub struct ControlPanelState {
    panel: ControlPanel,
    values: ControlValues,
    /// Controls edited since the last submission. Non-submitting widgets
    /// accumulate here until something submits, and the pill marks them so
    /// an unsent form is never mistaken for applied state.
    edited: HashSet<String>,
    pending_submit: Option<PendingSubmit>,
    focus: FocusOwner,
    dropdown_start: usize,
    filter: String,
    text_buffer: String,
    /// Slider being dragged, submitted when the drag ends rather than on
    /// every cursor sample, so one drag is one request.
    dragging_slider: Option<String>,
    collapsed_groups: HashSet<String>,
    /// Control values from `--control`, applied over the first panel a server
    /// sends so a scripted run starts where it asked to.
    initial_values: ControlValues,
    initial_values_applied: bool,
    unknown_controls_reported: bool,
    /// Revision the spawned widget entities were built for; `None` until the
    /// first build. Drives the despawn/respawn in [`rebuild_control_widgets`].
    rendered_revision: Option<u64>,
    pub pill_expanded: bool,
    pub projected_points: usize,
    pub projected_total: usize,
    /// A failed request, as opposed to the server rejecting the values it was
    /// sent (which arrives as `panel.error`).
    pub last_error: Option<String>,
    slice_depth_cells: f32,
}

impl Default for ControlPanelState {
    fn default() -> Self {
        Self::new(ControlValues::new())
    }
}

impl ControlPanelState {
    pub fn new(initial_values: ControlValues) -> Self {
        Self {
            panel: ControlPanel::default(),
            values: ControlValues::new(),
            edited: HashSet::new(),
            pending_submit: None,
            focus: FocusOwner::None,
            dropdown_start: 0,
            filter: String::new(),
            text_buffer: String::new(),
            dragging_slider: None,
            collapsed_groups: HashSet::new(),
            initial_values,
            initial_values_applied: false,
            unknown_controls_reported: false,
            rendered_revision: None,
            pill_expanded: false,
            projected_points: 0,
            projected_total: 0,
            last_error: None,
            slice_depth_cells: 0.0,
        }
    }

    pub fn panel(&self) -> &ControlPanel {
        &self.panel
    }

    pub fn values(&self) -> &ControlValues {
        &self.values
    }

    /// Control ids `--control` named that no server panel has defined. Empty
    /// until the first panel arrives.
    /// Reported once, not per snapshot: a streaming load delivers many, and
    /// the same warning on each would bury the rest of the output.
    pub fn take_unknown_initial_controls(&mut self) -> Vec<String> {
        if !self.initial_values_applied || self.unknown_controls_reported {
            return Vec::new();
        }
        self.unknown_controls_reported = true;
        self.initial_values
            .keys()
            .filter(|id| self.panel.find(id).is_none())
            .cloned()
            .collect()
    }

    /// Adopts the panel a snapshot carried.
    ///
    /// Server values win except where the user has an unsubmitted edit, so a
    /// streaming snapshot arriving mid-typing cannot overwrite what is being
    /// typed.
    pub fn apply_panel(&mut self, panel: ControlPanel, points: usize, total: usize) {
        for (control_id, value) in panel.values() {
            if self.edited.contains(&control_id) {
                continue;
            }
            self.values.insert(control_id, value);
        }
        if !self.initial_values_applied {
            for (control_id, value) in &self.initial_values {
                self.values.insert(control_id.clone(), value.clone());
            }
            self.initial_values_applied = true;
        }
        // A control the server dropped must not keep sending its old value.
        self.values.retain(|id, _| panel.find(id).is_some());
        self.edited.retain(|id| panel.find(id).is_some());
        // Every piece of state naming a control id is reconciled here, so
        // none can outlive the control it points at.
        if matches!(&self.focus, FocusOwner::Dropdown(id) | FocusOwner::TextField(id) if panel.find(id).is_none())
        {
            self.clear_focus();
        }
        if self
            .dragging_slider
            .as_ref()
            .is_some_and(|id| panel.find(id).is_none())
        {
            self.dragging_slider = None;
        }
        self.panel = panel;
        self.projected_points = points;
        self.projected_total = total;
        self.last_error = None;
    }

    pub fn set_request_error(&mut self, error: impl Into<String>) {
        self.last_error = Some(error.into());
    }

    /// The error to show: a failed request first, else the server rejecting
    /// the values it was sent.
    pub fn error(&self) -> Option<&str> {
        self.last_error.as_deref().or(self.panel.error.as_deref())
    }

    pub fn take_pending_submit(&mut self) -> Option<PendingSubmit> {
        self.pending_submit.take()
    }

    /// Queues a submission carrying every current value. Used by the widget
    /// handlers and by scripted runs.
    pub fn submit(&mut self, activated: Option<String>) {
        self.pending_submit = Some(PendingSubmit {
            values: self.values.clone(),
            activated,
        });
        self.edited.clear();
        self.last_error = None;
        self.clear_focus();
    }

    /// Records a value, then submits if the widget declared that it submits.
    pub fn set_value(&mut self, control_id: &str, value: ControlValue) {
        if self.values.get(control_id) == Some(&value) {
            return;
        }
        self.values.insert(control_id.to_owned(), value);
        let submits = self
            .panel
            .find(control_id)
            .is_some_and(ControlWidget::submits);
        if submits {
            self.submit(Some(control_id.to_owned()));
        } else {
            self.edited.insert(control_id.to_owned());
        }
    }

    pub fn value(&self, control_id: &str) -> Option<&ControlValue> {
        self.values.get(control_id)
    }

    /// Whether this control has an edit that has not been sent yet.
    pub fn is_edited(&self, control_id: &str) -> bool {
        self.edited.contains(control_id)
    }

    pub fn focus(&self) -> &FocusOwner {
        &self.focus
    }

    /// Whether the keyboard and wheel belong to the panel, so camera controls
    /// stand down.
    pub fn input_focused(&self) -> bool {
        self.focus != FocusOwner::None
    }

    pub fn clear_focus(&mut self) {
        self.focus = FocusOwner::None;
        self.filter.clear();
        self.text_buffer.clear();
        self.dropdown_start = 0;
    }

    pub fn dropdown_active(&self) -> bool {
        matches!(self.focus, FocusOwner::Dropdown(_))
    }

    fn open_dropdown_id(&self) -> Option<&str> {
        match &self.focus {
            FocusOwner::Dropdown(control_id) => Some(control_id),
            _ => None,
        }
    }

    fn focused_text_id(&self) -> Option<&str> {
        match &self.focus {
            FocusOwner::TextField(control_id) => Some(control_id),
            _ => None,
        }
    }

    pub(crate) fn toggle_pill(&mut self) {
        self.pill_expanded = !self.pill_expanded;
        if !self.pill_expanded {
            self.clear_focus();
        }
    }

    pub(crate) fn toggle_dropdown(&mut self, control_id: &str) {
        if self.open_dropdown_id() == Some(control_id) {
            self.clear_focus();
            return;
        }
        self.clear_focus();
        self.focus = FocusOwner::Dropdown(control_id.to_owned());
        self.center_dropdown_on_selection();
    }

    pub(crate) fn focus_text(&mut self, control_id: &str) {
        if self.focused_text_id() == Some(control_id) {
            self.clear_focus();
            return;
        }
        let existing = self
            .values
            .get(control_id)
            .map(ControlValue::as_text)
            .unwrap_or_default();
        self.clear_focus();
        self.text_buffer = existing;
        self.focus = FocusOwner::TextField(control_id.to_owned());
    }

    /// Commits the focused text field's buffer, submitting if it submits.
    pub(crate) fn commit_text(&mut self) {
        let Some(control_id) = self.focused_text_id().map(str::to_owned) else {
            return;
        };
        let value = ControlValue::Text(self.text_buffer.clone());
        self.clear_focus();
        self.set_value(&control_id, value);
    }

    pub(crate) fn toggle_group(&mut self, group_id: &str) {
        if !self.collapsed_groups.remove(group_id) {
            self.collapsed_groups.insert(group_id.to_owned());
        }
    }

    pub fn group_collapsed(&self, group_id: &str) -> bool {
        self.collapsed_groups.contains(group_id)
    }

    /// Whether every enclosing group of this widget is expanded.
    pub fn widget_visible(&self, control_id: &str) -> bool {
        self.ancestors(control_id)
            .iter()
            .all(|group_id| !self.group_collapsed(group_id))
    }

    fn ancestors(&self, control_id: &str) -> Vec<String> {
        let mut path = Vec::new();
        find_ancestors(&self.panel.widgets, control_id, &mut path);
        path
    }

    pub fn slice_depth_cells(&self) -> f32 {
        self.slice_depth_cells
    }

    pub(crate) fn lower_slice_depth(&mut self) {
        self.slice_depth_cells = (self.slice_depth_cells - CONTROL_SLICE_STEP_CELLS).max(0.0);
    }

    pub(crate) fn raise_slice_depth(&mut self) {
        self.slice_depth_cells =
            (self.slice_depth_cells + CONTROL_SLICE_STEP_CELLS).min(CONTROL_SLICE_MAX_CELLS);
    }

    pub(crate) fn drag_slider(&mut self, control_id: &str, normalized_x: f32) {
        let Some(ControlWidget::Slider {
            minimum,
            maximum,
            step,
            ..
        }) = self.panel.find(control_id)
        else {
            return;
        };
        let (minimum, maximum, step) = (*minimum, *maximum, *step);
        let span = maximum - minimum;
        if span <= 0.0 {
            return;
        }
        let raw = minimum + span * f64::from(normalized_x.clamp(0.0, 1.0));
        let snapped = if step > 0.0 {
            minimum + ((raw - minimum) / step).round() * step
        } else {
            raw
        };
        let value = ControlValue::Number(snapped.clamp(minimum, maximum));
        // A drag is one gesture: the value tracks the cursor locally and the
        // submission waits for release, so dragging is not one request per
        // frame.
        if self.values.get(control_id) != Some(&value) {
            self.values.insert(control_id.to_owned(), value);
            self.edited.insert(control_id.to_owned());
        }
        self.dragging_slider = Some(control_id.to_owned());
    }

    /// Ends a slider drag, submitting the value it landed on.
    pub(crate) fn release_slider(&mut self) {
        let Some(control_id) = self.dragging_slider.take() else {
            return;
        };
        if self
            .panel
            .find(&control_id)
            .is_some_and(ControlWidget::submits)
        {
            self.submit(Some(control_id));
        }
    }

    /// Where a slider's fill bar ends, as a percentage of its track.
    pub fn slider_percent(&self, control_id: &str) -> f32 {
        let Some(ControlWidget::Slider {
            minimum, maximum, ..
        }) = self.panel.find(control_id)
        else {
            return 0.0;
        };
        let span = maximum - minimum;
        if span <= 0.0 {
            return 0.0;
        }
        let value = self
            .values
            .get(control_id)
            .and_then(ControlValue::as_number)
            .unwrap_or(*minimum);
        (((value - minimum) / span) as f32).clamp(0.0, 1.0) * 100.0
    }

    pub(crate) fn push_filter_char(&mut self, character: char) {
        match &self.focus {
            FocusOwner::Dropdown(_) => {
                if self.filter.chars().count() < CONTROL_FILTER_MAX_CHARS {
                    self.filter.push(character);
                    self.dropdown_start = 0;
                }
            }
            FocusOwner::TextField(_) => {
                if self.text_buffer.chars().count() < CONTROL_TEXT_MAX_CHARS {
                    self.text_buffer.push(character);
                }
            }
            FocusOwner::None => {}
        }
    }

    pub(crate) fn pop_filter_char(&mut self) {
        match &self.focus {
            FocusOwner::Dropdown(_) => {
                self.filter.pop();
                self.dropdown_start = 0;
            }
            FocusOwner::TextField(_) => {
                self.text_buffer.pop();
            }
            FocusOwner::None => {}
        }
    }

    pub(crate) fn scroll_dropdown(&mut self, rows: i32) {
        if rows < 0 {
            self.dropdown_start = self
                .dropdown_start
                .saturating_sub(rows.unsigned_abs() as usize);
        } else {
            self.dropdown_start =
                (self.dropdown_start + rows as usize).min(self.maximum_dropdown_start());
        }
    }

    /// Options of the open select, narrowed by the live filter.
    fn filtered_options(&self) -> Vec<&generation_api::ControlOption> {
        let Some(ControlWidget::Select { options, .. }) =
            self.open_dropdown_id().and_then(|id| self.panel.find(id))
        else {
            return Vec::new();
        };
        if self.filter.is_empty() {
            return options.iter().collect();
        }
        let needle = self.filter.to_lowercase();
        options
            .iter()
            .filter(|option| {
                option.label.to_lowercase().contains(&needle)
                    || option.value.to_lowercase().contains(&needle)
            })
            .collect()
    }

    pub(crate) fn dropdown_option(
        &self,
        row_index: usize,
    ) -> Option<&generation_api::ControlOption> {
        if row_index >= CONTROL_DROPDOWN_VISIBLE_OPTIONS {
            return None;
        }
        self.filtered_options()
            .get(self.dropdown_start + row_index)
            .copied()
    }

    pub(crate) fn select_dropdown_option(&mut self, row_index: usize) {
        let Some(control_id) = self.open_dropdown_id().map(str::to_owned) else {
            return;
        };
        let Some(value) = self
            .dropdown_option(row_index)
            .map(|option| option.value.clone())
        else {
            return;
        };
        self.clear_focus();
        self.set_value(&control_id, ControlValue::Text(value));
    }

    fn center_dropdown_on_selection(&mut self) {
        let selected = self
            .open_dropdown_id()
            .and_then(|id| self.values.get(id))
            .map(ControlValue::as_text);
        let position = selected
            .and_then(|selected| {
                self.filtered_options()
                    .iter()
                    .position(|option| option.value == selected)
            })
            .unwrap_or(0);
        self.dropdown_start = position.saturating_sub(CONTROL_DROPDOWN_VISIBLE_OPTIONS / 2);
        self.clamp_dropdown_start();
    }

    fn clamp_dropdown_start(&mut self) {
        self.dropdown_start = self.dropdown_start.min(self.maximum_dropdown_start());
    }

    fn maximum_dropdown_start(&self) -> usize {
        self.filtered_options()
            .len()
            .saturating_sub(CONTROL_DROPDOWN_VISIBLE_OPTIONS)
    }

    fn dropdown_range_label(&self) -> String {
        let filtered = self.filtered_options().len();
        if filtered == 0 {
            return "0 of 0".to_owned();
        }
        let first = self.dropdown_start + 1;
        let last = (self.dropdown_start + CONTROL_DROPDOWN_VISIBLE_OPTIONS).min(filtered);
        format!("{first}-{last} of {filtered}")
    }

    /// Text shown on a widget's value button.
    fn display_value(&self, control_id: &str) -> String {
        if self.focused_text_id() == Some(control_id) {
            return format!("{}_", self.text_buffer);
        }
        let Some(value) = self.values.get(control_id) else {
            return String::new();
        };
        match self.panel.find(control_id) {
            // A select shows the option's label, not the raw value a server
            // keys it by.
            Some(ControlWidget::Select { options, .. }) => {
                let text = value.as_text();
                options
                    .iter()
                    .find(|option| option.value == text)
                    .map(|option| option.label.clone())
                    .unwrap_or(text)
            }
            _ => value.as_text(),
        }
    }
}

fn find_ancestors(widgets: &[ControlWidget], control_id: &str, path: &mut Vec<String>) -> bool {
    for widget in widgets {
        if widget.id() == control_id {
            return true;
        }
        if widget.children().is_empty() {
            continue;
        }
        path.push(widget.id().to_owned());
        if find_ancestors(widget.children(), control_id, path) {
            return true;
        }
        path.pop();
    }
    false
}

/// Fixed-height stat block, so the widgets below never move as a server's
/// numbers change. Ends with a reserved error line.
pub(crate) fn control_stats_text(state: &ControlPanelState) -> String {
    let mut lines: Vec<String> = state
        .panel
        .stats
        .iter()
        .take(CONTROL_STAT_LINES)
        .map(|stat| {
            format!(
                "{:<14}{:>14}",
                truncate_ui_label(&stat.label, 14),
                truncate_ui_label(&stat.value, 14)
            )
        })
        .collect();
    while lines.len() < CONTROL_STAT_LINES {
        lines.push(" ".to_owned());
    }
    lines.push(if state.slice_depth_cells > 0.0 {
        format!("slice: hide nearest {:.1} cells", state.slice_depth_cells)
    } else {
        "slice: off".to_owned()
    });
    lines.push(
        state
            .error()
            .map(|error| format!("error: {}", truncate_ui_label(error, 28)))
            .unwrap_or_else(|| " ".to_owned()),
    );
    lines.join("\n")
}

pub(crate) fn control_summary_text(state: &ControlPanelState) -> String {
    if state.panel.summary.is_empty() {
        return format!("{:>6} pts", crate::compact_usize(state.projected_points));
    }
    truncate_ui_label(&state.panel.summary, 30)
}

/// Always two lines: range summary plus the live filter prompt.
pub(crate) fn control_dropdown_title(state: &ControlPanelState) -> String {
    let Some(control_id) = state.open_dropdown_id() else {
        return "Select value\n ".to_owned();
    };
    let label = state
        .panel
        .find(control_id)
        .map(|widget| widget.label().to_owned())
        .unwrap_or_else(|| control_id.to_owned());
    let filter_line = if state.filter.is_empty() {
        "type to filter, scroll to browse".to_owned()
    } else {
        format!("filter: {}_", state.filter)
    };
    format!(
        "{}  {}\n{}",
        truncate_ui_label(&label, 14),
        state.dropdown_range_label(),
        filter_line
    )
}

pub(crate) fn control_dropdown_option_label(state: &ControlPanelState, row_index: usize) -> String {
    let Some(control_id) = state.open_dropdown_id() else {
        return String::new();
    };
    let Some(option) = state.dropdown_option(row_index) else {
        return String::new();
    };
    let selected = state
        .values
        .get(control_id)
        .map(ControlValue::as_text)
        .is_some_and(|value| value == option.value);
    format!(
        "{}{}{}",
        truncate_ui_label(&option.label, 20),
        option
            .detail
            .as_ref()
            .map(|detail| format!("  ({detail})"))
            .unwrap_or_default(),
        if selected { "  [selected]" } else { "" },
    )
}

/// Label on a widget's own row: the server's label, marked when the control
/// holds an edit that has not been submitted.
pub(crate) fn control_widget_label(state: &ControlPanelState, control_id: &str) -> String {
    let Some(widget) = state.panel.find(control_id) else {
        return String::new();
    };
    format!(
        "{}{}",
        truncate_ui_label(widget.label(), 22),
        if state.is_edited(control_id) {
            " *"
        } else {
            ""
        }
    )
}

pub(crate) fn control_widget_value(state: &ControlPanelState, control_id: &str) -> String {
    truncate_ui_label(&state.display_value(control_id), 24)
}

pub(crate) fn control_widget_detail(state: &ControlPanelState, control_id: &str) -> String {
    state
        .panel
        .find(control_id)
        .and_then(ControlWidget::detail)
        .map(|detail| truncate_ui_label(detail, 34))
        .unwrap_or_default()
}

/// A group header shows its collapse state; other marks are checkbox ticks.
pub(crate) fn control_widget_mark(state: &ControlPanelState, control_id: &str) -> String {
    match state.panel.find(control_id) {
        Some(ControlWidget::Group { .. }) => {
            if state.group_collapsed(control_id) {
                "+".to_owned()
            } else {
                "-".to_owned()
            }
        }
        Some(ControlWidget::Toggle { .. }) => {
            if state
                .values
                .get(control_id)
                .is_some_and(ControlValue::as_bool)
            {
                "X".to_owned()
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// The pill's own buttons (header, slice depth, dropdown rows), which keep
/// the statically-typed `ViewerUiButton` of the shell.
type ShellButtonQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Interaction, &'static ViewerUiButton),
    (Changed<Interaction>, With<Button>),
>;
/// Buttons belonging to server-described widgets, which carry a control id
/// rather than a compile-time variant.
type WidgetButtonQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Interaction, &'static ControlWidgetButton),
    (Changed<Interaction>, With<Button>),
>;
/// Sliders track the held cursor, so they are read every frame rather than
/// only when their interaction changes.
type WidgetSliderQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static Interaction,
        &'static ControlWidgetButton,
        Option<&'static RelativeCursorPosition>,
    ),
    With<Button>,
>;
type ShellButtonColorQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static ViewerUiButton,
        &'static Interaction,
        &'static mut BackgroundColor,
    ),
    (With<Button>, Without<ControlWidgetButton>),
>;
type WidgetButtonColorQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static ControlWidgetButton,
        &'static Interaction,
        &'static mut BackgroundColor,
    ),
    With<Button>,
>;

pub fn handle_control_buttons(
    mut state: ResMut<ControlPanelState>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    slider_query: WidgetSliderQuery,
    shell_query: ShellButtonQuery,
    widget_query: WidgetButtonQuery,
) {
    // A slider follows the held cursor rather than discrete presses, so it is
    // read before the press-driven buttons below.
    if mouse_buttons.pressed(MouseButton::Left) {
        for (interaction, button, cursor_position) in &slider_query {
            if *interaction != Interaction::Pressed || button.action != ControlAction::SliderDrag {
                continue;
            }
            let Some(position) = cursor_position.and_then(|cursor| cursor.normalized) else {
                continue;
            };
            state.drag_slider(&button.control_id, position.x);
        }
    } else {
        state.release_slider();
    }

    for (interaction, button) in &shell_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button {
            ViewerUiButton::ToggleControlPanelPill => state.toggle_pill(),
            ViewerUiButton::SelectControlDropdownOption(row_index) => {
                state.select_dropdown_option(*row_index);
            }
            ViewerUiButton::SliceDepthLower => state.lower_slice_depth(),
            ViewerUiButton::SliceDepthHigher => state.raise_slice_depth(),
            _ => {}
        }
    }

    for (interaction, button) in &widget_query {
        if *interaction != Interaction::Pressed
            || state
                .panel()
                .find(&button.control_id)
                .is_some_and(ControlWidget::disabled)
        {
            continue;
        }
        let control_id = button.control_id.clone();
        match button.action {
            ControlAction::Activate => state.submit(Some(control_id)),
            ControlAction::ToggleDropdown => state.toggle_dropdown(&control_id),
            ControlAction::ToggleGroup => state.toggle_group(&control_id),
            ControlAction::FocusText => state.focus_text(&control_id),
            ControlAction::ToggleBool => {
                let toggled = !state.value(&control_id).is_some_and(ControlValue::as_bool);
                state.set_value(&control_id, ControlValue::Bool(toggled));
            }
            // Handled above, against the held cursor rather than a press.
            ControlAction::SliderDrag => {}
        }
    }
}

/// While a dropdown is open or a text field focused, typing edits it.
/// Escape/enter handling stays with the pause menu system so a single
/// keypress cannot both close a control and open the pause menu.
pub fn handle_control_keyboard(
    mut state: ResMut<ControlPanelState>,
    mut keyboard_events: EventReader<bevy::input::keyboard::KeyboardInput>,
) {
    use bevy::input::keyboard::Key;

    if !state.input_focused() {
        keyboard_events.clear();
        return;
    }
    for event in keyboard_events.read() {
        if !event.state.is_pressed() {
            continue;
        }
        match &event.logical_key {
            Key::Character(characters) => {
                for character in characters.chars() {
                    if !character.is_control() {
                        state.push_filter_char(character);
                    }
                }
            }
            Key::Space => state.push_filter_char(' '),
            Key::Backspace => state.pop_filter_char(),
            Key::Enter => state.commit_text(),
            _ => {}
        }
    }
}

/// While a dropdown is open, the mouse wheel scrolls its option list instead
/// of adjusting navigation speed.
pub fn handle_control_dropdown_scroll(
    mut state: ResMut<ControlPanelState>,
    mut wheel_events: EventReader<bevy::input::mouse::MouseWheel>,
    mut scroll_accumulator: Local<f32>,
) {
    use bevy::input::mouse::MouseScrollUnit;

    if !state.dropdown_active() {
        wheel_events.clear();
        *scroll_accumulator = 0.0;
        return;
    }
    for wheel in wheel_events.read() {
        *scroll_accumulator += match wheel.unit {
            MouseScrollUnit::Line => wheel.y,
            MouseScrollUnit::Pixel => wheel.y / 20.0,
        };
    }
    let rows = scroll_accumulator.trunc() as i32;
    if rows != 0 {
        *scroll_accumulator -= rows as f32;
        state.scroll_dropdown(-rows);
    }
}

/// Despawns and respawns the widget subtree when the server's panel revision
/// changes.
///
/// Ordered inside the button-handling chain and before the update systems, so
/// a click is fully applied to [`ControlPanelState`] before the entities it
/// landed on are torn down, and the respawned entities are populated the same
/// frame. Nothing else rebuilds: a value change, a new stat, or collapsing a
/// group leaves these entities alone.
pub fn rebuild_control_widgets(
    mut commands: Commands,
    mut state: ResMut<ControlPanelState>,
    root_query: Query<Entity, With<ControlWidgetRoot>>,
) {
    if state.rendered_revision == Some(state.panel.revision) {
        return;
    }
    let Ok(root) = root_query.get_single() else {
        return;
    };
    state.rendered_revision = Some(state.panel.revision);
    let widgets = state.panel.widgets.clone();
    commands.entity(root).despawn_descendants();
    commands.entity(root).with_children(|parent| {
        for widget in &widgets {
            spawn_widget(parent, widget, 0);
        }
    });
}

pub fn update_control_text(
    state: Res<ControlPanelState>,
    mut shell_text: Query<(&mut Text, &ViewerUiText), Without<ControlWidgetText>>,
    mut widget_text: Query<(&mut Text, &ControlWidgetText)>,
) {
    for (mut text, text_kind) in &mut shell_text {
        match text_kind {
            ViewerUiText::ControlPanelSummary => **text = control_summary_text(&state),
            ViewerUiText::ControlPanelStats => **text = control_stats_text(&state),
            ViewerUiText::ControlDropdownTitle => **text = control_dropdown_title(&state),
            ViewerUiText::ControlDropdownOption(row_index) => {
                **text = control_dropdown_option_label(&state, *row_index);
            }
            _ => {}
        }
    }
    for (mut text, widget) in &mut widget_text {
        **text = match widget.role {
            ControlTextRole::Label => control_widget_label(&state, &widget.control_id),
            ControlTextRole::Value => control_widget_value(&state, &widget.control_id),
            ControlTextRole::Mark => control_widget_mark(&state, &widget.control_id),
            ControlTextRole::Detail => control_widget_detail(&state, &widget.control_id),
        };
    }
}

pub fn update_control_panels(
    state: Res<ControlPanelState>,
    mut shell_panels: Query<(&mut Node, &ViewerUiPanel), Without<ControlWidgetPanel>>,
    mut widget_panels: Query<(&mut Node, &ControlWidgetPanel)>,
) {
    for (mut node, panel) in &mut shell_panels {
        match panel {
            ViewerUiPanel::ControlPanelPill => {
                node.width = if state.pill_expanded {
                    Val::Px(300.0)
                } else {
                    Val::Px(218.0)
                };
            }
            ViewerUiPanel::ControlPanelExpanded => node.display = display_if(state.pill_expanded),
            ViewerUiPanel::ControlDropdown => {
                node.display = display_if(state.pill_expanded && state.dropdown_active());
            }
            ViewerUiPanel::ControlDropdownOption(row_index) => {
                node.display = display_if(
                    state.pill_expanded
                        && state.dropdown_active()
                        && state.dropdown_option(*row_index).is_some(),
                );
            }
            _ => {}
        }
    }
    for (mut node, panel) in &mut widget_panels {
        match panel.role {
            ControlWidgetPanelRole::GroupChildren => {
                node.display = display_if(!state.group_collapsed(&panel.control_id));
            }
            ControlWidgetPanelRole::SliderFill => {
                node.width = Val::Percent(state.slider_percent(&panel.control_id));
            }
        }
    }
}

pub fn update_control_button_colors(
    state: Res<ControlPanelState>,
    mut shell_buttons: ShellButtonColorQuery,
    mut widget_buttons: WidgetButtonColorQuery,
) {
    for (button, interaction, color) in &mut shell_buttons {
        match button {
            ViewerUiButton::ToggleControlPanelPill => set_header_color(color, interaction),
            ViewerUiButton::SelectControlDropdownOption(row_index) => {
                let selected = state
                    .dropdown_option(*row_index)
                    .zip(state.open_dropdown_id().and_then(|id| state.values.get(id)))
                    .is_some_and(|(option, value)| option.value == value.as_text());
                set_button_color(color, selected, interaction);
            }
            ViewerUiButton::SliceDepthLower | ViewerUiButton::SliceDepthHigher => {
                set_button_color(color, false, interaction);
            }
            _ => {}
        }
    }
    for (button, interaction, mut color) in &mut widget_buttons {
        if state
            .panel()
            .find(&button.control_id)
            .is_some_and(ControlWidget::disabled)
        {
            *color = crate::disabled_button_color().into();
            continue;
        }
        let active = match button.action {
            ControlAction::ToggleDropdown => state.open_dropdown_id() == Some(&button.control_id),
            ControlAction::FocusText => state.focused_text_id() == Some(&button.control_id),
            ControlAction::ToggleBool => state
                .value(&button.control_id)
                .is_some_and(ControlValue::as_bool),
            ControlAction::Activate | ControlAction::SliderDrag | ControlAction::ToggleGroup => {
                false
            }
        };
        set_button_color(color, active, interaction);
    }
}

// ---------------------------------------------------------------------------
// Spawning
// ---------------------------------------------------------------------------

/// The panel pill: a static shell whose widget root is filled from the
/// server's schema by [`rebuild_control_widgets`].
pub(crate) fn spawn_control_panel_pill(parent: &mut ChildBuilder) {
    parent
        .spawn(pill_node(218.0, ViewerUiPanel::ControlPanelPill))
        .with_children(|pill| {
            spawn_pill_button(
                pill,
                ViewerUiButton::ToggleControlPanelPill,
                ViewerUiText::ControlPanelSummary,
                13.0,
            );
            pill.spawn(expanded_panel(ViewerUiPanel::ControlPanelExpanded))
                .with_children(|expanded| {
                    spawn_title(expanded, "View");
                    spawn_text(expanded, ViewerUiText::ControlPanelStats);
                    expanded.spawn((
                        Node {
                            width: Val::Percent(100.0),
                            flex_direction: FlexDirection::Column,
                            row_gap: Val::Px(6.0),
                            ..default()
                        },
                        ControlWidgetRoot,
                    ));
                    spawn_title(expanded, "Slice depth");
                    spawn_button_row(
                        expanded,
                        &[
                            (ViewerUiButton::SliceDepthLower, "Slice -"),
                            (ViewerUiButton::SliceDepthHigher, "Slice +"),
                        ],
                    );
                    spawn_control_dropdown_menu(expanded);
                });
        });
}

fn spawn_control_dropdown_menu(parent: &mut ChildBuilder) {
    parent
        .spawn((
            Node {
                display: Display::None,
                width: Val::Percent(100.0),
                padding: UiRect::axes(Val::Px(6.0), Val::Px(7.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderRadius::all(Val::Px(8.0)),
            BorderColor(Color::srgba(0.70, 0.76, 0.84, 0.42)),
            BackgroundColor(Color::srgba(0.045, 0.052, 0.068, 0.94)),
            ViewerUiPanel::ControlDropdown,
            Interaction::default(),
            FocusPolicy::Block,
        ))
        .with_children(|dropdown| {
            spawn_text(dropdown, ViewerUiText::ControlDropdownTitle);
            for row_index in 0..CONTROL_DROPDOWN_VISIBLE_OPTIONS {
                spawn_control_option_button(dropdown, row_index);
            }
        });
}

fn spawn_control_option_button(parent: &mut ChildBuilder, row_index: usize) {
    parent
        .spawn((
            Button,
            Node {
                display: Display::None,
                width: Val::Percent(100.0),
                min_height: Val::Px(28.0),
                padding: UiRect::horizontal(Val::Px(9.0)),
                justify_content: JustifyContent::FlexStart,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderRadius::all(Val::Px(6.0)),
            BorderColor(Color::srgba(0.64, 0.70, 0.78, 0.55)),
            BackgroundColor(button_color(false, false)),
            ViewerUiButton::SelectControlDropdownOption(row_index),
            ViewerUiPanel::ControlDropdownOption(row_index),
        ))
        .with_child((
            Text::new(""),
            TextFont {
                font_size: 11.0,
                ..default()
            },
            TextColor(Color::srgb(0.90, 0.93, 0.97)),
            ViewerUiText::ControlDropdownOption(row_index),
        ));
}

fn spawn_widget(parent: &mut ChildBuilder, widget: &ControlWidget, depth: usize) {
    match widget {
        ControlWidget::Group {
            id,
            collapsible,
            children,
            ..
        } => spawn_group(parent, id, *collapsible, children, depth),
        ControlWidget::Select { id, .. } => {
            spawn_value_row(parent, id, ControlAction::ToggleDropdown, depth)
        }
        ControlWidget::Text { id, .. } => {
            spawn_value_row(parent, id, ControlAction::FocusText, depth)
        }
        ControlWidget::Button { id, .. } => spawn_action_button(parent, id, depth),
        ControlWidget::Toggle { id, .. } => spawn_toggle_row(parent, id, depth),
        ControlWidget::Slider { id, .. } => spawn_slider_row(parent, id, depth),
    }
}

fn spawn_group(
    parent: &mut ChildBuilder,
    group_id: &str,
    collapsible: bool,
    children: &[ControlWidget],
    depth: usize,
) {
    parent.spawn(indented_column(depth)).with_children(|group| {
        spawn_group_header(group, group_id, collapsible);
        spawn_widget_detail(group, group_id);
        group
            .spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(6.0),
                    ..default()
                },
                ControlWidgetPanel {
                    control_id: group_id.to_owned(),
                    role: ControlWidgetPanelRole::GroupChildren,
                },
            ))
            .with_children(|body| {
                for child in children {
                    spawn_widget(body, child, depth + 1);
                }
            });
    });
}

fn spawn_group_header(parent: &mut ChildBuilder, group_id: &str, collapsible: bool) {
    if !collapsible {
        parent.spawn((
            Text::new(""),
            TextFont {
                font_size: 14.0,
                ..default()
            },
            TextColor(Color::srgb(0.96, 0.97, 0.99)),
            ControlWidgetText {
                control_id: group_id.to_owned(),
                role: ControlTextRole::Label,
            },
        ));
        return;
    }
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                min_height: Val::Px(26.0),
                padding: UiRect::horizontal(Val::Px(8.0)),
                justify_content: JustifyContent::FlexStart,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                ..default()
            },
            BorderRadius::all(Val::Px(6.0)),
            BackgroundColor(header_button_color(false)),
            ControlWidgetButton {
                control_id: group_id.to_owned(),
                action: ControlAction::ToggleGroup,
            },
        ))
        .with_children(|header| {
            header.spawn((
                Text::new(""),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgba(0.72, 0.78, 0.86, 0.9)),
                ControlWidgetText {
                    control_id: group_id.to_owned(),
                    role: ControlTextRole::Mark,
                },
            ));
            header.spawn((
                Text::new(""),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgb(0.96, 0.97, 0.99)),
                ControlWidgetText {
                    control_id: group_id.to_owned(),
                    role: ControlTextRole::Label,
                },
            ));
        });
}

/// A label beside a button showing the control's value — shared by select
/// (which opens the dropdown) and text (which takes keyboard focus).
fn spawn_value_row(
    parent: &mut ChildBuilder,
    control_id: &str,
    action: ControlAction,
    depth: usize,
) {
    parent
        .spawn(indented_column(depth))
        .with_children(|column| {
            column
                .spawn(Node {
                    height: Val::Px(30.0),
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new(""),
                        TextFont {
                            font_size: 12.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.86, 0.90, 0.96)),
                        Node {
                            width: Val::Px(56.0),
                            ..default()
                        },
                        ControlWidgetText {
                            control_id: control_id.to_owned(),
                            role: ControlTextRole::Label,
                        },
                    ));
                    row.spawn((
                        Button,
                        Node {
                            flex_grow: 1.0,
                            height: Val::Px(30.0),
                            padding: UiRect::horizontal(Val::Px(10.0)),
                            justify_content: JustifyContent::FlexStart,
                            align_items: AlignItems::Center,
                            border: UiRect::all(Val::Px(1.0)),
                            ..default()
                        },
                        BorderRadius::MAX,
                        BorderColor(Color::srgba(0.7, 0.76, 0.84, 0.75)),
                        BackgroundColor(button_color(false, false)),
                        ControlWidgetButton {
                            control_id: control_id.to_owned(),
                            action,
                        },
                    ))
                    .with_child((
                        Text::new(""),
                        TextFont {
                            font_size: 11.5,
                            ..default()
                        },
                        TextColor(Color::srgb(0.94, 0.96, 0.98)),
                        ControlWidgetText {
                            control_id: control_id.to_owned(),
                            role: ControlTextRole::Value,
                        },
                    ));
                });
            spawn_widget_detail(column, control_id);
        });
}

fn spawn_action_button(parent: &mut ChildBuilder, control_id: &str, depth: usize) {
    parent
        .spawn(indented_column(depth))
        .with_children(|column| {
            column
                .spawn((
                    Button,
                    Node {
                        width: Val::Percent(100.0),
                        min_height: Val::Px(30.0),
                        padding: UiRect::horizontal(Val::Px(10.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderRadius::MAX,
                    BorderColor(Color::srgba(0.7, 0.76, 0.84, 0.75)),
                    BackgroundColor(button_color(false, false)),
                    ControlWidgetButton {
                        control_id: control_id.to_owned(),
                        action: ControlAction::Activate,
                    },
                ))
                .with_child((
                    Text::new(""),
                    TextFont {
                        font_size: 12.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.94, 0.96, 0.98)),
                    ControlWidgetText {
                        control_id: control_id.to_owned(),
                        role: ControlTextRole::Label,
                    },
                ));
            spawn_widget_detail(column, control_id);
        });
}

fn spawn_toggle_row(parent: &mut ChildBuilder, control_id: &str, depth: usize) {
    parent
        .spawn(indented_column(depth))
        .with_children(|column| {
            column
                .spawn(Node {
                    height: Val::Px(26.0),
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Button,
                        Node {
                            width: Val::Px(20.0),
                            height: Val::Px(20.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            border: UiRect::all(Val::Px(1.0)),
                            ..default()
                        },
                        BorderRadius::all(Val::Px(4.0)),
                        BorderColor(Color::srgba(0.7, 0.76, 0.84, 0.75)),
                        BackgroundColor(button_color(false, false)),
                        ControlWidgetButton {
                            control_id: control_id.to_owned(),
                            action: ControlAction::ToggleBool,
                        },
                    ))
                    .with_child((
                        Text::new(""),
                        TextFont {
                            font_size: 14.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.94, 0.96, 0.98)),
                        ControlWidgetText {
                            control_id: control_id.to_owned(),
                            role: ControlTextRole::Mark,
                        },
                    ));
                    row.spawn((
                        Text::new(""),
                        TextFont {
                            font_size: 12.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.86, 0.90, 0.96)),
                        ControlWidgetText {
                            control_id: control_id.to_owned(),
                            role: ControlTextRole::Label,
                        },
                    ));
                });
            spawn_widget_detail(column, control_id);
        });
}

fn spawn_slider_row(parent: &mut ChildBuilder, control_id: &str, depth: usize) {
    parent
        .spawn(indented_column(depth))
        .with_children(|column| {
            column
                .spawn(Node {
                    width: Val::Percent(100.0),
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new(""),
                        TextFont {
                            font_size: 12.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.86, 0.90, 0.96)),
                        Node {
                            flex_grow: 1.0,
                            ..default()
                        },
                        ControlWidgetText {
                            control_id: control_id.to_owned(),
                            role: ControlTextRole::Label,
                        },
                    ));
                    row.spawn((
                        Text::new(""),
                        TextFont {
                            font_size: 12.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.94, 0.96, 0.98)),
                        ControlWidgetText {
                            control_id: control_id.to_owned(),
                            role: ControlTextRole::Value,
                        },
                    ));
                });
            column
                .spawn((
                    Button,
                    Node {
                        min_height: Val::Px(28.0),
                        width: Val::Percent(100.0),
                        padding: UiRect::horizontal(Val::Px(7.0)),
                        justify_content: JustifyContent::FlexStart,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderRadius::all(Val::Px(8.0)),
                    BorderColor(Color::srgba(0.64, 0.70, 0.78, 0.55)),
                    BackgroundColor(button_color(false, false)),
                    RelativeCursorPosition::default(),
                    ControlWidgetButton {
                        control_id: control_id.to_owned(),
                        action: ControlAction::SliderDrag,
                    },
                ))
                .with_children(|slider| {
                    slider
                        .spawn((
                            Node {
                                width: Val::Percent(100.0),
                                height: Val::Px(8.0),
                                position_type: PositionType::Relative,
                                ..default()
                            },
                            BorderRadius::MAX,
                            BackgroundColor(Color::srgba(0.06, 0.075, 0.095, 0.95)),
                        ))
                        .with_children(|track| {
                            track.spawn((
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: Val::Px(0.0),
                                    height: Val::Px(8.0),
                                    width: Val::Percent(0.0),
                                    ..default()
                                },
                                BorderRadius::MAX,
                                BackgroundColor(Color::srgb(0.22, 0.68, 1.0)),
                                ControlWidgetPanel {
                                    control_id: control_id.to_owned(),
                                    role: ControlWidgetPanelRole::SliderFill,
                                },
                            ));
                        });
                });
            spawn_widget_detail(column, control_id);
        });
}

/// The server's secondary line. Always spawned; it renders empty when the
/// widget has no detail, so a server adding one needs no structural change.
fn spawn_widget_detail(parent: &mut ChildBuilder, control_id: &str) {
    parent.spawn((
        Text::new(""),
        TextFont {
            font_size: 10.5,
            ..default()
        },
        TextColor(Color::srgba(0.70, 0.76, 0.84, 0.85)),
        ControlWidgetText {
            control_id: control_id.to_owned(),
            role: ControlTextRole::Detail,
        },
    ));
}

fn indented_column(depth: usize) -> Node {
    Node {
        width: Val::Percent(100.0),
        flex_direction: FlexDirection::Column,
        row_gap: Val::Px(4.0),
        padding: UiRect::left(Val::Px(depth as f32 * GROUP_INDENT)),
        ..default()
    }
}

/// Parses `--control id=value` pairs. Numbers and booleans are recognized so
/// a slider or toggle can be pinned from the command line without quoting
/// rules; everything else stays text.
pub fn parse_control_assignment(assignment: &str) -> Result<(String, ControlValue), String> {
    let Some((control_id, raw)) = assignment.split_once('=') else {
        return Err(format!("Expected id=value, got '{assignment}'"));
    };
    let control_id = control_id.trim();
    if control_id.is_empty() {
        return Err(format!("Missing control id in '{assignment}'"));
    }
    Ok((control_id.to_owned(), parse_control_value(raw.trim())))
}

fn parse_control_value(raw: &str) -> ControlValue {
    match raw {
        "true" => ControlValue::Bool(true),
        "false" => ControlValue::Bool(false),
        _ => raw
            .parse::<f64>()
            .map(ControlValue::Number)
            .unwrap_or_else(|_| ControlValue::Text(raw.to_owned())),
    }
}

/// Builds the value map for a run's `--control` flags.
pub fn control_values_from_assignments<'a>(
    assignments: impl IntoIterator<Item = &'a str>,
) -> Result<ControlValues, String> {
    let mut values = BTreeMap::new();
    for assignment in assignments {
        let (control_id, value) = parse_control_assignment(assignment)?;
        values.insert(control_id, value);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use generation_api::{ControlOption, StatLine};

    fn sample_panel(revision: u64) -> ControlPanel {
        ControlPanel {
            revision,
            title: "View".to_owned(),
            summary: "sphere  120 pts".to_owned(),
            stats: vec![StatLine {
                label: "shown".to_owned(),
                value: "120".to_owned(),
            }],
            error: None,
            widgets: vec![
                ControlWidget::Select {
                    id: "shape".to_owned(),
                    label: "Shape".to_owned(),
                    detail: None,
                    value: "sphere".to_owned(),
                    options: vec![
                        ControlOption {
                            value: "sphere".to_owned(),
                            label: "Sphere".to_owned(),
                            detail: None,
                        },
                        ControlOption {
                            value: "cube".to_owned(),
                            label: "Cube".to_owned(),
                            detail: None,
                        },
                        ControlOption {
                            value: "spiral".to_owned(),
                            label: "Spiral".to_owned(),
                            detail: None,
                        },
                    ],
                    disabled: false,
                    submits: true,
                },
                ControlWidget::Group {
                    id: "layout".to_owned(),
                    label: "Layout".to_owned(),
                    detail: None,
                    collapsible: true,
                    children: vec![
                        ControlWidget::Slider {
                            id: "radius".to_owned(),
                            label: "Radius".to_owned(),
                            detail: None,
                            value: 10.0,
                            minimum: 0.0,
                            maximum: 100.0,
                            step: 5.0,
                            disabled: false,
                            submits: true,
                        },
                        ControlWidget::Text {
                            id: "seed".to_owned(),
                            label: "Seed".to_owned(),
                            detail: None,
                            value: "42".to_owned(),
                            disabled: false,
                            // A form field: edits wait for a submitting control.
                            submits: false,
                        },
                    ],
                },
                ControlWidget::Button {
                    id: "randomize".to_owned(),
                    label: "Randomize".to_owned(),
                    detail: None,
                    disabled: false,
                    submits: true,
                },
            ],
        }
    }

    fn sample_state() -> ControlPanelState {
        let mut state = ControlPanelState::default();
        state.apply_panel(sample_panel(1), 120, 500);
        state
    }

    #[test]
    fn panel_seeds_values_from_the_server() {
        let state = sample_state();
        assert_eq!(
            state.value("shape"),
            Some(&ControlValue::Text("sphere".to_owned()))
        );
        assert_eq!(state.value("radius"), Some(&ControlValue::Number(10.0)));
        assert_eq!(state.projected_points, 120);
    }

    #[test]
    fn submitting_control_sends_every_value_and_names_what_was_activated() {
        let mut state = sample_state();
        state.set_value("shape", ControlValue::Text("cube".to_owned()));

        let submit = state.take_pending_submit().expect("change submitted");
        assert_eq!(submit.activated.as_deref(), Some("shape"));
        assert_eq!(
            submit.values.get("shape"),
            Some(&ControlValue::Text("cube".to_owned()))
        );
        // Unchanged controls travel too, so the server needs no session state.
        assert_eq!(
            submit.values.get("radius"),
            Some(&ControlValue::Number(10.0))
        );
        assert!(state.take_pending_submit().is_none());
    }

    #[test]
    fn non_submitting_control_waits_for_a_submitting_one() {
        let mut state = sample_state();
        state.set_value("seed", ControlValue::Text("7".to_owned()));

        assert!(
            state.take_pending_submit().is_none(),
            "a form field must not send on its own"
        );
        assert!(state.is_edited("seed"), "the pending edit is marked");

        // The button carries the accumulated edits with it.
        state.submit(Some("randomize".to_owned()));
        let submit = state.take_pending_submit().expect("button submitted");
        assert_eq!(submit.activated.as_deref(), Some("randomize"));
        assert_eq!(
            submit.values.get("seed"),
            Some(&ControlValue::Text("7".to_owned()))
        );
        assert!(!state.is_edited("seed"));
    }

    #[test]
    fn a_streamed_panel_does_not_overwrite_an_unsubmitted_edit() {
        let mut state = sample_state();
        state.set_value("seed", ControlValue::Text("7".to_owned()));

        state.apply_panel(sample_panel(1), 240, 500);

        assert_eq!(
            state.value("seed"),
            Some(&ControlValue::Text("7".to_owned())),
            "a snapshot arriving mid-edit must not reset the field"
        );
        assert_eq!(
            state.value("shape"),
            Some(&ControlValue::Text("sphere".to_owned())),
            "untouched controls still follow the server"
        );
    }

    #[test]
    fn dropping_a_control_stops_it_being_sent() {
        let mut state = sample_state();
        let mut panel = sample_panel(2);
        panel.widgets.retain(|widget| widget.id() != "shape");

        state.apply_panel(panel, 120, 500);

        assert!(state.value("shape").is_none());
    }

    /// Nothing may keep naming a control the server dropped — including a
    /// drag or a focus that was live when the panel changed under it.
    #[test]
    fn dropping_a_control_releases_a_drag_or_focus_on_it() {
        let mut state = sample_state();
        state.toggle_dropdown("shape");
        state.drag_slider("radius", 0.5);

        let mut panel = sample_panel(2);
        panel.widgets.clear();
        state.apply_panel(panel, 0, 0);

        assert_eq!(state.focus(), &FocusOwner::None);
        assert_eq!(
            state.dragging_slider, None,
            "a drag must not outlive the control it names"
        );
    }

    #[test]
    fn dropdown_filters_scrolls_and_selects() {
        let mut state = sample_state();
        state.toggle_dropdown("shape");
        assert!(state.dropdown_active());
        assert!(state.input_focused());

        for character in "sp".chars() {
            state.push_filter_char(character);
        }
        let labels: Vec<String> = (0..CONTROL_DROPDOWN_VISIBLE_OPTIONS)
            .filter_map(|row| state.dropdown_option(row).map(|o| o.label.clone()))
            .collect();
        assert_eq!(labels, vec!["Sphere".to_owned(), "Spiral".to_owned()]);

        state.pop_filter_char();
        state.pop_filter_char();
        assert_eq!(state.filtered_options().len(), 3);

        state.scroll_dropdown(100);
        assert_eq!(state.dropdown_start, state.maximum_dropdown_start());
        state.scroll_dropdown(-100);
        assert_eq!(state.dropdown_start, 0);

        state.select_dropdown_option(1);
        assert_eq!(
            state.value("shape"),
            Some(&ControlValue::Text("cube".to_owned()))
        );
        assert!(!state.dropdown_active(), "selecting closes the dropdown");
        assert_eq!(
            state.take_pending_submit().unwrap().activated.as_deref(),
            Some("shape")
        );
    }

    #[test]
    fn typing_into_a_text_field_commits_on_enter() {
        let mut state = sample_state();
        state.focus_text("seed");
        assert!(state.input_focused(), "camera controls must stand down");
        state.pop_filter_char();
        state.pop_filter_char();
        for character in "99".chars() {
            state.push_filter_char(character);
        }
        // The buffer shows a cursor while focused.
        assert_eq!(control_widget_value(&state, "seed"), "99_");

        state.commit_text();

        assert_eq!(
            state.value("seed"),
            Some(&ControlValue::Text("99".to_owned()))
        );
        assert!(!state.input_focused());
    }

    #[test]
    fn opening_a_dropdown_releases_a_focused_text_field() {
        let mut state = sample_state();
        state.focus_text("seed");
        state.toggle_dropdown("shape");

        assert!(state.dropdown_active());
        assert_eq!(
            state.focus(),
            &FocusOwner::Dropdown("shape".to_owned()),
            "one owner at a time, so Escape and the wheel have one thing to check"
        );
    }

    #[test]
    fn slider_drag_snaps_to_step_and_submits_once_on_release() {
        let mut state = sample_state();

        state.drag_slider("radius", 0.42);
        assert_eq!(state.value("radius"), Some(&ControlValue::Number(40.0)));
        assert!(
            state.take_pending_submit().is_none(),
            "a drag is one gesture, not one request per frame"
        );
        state.drag_slider("radius", 0.47);
        assert_eq!(state.value("radius"), Some(&ControlValue::Number(45.0)));

        state.release_slider();
        let submit = state.take_pending_submit().expect("release submitted");
        assert_eq!(submit.activated.as_deref(), Some("radius"));
        assert_eq!(state.slider_percent("radius"), 45.0);
    }

    #[test]
    fn slider_drag_clamps_outside_the_track() {
        let mut state = sample_state();
        state.drag_slider("radius", -3.0);
        assert_eq!(state.value("radius"), Some(&ControlValue::Number(0.0)));
        state.drag_slider("radius", 9.0);
        assert_eq!(state.value("radius"), Some(&ControlValue::Number(100.0)));
    }

    #[test]
    fn collapsing_a_group_is_local_and_hides_its_children() {
        let mut state = sample_state();
        assert!(state.widget_visible("radius"));

        state.toggle_group("layout");

        assert!(state.group_collapsed("layout"));
        assert!(!state.widget_visible("radius"));
        assert!(
            state.take_pending_submit().is_none(),
            "collapsing is display state and must never hit the server"
        );
        assert_eq!(
            state.rendered_revision, None,
            "nor may it trigger a widget rebuild"
        );
    }

    #[test]
    fn initial_control_values_override_the_first_panel_only() {
        let mut state = ControlPanelState::new(ControlValues::from([(
            "shape".to_owned(),
            ControlValue::Text("cube".to_owned()),
        )]));

        state.apply_panel(sample_panel(1), 10, 10);
        assert_eq!(
            state.value("shape"),
            Some(&ControlValue::Text("cube".to_owned()))
        );

        // Once applied, the server owns the value again.
        state.apply_panel(sample_panel(1), 10, 10);
        assert_eq!(
            state.value("shape"),
            Some(&ControlValue::Text("sphere".to_owned()))
        );
    }

    #[test]
    fn unknown_initial_controls_are_reported() {
        let mut state = ControlPanelState::new(ControlValues::from([(
            "nope".to_owned(),
            ControlValue::Text("1".to_owned()),
        )]));
        assert!(state.take_unknown_initial_controls().is_empty());

        state.apply_panel(sample_panel(1), 10, 10);

        assert_eq!(
            state.take_unknown_initial_controls(),
            vec!["nope".to_owned()]
        );
        assert!(
            state.take_unknown_initial_controls().is_empty(),
            "a streaming load delivers many snapshots; the warning is reported once"
        );
    }

    #[test]
    fn a_request_failure_outranks_a_server_rejection_then_clears() {
        let mut state = sample_state();
        let mut rejected = sample_panel(1);
        rejected.error = Some("two axes on one dimension".to_owned());
        state.apply_panel(rejected, 120, 500);
        assert_eq!(state.error(), Some("two axes on one dimension"));

        state.set_request_error("Projection reload failed");
        assert_eq!(state.error(), Some("Projection reload failed"));

        state.apply_panel(sample_panel(1), 120, 500);
        assert_eq!(state.error(), None);
    }

    #[test]
    fn stat_block_height_is_fixed_so_widgets_below_never_move() {
        let mut state = sample_state();
        let one_stat = control_stats_text(&state).lines().count();

        let mut panel = sample_panel(1);
        panel.stats = (0..4)
            .map(|index| StatLine {
                label: format!("stat {index}"),
                value: index.to_string(),
            })
            .collect();
        state.apply_panel(panel, 120, 500);

        assert_eq!(control_stats_text(&state).lines().count(), one_stat);
    }

    #[test]
    fn slice_depth_steps_and_clamps_at_zero() {
        let mut state = sample_state();
        assert_eq!(state.slice_depth_cells(), 0.0);

        state.lower_slice_depth();
        assert_eq!(state.slice_depth_cells(), 0.0);

        state.raise_slice_depth();
        state.raise_slice_depth();
        assert_eq!(state.slice_depth_cells(), 2.0 * CONTROL_SLICE_STEP_CELLS);

        state.lower_slice_depth();
        assert_eq!(state.slice_depth_cells(), CONTROL_SLICE_STEP_CELLS);
    }

    #[test]
    fn control_assignments_parse_into_typed_values() {
        let values =
            control_values_from_assignments(["shape=cube", "radius=12.5", "jitter=true"]).unwrap();
        assert_eq!(
            values.get("shape"),
            Some(&ControlValue::Text("cube".to_owned()))
        );
        assert_eq!(values.get("radius"), Some(&ControlValue::Number(12.5)));
        assert_eq!(values.get("jitter"), Some(&ControlValue::Bool(true)));

        let error = control_values_from_assignments(["shape"]).unwrap_err();
        assert!(error.contains("id=value"), "{error}");
    }
}
