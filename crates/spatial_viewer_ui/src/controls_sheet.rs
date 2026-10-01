//! The controls sheet: a pause-menu screen listing every [`Action`] with
//! what it is bound to, built from the bindings table itself so it lists
//! exactly what the viewer answers to. Each action's slots rebind it (see
//! [`BindingEditor`]), a held one can be made to toggle, and a filter box
//! narrows the list.

use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::ui::{FocusPolicy, RelativeCursorPosition};

use crate::input_bindings::{
    Action, ActionCategory, BindingEditor, BindingSlot, ControlBindings, HoldMode, SLOT_COUNT,
};
use crate::text_entry::{TextEntry, TextEntryBox, TextEntryTarget};
use crate::{
    display_if, menu_panel, set_button_color, set_disabled_button_color, spawn_menu_button,
    PauseMenuState, PauseScreen, ViewerUiButton, ViewerUiPanel,
};

const SHEET_WIDTH: f32 = 860.0;
const ROW_FONT_SIZE: f32 = 12.0;
const SLOT_WIDTH: f32 = 124.0;
const MODE_WIDTH: f32 = 62.0;
const RESET_WIDTH: f32 = 52.0;
const CELL_GAP: f32 = 6.0;
const CELL_HEIGHT: f32 = 22.0;
/// The slots side by side, where a row without its own shows what it
/// follows.
const BINDINGS_WIDTH: f32 = SLOT_WIDTH * SLOT_COUNT as f32 + CELL_GAP * (SLOT_COUNT as f32 - 1.0);
/// Pixels one line of a mouse wheel scrolls the list.
const WHEEL_LINE_PIXELS: f32 = 28.0;
const SCROLLBAR_WIDTH: f32 = 8.0;
/// Shortest the scrollbar's thumb gets, as a fraction of its track.
const MIN_THUMB_FRACTION: f32 = 0.08;
/// Most conflicts listed at once.
const SHOWN_CONFLICTS: usize = 3;

const EXPLAINER: &str = "Click a slot, then press a key, a mouse button or a combination \
such as Ctrl+Z; press it twice quickly to bind a double-tap. Esc cancels and Backspace \
empties the slot. Hold / Toggle picks whether a held control stays on until pressed \
again. Greyed rows follow the control they name. Changes are saved as you make them.";

const TEXT_COLOR: Color = Color::srgb(0.86, 0.90, 0.96);
const MUTED_COLOR: Color = Color::srgba(0.62, 0.68, 0.76, 0.90);
const BINDING_COLOR: Color = Color::srgb(0.98, 0.86, 0.46);
const CONFLICT_COLOR: Color = Color::srgb(1.0, 0.50, 0.45);
const CAPTURING_COLOR: Color = Color::srgb(0.96, 0.97, 0.99);

/// The filter the list is narrowed by, and whether it is being typed.
#[derive(Resource, Default)]
pub struct ControlsSheetState {
    filter: String,
    editing_filter: bool,
}

/// A button on the sheet.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlsSheetButton {
    /// Captures a new binding for one slot.
    Slot(BindingSlot),
    HoldMode(Action),
    Reset(Action),
    ResetAll,
    Filter,
}

/// A text the sheet keeps current.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlsSheetText {
    Slot(BindingSlot),
    HoldMode(Action),
    /// What an action without slots of its own is bound to.
    Follows(Action),
    Filter,
    /// Conflicts, and why the last change was not made.
    Warnings,
    NoMatches,
}

/// A part of the sheet that shows only at times.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlsSheetPart {
    /// An action's row, shown while it matches the filter.
    Row(Action),
    /// A category's title, shown while any of its rows is.
    Category(ActionCategory),
    /// Covers the screen while a binding is captured, so the click that
    /// binds a mouse button presses nothing under it.
    CaptureShield,
}

/// The list the mouse wheel scrolls: a window onto its content.
#[derive(Component)]
pub struct ControlsSheetList;

/// Everything the list holds, as tall as all of it.
#[derive(Component)]
pub struct ControlsSheetListContent;

/// The scrollbar beside the list: its track, which a press or drag scrolls
/// to, and the thumb showing which part of the list is in view.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlsSheetScrollbar {
    Track,
    Thumb,
}

pub(crate) fn spawn_controls_sheet(overlay: &mut ChildBuilder) {
    let mut panel = menu_panel(ViewerUiPanel::PauseControls, SHEET_WIDTH);
    panel.0.max_width = Val::Percent(96.0);
    panel.0.height = Val::Percent(90.0);
    overlay.spawn(panel).with_children(|sheet| {
        sheet.spawn((
            Text::new("Controls"),
            TextFont {
                font_size: 24.0,
                ..default()
            },
            TextColor(Color::srgb(0.96, 0.97, 0.99)),
        ));
        sheet.spawn((
            Text::new(EXPLAINER),
            TextFont {
                font_size: ROW_FONT_SIZE,
                ..default()
            },
            TextColor(MUTED_COLOR),
        ));
        sheet
            .spawn((
                Button,
                Node {
                    width: Val::Percent(100.0),
                    min_height: Val::Px(28.0),
                    padding: UiRect::horizontal(Val::Px(8.0)),
                    align_items: AlignItems::Center,
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BorderRadius::all(Val::Px(6.0)),
                BorderColor(Color::srgba(0.64, 0.70, 0.78, 0.55)),
                BackgroundColor(Color::srgba(0.06, 0.07, 0.09, 0.95)),
                ControlsSheetButton::Filter,
                TextEntryBox(TextEntryTarget::ControlsFilter),
            ))
            .with_child((
                Text::new(""),
                TextFont {
                    font_size: 12.5,
                    ..default()
                },
                TextColor(Color::srgb(0.94, 0.96, 0.98)),
                ControlsSheetText::Filter,
            ));
        sheet.spawn((
            Text::new(""),
            TextFont {
                font_size: ROW_FONT_SIZE,
                ..default()
            },
            TextColor(CONFLICT_COLOR),
            ControlsSheetText::Warnings,
        ));
        sheet
            .spawn(Node {
                width: Val::Percent(100.0),
                flex_grow: 1.0,
                flex_shrink: 1.0,
                min_height: Val::Px(0.0),
                column_gap: Val::Px(CELL_GAP),
                ..default()
            })
            .with_children(|list_area| {
                list_area
                    .spawn((
                        Node {
                            flex_grow: 1.0,
                            flex_direction: FlexDirection::Column,
                            overflow: Overflow::scroll_y(),
                            ..default()
                        },
                        ScrollPosition::default(),
                        ControlsSheetList,
                    ))
                    .with_children(|list| {
                        // Never shrunk to the window, so rows keep their
                        // height and the window scrolls over them.
                        list.spawn((
                            Node {
                                width: Val::Percent(100.0),
                                flex_shrink: 0.0,
                                flex_direction: FlexDirection::Column,
                                row_gap: Val::Px(2.0),
                                ..default()
                            },
                            ControlsSheetListContent,
                        ))
                        .with_children(|content| {
                            for category in ActionCategory::ALL {
                                spawn_category(content, category);
                            }
                            content.spawn((
                                Text::new("No control matches the filter."),
                                TextFont {
                                    font_size: ROW_FONT_SIZE,
                                    ..default()
                                },
                                TextColor(MUTED_COLOR),
                                ControlsSheetText::NoMatches,
                            ));
                        });
                    });
                list_area
                    .spawn((
                        Node {
                            width: Val::Px(SCROLLBAR_WIDTH),
                            height: Val::Percent(100.0),
                            flex_shrink: 0.0,
                            ..default()
                        },
                        BorderRadius::all(Val::Px(SCROLLBAR_WIDTH * 0.5)),
                        BackgroundColor(Color::srgba(0.78, 0.84, 0.92, 0.10)),
                        Interaction::default(),
                        FocusPolicy::Block,
                        RelativeCursorPosition::default(),
                        ControlsSheetScrollbar::Track,
                    ))
                    .with_child((
                        Node {
                            position_type: PositionType::Absolute,
                            width: Val::Percent(100.0),
                            ..default()
                        },
                        BorderRadius::all(Val::Px(SCROLLBAR_WIDTH * 0.5)),
                        BackgroundColor(Color::srgba(0.78, 0.84, 0.92, 0.55)),
                        ControlsSheetScrollbar::Thumb,
                    ));
            });
        sheet
            .spawn(Node {
                width: Val::Percent(100.0),
                column_gap: Val::Px(10.0),
                ..default()
            })
            .with_children(|footer| {
                spawn_cell_button(
                    footer,
                    ControlsSheetButton::ResetAll,
                    Text::new("Reset all"),
                    None,
                    Node {
                        width: Val::Percent(100.0),
                        min_height: Val::Px(34.0),
                        ..default()
                    },
                );
                spawn_menu_button(footer, ViewerUiButton::PauseBack, "Back");
            });
    });
    overlay.spawn((
        Node {
            display: Display::None,
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            right: Val::Px(0.0),
            top: Val::Px(0.0),
            bottom: Val::Px(0.0),
            ..default()
        },
        ZIndex(1),
        Interaction::default(),
        FocusPolicy::Block,
        ControlsSheetPart::CaptureShield,
    ));
}

fn spawn_category(list: &mut ChildBuilder, category: ActionCategory) {
    list.spawn((
        Text::new(category.title()),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(Color::srgba(0.62, 0.74, 0.90, 0.95)),
        Node {
            margin: UiRect::top(Val::Px(8.0)),
            ..default()
        },
        ControlsSheetPart::Category(category),
    ));
    for &action in Action::ALL
        .iter()
        .filter(|action| action.category() == category)
    {
        list.spawn((
            Node {
                width: Val::Percent(100.0),
                min_height: Val::Px(CELL_HEIGHT + 4.0),
                align_items: AlignItems::Center,
                column_gap: Val::Px(CELL_GAP),
                ..default()
            },
            ControlsSheetPart::Row(action),
        ))
        .with_children(|row| spawn_row(row, action));
    }
}

fn spawn_row(row: &mut ChildBuilder, action: Action) {
    let own_row = action.rebindable() || action.shares_with().is_none();
    row.spawn((
        Text::new(action.description()),
        TextFont {
            font_size: ROW_FONT_SIZE,
            ..default()
        },
        TextColor(if own_row { TEXT_COLOR } else { MUTED_COLOR }),
        Node {
            flex_grow: 1.0,
            flex_shrink: 1.0,
            ..default()
        },
    ));
    if action.rebindable() {
        for slot in 0..SLOT_COUNT {
            let slot = BindingSlot { action, slot };
            spawn_cell_button(
                row,
                ControlsSheetButton::Slot(slot),
                Text::new(""),
                Some(ControlsSheetText::Slot(slot)),
                cell_node(SLOT_WIDTH),
            );
        }
    } else {
        row.spawn(Node {
            width: Val::Px(BINDINGS_WIDTH),
            flex_shrink: 0.0,
            flex_direction: FlexDirection::Column,
            ..default()
        })
        .with_children(|cell| {
            cell.spawn((
                Text::new(""),
                TextFont {
                    font_size: ROW_FONT_SIZE,
                    ..default()
                },
                TextColor(MUTED_COLOR),
                ControlsSheetText::Follows(action),
            ));
            if let Some(other) = action.shares_with() {
                cell.spawn((
                    Text::new(format!("same as: {}", other.description())),
                    TextFont {
                        font_size: 10.5,
                        ..default()
                    },
                    TextColor(MUTED_COLOR),
                ));
            }
        });
    }
    if action.can_toggle() {
        spawn_cell_button(
            row,
            ControlsSheetButton::HoldMode(action),
            Text::new(""),
            Some(ControlsSheetText::HoldMode(action)),
            cell_node(MODE_WIDTH),
        );
    } else {
        row.spawn(cell_node(MODE_WIDTH));
    }
    if action.rebindable() || action.can_toggle() {
        spawn_cell_button(
            row,
            ControlsSheetButton::Reset(action),
            Text::new("Reset"),
            None,
            cell_node(RESET_WIDTH),
        );
    } else {
        row.spawn(cell_node(RESET_WIDTH));
    }
}

fn cell_node(width: f32) -> Node {
    Node {
        width: Val::Px(width),
        min_height: Val::Px(CELL_HEIGHT),
        flex_shrink: 0.0,
        ..default()
    }
}

fn spawn_cell_button(
    parent: &mut ChildBuilder,
    button: ControlsSheetButton,
    text: Text,
    text_kind: Option<ControlsSheetText>,
    node: Node,
) {
    parent
        .spawn((
            Button,
            Node {
                padding: UiRect::horizontal(Val::Px(6.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..node
            },
            BorderRadius::all(Val::Px(6.0)),
            BorderColor(Color::srgba(0.7, 0.76, 0.84, 0.55)),
            BackgroundColor(crate::button_color(false, false)),
            button,
        ))
        .with_children(|cell| {
            let mut label = cell.spawn((
                text,
                TextFont {
                    font_size: ROW_FONT_SIZE,
                    ..default()
                },
                TextColor(TEXT_COLOR),
            ));
            if let Some(text_kind) = text_kind {
                label.insert(text_kind);
            }
        });
}

/// Carries out the sheet's buttons and takes the filter as it is typed.
/// While a binding is being captured, presses belong to the capture.
pub fn handle_controls_sheet_buttons(
    buttons: Query<(&Interaction, &ControlsSheetButton), Changed<Interaction>>,
    mut bindings: ResMut<ControlBindings>,
    mut editor: ResMut<BindingEditor>,
    mut text_entry: ResMut<TextEntry>,
    mut sheet: ResMut<ControlsSheetState>,
) {
    let pressed = buttons
        .iter()
        .filter(|(interaction, _)| **interaction == Interaction::Pressed)
        .map(|(_, button)| *button);
    for button in pressed {
        if editor.holds_input() {
            break;
        }
        match button {
            ControlsSheetButton::Slot(slot) => editor.begin(slot),
            ControlsSheetButton::HoldMode(action) => {
                bindings.switch_hold_mode(action);
                editor.set_notice(None);
            }
            ControlsSheetButton::Reset(action) if !bindings.is_default(action) => {
                bindings.reset(action);
                editor.set_notice(None);
            }
            ControlsSheetButton::Reset(_) => {}
            ControlsSheetButton::ResetAll if !bindings.all_default() => {
                bindings.reset_all();
                editor.set_notice(None);
            }
            ControlsSheetButton::ResetAll => {}
            ControlsSheetButton::Filter => {
                if text_entry.editing() != Some(TextEntryTarget::ControlsFilter) {
                    text_entry.begin(TextEntryTarget::ControlsFilter, &sheet.filter);
                }
            }
        }
    }
    if let Some(text) = text_entry.text(TextEntryTarget::ControlsFilter) {
        if sheet.filter != text {
            sheet.filter = text.to_owned();
        }
    }
    let editing_filter = text_entry.editing() == Some(TextEntryTarget::ControlsFilter);
    if sheet.editing_filter != editing_filter {
        sheet.editing_filter = editing_filter;
    }
}

/// How much of the list is in view and how far down it is scrolled, from
/// the last layout.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ListView {
    /// The window's height over the content's, at most 1.
    shown_fraction: f32,
    /// Farthest the list scrolls, in logical pixels.
    max_offset: f32,
}

impl ListView {
    fn of(window: &ComputedNode, content: &ComputedNode) -> Self {
        let (window_height, content_height) = (window.size().y, content.size().y);
        Self {
            shown_fraction: if content_height > 0.0 {
                (window_height / content_height).min(1.0)
            } else {
                1.0
            },
            max_offset: (content_height - window_height).max(0.0) * window.inverse_scale_factor(),
        }
    }

    fn scrolls(self) -> bool {
        self.max_offset >= 1.0
    }

    fn thumb_fraction(self) -> f32 {
        self.shown_fraction.max(MIN_THUMB_FRACTION)
    }

    /// Where the thumb's top stands down its track, as a fraction of it.
    fn thumb_top(self, offset: f32) -> f32 {
        if !self.scrolls() {
            return 0.0;
        }
        (offset / self.max_offset).clamp(0.0, 1.0) * (1.0 - self.thumb_fraction())
    }

    /// The offset that centers the thumb on `track_position` (0 at the
    /// track's top, 1 at its bottom).
    fn offset_at(self, track_position: f32) -> f32 {
        let travel = 1.0 - self.thumb_fraction();
        if travel <= 0.0 {
            return 0.0;
        }
        let top = track_position - self.thumb_fraction() * 0.5;
        (top / travel).clamp(0.0, 1.0) * self.max_offset
    }
}

type ListQuery<'w, 's> =
    Query<'w, 's, (&'static ComputedNode, &'static mut ScrollPosition), With<ControlsSheetList>>;

/// Scrolls the list with the mouse wheel while the sheet shows, and to
/// wherever its scrollbar's track is pressed or dragged; shows the
/// scrollbar while the list is taller than its window, its thumb over the
/// part in view.
pub fn scroll_controls_sheet(
    mut wheel: EventReader<MouseWheel>,
    pause_menu: Res<PauseMenuState>,
    mut lists: ListQuery,
    contents: Query<&ComputedNode, With<ControlsSheetListContent>>,
    mut scrollbars: Query<(
        &mut Node,
        &ControlsSheetScrollbar,
        Option<&Interaction>,
        Option<&RelativeCursorPosition>,
    )>,
) {
    let pixels: f32 = wheel
        .read()
        .map(|event| match event.unit {
            MouseScrollUnit::Line => event.y * WHEEL_LINE_PIXELS,
            MouseScrollUnit::Pixel => event.y,
        })
        .sum();
    let showing = pause_menu.paused && pause_menu.screen == PauseScreen::Controls;
    let (Ok((window, mut position)), Ok(content)) = (lists.get_single_mut(), contents.get_single())
    else {
        return;
    };
    let view = ListView::of(window, content);
    let dragged_to = scrollbars
        .iter()
        .find_map(|(_, part, interaction, cursor)| {
            (*part == ControlsSheetScrollbar::Track && interaction == Some(&Interaction::Pressed))
                .then(|| cursor.and_then(|cursor| cursor.normalized))
                .flatten()
        });
    let offset = match dragged_to {
        Some(cursor) if showing => view.offset_at(cursor.y),
        _ if showing => (position.offset_y - pixels).clamp(0.0, view.max_offset),
        _ => position.offset_y,
    };
    if position.offset_y != offset {
        position.offset_y = offset;
    }
    for (mut node, part, _, _) in &mut scrollbars {
        match part {
            ControlsSheetScrollbar::Track => {
                let display = display_if(view.scrolls());
                if node.display != display {
                    node.display = display;
                }
            }
            ControlsSheetScrollbar::Thumb => {
                let (top, height) = (
                    Val::Percent(view.thumb_top(offset) * 100.0),
                    Val::Percent(view.thumb_fraction() * 100.0),
                );
                if node.top != top || node.height != height {
                    node.top = top;
                    node.height = height;
                }
            }
        }
    }
}

/// Everything the sheet's texts and rows are drawn from.
type SheetTextQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut Text,
        &'static mut TextColor,
        &'static mut Node,
        &'static ControlsSheetText,
    ),
>;

/// Writes the bindings, the filter and the warnings into the sheet, and
/// shows the rows the filter matches.
pub fn update_controls_sheet(
    bindings: Res<ControlBindings>,
    editor: Res<BindingEditor>,
    sheet: Res<ControlsSheetState>,
    added: Query<(), Added<ControlsSheetText>>,
    mut texts: SheetTextQuery,
    mut parts: Query<(&mut Node, &ControlsSheetPart), Without<ControlsSheetText>>,
) {
    let changed = bindings.is_changed() || editor.is_changed() || sheet.is_changed();
    if !changed && added.is_empty() {
        return;
    }
    let conflicts = bindings.conflicts();
    let conflicted = |slot: BindingSlot| {
        bindings.slots(slot.action)[slot.slot].is_some_and(|chord| {
            conflicts.iter().any(|conflict| {
                conflict.actions.contains(&slot.action) && conflict.chord.overlaps(chord)
            })
        })
    };
    let needle = sheet.filter.trim().to_lowercase();
    let shown = |action: Action| needle.is_empty() || matches_filter(&bindings, action, &needle);
    let any_shown = Action::ALL.iter().any(|&action| shown(action));

    for (mut text, mut color, mut node, kind) in &mut texts {
        let (shown_text, shown_color) = match *kind {
            ControlsSheetText::Slot(slot) => slot_text(&bindings, &editor, slot, conflicted(slot)),
            ControlsSheetText::HoldMode(action) => (
                match bindings.hold_mode(action) {
                    HoldMode::Hold => "Hold".to_owned(),
                    HoldMode::Toggle => "Toggle".to_owned(),
                },
                TEXT_COLOR,
            ),
            ControlsSheetText::Follows(action) => (bindings.label(action), MUTED_COLOR),
            ControlsSheetText::Filter => {
                (filter_text(&sheet.filter, sheet.editing_filter), color.0)
            }
            ControlsSheetText::Warnings => {
                let warnings = warnings_text(&editor, &conflicts);
                set_display(&mut node, !warnings.is_empty());
                (warnings, color.0)
            }
            ControlsSheetText::NoMatches => {
                set_display(&mut node, !any_shown);
                continue;
            }
        };
        if text.0 != shown_text {
            text.0 = shown_text;
        }
        if color.0 != shown_color {
            color.0 = shown_color;
        }
    }
    for (mut node, part) in &mut parts {
        let part_shown = match *part {
            ControlsSheetPart::Row(action) => shown(action),
            ControlsSheetPart::Category(category) => Action::ALL
                .iter()
                .any(|&action| action.category() == category && shown(action)),
            ControlsSheetPart::CaptureShield => editor.capturing().is_some(),
        };
        set_display(&mut node, part_shown);
    }
}

fn set_display(node: &mut Mut<Node>, shown: bool) {
    let display = display_if(shown);
    if node.display != display {
        node.display = display;
    }
}

/// Whether `action`'s description, category, binding or the action it
/// follows holds `needle` (lower case).
fn matches_filter(bindings: &ControlBindings, action: Action, needle: &str) -> bool {
    let follows = action.shares_with().map(Action::description);
    [
        Some(action.description()),
        Some(action.category().title()),
        follows,
    ]
    .into_iter()
    .flatten()
    .map(str::to_lowercase)
    .chain([bindings.label(action).to_lowercase()])
    .any(|haystack| haystack.contains(needle))
}

fn slot_text(
    bindings: &ControlBindings,
    editor: &BindingEditor,
    slot: BindingSlot,
    conflicted: bool,
) -> (String, Color) {
    if editor.capturing() == Some(slot) {
        let prompt = match editor.pending() {
            Some(chord) => format!("{}, again?", chord.name()),
            None => "Press an input".to_owned(),
        };
        return (prompt, CAPTURING_COLOR);
    }
    match bindings.slots(slot.action)[slot.slot] {
        None => ("-".to_owned(), MUTED_COLOR),
        Some(chord) if conflicted => (chord.name(), CONFLICT_COLOR),
        Some(chord) => (chord.name(), BINDING_COLOR),
    }
}

fn filter_text(filter: &str, editing: bool) -> String {
    match (editing, filter.is_empty()) {
        (true, _) => format!("{filter}|"),
        (false, true) => "Click to search the controls".to_owned(),
        (false, false) => filter.to_owned(),
    }
}

fn warnings_text(
    editor: &BindingEditor,
    conflicts: &[crate::input_bindings::BindingConflict],
) -> String {
    let conflict_lines = conflicts.iter().take(SHOWN_CONFLICTS).map(|conflict| {
        let [first, second] = conflict.actions;
        format!(
            "{} does both \"{}\" and \"{}\".",
            conflict.chord.name(),
            first.description(),
            second.description()
        )
    });
    let more = conflicts
        .len()
        .checked_sub(SHOWN_CONFLICTS)
        .filter(|&more| more > 0)
        .map(|more| format!("{more} more conflicts."));
    editor
        .notice()
        .map(str::to_owned)
        .into_iter()
        .chain(conflict_lines)
        .chain(more)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Colors the sheet's buttons: the slot being captured and a held action
/// that toggles show as active, and a reset with nothing to undo as
/// disabled.
pub fn update_controls_sheet_colors(
    bindings: Res<ControlBindings>,
    editor: Res<BindingEditor>,
    mut buttons: Query<(&ControlsSheetButton, &Interaction, &mut BackgroundColor), With<Button>>,
) {
    for (button, interaction, color) in &mut buttons {
        let (available, active) = match *button {
            ControlsSheetButton::Slot(slot) => (true, editor.capturing() == Some(slot)),
            ControlsSheetButton::HoldMode(action) => {
                (true, bindings.hold_mode(action) == HoldMode::Toggle)
            }
            ControlsSheetButton::Reset(action) => (!bindings.is_default(action), false),
            ControlsSheetButton::ResetAll => (!bindings.all_default(), false),
            ControlsSheetButton::Filter => continue,
        };
        if available {
            set_button_color(color, active, interaction);
        } else {
            set_disabled_button_color(color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_filter_matches_descriptions_categories_and_bindings() {
        let bindings = ControlBindings::default();
        assert!(matches_filter(&bindings, Action::Undo, "undo"));
        assert!(matches_filter(&bindings, Action::Undo, "editing"));
        assert!(matches_filter(&bindings, Action::Undo, "ctrl+z"));
        assert!(!matches_filter(&bindings, Action::Undo, "folder"));
        // A shared row is found by what it follows.
        assert!(matches_filter(
            &bindings,
            Action::AddToSelection,
            "move faster"
        ));
    }

    #[test]
    fn the_thumb_spans_the_part_in_view_and_a_drag_scrolls_to_it() {
        let view = ListView {
            shown_fraction: 0.25,
            max_offset: 300.0,
        };
        assert!(view.scrolls());
        assert_eq!(view.thumb_top(0.0), 0.0);
        assert_eq!(view.thumb_top(300.0), 0.75);
        assert_eq!(view.thumb_top(150.0), 0.375);
        // Pressing the track centers the thumb there.
        assert_eq!(view.offset_at(0.5), 150.0);
        assert_eq!(view.offset_at(0.0), 0.0);
        assert_eq!(view.offset_at(1.0), 300.0);

        let short = ListView {
            shown_fraction: 1.0,
            max_offset: 0.0,
        };
        assert!(!short.scrolls());
        assert_eq!(short.offset_at(0.7), 0.0);
    }

    #[test]
    fn every_category_has_a_row() {
        for category in ActionCategory::ALL {
            assert!(
                Action::ALL
                    .iter()
                    .any(|action| action.category() == category),
                "{category:?} has nothing to list"
            );
        }
    }
}
