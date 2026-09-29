//! Right-click menus: a small screen-space menu opened at the pointer for
//! whatever was clicked.
//!
//! The UI owns the widget and nothing about what it controls. Whoever owns
//! the clicked thing describes its menu as a [`ContextMenuModel`] whose items
//! carry that owner's own command type `C`, keeps the model current while
//! the menu is open ([`ContextMenu::update`]), and takes the commands the
//! user activates ([`ContextMenu::take_activated`]). Add one
//! [`ContextMenuPlugin`] per command type.
//!
//! The widget is rebuilt whenever what it shows changes: menus are a few
//! dozen nodes, and a press is always handled before the rebuild that
//! follows it within a frame.

use std::marker::PhantomData;

use bevy::prelude::*;
use bevy::transform::TransformSystem;
use bevy::ui::{FocusPolicy, UiSystem};
use bevy::window::PrimaryWindow;

use crate::{button_color, header_button_color, update_ui_input_capture, UiInputCapture};

const MENU_WIDTH: f32 = 300.0;
const SUBMENU_WIDTH: f32 = 280.0;
/// Closest the menu comes to a window edge.
const WINDOW_MARGIN: f32 = 8.0;
const ROW_MIN_HEIGHT: f32 = 26.0;
const STEP_BUTTON_WIDTH: f32 = 26.0;
const RESET_BUTTON_WIDTH: f32 = 48.0;
/// Wide enough for the values steppers show, so the buttons stay put as
/// the value changes.
const STEPPER_VALUE_MIN_WIDTH: f32 = 64.0;
/// Above every other UI.
const MENU_Z_INDEX: i32 = 100;
const PANEL_COLOR: Color = Color::srgba(0.035, 0.040, 0.052, 0.97);
const BORDER_COLOR: Color = Color::srgba(0.78, 0.84, 0.92, 0.60);
const TEXT_COLOR: Color = Color::srgb(0.94, 0.96, 0.98);
const VALUE_COLOR: Color = Color::srgb(0.72, 0.80, 0.90);
const HEADING_COLOR: Color = Color::srgba(0.62, 0.68, 0.76, 0.90);
const DISABLED_TEXT_COLOR: Color = Color::srgba(0.60, 0.64, 0.70, 0.55);
const TITLE_FONT_SIZE: f32 = 14.0;
const HEADING_FONT_SIZE: f32 = 11.0;
const ROW_FONT_SIZE: f32 = 13.0;

/// What a menu shows.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextMenuModel<C> {
    pub title: String,
    pub sections: Vec<ContextMenuSection<C>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextMenuSection<C> {
    pub heading: Option<String>,
    pub items: Vec<ContextMenuItem<C>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ContextMenuItem<C> {
    /// Runs `command` and closes the menu.
    Action { label: String, command: C },
    /// A setting shown on or off; clicking runs `command` and keeps the menu
    /// open.
    Toggle { label: String, on: bool, command: C },
    /// A row showing the current `value` that opens a submenu of
    /// `options`; picking one runs its command and closes the menu.
    Choice {
        label: String,
        value: String,
        options: Vec<ContextMenuOption<C>>,
    },
    /// A value stepped with − and + buttons, with a reset button while
    /// `reset` is offered; every button keeps the menu open.
    Stepper {
        label: String,
        value: String,
        decrease: C,
        increase: C,
        reset: Option<C>,
    },
    /// A read-only fact.
    Info { label: String, value: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContextMenuOption<C> {
    pub label: String,
    pub selected: bool,
    /// `None` shows the option dimmed and unpickable.
    pub command: Option<C>,
}

/// The one open menu of command type `C`, and the commands activated since
/// they were last taken.
#[derive(Resource)]
pub struct ContextMenu<C> {
    open: Option<OpenContextMenu<C>>,
    activated: Vec<C>,
    /// Bumped whenever what the widget shows changes.
    revision: u64,
}

struct OpenContextMenu<C> {
    /// Where the pointer opened the menu, in logical window pixels.
    anchor: Vec2,
    /// The menu's top-left once it has been fitted inside the window.
    placed_at: Option<Vec2>,
    model: ContextMenuModel<C>,
    submenu: Option<ItemAddress>,
    /// The open submenu's offset from its row once fitted inside the
    /// window, so a rebuild shows it in place rather than fitting it again.
    submenu_offset: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ItemAddress {
    section: usize,
    item: usize,
}

impl<C> Default for ContextMenu<C> {
    fn default() -> Self {
        Self {
            open: None,
            activated: Vec::new(),
            revision: 0,
        }
    }
}

impl<C: Clone + PartialEq> ContextMenu<C> {
    /// Opens the menu at `anchor` (logical window pixels), replacing any
    /// open one.
    pub fn open(&mut self, anchor: Vec2, model: ContextMenuModel<C>) {
        self.open = Some(OpenContextMenu {
            anchor,
            placed_at: None,
            model,
            submenu: None,
            submenu_offset: None,
        });
        self.revision += 1;
    }

    pub fn close(&mut self) {
        if self.open.take().is_some() {
            self.revision += 1;
        }
    }

    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// Replaces what the open menu shows, keeping its place and its open
    /// submenu while that is still a choice.
    pub fn update(&mut self, model: ContextMenuModel<C>) {
        let Some(open) = self.open.as_mut() else {
            return;
        };
        if open.model == model {
            return;
        }
        open.model = model;
        if open.submenu.is_some_and(|address| {
            !matches!(
                item_at(&open.model, address),
                Some(ContextMenuItem::Choice { .. })
            )
        }) {
            open.submenu = None;
        }
        self.revision += 1;
    }

    /// The commands activated since the last call, oldest first.
    pub fn take_activated(&mut self) -> Vec<C> {
        std::mem::take(&mut self.activated)
    }

    /// Hands `command` over as if picked from the menu, closing it when
    /// `close`.
    pub fn activate(&mut self, command: C, close: bool) {
        self.activated.push(command);
        if close {
            self.close();
        }
    }

    fn show_submenu(&mut self, submenu: Option<ItemAddress>) {
        let Some(open) = self.open.as_mut() else {
            return;
        };
        if open.submenu != submenu {
            open.submenu = submenu;
            open.submenu_offset = None;
            self.revision += 1;
        }
    }
}

fn item_at<C>(model: &ContextMenuModel<C>, address: ItemAddress) -> Option<&ContextMenuItem<C>> {
    model.sections.get(address.section)?.items.get(address.item)
}

/// Orders a menu's systems: owners apply the activated commands and refresh
/// the model between the two.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContextMenuSystems {
    /// Presses, hovers and dismissals are read.
    Input,
    /// The widget catches up with the menu.
    Render,
}

pub struct ContextMenuPlugin<C>(PhantomData<C>);

impl<C> Default for ContextMenuPlugin<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: Clone + PartialEq + Send + Sync + 'static> Plugin for ContextMenuPlugin<C> {
    fn build(&self, app: &mut App) {
        app.init_resource::<ContextMenu<C>>()
            .configure_sets(
                Update,
                ContextMenuSystems::Input.before(ContextMenuSystems::Render),
            )
            .add_systems(
                PreUpdate,
                capture_pointer_for_context_menu::<C>.after(update_ui_input_capture),
            )
            .add_systems(
                Update,
                (
                    handle_context_menu_input::<C>.in_set(ContextMenuSystems::Input),
                    (
                        sync_context_menu_widget::<C>,
                        color_context_menu_buttons::<C>,
                    )
                        .chain()
                        .in_set(ContextMenuSystems::Render),
                ),
            )
            // Fitting reads where layout put each panel on screen.
            .add_systems(
                PostUpdate,
                place_context_menu::<C>
                    .after(UiSystem::Layout)
                    .after(TransformSystem::TransformPropagate),
            );
    }
}

/// Every node of a menu's widget; the pointer over any of them is over the
/// menu.
#[derive(Component)]
struct ContextMenuNode<C>(PhantomData<C>);

impl<C> Default for ContextMenuNode<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// The widget's outer panel, built for one revision of the menu.
#[derive(Component)]
struct ContextMenuRoot<C> {
    revision: u64,
    marker: PhantomData<C>,
}

/// A submenu's panel, fitted inside the window like the menu itself.
#[derive(Component)]
struct ContextMenuSubmenu<C>(PhantomData<C>);

/// A row of the main panel; hovering one shows its submenu, if any, and
/// hides any other.
#[derive(Component)]
struct ContextMenuRow<C> {
    submenu: Option<ItemAddress>,
    marker: PhantomData<C>,
}

#[derive(Component)]
struct ContextMenuButton<C> {
    press: ButtonPress<C>,
    /// Drawn as the selected option, or the row whose submenu is open.
    active: bool,
}

enum ButtonPress<C> {
    Run {
        command: C,
        close: bool,
    },
    /// Shows the row's submenu; hovering already does, so this only
    /// matters without a hover (a touch screen).
    ShowSubmenu(ItemAddress),
    /// An unavailable option.
    Nothing,
}

/// Counts the open menu as a menu that owns the pointer, so world clicks
/// and the right-drag look stand down while it is open.
fn capture_pointer_for_context_menu<C: Clone + PartialEq + Send + Sync + 'static>(
    menu: Res<ContextMenu<C>>,
    mut capture: ResMut<UiInputCapture>,
) {
    capture.context_menu_open |= menu.is_open();
}

type RowQuery<'w, 's, C> =
    Query<'w, 's, (&'static Interaction, &'static ContextMenuRow<C>), Changed<Interaction>>;
type ButtonQuery<'w, 's, C> =
    Query<'w, 's, (&'static Interaction, &'static ContextMenuButton<C>), Changed<Interaction>>;

fn handle_context_menu_input<C: Clone + PartialEq + Send + Sync + 'static>(
    mut menu: ResMut<ContextMenu<C>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    rows: RowQuery<C>,
    buttons: ButtonQuery<C>,
    menu_nodes: Query<&Interaction, With<ContextMenuNode<C>>>,
) {
    if !menu.is_open() {
        return;
    }
    if keyboard.just_pressed(KeyCode::Escape) {
        menu.close();
        return;
    }
    for (interaction, row) in &rows {
        if *interaction != Interaction::None {
            menu.show_submenu(row.submenu);
        }
    }
    for (interaction, button) in &buttons {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match &button.press {
            ButtonPress::Run { command, close } => menu.activate(command.clone(), *close),
            ButtonPress::ShowSubmenu(address) => menu.show_submenu(Some(*address)),
            ButtonPress::Nothing => {}
        }
    }
    // A press anywhere else dismisses the menu. It does not reach the world
    // (the pointer capture keeps world clicks off while the menu is open),
    // but a press on other UI still acts on that UI.
    let pressed = mouse_buttons.get_just_pressed().next().is_some();
    if pressed
        && menu.is_open()
        && menu_nodes
            .iter()
            .all(|interaction| *interaction == Interaction::None)
    {
        menu.close();
    }
}

fn sync_context_menu_widget<C: Clone + PartialEq + Send + Sync + 'static>(
    mut commands: Commands,
    menu: Res<ContextMenu<C>>,
    roots: Query<(Entity, &ContextMenuRoot<C>)>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let mut current = false;
    for (entity, root) in &roots {
        if root.revision == menu.revision {
            current = true;
        } else {
            commands.entity(entity).despawn_recursive();
        }
    }
    if current {
        return;
    }
    let (Some(open), Ok(window)) = (menu.open.as_ref(), windows.get_single()) else {
        return;
    };
    spawn_menu_widget(&mut commands, open, menu.revision, window.size());
}

type SubmenuPlacementQuery<'w, 's, C> = Query<
    'w,
    's,
    (
        &'static ComputedNode,
        &'static GlobalTransform,
        &'static mut Node,
        &'static mut Visibility,
    ),
    (With<ContextMenuSubmenu<C>>, Without<ContextMenuRoot<C>>),
>;

/// Fits the menu, and its open submenu, inside the window once their sizes
/// are known. Each stays hidden until it stands where it will be shown.
fn place_context_menu<C: Clone + PartialEq + Send + Sync + 'static>(
    mut menu: ResMut<ContextMenu<C>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut roots: Query<(&ComputedNode, &mut Node, &mut Visibility), With<ContextMenuRoot<C>>>,
    mut submenus: SubmenuPlacementQuery<C>,
) {
    let (Some(open), Ok(window)) = (menu.open.as_mut(), windows.get_single()) else {
        return;
    };
    for (computed, mut node, mut visibility) in &mut roots {
        let size = computed.size() * computed.inverse_scale_factor();
        if size == Vec2::ZERO {
            continue;
        }
        let limit = (window.size() - size - WINDOW_MARGIN).max(Vec2::splat(WINDOW_MARGIN));
        let placed = open.anchor.clamp(Vec2::splat(WINDOW_MARGIN), limit);
        if node.left == Val::Px(placed.x) && node.top == Val::Px(placed.y) {
            open.placed_at = Some(placed);
            visibility.set_if_neq(Visibility::Inherited);
        } else {
            node.left = Val::Px(placed.x);
            node.top = Val::Px(placed.y);
        }
    }
    // A submenu hangs from its row, so only its vertical offset from the
    // row is moved: up when it would run off the bottom, down when off the
    // top.
    for (computed, transform, mut node, mut visibility) in &mut submenus {
        let scale = computed.inverse_scale_factor();
        let height = computed.size().y * scale;
        if height == 0.0 {
            continue;
        }
        let top = transform.translation().y * scale - height * 0.5;
        let lowest_top = (window.height() - WINDOW_MARGIN - height).max(WINDOW_MARGIN);
        let shift = top.clamp(WINDOW_MARGIN, lowest_top) - top;
        let offset = match node.top {
            Val::Px(offset) => offset,
            _ => 0.0,
        };
        if shift.abs() < 0.5 {
            open.submenu_offset = Some(offset);
            visibility.set_if_neq(Visibility::Inherited);
        } else {
            node.top = Val::Px(offset + shift);
        }
    }
}

/// Menu buttons whose hover changed, or that were just built.
type ButtonColorQuery<'w, 's, C> = Query<
    'w,
    's,
    (
        &'static Interaction,
        &'static ContextMenuButton<C>,
        &'static mut BackgroundColor,
    ),
    Or<(Changed<Interaction>, Added<ContextMenuButton<C>>)>,
>;

fn color_context_menu_buttons<C: Clone + PartialEq + Send + Sync + 'static>(
    mut buttons: ButtonColorQuery<C>,
) {
    for (interaction, button, mut color) in &mut buttons {
        let hovered =
            *interaction != Interaction::None && !matches!(button.press, ButtonPress::Nothing);
        *color = if button.active {
            button_color(true, hovered)
        } else {
            header_button_color(hovered)
        }
        .into();
    }
}

fn spawn_menu_widget<C: Clone + PartialEq + Send + Sync + 'static>(
    commands: &mut Commands,
    open: &OpenContextMenu<C>,
    revision: u64,
    window_size: Vec2,
) {
    let (position, visibility) = match open.placed_at {
        Some(placed) => (placed, Visibility::Inherited),
        None => (open.anchor, Visibility::Hidden),
    };
    // Submenus open toward the side with room for them.
    let submenu_placement = SubmenuPlacement {
        open_left: position.x + MENU_WIDTH + SUBMENU_WIDTH > window_size.x,
        fitted_offset: open.submenu_offset,
    };
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(position.x),
                top: Val::Px(position.y),
                width: Val::Px(MENU_WIDTH),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(6.0)),
                row_gap: Val::Px(1.0),
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderRadius::all(Val::Px(8.0)),
            BorderColor(BORDER_COLOR),
            BackgroundColor(PANEL_COLOR),
            GlobalZIndex(MENU_Z_INDEX),
            visibility,
            Interaction::default(),
            FocusPolicy::Block,
            ContextMenuNode::<C>::default(),
            ContextMenuRoot::<C> {
                revision,
                marker: PhantomData,
            },
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new(open.model.title.clone()),
                text_font(TITLE_FONT_SIZE),
                TextColor(TEXT_COLOR),
                Node {
                    padding: UiRect::new(Val::Px(6.0), Val::Px(6.0), Val::Px(2.0), Val::Px(4.0)),
                    ..default()
                },
            ));
            for (section_index, section) in open.model.sections.iter().enumerate() {
                if let Some(heading) = &section.heading {
                    panel.spawn((
                        Text::new(heading.to_uppercase()),
                        text_font(HEADING_FONT_SIZE),
                        TextColor(HEADING_COLOR),
                        Node {
                            padding: UiRect::new(
                                Val::Px(6.0),
                                Val::Px(6.0),
                                Val::Px(6.0),
                                Val::Px(1.0),
                            ),
                            ..default()
                        },
                    ));
                }
                for (item_index, item) in section.items.iter().enumerate() {
                    let address = ItemAddress {
                        section: section_index,
                        item: item_index,
                    };
                    spawn_item(panel, item, address, open.submenu, submenu_placement);
                }
            }
        });
}

fn spawn_item<C: Clone + PartialEq + Send + Sync + 'static>(
    panel: &mut ChildBuilder,
    item: &ContextMenuItem<C>,
    address: ItemAddress,
    open_submenu: Option<ItemAddress>,
    submenu_placement: SubmenuPlacement,
) {
    let row = |submenu: Option<ItemAddress>| {
        (
            Node {
                width: Val::Percent(100.0),
                min_height: Val::Px(ROW_MIN_HEIGHT),
                padding: UiRect::horizontal(Val::Px(6.0)),
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                ..default()
            },
            BorderRadius::all(Val::Px(5.0)),
            BackgroundColor(Color::NONE),
            Interaction::default(),
            ContextMenuNode::<C>::default(),
            ContextMenuRow::<C> {
                submenu,
                marker: PhantomData,
            },
        )
    };
    match item {
        ContextMenuItem::Action { label, command } => {
            panel
                .spawn((
                    row(None),
                    Button,
                    ContextMenuButton {
                        press: ButtonPress::Run {
                            command: command.clone(),
                            close: true,
                        },
                        active: false,
                    },
                ))
                .with_children(|row| spawn_label(row, label, TEXT_COLOR));
        }
        ContextMenuItem::Toggle { label, on, command } => {
            panel
                .spawn((
                    row(None),
                    Button,
                    ContextMenuButton {
                        press: ButtonPress::Run {
                            command: command.clone(),
                            close: false,
                        },
                        active: false,
                    },
                ))
                .with_children(|row| {
                    spawn_label(row, label, TEXT_COLOR);
                    spawn_label(row, if *on { "On" } else { "Off" }, VALUE_COLOR);
                });
        }
        ContextMenuItem::Choice {
            label,
            value,
            options,
        } => {
            let submenu_open = open_submenu == Some(address);
            panel
                .spawn((
                    row(Some(address)),
                    Button,
                    ContextMenuButton::<C> {
                        press: ButtonPress::ShowSubmenu(address),
                        active: submenu_open,
                    },
                ))
                .with_children(|row| {
                    spawn_label(row, label, TEXT_COLOR);
                    spawn_label(row, &format!("{value}  >"), VALUE_COLOR);
                    if submenu_open {
                        spawn_submenu(row, options, submenu_placement);
                    }
                });
        }
        ContextMenuItem::Stepper {
            label,
            value,
            decrease,
            increase,
            reset,
        } => {
            panel.spawn(row(None)).with_children(|row| {
                spawn_label(row, label, TEXT_COLOR);
                row.spawn(Node {
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(4.0),
                    ..default()
                })
                .with_children(|steps| {
                    if let Some(reset) = reset {
                        spawn_step_button(steps, "Reset", reset, RESET_BUTTON_WIDTH);
                    }
                    spawn_step_button(steps, "-", decrease, STEP_BUTTON_WIDTH);
                    steps
                        .spawn(Node {
                            min_width: Val::Px(STEPPER_VALUE_MIN_WIDTH),
                            justify_content: JustifyContent::Center,
                            ..default()
                        })
                        .with_children(|value_box| spawn_label(value_box, value, VALUE_COLOR));
                    spawn_step_button(steps, "+", increase, STEP_BUTTON_WIDTH);
                });
            });
        }
        ContextMenuItem::Info { label, value } => {
            panel.spawn(row(None)).with_children(|row| {
                spawn_label(row, label, HEADING_COLOR);
                row.spawn((
                    Text::new(value.clone()),
                    text_font(ROW_FONT_SIZE),
                    TextColor(VALUE_COLOR),
                    TextLayout::new_with_justify(JustifyText::Right),
                    Node {
                        max_width: Val::Px(MENU_WIDTH * 0.62),
                        ..default()
                    },
                ));
            });
        }
    }
}

/// Where an open submenu goes: to the side of its row with room for it,
/// and, once fitted inside the window, at the offset it was fitted to.
#[derive(Clone, Copy)]
struct SubmenuPlacement {
    open_left: bool,
    fitted_offset: Option<f32>,
}

fn spawn_submenu<C: Clone + PartialEq + Send + Sync + 'static>(
    row: &mut ChildBuilder,
    options: &[ContextMenuOption<C>],
    placement: SubmenuPlacement,
) {
    let (left, right) = if placement.open_left {
        (Val::Auto, Val::Percent(100.0))
    } else {
        (Val::Percent(100.0), Val::Auto)
    };
    row.spawn((
        Node {
            position_type: PositionType::Absolute,
            left,
            right,
            top: Val::Px(placement.fitted_offset.unwrap_or(0.0)),
            width: Val::Px(SUBMENU_WIDTH),
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(Val::Px(6.0)),
            row_gap: Val::Px(1.0),
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BorderRadius::all(Val::Px(8.0)),
        BorderColor(BORDER_COLOR),
        BackgroundColor(PANEL_COLOR),
        // Shown once fitted inside the window.
        if placement.fitted_offset.is_some() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        },
        Interaction::default(),
        FocusPolicy::Block,
        ContextMenuNode::<C>::default(),
        ContextMenuSubmenu::<C>(PhantomData),
    ))
    .with_children(|submenu| {
        for option in options {
            let (press, text_color) = match &option.command {
                Some(command) => (
                    ButtonPress::Run {
                        command: command.clone(),
                        close: true,
                    },
                    TEXT_COLOR,
                ),
                None => (ButtonPress::Nothing, DISABLED_TEXT_COLOR),
            };
            submenu
                .spawn((
                    Button,
                    Node {
                        width: Val::Percent(100.0),
                        min_height: Val::Px(ROW_MIN_HEIGHT),
                        padding: UiRect::horizontal(Val::Px(8.0)),
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BorderRadius::all(Val::Px(5.0)),
                    BackgroundColor(Color::NONE),
                    ContextMenuNode::<C>::default(),
                    ContextMenuButton {
                        press,
                        active: option.selected,
                    },
                ))
                .with_children(|button| spawn_label(button, &option.label, text_color));
        }
    });
}

fn spawn_step_button<C: Clone + PartialEq + Send + Sync + 'static>(
    parent: &mut ChildBuilder,
    label: &str,
    command: &C,
    width: f32,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(width),
                height: Val::Px(22.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderRadius::all(Val::Px(5.0)),
            BorderColor(Color::srgba(0.7, 0.76, 0.84, 0.55)),
            BackgroundColor(Color::NONE),
            ContextMenuNode::<C>::default(),
            ContextMenuButton {
                press: ButtonPress::Run {
                    command: command.clone(),
                    close: false,
                },
                active: false,
            },
        ))
        .with_children(|button| spawn_label(button, label, TEXT_COLOR));
}

fn spawn_label(parent: &mut ChildBuilder, label: &str, color: Color) {
    parent.spawn((Text::new(label), text_font(ROW_FONT_SIZE), TextColor(color)));
}

fn text_font(font_size: f32) -> TextFont {
    TextFont {
        font_size,
        ..default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    enum Command {
        Loop,
        Speed(u8),
    }

    fn model(speed_value: &str) -> ContextMenuModel<Command> {
        ContextMenuModel {
            title: "clip.mp4".to_owned(),
            sections: vec![ContextMenuSection {
                heading: None,
                items: vec![
                    ContextMenuItem::Toggle {
                        label: "Loop".to_owned(),
                        on: true,
                        command: Command::Loop,
                    },
                    ContextMenuItem::Choice {
                        label: "Speed".to_owned(),
                        value: speed_value.to_owned(),
                        options: vec![ContextMenuOption {
                            label: "2x".to_owned(),
                            selected: false,
                            command: Some(Command::Speed(2)),
                        }],
                    },
                ],
            }],
        }
    }

    const SPEED: ItemAddress = ItemAddress {
        section: 0,
        item: 1,
    };

    #[test]
    fn an_unchanged_model_leaves_the_widget_alone() {
        let mut menu = ContextMenu::default();
        menu.open(Vec2::ZERO, model("1x"));
        let revision = menu.revision;
        menu.update(model("1x"));
        assert_eq!(menu.revision, revision);
        menu.update(model("2x"));
        assert!(menu.revision > revision);
    }

    #[test]
    fn a_submenu_stays_open_across_updates_while_it_is_a_choice() {
        let mut menu = ContextMenu::default();
        menu.open(Vec2::ZERO, model("1x"));
        menu.show_submenu(Some(SPEED));
        menu.update(model("2x"));
        assert_eq!(menu.open.as_ref().expect("open").submenu, Some(SPEED));

        let mut shorter = model("1x");
        shorter.sections[0].items.pop();
        menu.update(shorter);
        assert_eq!(menu.open.as_ref().expect("open").submenu, None);
    }

    #[test]
    fn activating_hands_the_command_over_and_closes_only_when_asked() {
        let mut menu = ContextMenu::default();
        menu.open(Vec2::ZERO, model("1x"));
        menu.activate(Command::Loop, false);
        assert!(menu.is_open());
        menu.activate(Command::Speed(2), true);
        assert!(!menu.is_open());
        assert_eq!(menu.take_activated(), [Command::Loop, Command::Speed(2)]);
        assert!(menu.take_activated().is_empty());
    }
}
