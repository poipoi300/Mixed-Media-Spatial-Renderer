//! Every control the viewer answers to, in one table, and what each one is
//! bound to now.
//!
//! Each [`Action`] names one thing the user can do and carries its category,
//! its description and its default binding. Input handling asks for an
//! action ([`ControlInput`]) rather than for a key or button, and the
//! controls sheet lists [`Action::ALL`], so the two cannot drift apart: a
//! control added to the viewer is added here, and then it shows.
//!
//! An action is bound to up to [`SLOT_COUNT`] [`Chord`]s: an input, the
//! modifiers held with it, and whether it is tapped twice. The user rebinds
//! them on the controls sheet (see [`BindingEditor`]), and a held control
//! can be made to toggle instead. [`ControlBindings`] keeps what they chose.
//!
//! Some actions share another action's input with a different gesture (a
//! drag of the button that selects on a click, say). They name that action
//! ([`BindingSource::SharedWith`]), so they follow whatever it is bound to.

mod capture;
mod keys;

pub use capture::{capture_binding, BindingEditor, BindingSlot};

use std::collections::{HashMap, HashSet};

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use crate::{ControlPanelState, TextEntry};

/// How many bindings each action can have.
pub const SLOT_COUNT: usize = 2;

/// An action's bindings, one per slot; an empty slot is `None`.
pub type Slots = [Option<Chord>; SLOT_COUNT];

/// Longest gap between the two presses of a double-tap.
const DOUBLE_TAP_SECONDS: f64 = 0.3;

/// Whether text is being typed somewhere (into a control panel field or a
/// [`TextEntry`]), or a binding is being captured. Refreshed by
/// [`update_typing_focus`] once the UI has handled this frame's presses, so
/// [`ControlInput`] need not borrow the state it is read from, which the
/// systems that end typing change.
#[derive(Resource, Default)]
pub struct TypingFocus {
    typing: bool,
    capturing: bool,
}

pub fn update_typing_focus(
    control_panel: Res<ControlPanelState>,
    text_entry: Res<TextEntry>,
    editor: Res<BindingEditor>,
    mut focus: ResMut<TypingFocus>,
) {
    let typing = control_panel.input_focused() || text_entry.is_active();
    let capturing = editor.holds_input();
    if focus.typing != typing || focus.capturing != capturing {
        focus.typing = typing;
        focus.capturing = capturing;
    }
}

/// A modifier, pressed with either of its two keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modifier {
    Ctrl,
    Shift,
    Alt,
}

impl Modifier {
    /// In the order chords name them.
    pub const ALL: [Self; 3] = [Self::Ctrl, Self::Shift, Self::Alt];

    fn keys(self) -> [KeyCode; 2] {
        match self {
            Self::Ctrl => [KeyCode::ControlLeft, KeyCode::ControlRight],
            Self::Shift => [KeyCode::ShiftLeft, KeyCode::ShiftRight],
            Self::Alt => [KeyCode::AltLeft, KeyCode::AltRight],
        }
    }

    /// The modifier `key` is one of the keys of.
    fn of_key(key: KeyCode) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|modifier| modifier.keys().contains(&key))
    }

    fn name(self) -> &'static str {
        match self {
            Self::Ctrl => "Ctrl",
            Self::Shift => "Shift",
            Self::Alt => "Alt",
        }
    }

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// The modifiers held with a chord's input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Modifiers(u8);

impl Modifiers {
    pub const NONE: Self = Self(0);
    const CTRL: Self = Self(Modifier::Ctrl.bit());
    const CTRL_SHIFT: Self = Self(Modifier::Ctrl.bit() | Modifier::Shift.bit());

    pub fn with(self, modifier: Modifier) -> Self {
        Self(self.0 | modifier.bit())
    }

    pub fn contains(self, modifier: Modifier) -> bool {
        self.0 & modifier.bit() != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn is_within(self, other: Self) -> bool {
        self.0 & other.0 == self.0
    }

    fn iter(self) -> impl Iterator<Item = Modifier> {
        Modifier::ALL
            .into_iter()
            .filter(move |&modifier| self.contains(modifier))
    }

    /// The modifiers held on `keys` now.
    fn held(keys: &ButtonInput<KeyCode>) -> Self {
        Modifier::ALL
            .into_iter()
            .filter(|modifier| keys.any_pressed(modifier.keys()))
            .fold(Self::NONE, Self::with)
    }
}

/// One physical input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    Key(KeyCode),
    /// Either key of a modifier.
    Modifier(Modifier),
    Mouse(MouseButton),
    Wheel,
}

impl Input {
    /// Whether pressing one presses the other: they are the same, or one is
    /// a modifier and the other one of its keys.
    fn overlaps(self, other: Self) -> bool {
        match (self, other) {
            (Self::Modifier(modifier), Self::Key(key))
            | (Self::Key(key), Self::Modifier(modifier)) => modifier.keys().contains(&key),
            _ => self == other,
        }
    }

    fn is_keyboard(self) -> bool {
        matches!(self, Self::Key(_) | Self::Modifier(_))
    }

    fn name(self) -> String {
        match self {
            Self::Key(key) => keys::key_name(key),
            Self::Modifier(modifier) => modifier.name().to_owned(),
            Self::Mouse(button) => keys::button_name(button),
            Self::Wheel => "Mouse wheel".to_owned(),
        }
    }

    fn saved_name(self) -> Option<String> {
        match self {
            Self::Key(key) => Some(keys::saved_key_name(key)),
            Self::Modifier(modifier) => Some(modifier.name().to_owned()),
            Self::Mouse(button) => {
                keys::saved_button_name(button).map(|name| format!("{MOUSE_PREFIX}{name}"))
            }
            Self::Wheel => None,
        }
    }

    fn from_saved_name(name: &str) -> Option<Self> {
        if let Some(button) = name.strip_prefix(MOUSE_PREFIX) {
            return keys::button_from_saved_name(button).map(Self::Mouse);
        }
        Modifier::ALL
            .into_iter()
            .find(|modifier| modifier.name() == name)
            .map(Self::Modifier)
            .or_else(|| keys::key_from_saved_name(name).map(Self::Key))
    }
}

const MOUSE_PREFIX: &str = "Mouse";
const DOUBLE_TAP_TOKEN: &str = "Double";

/// What one binding slot holds: an input, the modifiers held with it, and
/// whether it is tapped twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Chord {
    pub modifiers: Modifiers,
    pub input: Input,
    /// Pressed twice in quick succession; held after the second press.
    pub double_tap: bool,
}

impl Chord {
    pub const fn plain(input: Input) -> Self {
        Self {
            modifiers: Modifiers::NONE,
            input,
            double_tap: false,
        }
    }

    /// Whether one of them going down also takes the other: overlapping
    /// inputs, held with the same modifiers and tapped the same way.
    pub fn overlaps(self, other: Self) -> bool {
        self.modifiers == other.modifiers
            && self.double_tap == other.double_tap
            && self.input.overlaps(other.input)
    }

    /// How the controls sheet writes it, such as `Ctrl+Shift+Z`.
    pub fn name(self) -> String {
        let chord: Vec<String> = self
            .modifiers
            .iter()
            .map(|modifier| modifier.name().to_owned())
            .chain([self.input.name()])
            .collect();
        let chord = chord.join("+");
        match (self.double_tap, self.input) {
            (false, _) => chord,
            (true, Input::Mouse(_)) => format!("Double-click {chord}"),
            (true, _) => format!("Double-tap {chord}"),
        }
    }

    /// How a saved file writes it: tokens joined by `+`, the input last,
    /// named so no keyboard layout changes it. `None` for an input no file
    /// can hold (the wheel is never rebound).
    pub fn to_saved(self) -> Option<String> {
        let tokens: Vec<String> = self
            .double_tap
            .then(|| DOUBLE_TAP_TOKEN.to_owned())
            .into_iter()
            .chain(
                self.modifiers
                    .iter()
                    .map(|modifier| modifier.name().to_owned()),
            )
            .chain([self.input.saved_name()?])
            .collect();
        Some(tokens.join("+"))
    }

    pub fn from_saved(saved: &str) -> Option<Self> {
        let mut tokens: Vec<&str> = saved.split('+').collect();
        let input = Input::from_saved_name(tokens.pop()?)?;
        let mut chord = Self::plain(input);
        for token in tokens {
            if token == DOUBLE_TAP_TOKEN {
                chord.double_tap = true;
                continue;
            }
            let modifier = Modifier::ALL
                .into_iter()
                .find(|modifier| modifier.name() == token)?;
            chord.modifiers = chord.modifiers.with(modifier);
        }
        Some(chord)
    }
}

/// How an action uses its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gesture {
    Press,
    /// Held down while something else happens, or for as long as it lasts.
    Hold,
    Click,
    Drag,
    Scroll,
}

/// Whether a held action lasts while its input is held, or turns on with
/// one press and off with the next.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HoldMode {
    #[default]
    Hold,
    Toggle,
}

/// The sections of the controls sheet, in the order it shows them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionCategory {
    Movement,
    Selection,
    Editing,
    Folders,
    Video,
    Interface,
}

impl ActionCategory {
    pub const ALL: [Self; 6] = [
        Self::Movement,
        Self::Selection,
        Self::Editing,
        Self::Folders,
        Self::Video,
        Self::Interface,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Movement => "Movement",
            Self::Selection => "Selection",
            Self::Editing => "Editing",
            Self::Folders => "Folders",
            Self::Video => "Video",
            Self::Interface => "Interface",
        }
    }
}

/// When an action is read, which decides which bindings get in each
/// other's way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionContext {
    /// In the scene, while no menu is open and nothing is being typed.
    World,
    /// In menus and while typing too: the keys that leave them.
    Everywhere,
}

impl ActionContext {
    fn overlaps(self, other: Self) -> bool {
        self == other || self == Self::Everywhere || other == Self::Everywhere
    }
}

/// Where an action's input comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingSource {
    /// These chords, one per slot.
    Own(&'static [Chord]),
    /// Whatever the named action is bound to, used with this action's own
    /// gesture.
    SharedWith(Action),
}

/// An action's binding as the viewer ships it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefaultBinding {
    pub gesture: Gesture,
    pub source: BindingSource,
}

const fn own(gesture: Gesture, chords: &'static [Chord]) -> DefaultBinding {
    DefaultBinding {
        gesture,
        source: BindingSource::Own(chords),
    }
}

const fn shared(gesture: Gesture, action: Action) -> DefaultBinding {
    DefaultBinding {
        gesture,
        source: BindingSource::SharedWith(action),
    }
}

const fn key(key: KeyCode) -> Chord {
    Chord::plain(Input::Key(key))
}

const fn with(modifiers: Modifiers, key: KeyCode) -> Chord {
    Chord {
        modifiers,
        input: Input::Key(key),
        double_tap: false,
    }
}

const fn double_tap(key: KeyCode) -> Chord {
    Chord {
        modifiers: Modifiers::NONE,
        input: Input::Key(key),
        double_tap: true,
    }
}

const fn button(button: MouseButton) -> Chord {
    Chord::plain(Input::Mouse(button))
}

/// Declares [`Action`] together with [`Action::ALL`], so no action can be
/// left out of the list the controls sheet shows, and with the stable name
/// each is saved under.
macro_rules! actions {
    ($($(#[$meta:meta])* $variant:ident),* $(,)?) => {
        /// Something the user can do with a key, a button or the wheel.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Action {
            $($(#[$meta])* $variant),*
        }

        impl Action {
            /// Every action, in the order the controls sheet lists them.
            pub const ALL: &'static [Action] = &[$(Action::$variant),*];

            /// The name its binding is saved under.
            pub fn key(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($variant)),*
                }
            }
        }
    };
}

actions! {
    MoveForward,
    MoveBack,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    MoveFaster,
    ChangeSpeed,
    Look,
    StepRight,
    StepLeft,
    StepUp,
    StepDown,
    /// Held with the up and down steps, steps into and out of the scene.
    StepThroughDepth,
    Select,
    AddToSelection,
    BoxSelect,
    MoveSelection,
    PushPullSelection,
    SelectAll,
    FitSelection,
    FitAll,
    ClearSelection,
    Undo,
    Redo,
    OpenFolder,
    CloseFolder,
    DeleteFolder,
    ContextMenu,
    PlayPause,
    SeekBack,
    SeekForward,
    PauseMenu,
    ShowControls,
    ToggleHud,
}

impl Action {
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|action| action.key() == key)
    }

    pub fn category(self) -> ActionCategory {
        match self {
            Self::MoveForward
            | Self::MoveBack
            | Self::MoveLeft
            | Self::MoveRight
            | Self::MoveUp
            | Self::MoveDown
            | Self::MoveFaster
            | Self::ChangeSpeed
            | Self::Look
            | Self::StepRight
            | Self::StepLeft
            | Self::StepUp
            | Self::StepDown
            | Self::StepThroughDepth => ActionCategory::Movement,
            Self::Select
            | Self::AddToSelection
            | Self::BoxSelect
            | Self::MoveSelection
            | Self::PushPullSelection
            | Self::SelectAll
            | Self::FitSelection
            | Self::FitAll
            | Self::ClearSelection => ActionCategory::Selection,
            Self::Undo | Self::Redo => ActionCategory::Editing,
            Self::OpenFolder | Self::CloseFolder | Self::DeleteFolder | Self::ContextMenu => {
                ActionCategory::Folders
            }
            Self::PlayPause | Self::SeekBack | Self::SeekForward => ActionCategory::Video,
            Self::PauseMenu | Self::ShowControls | Self::ToggleHud => ActionCategory::Interface,
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::MoveForward => "Move forward",
            Self::MoveBack => "Move back",
            Self::MoveLeft => "Strafe left",
            Self::MoveRight => "Strafe right",
            Self::MoveUp => "Move up, relative to the view",
            Self::MoveDown => "Move down, relative to the view",
            Self::MoveFaster => "Move faster",
            Self::ChangeSpeed => "Change base speed",
            Self::Look => "Look around",
            Self::StepRight => "Jump to the next image right",
            Self::StepLeft => "Jump to the next image left",
            Self::StepUp => "Jump to the next image up",
            Self::StepDown => "Jump to the next image down",
            Self::StepThroughDepth => "With up / down: jump forward / back instead",
            Self::Select => "Select an image or folder; click empty space to clear",
            Self::AddToSelection => "With a click or box: add to the selection",
            Self::BoxSelect => "Box-select, starting on empty space",
            Self::MoveSelection => "Move the selection, starting on it",
            Self::PushPullSelection => "While moving: push away / pull closer",
            Self::SelectAll => "Select every image and closed folder shown",
            Self::FitSelection => "Fit the selection to the screen",
            Self::FitAll => "Fit everything to the screen",
            Self::ClearSelection => "Clear the selection, before the pause menu",
            Self::Undo => "Undo the last arrangement change",
            Self::Redo => "Redo",
            Self::OpenFolder => "Open a closed folder",
            Self::CloseFolder => "Close an open folder, on a corner icon",
            Self::DeleteFolder => "Delete the selected folders",
            Self::ContextMenu => "Menu for what is under the pointer",
            Self::PlayPause => "Play or pause the hovered, else the selected videos",
            Self::SeekBack => "Seek back 5 s",
            Self::SeekForward => "Seek forward 5 s",
            Self::PauseMenu => "Pause menu; steps back through menus",
            Self::ShowControls => "Show these controls",
            Self::ToggleHud => "Hide or show the interface",
        }
    }

    /// Answers even while something is being typed: the key that ends the
    /// typing.
    fn reads_while_typing(self) -> bool {
        matches!(self, Self::PauseMenu)
    }

    pub fn context(self) -> ActionContext {
        match self {
            Self::PauseMenu | Self::ShowControls => ActionContext::Everywhere,
            _ => ActionContext::World,
        }
    }

    /// Only changes what another action does while it is held, so it may
    /// share an input with other actions of its kind (Shift can both speed
    /// up flying and step through depth), and it adds to what it shares an
    /// input with rather than taking the press. Asked only of actions that
    /// own their binding: one that shares another's follows that one.
    fn modifies_another(self) -> bool {
        matches!(self, Self::MoveFaster | Self::StepThroughDepth)
    }

    /// Must keep a binding: the way back out of every menu.
    pub fn required(self) -> bool {
        matches!(self, Self::PauseMenu)
    }

    /// Its own binding can be changed. The wheel's actions are not: nothing
    /// else scrolls.
    pub fn rebindable(self) -> bool {
        let binding = self.default_binding();
        matches!(binding.source, BindingSource::Own(_)) && binding.gesture != Gesture::Scroll
    }

    /// It is held, so it can toggle instead.
    pub fn can_toggle(self) -> bool {
        self.default_binding().gesture == Gesture::Hold
    }

    pub fn default_binding(self) -> DefaultBinding {
        use Gesture::{Click, Drag, Hold, Press, Scroll};
        const SHIFT: Chord = Chord::plain(Input::Modifier(Modifier::Shift));
        match self {
            Self::MoveForward => own(Hold, const { &[key(KeyCode::KeyW)] }),
            Self::MoveBack => own(Hold, const { &[key(KeyCode::KeyS)] }),
            Self::MoveLeft => own(Hold, const { &[key(KeyCode::KeyA)] }),
            Self::MoveRight => own(Hold, const { &[key(KeyCode::KeyD)] }),
            Self::MoveUp => own(Hold, const { &[key(KeyCode::Space)] }),
            Self::MoveDown => own(Hold, const { &[key(KeyCode::KeyC)] }),
            Self::MoveFaster => own(Hold, const { &[SHIFT] }),
            Self::ChangeSpeed => own(Scroll, const { &[Chord::plain(Input::Wheel)] }),
            Self::Look => own(Drag, const { &[button(MouseButton::Right)] }),
            Self::StepRight => own(Press, const { &[key(KeyCode::ArrowRight)] }),
            Self::StepLeft => own(Press, const { &[key(KeyCode::ArrowLeft)] }),
            Self::StepUp => own(Press, const { &[key(KeyCode::ArrowUp)] }),
            Self::StepDown => own(Press, const { &[key(KeyCode::ArrowDown)] }),
            Self::StepThroughDepth => own(Hold, const { &[key(KeyCode::ShiftRight)] }),
            Self::Select => own(Click, const { &[button(MouseButton::Left)] }),
            Self::AddToSelection => shared(Hold, Self::MoveFaster),
            Self::BoxSelect => shared(Drag, Self::Select),
            Self::MoveSelection => shared(Drag, Self::Select),
            Self::PushPullSelection => shared(Scroll, Self::ChangeSpeed),
            Self::SelectAll => own(Press, const { &[with(Modifiers::CTRL, KeyCode::KeyA)] }),
            Self::FitSelection => own(Press, const { &[key(KeyCode::KeyF)] }),
            Self::FitAll => own(Press, const { &[double_tap(KeyCode::KeyF)] }),
            Self::ClearSelection => shared(Press, Self::PauseMenu),
            Self::Undo => own(Press, const { &[with(Modifiers::CTRL, KeyCode::KeyZ)] }),
            Self::Redo => own(
                Press,
                const {
                    &[
                        with(Modifiers::CTRL_SHIFT, KeyCode::KeyZ),
                        with(Modifiers::CTRL, KeyCode::KeyY),
                    ]
                },
            ),
            Self::OpenFolder => shared(Click, Self::Select),
            Self::CloseFolder => shared(Click, Self::Select),
            Self::DeleteFolder => own(Press, const { &[key(KeyCode::Delete)] }),
            Self::ContextMenu => shared(Click, Self::Look),
            Self::PlayPause => own(Press, const { &[key(KeyCode::KeyK)] }),
            Self::SeekBack => own(Press, const { &[key(KeyCode::KeyJ)] }),
            Self::SeekForward => own(Press, const { &[key(KeyCode::KeyL)] }),
            Self::PauseMenu => own(Press, const { &[key(KeyCode::Escape)] }),
            Self::ShowControls => own(Press, const { &[key(KeyCode::F1)] }),
            Self::ToggleHud => own(Press, const { &[key(KeyCode::KeyH)] }),
        }
    }

    /// The action whose binding this one follows, if it shares one.
    pub fn shares_with(self) -> Option<Action> {
        match self.default_binding().source {
            BindingSource::Own(_) => None,
            BindingSource::SharedWith(action) => Some(action),
        }
    }

    /// Its own default slots, or `None` when it shares another's.
    fn default_slots(self) -> Option<Slots> {
        let BindingSource::Own(chords) = self.default_binding().source else {
            return None;
        };
        let mut slots = [None; SLOT_COUNT];
        for (slot, &chord) in slots.iter_mut().zip(chords) {
            *slot = Some(chord);
        }
        Some(slots)
    }
}

/// Why a change to a binding was not made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingRefusal {
    /// The action follows another's binding, or scrolls.
    NotRebindable,
    /// It would leave a required action with nothing bound.
    LastRequiredBinding,
}

impl BindingRefusal {
    pub fn message(self) -> &'static str {
        match self {
            Self::NotRebindable => "That control cannot be rebound.",
            Self::LastRequiredBinding => {
                "The pause menu keeps at least one binding: it is the way out of every menu."
            }
        }
    }
}

/// Two actions a chord takes at once, in [`Action::ALL`] order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingConflict {
    pub chord: Chord,
    pub actions: [Action; 2],
}

/// What each action is bound to now, and which held actions toggle. Starts
/// from every action's default binding.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct ControlBindings {
    slots: HashMap<Action, Slots>,
    toggled: HashSet<Action>,
}

impl Default for ControlBindings {
    fn default() -> Self {
        Self {
            slots: Action::ALL
                .iter()
                .filter_map(|&action| Some((action, action.default_slots()?)))
                .collect(),
            toggled: HashSet::new(),
        }
    }
}

impl ControlBindings {
    /// The defaults, changed by what was saved: `slots` for the actions
    /// rebound and `toggled` for the held actions made to toggle. What
    /// cannot apply (an action that cannot be rebound, a required one left
    /// unbound) keeps its default.
    pub fn with_saved(
        slots: impl IntoIterator<Item = (Action, Slots)>,
        toggled: impl IntoIterator<Item = Action>,
    ) -> Self {
        let mut bindings = Self::default();
        for (action, saved) in slots {
            let unbound = saved.iter().all(Option::is_none);
            if action.rebindable() && !(unbound && action.required()) {
                bindings.slots.insert(action, saved);
            }
        }
        bindings.toggled = toggled
            .into_iter()
            .filter(|action| action.can_toggle())
            .collect();
        bindings
    }

    /// The slots of `action`, or of the action it shares.
    pub fn slots(&self, action: Action) -> Slots {
        let owner = action.shares_with().unwrap_or(action);
        self.slots
            .get(&owner)
            .copied()
            .unwrap_or([None; SLOT_COUNT])
    }

    /// The chords that trigger `action`, following a shared binding.
    pub fn chords(&self, action: Action) -> impl Iterator<Item = Chord> {
        self.slots(action).into_iter().flatten()
    }

    /// Binds `chord` in `slot` of `action`, or empties the slot. A chord
    /// already in the action's other slot moves here.
    pub fn set_slot(
        &mut self,
        action: Action,
        slot: usize,
        chord: Option<Chord>,
    ) -> Result<(), BindingRefusal> {
        if !action.rebindable() || slot >= SLOT_COUNT {
            return Err(BindingRefusal::NotRebindable);
        }
        let mut slots = self.slots(action);
        slots[slot] = chord;
        if chord.is_some() {
            for (other, bound) in slots.iter_mut().enumerate() {
                if other != slot && *bound == chord {
                    *bound = None;
                }
            }
        }
        if action.required() && slots.iter().all(Option::is_none) {
            return Err(BindingRefusal::LastRequiredBinding);
        }
        self.slots.insert(action, slots);
        Ok(())
    }

    pub fn hold_mode(&self, action: Action) -> HoldMode {
        if self.toggled.contains(&action) {
            HoldMode::Toggle
        } else {
            HoldMode::Hold
        }
    }

    /// Switches a held action between holding and toggling.
    pub fn switch_hold_mode(&mut self, action: Action) {
        if action.can_toggle() && !self.toggled.remove(&action) {
            self.toggled.insert(action);
        }
    }

    /// Whether `action` is bound and behaves as it ships.
    pub fn is_default(&self, action: Action) -> bool {
        let slots_default = action
            .default_slots()
            .is_none_or(|slots| self.slots.get(&action).is_some_and(|bound| *bound == slots));
        slots_default && !self.toggled.contains(&action)
    }

    pub fn all_default(&self) -> bool {
        Action::ALL.iter().all(|&action| self.is_default(action))
    }

    /// Restores `action`'s default binding and hold mode.
    pub fn reset(&mut self, action: Action) {
        if let Some(slots) = action.default_slots() {
            self.slots.insert(action, slots);
        }
        self.toggled.remove(&action);
    }

    pub fn reset_all(&mut self) {
        *self = Self::default();
    }

    /// The actions whose own slots differ from their defaults, with them.
    pub fn changed_slots(&self) -> impl Iterator<Item = (Action, Slots)> + '_ {
        Action::ALL.iter().filter_map(|&action| {
            let slots = *self.slots.get(&action)?;
            (Some(slots) != action.default_slots()).then_some((action, slots))
        })
    }

    /// The held actions that toggle, in [`Action::ALL`] order.
    pub fn toggled(&self) -> impl Iterator<Item = Action> + '_ {
        Action::ALL
            .iter()
            .copied()
            .filter(|action| self.toggled.contains(action))
    }

    /// Every pair of actions one chord takes at once where they get in each
    /// other's way: read at the same time, and not both only changing what
    /// another action does. Actions sharing a binding share it on purpose,
    /// so only the actions that own one are compared.
    pub fn conflicts(&self) -> Vec<BindingConflict> {
        let owners: Vec<Action> = Action::ALL
            .iter()
            .copied()
            .filter(|action| action.shares_with().is_none())
            .collect();
        let mut conflicts = Vec::new();
        for (index, &first) in owners.iter().enumerate() {
            for &second in &owners[index + 1..] {
                if !first.context().overlaps(second.context())
                    || (first.modifies_another() && second.modifies_another())
                {
                    continue;
                }
                for chord in self.chords(first) {
                    if self.chords(second).any(|other| chord.overlaps(other)) {
                        conflicts.push(BindingConflict {
                            chord,
                            actions: [first, second],
                        });
                    }
                }
            }
        }
        conflicts
    }

    /// Whether a more specific chord bound elsewhere on `chord`'s input
    /// takes the press instead: one with more modifiers, all `held`, so
    /// Ctrl+Shift+Z redoes without also undoing as Ctrl+Z would; or, on a
    /// second tap (`doubled`), a double-tap with the same modifiers, so the
    /// second press of F fits everything without fitting the selection
    /// again. An action that only modifies another never takes a press.
    fn outranked(&self, action: Action, chord: Chord, held: Modifiers, doubled: bool) -> bool {
        self.slots.iter().any(|(&owner, slots)| {
            !owner.modifies_another()
                && owner.context().overlaps(action.context())
                && slots.iter().flatten().any(|other| {
                    let more_modifiers = other.double_tap == chord.double_tap
                        && other.modifiers != chord.modifiers
                        && chord.modifiers.is_within(other.modifiers)
                        && other.modifiers.is_within(held);
                    let double_tapped = doubled
                        && other.double_tap
                        && !chord.double_tap
                        && other.modifiers == chord.modifiers;
                    other.input.overlaps(chord.input) && (more_modifiers || double_tapped)
                })
        })
    }

    /// How the controls sheet writes `action`'s binding: its chords, then
    /// how they are used.
    pub fn label(&self, action: Action) -> String {
        let chords: Vec<String> = self.chords(action).map(Chord::name).collect();
        if chords.is_empty() {
            return "Unbound".to_owned();
        }
        let chords = chords.join(" / ");
        match action.default_binding().gesture {
            Gesture::Press | Gesture::Scroll => chords,
            Gesture::Hold => match self.hold_mode(action) {
                HoldMode::Hold => format!("{chords} (hold)"),
                HoldMode::Toggle => format!("{chords} (toggle)"),
            },
            Gesture::Click => format!("{chords} click"),
            Gesture::Drag => format!("{chords} drag"),
        }
    }
}

/// The inputs whose latest press was the second of a double-tap.
#[derive(Default)]
struct InputTaps {
    /// When each input was last pressed, if that press may still start a
    /// double-tap.
    last_press: HashMap<Input, f64>,
    doubled: HashSet<Input>,
}

impl InputTaps {
    fn record(&mut self, input: Input, now: f64) {
        let doubled = self
            .last_press
            .remove(&input)
            .is_some_and(|at| now - at <= DOUBLE_TAP_SECONDS);
        if doubled {
            self.doubled.insert(input);
        } else {
            self.doubled.remove(&input);
            self.last_press.insert(input, now);
        }
    }

    fn doubled(&self, input: Input) -> bool {
        match input {
            Input::Modifier(modifier) => modifier
                .keys()
                .into_iter()
                .any(|key| self.doubled.contains(&Input::Key(key))),
            input => self.doubled.contains(&input),
        }
    }
}

/// The held actions that toggle and are on, this frame and the last.
#[derive(Default)]
struct ActionToggles {
    on: HashSet<Action>,
    was_on: HashSet<Action>,
}

/// What input reading carries from frame to frame: taps, for double-taps,
/// and which toggling actions are on.
#[derive(Resource, Default)]
pub struct ControlInputState {
    taps: InputTaps,
    toggles: ActionToggles,
}

/// Records this frame's presses as taps and switches the toggling actions
/// pressed. Runs after the input update and before anything reads
/// [`ControlInput`].
pub fn update_control_input_state(
    real_time: Res<Time<Real>>,
    bindings: Res<ControlBindings>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    focus: Res<TypingFocus>,
    mut state: ResMut<ControlInputState>,
) {
    let now = real_time.elapsed_secs_f64();
    let ControlInputState { taps, toggles } = &mut *state;
    for &key in keys.get_just_pressed() {
        taps.record(Input::Key(key), now);
    }
    for &button in mouse.get_just_pressed() {
        taps.record(Input::Mouse(button), now);
    }
    toggles.was_on.clone_from(&toggles.on);
    toggles
        .on
        .retain(|&action| bindings.hold_mode(action) == HoldMode::Toggle);
    let reading = InputReading {
        bindings: &bindings,
        keys: &keys,
        mouse: &mouse,
        taps,
        focus: &focus,
    };
    for &action in Action::ALL {
        if bindings.hold_mode(action) == HoldMode::Toggle
            && reading.action(action, Phase::JustPressed)
            && !toggles.on.remove(&action)
        {
            toggles.on.insert(action);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Held,
    JustPressed,
    JustReleased,
}

/// This frame's inputs, read as bindings.
struct InputReading<'a> {
    bindings: &'a ControlBindings,
    keys: &'a ButtonInput<KeyCode>,
    mouse: &'a ButtonInput<MouseButton>,
    taps: &'a InputTaps,
    focus: &'a TypingFocus,
}

impl InputReading<'_> {
    /// Whether any chord of `action` is in `phase`. Nothing is while a
    /// binding is being captured; keys stand down while text is typed.
    fn action(&self, action: Action, phase: Phase) -> bool {
        if self.focus.capturing {
            return false;
        }
        let keys_stand_down = self.focus.typing && !action.reads_while_typing();
        let held = Modifiers::held(self.keys);
        self.bindings.chords(action).any(|chord| {
            !(keys_stand_down && chord.input.is_keyboard())
                && (!chord.double_tap || self.taps.doubled(chord.input))
                && self.input(chord.input, phase)
                // Modifiers start a press; letting go of one first still
                // ends it.
                && (phase == Phase::JustReleased
                    || (chord.modifiers.is_within(held)
                        && !self.bindings.outranked(
                            action,
                            chord,
                            held,
                            self.taps.doubled(chord.input),
                        )))
        })
    }

    fn input(&self, input: Input, phase: Phase) -> bool {
        match (input, phase) {
            (Input::Key(key), Phase::Held) => self.keys.pressed(key),
            (Input::Key(key), Phase::JustPressed) => self.keys.just_pressed(key),
            (Input::Key(key), Phase::JustReleased) => self.keys.just_released(key),
            (Input::Modifier(modifier), Phase::Held) => self.keys.any_pressed(modifier.keys()),
            (Input::Modifier(modifier), Phase::JustPressed) => {
                self.keys.any_just_pressed(modifier.keys())
            }
            // Released once neither key is held any more.
            (Input::Modifier(modifier), Phase::JustReleased) => {
                self.keys.any_just_released(modifier.keys())
                    && !self.keys.any_pressed(modifier.keys())
            }
            (Input::Mouse(button), Phase::Held) => self.mouse.pressed(button),
            (Input::Mouse(button), Phase::JustPressed) => self.mouse.just_pressed(button),
            (Input::Mouse(button), Phase::JustReleased) => self.mouse.just_released(button),
            // The wheel is read as scroll events, by the actions that scroll.
            (Input::Wheel, _) => false,
        }
    }
}

/// Reads actions from this frame's keyboard and mouse. Keys stand down while
/// something is being typed, so typing never also flies the camera or fires
/// a shortcut; mouse buttons do not, and neither does the key that ends the
/// typing. Everything stands down while a binding is being captured. A held
/// action that toggles reads as held while it is on.
#[derive(SystemParam)]
pub struct ControlInput<'w> {
    bindings: Res<'w, ControlBindings>,
    keys: Res<'w, ButtonInput<KeyCode>>,
    mouse: Res<'w, ButtonInput<MouseButton>>,
    focus: Res<'w, TypingFocus>,
    state: Res<'w, ControlInputState>,
}

impl ControlInput<'_> {
    pub fn pressed(&self, action: Action) -> bool {
        match self.bindings.hold_mode(action) {
            HoldMode::Toggle => !self.focus.capturing && self.state.toggles.on.contains(&action),
            HoldMode::Hold => self.reading().action(action, Phase::Held),
        }
    }

    pub fn just_pressed(&self, action: Action) -> bool {
        match self.bindings.hold_mode(action) {
            HoldMode::Toggle => {
                self.state.toggles.on.contains(&action)
                    && !self.state.toggles.was_on.contains(&action)
            }
            HoldMode::Hold => self.reading().action(action, Phase::JustPressed),
        }
    }

    pub fn just_released(&self, action: Action) -> bool {
        match self.bindings.hold_mode(action) {
            HoldMode::Toggle => {
                self.state.toggles.was_on.contains(&action)
                    && !self.state.toggles.on.contains(&action)
            }
            HoldMode::Hold => self.reading().action(action, Phase::JustReleased),
        }
    }

    /// Something is being typed: the keyboard belongs to it.
    pub fn typing(&self) -> bool {
        self.focus.typing
    }

    pub fn bindings(&self) -> &ControlBindings {
        &self.bindings
    }

    fn reading(&self) -> InputReading<'_> {
        InputReading {
            bindings: &self.bindings,
            keys: &self.keys,
            mouse: &self.mouse,
            taps: &self.state.taps,
            focus: &self.focus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_action_has_a_binding_that_fits_its_slots() {
        let bindings = ControlBindings::default();
        for &action in Action::ALL {
            assert!(
                bindings.chords(action).next().is_some(),
                "{action:?} has nothing bound"
            );
            assert!(!action.description().is_empty());
            if let BindingSource::Own(chords) = action.default_binding().source {
                assert!(chords.len() <= SLOT_COUNT, "{action:?} has too many");
            }
            assert_eq!(Action::from_key(action.key()), Some(action));
        }
    }

    #[test]
    fn the_defaults_have_no_conflicts() {
        assert_eq!(ControlBindings::default().conflicts(), Vec::new());
    }

    #[test]
    fn a_shared_binding_follows_the_action_it_shares() {
        let mut bindings = ControlBindings::default();
        assert_eq!(bindings.label(Action::BoxSelect), "Left mouse drag");
        bindings
            .set_slot(
                Action::Select,
                0,
                Some(Chord::plain(Input::Mouse(MouseButton::Middle))),
            )
            .unwrap();
        assert_eq!(bindings.label(Action::BoxSelect), "Middle mouse drag");
        assert_eq!(bindings.label(Action::Select), "Middle mouse click");
        assert_eq!(
            bindings.set_slot(Action::BoxSelect, 0, None),
            Err(BindingRefusal::NotRebindable)
        );
    }

    #[test]
    fn labels_name_chords() {
        let bindings = ControlBindings::default();
        assert_eq!(bindings.label(Action::MoveFaster), "Shift (hold)");
        assert_eq!(bindings.label(Action::MoveDown), "C (hold)");
        assert_eq!(
            bindings.label(Action::StepThroughDepth),
            "Right Shift (hold)"
        );
        assert_eq!(bindings.label(Action::FitSelection), "F");
        assert_eq!(bindings.label(Action::FitAll), "Double-tap F");
        assert_eq!(bindings.label(Action::Redo), "Ctrl+Shift+Z / Ctrl+Y");
        assert_eq!(bindings.label(Action::PauseMenu), "Esc");
        assert_eq!(bindings.label(Action::ChangeSpeed), "Mouse wheel");
    }

    #[test]
    fn chords_read_back_from_their_saved_form() {
        let chords = [
            with(Modifiers::CTRL_SHIFT, KeyCode::KeyZ),
            double_tap(KeyCode::KeyF),
            Chord::plain(Input::Modifier(Modifier::Shift)),
            Chord {
                modifiers: Modifiers::CTRL,
                input: Input::Mouse(MouseButton::Back),
                double_tap: true,
            },
            key(KeyCode::NumpadAdd),
            key(KeyCode::Equal),
        ];
        for chord in chords {
            let saved = chord.to_saved().expect("savable");
            assert_eq!(Chord::from_saved(&saved), Some(chord), "{saved}");
        }
        assert_eq!(
            with(Modifiers::CTRL_SHIFT, KeyCode::KeyZ)
                .to_saved()
                .as_deref(),
            Some("Ctrl+Shift+KeyZ")
        );
        assert_eq!(Chord::from_saved("Ctrl+Nonsense"), None);
        assert_eq!(Chord::from_saved("Hyper+KeyA"), None);
    }

    #[test]
    fn rebinding_moves_a_duplicate_and_keeps_the_pause_menu_bound() {
        let mut bindings = ControlBindings::default();
        let y = with(Modifiers::CTRL, KeyCode::KeyY);
        bindings.set_slot(Action::Redo, 0, Some(y)).unwrap();
        assert_eq!(bindings.slots(Action::Redo), [Some(y), None]);
        assert!(!bindings.is_default(Action::Redo));
        bindings.reset(Action::Redo);
        assert!(bindings.is_default(Action::Redo));

        assert_eq!(
            bindings.set_slot(Action::PauseMenu, 0, None),
            Err(BindingRefusal::LastRequiredBinding)
        );
        assert_eq!(bindings.label(Action::PauseMenu), "Esc");
        bindings
            .set_slot(Action::PauseMenu, 1, Some(key(KeyCode::KeyP)))
            .unwrap();
        bindings.set_slot(Action::PauseMenu, 0, None).unwrap();
        assert_eq!(bindings.label(Action::PauseMenu), "P");
    }

    #[test]
    fn conflicts_are_triggers_read_together_and_modifiers_may_stack() {
        let mut bindings = ControlBindings::default();
        bindings
            .set_slot(Action::PlayPause, 0, Some(key(KeyCode::KeyF)))
            .unwrap();
        assert_eq!(
            bindings.conflicts(),
            vec![BindingConflict {
                chord: key(KeyCode::KeyF),
                actions: [Action::FitSelection, Action::PlayPause],
            }]
        );
        bindings.reset_all();
        // Both only change what another action does.
        bindings
            .set_slot(Action::StepThroughDepth, 0, Some(key(KeyCode::ShiftLeft)))
            .unwrap();
        assert!(bindings.conflicts().is_empty());
        // Held with Ctrl, W is a different chord from W.
        bindings
            .set_slot(Action::Undo, 0, Some(with(Modifiers::CTRL, KeyCode::KeyW)))
            .unwrap();
        assert!(bindings.conflicts().is_empty());
        bindings
            .set_slot(Action::ShowControls, 0, Some(key(KeyCode::KeyW)))
            .unwrap();
        assert_eq!(bindings.conflicts().len(), 1);
    }

    #[test]
    fn the_chord_with_the_most_held_modifiers_takes_the_press() {
        let bindings = ControlBindings::default();
        let ctrl_z = with(Modifiers::CTRL, KeyCode::KeyZ);
        assert!(bindings.outranked(Action::Undo, ctrl_z, Modifiers::CTRL_SHIFT, false));
        assert!(!bindings.outranked(Action::Undo, ctrl_z, Modifiers::CTRL, false));
        let ctrl_shift_z = with(Modifiers::CTRL_SHIFT, KeyCode::KeyZ);
        assert!(!bindings.outranked(Action::Redo, ctrl_shift_z, Modifiers::CTRL_SHIFT, false));
    }

    #[test]
    fn a_second_tap_goes_to_the_double_tap_alone() {
        let mut bindings = ControlBindings::default();
        let f = key(KeyCode::KeyF);
        assert!(!bindings.outranked(Action::FitSelection, f, Modifiers::NONE, false));
        assert!(bindings.outranked(Action::FitSelection, f, Modifiers::NONE, true));
        // Held with Ctrl, F is a different chord, which no double-tap takes.
        let ctrl_f = with(Modifiers::CTRL, KeyCode::KeyF);
        bindings
            .set_slot(Action::PlayPause, 0, Some(ctrl_f))
            .unwrap();
        assert!(!bindings.outranked(Action::PlayPause, ctrl_f, Modifiers::CTRL, true));
    }

    #[test]
    fn an_action_that_modifies_another_adds_to_the_press() {
        let mut bindings = ControlBindings::default();
        bindings
            .set_slot(Action::MoveFaster, 0, Some(double_tap(KeyCode::KeyW)))
            .unwrap();
        let w = key(KeyCode::KeyW);
        assert!(!bindings.outranked(Action::MoveForward, w, Modifiers::NONE, true));
    }

    #[test]
    fn a_second_press_soon_after_the_first_is_a_double_tap() {
        let mut taps = InputTaps::default();
        let f = Input::Key(KeyCode::KeyF);
        taps.record(f, 1.0);
        assert!(!taps.doubled(f));
        taps.record(f, 1.2);
        assert!(taps.doubled(f));
        // A third press starts over.
        taps.record(f, 1.3);
        assert!(!taps.doubled(f));
        taps.record(f, 2.0);
        assert!(!taps.doubled(f));

        let right_shift = Input::Key(KeyCode::ShiftRight);
        taps.record(right_shift, 3.0);
        taps.record(right_shift, 3.1);
        assert!(taps.doubled(Input::Modifier(Modifier::Shift)));
    }

    #[test]
    fn saved_bindings_apply_where_they_can() {
        let bindings = ControlBindings::with_saved(
            [
                (Action::MoveDown, [Some(key(KeyCode::KeyQ)), None]),
                (Action::BoxSelect, [Some(key(KeyCode::KeyB)), None]),
                (Action::PauseMenu, [None, None]),
            ],
            [Action::MoveFaster, Action::FitSelection],
        );
        assert_eq!(bindings.label(Action::MoveDown), "Q (hold)");
        assert_eq!(bindings.label(Action::BoxSelect), "Left mouse drag");
        assert_eq!(bindings.label(Action::PauseMenu), "Esc");
        assert_eq!(bindings.hold_mode(Action::MoveFaster), HoldMode::Toggle);
        assert_eq!(bindings.label(Action::MoveFaster), "Shift (toggle)");
        assert_eq!(bindings.hold_mode(Action::FitSelection), HoldMode::Hold);
        assert_eq!(
            bindings.changed_slots().collect::<Vec<_>>(),
            vec![(Action::MoveDown, [Some(key(KeyCode::KeyQ)), None])]
        );
        assert_eq!(
            bindings.toggled().collect::<Vec<_>>(),
            vec![Action::MoveFaster]
        );
    }

    /// Keys are read through [`ControlInput`] alone, so every key the viewer
    /// answers to is in the table above and on the controls sheet. Mouse
    /// buttons here are the widgets' own (slider drags, a press outside a
    /// text box); the viewer's test covers the buttons its world answers to.
    #[test]
    fn no_key_is_read_outside_the_bindings() {
        fn check(directory: &std::path::Path) {
            for entry in std::fs::read_dir(directory).expect("source directory") {
                let path = entry.expect("source entry").path();
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("");
                if name == "input_bindings.rs" || name == "input_bindings" {
                    continue;
                }
                if path.is_dir() {
                    check(&path);
                    continue;
                }
                if path.extension().is_none_or(|ext| ext != "rs") {
                    continue;
                }
                let source = std::fs::read_to_string(&path).expect("source file");
                assert!(
                    !source.contains("KeyCode::"),
                    "{name} reads a key directly; bind it to an Action instead"
                );
            }
        }
        check(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"));
    }
}
