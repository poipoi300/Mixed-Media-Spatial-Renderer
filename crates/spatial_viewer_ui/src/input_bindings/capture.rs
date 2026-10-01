//! Rebinding: the controls sheet picks a slot, and the next input pressed
//! becomes its binding.
//!
//! A key or mouse button binds with the modifiers held with it; pressed
//! twice in quick succession, it binds as a double-tap, so a press waits
//! [`DOUBLE_TAP_SECONDS`] for a second one before it binds. A modifier
//! pressed and let go with nothing pressed after it binds by itself, with no
//! other modifier. Esc cancels and Backspace
//! empties the slot, so neither can be captured.

use bevy::prelude::*;

use super::keys::{BINDABLE_BUTTONS, BINDABLE_KEYS, CANCEL_KEY, CLEAR_KEY};
use super::{Action, Chord, ControlBindings, Input, Modifier, Modifiers, DOUBLE_TAP_SECONDS};

/// One binding slot of one action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingSlot {
    pub action: Action,
    pub slot: usize,
}

/// The slot being captured, and what the last capture could not do.
#[derive(Resource, Default)]
pub struct BindingEditor {
    capturing: Option<BindingSlot>,
    /// The frame that began the capture has passed, so the click that began
    /// it is not taken for the binding.
    listening: bool,
    /// A press waiting to learn whether a second tap follows, and when it
    /// went down.
    pending: Option<(Chord, f64)>,
    /// The capture ended this frame, so the key that ended it reaches
    /// nothing else.
    ended_this_frame: bool,
    notice: Option<&'static str>,
}

impl BindingEditor {
    /// Starts capturing a binding for `slot`, replacing any other capture.
    pub fn begin(&mut self, slot: BindingSlot) {
        self.capturing = Some(slot);
        self.listening = false;
        self.pending = None;
        self.notice = None;
    }

    pub fn capturing(&self) -> Option<BindingSlot> {
        self.capturing
    }

    /// The press waiting for a possible second tap.
    pub fn pending(&self) -> Option<Chord> {
        self.pending.map(|(chord, _)| chord)
    }

    /// Why the last change to a binding was not made.
    pub fn notice(&self) -> Option<&'static str> {
        self.notice
    }

    pub fn set_notice(&mut self, notice: Option<&'static str>) {
        self.notice = notice;
    }

    /// Input belongs to the capture: it is under way, or ended this frame.
    pub(crate) fn holds_input(&self) -> bool {
        self.capturing.is_some() || self.ended_this_frame
    }

    fn end(&mut self) {
        self.capturing = None;
        self.pending = None;
        self.ended_this_frame = true;
    }

    /// Binds `chord` (or empties the slot) and ends the capture.
    fn finish(&mut self, bindings: &mut ControlBindings, chord: Option<Chord>) {
        if let Some(target) = self.capturing {
            self.notice = bindings
                .set_slot(target.action, target.slot, chord)
                .err()
                .map(|refusal| refusal.message());
        }
        self.end();
    }
}

/// Takes the next input pressed as the binding of the slot being captured.
/// Runs after the controls sheet's buttons, so the click that begins a
/// capture is seen by it first.
pub fn capture_binding(
    real_time: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut editor: ResMut<BindingEditor>,
    mut bindings: ResMut<ControlBindings>,
) {
    // Written only when it changes, so the sheet redraws only then.
    if editor.ended_this_frame {
        editor.ended_this_frame = false;
    }
    if editor.capturing.is_none() {
        return;
    }
    if !editor.listening {
        editor.listening = true;
        return;
    }
    if keys.just_pressed(CANCEL_KEY) {
        editor.end();
        return;
    }
    if keys.just_pressed(CLEAR_KEY) {
        editor.finish(&mut bindings, None);
        return;
    }
    let now = real_time.elapsed_secs_f64();
    let held = Modifiers::held(&keys);
    let pressed = newly_pressed(&keys, &mouse, held);
    if let Some((first, at)) = editor.pending {
        match pressed {
            Some(second) if second == first => editor.finish(
                &mut bindings,
                Some(Chord {
                    double_tap: true,
                    ..first
                }),
            ),
            // Anything else pressed, or no second tap in time, confirms the
            // single press.
            Some(_) => editor.finish(&mut bindings, Some(first)),
            None if now - at > DOUBLE_TAP_SECONDS => editor.finish(&mut bindings, Some(first)),
            None => {}
        }
        return;
    }
    if let Some(chord) = pressed {
        editor.pending = Some((chord, now));
        return;
    }
    let released_modifier = BINDABLE_KEYS
        .iter()
        .map(|&(key, _)| key)
        .find(|&key| Modifier::of_key(key).is_some() && keys.just_released(key));
    if let Some(key) = released_modifier {
        editor.finish(&mut bindings, Some(Chord::plain(Input::Key(key))));
    }
}

/// The first key or button (in the order they are listed) to go down this
/// frame that is not a modifier, as a chord with the modifiers held.
fn newly_pressed(
    keys: &ButtonInput<KeyCode>,
    mouse: &ButtonInput<MouseButton>,
    held: Modifiers,
) -> Option<Chord> {
    let key = BINDABLE_KEYS
        .iter()
        .map(|&(key, _)| key)
        .find(|&key| {
            keys.just_pressed(key)
                && Modifier::of_key(key).is_none()
                && key != CANCEL_KEY
                && key != CLEAR_KEY
        })
        .map(Input::Key);
    let button = || {
        BINDABLE_BUTTONS
            .iter()
            .map(|&(button, _, _)| button)
            .find(|&button| mouse.just_pressed(button))
            .map(Input::Mouse)
    };
    key.or_else(button).map(|input| Chord {
        modifiers: held,
        ..Chord::plain(input)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn capture_app() -> App {
        let mut app = App::new();
        app.init_resource::<Time<Real>>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<BindingEditor>()
            .init_resource::<ControlBindings>()
            .add_systems(Update, capture_binding);
        // A real clock's first update only starts it.
        app.world_mut()
            .resource_mut::<Time<Real>>()
            .update_with_duration(Duration::ZERO);
        app
    }

    fn begin(app: &mut App, action: Action, slot: usize) {
        app.world_mut()
            .resource_mut::<BindingEditor>()
            .begin(BindingSlot { action, slot });
        // The frame that began it.
        app.update();
    }

    fn press(app: &mut App, key: KeyCode) {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(key);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
    }

    fn release(app: &mut App, key: KeyCode) {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .release(key);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
    }

    fn wait(app: &mut App, seconds: f64) {
        app.world_mut()
            .resource_mut::<Time<Real>>()
            .update_with_duration(Duration::from_secs_f64(seconds));
        app.update();
    }

    fn label(app: &App, action: Action) -> String {
        app.world().resource::<ControlBindings>().label(action)
    }

    #[test]
    fn a_key_with_modifiers_binds_once_no_second_tap_comes() {
        let mut app = capture_app();
        begin(&mut app, Action::FitSelection, 0);
        press(&mut app, KeyCode::ControlLeft);
        press(&mut app, KeyCode::KeyG);
        assert!(app.world().resource::<BindingEditor>().pending().is_some());
        wait(&mut app, DOUBLE_TAP_SECONDS + 0.1);
        assert_eq!(label(&app, Action::FitSelection), "Ctrl+G");
        assert!(app
            .world()
            .resource::<BindingEditor>()
            .capturing()
            .is_none());
    }

    #[test]
    fn two_quick_presses_bind_a_double_tap() {
        let mut app = capture_app();
        begin(&mut app, Action::PlayPause, 1);
        press(&mut app, KeyCode::KeyP);
        release(&mut app, KeyCode::KeyP);
        press(&mut app, KeyCode::KeyP);
        assert_eq!(label(&app, Action::PlayPause), "K / Double-tap P");
    }

    #[test]
    fn a_modifier_let_go_alone_binds_by_itself() {
        let mut app = capture_app();
        begin(&mut app, Action::MoveDown, 0);
        press(&mut app, KeyCode::ControlRight);
        release(&mut app, KeyCode::ControlRight);
        assert_eq!(label(&app, Action::MoveDown), "Right Ctrl (hold)");
    }

    #[test]
    fn escape_cancels_and_backspace_empties_the_slot() {
        let mut app = capture_app();
        begin(&mut app, Action::Redo, 1);
        press(&mut app, KeyCode::Escape);
        assert_eq!(label(&app, Action::Redo), "Ctrl+Shift+Z / Ctrl+Y");
        assert!(app.world().resource::<BindingEditor>().holds_input());
        app.update();
        assert!(!app.world().resource::<BindingEditor>().holds_input());

        begin(&mut app, Action::Redo, 1);
        press(&mut app, KeyCode::Backspace);
        release(&mut app, KeyCode::Backspace);
        assert_eq!(label(&app, Action::Redo), "Ctrl+Shift+Z");

        begin(&mut app, Action::PauseMenu, 0);
        press(&mut app, KeyCode::Backspace);
        assert_eq!(label(&app, Action::PauseMenu), "Esc");
        assert!(app.world().resource::<BindingEditor>().notice().is_some());
    }
}
