//! The user's control bindings, kept between sessions in `controls.json` in
//! the per-user state directory.
//!
//! Only how they differ from the defaults is kept, so a later viewer's new
//! defaults still reach every control the user has not changed. A change is
//! written as soon as it is made: they are rare, and small.

use std::collections::BTreeMap;
use std::path::PathBuf;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use spatial_viewer_ui::{Action, Chord, ControlBindings, Slots, SLOT_COUNT};

use crate::catalog_session::user_state_directory;
use crate::media_settings::{read_state_file, SettingsWriter};

const CONTROLS_FILE_NAME: &str = "controls.json";

/// What the file holds. Actions are named by [`Action::key`] and chords by
/// [`Chord::to_saved`], so an action or chord a later viewer no longer
/// knows is skipped.
#[derive(Serialize, Deserialize, Default, Debug, PartialEq)]
#[serde(default)]
struct SavedControls {
    /// The actions bound differently from their defaults: each slot's
    /// chord, or null where the user emptied it.
    bindings: BTreeMap<String, Vec<Option<String>>>,
    /// The held actions the user made toggle.
    toggled: Vec<String>,
}

impl SavedControls {
    fn of(bindings: &ControlBindings) -> Self {
        Self {
            bindings: bindings
                .changed_slots()
                .map(|(action, slots)| {
                    let saved = slots
                        .iter()
                        .map(|chord| chord.and_then(Chord::to_saved))
                        .collect();
                    (action.key().to_owned(), saved)
                })
                .collect(),
            toggled: bindings
                .toggled()
                .map(|action| action.key().to_owned())
                .collect(),
        }
    }

    /// The bindings saved, over the defaults. An action whose chords do
    /// not all read back keeps its default, rather than losing a slot.
    fn bindings(&self) -> ControlBindings {
        let slots = self.bindings.iter().filter_map(|(key, saved)| {
            let action = Action::from_key(key)?;
            let mut slots: Slots = [None; SLOT_COUNT];
            for (slot, chord) in slots.iter_mut().zip(saved) {
                *slot = match chord {
                    Some(chord) => Some(Chord::from_saved(chord)?),
                    None => None,
                };
            }
            Some((action, slots))
        });
        let toggled = self.toggled.iter().filter_map(|key| Action::from_key(key));
        ControlBindings::with_saved(slots, toggled)
    }
}

#[derive(Resource)]
pub(crate) struct ControlsStore {
    /// Where the bindings are saved; `None` keeps them in memory only.
    file: Option<PathBuf>,
    writer: SettingsWriter,
}

impl ControlsStore {
    /// The saved bindings, or the defaults when there are none. A file that
    /// cannot be read is moved aside rather than overwritten by the next
    /// save.
    pub(crate) fn load() -> (Self, ControlBindings) {
        let file = user_state_directory().map(|directory| directory.join(CONTROLS_FILE_NAME));
        let (saved, file) = match file {
            Some(file) => match read_state_file::<SavedControls>(&file, "control bindings") {
                Ok(saved) => (saved.unwrap_or_default(), Some(file)),
                Err(()) => (SavedControls::default(), None),
            },
            None => (SavedControls::default(), None),
        };
        (Self::with_file(file), saved.bindings())
    }

    /// The defaults, kept in memory only, for runs (benchmarks, the
    /// performance harness) that must neither follow nor change the user's.
    pub(crate) fn in_memory() -> (Self, ControlBindings) {
        (Self::with_file(None), ControlBindings::default())
    }

    fn with_file(file: Option<PathBuf>) -> Self {
        Self {
            file,
            writer: SettingsWriter::new("control bindings"),
        }
    }
}

/// Saves the bindings whenever the controls sheet changes them.
pub(crate) fn keep_control_bindings(
    bindings: Res<ControlBindings>,
    mut store: ResMut<ControlsStore>,
) {
    let failed = store.writer.take_failure();
    if !failed && (!bindings.is_changed() || bindings.is_added()) {
        return;
    }
    let Some(file) = store.file.clone() else {
        return;
    };
    match serde_json::to_string_pretty(&SavedControls::of(&bindings)) {
        // Waited for: a change is one click, and the next may be the last
        // before the app closes.
        Ok(contents) => store.writer.write(file, contents, true),
        Err(error) => eprintln!("Failed to encode control bindings: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_changes_are_saved_and_they_read_back() {
        assert_eq!(
            SavedControls::of(&ControlBindings::default()),
            SavedControls::default()
        );
        let mut bindings = ControlBindings::default();
        let fit = bindings.slots(Action::FitSelection)[0];
        bindings.set_slot(Action::MoveDown, 1, fit).unwrap();
        bindings.set_slot(Action::Redo, 1, None).unwrap();
        bindings.switch_hold_mode(Action::MoveFaster);

        let saved = SavedControls::of(&bindings);
        assert_eq!(saved.bindings.len(), 2);
        assert_eq!(
            saved.bindings["Redo"],
            vec![Some("Ctrl+Shift+KeyZ".to_owned()), None]
        );
        assert_eq!(saved.toggled, vec!["MoveFaster".to_owned()]);
        let json = serde_json::to_string(&saved).unwrap();
        let read: SavedControls = serde_json::from_str(&json).unwrap();
        assert_eq!(read.bindings(), bindings);
    }

    #[test]
    fn what_a_later_viewer_does_not_know_is_skipped() {
        let saved: SavedControls = serde_json::from_str(
            r#"{
                "bindings": {
                    "Teleport": ["KeyT", null],
                    "MoveDown": ["KeyQ", "Warp+KeyX"],
                    "MoveUp": ["KeyE"]
                },
                "toggled": ["Sprint", "MoveFaster"]
            }"#,
        )
        .unwrap();
        let bindings = saved.bindings();
        assert_eq!(bindings.label(Action::MoveDown), "C (hold)");
        assert_eq!(bindings.label(Action::MoveUp), "E (hold)");
        assert_eq!(
            bindings.hold_mode(Action::MoveFaster),
            spatial_viewer_ui::HoldMode::Toggle
        );
    }
}
