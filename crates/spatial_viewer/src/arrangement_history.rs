//! Undo and redo for what the user arranges by hand.
//!
//! Every change to the [`ManualArrangement`] that settles while no drag is
//! under way is one step: a whole drag, a new or deleted folder, a rename.
//! Opening and closing folders arranges nothing, so it is not undone.
//! Starting another view starts a fresh history.

use bevy::prelude::*;
use spatial_viewer_ui::{Action, ControlInput, EditControls, EditRequest, PauseMenuState};

use crate::catalog_load::CatalogLoadTask;
use crate::folders::{ArrangementSnapshot, FolderViewState, ManualArrangement};
use crate::manual_spacing::SelectionState;

/// Most steps kept to undo.
const MAX_UNDO_STEPS: usize = 100;

#[derive(Resource, Default)]
pub(crate) struct ArrangementHistory {
    /// The arrangement as the last step left it.
    current: Option<ArrangementSnapshot>,
    /// The view `current` belongs to.
    view: u64,
    undo: Vec<ArrangementSnapshot>,
    redo: Vec<ArrangementSnapshot>,
}

impl ArrangementHistory {
    pub(crate) fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub(crate) fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Takes a settled change to the arrangement as a step.
    pub(crate) fn record(&mut self, arrangement: &ManualArrangement) {
        if self.view != arrangement.view() || self.current.is_none() {
            self.view = arrangement.view();
            self.current = Some(arrangement.snapshot());
            self.undo.clear();
            self.redo.clear();
            return;
        }
        let Some(current) = self.current.as_ref() else {
            return;
        };
        if current.revision == arrangement.revision() {
            return;
        }
        let previous = self
            .current
            .replace(arrangement.snapshot())
            .expect("checked above");
        // A change that only renumbers the same arrangement is no step.
        if previous.contents == self.current.as_ref().expect("just set").contents {
            return;
        }
        self.undo.push(previous);
        if self.undo.len() > MAX_UNDO_STEPS {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    /// Puts the arrangement back as it was before the last step. Returns
    /// whether there was one.
    pub(crate) fn undo(&mut self, arrangement: &mut ManualArrangement) -> bool {
        self.step(arrangement, true)
    }

    /// Makes the last undone step again. Returns whether there was one.
    pub(crate) fn redo(&mut self, arrangement: &mut ManualArrangement) -> bool {
        self.step(arrangement, false)
    }

    fn step(&mut self, arrangement: &mut ManualArrangement, back: bool) -> bool {
        // An unrecorded change is the newest step, so it goes first.
        self.record(arrangement);
        let (from, to) = if back {
            (&mut self.undo, &mut self.redo)
        } else {
            (&mut self.redo, &mut self.undo)
        };
        let Some(target) = from.pop() else {
            return false;
        };
        if let Some(current) = self.current.take() {
            to.push(current);
        }
        arrangement.restore(&target);
        self.current = Some(arrangement.snapshot());
        true
    }
}

/// Applies the edit pill's buttons and the undo and redo keys, and
/// publishes whether there is anything to undo or redo. The keys wait while
/// a menu is open or a drag is changing the arrangement.
pub(crate) fn apply_edit_requests(
    input: ControlInput,
    pause_menu: Res<PauseMenuState>,
    selection: Res<SelectionState>,
    task: Res<CatalogLoadTask>,
    mut history: ResMut<ArrangementHistory>,
    mut controls: ResMut<EditControls>,
    mut folder_view: ResMut<FolderViewState>,
) {
    let key_request = || {
        if pause_menu.paused || selection.pressing() {
            None
        } else if input.just_pressed(Action::Redo) {
            Some(EditRequest::Redo)
        } else if input.just_pressed(Action::Undo) {
            Some(EditRequest::Undo)
        } else {
            None
        }
    };
    if let Some(request) = controls.take_request().or_else(key_request) {
        step_arrangement(
            &mut history,
            &mut task.arrangement(),
            &mut folder_view,
            request == EditRequest::Undo,
        );
    }
    let (can_undo, can_redo) = (history.can_undo(), history.can_redo());
    if controls.can_undo != can_undo || controls.can_redo != can_redo {
        controls.can_undo = can_undo;
        controls.can_redo = can_redo;
    }
}

/// Undoes or redoes a step, laying the scene out again for it.
pub(crate) fn step_arrangement(
    history: &mut ArrangementHistory,
    arrangement: &mut ManualArrangement,
    folder_view: &mut FolderViewState,
    back: bool,
) {
    let stepped = if back {
        history.undo(arrangement)
    } else {
        history.redo(arrangement)
    };
    if stepped {
        folder_view.relayout(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::folders::ManualPlacement;

    #[test]
    fn undo_and_redo_walk_the_steps_and_a_new_step_drops_the_redos() {
        let mut arrangement = ManualArrangement::default();
        let mut history = ArrangementHistory::default();
        history.record(&arrangement);
        arrangement.place(1, ManualPlacement::Loose(Vec3::X));
        history.record(&arrangement);
        arrangement.place(1, ManualPlacement::Loose(Vec3::Y));
        // Not yet recorded: undo takes it as the newest step.
        assert!(history.undo(&mut arrangement));
        assert_eq!(
            arrangement.placement(1),
            Some(&ManualPlacement::Loose(Vec3::X))
        );
        assert!(history.undo(&mut arrangement));
        assert_eq!(arrangement.placement(1), None);
        assert!(!history.undo(&mut arrangement));

        assert!(history.redo(&mut arrangement));
        assert_eq!(
            arrangement.placement(1),
            Some(&ManualPlacement::Loose(Vec3::X))
        );
        arrangement.place(2, ManualPlacement::Loose(Vec3::Z));
        history.record(&arrangement);
        assert!(!history.can_redo());
        assert!(history.can_undo());
    }

    #[test]
    fn a_new_view_starts_a_fresh_history() {
        let mut arrangement = ManualArrangement::default();
        let mut history = ArrangementHistory::default();
        history.record(&arrangement);
        arrangement.place(1, ManualPlacement::Loose(Vec3::X));
        history.record(&arrangement);
        assert!(history.can_undo());
        arrangement.start_view(Default::default(), 0);
        history.record(&arrangement);
        assert!(!history.can_undo());
        assert!(!history.undo(&mut arrangement));
    }
}
