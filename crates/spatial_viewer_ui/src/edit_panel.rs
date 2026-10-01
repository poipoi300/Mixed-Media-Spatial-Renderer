//! The edit pill: undo and redo for how things are arranged by hand, moves
//! and folders alike.

use bevy::prelude::*;

use crate::{
    pill_node, set_button_color, set_disabled_button_color, spawn_button_row,
    ButtonInteractionQuery, UiButtonColorQuery, ViewerUiButton, ViewerUiPanel,
};

const WIDTH: f32 = 190.0;

/// A button press waiting for the viewer, which owns the history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditRequest {
    Undo,
    Redo,
}

/// Whether there is anything to undo or redo, published by the viewer, and
/// the press the pill made.
#[derive(Resource, Default)]
pub struct EditControls {
    pub can_undo: bool,
    pub can_redo: bool,
    request: Option<EditRequest>,
}

impl EditControls {
    pub fn take_request(&mut self) -> Option<EditRequest> {
        self.request.take()
    }
}

pub(crate) fn spawn_edit_pill(parent: &mut ChildBuilder) {
    parent
        .spawn(pill_node(WIDTH, ViewerUiPanel::EditPill))
        .with_children(|pill| {
            spawn_button_row(
                pill,
                &[
                    (ViewerUiButton::UndoArrangement, "Undo"),
                    (ViewerUiButton::RedoArrangement, "Redo"),
                ],
            );
        });
}

pub fn handle_edit_buttons(
    mut controls: ResMut<EditControls>,
    interaction_query: ButtonInteractionQuery,
) {
    for (interaction, button) in &interaction_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match *button {
            ViewerUiButton::UndoArrangement if controls.can_undo => {
                controls.request = Some(EditRequest::Undo);
            }
            ViewerUiButton::RedoArrangement if controls.can_redo => {
                controls.request = Some(EditRequest::Redo);
            }
            _ => {}
        }
    }
}

pub fn update_edit_button_colors(
    controls: Res<EditControls>,
    mut button_query: UiButtonColorQuery,
) {
    for (button, interaction, color) in &mut button_query {
        let available = match button {
            ViewerUiButton::UndoArrangement => controls.can_undo,
            ViewerUiButton::RedoArrangement => controls.can_redo,
            _ => continue,
        };
        if available {
            set_button_color(color, false, interaction);
        } else {
            set_disabled_button_color(color);
        }
    }
}
