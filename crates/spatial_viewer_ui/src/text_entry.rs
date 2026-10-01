//! Typing outside the control panel: renaming a folder in a small prompt,
//! the search box, and the controls sheet's filter.
//!
//! One [`TextEntry`] edits one piece of text at a time, for one
//! [`TextEntryTarget`]. The viewer starts an edit, reads the text as it is
//! typed ([`TextEntry::text`]) and takes what Enter submits
//! ([`TextEntry::take_submitted`]). Escape ends an edit (see the viewer's
//! escape handling), and so does a press anywhere outside the box it is
//! typed into ([`TextEntryBox`]).

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use bevy::ui::FocusPolicy;

const MAX_CHARS: usize = 80;
const PROMPT_WIDTH: f32 = 360.0;
const PROMPT_Z_INDEX: i32 = 90;

/// What an edit is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextEntryTarget {
    /// A folder's name, typed into the rename prompt. Enter submits it and
    /// ends the edit.
    FolderName,
    /// The search box. The text applies as it is typed; Enter submits it and
    /// keeps editing.
    Search,
    /// The controls sheet's filter. The text applies as it is typed; Enter
    /// ends the edit.
    ControlsFilter,
}

impl TextEntryTarget {
    fn ends_on_submit(self) -> bool {
        match self {
            Self::FolderName | Self::ControlsFilter => true,
            Self::Search => false,
        }
    }

    /// Enter hands the text over, rather than only ending the edit.
    fn submits(self) -> bool {
        match self {
            Self::FolderName | Self::Search => true,
            Self::ControlsFilter => false,
        }
    }
}

/// The UI node an edit for its target is typed into: a press anywhere else
/// ends the edit.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub struct TextEntryBox(pub TextEntryTarget);

#[derive(Resource, Default)]
pub struct TextEntry {
    editing: Option<TextEntryTarget>,
    text: String,
    submitted: Option<(TextEntryTarget, String)>,
}

impl TextEntry {
    /// Starts editing `initial` for `target`, replacing any other edit.
    pub fn begin(&mut self, target: TextEntryTarget, initial: &str) {
        self.editing = Some(target);
        self.text = initial.chars().take(MAX_CHARS).collect();
    }

    /// Ends the edit without submitting it.
    pub fn cancel(&mut self) {
        self.editing = None;
    }

    pub fn is_active(&self) -> bool {
        self.editing.is_some()
    }

    pub fn editing(&self) -> Option<TextEntryTarget> {
        self.editing
    }

    /// The text being typed for `target`, while it is being edited.
    pub fn text(&self, target: TextEntryTarget) -> Option<&str> {
        (self.editing == Some(target)).then_some(self.text.as_str())
    }

    /// What Enter submitted for `target` since this was last called.
    pub fn take_submitted(&mut self, target: TextEntryTarget) -> Option<String> {
        if self
            .submitted
            .as_ref()
            .is_some_and(|(submitted, _)| *submitted == target)
        {
            return self.submitted.take().map(|(_, text)| text);
        }
        None
    }

    fn push(&mut self, character: char) {
        if self.text.chars().count() < MAX_CHARS {
            self.text.push(character);
        }
    }

    fn submit(&mut self) {
        let Some(target) = self.editing else {
            return;
        };
        if target.submits() {
            self.submitted = Some((target, self.text.clone()));
        }
        if target.ends_on_submit() {
            self.editing = None;
        }
    }
}

/// Types into the edit in progress, and ends it when a press lands outside
/// its box.
pub fn handle_text_entry_keyboard(
    mut entry: ResMut<TextEntry>,
    mut keyboard_events: EventReader<KeyboardInput>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    boxes: Query<(&Interaction, &TextEntryBox)>,
) {
    let Some(editing) = entry.editing else {
        keyboard_events.clear();
        return;
    };
    let pressed = [MouseButton::Left, MouseButton::Right]
        .into_iter()
        .any(|button| mouse_buttons.just_pressed(button));
    let on_box = boxes
        .iter()
        .any(|(interaction, text_box)| text_box.0 == editing && *interaction != Interaction::None);
    if pressed && !on_box {
        entry.cancel();
        keyboard_events.clear();
        return;
    }
    for event in keyboard_events.read() {
        if !event.state.is_pressed() {
            continue;
        }
        match &event.logical_key {
            Key::Character(characters) => {
                for character in characters
                    .chars()
                    .filter(|character| !character.is_control())
                {
                    entry.push(character);
                }
            }
            Key::Space => entry.push(' '),
            Key::Backspace => {
                entry.text.pop();
            }
            Key::Enter => entry.submit(),
            _ => {}
        }
    }
}

/// The rename prompt's parts.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub enum RenamePromptPart {
    Panel,
    Text,
}

pub(crate) fn spawn_rename_prompt(parent: &mut ChildBuilder) {
    parent
        .spawn((
            Node {
                display: Display::None,
                position_type: PositionType::Absolute,
                top: Val::Percent(18.0),
                left: Val::Percent(50.0),
                width: Val::Px(PROMPT_WIDTH),
                margin: UiRect::left(Val::Px(-PROMPT_WIDTH * 0.5)),
                padding: UiRect::all(Val::Px(14.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(8.0),
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderRadius::all(Val::Px(8.0)),
            BorderColor(Color::srgba(0.78, 0.84, 0.92, 0.60)),
            BackgroundColor(Color::srgba(0.035, 0.040, 0.052, 0.97)),
            ZIndex(PROMPT_Z_INDEX),
            Interaction::default(),
            FocusPolicy::Block,
            RenamePromptPart::Panel,
            TextEntryBox(TextEntryTarget::FolderName),
        ))
        .with_children(|prompt| {
            prompt.spawn((
                Text::new("Rename folder"),
                TextFont {
                    font_size: 15.0,
                    ..default()
                },
                TextColor(Color::srgb(0.96, 0.97, 0.99)),
            ));
            prompt
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        min_height: Val::Px(30.0),
                        padding: UiRect::horizontal(Val::Px(8.0)),
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderRadius::all(Val::Px(6.0)),
                    BorderColor(Color::srgb(0.22, 0.68, 1.0)),
                    BackgroundColor(Color::srgba(0.06, 0.07, 0.09, 0.95)),
                ))
                .with_child((
                    Text::new(""),
                    TextFont {
                        font_size: 13.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.94, 0.96, 0.98)),
                    RenamePromptPart::Text,
                ));
            prompt.spawn((
                Text::new("Enter to save, Esc to cancel. Empty restores the default name."),
                TextFont {
                    font_size: 11.0,
                    ..default()
                },
                TextColor(Color::srgba(0.70, 0.76, 0.84, 0.85)),
            ));
        });
}

pub fn update_rename_prompt(
    entry: Res<TextEntry>,
    mut panels: Query<(&mut Node, &RenamePromptPart)>,
    mut texts: Query<(&mut Text, &RenamePromptPart)>,
) {
    let text = entry.text(TextEntryTarget::FolderName);
    for (mut node, part) in &mut panels {
        if *part == RenamePromptPart::Panel {
            node.display = crate::display_if(text.is_some());
        }
    }
    let Some(text) = text else {
        return;
    };
    for (mut shown, part) in &mut texts {
        if *part == RenamePromptPart::Text {
            let with_caret = format!("{text}|");
            if **shown != with_caret {
                **shown = with_caret;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folder_name_ends_on_submit_and_search_keeps_editing() {
        let mut entry = TextEntry::default();
        entry.begin(TextEntryTarget::FolderName, "Old");
        entry.push('!');
        entry.submit();
        assert!(!entry.is_active());
        assert_eq!(
            entry.take_submitted(TextEntryTarget::FolderName).as_deref(),
            Some("Old!")
        );
        assert_eq!(entry.take_submitted(TextEntryTarget::FolderName), None);

        entry.begin(TextEntryTarget::Search, "cat");
        entry.submit();
        assert_eq!(entry.text(TextEntryTarget::Search), Some("cat"));
        assert_eq!(entry.take_submitted(TextEntryTarget::FolderName), None);
        assert_eq!(
            entry.take_submitted(TextEntryTarget::Search).as_deref(),
            Some("cat")
        );

        entry.begin(TextEntryTarget::ControlsFilter, "zoom");
        entry.submit();
        assert!(!entry.is_active());
        assert_eq!(entry.take_submitted(TextEntryTarget::ControlsFilter), None);
    }
}
