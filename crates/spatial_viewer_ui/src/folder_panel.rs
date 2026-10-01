//! The folders pill: how many folders the scene has and how many are open,
//! with buttons that open or close them all at once, and which parts of
//! every folder show.

use bevy::prelude::*;

use crate::{
    display_if, expanded_panel, pill_node, set_button_color, set_header_color, spawn_button_row,
    spawn_checkbox_row, spawn_pill_button, ButtonInteractionQuery, UiButtonColorQuery,
    UiPanelQuery, UiTextQuery, ViewerUiButton, ViewerUiPanel, ViewerUiText,
};

const COLLAPSED_WIDTH: f32 = 190.0;
const EXPANDED_WIDTH: f32 = 260.0;

/// A button press waiting for the viewer, which owns the folder state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FolderRequest {
    OpenAll,
    CloseAll,
}

/// Which parts of every folder show. With all of them off a folder shows
/// nothing until it is selected or something is dragged over it, and still
/// takes presses where its cube stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FolderDisplay {
    /// Tags (count and name), unless a folder says otherwise.
    pub tags: bool,
    /// The icon that closes an open folder.
    pub close_icons: bool,
    /// The cube's walls.
    pub backgrounds: bool,
}

impl Default for FolderDisplay {
    fn default() -> Self {
        Self {
            tags: true,
            close_icons: true,
            backgrounds: true,
        }
    }
}

/// What the pill shows, published by the viewer, the press it made, and
/// which parts of every folder show.
#[derive(Resource)]
pub struct FolderControls {
    pub folder_count: usize,
    pub open_count: usize,
    pub display: FolderDisplay,
    request: Option<FolderRequest>,
    pill_expanded: bool,
}

impl Default for FolderControls {
    fn default() -> Self {
        Self::with_display(FolderDisplay::default())
    }
}

impl FolderControls {
    pub fn with_display(display: FolderDisplay) -> Self {
        Self {
            folder_count: 0,
            open_count: 0,
            display,
            request: None,
            pill_expanded: false,
        }
    }

    pub fn take_request(&mut self) -> Option<FolderRequest> {
        self.request.take()
    }
}

pub(crate) fn spawn_folder_pill(parent: &mut ChildBuilder) {
    parent
        .spawn(pill_node(COLLAPSED_WIDTH, ViewerUiPanel::FolderPill))
        .with_children(|pill| {
            spawn_pill_button(
                pill,
                ViewerUiButton::ToggleFolderPill,
                ViewerUiText::FolderSummary,
                13.0,
            );
            pill.spawn(expanded_panel(ViewerUiPanel::FolderExpanded))
                .with_children(|expanded| {
                    spawn_button_row(
                        expanded,
                        &[
                            (ViewerUiButton::OpenAllFolders, "Open all"),
                            (ViewerUiButton::CloseAllFolders, "Close all"),
                        ],
                    );
                    for (button, mark, label) in [
                        (
                            ViewerUiButton::ToggleFolderTags,
                            ViewerUiText::FolderTagsMark,
                            "Show folder tags",
                        ),
                        (
                            ViewerUiButton::ToggleFolderCloseIcons,
                            ViewerUiText::FolderCloseIconsMark,
                            "Show minimize icon",
                        ),
                        (
                            ViewerUiButton::ToggleFolderBackgrounds,
                            ViewerUiText::FolderBackgroundsMark,
                            "Show background",
                        ),
                    ] {
                        spawn_checkbox_row(expanded, button, mark, label);
                    }
                });
        });
}

pub fn handle_folder_buttons(
    mut controls: ResMut<FolderControls>,
    interaction_query: ButtonInteractionQuery,
) {
    for (interaction, button) in &interaction_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match *button {
            ViewerUiButton::ToggleFolderPill => controls.pill_expanded = !controls.pill_expanded,
            ViewerUiButton::OpenAllFolders => controls.request = Some(FolderRequest::OpenAll),
            ViewerUiButton::CloseAllFolders => controls.request = Some(FolderRequest::CloseAll),
            ViewerUiButton::ToggleFolderTags => {
                controls.display.tags = !controls.display.tags;
            }
            ViewerUiButton::ToggleFolderCloseIcons => {
                controls.display.close_icons = !controls.display.close_icons;
            }
            ViewerUiButton::ToggleFolderBackgrounds => {
                controls.display.backgrounds = !controls.display.backgrounds;
            }
            _ => {}
        }
    }
}

pub fn update_folder_text(controls: Res<FolderControls>, mut text_query: UiTextQuery) {
    for (mut text, text_kind) in &mut text_query {
        match *text_kind {
            ViewerUiText::FolderSummary => {
                **text = folder_summary(controls.folder_count, controls.open_count);
            }
            ViewerUiText::FolderTagsMark => **text = check_mark(controls.display.tags),
            ViewerUiText::FolderCloseIconsMark => {
                **text = check_mark(controls.display.close_icons);
            }
            ViewerUiText::FolderBackgroundsMark => {
                **text = check_mark(controls.display.backgrounds);
            }
            _ => {}
        }
    }
}

pub fn update_folder_panels(controls: Res<FolderControls>, mut panel_query: UiPanelQuery) {
    for (mut node, panel) in &mut panel_query {
        match panel {
            ViewerUiPanel::FolderPill => {
                node.width = Val::Px(if controls.pill_expanded {
                    EXPANDED_WIDTH
                } else {
                    COLLAPSED_WIDTH
                });
            }
            ViewerUiPanel::FolderExpanded => {
                node.display = display_if(controls.pill_expanded);
            }
            _ => {}
        }
    }
}

pub fn update_folder_button_colors(
    controls: Res<FolderControls>,
    mut button_query: UiButtonColorQuery,
) {
    for (button, interaction, color) in &mut button_query {
        match button {
            ViewerUiButton::ToggleFolderPill => set_header_color(color, interaction),
            ViewerUiButton::OpenAllFolders | ViewerUiButton::CloseAllFolders => {
                set_button_color(color, false, interaction)
            }
            ViewerUiButton::ToggleFolderTags => {
                set_button_color(color, controls.display.tags, interaction)
            }
            ViewerUiButton::ToggleFolderCloseIcons => {
                set_button_color(color, controls.display.close_icons, interaction)
            }
            ViewerUiButton::ToggleFolderBackgrounds => {
                set_button_color(color, controls.display.backgrounds, interaction)
            }
            _ => {}
        }
    }
}

fn check_mark(checked: bool) -> String {
    if checked { "X" } else { "" }.to_owned()
}

fn folder_summary(folder_count: usize, open_count: usize) -> String {
    match (folder_count, open_count) {
        (0, _) => "Folders: none".to_owned(),
        (count, 0) => format!("Folders: {count}"),
        (count, open) => format!("Folders: {open}/{count} open"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_counts_open_folders_only_when_some_are() {
        assert_eq!(folder_summary(0, 0), "Folders: none");
        assert_eq!(folder_summary(12, 0), "Folders: 12");
        assert_eq!(folder_summary(12, 3), "Folders: 3/12 open");
    }
}
