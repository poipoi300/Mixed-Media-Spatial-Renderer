//! The search pill: a box that finds images by file name as it is typed,
//! with buttons that select the matches the scene shows and step the view
//! from one match to the next.

use bevy::prelude::*;

use crate::text_entry::{TextEntry, TextEntryBox, TextEntryTarget};
use crate::{
    display_if, expanded_panel, pill_node, set_button_color, set_disabled_button_color,
    set_header_color, spawn_button_row, spawn_pill_button, spawn_text, ButtonInteractionQuery,
    UiButtonColorQuery, UiPanelQuery, UiTextQuery, ViewerUiButton, ViewerUiPanel, ViewerUiText,
};

const COLLAPSED_WIDTH: f32 = 190.0;
const EXPANDED_WIDTH: f32 = 260.0;
const QUERY_SHOWN_CHARS: usize = 28;

/// A button press waiting for the viewer, which finds the matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchRequest {
    /// Select every match the scene shows.
    SelectShown,
    /// Fly to the next match, opening the folders it is in.
    Next,
}

/// The query, what the viewer found for it, and the press it made.
#[derive(Resource, Default)]
pub struct SearchControls {
    query: String,
    pub match_count: usize,
    /// Matches not hidden in a closed folder.
    pub shown_count: usize,
    request: Option<SearchRequest>,
    pill_expanded: bool,
}

impl SearchControls {
    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn take_request(&mut self) -> Option<SearchRequest> {
        self.request.take()
    }
}

pub(crate) fn spawn_search_pill(parent: &mut ChildBuilder) {
    parent
        .spawn(pill_node(COLLAPSED_WIDTH, ViewerUiPanel::SearchPill))
        .with_children(|pill| {
            spawn_pill_button(
                pill,
                ViewerUiButton::ToggleSearchPill,
                ViewerUiText::SearchSummary,
                13.0,
            );
            pill.spawn(expanded_panel(ViewerUiPanel::SearchExpanded))
                .with_children(|expanded| {
                    expanded
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
                            ViewerUiButton::FocusSearch,
                            TextEntryBox(TextEntryTarget::Search),
                        ))
                        .with_child((
                            Text::new(""),
                            TextFont {
                                font_size: 12.5,
                                ..default()
                            },
                            TextColor(Color::srgb(0.94, 0.96, 0.98)),
                            ViewerUiText::SearchQuery,
                        ));
                    spawn_text(expanded, ViewerUiText::SearchResults);
                    spawn_button_row(
                        expanded,
                        &[
                            (ViewerUiButton::SelectSearchMatches, "Select shown"),
                            (ViewerUiButton::NextSearchMatch, "Next"),
                        ],
                    );
                });
        });
}

pub fn handle_search_buttons(
    mut controls: ResMut<SearchControls>,
    mut text_entry: ResMut<TextEntry>,
    interaction_query: ButtonInteractionQuery,
) {
    for (interaction, button) in &interaction_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match *button {
            ViewerUiButton::ToggleSearchPill => controls.pill_expanded = !controls.pill_expanded,
            ViewerUiButton::FocusSearch => {
                if text_entry.editing() == Some(TextEntryTarget::Search) {
                    text_entry.cancel();
                } else {
                    text_entry.begin(TextEntryTarget::Search, &controls.query);
                }
            }
            ViewerUiButton::SelectSearchMatches if controls.shown_count > 0 => {
                controls.request = Some(SearchRequest::SelectShown);
            }
            ViewerUiButton::NextSearchMatch if controls.match_count > 0 => {
                controls.request = Some(SearchRequest::Next);
            }
            _ => {}
        }
    }
}

/// Takes the query as it is typed, and Enter as a press of Next.
pub fn apply_search_typing(
    mut controls: ResMut<SearchControls>,
    mut text_entry: ResMut<TextEntry>,
) {
    if let Some(text) = text_entry.text(TextEntryTarget::Search) {
        if controls.query != text {
            controls.query = text.to_owned();
        }
    }
    if text_entry.take_submitted(TextEntryTarget::Search).is_some() && controls.match_count > 0 {
        controls.request = Some(SearchRequest::Next);
    }
}

pub fn update_search_text(
    controls: Res<SearchControls>,
    text_entry: Res<TextEntry>,
    mut text_query: UiTextQuery,
) {
    let editing = text_entry.editing() == Some(TextEntryTarget::Search);
    for (mut text, text_kind) in &mut text_query {
        let shown = match *text_kind {
            ViewerUiText::SearchSummary => search_summary(&controls),
            ViewerUiText::SearchQuery => query_line(&controls.query, editing),
            ViewerUiText::SearchResults => results_line(&controls),
            _ => continue,
        };
        if **text != shown {
            **text = shown;
        }
    }
}

pub fn update_search_panels(controls: Res<SearchControls>, mut panel_query: UiPanelQuery) {
    for (mut node, panel) in &mut panel_query {
        match panel {
            ViewerUiPanel::SearchPill => {
                node.width = Val::Px(if controls.pill_expanded {
                    EXPANDED_WIDTH
                } else {
                    COLLAPSED_WIDTH
                });
            }
            ViewerUiPanel::SearchExpanded => node.display = display_if(controls.pill_expanded),
            _ => {}
        }
    }
}

pub fn update_search_button_colors(
    controls: Res<SearchControls>,
    text_entry: Res<TextEntry>,
    mut button_query: UiButtonColorQuery,
) {
    for (button, interaction, color) in &mut button_query {
        let (available, active) = match button {
            ViewerUiButton::ToggleSearchPill => {
                set_header_color(color, interaction);
                continue;
            }
            ViewerUiButton::FocusSearch => {
                (true, text_entry.editing() == Some(TextEntryTarget::Search))
            }
            ViewerUiButton::SelectSearchMatches => (controls.shown_count > 0, false),
            ViewerUiButton::NextSearchMatch => (controls.match_count > 0, false),
            _ => continue,
        };
        if available {
            set_button_color(color, active, interaction);
        } else {
            set_disabled_button_color(color);
        }
    }
}

fn search_summary(controls: &SearchControls) -> String {
    if controls.query.is_empty() {
        "Search".to_owned()
    } else {
        format!("Search: {}", controls.match_count)
    }
}

fn query_line(query: &str, editing: bool) -> String {
    let tail: String = {
        let count = query.chars().count();
        query
            .chars()
            .skip(count.saturating_sub(QUERY_SHOWN_CHARS))
            .collect()
    };
    match (editing, query.is_empty()) {
        (true, _) => format!("{tail}|"),
        (false, true) => "Click to type a file name".to_owned(),
        (false, false) => tail,
    }
}

fn results_line(controls: &SearchControls) -> String {
    match (controls.query.is_empty(), controls.match_count) {
        (true, _) => "Finds images by file name.".to_owned(),
        (false, 0) => "No matches".to_owned(),
        (false, count) if controls.shown_count == count => format!("{count} matches"),
        (false, count) => format!(
            "{count} matches, {} in closed folders",
            count - controls.shown_count
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_say_how_many_are_hidden_in_folders() {
        let mut controls = SearchControls {
            query: "cat".to_owned(),
            match_count: 5,
            shown_count: 3,
            ..default()
        };
        assert_eq!(results_line(&controls), "5 matches, 2 in closed folders");
        controls.shown_count = 5;
        assert_eq!(results_line(&controls), "5 matches");
        controls.match_count = 0;
        assert_eq!(results_line(&controls), "No matches");
        assert_eq!(search_summary(&controls), "Search: 0");
    }

    #[test]
    fn a_long_query_shows_its_end() {
        let query = "a".repeat(40) + "end";
        assert!(query_line(&query, false).ends_with("end"));
        assert_eq!(query_line(&query, false).chars().count(), QUERY_SHOWN_CHARS);
    }
}
