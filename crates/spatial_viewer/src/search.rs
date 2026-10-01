//! Search: finds the images whose file name holds the query, selects the
//! ones the scene shows, and steps the view from one match to the next,
//! opening the folders a match is hidden in on the way.

use std::collections::HashSet;
use std::path::Path;

use bevy::prelude::*;
use spatial_api::ProjectionPage;
use spatial_viewer_ui::{SearchControls, SearchRequest};

use crate::fit_view::{fly_to_fit, CameraFlight};
use crate::folders::FolderViewState;
use crate::manual_spacing::SelectionState;
use crate::{ExplorerScene, FlyCamera};

#[derive(Resource, Default)]
pub(crate) struct SearchState {
    /// The matches, in file name order, and the query and layout they were
    /// found for.
    matches: Vec<usize>,
    found_for: Option<(String, u64)>,
    /// Which match Next goes to.
    next: usize,
    /// The match the view flies to once the scene shows it.
    pending: Option<usize>,
}

/// The images of `projection` whose file name holds `query`, ignoring case,
/// in file name order. Nothing matches an empty query.
fn find_matches(projection: &ProjectionPage, query: &str) -> Vec<usize> {
    let needle = query.trim().to_lowercase();
    if needle.is_empty() {
        return Vec::new();
    }
    let mut matches: Vec<(String, usize)> = projection
        .points
        .iter()
        .filter_map(|point| {
            let name = Path::new(&point.path)
                .file_name()?
                .to_string_lossy()
                .to_lowercase();
            name.contains(&needle).then_some((name, point.image_id))
        })
        .collect();
    matches.sort();
    matches.into_iter().map(|(_, image_id)| image_id).collect()
}

/// Keeps the search pill's counts current and carries out its buttons.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_search(
    real_time: Res<Time<Real>>,
    mut controls: ResMut<SearchControls>,
    mut state: ResMut<SearchState>,
    scene: Res<ExplorerScene>,
    mut selection: ResMut<SelectionState>,
    mut folder_view: ResMut<FolderViewState>,
    mut flight: ResMut<CameraFlight>,
    camera: Query<(&Transform, &Projection), With<FlyCamera>>,
) {
    let found_for = (controls.query().to_owned(), scene.folder_generation);
    // Full size in the scene, so neither hidden nor a closed folder's
    // preview.
    let shown: HashSet<usize> = scene
        .image_points
        .iter()
        .filter(|point| point.scale >= 1.0)
        .map(|point| point.image_id)
        .collect();
    if state.found_for.as_ref() != Some(&found_for) {
        if state
            .found_for
            .as_ref()
            .is_none_or(|(query, _)| *query != found_for.0)
        {
            state.next = 0;
        }
        state.matches = find_matches(&scene.projection, &found_for.0);
        state.found_for = Some(found_for);
        let shown_count = state
            .matches
            .iter()
            .filter(|image_id| shown.contains(image_id))
            .count();
        if controls.match_count != state.matches.len() || controls.shown_count != shown_count {
            controls.match_count = state.matches.len();
            controls.shown_count = shown_count;
        }
    }

    match controls.take_request() {
        Some(SearchRequest::SelectShown) => {
            selection.select_only_images(
                state
                    .matches
                    .iter()
                    .copied()
                    .filter(|image_id| shown.contains(image_id)),
            );
        }
        Some(SearchRequest::Next) if !state.matches.is_empty() => {
            let image_id = state.matches[state.next % state.matches.len()];
            state.next += 1;
            state.pending = Some(image_id);
        }
        _ => {}
    }

    let Some(image_id) = state.pending else {
        return;
    };
    if !state.matches.contains(&image_id) {
        // Gone from the scene, or no longer a match.
        state.pending = None;
        return;
    }
    if !shown.contains(&image_id) {
        // Opens the folders it is in, once the scene knows them and as long
        // as any of them is closed.
        let enclosing: Vec<_> = scene
            .folders
            .folder_of(image_id)
            .and_then(|folder| scene.folders.index_of(&folder.key))
            .map(|index| scene.folders.ancestry_keys(index).cloned().collect())
            .unwrap_or_default();
        if enclosing.iter().any(|key| !folder_view.is_open(key)) {
            folder_view.open_all(enclosing.iter(), real_time.elapsed_secs());
        }
        return;
    }
    let (Some(point), Ok((camera_transform, Projection::Perspective(perspective)))) = (
        scene
            .image_points
            .iter()
            .find(|point| point.image_id == image_id),
        camera.get_single(),
    ) else {
        return;
    };
    state.pending = None;
    let half = scene.billboard_world_size * 0.5;
    let right = camera_transform.rotation * Vec3::X * half;
    let up = camera_transform.rotation * Vec3::Y * half;
    let corners = [
        point.position - right - up,
        point.position + right - up,
        point.position + right + up,
        point.position - right + up,
    ];
    fly_to_fit(
        &mut flight,
        camera_transform,
        perspective,
        &corners,
        scene.billboard_world_size,
    );
    selection.select_only_images([image_id]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatial_api::ProjectionPoint;

    #[test]
    fn matches_are_file_names_holding_the_query_in_name_order() {
        let point = |image_id: usize, path: &str| ProjectionPoint {
            image_id,
            path: path.to_owned(),
            position: [0.0; 3],
            group: None,
            width: None,
            height: None,
            media_type: "image".to_owned(),
            duration_seconds: None,
            coordinate_labels: [None, None, None],
        };
        let projection = ProjectionPage {
            points: vec![
                point(0, "cats/Zebra_cat.png"),
                point(1, "cats/dog.png"),
                point(2, "dogs/a_CAT.jpg"),
            ],
            ..default()
        };
        assert_eq!(find_matches(&projection, " Cat "), vec![2, 0]);
        // Folder names are not searched.
        assert_eq!(find_matches(&projection, "cats"), Vec::<usize>::new());
        assert!(find_matches(&projection, "").is_empty());
    }
}
