//! Loading a catalog and turning it into the live scene.
//!
//! A load runs on a background thread that streams NDJSON snapshots from the
//! Python API; each snapshot is a complete view of everything discovered so
//! far, so coordinates move as dimension extents grow. Whatever can be
//! derived without ECS or asset access (bounds, billboard points, the point
//! cloud partition, the pending spatial index) is computed off-thread in
//! [`PreparedProjection`] so the main thread only has to swap the results in.
//!
//! Applying a snapshot avoids a full teardown whenever it can:
//! [`projection_preserves_billboards`] checks that every retained record kept
//! its identity and label contract, and only a genuine change (new axes, a
//! changed path) tears down decoded textures. Positions are pinned per image
//! for the lifetime of a load by [`stabilize_streamed_projection_positions`],
//! so a partial catalog never shuffles the scene around the player.

use std::collections::HashMap;
use std::sync::mpsc::{channel, sync_channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use generation_api::{CatalogStreamEvent, ControlPanel, ProjectionPage};
use generation_geometry::{
    api_position_to_viewer, estimate_smallest_axis_gap, projection_bounds, Bounds3,
};
use generation_viewer_ui::{
    BillboardControls, BillboardStats, ControlPanelState, NavigationSettings, NavigationTargets,
    PauseMenuState, PendingSubmit, StartMenuState,
};

use crate::axis_gizmo::AxisGizmoLabels;
use crate::catalog_session::save_last_catalog_roots;
use crate::image_loading::{BillboardPoint, ImageLoadingState, PreparedPendingIndex};
use crate::manual_spacing::{ImagePointIndex, SelectionHighlights, SelectionState};
use crate::point_cloud::{point_color, PointCloud, PointCloudLayout, PointCloudPoint, ViewBounds};
use crate::{
    axis_label, billboard_axis_labels, drop_in_background, initial_camera_distance,
    navigation_speed, place_camera_at_initial_view, point_cloud_point_size,
    set_scene_camera_far_plane, BillboardEntitiesQuery, ExplorerScene, FlyCameraProjectionQuery,
    FlyCameraTransformQuery,
};

/// Result of a background catalog load kicked off from the start menu.
pub(crate) struct LoadedCatalog {
    pub panel: ControlPanel,
    pub projection: ProjectionPage,
    pub complete: bool,
    pub roots: Option<Vec<String>>,
}

/// Everything a scene needs from a projection that can be derived without
/// touching ECS or assets. Built on the catalog load thread so applying a
/// snapshot on the main thread costs O(chunks), not O(points).
pub(crate) struct PreparedProjection {
    pub panel: ControlPanel,
    pub projection: ProjectionPage,
    pub complete: bool,
    pub roots: Option<Vec<String>>,
    bounds: Bounds3,
    image_points: Vec<BillboardPoint>,
    nearest_gap: Option<f32>,
    navigation_targets: NavigationTargets,
    point_index: ImagePointIndex,
    point_cloud_layout: PointCloudLayout,
    /// Pending-load index over every point, used when the snapshot resets
    /// the billboard cache (first snapshot or changed identity contract).
    /// Streaming snapshots that preserve billboards reconcile in place
    /// instead and leave this unused.
    pending_index: PreparedPendingIndex,
}

impl PreparedProjection {
    fn new(loaded: LoadedCatalog) -> Self {
        let projection = loaded.projection;
        let bounds = projection_bounds(&projection.points);
        let image_points = projection_billboard_points(&projection);
        let nearest_gap = estimate_smallest_axis_gap(&projection.points);
        let navigation_targets =
            NavigationTargets::from_positions(image_points.iter().map(|point| point.position));
        let point_index = ImagePointIndex::new(&image_points);
        let point_cloud_layout = PointCloudLayout::new(
            point_cloud_points(&projection),
            point_cloud_point_size(&bounds),
        );
        let pending_index = PreparedPendingIndex::new(image_points.clone());
        Self {
            panel: loaded.panel,
            projection,
            complete: loaded.complete,
            roots: loaded.roots,
            bounds,
            image_points,
            nearest_gap,
            navigation_targets,
            point_index,
            point_cloud_layout,
            pending_index,
        }
    }

    /// Re-derives the snapshot with `overrides` applied to the matching
    /// projection points. O(points), so callers reserve it for the rare
    /// case of a manual drag during streaming.
    fn with_position_overrides(mut self, overrides: &HashMap<usize, Vec3>) -> Self {
        for point in &mut self.projection.points {
            if let Some(position) = overrides.get(&point.image_id) {
                point.position = [position.x, position.y, position.z];
            }
        }
        Self::new(LoadedCatalog {
            panel: self.panel,
            projection: self.projection,
            complete: self.complete,
            roots: self.roots,
        })
    }
}

/// Receiver for the catalog load running on a background thread, so the
/// blocking HTTP requests cannot freeze the UI. The `Mutex` only satisfies
/// the `Sync` bound resources need; systems access it via `get_mut`.
///
/// The load thread also owns streamed-position stabilisation and the
/// derived-data precomputation ([`PreparedProjection`]).
#[derive(Resource, Default)]
pub(crate) struct CatalogLoadTask {
    receiver: Mutex<Option<Receiver<Result<PreparedProjection, String>>>>,
    first_snapshot: bool,
    catalog: bool,
    /// Latest control submission waiting for the in-flight load to finish.
    /// Only the newest is kept: an older set of values is superseded, never
    /// queued behind the new one.
    pending_submit: Option<PendingSubmit>,
    /// Positions the user dragged images to since this load began. Applied
    /// over the load thread's first-streamed pins so drags survive later
    /// snapshots of the same stream.
    pub dragged_positions: HashMap<usize, Vec3>,
}

/// Runs on the catalog load thread: pins every image to its first streamed
/// coordinate, then derives the scene data the main thread would otherwise
/// compute per snapshot. Returns `false` when the viewer dropped the load.
pub(crate) fn prepare_loaded_catalog(
    sender: &std::sync::mpsc::SyncSender<Result<PreparedProjection, String>>,
    stable_positions: &mut HashMap<usize, [f32; 3]>,
    mut loaded: LoadedCatalog,
) -> bool {
    stabilize_streamed_projection_positions(&mut loaded.projection, stable_positions);
    sender.send(Ok(PreparedProjection::new(loaded))).is_ok()
}

/// Lifetime guard for automatic player placement. The camera may be placed
/// when the first non-empty coordinate set arrives, but catalog streaming,
/// reloads, and projection changes never own its position afterward.
#[derive(Resource, Default)]
pub(crate) struct InitialPlayerPlacement {
    pub positioned: bool,
}

/// Receiver for the native folder picker dialog running on a background
/// thread; `None` payload means the dialog was cancelled.
#[derive(Resource, Default)]
pub(crate) struct FolderPickTask(pub Mutex<Option<Receiver<Option<String>>>>);

/// Folder the picker dialog opened last time, so re-opening it starts where
/// the user left off.
#[derive(Resource, Default)]
pub(crate) struct LastPickedFolder(pub Option<String>);

/// Catalog roots saved by the previous session. Consumed once by
/// `restore_last_catalog` on the first frame to kick off the initial load.
#[derive(Resource, Default)]
pub(crate) struct RestoredCatalogRoots(pub Vec<String>);

/// Bundled so `poll_catalog_load_task` stays within Bevy's supported
/// system-function parameter arity.
#[derive(SystemParam)]
pub(crate) struct CatalogMenuState<'w> {
    start_menu: ResMut<'w, StartMenuState>,
    pause_menu: ResMut<'w, PauseMenuState>,
    initial_player_placement: ResMut<'w, InitialPlayerPlacement>,
}

#[derive(SystemParam)]
pub(crate) struct SceneReloadQueries<'w, 's> {
    billboards: BillboardEntitiesQuery<'w, 's>,
    camera: FlyCameraTransformQuery<'w, 's>,
    camera_projection: FlyCameraProjectionQuery<'w, 's>,
}

/// Asset stores a scene reload writes to; bundled so
/// `poll_catalog_load_task` stays within Bevy's supported system-function
/// parameter arity.
#[derive(SystemParam)]
pub(crate) struct SceneReloadAssets<'w> {
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    images: ResMut<'w, Assets<Image>>,
}

pub(crate) fn point_cloud_points(
    projection: &generation_api::ProjectionPage,
) -> Vec<PointCloudPoint> {
    projection
        .points
        .iter()
        .map(|point| {
            let position = api_position_to_viewer(point.position);
            PointCloudPoint {
                image_id: point.image_id,
                position: Vec3::new(position.x, position.y, position.z),
                color: point_color(point.image_id),
            }
        })
        .collect()
}

pub(crate) fn projection_billboard_points(
    projection: &generation_api::ProjectionPage,
) -> Vec<BillboardPoint> {
    projection
        .points
        .iter()
        .map(|point| {
            let position = api_position_to_viewer(point.position);
            BillboardPoint {
                image_id: point.image_id,
                path: Arc::from(point.path.as_str()),
                position: Vec3::new(position.x, position.y, position.z),
                is_video: point.media_type == "video",
                duration_seconds: point.duration_seconds,
                source_size: point
                    .width
                    .zip(point.height)
                    .map(|(width, height)| UVec2::new(width, height)),
                coordinate_labels: point
                    .coordinate_labels
                    .clone()
                    .map(|label| label.map(Arc::from)),
            }
        })
        .collect()
}

/// First-frame hook: if the previous session saved catalog roots, put them
/// in the start menu and request a load exactly as "Load selected" would,
/// so the window opens on the last-used catalog without blocking startup.
pub(crate) fn restore_last_catalog(
    mut restored: ResMut<RestoredCatalogRoots>,
    mut start_menu: ResMut<StartMenuState>,
    mut pause_menu: ResMut<PauseMenuState>,
) {
    if restored.0.is_empty() {
        return;
    }
    for root in std::mem::take(&mut restored.0) {
        start_menu.add_root(root);
    }
    start_menu.request_load();
    // Show the catalog screen while the restore runs so the status line and
    // the loaded roots are visible; a successful load closes it again.
    pause_menu.open_catalog_options_screen();
}

pub(crate) fn handle_start_menu_requests(
    mut start_menu: ResMut<StartMenuState>,
    mut pause_menu: ResMut<PauseMenuState>,
    scene: Res<ExplorerScene>,
    controls: Res<ControlPanelState>,
    mut folder_pick_task: ResMut<FolderPickTask>,
    mut catalog_load_task: ResMut<CatalogLoadTask>,
    last_folder: Res<LastPickedFolder>,
) {
    if start_menu.take_use_current_request() {
        pause_menu.back();
    }

    if start_menu.take_clear_request() {
        start_menu.roots.clear();
        start_menu.set_status("Folder selection cleared.");
    }

    if let Some(index) = start_menu.take_remove_request() {
        start_menu.remove_root(index);
        start_menu.set_status("Folder removed.");
    }

    let pick_slot = folder_pick_task.0.get_mut().expect("folder pick task lock");
    if start_menu.take_add_folder_request() && pick_slot.is_none() {
        let (sender, receiver) = channel();
        let initial_directory = last_folder
            .0
            .clone()
            .or_else(|| start_menu.roots.last().cloned());
        // The native dialog blocks until dismissed, so it runs on its own
        // thread and reports back through the channel.
        std::thread::spawn(move || {
            let _ = sender.send(pick_folder_dialog(initial_directory.as_deref()));
        });
        *pick_slot = Some(receiver);
        start_menu.picking_folder = true;
        start_menu.set_status("Choose a folder in the system dialog...");
    }

    if !start_menu.take_load_request() {
        return;
    }
    if start_menu.roots.is_empty() {
        start_menu.set_status("Add at least one folder first.");
        return;
    }
    let load_slot = catalog_load_task
        .receiver
        .get_mut()
        .expect("catalog load task lock");
    if load_slot.is_some() {
        return;
    }

    // Catalog reload + projection are blocking HTTP calls that can take
    // seconds; running them inline froze the whole app on "Load selected".
    let (sender, receiver) = sync_channel(1);
    let client = scene.client.clone();
    // A fresh load starts from whatever controls the panel currently holds,
    // so reloading a catalog keeps the view the user set up.
    let request = scene
        .projection_request()
        .with_roots(start_menu.roots.clone())
        .with_controls(controls.values().clone(), None);
    std::thread::spawn(move || {
        let mut stable_positions = HashMap::new();
        let result = client.stream_catalog(&request, |event| match event {
            CatalogStreamEvent::Snapshot {
                panel,
                projection,
                complete,
                roots,
            } => prepare_loaded_catalog(
                &sender,
                &mut stable_positions,
                LoadedCatalog {
                    panel,
                    projection: *projection,
                    complete,
                    roots: Some(roots),
                },
            ),
            CatalogStreamEvent::Error { message } => sender.send(Err(message)).is_ok(),
            CatalogStreamEvent::Heartbeat => true,
        });
        if let Err(error) = result {
            let _ = sender.send(Err(error));
        }
    });
    *load_slot = Some(receiver);
    catalog_load_task.first_snapshot = true;
    catalog_load_task.catalog = true;
    catalog_load_task.pending_submit = None;
    catalog_load_task.dragged_positions.clear();
    start_menu.loading = true;
    start_menu.set_status("Loading catalog...");
}

pub(crate) fn poll_folder_pick_task(
    mut start_menu: ResMut<StartMenuState>,
    mut folder_pick_task: ResMut<FolderPickTask>,
    mut last_folder: ResMut<LastPickedFolder>,
) {
    let pick_slot = folder_pick_task.0.get_mut().expect("folder pick task lock");
    let Some(receiver) = pick_slot.as_ref() else {
        return;
    };
    match receiver.try_recv() {
        Ok(Some(folder)) => {
            last_folder.0 = Some(folder.clone());
            start_menu.add_root(folder);
            start_menu.set_status("Folder added.");
        }
        Ok(None) => start_menu.set_status("No folder selected."),
        Err(TryRecvError::Empty) => return,
        Err(TryRecvError::Disconnected) => {
            start_menu.set_status("Folder dialog failed.");
        }
    }
    *pick_slot = None;
    start_menu.picking_folder = false;
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn poll_catalog_load_task(
    mut commands: Commands,
    mut catalog_load_task: ResMut<CatalogLoadTask>,
    mut menu: CatalogMenuState,
    mut controls: ResMut<ControlPanelState>,
    mut scene: ResMut<ExplorerScene>,
    billboard_controls: Res<BillboardControls>,
    mut loading_state: ResMut<ImageLoadingState>,
    mut navigation_settings: ResMut<NavigationSettings>,
    mut navigation_targets: ResMut<NavigationTargets>,
    mut billboard_stats: ResMut<BillboardStats>,
    mut cloud: ResMut<PointCloud>,
    mut view_bounds: ResMut<ViewBounds>,
    mut gizmo_labels: ResMut<AxisGizmoLabels>,
    mut assets: SceneReloadAssets,
    mut scene_queries: SceneReloadQueries,
) {
    let load_slot = catalog_load_task
        .receiver
        .get_mut()
        .expect("catalog load task lock");
    let Some(receiver) = load_slot.as_ref() else {
        return;
    };
    let result = match receiver.try_recv() {
        Ok(result) => result,
        Err(TryRecvError::Empty) => return,
        Err(TryRecvError::Disconnected) => Err("Catalog load thread failed.".to_owned()),
    };
    match result {
        Ok(mut loaded) => {
            // The load thread pinned every image to its first streamed
            // coordinate. Manual drags happen here on the main thread, so
            // images moved since this load began override that pin; the
            // derived scene data is re-derived only when a drag happened.
            if !catalog_load_task.dragged_positions.is_empty() {
                loaded = loaded.with_position_overrides(&catalog_load_task.dragged_positions);
            }
            let first_snapshot = catalog_load_task.first_snapshot;
            let position_player =
                !menu.initial_player_placement.positioned && !loaded.projection.points.is_empty();
            catalog_load_task.first_snapshot = false;
            if position_player {
                menu.initial_player_placement.positioned = true;
            }
            if loaded.complete {
                *catalog_load_task
                    .receiver
                    .get_mut()
                    .expect("catalog load task lock") = None;
                menu.start_menu.loading = false;
            }
            // Preserve textures while coordinates move. A changed identity or
            // label contract requires recreating the outgoing billboards.
            let reset = first_snapshot
                || !projection_preserves_billboards(&scene.projection, &loaded.projection);
            let complete = loaded.complete;
            let roots = loaded.roots.clone();
            apply_projection_to_scene(
                &mut commands,
                &mut scene,
                &mut controls,
                &billboard_controls,
                &mut loading_state,
                &mut navigation_settings,
                &mut navigation_targets,
                &mut billboard_stats,
                &mut cloud,
                &mut view_bounds,
                &mut gizmo_labels,
                &mut assets.meshes,
                &mut assets.materials,
                &mut assets.images,
                &mut scene_queries,
                loaded,
                reset,
                position_player,
            );
            menu.start_menu.set_status(catalog_load_status(
                scene.projection.points.len(),
                scene.projection.total,
                complete,
            ));
            if complete {
                if let Some(roots) = roots {
                    if std::env::var_os("GENERATION_VIEWER_PERF_CONFIG").is_none() {
                        if let Err(error) = save_last_catalog_roots(&roots) {
                            eprintln!("Failed to remember catalog roots: {error}");
                        }
                    }
                }
            }
            if catalog_load_task.catalog
                && first_snapshot
                && (!scene.projection.points.is_empty() || complete)
            {
                menu.pause_menu.resume();
            }
        }
        Err(error) => {
            *catalog_load_task
                .receiver
                .get_mut()
                .expect("catalog load task lock") = None;
            menu.start_menu.loading = false;
            controls.set_request_error(error.clone());
            menu.start_menu.set_status(error);
        }
    }
}

pub(crate) fn projection_preserves_billboards(
    previous: &ProjectionPage,
    next: &ProjectionPage,
) -> bool {
    // Axis meaning is the label contract: when a server relabels an axis,
    // every billboard's coordinate label changes with it.
    if previous.axis_labels != next.axis_labels {
        return false;
    }
    let next_by_id: HashMap<_, _> = next
        .points
        .iter()
        .map(|point| (point.image_id, point))
        .collect();
    previous.points.iter().all(|point| {
        next_by_id.get(&point.image_id).is_some_and(|next| {
            point.path == next.path && point.coordinate_labels == next.coordinate_labels
        })
    })
}

/// A partial catalog does not yet know the final dimension extents, so a
/// freshly recomputed projection may move records that were already shown.
/// Give every image its first streamed coordinate for the lifetime of this
/// load; newly discovered images still appear immediately at their own first
/// A partial catalog does not yet know its final extents, so a freshly
/// recomputed projection may move records that were already shown. Give every
/// image its first streamed coordinate for the lifetime of this load; newly
/// discovered images still appear immediately at their own first coordinate
/// without shifting the existing scene around the player.
pub(crate) fn stabilize_streamed_projection_positions(
    projection: &mut ProjectionPage,
    stable_positions: &mut HashMap<usize, [f32; 3]>,
) {
    for point in &mut projection.points {
        let stable = stable_positions
            .entry(point.image_id)
            .or_insert(point.position);
        point.position = *stable;
    }
}

/// Tears down the current projection scene (markers, billboards, gizmo) and
/// rebuilds it for `projection`, updating navigation, loading, and camera
/// state to match. Shared by catalog reloads and dimension changes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_projection_to_scene(
    commands: &mut Commands,
    scene: &mut ExplorerScene,
    controls: &mut ControlPanelState,
    billboard_controls: &BillboardControls,
    loading_state: &mut ImageLoadingState,
    navigation_settings: &mut NavigationSettings,
    navigation_targets: &mut NavigationTargets,
    billboard_stats: &mut BillboardStats,
    cloud: &mut PointCloud,
    view_bounds: &mut ViewBounds,
    gizmo_labels: &mut AxisGizmoLabels,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    scene_queries: &mut SceneReloadQueries,
    prepared: PreparedProjection,
    reset_billboards: bool,
    position_player: bool,
) {
    if reset_billboards {
        for (entity, billboard) in scene_queries.billboards.iter() {
            billboard.remove_assets(images, materials);
            commands.entity(entity).despawn_recursive();
        }
    }

    let PreparedProjection {
        panel,
        projection,
        bounds,
        image_points,
        nearest_gap,
        navigation_targets: prepared_targets,
        point_index,
        point_cloud_layout,
        pending_index,
        ..
    } = prepared;
    let nearest_gap = nearest_gap.unwrap_or(scene.coordinate_spacing);
    let camera_speed = navigation_speed(nearest_gap, scene.coordinate_spacing);
    let navigation_reference_distance = scene
        .coordinate_spacing
        .max(scene.billboard_world_size)
        .max(1.0);
    let initial_view_distance =
        initial_camera_distance(scene.coordinate_spacing, scene.billboard_world_size);
    let axis_labels = projection.axis_labels.clone().map(axis_label);

    cloud.rebuild_from_layout(commands, meshes, materials, point_cloud_layout);
    *gizmo_labels = AxisGizmoLabels {
        dimension_labels: axis_labels.clone(),
    };
    if reset_billboards {
        *view_bounds = ViewBounds::default();
        loading_state.clear_staged_uploads(images, materials);
        loading_state.reset_with_pending(
            pending_index,
            billboard_controls.texture_budget_bytes(),
            scene.image_concurrency,
            billboard_controls.max_texture_side,
        );
    } else {
        loading_state.update_points(image_points.clone());
        drop_in_background(pending_index);
        for (_, billboard) in scene_queries.billboards.iter() {
            cloud.set_point_visible(billboard.image_id, false);
        }
    }
    drop_in_background(std::mem::replace(navigation_targets, prepared_targets));
    // Manual selection state points at billboards of the outgoing projection;
    // start the new scene with a clean slate and a matching id index.
    commands.insert_resource(point_index);
    if reset_billboards {
        commands.insert_resource(SelectionState::default());
        commands.insert_resource(SelectionHighlights::default());
    }
    if position_player {
        navigation_settings.base_speed = camera_speed;
        navigation_settings.retarget_reference_distance(navigation_reference_distance);
        navigation_settings.status.effective_speed = camera_speed;
    }
    *billboard_stats = BillboardStats {
        entity_count: image_points.len(),
        pending: image_points.len(),
        ..default()
    };

    scene.bounds = bounds;
    drop_in_background(std::mem::replace(&mut scene.image_points, image_points));
    scene.camera_speed = camera_speed;
    scene.navigation_reference_distance = navigation_reference_distance;
    scene.initial_view_distance = initial_view_distance;
    scene.axis_labels = axis_labels;
    commands.insert_resource(billboard_axis_labels(&projection.axis_labels));
    drop_in_background(std::mem::replace(&mut scene.projection, projection));
    controls.apply_panel(panel, scene.projection.points.len(), scene.projection.total);
    // A `--control` naming something this server does not publish would
    // otherwise be silently ignored, leaving a scripted run looking correct
    // while using a different value than it asked for.
    let unknown = controls.take_unknown_initial_controls();
    if !unknown.is_empty() {
        controls.set_request_error(format!("Unknown --control: {}", unknown.join(", ")));
        eprintln!(
            "This API does not offer these controls: {}. Open the View pill to see what it does offer.",
            unknown.join(", ")
        );
    }

    if let (true, Some(point), Ok((mut transform, mut camera))) = (
        position_player,
        scene.projection.points.first(),
        scene_queries.camera.get_single_mut(),
    ) {
        place_camera_at_initial_view(
            &mut transform,
            &mut camera,
            api_position_to_viewer(point.position),
            scene.initial_view_distance,
        );
    }
    if let Ok(mut camera_projection) = scene_queries.camera_projection.get_single_mut() {
        set_scene_camera_far_plane(&mut camera_projection, scene.bounds.max_extent);
    }
}

/// Turns a control submission from the panel into a projection request.
///
/// The server owns control identity, so unlike the dimension-index remapping
/// this replaced, a submission needs no translation between snapshots: the
/// values go out exactly as the panel holds them.
pub(crate) fn apply_control_submit_requests(
    mut controls: ResMut<ControlPanelState>,
    scene: Res<ExplorerScene>,
    start_menu: Res<StartMenuState>,
    mut task: ResMut<CatalogLoadTask>,
) {
    // Keep only the newest submission and at most one request in flight, so
    // holding down a button cannot queue a backlog of reprojections.
    if let Some(submit) = controls.take_pending_submit() {
        task.pending_submit = Some(submit);
    }
    if task
        .receiver
        .get_mut()
        .expect("catalog load task lock")
        .is_some()
    {
        return;
    }
    let Some(submit) = task.pending_submit.take() else {
        return;
    };

    let request = scene
        .projection_request()
        .with_roots(start_menu.roots.clone())
        .with_controls(submit.values, submit.activated);
    let client = scene.client.clone();
    let (sender, receiver) = sync_channel(1);
    std::thread::spawn(move || match client.projection(&request) {
        Ok(snapshot) => {
            prepare_loaded_catalog(
                &sender,
                &mut HashMap::new(),
                LoadedCatalog {
                    panel: snapshot.panel,
                    projection: snapshot.projection,
                    complete: true,
                    roots: None,
                },
            );
        }
        Err(error) => {
            let _ = sender.send(Err(format!("Projection reload failed: {error}")));
        }
    });
    *task.receiver.get_mut().expect("catalog load task lock") = Some(receiver);
    task.first_snapshot = true;
    task.catalog = false;
    task.dragged_positions.clear();
}

/// Opens a native folder picker, preferring `pwsh` (PowerShell 7 / .NET
/// Core), where `FolderBrowserDialog` is the modern Vista-style dialog with
/// an editable path bar — so a path can be typed or pasted directly. The
/// legacy `powershell` fallback pre-selects `initial_directory` instead.
/// `initial_directory` re-opens the dialog where the user last picked.
pub(crate) fn pick_folder_dialog(initial_directory: Option<&str>) -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        let script = r#"
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.ShowNewFolderButton = $false
try {
    $dialog.Description = 'Select generation media folder'
    $dialog.UseDescriptionForTitle = $true
} catch {}
if ($env:GENERATION_VIEWER_PICKER_START) {
    try { $dialog.InitialDirectory = $env:GENERATION_VIEWER_PICKER_START }
    catch { $dialog.SelectedPath = $env:GENERATION_VIEWER_PICKER_START }
}
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Out.Write($dialog.SelectedPath)
}
"#;
        for (shell, extra_args) in [("pwsh", [].as_slice()), ("powershell", ["-STA"].as_slice())] {
            let mut command = std::process::Command::new(shell);
            command.arg("-NoProfile");
            command.args(extra_args);
            command.args(["-Command", script]);
            if let Some(initial) = initial_directory {
                command.env("GENERATION_VIEWER_PICKER_START", initial);
            }
            let Ok(output) = command.output() else {
                continue;
            };
            if !output.status.success() {
                continue;
            }
            let selected = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            return if selected.is_empty() {
                None
            } else {
                Some(selected)
            };
        }
        None
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = initial_directory;
        None
    }
}

/// While streaming, `total` counts only the points discovered so far, so it
/// must not read as the catalog's size. `shown` falls below `total` only when
/// the projection limit caps the page.
fn catalog_load_status(shown: usize, total: usize, complete: bool) -> String {
    let progress = if complete {
        format!("Loaded {total} points")
    } else {
        format!("Streaming: {total} points found so far")
    };
    if shown < total {
        format!("{progress}; showing {shown} (--limit).")
    } else {
        format!("{progress}.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_status_marks_streamed_totals_as_partial_and_reports_the_limit() {
        assert_eq!(
            catalog_load_status(1024, 1024, false),
            "Streaming: 1024 points found so far."
        );
        assert_eq!(catalog_load_status(85, 85, true), "Loaded 85 points.");
        assert_eq!(
            catalog_load_status(10_000, 52_000, true),
            "Loaded 52000 points; showing 10000 (--limit)."
        );
    }
    use crate::tests::test_scene;
    use crate::{FlyCamera, CAMERA_VELOCITY_RESPONSE_PER_SECOND};
    use std::sync::mpsc::sync_channel;

    fn test_snapshot(points: &[(usize, [f32; 3])], complete: bool) -> LoadedCatalog {
        let mut projection = test_scene().projection;
        projection.points = points
            .iter()
            .map(|&(image_id, position)| generation_api::ProjectionPoint {
                image_id,
                position,
                path: format!("{image_id}.png"),
                width: Some(64),
                height: Some(64),
                media_type: "image".to_owned(),
                duration_seconds: None,
                coordinate_labels: [None, None, None],
            })
            .collect();
        projection.total = points.len();
        LoadedCatalog {
            panel: ControlPanel::default(),
            projection,
            complete,
            roots: None,
        }
    }

    #[test]
    fn streamed_snapshots_position_player_once_and_keep_camera_during_updates() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(test_scene());
        app.insert_resource(ControlPanelState::default());
        app.init_resource::<StartMenuState>();
        app.init_resource::<PauseMenuState>();
        app.insert_resource(BillboardControls::new(8, 64));
        app.init_resource::<BillboardStats>();
        app.insert_resource(NavigationSettings::new(2.0, 6.0));
        app.insert_resource(NavigationTargets::from_positions([]));
        app.insert_resource(ImageLoadingState::new(Vec::new(), 8, 1, 64));
        app.init_resource::<PointCloud>();
        app.init_resource::<ViewBounds>();
        app.init_resource::<Assets<Mesh>>();
        app.init_resource::<Assets<StandardMaterial>>();
        app.init_resource::<Assets<Image>>();
        let camera = app
            .world_mut()
            .spawn((
                Transform::default(),
                FlyCamera {
                    yaw: 0.0,
                    pitch: 0.0,
                    sensitivity: 0.0025,
                    velocity: Vec3::ZERO,
                    velocity_response: CAMERA_VELOCITY_RESPONSE_PER_SECOND,
                },
                Projection::default(),
            ))
            .id();
        let (sender, receiver) = sync_channel(1);
        app.insert_resource(CatalogLoadTask {
            receiver: Mutex::new(Some(receiver)),
            first_snapshot: true,
            catalog: true,
            pending_submit: None,
            dragged_positions: HashMap::new(),
        });
        app.init_resource::<InitialPlayerPlacement>();
        app.init_resource::<AxisGizmoLabels>();
        app.add_systems(Update, poll_catalog_load_task);
        // Snapshots go through the same stabilising preparation the load
        // thread applies, sharing one pin map across the stream.
        let mut stable_positions = HashMap::new();
        assert!(prepare_loaded_catalog(
            &sender,
            &mut stable_positions,
            test_snapshot(&[(0, [100.0, 20.0, 30.0])], false),
        ));
        app.update();
        let initial = *app.world().get::<Transform>(camera).unwrap();
        assert_eq!(initial.translation, Vec3::new(115.0, 26.75, 45.0));
        assert_eq!(
            app.world()
                .resource::<ExplorerScene>()
                .projection
                .points
                .len(),
            1
        );
        assert!(!app.world().resource::<PauseMenuState>().paused);
        let moved = Vec3::new(101.0, 25.0, 55.0);
        app.world_mut()
            .get_mut::<Transform>(camera)
            .unwrap()
            .translation = moved;
        assert!(prepare_loaded_catalog(
            &sender,
            &mut stable_positions,
            test_snapshot(&[(0, [200.0, 20.0, 30.0]), (1, [300.0, 20.0, 30.0])], true,),
        ));
        app.update();
        assert_eq!(
            app.world().get::<Transform>(camera).unwrap().translation,
            moved
        );
        assert_eq!(
            app.world().resource::<ExplorerScene>().image_points[0]
                .position
                .x,
            100.0
        );
        assert_eq!(app.world().resource::<PointCloud>().point_count(), 2);
        assert!(!app.world().resource::<StartMenuState>().loading);
    }

    #[test]
    fn streaming_identity_check_allows_coordinate_changes_but_rejects_id_reuse() {
        let previous = test_snapshot(&[(0, [0.0; 3])], false).projection;
        let mut next = test_snapshot(&[(0, [100.0; 3]), (1, [200.0; 3])], true).projection;
        assert!(projection_preserves_billboards(&previous, &next));
        next.points[0].path = "different.png".into();
        assert!(!projection_preserves_billboards(&previous, &next));
    }
}
