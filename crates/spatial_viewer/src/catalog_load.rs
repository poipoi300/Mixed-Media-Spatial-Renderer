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
//!
//! The API places points in cube units and names their groups; the viewer
//! lays them out under its own folder state ([`lay_out_points`]). Opening or
//! closing a folder re-derives the current projection on a background
//! thread ([`CatalogLoadTask::start_relayout`]) and applies it like a
//! snapshot, so textures survive.

use std::collections::HashMap;
use std::sync::mpsc::{channel, sync_channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use spatial_api::{CatalogStreamEvent, ControlPanel, ProjectionPage};
use spatial_geometry::{bounds_of, estimate_smallest_axis_gap, Bounds3};
use spatial_viewer_ui::{
    Animation, AnimationSettings, BillboardControls, BillboardStats, ControlPanelState,
    NavigationSettings, NavigationTargets, PauseMenuState, PendingSubmit, StartMenuState,
    ViewSettings,
};

use crate::arrangement_store::{ArrangementStore, SavedCamera};
use crate::axis_gizmo::AxisGizmoLabels;
use crate::catalog_session::save_last_catalog_roots;
use crate::folders::{
    lay_out_points, BillboardGrowth, BillboardMotion, FolderLayoutInput, FolderScene,
    FolderViewState, LaidOutProjection, ManualArrangement, RetiringBillboard,
};
use crate::image_loading::{
    BillboardPoint, ImageLoadingState, MediaBillboard, PreparedPendingIndex,
};
use crate::manual_spacing::{ImagePointIndex, SelectionHighlights, SelectionState};
use crate::point_cloud::{point_color, PointCloud, PointCloudLayout, PointCloudPoint, ViewBounds};
use crate::{
    axis_label, billboard_axis_labels, drop_in_background, initial_camera_distance,
    navigation_speed, place_camera_at_initial_view, place_camera_at_pose, point_cloud_point_size,
    set_scene_camera_far_plane, BillboardEntitiesQuery, ExplorerScene, FlyCamera,
    FlyCameraProjectionQuery, FlyCameraTransformQuery,
};

/// Result of a background catalog load kicked off from the start menu.
pub(crate) struct LoadedCatalog {
    /// `None` for a relayout, which changes positions and nothing else.
    pub panel: Option<ControlPanel>,
    pub projection: ProjectionPage,
    pub complete: bool,
    pub roots: Option<Vec<String>>,
}

/// Everything a scene needs from a projection that can be derived without
/// touching ECS or assets. Built on the catalog load thread so applying a
/// snapshot on the main thread costs O(chunks), not O(points).
pub(crate) struct PreparedProjection {
    pub panel: Option<ControlPanel>,
    pub projection: ProjectionPage,
    pub complete: bool,
    pub roots: Option<Vec<String>>,
    /// The folder state `image_points` were laid out under; a snapshot
    /// prepared under an outdated one is re-derived.
    folders: FolderScene,
    /// Likewise the [`ManualArrangement`] revision.
    arrangement_revision: u64,
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
    pub(crate) fn new(loaded: LoadedCatalog, input: &FolderLayoutInput) -> Self {
        let projection = loaded.projection;
        let LaidOutProjection {
            points: image_points,
            folders,
        } = lay_out_points(&projection, input);
        let positions = || image_points.iter().map(|point| point.position);
        let bounds = bounds_of(positions());
        let nearest_gap = estimate_smallest_axis_gap(positions());
        let navigation_targets = NavigationTargets::from_positions(positions());
        let point_index = ImagePointIndex::new(&image_points);
        let point_cloud_layout = PointCloudLayout::new(
            point_cloud_points(&image_points),
            point_cloud_point_size(&bounds),
        );
        let pending_index = PreparedPendingIndex::new(image_points.clone());
        Self {
            panel: loaded.panel,
            projection,
            complete: loaded.complete,
            roots: loaded.roots,
            folders,
            arrangement_revision: input.arrangement.revision,
            bounds,
            image_points,
            nearest_gap,
            navigation_targets,
            point_index,
            point_cloud_layout,
            pending_index,
        }
    }

    /// Re-derives the snapshot under `input`. O(points), so callers reserve
    /// it for the rare case of a manual drag or a folder change while a
    /// snapshot was in flight.
    fn relaid_out(self, input: &FolderLayoutInput) -> Self {
        Self::new(
            LoadedCatalog {
                panel: self.panel,
                projection: self.projection,
                complete: self.complete,
                roots: self.roots,
            },
            input,
        )
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
    /// What the user arranged by hand since this load began, laid out over
    /// every later snapshot of it and every folder change.
    arrangement: Arc<Mutex<ManualArrangement>>,
}

impl CatalogLoadTask {
    pub(crate) fn arrangement(&self) -> MutexGuard<'_, ManualArrangement> {
        self.arrangement.lock().expect("manual arrangement lock")
    }

    /// No load or relayout is in flight.
    pub(crate) fn is_idle(&mut self) -> bool {
        self.receiver
            .get_mut()
            .expect("catalog load task lock")
            .is_none()
    }

    /// Re-derives `projection` under `input` on a background thread. It
    /// arrives through [`poll_catalog_load_task`] as a snapshot that changes
    /// which images show and where, and nothing else. Only call while idle.
    pub(crate) fn start_relayout(&mut self, projection: ProjectionPage, input: FolderLayoutInput) {
        let relayout = LoadedCatalog {
            panel: None,
            projection,
            complete: true,
            roots: None,
        };
        let (sender, receiver) = sync_channel(1);
        std::thread::spawn(move || {
            let _ = sender.send(Ok(PreparedProjection::new(relayout, &input)));
        });
        *self.receiver.get_mut().expect("catalog load task lock") = Some(receiver);
        self.first_snapshot = false;
        self.catalog = false;
    }
}

/// Runs on the catalog load thread: pins every ungrouped image to its first
/// streamed coordinate, then derives the scene data the main thread would
/// otherwise compute per snapshot. Returns `false` when the viewer dropped
/// the load.
pub(crate) fn prepare_loaded_catalog(
    sender: &std::sync::mpsc::SyncSender<Result<PreparedProjection, String>>,
    stable_positions: &mut HashMap<usize, [f32; 3]>,
    mut loaded: LoadedCatalog,
    input: &FolderLayoutInput,
) -> bool {
    stabilize_streamed_projection_positions(&mut loaded.projection, stable_positions);
    sender
        .send(Ok(PreparedProjection::new(loaded, input)))
        .is_ok()
}

/// Lifetime guard for automatic player placement. The camera may be placed
/// when the first non-empty coordinate set arrives, but catalog streaming,
/// reloads, and projection changes never own its position afterward.
#[derive(Resource, Default)]
pub(crate) struct InitialPlayerPlacement {
    pub positioned: bool,
    /// Where the camera was left in the view being loaded, placed instead of
    /// the initial view.
    pub remembered: Option<SavedCamera>,
}

impl InitialPlayerPlacement {
    /// The camera stands where the current view put it: no load is still
    /// to place it.
    pub(crate) fn settled(&self) -> bool {
        self.positioned && self.remembered.is_none()
    }
}

/// Where a load puts the camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PlayerPlacement {
    /// Looking at the first image, as a catalog first opens.
    Initial,
    /// Where it was left in this view.
    Remembered(SavedCamera),
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

/// Billboards with what a relayout moves them by: where and how large they
/// are drawn, and any motion they are already in.
type BillboardMotionQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static MediaBillboard,
        &'static Transform,
        &'static BillboardGrowth,
        Option<&'static mut BillboardMotion>,
    ),
    Without<FlyCamera>,
>;

#[derive(SystemParam)]
pub(crate) struct SceneReloadQueries<'w, 's> {
    billboards: BillboardEntitiesQuery<'w, 's>,
    /// Retained billboards spring to where a relayout moved them.
    billboard_motion: BillboardMotionQuery<'w, 's>,
    /// Whether a toggled folder's images leave in a stagger.
    animations: Res<'w, AnimationSettings>,
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

pub(crate) fn point_cloud_points(image_points: &[BillboardPoint]) -> Vec<PointCloudPoint> {
    image_points
        .iter()
        .map(|point| PointCloudPoint {
            image_id: point.image_id,
            position: point.position,
            color: point_color(point.image_id),
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_start_menu_requests(
    mut start_menu: ResMut<StartMenuState>,
    mut pause_menu: ResMut<PauseMenuState>,
    scene: Res<ExplorerScene>,
    controls: Res<ControlPanelState>,
    mut folder_view: ResMut<FolderViewState>,
    mut folder_pick_task: ResMut<FolderPickTask>,
    mut catalog_load_task: ResMut<CatalogLoadTask>,
    mut arrangements: ResMut<ArrangementStore>,
    last_folder: Res<LastPickedFolder>,
    view_settings: Res<ViewSettings>,
    mut initial_player_placement: ResMut<InitialPlayerPlacement>,
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
    if !catalog_load_task.is_idle() {
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
    // A catalog opens with every folder closed, arranged as the user left
    // this view.
    folder_view.reset();
    arrangements.switch_view(
        ArrangementStore::view_key(&start_menu.roots, controls.values()),
        &mut catalog_load_task.arrangement(),
        &scene.projection,
        scene.folders.origin,
    );
    // Back where the camera was left in this view, once its layout arrives.
    let remembered = arrangements
        .current_camera()
        .filter(|_| view_settings.remember_camera);
    if remembered.is_some() {
        initial_player_placement.positioned = false;
    }
    initial_player_placement.remembered = remembered;
    let input = folder_view.layout_input(
        scene.billboard_world_size,
        &FolderScene::default(),
        &catalog_load_task.arrangement(),
    );
    let arrangement = catalog_load_task.arrangement.clone();
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
                    panel: Some(panel),
                    projection: *projection,
                    complete,
                    roots: Some(roots),
                },
                &input.with_arrangement(&arrangement.lock().expect("manual arrangement lock")),
            ),
            CatalogStreamEvent::Error { message } => sender.send(Err(message)).is_ok(),
            CatalogStreamEvent::Heartbeat => true,
        });
        if let Err(error) = result {
            let _ = sender.send(Err(error));
        }
    });
    *catalog_load_task
        .receiver
        .get_mut()
        .expect("catalog load task lock") = Some(receiver);
    catalog_load_task.first_snapshot = true;
    catalog_load_task.catalog = true;
    catalog_load_task.pending_submit = None;
    start_menu.loading = true;
    start_menu.set_status("Loading catalog...");
}

/// Puts the catalog back as it first opened when "Reset layout" is pressed:
/// nothing arranged by hand in this view (one undo step brings it back),
/// every folder closed and laid out from the layout's own origin, nothing
/// selected, and the camera placed again once the new layout arrives.
pub(crate) fn reset_catalog_layout(
    mut start_menu: ResMut<StartMenuState>,
    task: Res<CatalogLoadTask>,
    mut folder_view: ResMut<FolderViewState>,
    mut selection: ResMut<SelectionState>,
    mut initial_player_placement: ResMut<InitialPlayerPlacement>,
) {
    if !start_menu.take_reset_layout_request() {
        return;
    }
    task.arrangement().clear();
    folder_view.reset_to_first_open();
    selection.clear();
    initial_player_placement.positioned = false;
    initial_player_placement.remembered = None;
    start_menu.set_status(
        "Layout reset as the catalog first opened. Undo brings back what you arranged.",
    );
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
    mut folder_view: ResMut<FolderViewState>,
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
            // An image may have been moved, or a folder made, opened or
            // closed, while the snapshot was being laid out; only then is it
            // laid out again, here.
            if loaded.arrangement_revision != catalog_load_task.arrangement().revision()
                || loaded.folders.revision != folder_view.revision()
            {
                let input = folder_view.layout_input(
                    scene.billboard_world_size,
                    &scene.folders,
                    &catalog_load_task.arrangement(),
                );
                loaded = loaded.relaid_out(&input);
            }
            let first_snapshot = catalog_load_task.first_snapshot;
            let placement = &mut menu.initial_player_placement;
            let player_placement = (!placement.positioned && !loaded.projection.points.is_empty())
                .then(|| {
                    placement.positioned = true;
                    placement
                        .remembered
                        .take()
                        .map_or(PlayerPlacement::Initial, PlayerPlacement::Remembered)
                });
            catalog_load_task.first_snapshot = false;
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
                player_placement,
            );
            menu.start_menu.set_status(catalog_load_status(
                scene.projection.points.len(),
                scene.projection.total,
                complete,
            ));
            if complete {
                if let Some(roots) = roots {
                    if std::env::var_os("SPATIAL_VIEWER_PERF_CONFIG").is_none() {
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

/// A partial catalog does not yet know its final extents, so a freshly
/// recomputed projection may move records that were already shown. Give every
/// ungrouped image its first streamed coordinate for the lifetime of this
/// load; newly discovered images still appear immediately at their own first
/// coordinate without shifting the existing scene around the player.
///
/// Grouped images are laid out by the viewer from their group's slot, which
/// a newly discovered value can shift: pinning them would stack two groups
/// in one slot. They move instead, springing to their new places.
pub(crate) fn stabilize_streamed_projection_positions(
    projection: &mut ProjectionPage,
    stable_positions: &mut HashMap<usize, [f32; 3]>,
) {
    for point in &mut projection.points {
        if point.group.is_some() {
            continue;
        }
        point.position = *stable_positions
            .entry(point.image_id)
            .or_insert(point.position);
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
    player_placement: Option<PlayerPlacement>,
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
        folders,
        bounds,
        image_points,
        nearest_gap,
        navigation_targets: prepared_targets,
        point_index,
        point_cloud_layout,
        pending_index,
        ..
    } = prepared;
    let cell = scene.billboard_world_size;
    let nearest_gap = nearest_gap.unwrap_or(cell);
    let camera_speed = navigation_speed(nearest_gap, cell);
    let navigation_reference_distance = cell.max(scene.billboard_world_size).max(1.0);
    let initial_view_distance = initial_camera_distance(cell, scene.billboard_world_size);
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
        drop_in_background(pending_index);
        // `projection_preserves_billboards` already vetted every identity.
        let retired = loading_state
            .update_points(image_points.clone())
            .expect("a preserving snapshot keeps every billboard's identity");
        for billboard in retired {
            // Back into the folder that now hides it, from the size it is
            // drawn at, grown or part way through a move.
            let target = folders
                .folder_of(billboard.image_id)
                .map_or(billboard.position, |folder| folder.center);
            let drawn_scale = scene_queries
                .billboard_motion
                .get(billboard.entity)
                .map_or(billboard.scale, |(_, _, transform, _, _)| transform.scale.x);
            commands
                .entity(billboard.entity)
                .remove::<(MediaBillboard, BillboardMotion, BillboardGrowth)>()
                .insert(RetiringBillboard::new(
                    target,
                    drawn_scale,
                    billboard.surface_assets,
                ));
        }
        for (_, billboard) in scene_queries.billboards.iter() {
            cloud.set_point_visible(billboard.image_id, false);
        }
        follow_loaded_points(commands, loading_state, &folders, cell, scene_queries);
    }
    drop_in_background(std::mem::replace(navigation_targets, prepared_targets));
    // Manual selection state points at billboards of the outgoing projection;
    // start the new scene with a clean slate and a matching id index.
    commands.insert_resource(point_index);
    if reset_billboards {
        commands.insert_resource(SelectionState::default());
        commands.insert_resource(SelectionHighlights::default());
    }
    if let Some(placement) = player_placement {
        match placement {
            PlayerPlacement::Initial => navigation_settings.base_speed = camera_speed,
            PlayerPlacement::Remembered(camera) => {
                navigation_settings.set_base_speed(camera.base_speed);
            }
        }
        navigation_settings.retarget_reference_distance(navigation_reference_distance);
        navigation_settings.status.effective_speed = navigation_settings.base_speed;
    }
    *billboard_stats = BillboardStats {
        entity_count: image_points.len(),
        pending: image_points.len(),
        ..default()
    };

    scene.folders = folders;
    scene.folder_generation += 1;
    scene.bounds = bounds;
    drop_in_background(std::mem::replace(&mut scene.image_points, image_points));
    scene.camera_speed = camera_speed;
    scene.navigation_reference_distance = navigation_reference_distance;
    scene.initial_view_distance = initial_view_distance;
    scene.axis_labels = axis_labels;
    commands.insert_resource(billboard_axis_labels(&projection.axis_labels));
    drop_in_background(std::mem::replace(&mut scene.projection, projection));
    if let Some(panel) = panel {
        controls.apply_panel(panel, scene.projection.points.len(), scene.projection.total);
        // A `--control` naming something this server does not publish would
        // otherwise be silently ignored, leaving a scripted run looking
        // correct while using a different value than it asked for.
        let unknown = controls.take_unknown_initial_controls();
        if !unknown.is_empty() {
            controls.set_request_error(format!("Unknown --control: {}", unknown.join(", ")));
            eprintln!(
                "This API does not offer these controls: {}. Open the View pill to see what it does offer.",
                unknown.join(", ")
            );
        }
    }

    if let (Some(placement), Ok((mut transform, mut camera))) =
        (player_placement, scene_queries.camera.get_single_mut())
    {
        match placement {
            PlayerPlacement::Initial => {
                if let Some(point) = scene.image_points.first() {
                    place_camera_at_initial_view(
                        &mut transform,
                        &mut camera,
                        point.position,
                        scene.initial_view_distance,
                    );
                }
            }
            PlayerPlacement::Remembered(saved) => place_camera_at_pose(
                &mut transform,
                &mut camera,
                scene.folders.origin + Vec3::from_array(saved.position),
                saved.yaw,
                saved.pitch,
            ),
        }
    }
    if let Ok(mut camera_projection) = scene_queries.camera_projection.get_single_mut() {
        set_scene_camera_far_plane(&mut camera_projection, scene.bounds.max_extent);
    }
}

/// Starts every retained billboard springing to where and how large its
/// loaded record now says it belongs. The images of a folder just toggled
/// leave in a stagger, nearest to its center first, while folders animate
/// opening and closing.
fn follow_loaded_points(
    commands: &mut Commands,
    loading_state: &ImageLoadingState,
    folders: &FolderScene,
    cube_size: f32,
    scene_queries: &mut SceneReloadQueries,
) {
    let staggers = scene_queries.animations.plays(Animation::FolderSlide);
    for (entity, billboard, transform, growth, motion) in &mut scene_queries.billboard_motion {
        let Some(point) = loading_state.loaded_point(billboard.image_id) else {
            continue;
        };
        let layout_scale = transform.scale.x / growth.factor();
        if point.position == transform.translation && point.scale == layout_scale {
            continue;
        }
        let delay = if staggers {
            folders.stagger_seconds(billboard.image_id, point.position, cube_size)
        } else {
            0.0
        };
        BillboardMotion::retarget(commands, entity, motion, delay);
    }
}

/// Turns a control submission from the panel into a projection request.
///
/// The server owns control identity, so unlike the dimension-index remapping
/// this replaced, a submission needs no translation between snapshots: the
/// values go out exactly as the panel holds them.
pub(crate) fn apply_control_submit_requests(
    mut controls: ResMut<ControlPanelState>,
    mut folder_view: ResMut<FolderViewState>,
    scene: Res<ExplorerScene>,
    start_menu: Res<StartMenuState>,
    mut task: ResMut<CatalogLoadTask>,
    mut arrangements: ResMut<ArrangementStore>,
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

    // A new projection starts from the origin, arranged as the user left
    // this view; folders that keep their key under it stay open.
    arrangements.switch_view(
        ArrangementStore::view_key(&start_menu.roots, &submit.values),
        &mut task.arrangement(),
        &scene.projection,
        scene.folders.origin,
    );
    let request = scene
        .projection_request()
        .with_roots(start_menu.roots.clone())
        .with_controls(submit.values, submit.activated);
    let client = scene.client.clone();
    let input = folder_view.layout_input(
        scene.billboard_world_size,
        &FolderScene::default(),
        &task.arrangement(),
    );
    let (sender, receiver) = sync_channel(1);
    std::thread::spawn(move || match client.projection(&request) {
        Ok(snapshot) => {
            prepare_loaded_catalog(
                &sender,
                &mut HashMap::new(),
                LoadedCatalog {
                    panel: Some(snapshot.panel),
                    projection: snapshot.projection,
                    complete: true,
                    roots: None,
                },
                &input,
            );
        }
        Err(error) => {
            let _ = sender.send(Err(format!("Projection reload failed: {error}")));
        }
    });
    *task.receiver.get_mut().expect("catalog load task lock") = Some(receiver);
    task.first_snapshot = true;
    task.catalog = false;
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
    $dialog.Description = 'Select media folder'
    $dialog.UseDescriptionForTitle = $true
} catch {}
if ($env:SPATIAL_VIEWER_PICKER_START) {
    try { $dialog.InitialDirectory = $env:SPATIAL_VIEWER_PICKER_START }
    catch { $dialog.SelectedPath = $env:SPATIAL_VIEWER_PICKER_START }
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
                command.env("SPATIAL_VIEWER_PICKER_START", initial);
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
    use crate::folders::{apply_folder_changes, FolderKey, ManualPlacement, FOLDER_PREVIEW_SCALE};
    use crate::tests::test_scene;
    use crate::{FlyCamera, CAMERA_VELOCITY_RESPONSE_PER_SECOND};
    use spatial_api::PointGroup;
    use spatial_viewer_ui::BillboardFacingSettings;
    use std::sync::mpsc::sync_channel;

    fn test_snapshot(points: &[(usize, [f32; 3])], complete: bool) -> LoadedCatalog {
        let mut projection = test_scene().projection;
        projection.points = points
            .iter()
            .map(|&(image_id, position)| spatial_api::ProjectionPoint {
                image_id,
                position,
                group: None,
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
            panel: Some(ControlPanel::default()),
            projection,
            complete,
            roots: None,
        }
    }

    /// Everything `poll_catalog_load_task` and `apply_folder_changes` read,
    /// with a camera and a load task fed by the returned sender.
    fn catalog_app() -> (
        App,
        Entity,
        std::sync::mpsc::SyncSender<Result<PreparedProjection, String>>,
    ) {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(test_scene());
        app.insert_resource(ControlPanelState::default());
        app.init_resource::<FolderViewState>();
        app.init_resource::<StartMenuState>();
        app.init_resource::<PauseMenuState>();
        app.init_resource::<AnimationSettings>();
        app.insert_resource(BillboardControls::new(8, 64));
        app.init_resource::<BillboardFacingSettings>();
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
            arrangement: Arc::default(),
        });
        app.init_resource::<InitialPlayerPlacement>();
        app.init_resource::<AxisGizmoLabels>();
        app.add_systems(
            Update,
            (poll_catalog_load_task, apply_folder_changes).chain(),
        );
        (app, camera, sender)
    }

    #[test]
    fn a_remembered_camera_is_placed_instead_of_the_initial_view() {
        let (mut app, camera, sender) = catalog_app();
        let saved = SavedCamera {
            position: [3.0, -2.0, 40.0],
            yaw: 0.7,
            pitch: -0.3,
            base_speed: 9.0,
        };
        app.world_mut()
            .resource_mut::<InitialPlayerPlacement>()
            .remembered = Some(saved);
        let input = FolderViewState::default().layout_input(
            test_scene().billboard_world_size,
            &FolderScene::default(),
            &ManualArrangement::default(),
        );
        assert!(prepare_loaded_catalog(
            &sender,
            &mut HashMap::new(),
            test_snapshot(&[(0, [100.0, 20.0, 30.0])], true),
            &input,
        ));
        app.update();
        let transform = app.world().get::<Transform>(camera).unwrap();
        let origin = app.world().resource::<ExplorerScene>().folders.origin;
        assert!(
            transform
                .translation
                .distance(origin + Vec3::new(3.0, -2.0, 40.0))
                < 1e-4
        );
        let fly_camera = app.world().get::<FlyCamera>(camera).unwrap();
        assert_eq!((fly_camera.yaw, fly_camera.pitch), (0.7, -0.3));
        assert_eq!(app.world().resource::<NavigationSettings>().base_speed, 9.0);
        let placement = app.world().resource::<InitialPlayerPlacement>();
        assert!(placement.settled());
    }

    #[test]
    fn streamed_snapshots_position_player_once_and_keep_camera_during_updates() {
        let (mut app, camera, sender) = catalog_app();
        let cube_size = test_scene().billboard_world_size;
        let input = FolderViewState::default().layout_input(
            cube_size,
            &FolderScene::default(),
            &ManualArrangement::default(),
        );
        // Snapshots go through the same stabilising preparation the load
        // thread applies, sharing one pin map across the stream.
        let mut stable_positions = HashMap::new();
        assert!(prepare_loaded_catalog(
            &sender,
            &mut stable_positions,
            test_snapshot(&[(0, [100.0, 20.0, 30.0])], false),
            &input,
        ));
        app.update();
        let first_world = Vec3::new(100.0, 20.0, 30.0) * cube_size;
        let distance = initial_camera_distance(cube_size, cube_size);
        let initial = *app.world().get::<Transform>(camera).unwrap();
        assert!(
            initial
                .translation
                .distance(first_world + Vec3::new(distance, distance * 0.45, distance))
                < 1e-3
        );
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
            &input,
        ));
        app.update();
        assert_eq!(
            app.world().get::<Transform>(camera).unwrap().translation,
            moved
        );
        assert_eq!(
            app.world().resource::<ExplorerScene>().image_points[0].position,
            first_world
        );
        assert_eq!(app.world().resource::<PointCloud>().point_count(), 2);
        assert!(!app.world().resource::<StartMenuState>().loading);
    }

    /// Runs frames until the scene has caught up with the folder state.
    fn settle_folders(app: &mut App) {
        for _ in 0..500 {
            app.update();
            let revision = app.world().resource::<FolderViewState>().revision();
            if app.world().resource::<ExplorerScene>().folders.revision == revision {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("the folder relayout never arrived");
    }

    #[test]
    fn opening_a_folder_reveals_its_images_and_keeps_drags() {
        let (mut app, _camera, sender) = catalog_app();
        let mut snapshot = test_snapshot(
            &[
                (0, [0.0; 3]),
                (1, [0.0; 3]),
                (2, [0.0; 3]),
                (3, [0.0; 3]),
                (4, [0.0; 3]),
            ],
            true,
        );
        for (point, offset) in snapshot
            .projection
            .points
            .iter_mut()
            .zip([-2.0, -1.0, 0.0, 1.0, 2.0])
        {
            point.group = Some(PointGroup {
                key: "folder".to_owned(),
                index: [0, 0, 0],
                offset: [offset, 0.0, 0.0],
            });
        }
        let cube_size = test_scene().billboard_world_size;
        let input = FolderViewState::default().layout_input(
            cube_size,
            &FolderScene::default(),
            &ManualArrangement::default(),
        );
        assert!(prepare_loaded_catalog(
            &sender,
            &mut HashMap::new(),
            snapshot,
            &input
        ));
        app.update();

        // Closed: four previews, small, and the fifth image hidden.
        let scene = app.world().resource::<ExplorerScene>();
        assert_eq!(scene.image_points.len(), 4);
        assert!(scene.image_points.iter().all(|point| point.scale < 1.0));
        let folder = scene
            .folders
            .get(&FolderKey::group("folder"))
            .unwrap()
            .clone();
        let hidden = (0..5)
            .find(|image_id| !folder.previews.contains(image_id))
            .unwrap();

        let dragged = Vec3::new(0.0, 40.0, 0.0);
        app.world()
            .resource::<CatalogLoadTask>()
            .arrangement()
            .place(folder.previews[0], ManualPlacement::Loose(dragged));
        app.world_mut()
            .resource_mut::<FolderViewState>()
            .toggle(&folder.key, 0.0);
        settle_folders(&mut app);

        let scene = app.world().resource::<ExplorerScene>();
        let point = |image_id: usize| {
            scene
                .image_points
                .iter()
                .find(|point| point.image_id == image_id)
                .unwrap()
        };
        assert_eq!(scene.image_points.len(), 5);
        assert_eq!(point(hidden).scale, 1.0);
        // Where its folder, now four images wide, packs it.
        let opened = scene.folders.get(&folder.key).unwrap();
        let (_, offset) = opened
            .members
            .iter()
            .find(|(image_id, _)| *image_id == hidden)
            .unwrap();
        assert_eq!(point(hidden).position, opened.home + *offset * cube_size);
        // The toggled folder stays where it was.
        assert_eq!(
            scene
                .folders
                .get(&FolderKey::group("folder"))
                .unwrap()
                .center,
            folder.center
        );
        assert_eq!(point(folder.previews[0]).position, dragged);
        assert_eq!(app.world().resource::<PointCloud>().point_count(), 5);

        // Closed again, the folder holds the four images left in it, all of
        // them previews now; the dragged one stays where it was put.
        app.world_mut()
            .resource_mut::<FolderViewState>()
            .toggle(&folder.key, 1.0);
        settle_folders(&mut app);
        let scene = app.world().resource::<ExplorerScene>();
        let closed = scene.folders.get(&folder.key).unwrap();
        assert_eq!(closed.members.len(), 4);
        assert!(closed.previews.contains(&hidden));
        assert!(!closed.previews.contains(&folder.previews[0]));
        let point = |image_id: usize| {
            scene
                .image_points
                .iter()
                .find(|point| point.image_id == image_id)
                .unwrap()
        };
        assert_eq!(point(hidden).scale, FOLDER_PREVIEW_SCALE);
        assert_eq!(point(folder.previews[0]).position, dragged);
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
