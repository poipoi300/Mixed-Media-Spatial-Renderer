use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use ab_glyph::FontArc;
use bevy::{math::primitives::Rectangle, prelude::*};
use generation_geometry::normalized_or;
use generation_viewer_ui::{
    BillboardControls, BillboardFacingAxis, BillboardFacingSettings, BillboardStats,
    NavigationSettings, PauseMenuState,
};

use crate::{
    axis_gizmo::{render_label_text, LabelTextLine},
    background_work::{spawn_background, CancelToken, CompletedWork},
    decode_budget::DecodeBudget,
    load_sampling::{load_value, load_value_sigma, WeightedReservoir},
    media_decode::{
        billboard_surface_bytes, billboard_surface_to_bevy_image, decode_first_video_frame,
        encode_billboard_surface, fit_image_to_square, open_image_file, BillboardTextureEncoding,
        BillboardTextureFormat, DecodedBillboardImage, DecodedImage, SurfaceEncodeTally,
    },
    point_cloud::{PointCloud, ViewBounds},
    FlyCamera,
};

/// Coordinate label block height contributed per text line, as a fraction of
/// the billboard's world size (e.g. 3 axis lines get 3x this block height).
const COORDINATE_LABEL_LINE_HEIGHT_FACTOR: f32 = 0.085;
/// Gap between the bottom of a billboard and the top of its coordinate
/// label, as a fraction of the billboard's world size.
const COORDINATE_LABEL_GAP_FACTOR: f32 = 0.10;
/// Small nudge toward the camera so the label never z-fights the billboard
/// plane it hangs below.
const COORDINATE_LABEL_FORWARD_OFFSET_FACTOR: f32 = 0.01;
/// Cap on how wide (in billboard-size multiples) the coordinate text is
/// allowed to grow before it wraps onto another line.
const COORDINATE_LABEL_MAX_WIDTH_FACTOR: f32 = 2.5;
const COORDINATE_LABEL_FONT_SIZE: f32 = 46.0;
const COORDINATE_LABEL_COLOR: [u8; 4] = [235, 240, 248, 255];

const BILLBOARD_ORIENTATION_MAX_DISTANCE_FACTOR: f32 = 96.0;
const BILLBOARD_NEAR_VISIBLE_DOT: f32 = -0.15;
const BILLBOARD_GRID_LOCAL_RADIUS: i32 = 6;
const BILLBOARD_PENDING_FALLBACK_CELL_LIMIT: usize = 64;
const BILLBOARD_COORDINATE_LABEL_SPAWNS_PER_FRAME: usize = 2;
const BILLBOARD_RECEIVE_MAX_UPLOADS_PER_FRAME: usize = 8;
const BILLBOARD_RECEIVE_MAX_SPAWNS_PER_FRAME: usize = 8;
const BILLBOARD_RECEIVE_FRAME_BUDGET: Duration = Duration::from_millis(6);
/// Soft ceiling on texture bytes uploaded per frame.
///
/// A billboard's texture is one contiguous surface, so this cannot split a
/// large image across frames — `upload_fits_frame_budget` always admits the
/// first upload of a frame however large it is, and this then stops anything
/// else joining it. The median catalog image is ~8.9 MB of BC7 and the
/// largest ~90 MB.
pub const BILLBOARD_GPU_UPLOAD_BYTES_PER_FRAME: usize = 4 * 1024 * 1024;
/// Largest square side a video frame is decoded to.
///
/// Stills are decoded at source resolution, but a playing video replaces its
/// frame at its playback rate, so spending a large upload on one would waste
/// the budget on pixels discarded before they are seen.
pub const BILLBOARD_VIDEO_TEXTURE_SIDE: u32 = 1024;
const BILLBOARD_CAMERA_MOVE_THRESHOLD_SQUARED: f32 = 0.0025;
const BILLBOARD_CAMERA_ROTATION_DOT_THRESHOLD: f32 = 0.999_98;
/// How far ahead of the camera quality and priority decisions look, in
/// seconds of travel at the current velocity. Sized to cover a decode plus
/// its GPU upload so an image the camera is flying toward is decoded once,
/// at the resolution it needs on arrival, instead of being previewed low and
/// immediately re-decoded.
const BILLBOARD_QUALITY_LOOKAHEAD_SECONDS: f32 = 0.75;
/// Cap on the lookahead displacement, in scene reference distances. Without
/// it a fast cruise would place the projected camera far across the catalog
/// and ask for full quality nearly everywhere.
const BILLBOARD_QUALITY_LOOKAHEAD_MAX_STEPS: f32 = 12.0;
/// Pending cells inspected per frame while offering candidates to the
/// sampler. The walk starts at the camera and spirals outward, so this bounds
/// the per-frame cost without biasing *which* points win: the sample is
/// weighted, and cells near the camera are visited first and every frame.
const BILLBOARD_PENDING_CELLS_PER_FRAME: usize = 192;

#[derive(Component)]
pub struct GenerationBillboard {
    pub image_id: usize,
    pub path: Arc<str>,
    pub is_video: bool,
    pub duration_seconds: Option<f32>,
    /// Source pixel dimensions, when the catalog knows them.
    pub source_size: Option<UVec2>,
    pub texture_side: u32,
    pub surface_assets: BillboardSurfaceAssets,
}

impl GenerationBillboard {
    /// Half extents, in billboard-local units, of the picture inside the
    /// square billboard of `world_size`. Sources are letterboxed into the
    /// square with transparent padding, so a landscape picture leaves empty
    /// bands above and below it; an unknown size fills the square.
    ///
    /// TODO: verify, and fix if true, that a video with rotation metadata
    /// (phone footage stored landscape, tagged to play portrait) gets a
    /// sideways rect here: the catalog's `video_metadata`
    /// (`generation_explorer/catalog.py`) is suspected to report the stored
    /// size, while ffmpeg auto-rotates the decoded frames. Picking and strip
    /// placement would then not match the picture. Fix by having the catalog
    /// report the displayed size, or by deriving the rect from the decode.
    pub fn content_half_extents(&self, world_size: f32) -> Vec2 {
        let half_size = world_size * 0.5;
        match self.source_size {
            Some(size) if size.x > 0 && size.y > 0 => {
                let longest_side = size.max_element() as f32;
                size.as_vec2() / longest_side * half_size
            }
            _ => Vec2::splat(half_size),
        }
    }

    pub fn remove_assets(
        &self,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) {
        self.surface_assets.remove(images, materials);
    }
}

/// The one quad, texture and material that draw a billboard's image.
///
/// A billboard used to own a grid of these. Collapsing to one surface is
/// what removes the seam (see [`BillboardSurface`]); it also cuts the mean
/// billboard from 14.4 draw calls to 1.
#[derive(Clone, Default)]
pub struct BillboardSurfaceAssets {
    /// `None` before the surface has been attached, and for the degenerate
    /// decode that produced no image pixels at all.
    pub entity: Option<Entity>,
    pub texture_handle: Option<Handle<Image>>,
    pub material_handle: Option<Handle<StandardMaterial>>,
}

impl BillboardSurfaceAssets {
    fn remove(&self, images: &mut Assets<Image>, materials: &mut Assets<StandardMaterial>) {
        if let Some(material) = &self.material_handle {
            materials.remove(material.id());
        }
        if let Some(texture) = &self.texture_handle {
            images.remove(texture.id());
        }
    }
}

/// Strings are shared (`Arc<str>`) because the whole point list is cloned
/// into the loading state on every catalog snapshot; with 50k points, owned
/// strings made that clone alone cost tens of milliseconds on the main
/// thread.
#[derive(Clone)]
pub struct BillboardPoint {
    pub image_id: usize,
    pub path: Arc<str>,
    pub position: Vec3,
    pub is_video: bool,
    pub duration_seconds: Option<f32>,
    /// Source pixel dimensions, when the catalog knows them. Used to charge
    /// a texture against the VRAM budget *before* it is decoded, so a burst
    /// of scheduled loads cannot overshoot the budget on arrival.
    pub source_size: Option<UVec2>,
    /// Human-readable dimension value backing each axis (e.g. an acquisition
    /// date or seed), aligned to the currently selected X/Y/Z dimensions;
    /// `None` where an axis has no dimension assigned.
    pub coordinate_labels: [Option<Arc<str>>; 3],
}

/// Bytes charged for a point whose source dimensions the catalog does not
/// report.
///
/// The mean of a sampled catalog (~3800 px on a side) rather than zero: a point
/// charged nothing could be admitted without limit and could never be
/// evicted for size, so a handful of unknown-size images could exhaust VRAM
/// while the budget still read as satisfied.
const BILLBOARD_UNKNOWN_SOURCE_SIDE: u32 = 3800;

impl BillboardPoint {
    /// VRAM this point's texture will occupy once resident.
    ///
    /// Counts only the image rectangle, which is all the decode encodes: a
    /// portrait image on a square canvas leaves a wide letterbox margin that
    /// is never encoded or uploaded.
    pub fn texture_bytes(&self, format: BillboardTextureFormat) -> usize {
        let source = self
            .source_size
            .unwrap_or(UVec2::splat(BILLBOARD_UNKNOWN_SOURCE_SIDE));
        billboard_surface_bytes(source, format)
    }
}

#[derive(Resource)]
pub struct BillboardMesh(pub Handle<Mesh>);

/// World-space size of a billboard's square mesh, needed to scale and
/// position its coordinate label relative to it.
#[derive(Resource)]
pub struct BillboardWorldSize(pub f32);

/// Human-readable name of the dimension currently assigned to each of the
/// X/Y/Z projection axes (e.g. "Seed", "Acquisition Date"); `None` where an
/// axis has no dimension assigned. Updated whenever the projection reloads
/// with a different axis selection.
#[derive(Resource, Default)]
pub struct BillboardAxisLabels(pub [Option<String>; 3]);

/// Font used to render each billboard's coordinate label; shared with the
/// axis gizmo and video control labels. `None` when the bundled font failed
/// to load, in which case no coordinate labels are rendered.
#[derive(Resource)]
pub struct BillboardLabelFont(pub Option<FontArc>);

/// Marker for a billboard's coordinate-text child entity, toggled by the
/// "Show coordinates" checkbox.
#[derive(Component)]
pub struct BillboardCoordinateLabel;

#[derive(Component)]
pub(crate) struct BillboardCoordinateLabelReady;

#[derive(Resource)]
pub struct ImageLoadingState {
    pending: SpatialPendingIndex,
    in_flight: HashMap<usize, InFlightLoad>,
    loaded: HashMap<usize, LoadedBillboardRecord>,
    failed: HashSet<usize>,
    completed: CompletedWork<ImageLoadResult>,
    /// VRAM ceiling for resident billboard textures, from the Billboards
    /// panel slider. Replaces a count cap: images differ ~100x in size, so a
    /// count cannot bound the memory that actually runs out.
    texture_budget_bytes: usize,
    max_in_flight: usize,
    /// Advanced once per frame so the weighted draw is deterministic per run.
    sample_seed: u64,
    /// Format decodes encode their surface in, decided once from the
    /// negotiated render device.
    texture_format: BillboardTextureFormat,
    last_reported_completed: usize,
    last_upload_count: usize,
    last_upload_bytes: usize,
    /// Running total of how alpha classification encoded every image
    /// decoded this run.
    encode_tally: SurfaceEncodeTally,
}

struct InFlightLoad {
    point: BillboardPoint,
    texture_limit: u32,
    /// Cancels this decode when its billboard is evicted before it finishes.
    cancel: CancelToken,
}

struct LoadedBillboardRecord {
    point: BillboardPoint,
    entity: Entity,
    texture_side: u32,
    texture_limit: u32,
    surface_assets: BillboardSurfaceAssets,
}

struct ImageLoadResult {
    point: BillboardPoint,
    texture_limit: u32,
    /// How alpha classification encoded this image, captured at decode time.
    ///
    /// Held separately from `result` because the surface is moved out of it
    /// on upload, so the decoded image is not intact by the time the
    /// receiving system finishes with it.
    encode_tally: SurfaceEncodeTally,
    /// Set once the tally has been folded into the run total, so a result
    /// revisited across frames is counted exactly once.
    encode_tally_counted: bool,
    result: Result<DecodedBillboardImage, String>,
    /// Set when the worker stopped early because the load was cancelled. The
    /// receiving system drops such results without recording a failure: the
    /// image did not fail to decode, it simply stopped being wanted.
    cancelled: bool,
    /// Set once the surface has been uploaded, which happens in a single
    /// step: unlike the tile grid this replaced, there is no partial state
    /// to carry across frames.
    uploaded_surface: Option<UploadedBillboardSurface>,
}

struct UploadedBillboardSurface {
    layout: BillboardSurfaceLayout,
    texture_handle: Handle<Image>,
    material_handle: Handle<StandardMaterial>,
}

/// Where a billboard's surface sits on its square, in texel coordinates.
#[derive(Clone, Copy)]
struct BillboardSurfaceLayout {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy)]
struct SchedulingView {
    position: Vec3,
    forward: Vec3,
    /// Where the camera is expected to be after
    /// [`BILLBOARD_QUALITY_LOOKAHEAD_SECONDS`] of travel at its current
    /// velocity, clamped to [`BILLBOARD_QUALITY_LOOKAHEAD_MAX_STEPS`] scene
    /// steps. Equal to `position` when the camera is stationary.
    lookahead_position: Vec3,
}

impl SchedulingView {
    /// Offset from the camera to `position`, measured from whichever of the
    /// current and projected camera positions is nearer to it. Every quality
    /// and priority decision uses this rather than the raw distance, so an
    /// image being approached is treated as already close: it is decoded at
    /// its arrival resolution in one pass instead of landing as a low preview
    /// that has to be thrown away a moment later.
    fn approach_offset(self, position: Vec3) -> Vec3 {
        let current = position - self.position;
        let projected = position - self.lookahead_position;
        if projected.length_squared() < current.length_squared() {
            projected
        } else {
            current
        }
    }
}

#[derive(Clone, Copy)]
pub struct BillboardFacingSnapshot {
    position: Vec3,
    rotation: Quat,
    axis: BillboardFacingAxis,
}

/// A pending point identified by where it sits in the index, so it can be
/// offered to the sampler without being removed and then put back.
#[derive(Clone, Copy)]
struct PendingCandidate {
    cell: SpatialCell,
    index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SpatialCell {
    x: i32,
    y: i32,
    z: i32,
}

/// Pending-load spatial index for a whole point set, built ahead of a scene
/// reset (on the catalog load thread) so the reset itself is O(1).
pub struct PreparedPendingIndex(SpatialPendingIndex);

impl PreparedPendingIndex {
    pub fn new(points: Vec<BillboardPoint>) -> Self {
        Self(SpatialPendingIndex::new(points))
    }
}

/// Points with no texture yet, bucketed by position so the sampler can offer
/// the ones near the camera without walking the whole catalog.
pub struct SpatialPendingIndex {
    cell_size: f32,
    cells: HashMap<SpatialCell, Vec<BillboardPoint>>,
    ids: HashSet<usize>,
    len: usize,
}

pub fn billboard_world_size(coordinate_spacing: f32, scale: f32) -> f32 {
    coordinate_spacing.max(1.0) * scale.clamp(0.1, 1.5)
}

pub fn create_billboard_mesh(meshes: &mut Assets<Mesh>, world_size: f32) -> BillboardMesh {
    BillboardMesh(meshes.add(Rectangle::new(world_size, world_size)))
}

impl ImageLoadingState {
    pub fn new(
        points: Vec<BillboardPoint>,
        texture_budget_bytes: usize,
        max_in_flight: usize,
        _max_texture_side: u32,
    ) -> Self {
        Self {
            pending: SpatialPendingIndex::new(points),
            in_flight: HashMap::new(),
            loaded: HashMap::new(),
            failed: HashSet::new(),
            completed: CompletedWork::default(),
            texture_budget_bytes,
            max_in_flight: max_in_flight.max(1),
            sample_seed: 0,
            texture_format: BillboardTextureFormat::default(),
            last_reported_completed: 0,
            last_upload_count: 0,
            last_upload_bytes: 0,
            encode_tally: SurfaceEncodeTally::default(),
        }
    }

    /// Replaces every loading structure for a new point set, with the
    /// pending index already built off-thread. The outgoing state (tens of thousands of points and their hash maps for
    /// a large catalog) is dropped on a background thread so its
    /// deallocation does not land in this frame either.
    pub fn reset_with_pending(
        &mut self,
        pending: PreparedPendingIndex,
        texture_budget_bytes: usize,
        max_in_flight: usize,
        _max_texture_side: u32,
    ) {
        let outgoing_pending = std::mem::replace(&mut self.pending, pending.0);
        let outgoing_in_flight = std::mem::take(&mut self.in_flight);
        let outgoing_loaded = std::mem::take(&mut self.loaded);
        // Every decode for the outgoing catalog is now pointless work.
        for load in outgoing_in_flight.values() {
            load.cancel.cancel();
        }
        self.failed.clear();
        self.completed = CompletedWork::default();
        self.texture_budget_bytes = texture_budget_bytes;
        self.max_in_flight = max_in_flight.max(1);
        self.last_reported_completed = 0;
        self.last_upload_count = 0;
        self.last_upload_bytes = 0;
        spawn_background(move || {
            drop(outgoing_pending);
            drop(outgoing_in_flight);
            drop(outgoing_loaded);
        });
    }

    pub fn clear_staged_uploads(
        &mut self,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) {
        while let Some(completed) = self.completed.pop_front() {
            remove_uploaded_billboard_surface(&completed.uploaded_surface, images, materials);
        }
    }

    /// Reconciles a complete streamed catalog snapshot without resetting
    /// matching loaded or decoding media. Returns `false` when a retained
    /// billboard's ID/path identity is missing or changed; callers must then
    /// tear down the scene and call [`Self::reset`] instead.
    pub fn update_points(&mut self, points: Vec<BillboardPoint>) -> bool {
        let mut points_by_id = HashMap::with_capacity(points.len());
        for point in points {
            if points_by_id.insert(point.image_id, point).is_some() {
                return false;
            }
        }

        for record in self.loaded.values() {
            let Some(point) = points_by_id.get(&record.point.image_id) else {
                return false;
            };
            if point.path != record.point.path {
                return false;
            }
        }
        for load in self.in_flight.values() {
            let Some(point) = points_by_id.get(&load.point.image_id) else {
                return false;
            };
            if point.path != load.point.path {
                return false;
            }
        }

        for record in self.loaded.values_mut() {
            let mut updated = points_by_id
                .get(&record.point.image_id)
                .expect("loaded billboard identity was validated above")
                .clone();
            updated.position = record.point.position;
            record.point = updated;
        }
        for load in self.in_flight.values_mut() {
            let mut updated = points_by_id
                .get(&load.point.image_id)
                .expect("in-flight billboard identity was validated above")
                .clone();
            updated.position = load.point.position;
            load.point = updated;
        }
        for image_id in self.loaded.keys().chain(self.in_flight.keys()) {
            points_by_id.remove(image_id);
        }

        self.failed
            .retain(|image_id| points_by_id.contains_key(image_id));
        self.pending = SpatialPendingIndex::new(
            points_by_id
                .into_values()
                .filter(|point| !self.failed.contains(&point.image_id))
                .collect(),
        );
        true
    }

    fn return_to_pending(&mut self, point: BillboardPoint) {
        if self.loaded.contains_key(&point.image_id)
            || self.in_flight.contains_key(&point.image_id)
            || self.failed.contains(&point.image_id)
        {
            return;
        }
        self.pending.insert(point);
    }

    /// VRAM the resident textures occupy, plus what the in-flight decodes
    /// will occupy when they land.
    ///
    /// In-flight loads are charged before they arrive because a decode takes
    /// hundreds of milliseconds: a budget that only counted what had already
    /// landed would admit a whole frame's worth of scheduling and overshoot
    /// once they all completed.
    fn loaded_bytes_used(&self) -> usize {
        let resident: usize = self
            .loaded
            .values()
            .map(|record| record.point.texture_bytes(self.texture_format))
            .sum();
        let incoming: usize = self
            .in_flight
            .iter()
            .filter(|(image_id, _)| !self.loaded.contains_key(image_id))
            .map(|(_, load)| load.point.texture_bytes(self.texture_format))
            .sum();
        resident + incoming
    }

    /// Poster decodes currently running, counted against the decode budget.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Decode slots the image scheduler could put to use right now, used as
    /// the demand signal for the decode budget.
    pub fn pending_demand(&self) -> usize {
        self.pending.len().min(self.max_in_flight)
    }

    /// Seed for this frame's weighted draw. Advancing per frame keeps
    /// selection reproducible for a given run while ensuring successive
    /// frames do not repeat the same draw from an unchanged candidate set.
    fn next_sample_seed(&mut self) -> u64 {
        self.sample_seed = self.sample_seed.wrapping_add(1);
        self.sample_seed
    }

    /// Records a manually moved position for a loaded billboard so eviction
    /// scoring and a later return to the pending queue keep the new location.
    pub fn update_loaded_position(&mut self, image_id: usize, position: Vec3) {
        if let Some(record) = self.loaded.get_mut(&image_id) {
            record.point.position = position;
        }
    }

    /// Whether admitting `point` would exceed the VRAM budget.
    fn cache_is_full_for(&self, point: &BillboardPoint) -> bool {
        self.loaded_bytes_used() + point.texture_bytes(self.texture_format)
            > self.texture_budget_bytes
    }

    fn cache_over_limit(&self) -> bool {
        // Never evict the last resident billboard: a budget smaller than one
        // image would otherwise evict everything and re-decode forever,
        // showing an empty view instead of one image.
        self.loaded.len() > 1 && self.loaded_bytes_used() > self.texture_budget_bytes
    }

    /// Cancels in-flight decodes for points the camera can no longer reach,
    /// returning how many were stopped.
    ///
    /// A point outside [`ViewBounds`] is hidden behind the opaque boundary
    /// wall: finishing its decode could not show the user anything, and the
    /// worker it holds is one a visible point could have used. This matters
    /// more than it used to — a decode now also pays for block compression,
    /// so an abandoned one wastes several times what it did before.
    ///
    /// The test is exactly the one [`candidate_value`] scores zero for, so a
    /// load is only cancelled on the same grounds that would have stopped it
    /// from ever being scheduled.
    fn cancel_unreachable_loads(&mut self, view: SchedulingView, bounds: &ViewBounds) -> usize {
        let mut cancelled = 0;
        for load in self.in_flight.values() {
            if load.cancel.is_cancelled() {
                continue;
            }
            if bounds.is_beyond(view.position, load.point.position) {
                load.cancel.cancel();
                cancelled += 1;
            }
        }
        cancelled
    }

    fn farthest_loaded_billboard(&self, view: SchedulingView) -> Option<usize> {
        self.loaded
            .values()
            .max_by(|left, right| {
                loaded_eviction_score(left, view).total_cmp(&loaded_eviction_score(right, view))
            })
            .map(|record| record.point.image_id)
    }
}

/// One unit of decode work the scheduler may start this frame: a point with
/// no texture yet, referenced by its slot in the pending index so only the
/// drawn ones are removed from it.
///
/// Every image is decoded once at source resolution, so there is no second
/// kind of work (a quality refresh) to compete with these.
struct LoadCandidate {
    candidate: PendingCandidate,
}

#[allow(clippy::too_many_arguments)]
pub fn schedule_image_loads(
    mut commands: Commands,
    mut state: ResMut<ImageLoadingState>,
    controls: Res<BillboardControls>,
    pause_menu: Res<PauseMenuState>,
    navigation: Res<NavigationSettings>,
    view_bounds: Res<ViewBounds>,
    texture_encoding: Res<BillboardTextureEncoding>,
    mut budget: ResMut<DecodeBudget>,
    camera_query: Query<(&Transform, &FlyCamera)>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cloud: ResMut<PointCloud>,
) {
    if pause_menu.paused {
        return;
    }

    state.texture_format = texture_encoding.format;
    state.texture_budget_bytes = controls.texture_budget_bytes();

    let view = scheduling_view(camera_query.get_single().ok(), &navigation);

    // Free workers held by decodes the camera has put out of reach before
    // this frame's scheduling looks for capacity.
    state.cancel_unreachable_loads(view, &view_bounds);

    evict_over_cache_limit(
        &mut commands,
        &mut state,
        &mut images,
        &mut materials,
        &mut cloud,
        view,
    );

    // Fill every free worker from ONE weighted sample over all candidates.
    //
    // Every image is decoded once, at source resolution, so the only work
    // that competes here is points that have never been decoded. They are
    // scored by how much of the view they occupy, so what gets loaded falls
    // out of the score rather than out of a ration.
    let capacity = state
        .max_in_flight
        .saturating_sub(state.in_flight.len())
        .min(budget.remaining());
    if capacity == 0 {
        return;
    }
    let sigma = load_value_sigma(navigation.reference_distance);
    let mut reservoir = WeightedReservoir::new(capacity, state.next_sample_seed());

    // Points with no texture yet, offered from the cells nearest the camera
    // outward. Taking them out of the index to offer them would strand the
    // ones not drawn, so candidates are referenced by cell and index and only
    // the drawn ones are removed.
    // A full cache does not stop first loads: one replacement at a time may
    // still displace a worse-placed resident, which is how camera movement
    // improves cache locality. The gate above is on workers, not on slots.
    state
        .pending
        .offer_candidates(&mut reservoir, view, &view_bounds, sigma);

    let first_loads = reservoir.take();
    for point in state.pending.take_candidates(first_loads) {
        if budget.take_allowance(1) == 0 {
            state.pending.insert(point);
            continue;
        }
        // Once the cache is full a new load must displace a resident, and
        // only one such replacement may be in flight: the resident stays
        // visible until its replacement is ready, and receive_image_loads
        // performs the swap.
        if state.cache_is_full_for(&point) {
            let replaces_a_worse_resident = state
                .farthest_loaded_billboard(view)
                .and_then(|image_id| state.loaded.get(&image_id))
                .is_some_and(|resident| replacement_improves_cache(&point, resident, view));
            let replacement_in_flight = state
                .in_flight
                .keys()
                .any(|image_id| !state.loaded.contains_key(image_id));
            if !replaces_a_worse_resident || replacement_in_flight {
                state.pending.insert(point);
                continue;
            }
        }
        let texture_limit = billboard_texture_limit(&controls, *texture_encoding, point.is_video);
        start_image_load(&mut state, point, texture_limit);
    }
}

pub fn publish_image_loading_stats(
    state: Res<ImageLoadingState>,
    navigation: Res<NavigationSettings>,
    camera_query: Query<(&Transform, &FlyCamera)>,
    mut stats: ResMut<BillboardStats>,
) {
    stats.loaded = state.loaded.len();
    stats.failed = state.failed.len();
    stats.in_flight = state.in_flight.len();
    stats.pending = state.pending.len();
    // Every pending point is now schedulable: candidates are weighted rather
    // than deferred, so nothing sits in the index ineligible for selection.
    stats.schedulable_pending = state.pending.len();
    stats.uploaded_textures = state.last_upload_count;
    stats.uploaded_bytes = state.last_upload_bytes;
    stats.upload_queue = state.completed.len();
    stats.texture_bytes_used = state.loaded_bytes_used();
    stats.texture_budget_bytes = state.texture_budget_bytes;

    let view = scheduling_view(camera_query.get_single().ok(), &navigation);
    let (visible, textured) = visible_coverage_counts(&state, view, &navigation);
    stats.visible_billboards = visible;
    stats.visible_textured = textured;

    let tally = &state.encode_tally;
    stats.encode_opaque_surfaces = tally.opaque_surfaces;
    stats.encode_mixed_surfaces = tally.mixed_surfaces;
    stats.encode_opaque_texels = tally.opaque_texels;
    stats.encode_mixed_texels = tally.mixed_texels;
    stats.encode_opaque_nanos = tally.opaque_encode_nanos;
    stats.encode_mixed_nanos = tally.mixed_encode_nanos;
}

/// Points near enough and in front of the camera to be worth showing, and
/// how many of those actually have a texture.
///
/// With one resolution per image there is no longer a "wrong quality" to
/// measure — every loaded billboard is at the only resolution there is. What
/// still varies, and what the VRAM budget directly controls, is *coverage*:
/// whether the thing you are looking at is there at all. That is the number
/// worth watching.
///
/// "In view" reuses the orientation test rather than inventing a second
/// notion of visibility, so this counts exactly the billboards the renderer
/// is turning toward the camera.
fn visible_coverage_counts(
    state: &ImageLoadingState,
    view: SchedulingView,
    navigation: &NavigationSettings,
) -> (usize, usize) {
    let max_distance = billboard_orientation_max_distance(navigation);
    let max_distance_squared = max_distance * max_distance;
    let in_view = |position: Vec3| {
        billboard_should_update(view.position, view.forward, position, max_distance_squared)
    };
    let textured = state
        .loaded
        .values()
        .filter(|record| in_view(record.point.position))
        .count();
    // Pending points count toward what *should* be visible: a hole in the
    // view is exactly a point that is in range and has no texture yet.
    let missing = state.pending.positions().filter(|p| in_view(*p)).count();
    (textured + missing, textured)
}

#[allow(clippy::too_many_arguments)]
pub fn receive_image_loads(
    mut commands: Commands,
    mut state: ResMut<ImageLoadingState>,
    pause_menu: Res<PauseMenuState>,
    billboard_mesh: Res<BillboardMesh>,
    billboard_world_size: Res<BillboardWorldSize>,
    navigation: Res<NavigationSettings>,
    camera_query: Query<(&Transform, &FlyCamera)>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cloud: ResMut<PointCloud>,
    mut billboard_query: Query<&mut GenerationBillboard>,
) {
    state.last_upload_count = 0;
    state.last_upload_bytes = 0;
    if pause_menu.paused {
        return;
    }

    let frame_start = Instant::now();
    let view = scheduling_view(camera_query.get_single().ok(), &navigation);
    let mut uploaded_textures = 0;
    let mut uploaded_bytes = 0usize;
    let mut spawned_billboards = 0;
    let mut processed_results = 0;

    while uploaded_textures < BILLBOARD_RECEIVE_MAX_UPLOADS_PER_FRAME
        && spawned_billboards < BILLBOARD_RECEIVE_MAX_SPAWNS_PER_FRAME
        && frame_start.elapsed() < BILLBOARD_RECEIVE_FRAME_BUDGET
    {
        let Some(mut completed) = state.completed.pop_front() else {
            break;
        };
        let completed_bytes = completed
            .result
            .as_ref()
            .ok()
            .and_then(|decoded| decoded.surface.as_ref())
            .map_or(0, |surface| surface.pixels.len());
        if !upload_fits_frame_budget(uploaded_textures, uploaded_bytes, completed_bytes) {
            state.completed.push_front(completed);
            break;
        }
        processed_results += 1;

        // Count the encode split once per decode, before any of the paths
        // below can drop or requeue this result. The work was performed by
        // the worker regardless of whether the surface ends up on screen.
        if !completed.encode_tally_counted {
            completed.encode_tally_counted = true;
            let tally = completed.encode_tally;
            state.encode_tally.add(&tally);
        }

        // A cancelled decode stopped early on purpose: drop it without
        // recording a failure, and let the point become schedulable again so
        // a later pass can decode it at whatever quality it needs then.
        if completed.cancelled {
            if let Some(in_flight) = state.in_flight.remove(&completed.point.image_id) {
                remove_uploaded_billboard_surface(
                    &completed.uploaded_surface,
                    &mut images,
                    &mut materials,
                );
                state.return_to_pending(in_flight.point);
            }
            continue;
        }

        let Some(in_flight) = state.in_flight.get(&completed.point.image_id) else {
            remove_uploaded_billboard_surface(
                &completed.uploaded_surface,
                &mut images,
                &mut materials,
            );
            continue;
        };
        if in_flight.texture_limit != completed.texture_limit {
            let in_flight = state
                .in_flight
                .remove(&completed.point.image_id)
                .expect("in-flight load was inspected above");
            remove_uploaded_billboard_surface(
                &completed.uploaded_surface,
                &mut images,
                &mut materials,
            );
            state.return_to_pending(in_flight.point);
            continue;
        }

        let point = in_flight.point.clone();
        let current_limit = state
            .loaded
            .get(&point.image_id)
            .map(|record| record.texture_limit);

        if let Ok(decoded) = &mut completed.result {
            if let Some(surface) = decoded.surface.take() {
                let layout = BillboardSurfaceLayout {
                    x: surface.x,
                    y: surface.y,
                    width: surface.width,
                    height: surface.height,
                };
                let texture_handle = images.add(billboard_surface_to_bevy_image(surface));
                let material_handle = materials.add(StandardMaterial {
                    base_color_texture: Some(texture_handle.clone()),
                    alpha_mode: BILLBOARD_SURFACE_ALPHA_MODE,
                    cull_mode: None,
                    unlit: true,
                    ..default()
                });
                completed.uploaded_surface = Some(UploadedBillboardSurface {
                    layout,
                    texture_handle,
                    material_handle,
                });
                uploaded_textures += 1;
                uploaded_bytes += completed_bytes;
            }
        }

        state.in_flight.remove(&point.image_id);
        match completed.result {
            Ok(decoded) => {
                if current_limit.is_none() && state.cache_is_full_for(&point) {
                    let replacement = state.farthest_loaded_billboard(view).filter(|image_id| {
                        state.loaded.get(image_id).is_some_and(|resident| {
                            replacement_improves_cache(&point, resident, view)
                        })
                    });
                    let Some(image_id) = replacement else {
                        remove_uploaded_billboard_surface(
                            &completed.uploaded_surface,
                            &mut images,
                            &mut materials,
                        );
                        state.return_to_pending(point);
                        continue;
                    };
                    evict_loaded_billboard(
                        &mut commands,
                        &mut state,
                        image_id,
                        &mut images,
                        &mut materials,
                        &mut cloud,
                        true,
                    );
                }
                let texture_side = decoded.side;

                if let Some(record) = state.loaded.get(&point.image_id) {
                    let entity = record.entity;
                    let old_assets = record.surface_assets.clone();
                    let surface_assets = attach_billboard_surface(
                        &mut commands,
                        entity,
                        &billboard_mesh.0,
                        texture_side,
                        billboard_world_size.0,
                        completed.uploaded_surface,
                    );
                    if let Some(old_entity) = old_assets.entity {
                        commands.entity(old_entity).despawn_recursive();
                    }
                    old_assets.remove(&mut images, &mut materials);
                    if let Ok(mut billboard) = billboard_query.get_mut(entity) {
                        billboard.texture_side = texture_side;
                        billboard.surface_assets = surface_assets.clone();
                    }
                    let record = state
                        .loaded
                        .get_mut(&point.image_id)
                        .expect("loaded billboard was inspected above");
                    record.point = point;
                    record.texture_side = texture_side;
                    record.texture_limit = completed.texture_limit;
                    record.surface_assets = surface_assets;
                    continue;
                }

                if state.cache_is_full_for(&point) {
                    remove_uploaded_billboard_surface(
                        &completed.uploaded_surface,
                        &mut images,
                        &mut materials,
                    );
                    state.return_to_pending(point);
                    continue;
                }

                let entity = commands
                    .spawn((
                        Transform::from_translation(point.position),
                        Visibility::Inherited,
                        Name::new(format!("generation image {}", point.image_id)),
                    ))
                    .id();
                let surface_assets = attach_billboard_surface(
                    &mut commands,
                    entity,
                    &billboard_mesh.0,
                    texture_side,
                    billboard_world_size.0,
                    completed.uploaded_surface,
                );
                commands.entity(entity).insert(GenerationBillboard {
                    image_id: point.image_id,
                    path: point.path.clone(),
                    is_video: point.is_video,
                    duration_seconds: point.duration_seconds,
                    source_size: point.source_size,
                    texture_side,
                    surface_assets: surface_assets.clone(),
                });
                spawned_billboards += 1;

                cloud.set_point_visible(point.image_id, false);
                state.loaded.insert(
                    point.image_id,
                    LoadedBillboardRecord {
                        point,
                        entity,
                        texture_side,
                        texture_limit: completed.texture_limit,
                        surface_assets,
                    },
                );
            }
            Err(error) => {
                remove_uploaded_billboard_surface(
                    &completed.uploaded_surface,
                    &mut images,
                    &mut materials,
                );
                state.failed.insert(point.image_id);
                eprintln!("Failed to load image {}: {error}", point.image_id);
            }
        }
    }

    state.last_upload_count = uploaded_textures;
    state.last_upload_bytes = uploaded_bytes;

    if processed_results == 0 {
        return;
    }

    let completed_count = state.loaded.len() + state.failed.len();
    let finished_loading_window = state.loaded_bytes_used() >= state.texture_budget_bytes
        || (state.pending.is_empty() && state.in_flight.is_empty() && state.completed.is_empty());
    if completed_count >= state.last_reported_completed + 50
        || (finished_loading_window && completed_count != state.last_reported_completed)
    {
        println!(
            "Loaded {} image textures ({} failed, {} in flight, {} pending, {} queued, {:.1}/{} GiB VRAM)",
            state.loaded.len(),
            state.failed.len(),
            state.in_flight.len(),
            state.pending.len(),
            state.completed.len(),
            state.loaded_bytes_used() as f64 / (1024.0 * 1024.0 * 1024.0),
            state.texture_budget_bytes / (1024 * 1024 * 1024),
        );
        state.last_reported_completed = completed_count;
    }
}

fn upload_fits_frame_budget(
    uploaded_count: usize,
    uploaded_bytes: usize,
    next_bytes: usize,
) -> bool {
    uploaded_count == 0
        || uploaded_bytes.saturating_add(next_bytes) <= BILLBOARD_GPU_UPLOAD_BYTES_PER_FRAME
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_visible_coordinate_labels(
    mut commands: Commands,
    controls: Res<BillboardControls>,
    state: Res<ImageLoadingState>,
    label_font: Res<BillboardLabelFont>,
    axis_labels: Res<BillboardAxisLabels>,
    billboard_world_size: Res<BillboardWorldSize>,
    ready: Query<(), With<BillboardCoordinateLabelReady>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !controls.show_coordinates {
        return;
    }
    let Some(font) = label_font.0.as_ref() else {
        return;
    };

    let candidates = state
        .loaded
        .values()
        .filter(|record| ready.get(record.entity).is_err())
        .take(BILLBOARD_COORDINATE_LABEL_SPAWNS_PER_FRAME)
        .map(|record| (record.entity, record.point.coordinate_labels.clone()))
        .collect::<Vec<_>>();
    for (entity, coordinate_labels) in candidates {
        commands
            .entity(entity)
            .insert(BillboardCoordinateLabelReady);
        let lines = coordinate_label_lines(&axis_labels.0, &coordinate_labels);
        spawn_coordinate_label(
            &mut commands,
            &mut meshes,
            &mut materials,
            &mut images,
            font,
            entity,
            &lines,
            billboard_world_size.0,
            true,
        );
    }
}

pub fn face_billboards_to_camera(
    settings: Res<BillboardFacingSettings>,
    navigation: Res<NavigationSettings>,
    pause_menu: Res<PauseMenuState>,
    mut stats: ResMut<BillboardStats>,
    camera_query: Query<&Transform, (With<FlyCamera>, Without<GenerationBillboard>)>,
    mut billboard_query: Query<&mut Transform, With<GenerationBillboard>>,
    mut last_snapshot: Local<Option<BillboardFacingSnapshot>>,
) {
    if pause_menu.paused {
        stats.orientation_updated = 0;
        stats.orientation_checked = 0;
        stats.orientation_skipped = billboard_query.iter().len();
        return;
    }
    let Ok(camera_transform) = camera_query.get_single() else {
        return;
    };

    let snapshot = BillboardFacingSnapshot {
        position: camera_transform.translation,
        rotation: camera_transform.rotation,
        axis: settings.axis,
    };
    if let Some(previous) = *last_snapshot {
        if !billboard_facing_snapshot_changed(previous, snapshot) {
            stats.orientation_updated = 0;
            stats.orientation_checked = 0;
            stats.orientation_skipped = billboard_query.iter().len();
            return;
        }
    }
    *last_snapshot = Some(snapshot);

    let camera_position = camera_transform.translation;
    let camera_forward = camera_transform.rotation.mul_vec3(Vec3::NEG_Z);
    let camera_up = camera_transform.rotation.mul_vec3(Vec3::Y);
    let viewport_normal = camera_transform.rotation.mul_vec3(Vec3::Z);
    let max_distance = billboard_orientation_max_distance(&navigation);
    let max_distance_squared = max_distance * max_distance;
    let mut updated = 0;
    let mut checked = 0;
    let mut skipped = 0;
    for mut transform in &mut billboard_query {
        checked += 1;
        if !billboard_should_update(
            camera_position,
            camera_forward,
            transform.translation,
            max_distance_squared,
        ) {
            skipped += 1;
            continue;
        }
        let to_camera = camera_transform.translation - transform.translation;
        if to_camera.length_squared() <= f32::EPSILON {
            skipped += 1;
            continue;
        }
        transform.rotation = billboard_rotation(
            to_camera.normalize(),
            viewport_normal,
            camera_up,
            settings.axis,
        );
        updated += 1;
    }
    stats.orientation_updated = updated;
    stats.orientation_checked = checked;
    stats.orientation_skipped = skipped;
}

fn billboard_should_update(
    camera_position: Vec3,
    camera_forward: Vec3,
    billboard_position: Vec3,
    max_distance_squared: f32,
) -> bool {
    let to_billboard = billboard_position - camera_position;
    if to_billboard.length_squared() > max_distance_squared {
        return false;
    }
    normalized_or(to_billboard, camera_forward).dot(camera_forward) >= BILLBOARD_NEAR_VISIBLE_DOT
}

fn billboard_orientation_max_distance(navigation: &NavigationSettings) -> f32 {
    navigation.reference_distance * BILLBOARD_ORIENTATION_MAX_DISTANCE_FACTOR
}

fn billboard_facing_snapshot_changed(
    previous: BillboardFacingSnapshot,
    current: BillboardFacingSnapshot,
) -> bool {
    if previous.axis != current.axis {
        return true;
    }
    if previous.position.distance_squared(current.position)
        > BILLBOARD_CAMERA_MOVE_THRESHOLD_SQUARED
    {
        return true;
    }
    previous.rotation.dot(current.rotation).abs() < BILLBOARD_CAMERA_ROTATION_DOT_THRESHOLD
}

/// Builds one "Axis label: value" line per axis that has both a dimension
/// assigned and a recorded value for this point (e.g. "Acquisition Date:
/// 2024-03-15"); axes with no dimension selected are omitted entirely.
fn coordinate_label_lines(
    axis_labels: &[Option<String>; 3],
    coordinate_labels: &[Option<Arc<str>>; 3],
) -> Vec<String> {
    axis_labels
        .iter()
        .zip(coordinate_labels.iter())
        .filter_map(|(axis_label, value)| {
            Some(format!("{}: {}", axis_label.as_deref()?, value.as_deref()?))
        })
        .collect()
}

/// Bakes each of `lines` into a text texture and spawns it as a child of the
/// billboard entity, hanging just below it. As a child, the label inherits
/// the billboard's camera-facing rotation for free instead of needing its
/// own per-frame orientation system. Spawns nothing if `lines` is empty (no
/// dimension is assigned to any axis).
#[allow(clippy::too_many_arguments)]
fn spawn_coordinate_label(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    font: &FontArc,
    billboard_entity: Entity,
    lines: &[String],
    billboard_world_size: f32,
    visible: bool,
) {
    if lines.is_empty() {
        return;
    }
    let text_lines: Vec<LabelTextLine> = lines
        .iter()
        .map(|line| LabelTextLine {
            text: line,
            font_size: COORDINATE_LABEL_FONT_SIZE,
            color: COORDINATE_LABEL_COLOR,
        })
        .collect();
    let rendered = render_label_text(
        font,
        &text_lines,
        billboard_world_size * COORDINATE_LABEL_LINE_HEIGHT_FACTOR * lines.len() as f32,
        billboard_world_size * COORDINATE_LABEL_MAX_WIDTH_FACTOR,
    );
    let texture = images.add(rendered.image);
    let mesh = meshes.add(Rectangle::new(rendered.world_size.x, rendered.world_size.y));
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(texture),
        // Texture is premultiplied (see `render_label_text`/`blend_label_pixel`).
        alpha_mode: AlphaMode::Premultiplied,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let offset = Vec3::new(
        0.0,
        -(billboard_world_size * 0.5)
            - billboard_world_size * COORDINATE_LABEL_GAP_FACTOR
            - rendered.world_size.y * 0.5,
        billboard_world_size * COORDINATE_LABEL_FORWARD_OFFSET_FACTOR,
    );
    commands.entity(billboard_entity).with_children(|parent| {
        parent.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_translation(offset),
            if visible {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            },
            BillboardCoordinateLabel,
            Name::new("billboard coordinates"),
        ));
    });
}

/// Shows or hides every billboard's coordinate label when the "Show
/// coordinates" checkbox changes; labels are already baked at spawn time, so
/// this only ever flips visibility, never re-renders text.
pub fn update_billboard_coordinate_label_visibility(
    controls: Res<BillboardControls>,
    mut labels: Query<&mut Visibility, With<BillboardCoordinateLabel>>,
) {
    if !controls.is_changed() {
        return;
    }
    let visibility = if controls.show_coordinates {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut label_visibility in &mut labels {
        *label_visibility = visibility;
    }
}

fn evict_over_cache_limit(
    commands: &mut Commands,
    state: &mut ImageLoadingState,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
    cloud: &mut PointCloud,
    view: SchedulingView,
) {
    while state.cache_over_limit() && !state.loaded.is_empty() {
        let Some(image_id) = state.farthest_loaded_billboard(view) else {
            break;
        };
        evict_loaded_billboard(commands, state, image_id, images, materials, cloud, true);
    }
}

fn evict_loaded_billboard(
    commands: &mut Commands,
    state: &mut ImageLoadingState,
    image_id: usize,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
    cloud: &mut PointCloud,
    return_to_pending: bool,
) {
    // Any refresh still decoding for this billboard is now for a texture
    // nothing will display.
    if let Some(load) = state.in_flight.remove(&image_id) {
        load.cancel.cancel();
    }
    let Some(record) = state.loaded.remove(&image_id) else {
        return;
    };
    commands.entity(record.entity).despawn_recursive();
    record.surface_assets.remove(images, materials);
    cloud.set_point_visible(record.point.image_id, true);
    if return_to_pending {
        state.return_to_pending(record.point);
    }
}

fn start_image_load(state: &mut ImageLoadingState, point: BillboardPoint, texture_limit: u32) {
    let cancel = CancelToken::new();
    state.in_flight.insert(
        point.image_id,
        InFlightLoad {
            point: point.clone(),
            texture_limit,
            cancel: cancel.clone(),
        },
    );

    let texture_format = state.texture_format;
    let completed = state.completed.clone();
    spawn_background(move || {
        let outcome = decode_billboard_image(&point, texture_limit, &cancel);
        let cancelled = cancel.is_cancelled();
        let result = match outcome {
            Ok(decoded) => Ok(encode_billboard_surface(&decoded, texture_format)),
            Err(error) => Err(error.to_string()),
        };
        let encode_tally = result
            .as_ref()
            .map(|decoded| decoded.encode_tally)
            .unwrap_or_default();
        completed.push(ImageLoadResult {
            point,
            texture_limit,
            encode_tally,
            encode_tally_counted: false,
            result,
            cancelled,
            uploaded_surface: None,
        });
    });
}

/// Runs one poster decode on a worker thread, polling `cancel` at each point
/// where the remaining work is expensive. A cancelled decode returns an error
/// the receiving system discards rather than recording as a failure.
fn decode_billboard_image(
    point: &BillboardPoint,
    texture_limit: u32,
    cancel: &CancelToken,
) -> anyhow::Result<DecodedImage> {
    if cancel.is_cancelled() {
        anyhow::bail!("cancelled");
    }
    if point.is_video {
        // ffmpeg runs as one uninterruptible child process, so the only
        // useful checks are before starting it and after it returns.
        let frame = decode_first_video_frame(Path::new(&*point.path), texture_limit)?;
        if cancel.is_cancelled() {
            anyhow::bail!("cancelled");
        }
        return Ok(frame);
    }

    let source = open_image_file(Path::new(&*point.path))?;
    // Fitting is the other half of the cost, and by now the camera has had
    // the whole file read to move on.
    if cancel.is_cancelled() {
        anyhow::bail!("cancelled");
    }
    fit_image_to_square(source, texture_limit)
}

fn remove_uploaded_billboard_surface(
    surface: &Option<UploadedBillboardSurface>,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
) {
    let Some(surface) = surface else {
        return;
    };
    materials.remove(surface.material_handle.id());
    images.remove(surface.texture_handle.id());
}

fn attach_billboard_surface(
    commands: &mut Commands,
    billboard_entity: Entity,
    billboard_mesh: &Handle<Mesh>,
    texture_side: u32,
    billboard_world_size: f32,
    surface: Option<UploadedBillboardSurface>,
) -> BillboardSurfaceAssets {
    let Some(surface) = surface else {
        return BillboardSurfaceAssets::default();
    };
    let transform = billboard_surface_transform(surface.layout, texture_side, billboard_world_size);
    let mut entity = None;
    commands.entity(billboard_entity).with_children(|parent| {
        entity = Some(
            parent
                .spawn((
                    Mesh3d(billboard_mesh.clone()),
                    MeshMaterial3d(surface.material_handle.clone()),
                    transform,
                    Name::new("billboard image surface"),
                ))
                .id(),
        );
    });
    BillboardSurfaceAssets {
        entity,
        texture_handle: Some(surface.texture_handle),
        material_handle: Some(surface.material_handle),
    }
}

/// Billboard surfaces are alpha-masked, not alpha-blended.
///
/// `AlphaMode::Blend` puts a mesh in the sorted `Transparent3d` phase, which
/// disables depth writes and orders draws by the view-space Z of the mesh
/// *origin* alone. Neither property survives this scene:
///
/// * Without depth writes, a nearer billboard cannot occlude a farther one,
///   so the phase order is the only thing deciding what is on top.
/// * Camera-facing quads intersect and interleave whenever their planes are
///   not parallel to each other, and a single per-mesh distance cannot order
///   meshes that overlap in depth. Two billboards a hair apart also swap
///   order the instant the camera moves across the midpoint between them.
///
/// The result is quads that flicker in front of each other and swap layering
/// as the camera moves — z-fighting in the phase sort rather than in the
/// depth buffer, which is why nudging positions never fixed it.
///
/// `Mask` discards below the cutoff and treats the rest as opaque, which
/// routes surfaces into the depth-writing `AlphaMask3d` phase. Depth then
/// resolves overlap per fragment, at any camera angle, independent of draw
/// order.
///
/// The surface is cropped to the image rectangle, so the letterbox padding
/// is not in the texture at all and the mask has nothing of its own to
/// discard. What it still does is honour a source image's own transparency:
/// the 6.8% of this catalog's PNGs that carry alpha. The cutoff is low so a
/// feathered edge stays visible rather than being cut away.
const BILLBOARD_SURFACE_ALPHA_MODE: AlphaMode = AlphaMode::Mask(0.05);

/// Places the surface's quad on the billboard's square, in texel
/// coordinates.
///
/// The surface covers only the image rectangle, not the whole square, so the
/// letterbox margin is simply not drawn. Coordinates stay relative to the
/// square so the placement is independent of how the image sits within it,
/// and every billboard keeps the same world footprint whatever its aspect.
fn billboard_surface_transform(
    surface: BillboardSurfaceLayout,
    texture_side: u32,
    billboard_world_size: f32,
) -> Transform {
    let side = texture_side.max(1) as f32;
    let width_fraction = surface.width as f32 / side;
    let height_fraction = surface.height as f32 / side;
    let center_x = (surface.x as f32 + surface.width as f32 * 0.5) / side - 0.5;
    let center_y = 0.5 - (surface.y as f32 + surface.height as f32 * 0.5) / side;
    Transform {
        translation: Vec3::new(
            center_x * billboard_world_size,
            center_y * billboard_world_size,
            0.0,
        ),
        scale: Vec3::new(width_fraction, height_fraction, 1.0),
        ..default()
    }
}

impl SpatialPendingIndex {
    fn new(points: Vec<BillboardPoint>) -> Self {
        let cell_size = pending_cell_size(&points);
        let mut index = Self {
            cell_size,
            cells: HashMap::new(),
            ids: HashSet::new(),
            len: 0,
        };
        for point in points {
            index.insert(point);
        }
        index
    }

    fn insert(&mut self, point: BillboardPoint) {
        if !self.ids.insert(point.image_id) {
            return;
        }
        let cell = self.cell_for(point.position);
        self.cells.entry(cell).or_default().push(point);
        self.len += 1;
    }

    /// Offers pending points to the sampler, walking cells outward from the
    /// camera.
    ///
    /// Points are offered by reference (cell plus index) rather than removed:
    /// a removed candidate that lost the draw would have to be reinserted,
    /// and reinsertion invalidates the indices of everything offered after
    /// it. Only the drawn points are taken, in [`Self::take_candidates`].
    ///
    /// The outward walk is what keeps this affordable on a large catalog. It
    /// visits at most [`BILLBOARD_PENDING_CELLS_PER_FRAME`] cells, but always
    /// the nearest ones first, so the points that dominate the weighting are
    /// offered every frame while the far tail is merely sampled over time.
    fn offer_candidates(
        &mut self,
        reservoir: &mut WeightedReservoir<LoadCandidate>,
        view: SchedulingView,
        view_bounds: &ViewBounds,
        sigma: f32,
    ) {
        if self.cells.is_empty() {
            return;
        }

        let camera_cell = self.cell_for(view.position);
        let mut cells_visited = 0;
        let mut offers = Vec::new();
        'outward: for radius in 0..=BILLBOARD_GRID_LOCAL_RADIUS {
            for x in (camera_cell.x - radius)..=(camera_cell.x + radius) {
                for y in (camera_cell.y - radius)..=(camera_cell.y + radius) {
                    for z in (camera_cell.z - radius)..=(camera_cell.z + radius) {
                        if radius > 0
                            && (x - camera_cell.x)
                                .abs()
                                .max((y - camera_cell.y).abs())
                                .max((z - camera_cell.z).abs())
                                != radius
                        {
                            continue;
                        }
                        let cell = SpatialCell { x, y, z };
                        let Some(points) = self.cells.get(&cell) else {
                            continue;
                        };
                        cells_visited += 1;
                        for (index, point) in points.iter().enumerate() {
                            let value = candidate_value(point.position, view, view_bounds, sigma);
                            if value <= 0.0 {
                                continue;
                            }
                            offers.push((
                                value,
                                LoadCandidate {
                                    candidate: PendingCandidate { cell, index },
                                },
                            ));
                        }
                        if cells_visited >= BILLBOARD_PENDING_CELLS_PER_FRAME {
                            break 'outward;
                        }
                    }
                }
            }
        }

        // A sparse streamed catalog can leave the camera with no populated
        // cell nearby; sample a bounded slice of the rest so those points are
        // still reachable rather than waiting for the camera to arrive.
        if offers.is_empty() {
            let cells = self
                .cells
                .keys()
                .copied()
                .take(BILLBOARD_PENDING_FALLBACK_CELL_LIMIT)
                .collect::<Vec<_>>();
            for cell in cells {
                let Some(points) = self.cells.get(&cell) else {
                    continue;
                };
                for (index, point) in points.iter().enumerate() {
                    let value = candidate_value(point.position, view, view_bounds, sigma);
                    if value <= 0.0 {
                        continue;
                    }
                    offers.push((
                        value,
                        LoadCandidate {
                            candidate: PendingCandidate { cell, index },
                        },
                    ));
                }
            }
        }

        for (value, candidate) in offers {
            reservoir.offer(value, candidate);
        }
    }

    /// Removes the drawn points from the index and returns them.
    ///
    /// Removal is `swap_remove`, which moves a cell's last point into the
    /// vacated slot, so within each cell the highest index is taken first and
    /// every remaining index in the batch stays valid.
    fn take_candidates(&mut self, mut candidates: Vec<LoadCandidate>) -> Vec<BillboardPoint> {
        candidates.sort_by(|left, right| {
            left.candidate
                .cell
                .cmp(&right.candidate.cell)
                .then_with(|| right.candidate.index.cmp(&left.candidate.index))
        });
        let mut points = Vec::with_capacity(candidates.len());
        for LoadCandidate { candidate } in candidates {
            let Some(cell_points) = self.cells.get_mut(&candidate.cell) else {
                continue;
            };
            if candidate.index >= cell_points.len() {
                continue;
            }
            let point = cell_points.swap_remove(candidate.index);
            if cell_points.is_empty() {
                self.cells.remove(&candidate.cell);
            }
            self.ids.remove(&point.image_id);
            self.len -= 1;
            points.push(point);
        }
        points
    }

    fn cell_for(&self, position: Vec3) -> SpatialCell {
        SpatialCell {
            x: (position.x / self.cell_size).floor() as i32,
            y: (position.y / self.cell_size).floor() as i32,
            z: (position.z / self.cell_size).floor() as i32,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Where every point still awaiting a texture sits, for coverage stats.
    fn positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.cells
            .values()
            .flat_map(|points| points.iter().map(|point| point.position))
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

fn pending_cell_size(points: &[BillboardPoint]) -> f32 {
    if points.len() <= 1 {
        return 1.0;
    }

    let mut min = points[0].position;
    let mut max = points[0].position;
    for point in points.iter().skip(1) {
        min = min.min(point.position);
        max = max.max(point.position);
    }
    let extent = max - min;
    let max_extent = extent.x.max(extent.y).max(extent.z).max(1.0);
    let target_cells_per_axis = (points.len() as f32).cbrt().ceil().max(1.0);
    (max_extent / target_cells_per_axis).max(1.0)
}

/// Builds the view every scheduling decision reads, including where the
/// camera will be shortly. The projection comes from the fly camera's own
/// velocity rather than a frame-to-frame position delta: the velocity is
/// already inertia-smoothed, is available on the very first frame of a move,
/// and does not read as motion when the camera is only rotating.
fn scheduling_view(
    camera: Option<(&Transform, &FlyCamera)>,
    navigation: &NavigationSettings,
) -> SchedulingView {
    let Some((transform, fly_camera)) = camera else {
        return SchedulingView {
            position: Vec3::ZERO,
            forward: Vec3::Z,
            lookahead_position: Vec3::ZERO,
        };
    };

    let max_lookahead = navigation.reference_distance * BILLBOARD_QUALITY_LOOKAHEAD_MAX_STEPS;
    let travel = fly_camera.velocity * BILLBOARD_QUALITY_LOOKAHEAD_SECONDS;
    let lookahead = travel.clamp_length_max(max_lookahead);
    SchedulingView {
        position: transform.translation,
        forward: transform.rotation.mul_vec3(Vec3::NEG_Z),
        lookahead_position: transform.translation + lookahead,
    }
}

/// Resolution every billboard decode targets.
///
/// One value, not a ladder. Sizing images down by distance cost a visible
/// quality drop on nearly four in five on-screen billboards while saving
/// only ~12% of decode time, because reading and parsing the file dominates
/// a decode regardless of the target size. `max_texture_side` is a global
/// cap, `0` meaning source resolution.
///
/// Videos stay capped at [`BILLBOARD_VIDEO_TEXTURE_SIDE`]: a playing video
/// replaces its frame at its playback rate, so decoding one at source
/// resolution would spend the upload budget on pixels discarded unseen.
fn billboard_texture_limit(
    controls: &BillboardControls,
    encoding: BillboardTextureEncoding,
    is_video: bool,
) -> u32 {
    let requested = if is_video {
        if controls.max_texture_side == 0 {
            BILLBOARD_VIDEO_TEXTURE_SIDE
        } else {
            controls.max_texture_side.min(BILLBOARD_VIDEO_TEXTURE_SIDE)
        }
    } else {
        controls.max_texture_side
    };
    // `0` means source resolution, which the device limit still bounds: the
    // whole image is one texture now, so a source larger than
    // `max_texture_dimension_2d` would be rejected outright by wgpu rather
    // than merely look wrong.
    if requested == 0 {
        encoding.max_texture_side
    } else {
        requested.min(encoding.max_texture_side)
    }
}

/// Weight of a candidate at `position`: the Gaussian "p score", measured
/// from whichever of the camera's current and projected position is nearer,
/// so an image being flown toward is weighted as though already close.
///
/// Returns exactly zero outside the render boundary. That is not a cap on how
/// far the sampler may reach — it is the fact that those points are hidden
/// behind the opaque boundary walls and cannot be seen at any quality, so
/// decoding one could never show the user anything. Every point that *is*
/// rendered keeps a non-zero weight, however small.
fn candidate_value(
    position: Vec3,
    view: SchedulingView,
    view_bounds: &ViewBounds,
    sigma: f32,
) -> f32 {
    if view_bounds.is_beyond(view.position, position) {
        return 0.0;
    }
    load_value(
        view.approach_offset(position),
        Vec3::ZERO,
        view.forward,
        sigma,
    )
}

fn pending_priority_score(position: Vec3, view: SchedulingView) -> f32 {
    let to_point = view.approach_offset(position);
    let distance = to_point.length().max(0.001);
    let direction = to_point / distance;
    let forward_alignment = direction.dot(view.forward);
    let visibility_penalty = if forward_alignment >= 0.0 {
        0.0
    } else if forward_alignment >= BILLBOARD_NEAR_VISIBLE_DOT {
        distance * 0.75
    } else {
        distance * 3.0
    };
    let screen_size_bonus = 1.0 / distance;

    distance + visibility_penalty - (forward_alignment.max(0.0) * 2.0) - screen_size_bonus
}

fn loaded_eviction_score(record: &LoadedBillboardRecord, view: SchedulingView) -> f32 {
    let to_point = view.approach_offset(record.point.position);
    let distance_squared = to_point.length_squared();
    let direction = normalized_or(to_point, view.forward);
    let forward_alignment = direction.dot(view.forward);
    let visibility_penalty = if forward_alignment < BILLBOARD_NEAR_VISIBLE_DOT {
        distance_squared * 2.0
    } else {
        0.0
    };
    distance_squared + visibility_penalty + record.texture_side as f32 * 0.001
}

/// Whether admitting `candidate` in place of `resident` improves the cache.
///
/// This is what bounds the sampler's tail without a cap: a far point that
/// wins an unlikely draw still has to be worth more than the worst resident
/// to displace it. When it is not, the decode is skipped and the point is
/// returned to the pending index, so an improbable draw costs one comparison
/// rather than a wasted decode and an evicted near billboard.
fn replacement_improves_cache(
    candidate: &BillboardPoint,
    resident: &LoadedBillboardRecord,
    view: SchedulingView,
) -> bool {
    pending_priority_score(candidate.position, view)
        < pending_priority_score(resident.point.position, view)
}

pub(crate) fn billboard_rotation(
    to_camera: Vec3,
    viewport_normal: Vec3,
    camera_up: Vec3,
    axis: BillboardFacingAxis,
) -> Quat {
    match axis {
        BillboardFacingAxis::All => all_axes_billboard_rotation(viewport_normal, camera_up),
        BillboardFacingAxis::X | BillboardFacingAxis::Y | BillboardFacingAxis::Z => {
            axis_locked_billboard_rotation(to_camera, axis_vector(axis))
        }
    }
}

fn axis_locked_billboard_rotation(to_camera: Vec3, up: Vec3) -> Quat {
    let normal = projected_normal(to_camera, up).unwrap_or_else(|| fallback_normal(up));
    rotation_from_axes(up.cross(normal).normalize(), up, normal)
}

fn all_axes_billboard_rotation(viewport_normal: Vec3, camera_up: Vec3) -> Quat {
    let normal = normalized_or(viewport_normal, Vec3::Z);
    let up = projected_normal(camera_up, normal).unwrap_or_else(|| fallback_normal(normal));
    rotation_from_axes(
        up.cross(normal).normalize(),
        normal.cross(up.cross(normal)).normalize(),
        normal,
    )
}

fn rotation_from_axes(right: Vec3, up: Vec3, normal: Vec3) -> Quat {
    Quat::from_mat3(&Mat3::from_cols(right, up, normal))
}

fn projected_normal(vector: Vec3, fixed_axis: Vec3) -> Option<Vec3> {
    let projected = vector - fixed_axis * vector.dot(fixed_axis);
    (projected.length_squared() > f32::EPSILON).then(|| projected.normalize())
}

fn fallback_normal(fixed_axis: Vec3) -> Vec3 {
    let candidate = if fixed_axis.dot(Vec3::Y).abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    projected_normal(candidate, fixed_axis).unwrap_or(Vec3::X)
}

fn axis_vector(axis: BillboardFacingAxis) -> Vec3 {
    match axis {
        BillboardFacingAxis::X => Vec3::X,
        BillboardFacingAxis::Y => Vec3::Y,
        BillboardFacingAxis::Z => Vec3::Z,
        BillboardFacingAxis::All => unreachable!("all-axes billboards do not use a locked axis"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Surfaces must stay in a depth-writing phase. A blended alpha mode
    /// moves them into the sorted transparent phase, where per-mesh distance
    /// sorting cannot order intersecting camera-facing quads, so billboards
    /// visibly swap layering as the camera moves.
    #[test]
    fn billboard_surfaces_use_a_depth_writing_alpha_mode() {
        assert!(
            matches!(
                BILLBOARD_SURFACE_ALPHA_MODE,
                AlphaMode::Opaque | AlphaMode::Mask(_)
            ),
            "billboard surfaces must not be alpha-blended: {BILLBOARD_SURFACE_ALPHA_MODE:?}"
        );
    }

    /// The cutoff exists for a source image's own transparency: fully
    /// transparent texels must be discarded while a feathered edge survives.
    #[test]
    fn billboard_alpha_cutoff_discards_transparency_without_clipping_soft_edges() {
        let AlphaMode::Mask(cutoff) = BILLBOARD_SURFACE_ALPHA_MODE else {
            panic!("expected a masked alpha mode: {BILLBOARD_SURFACE_ALPHA_MODE:?}");
        };
        assert!(cutoff > 0.0, "fully transparent texels must be discarded");
        assert!(
            cutoff <= 0.1,
            "cutoff {cutoff} would clip a source image's own soft edges"
        );
    }

    /// A billboard draws exactly one quad, so there is no interior edge for
    /// float error to open a seam along.
    ///
    /// This is the whole reason for the single surface. The tile grid it
    /// replaced put each tile on its own model matrix; two neighbours
    /// reached their shared edge through different matrices and differed by
    /// an ULP or two once rotation and world position were applied, so with
    /// MSAA off a boundary pixel was claimed by both quads or by neither —
    /// the latter showing the clear colour as a dark line.
    #[test]
    fn a_billboard_draws_one_quad_with_no_interior_edge() {
        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<StandardMaterial>>();

        let (billboard, mesh, surface) = {
            let world = app.world_mut();
            let billboard = world.spawn(Transform::default()).id();
            let mesh = world
                .resource_mut::<Assets<Mesh>>()
                .add(Rectangle::new(1.0, 1.0));
            let texture = world.resource_mut::<Assets<Image>>().add(Image::default());
            let material = world
                .resource_mut::<Assets<StandardMaterial>>()
                .add(StandardMaterial::default());
            (
                billboard,
                mesh,
                UploadedBillboardSurface {
                    layout: BillboardSurfaceLayout {
                        x: 0,
                        y: 512,
                        width: 2048,
                        height: 1024,
                    },
                    texture_handle: texture,
                    material_handle: material,
                },
            )
        };

        let mut queue = bevy::ecs::world::CommandQueue::default();
        let assets = {
            let mut commands = Commands::new(&mut queue, app.world());
            attach_billboard_surface(&mut commands, billboard, &mesh, 2048, 2.0, Some(surface))
        };
        queue.apply(app.world_mut());

        let quad = assets.entity.expect("the surface is attached as one quad");
        assert_eq!(
            app.world()
                .entity(billboard)
                .get::<Children>()
                .map(|c| c.len()),
            Some(1),
            "a billboard draws exactly one quad"
        );
        let transform = app
            .world()
            .entity(quad)
            .get::<Transform>()
            .expect("the quad is placed");
        // A 2048x1024 surface on a 2048 square: full width, half height,
        // centred vertically because the letterbox margin is symmetric.
        assert!((transform.scale.x - 1.0).abs() < 1e-6);
        assert!((transform.scale.y - 0.5).abs() < 1e-6);
        assert!(transform.translation.abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    /// A billboard's texture is one contiguous surface, so the device's
    /// `max_texture_dimension_2d` bounds the whole image rather than a
    /// single 1024 px tile. Exceeding it is not a soft failure — wgpu
    /// rejects the texture — so source resolution has to be clamped to it.
    #[test]
    fn source_resolution_is_still_bounded_by_the_device_texture_limit() {
        let encoding = BillboardTextureEncoding {
            format: BillboardTextureFormat::Bc7,
            max_texture_side: 8192,
        };
        // `0` means source resolution, which is where the catalog's 11656 px
        // images would otherwise land.
        let controls = BillboardControls::new(1024, 0);
        assert_eq!(
            billboard_texture_limit(&controls, encoding, false),
            8192,
            "source resolution must be clamped to what the device accepts"
        );

        // An explicit cap below the device limit still wins.
        let capped = BillboardControls::new(1024, 4096);
        assert_eq!(billboard_texture_limit(&capped, encoding, false), 4096);

        // And a cap above it does not.
        let over = BillboardControls::new(1024, 16384);
        assert_eq!(billboard_texture_limit(&over, encoding, false), 8192);

        // Video is capped tighter still, because its frames are replaced 12
        // times a second.
        assert_eq!(
            billboard_texture_limit(&controls, encoding, true),
            BILLBOARD_VIDEO_TEXTURE_SIDE
        );
    }

    /// A surface covering only part of the square must be centred on the
    /// image it came from, not on the square.
    #[test]
    fn a_cropped_surface_stays_centred_on_its_image() {
        // A 1024-wide image centred on a 3072 square sits dead centre and
        // occupies a third of the billboard's width.
        let centre = billboard_surface_transform(
            BillboardSurfaceLayout {
                x: 1024,
                y: 1024,
                width: 1024,
                height: 1024,
            },
            3072,
            3.0,
        );
        assert!(centre.translation.abs_diff_eq(Vec3::ZERO, 1e-6));
        assert!((centre.scale.x - 1.0 / 3.0).abs() < 1e-6);
    }

    /// An off-centre surface must be placed off-centre by the same fraction
    /// of the billboard, so a letterboxed image lands where its pixels are.
    #[test]
    fn a_surface_is_placed_where_it_sits_on_the_square() {
        // Occupies the left half of a 2048 square: centre at x = -1/4 of the
        // billboard, i.e. -0.5 world units at world size 2.
        let left = billboard_surface_transform(
            BillboardSurfaceLayout {
                x: 0,
                y: 0,
                width: 1024,
                height: 2048,
            },
            2048,
            2.0,
        );
        assert!((left.translation.x + 0.5).abs() < 1e-6);
        assert!((left.scale.x - 0.5).abs() < 1e-6);
        assert!((left.scale.y - 1.0).abs() < 1e-6);
    }

    fn stationary_view(position: Vec3, forward: Vec3) -> SchedulingView {
        SchedulingView {
            position,
            forward,
            lookahead_position: position,
        }
    }

    fn test_point(image_id: usize, position: Vec3) -> BillboardPoint {
        BillboardPoint {
            image_id,
            path: format!("image-{image_id}.png").into(),
            position,
            is_video: false,
            duration_seconds: None,
            source_size: None,
            coordinate_labels: [None, None, None],
        }
    }

    /// Draws from the pending index the way the scheduler does.
    fn draw_pending(
        pending: &mut SpatialPendingIndex,
        view: SchedulingView,
        bounds: &ViewBounds,
        navigation: &NavigationSettings,
        capacity: usize,
        seed: u64,
    ) -> Vec<BillboardPoint> {
        let sigma = load_value_sigma(navigation.reference_distance);
        let mut reservoir = WeightedReservoir::new(capacity, seed);
        pending.offer_candidates(&mut reservoir, view, bounds, sigma);
        pending.take_candidates(reservoir.take())
    }

    #[test]
    fn near_visible_points_dominate_the_draw() {
        let bounds = ViewBounds::default();
        let navigation = NavigationSettings::new(4.0, 10.0);
        let view = stationary_view(Vec3::ZERO, Vec3::Z);

        // One near point ahead, one far ahead, one just behind. Over many
        // independent single draws the near one should win nearly always,
        // without the far ones being formally excluded.
        let mut near_wins = 0;
        let trials = 200;
        for seed in 0..trials {
            let mut pending = SpatialPendingIndex::new(vec![
                test_point(1, Vec3::new(0.0, 0.0, 100.0)),
                test_point(2, Vec3::new(0.0, 0.0, 3.0)),
                test_point(3, Vec3::new(0.0, 0.0, -1.0)),
            ]);
            let drawn = draw_pending(&mut pending, view, &bounds, &navigation, 1, seed);
            assert_eq!(drawn.len(), 1);
            assert_eq!(pending.len(), 2, "only the drawn point leaves the index");
            if drawn[0].image_id == 2 {
                near_wins += 1;
            }
        }

        assert!(
            near_wins > trials * 3 / 4,
            "near point won {near_wins}/{trials} draws; expected a large majority"
        );
    }

    #[test]
    fn points_outside_the_render_boundary_are_never_drawn() {
        let navigation = NavigationSettings::new(4.0, 10.0);
        let view = stationary_view(Vec3::ZERO, Vec3::Z);
        // Boundary walls hide anything past this radius, so decoding it
        // could not show the user anything.
        let mut bounds = ViewBounds::default();
        bounds.radius = 50.0;

        assert_eq!(
            candidate_value(Vec3::Z * 500.0, view, &bounds, 40.0),
            0.0,
            "a point behind the boundary wall has no value at any distance"
        );

        for seed in 0..40 {
            let mut pending =
                SpatialPendingIndex::new(vec![test_point(1, Vec3::new(0.0, 0.0, 500.0))]);
            let drawn = draw_pending(&mut pending, view, &bounds, &navigation, 4, seed);
            assert!(drawn.is_empty(), "out-of-bounds point was drawn");
        }
    }

    #[test]
    fn a_draw_removes_exactly_the_points_it_returns() {
        let bounds = ViewBounds::default();
        let navigation = NavigationSettings::new(4.0, 10.0);
        let view = stationary_view(Vec3::ZERO, Vec3::Z);
        // All in one cell, so every removal is a `swap_remove` that shuffles
        // the indices recorded when the batch was offered.
        let mut pending = SpatialPendingIndex::new(
            (0..12)
                .map(|index| test_point(index, Vec3::Z * (index as f32 + 1.0)))
                .collect(),
        );

        let drawn = draw_pending(&mut pending, view, &bounds, &navigation, 5, 9);

        let returned = drawn
            .iter()
            .map(|point| point.image_id)
            .collect::<HashSet<_>>();
        assert_eq!(returned.len(), 5, "no point may be returned twice");
        assert_eq!(pending.len(), 7);
        let remaining = pending
            .cells
            .values()
            .flatten()
            .map(|point| point.image_id)
            .collect::<HashSet<_>>();
        assert!(
            remaining.is_disjoint(&returned),
            "a returned point must no longer be in the index"
        );
        assert_eq!(remaining.len(), 7);
    }

    #[test]
    fn every_pending_point_stays_reachable_across_repeated_draws() {
        let bounds = ViewBounds::default();
        let navigation = NavigationSettings::new(4.0, 10.0);
        let view = stationary_view(Vec3::ZERO, Vec3::Z);
        // Spread across the near field: with no lanes and no caps, draining
        // must not strand anything, which is the property that replaces the
        // old deterministic queue's exhaustiveness.
        let mut pending = SpatialPendingIndex::new(
            (0..60)
                .map(|index| {
                    test_point(
                        index,
                        Vec3::new((index % 5) as f32 * 4.0, 0.0, (index / 5) as f32 * 4.0),
                    )
                })
                .collect(),
        );

        let mut drained = HashSet::new();
        for seed in 0..200 {
            if pending.is_empty() {
                break;
            }
            for point in draw_pending(&mut pending, view, &bounds, &navigation, 4, seed) {
                assert!(drained.insert(point.image_id), "point drawn twice");
            }
        }

        assert_eq!(
            drained.len(),
            60,
            "the sampler must eventually reach every point"
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn a_decode_the_camera_left_behind_is_cancelled() {
        let mut state = ImageLoadingState::new(Vec::new(), 64 * 1024 * 1024, 4, 0);
        let point = test_point(1, Vec3::Z * 20.0);
        let cancel = CancelToken::new();
        state.in_flight.insert(
            point.image_id,
            InFlightLoad {
                point,
                texture_limit: 0,
                cancel: cancel.clone(),
            },
        );
        let mut bounds = ViewBounds::default();
        bounds.radius = 100.0;

        // Inside the boundary: the decode can still be shown.
        let near = stationary_view(Vec3::ZERO, Vec3::Z);
        assert_eq!(state.cancel_unreachable_loads(near, &bounds), 0);
        assert!(!cancel.is_cancelled());

        // The camera fled far enough that the point is behind the boundary
        // wall, so finishing the decode could not show the user anything.
        let far = stationary_view(Vec3::Z * -600.0, Vec3::Z);
        assert_eq!(state.cancel_unreachable_loads(far, &bounds), 1);
        assert!(cancel.is_cancelled());
        // Idempotent: a second pass does not re-count an already-stopped load.
        assert_eq!(state.cancel_unreachable_loads(far, &bounds), 0);
    }

    #[test]
    fn pending_index_deduplicates_returned_points() {
        let point = test_point(7, Vec3::ONE);
        let mut pending = SpatialPendingIndex::new(Vec::new());

        pending.insert(point.clone());
        pending.insert(point);

        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn lookahead_is_capped_and_ignores_a_stationary_camera() {
        let navigation = NavigationSettings::new(4.0, 10.0);
        let transform = Transform::from_translation(Vec3::ZERO);
        let stopped = FlyCamera {
            yaw: 0.0,
            pitch: 0.0,
            sensitivity: 1.0,
            velocity: Vec3::ZERO,
            velocity_response: 1.0,
        };
        let view = scheduling_view(Some((&transform, &stopped)), &navigation);
        assert_eq!(view.lookahead_position, view.position);

        let sprinting = FlyCamera {
            velocity: Vec3::Z * 10_000.0,
            ..stopped
        };
        let view = scheduling_view(Some((&transform, &sprinting)), &navigation);
        assert_eq!(
            view.lookahead_position.length(),
            navigation.reference_distance * BILLBOARD_QUALITY_LOOKAHEAD_MAX_STEPS
        );
    }

    #[test]
    fn approach_offset_never_exceeds_the_direct_distance() {
        let view = SchedulingView {
            position: Vec3::ZERO,
            forward: Vec3::Z,
            lookahead_position: Vec3::Z * 50.0,
        };
        // Behind the camera: the projection moves away from the point, so the
        // current position stays authoritative and nothing is demoted.
        let behind = Vec3::Z * -20.0;
        assert_eq!(view.approach_offset(behind), behind);
    }

    /// A point of a known source size, so byte-budget tests can state their
    /// expectations in whole textures rather than in magic numbers.
    fn sized_point(image_id: usize, position: Vec3, side: u32) -> BillboardPoint {
        let mut point = test_point(image_id, position);
        point.source_size = Some(UVec2::new(side, side));
        point
    }

    #[test]
    fn a_texture_costs_its_block_aligned_square_in_the_chosen_format() {
        let point = sized_point(1, Vec3::ZERO, 1024);
        // BC7 is one byte per texel, RGBA8 four.
        assert_eq!(
            point.texture_bytes(BillboardTextureFormat::Bc7),
            1024 * 1024
        );
        assert_eq!(
            point.texture_bytes(BillboardTextureFormat::Rgba8),
            1024 * 1024 * 4
        );

        // An unaligned source is charged for the square actually decoded.
        let odd = sized_point(2, Vec3::ZERO, 1023);
        assert_eq!(odd.texture_bytes(BillboardTextureFormat::Bc7), 1024 * 1024);
    }

    #[test]
    fn a_point_of_unknown_size_is_charged_rather_than_free() {
        // A catalog does not always report dimensions. Charging zero would
        // admit such points without limit and make them un-evictable for
        // size, so they are charged the catalog mean instead.
        let unknown = test_point(1, Vec3::ZERO);
        assert_eq!(unknown.source_size, None);
        assert_eq!(
            unknown.texture_bytes(BillboardTextureFormat::Bc7),
            billboard_surface_bytes(
                UVec2::splat(BILLBOARD_UNKNOWN_SOURCE_SIDE),
                BillboardTextureFormat::Bc7,
            )
        );
    }

    #[test]
    fn eviction_frees_bytes_until_the_resident_set_fits_the_budget() {
        // Three 1024px BC7 textures are 1 MiB each; a 2 MiB budget fits two.
        let mut state = ImageLoadingState::new(Vec::new(), 2 * 1024 * 1024, 4, 0);
        state.texture_format = BillboardTextureFormat::Bc7;
        for (image_id, distance) in [(1usize, 5.0f32), (2, 50.0), (3, 500.0)] {
            let point = sized_point(image_id, Vec3::Z * distance, 1024);
            state.loaded.insert(
                image_id,
                LoadedBillboardRecord {
                    point,
                    entity: Entity::PLACEHOLDER,
                    texture_side: 1024,
                    texture_limit: 0,
                    surface_assets: BillboardSurfaceAssets::default(),
                },
            );
        }
        assert_eq!(state.loaded_bytes_used(), 3 * 1024 * 1024);
        assert!(state.cache_over_limit());

        // Evicting the worst-placed resident is enough to fit the budget,
        // and the farthest one is the one that goes.
        let view = stationary_view(Vec3::ZERO, Vec3::Z);
        let farthest = state
            .farthest_loaded_billboard(view)
            .expect("a resident to evict");
        assert_eq!(farthest, 3, "the farthest billboard is evicted first");
        state.loaded.remove(&farthest);
        assert!(!state.cache_over_limit());
    }

    #[test]
    fn a_budget_smaller_than_one_texture_still_keeps_one() {
        // Otherwise the loader would evict everything, re-decode, and evict
        // again forever — showing an empty view rather than one image.
        let mut state = ImageLoadingState::new(Vec::new(), 1, 4, 0);
        state.texture_format = BillboardTextureFormat::Bc7;
        state.loaded.insert(
            1,
            LoadedBillboardRecord {
                point: sized_point(1, Vec3::ZERO, 4096),
                entity: Entity::PLACEHOLDER,
                texture_side: 4096,
                texture_limit: 0,
                surface_assets: BillboardSurfaceAssets::default(),
            },
        );

        assert!(state.loaded_bytes_used() > state.texture_budget_bytes);
        assert!(
            !state.cache_over_limit(),
            "the last resident is never evicted"
        );
    }

    #[test]
    fn a_large_texture_is_charged_more_than_a_small_one() {
        // The whole point of a byte budget over a count: catalog images
        // differ ~100x in size, so counting them cannot bound VRAM.
        let small = sized_point(1, Vec3::ZERO, 1024);
        let large = sized_point(2, Vec3::ZERO, 8192);
        assert_eq!(
            large.texture_bytes(BillboardTextureFormat::Bc7),
            64 * small.texture_bytes(BillboardTextureFormat::Bc7)
        );
    }

    #[test]
    fn nearer_visible_candidate_replaces_farther_resident() {
        let view = SchedulingView {
            position: Vec3::ZERO,
            forward: Vec3::Z,
            lookahead_position: Vec3::ZERO,
        };
        let candidate = test_point(2, Vec3::Z * 5.0);
        let resident = LoadedBillboardRecord {
            point: test_point(1, Vec3::Z * 100.0),
            entity: Entity::PLACEHOLDER,
            texture_side: 64,
            texture_limit: 64,
            surface_assets: BillboardSurfaceAssets::default(),
        };

        assert!(replacement_improves_cache(&candidate, &resident, view));
        assert!(!replacement_improves_cache(
            &resident.point,
            &resident,
            view
        ));
    }

    #[test]
    fn replacement_decode_does_not_evict_before_it_is_ready() {
        let mut state = ImageLoadingState::new(Vec::new(), 1, 2, 1024);
        let resident = test_point(1, Vec3::Z * 100.0);
        state.loaded.insert(
            resident.image_id,
            LoadedBillboardRecord {
                point: resident,
                entity: Entity::PLACEHOLDER,
                texture_side: 64,
                texture_limit: 64,
                surface_assets: BillboardSurfaceAssets::default(),
            },
        );
        state.in_flight.insert(
            2,
            InFlightLoad {
                point: test_point(2, Vec3::Z * 5.0),
                texture_limit: 64,
                cancel: CancelToken::new(),
            },
        );

        assert_eq!(state.loaded.len(), 1);
        assert!(!state.cache_over_limit());
        // The in-flight replacement is charged against the budget before it
        // lands, so a burst of scheduling cannot overshoot on arrival.
        assert_eq!(
            state.loaded_bytes_used(),
            2 * billboard_surface_bytes(
                UVec2::splat(BILLBOARD_UNKNOWN_SOURCE_SIDE),
                BillboardTextureFormat::default(),
            )
        );
    }

    #[test]
    fn full_cache_schedules_nearer_candidate_without_evicting_resident() {
        let resident = test_point(1, Vec3::Z * 100.0);
        let candidate = test_point(2, Vec3::Z * 5.0);
        let mut state = ImageLoadingState::new(vec![candidate], 1, 2, 1024);
        state.loaded.insert(
            resident.image_id,
            LoadedBillboardRecord {
                point: resident,
                entity: Entity::PLACEHOLDER,
                texture_side: 64,
                texture_limit: 64,
                surface_assets: BillboardSurfaceAssets::default(),
            },
        );
        let mut pause_menu = PauseMenuState::default();
        pause_menu.resume();

        let mut app = App::new();
        app.insert_resource(state)
            .insert_resource(BillboardControls::new(1, 1024))
            .insert_resource(pause_menu)
            .insert_resource(NavigationSettings::new(4.0, 10.0))
            .insert_resource(ViewBounds::default())
            .insert_resource(BillboardTextureEncoding::default())
            .insert_resource(DecodeBudget::new(2))
            .init_resource::<Assets<Image>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<PointCloud>()
            .add_systems(Update, schedule_image_loads);

        app.update();

        let state = app.world().resource::<ImageLoadingState>();
        assert!(state.in_flight.contains_key(&2));
        assert!(state.loaded.contains_key(&1));
        assert_eq!(state.loaded.len(), 1);
    }

    #[test]
    fn upload_budget_allows_one_oversized_texture_but_not_a_burst() {
        assert!(upload_fits_frame_budget(
            0,
            0,
            BILLBOARD_GPU_UPLOAD_BYTES_PER_FRAME * 2
        ));
        assert!(!upload_fits_frame_budget(
            1,
            BILLBOARD_GPU_UPLOAD_BYTES_PER_FRAME,
            1
        ));
    }

    #[test]
    fn points_behind_the_camera_keep_a_small_but_real_weight() {
        let view = stationary_view(Vec3::ZERO, Vec3::Z);
        let bounds = ViewBounds::default();
        let sigma = load_value_sigma(10.0);

        // The old scheduler deferred everything behind the camera outside a
        // fixed buffer radius. The sampler instead down-weights it: a
        // turn-around still finds a warm cache, and distance decides the odds
        // rather than a hard radius.
        let just_behind = candidate_value(Vec3::NEG_Z * 20.0, view, &bounds, sigma);
        let far_behind = candidate_value(Vec3::NEG_Z * 250.0, view, &bounds, sigma);
        let ahead = candidate_value(Vec3::Z * 20.0, view, &bounds, sigma);

        assert!(ahead > just_behind);
        assert!(just_behind > far_behind);
        assert!(far_behind > 0.0, "nothing renderable is excluded outright");
    }

    #[test]
    fn camera_snapshot_ignores_sub_threshold_motion() {
        let previous = BillboardFacingSnapshot {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            axis: BillboardFacingAxis::Y,
        };
        let current = BillboardFacingSnapshot {
            position: Vec3::new(0.01, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            axis: BillboardFacingAxis::Y,
        };

        assert!(!billboard_facing_snapshot_changed(previous, current));
    }

    #[test]
    fn camera_snapshot_tracks_axis_changes() {
        let previous = BillboardFacingSnapshot {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            axis: BillboardFacingAxis::Y,
        };
        let current = BillboardFacingSnapshot {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            axis: BillboardFacingAxis::Z,
        };

        assert!(billboard_facing_snapshot_changed(previous, current));
    }

    #[test]
    fn default_billboard_axis_is_all() {
        assert_eq!(
            BillboardFacingSettings::default().axis,
            BillboardFacingAxis::All
        );
    }

    #[test]
    fn billboard_update_guard_rejects_behind_camera() {
        assert!(!billboard_should_update(
            Vec3::ZERO,
            Vec3::Z,
            Vec3::NEG_Z,
            100.0,
        ));
    }

    #[test]
    fn billboard_update_guard_rejects_far_billboard() {
        assert!(!billboard_should_update(
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z * 20.0,
            100.0,
        ));
    }

    #[test]
    fn billboard_update_guard_accepts_near_billboard_in_front() {
        assert!(billboard_should_update(
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z * 5.0,
            100.0,
        ));
    }

    #[test]
    fn face_billboards_system_runs_without_query_conflicts() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(BillboardFacingSettings::default());
        app.insert_resource(NavigationSettings::new(4.0, 10.0));
        app.insert_resource(PauseMenuState::default());
        app.insert_resource(BillboardStats::default());
        app.add_systems(Update, face_billboards_to_camera);

        app.update();
    }

    #[test]
    fn billboard_rotation_around_y_preserves_vertical_axis() {
        let rotation = billboard_rotation(
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::Z,
            Vec3::Y,
            BillboardFacingAxis::Y,
        );

        assert_vec3_close(rotation.mul_vec3(Vec3::Y), Vec3::Y);
    }

    #[test]
    fn billboard_rotation_preserves_selected_axis_as_image_up() {
        for (axis, expected_up) in [
            (BillboardFacingAxis::X, Vec3::X),
            (BillboardFacingAxis::Y, Vec3::Y),
            (BillboardFacingAxis::Z, Vec3::Z),
        ] {
            let rotation =
                billboard_rotation(Vec3::new(1.0, 1.0, 1.0).normalize(), Vec3::Z, Vec3::Y, axis);

            assert_vec3_close(rotation.mul_vec3(Vec3::Y), expected_up);
        }
    }

    #[test]
    fn billboard_rotation_tracks_projected_camera_direction_for_each_axis() {
        for (axis, to_camera) in [
            (BillboardFacingAxis::X, Vec3::new(0.0, 1.0, 1.0)),
            (BillboardFacingAxis::Y, Vec3::new(1.0, 0.0, 1.0)),
            (BillboardFacingAxis::Z, Vec3::new(1.0, 1.0, 0.0)),
        ] {
            let axis_vector = axis_vector(axis);
            let expected_normal =
                (to_camera - axis_vector * to_camera.dot(axis_vector)).normalize();
            let rotation = billboard_rotation(to_camera.normalize(), Vec3::Z, Vec3::Y, axis);

            assert_vec3_close(rotation.mul_vec3(Vec3::Z), expected_normal);
        }
    }

    #[test]
    fn all_axes_billboard_rotation_points_normal_to_viewport() {
        let viewport_normal = Vec3::new(0.3, -0.2, 1.0).normalize();

        let rotation =
            billboard_rotation(Vec3::X, viewport_normal, Vec3::Y, BillboardFacingAxis::All);

        assert_vec3_close(rotation.mul_vec3(Vec3::Z), viewport_normal);
    }

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        let delta = actual - expected;
        assert!(
            delta.length() < 0.0001,
            "actual {actual:?} was not close to expected {expected:?}"
        );
    }

    #[test]
    fn content_half_extents_follow_the_letterboxed_source_aspect() {
        let billboard = |source_size| GenerationBillboard {
            image_id: 0,
            path: "video.mp4".into(),
            is_video: true,
            duration_seconds: None,
            source_size,
            texture_side: 64,
            surface_assets: BillboardSurfaceAssets::default(),
        };
        let landscape = billboard(Some(UVec2::new(1920, 1080))).content_half_extents(2.0);
        assert!((landscape - Vec2::new(1.0, 0.5625)).length() < 1e-5);
        let portrait = billboard(Some(UVec2::new(500, 1000))).content_half_extents(2.0);
        assert!((portrait - Vec2::new(0.5, 1.0)).length() < 1e-5);
        assert_eq!(billboard(None).content_half_extents(2.0), Vec2::ONE);
    }
}
