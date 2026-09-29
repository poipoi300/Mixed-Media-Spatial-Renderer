//! Manual spacing: click-to-select billboards and drag the selection toward
//! the mouse to re-space items by hand.
//!
//! Selection: a click on an unselected item selects only it, Shift+click
//! adds it instead, and a click on a selected item (with or without Shift)
//! deselects it. A click on empty world clears the selection (Shift+click
//! there keeps it). Pressing a selected item keeps the whole selection so it
//! can be dragged as a group, and a press held past `CLICK_HOLD_SECONDS`
//! without moving counts as a hold, not a click, so it deselects nothing.
//!
//! Selection is data-oriented to stay proportional to what the user can
//! actually interact with, not the catalog size: selected items live in one
//! id set inside [`SelectionState`], hit-testing only walks the loaded
//! billboard entities (bounded by the texture cache limit), and drag edits
//! write straight into the existing flat position stores — the loaded record
//! in [`ImageLoadingState`], the chunked [`PointCloud`] arrays, and
//! `ExplorerScene::image_points` via the [`ImagePointIndex`] id map — so a
//! catalog of tens of thousands of points never pays per-point cost for a
//! drag of a few items.
//!
//! Dragging is anchored to the cursor's projection into the world: each
//! frame the selection rotates around the player by the delta between the
//! total arcs from the drag-start cursor ray to the previous and current
//! rays (see [`cursor_drag_rotation`]), so the grabbed item stays glued
//! under the cursor 1:1 regardless of FOV, window size, render-target
//! scale, or perspective nonlinearity, and the accumulated rotation is
//! path-independent — no twist builds up along curved cursor paths. The
//! rotation is rigid — depth stays normalized and relative spacing is
//! preserved, so a group never collapses into a stack. Flying while
//! dragging carries the selection with the player, and the scroll wheel
//! pushes the selection away (up) or pulls it closer (down).
//!
//! A world-space translate gizmo — three world-axis arrows at the selection
//! centroid, distance-scaled to constant screen size — offers precise
//! single-axis moves, Blender-style: dragging an arrow tracks the point on
//! that axis closest to the cursor ray, leaving the other two coordinates
//! untouched. With "Snap drag to grid" enabled the gizmo steps live on a
//! grid whose cell is the billboard's world size (the displayed image
//! square), so separately placed items land exactly flush against one
//! another; a free drag snaps to the same grid on release.

use std::collections::{HashMap, HashSet};

use std::f32::consts::FRAC_PI_2;

use bevy::ecs::system::SystemParam;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::math::primitives::{Cone, Cuboid, Rectangle};
use bevy::prelude::*;
use bevy::render::camera::{ClearColorConfig, RenderTarget};
use bevy::render::view::RenderLayers;
use bevy::window::PrimaryWindow;
use spatial_viewer_ui::{
    BillboardControls, BillboardFacingAxis, BillboardFacingSettings, NavigationTargets,
    PauseMenuState, RenderResolutionSettings, UiInputCapture,
};

use crate::axis_gizmo::cursor_over_axis_gizmo;
use crate::catalog_load::CatalogLoadTask;
use crate::image_loading::{
    billboard_rotation, BillboardPoint, BillboardWorldSize, ImageLoadingState, MediaBillboard,
};
use crate::point_cloud::PointCloud;
use crate::video_controls::ray_rect_hit;
use crate::video_strip::{nearest_strip_hit, VideoStripHitQuery};
use crate::{cursor_to_render_target_position, ExplorerScene, FlyCamera, SceneRenderTarget};

/// Raw mouse motion (device pixels) a held left button must accumulate over
/// an item before the press counts as a drag; below this it is a plain click
/// that only changes the selection. Matches the right-drag look threshold.
const SELECTION_DRAG_MOTION_THRESHOLD: f32 = 4.0;
/// A motionless press held at least this long is a hold rather than a
/// click, so releasing it leaves the selection as the press made it.
const CLICK_HOLD_SECONDS: f32 = 0.35;
/// Distance multiplier per scroll-wheel notch while dragging: up pushes the
/// selection away from the player, down pulls it closer.
const DRAG_DISTANCE_SCROLL_FACTOR: f32 = 1.12;
/// Width of the accent rim a highlight quad leaves visible around the
/// selected picture, as a fraction of the billboard's edge (split between
/// opposite sides).
const HIGHLIGHT_RIM_FRACTION: f32 = 0.03;
/// How far behind the billboard plane (toward -Z in billboard-local space,
/// away from the camera) the highlight quad sits to avoid z-fighting.
const HIGHLIGHT_BEHIND_OFFSET_FACTOR: f32 = 0.02;
/// Matches the point cloud's `SELECTED` palette entry.
const HIGHLIGHT_COLOR: Color = Color::srgba(1.0, 0.86, 0.24, 0.9);
/// Fraction of the camera distance the translate gizmo spans, keeping its
/// on-screen size constant while flying.
const TRANSLATE_GIZMO_SCREEN_FACTOR: f32 = 0.09;
/// Arrow shaft length in gizmo-local units (the root scales them to world
/// size).
const TRANSLATE_GIZMO_SHAFT_LENGTH: f32 = 0.78;
const TRANSLATE_GIZMO_SHAFT_THICKNESS: f32 = 0.035;
const TRANSLATE_GIZMO_HEAD_LENGTH: f32 = 0.22;
const TRANSLATE_GIZMO_HEAD_RADIUS: f32 = 0.07;
/// Grab distance around an arrow, in gizmo-local units.
const TRANSLATE_GIZMO_GRAB_RADIUS: f32 = 0.14;
/// Total arrow reach in gizmo-local units (shaft plus head).
const TRANSLATE_GIZMO_ARROW_LENGTH: f32 =
    TRANSLATE_GIZMO_SHAFT_LENGTH + TRANSLATE_GIZMO_HEAD_LENGTH;
/// Blender-style axis colors: X red, Y green, Z blue.
const TRANSLATE_GIZMO_AXIS_COLORS: [Color; 3] = [
    Color::srgb(0.93, 0.29, 0.33),
    Color::srgb(0.38, 0.85, 0.42),
    Color::srgb(0.32, 0.56, 1.0),
];
const WORLD_AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
/// Render layer for the gizmo and its overlay camera (0 is the scene, 1 the
/// corner axis gizmo, 2 the presentation quad).
const TRANSLATE_GIZMO_RENDER_LAYER: usize = 3;
/// Minimum on-screen length (render-target pixels) an arrow must project to
/// before axis dragging responds; below this the axis points so directly
/// at or away from the viewer that no meaningful drag direction exists.
const TRANSLATE_GIZMO_MIN_SCREEN_LENGTH: f32 = 4.0;

/// One active left press: what it grabbed and the state of its drag.
struct PressState {
    kind: PressKind,
    /// Any selected item actually moved during this press, so the navigation
    /// targets need a rebuild on release.
    moved_any: bool,
}

enum PressKind {
    /// A press over a billboard: the selection changed on press (see
    /// [`SelectionState::press_billboard`]), a quick motionless release
    /// applies `release`, and a drag past the motion threshold free-moves
    /// the selection around the player, glued to the cursor ray.
    Billboard {
        image_id: usize,
        release: ClickRelease,
        /// Real-time seconds when the press began, telling a click from a
        /// hold.
        pressed_at_seconds: f32,
        accumulated_motion: f32,
        dragging: bool,
        /// Cursor ray direction when the drag began. Each frame's rotation
        /// is the delta between total-from-anchor arcs, which telescopes to
        /// one twist-free arc — chaining raw frame-to-frame arcs instead
        /// would accumulate roll along curved cursor paths (spherical
        /// holonomy), visibly orbiting off-center items around the grabbed
        /// one.
        drag_anchor_direction: Vec3,
        /// World-space cursor ray direction last frame; the drag rotation is
        /// the arc delta from here to the current frame's ray, keeping the
        /// grabbed item glued under the cursor. Held unchanged while the
        /// cursor is outside the window, so the selection re-syncs the
        /// moment it returns.
        last_cursor_direction: Vec3,
        /// Where the player was last frame, so flight during the drag
        /// carries the selection by the same displacement.
        last_camera_position: Vec3,
    },
    /// A press on a translate-gizmo arrow: the selection translates rigidly
    /// along one world axis. Cursor motion is projected onto the axis's
    /// on-screen direction (Blender-style), so motion perpendicular to the
    /// axis line contributes nothing — the closest-point-on-axis approach
    /// swung wildly along depth-facing axes once the cursor left the axis's
    /// screen line. World-anchored — player flight does not carry it.
    GizmoAxis {
        axis_index: usize,
        /// Gizmo origin when the arrow was grabbed; the axis line stays
        /// anchored here for the whole drag.
        origin: Vec3,
        /// Cursor position (render-target pixels) last frame; only the
        /// frame-to-frame delta along the axis's screen direction moves the
        /// selection.
        last_cursor: Vec2,
        /// Accumulated unsnapped axis offset driven by the cursor; the grid
        /// target quantizes from this, so steps don't ratchet.
        raw_offset: f32,
        /// Offset already applied to the selection, so each frame only adds
        /// the difference toward the (possibly grid-snapped) target.
        applied_offset: f32,
    },
}

/// What a click (a quick, motionless release) on a billboard does to the
/// selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClickRelease {
    Keep,
    /// A click on a selected item removes it.
    Deselect,
}

/// Which items are selected and the state of the in-progress press/drag.
#[derive(Resource, Default)]
pub struct SelectionState {
    selected: HashSet<usize>,
    press: Option<PressState>,
}

impl SelectionState {
    pub fn is_selected(&self, image_id: usize) -> bool {
        self.selected.contains(&image_id)
    }

    /// Applies a press on a billboard and returns what a click then does. An
    /// unselected item becomes the only selected one, or joins the selection
    /// with Shift (`extend`). A selected item keeps the whole selection so it
    /// can be dragged as a group; removing it waits for the release, since
    /// only a click (not a drag or a hold) deselects it.
    fn press_billboard(&mut self, image_id: usize, extend: bool) -> ClickRelease {
        if self.selected.contains(&image_id) {
            return ClickRelease::Deselect;
        }
        if !extend {
            self.selected.clear();
        }
        self.selected.insert(image_id);
        ClickRelease::Keep
    }

    fn release_click(&mut self, image_id: usize, release: ClickRelease) {
        match release {
            ClickRelease::Keep => {}
            ClickRelease::Deselect => {
                self.selected.remove(&image_id);
            }
        }
    }

    /// A press on empty world clears the selection unless Shift extends it.
    fn press_empty(&mut self, extend: bool) {
        if !extend {
            self.selected.clear();
        }
    }

    /// A free left-drag of the selection is currently live; the scroll wheel
    /// then belongs to drag distance, not fly speed. (An axis-gizmo drag
    /// does not use the wheel, so it leaves fly speed alone.)
    pub fn drag_active(&self) -> bool {
        self.press
            .as_ref()
            .is_some_and(|press| matches!(press.kind, PressKind::Billboard { dragging: true, .. }))
    }
}

/// Marker for the translate gizmo's root entity.
#[derive(Component)]
pub struct TranslateGizmoRoot;

/// Marker for the overlay camera that draws the gizmo's render layer into
/// the scene render target on top of the already-rendered scene.
#[derive(Component)]
pub struct TranslateGizmoCamera;

/// The world-space translate gizmo shown on the current selection. `origin`
/// and `scale` mirror the spawned root's transform so press hit-testing can
/// run from data, without querying the arrow entities.
#[derive(Resource, Default)]
pub struct TranslateGizmo {
    root: Option<Entity>,
    camera: Option<Entity>,
    origin: Vec3,
    scale: f32,
}

impl TranslateGizmo {
    /// The arrow under the ray while the gizmo is shown. The gizmo draws
    /// over the whole scene, so every press handler asks this first.
    pub(crate) fn axis_under_ray(&self, ray_origin: Vec3, ray_direction: Vec3) -> Option<usize> {
        self.root?;
        hit_translate_gizmo_axis(self.origin, self.scale, ray_origin, ray_direction)
    }
}

/// Marker for the accent quad parented under each selected billboard.
#[derive(Component)]
pub struct SelectionHighlight;

/// Highlight entities keyed by image id, plus the one mesh/material every
/// highlight quad shares.
#[derive(Resource, Default)]
pub struct SelectionHighlights {
    by_image_id: HashMap<usize, Entity>,
    mesh: Option<Handle<Mesh>>,
    material: Option<Handle<StandardMaterial>>,
}

impl SelectionHighlights {
    /// A unit quad, scaled per highlight to its picture, and the accent
    /// material.
    fn shared_assets(
        &mut self,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
    ) -> (Handle<Mesh>, Handle<StandardMaterial>) {
        let mesh = self
            .mesh
            .get_or_insert_with(|| meshes.add(Rectangle::new(1.0, 1.0)))
            .clone();
        let material = self
            .material
            .get_or_insert_with(|| {
                materials.add(StandardMaterial {
                    base_color: HIGHLIGHT_COLOR,
                    alpha_mode: AlphaMode::Blend,
                    cull_mode: None,
                    unlit: true,
                    ..default()
                })
            })
            .clone();
        (mesh, material)
    }
}

/// Image id -> index into `ExplorerScene::image_points`, rebuilt whenever a
/// projection is (re)loaded, so drag edits write the canonical point store
/// without a linear scan per moved item.
#[derive(Resource, Default)]
pub struct ImagePointIndex(HashMap<usize, usize>);

impl ImagePointIndex {
    pub fn new(points: &[BillboardPoint]) -> Self {
        Self(
            points
                .iter()
                .enumerate()
                .map(|(index, point)| (point.image_id, index))
                .collect(),
        )
    }

    fn get(&self, image_id: usize) -> Option<usize> {
        self.0.get(&image_id).copied()
    }
}

/// Every flat position store a drag edit must keep coherent.
#[derive(SystemParam)]
pub struct SelectionDragStores<'w> {
    scene: ResMut<'w, ExplorerScene>,
    index: Res<'w, ImagePointIndex>,
    loading: ResMut<'w, ImageLoadingState>,
    cloud: ResMut<'w, PointCloud>,
    navigation_targets: ResMut<'w, NavigationTargets>,
    catalog_load_task: ResMut<'w, CatalogLoadTask>,
}

/// Bundled so `handle_selection_and_drag` stays within Bevy's supported
/// system-function parameter arity.
#[derive(SystemParam)]
pub struct SelectionDragQueries<'w, 's> {
    window: Query<'w, 's, &'static Window, With<PrimaryWindow>>,
    camera: Query<
        'w,
        's,
        (
            &'static Camera,
            &'static GlobalTransform,
            &'static Transform,
        ),
        With<FlyCamera>,
    >,
    billboards: Query<
        'w,
        's,
        (
            &'static MediaBillboard,
            &'static GlobalTransform,
            &'static Visibility,
        ),
    >,
    billboard_transforms:
        Query<'w, 's, (&'static MediaBillboard, &'static mut Transform), Without<FlyCamera>>,
    video_strips: VideoStripHitQuery<'w, 's>,
}

/// Left-click selection and drag-to-move. A press over a billboard selects
/// it following the conventions in the module docs (Shift extends), a press
/// over empty world clears the selection, and holding the press past
/// the motion threshold drags every selected item rigidly around the player,
/// glued to the cursor's world ray, with player flight carrying the
/// selection and the scroll wheel controlling its distance. A press on a
/// translate-gizmo arrow instead starts a world-axis-constrained drag (grid
/// stepped when "Snap drag to grid" is on). Presses on a video control strip
/// never reach here: screen strips are UI and world strips are hit-tested
/// first.
#[allow(clippy::too_many_arguments)]
pub fn handle_selection_and_drag(
    real_time: Res<Time<Real>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    pause_menu: Res<PauseMenuState>,
    ui_capture: Res<UiInputCapture>,
    render_resolution: Res<RenderResolutionSettings>,
    facing: Res<BillboardFacingSettings>,
    billboard_controls: Res<BillboardControls>,
    gizmo: Res<TranslateGizmo>,
    mut selection: ResMut<SelectionState>,
    mut stores: SelectionDragStores,
    mut queries: SelectionDragQueries,
) {
    let selection = &mut *selection;
    if pause_menu.paused {
        mouse_motion.clear();
        mouse_wheel.clear();
        selection.press = None;
        return;
    }

    // Release first: a click settles the selection (see `ClickRelease`),
    // a finished free drag optionally snaps the dropped items to the
    // image-size grid, and any move publishes the new positions to the
    // teleport targets. An axis drag already stepped on the grid live.
    if !mouse_buttons.pressed(MouseButton::Left) {
        if let Some(press) = selection.press.take() {
            if let PressKind::Billboard {
                image_id,
                release,
                pressed_at_seconds,
                dragging,
                ..
            } = press.kind
            {
                let held = real_time.elapsed_secs() - pressed_at_seconds >= CLICK_HOLD_SECONDS;
                if !dragging && !held {
                    selection.release_click(image_id, release);
                }
                if press.moved_any && billboard_controls.snap_to_grid {
                    snap_selection_to_grid(selection, &mut stores, &mut queries, facing.axis);
                }
            }
            if press.moved_any {
                stores.navigation_targets.replace_positions(
                    stores.scene.image_points.iter().map(|point| point.position),
                );
            }
        }
        mouse_motion.clear();
        return;
    }

    if mouse_buttons.just_pressed(MouseButton::Left) {
        mouse_motion.clear();
        // A click the UI owns or a click inside the axis gizmo canvas must
        // neither select nor clear.
        if ui_capture.blocks_world_clicks() {
            return;
        }
        let Ok(window) = queries.window.get_single() else {
            return;
        };
        if cursor_over_axis_gizmo(window) {
            return;
        }
        let Ok((camera, camera_global, camera_transform)) = queries.camera.get_single() else {
            return;
        };
        let Some((ray_origin, ray_direction)) =
            cursor_world_ray(window, camera, camera_global, &render_resolution)
        else {
            return;
        };
        // The translate gizmo draws over everything around the selection,
        // so its arrows win the press over strips and billboards alike.
        // Hit-tested with the same (lagged) camera pose the gizmo was
        // rendered with.
        if !selection.selected.is_empty() {
            if let Some(axis_index) = gizmo.axis_under_ray(ray_origin, ray_direction) {
                let Some(cursor_position) = cursor_render_position(window, &render_resolution)
                else {
                    return;
                };
                selection.press = Some(PressState {
                    kind: PressKind::GizmoAxis {
                        axis_index,
                        origin: gizmo.origin,
                        last_cursor: cursor_position,
                        raw_offset: 0.0,
                        applied_offset: 0.0,
                    },
                    moved_any: false,
                });
                return;
            }
        }
        let billboard_hit = nearest_billboard_hit(
            ray_origin,
            ray_direction,
            stores.scene.billboard_world_size,
            &queries.billboards,
        );
        // Presses on a visible video control strip belong to playback, which
        // already acted on them this frame, and never touch the selection.
        let occluder_distance = billboard_hit.map(|hit| hit.distance);
        if nearest_strip_hit(
            ray_origin,
            ray_direction,
            &queries.video_strips,
            occluder_distance,
        )
        .is_some()
        {
            return;
        }
        let extend = keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
        match billboard_hit.map(|hit| hit.image_id) {
            Some(image_id) => {
                // The drag anchor lives in the fresh camera pose so the
                // first drag frame starts without the propagation lag.
                let Some((_, anchor_direction)) = cursor_world_ray(
                    window,
                    camera,
                    &GlobalTransform::from(*camera_transform),
                    &render_resolution,
                ) else {
                    return;
                };
                let release = selection.press_billboard(image_id, extend);
                selection.press = Some(PressState {
                    kind: PressKind::Billboard {
                        image_id,
                        release,
                        pressed_at_seconds: real_time.elapsed_secs(),
                        accumulated_motion: 0.0,
                        dragging: false,
                        drag_anchor_direction: anchor_direction,
                        last_cursor_direction: anchor_direction,
                        last_camera_position: camera_transform.translation,
                    },
                    moved_any: false,
                });
            }
            None => selection.press_empty(extend),
        }
        return;
    }

    let Some(press) = selection.press.as_mut() else {
        mouse_motion.clear();
        mouse_wheel.clear();
        return;
    };
    let Ok((camera, _, camera_transform)) = queries.camera.get_single() else {
        return;
    };
    let camera_position = camera_transform.translation;
    let fresh_camera_pose = GlobalTransform::from(*camera_transform);
    let camera_up = camera_transform.rotation.mul_vec3(Vec3::Y);
    let viewport_normal = camera_transform.rotation.mul_vec3(Vec3::Z);

    match &mut press.kind {
        PressKind::Billboard {
            accumulated_motion,
            dragging,
            drag_anchor_direction,
            last_cursor_direction,
            last_camera_position,
            ..
        } => {
            if !*dragging {
                // Flight before the drag threshold is crossed must not later
                // count as a carry displacement, so keep the anchor fresh.
                *last_camera_position = camera_position;
                *accumulated_motion += mouse_motion
                    .read()
                    .map(|motion| motion.delta.length())
                    .sum::<f32>();
                if *accumulated_motion < SELECTION_DRAG_MOTION_THRESHOLD {
                    return;
                }
                *dragging = true;
            }
            if selection.selected.is_empty() {
                return;
            }

            // The mouse's contribution comes from the cursor ray, not
            // motion deltas.
            mouse_motion.clear();
            let scroll_steps: f32 = mouse_wheel.read().map(|wheel| wheel.y).sum();
            let distance_scale = DRAG_DISTANCE_SCROLL_FACTOR.powf(scroll_steps);
            // Flying while dragging carries the selection: every item keeps
            // its camera-relative offset, which adds the player's velocity
            // to it.
            let player_delta = camera_position - *last_camera_position;
            *last_camera_position = camera_position;

            // Rotate the selection by the arc the cursor ray swept since
            // last frame; while the cursor is outside the window the ray is
            // unavailable, so the stored direction holds and the selection
            // re-syncs on return. The ray is built from the camera's fresh
            // `Transform` — its `GlobalTransform` is only propagated in
            // PostUpdate, and anchoring to that lagged pose made the
            // selection stutter one frame behind view movement mid-drag.
            let rotation = match queries.window.get_single().ok().and_then(|window| {
                cursor_world_ray(window, camera, &fresh_camera_pose, &render_resolution)
            }) {
                Some((_, cursor_direction)) if cursor_direction != *last_cursor_direction => {
                    // Re-anchor before the anchor arc nears 180°, where
                    // `from_rotation_arc` degenerates; path independence
                    // is only interrupted across this rare boundary.
                    if drag_anchor_direction.dot(cursor_direction) < 0.0 {
                        *drag_anchor_direction = *last_cursor_direction;
                    }
                    let rotation = cursor_drag_rotation(
                        *drag_anchor_direction,
                        *last_cursor_direction,
                        cursor_direction,
                    );
                    *last_cursor_direction = cursor_direction;
                    rotation
                }
                // A still cursor (or one outside the window) contributes
                // no rotation; flight carry and scroll still apply.
                _ => Quat::IDENTITY,
            };

            if rotation == Quat::IDENTITY && player_delta == Vec3::ZERO && scroll_steps == 0.0 {
                return;
            }

            let min_distance = stores.scene.billboard_world_size;
            for (billboard, mut transform) in &mut queries.billboard_transforms {
                if !selection.selected.contains(&billboard.image_id) {
                    continue;
                }
                // Adding `player_delta` first keeps the offset the item had
                // relative to the camera before this frame's flight.
                let carried_offset = transform.translation + player_delta - camera_position;
                let offset =
                    scale_offset_depth(rotation * carried_offset, distance_scale, min_distance);
                let position = camera_position + offset;
                if position.distance_squared(transform.translation) <= f32::EPSILON {
                    continue;
                }
                place_item(
                    &mut transform,
                    billboard.image_id,
                    position,
                    camera_position,
                    viewport_normal,
                    camera_up,
                    facing.axis,
                    &mut stores,
                );
                press.moved_any = true;
            }
        }
        PressKind::GizmoAxis {
            axis_index,
            origin,
            last_cursor,
            raw_offset,
            applied_offset,
        } => {
            mouse_motion.clear();
            if selection.selected.is_empty() {
                return;
            }
            let Some(cursor_position) = queries
                .window
                .get_single()
                .ok()
                .and_then(|window| cursor_render_position(window, &render_resolution))
            else {
                return;
            };
            let cursor_delta = cursor_position - *last_cursor;
            *last_cursor = cursor_position;
            if cursor_delta == Vec2::ZERO {
                return;
            }
            // Only cursor travel along the axis's on-screen direction moves
            // the selection; the world amount comes from how many world
            // units one on-screen pixel of the axis represents right now.
            let axis = WORLD_AXES[*axis_index];
            let anchor = *origin + axis * *applied_offset;
            let probe_length = (gizmo.scale * TRANSLATE_GIZMO_ARROW_LENGTH).max(f32::EPSILON);
            let Some((screen_direction, units_per_pixel)) =
                axis_screen_mapping(camera, &fresh_camera_pose, anchor, axis, probe_length)
            else {
                return;
            };
            *raw_offset += cursor_delta.dot(screen_direction) * units_per_pixel;
            let snap_cell = billboard_controls
                .snap_to_grid
                .then_some(stores.scene.billboard_world_size);
            let target_offset = axis_drag_offset(*raw_offset, origin[*axis_index], snap_cell);
            let step = target_offset - *applied_offset;
            if step.abs() <= f32::EPSILON {
                return;
            }
            *applied_offset = target_offset;
            let delta = axis * step;

            for (billboard, mut transform) in &mut queries.billboard_transforms {
                if !selection.selected.contains(&billboard.image_id) {
                    continue;
                }
                let position = transform.translation + delta;
                place_item(
                    &mut transform,
                    billboard.image_id,
                    position,
                    camera_position,
                    viewport_normal,
                    camera_up,
                    facing.axis,
                    &mut stores,
                );
                press.moved_any = true;
            }
        }
    }
}

/// Writes a manually placed position into the billboard entity and every
/// flat store that mirrors it, re-facing the billboard toward the camera
/// (the facing system skips work while the camera holds still, so a moved
/// billboard has to be re-faced here).
#[allow(clippy::too_many_arguments)]
fn place_item(
    transform: &mut Transform,
    image_id: usize,
    position: Vec3,
    camera_position: Vec3,
    viewport_normal: Vec3,
    camera_up: Vec3,
    facing_axis: BillboardFacingAxis,
    stores: &mut SelectionDragStores,
) {
    transform.translation = position;
    let to_camera = camera_position - position;
    if to_camera.length_squared() > f32::EPSILON {
        transform.rotation = billboard_rotation(
            to_camera.normalize(),
            viewport_normal,
            camera_up,
            facing_axis,
        );
    }

    stores.loading.update_loaded_position(image_id, position);
    stores.cloud.set_point_position(image_id, position);
    stores
        .catalog_load_task
        .dragged_positions
        .insert(image_id, position);
    if let Some(point_index) = stores.index.get(image_id) {
        if let Some(point) = stores.scene.image_points.get_mut(point_index) {
            if point.image_id == image_id {
                point.position = position;
            }
        }
    }
}

/// Snaps every selected billboard onto the nearest point of the image-size
/// grid when a free drag is released with "Snap drag to grid" enabled. The
/// cell is the billboard's world size — the displayed image square — so any
/// two separately dropped items land exactly flush against one another.
fn snap_selection_to_grid(
    selection: &SelectionState,
    stores: &mut SelectionDragStores,
    queries: &mut SelectionDragQueries,
    facing_axis: BillboardFacingAxis,
) {
    let Ok((_, _, camera_transform)) = queries.camera.get_single() else {
        return;
    };
    let camera_position = camera_transform.translation;
    let camera_up = camera_transform.rotation.mul_vec3(Vec3::Y);
    let viewport_normal = camera_transform.rotation.mul_vec3(Vec3::Z);
    let spacing = stores.scene.billboard_world_size;

    for (billboard, mut transform) in &mut queries.billboard_transforms {
        if !selection.selected.contains(&billboard.image_id) {
            continue;
        }
        let snapped = snap_position_to_grid(transform.translation, spacing);
        if snapped == transform.translation {
            continue;
        }
        place_item(
            &mut transform,
            billboard.image_id,
            snapped,
            camera_position,
            viewport_normal,
            camera_up,
            facing_axis,
            stores,
        );
    }
}

/// Nearest lattice point of the `spacing`-sized coordinate grid.
fn snap_position_to_grid(position: Vec3, spacing: f32) -> Vec3 {
    if spacing <= f32::EPSILON {
        return position;
    }
    (position / spacing).round() * spacing
}

/// Target offset of a gizmo axis drag: the raw cursor-derived offset or,
/// with grid snapping, the offset that lands the gizmo origin's coordinate
/// on the image-size grid — so an item stepped along an axis sits flush
/// against any other grid-placed item.
fn axis_drag_offset(raw_offset: f32, origin_coordinate: f32, snap_cell: Option<f32>) -> f32 {
    match snap_cell {
        Some(cell) if cell > f32::EPSILON => {
            ((origin_coordinate + raw_offset) / cell).round() * cell - origin_coordinate
        }
        _ => raw_offset,
    }
}

/// Parameter `s` of the point on the line `origin + s * axis` closest to
/// the cursor ray (`axis` and `ray_direction` unit length); `None` when the
/// axis is (nearly) parallel to the ray and every point is equally close.
fn closest_axis_parameter(
    origin: Vec3,
    axis: Vec3,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<f32> {
    let to_origin = origin - ray_origin;
    let alignment = axis.dot(ray_direction);
    let denominator = 1.0 - alignment * alignment;
    if denominator <= 1.0e-5 {
        return None;
    }
    let along_axis = axis.dot(to_origin);
    let along_ray = ray_direction.dot(to_origin);
    Some((alignment * along_ray - along_axis) / denominator)
}

/// Which arrow (index into [`WORLD_AXES`]) of the translate gizmo the cursor
/// ray grabs, if any: the ray must pass within the grab radius of an arrow's
/// segment; the closest approach wins.
fn hit_translate_gizmo_axis(
    origin: Vec3,
    scale: f32,
    ray_origin: Vec3,
    ray_direction: Vec3,
) -> Option<usize> {
    let grab_radius = scale * TRANSLATE_GIZMO_GRAB_RADIUS;
    let arrow_length = scale * TRANSLATE_GIZMO_ARROW_LENGTH;
    let mut best: Option<(usize, f32)> = None;
    for (axis_index, axis) in WORLD_AXES.iter().enumerate() {
        let Some(parameter) = closest_axis_parameter(origin, *axis, ray_origin, ray_direction)
        else {
            continue;
        };
        let arrow_point = origin + *axis * parameter.clamp(0.0, arrow_length);
        let to_point = arrow_point - ray_origin;
        let along_ray = to_point.dot(ray_direction).max(0.0);
        let distance = (to_point - ray_direction * along_ray).length();
        if distance <= grab_radius && best.is_none_or(|(_, best_distance)| distance < best_distance)
        {
            best = Some((axis_index, distance));
        }
    }
    best.map(|(axis_index, _)| axis_index)
}

type GizmoCameraQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Transform, &'static Projection),
    (
        With<FlyCamera>,
        Without<TranslateGizmoRoot>,
        Without<TranslateGizmoCamera>,
    ),
>;
type GizmoBillboardQuery<'w, 's> = Query<
    'w,
    's,
    (&'static MediaBillboard, &'static Transform),
    (
        Without<TranslateGizmoRoot>,
        Without<TranslateGizmoCamera>,
        Without<FlyCamera>,
    ),
>;
type GizmoRootQuery<'w, 's> = Query<
    'w,
    's,
    &'static mut Transform,
    (With<TranslateGizmoRoot>, Without<TranslateGizmoCamera>),
>;
type GizmoOverlayCameraQuery<'w, 's> = Query<
    'w,
    's,
    (&'static mut Transform, &'static mut Projection),
    (
        With<TranslateGizmoCamera>,
        Without<TranslateGizmoRoot>,
        Without<FlyCamera>,
    ),
>;

/// Keeps the translate gizmo attached to the selection: spawned while any
/// selected billboard is loaded, positioned at the selection's centroid, and
/// scaled with camera distance for a constant on-screen size. Despawned when
/// the selection empties (including on projection reloads, which reset
/// [`SelectionState`]). Also owns the overlay camera that renders the gizmo
/// layer into the scene render target on top of the scene, keeping its pose
/// and projection locked to the fly camera's.
#[allow(clippy::too_many_arguments)]
pub fn sync_translate_gizmo(
    mut commands: Commands,
    selection: Res<SelectionState>,
    mut gizmo: ResMut<TranslateGizmo>,
    render_target: Res<SceneRenderTarget>,
    camera_query: GizmoCameraQuery,
    billboards: GizmoBillboardQuery,
    mut roots: GizmoRootQuery,
    mut overlay_cameras: GizmoOverlayCameraQuery,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut centroid_sum = Vec3::ZERO;
    let mut selected_count = 0;
    if !selection.selected.is_empty() {
        for (billboard, transform) in &billboards {
            if selection.selected.contains(&billboard.image_id) {
                centroid_sum += transform.translation;
                selected_count += 1;
            }
        }
    }
    if selected_count == 0 {
        if let Some(root) = gizmo.root.take() {
            commands.entity(root).despawn_recursive();
        }
        if let Some(camera) = gizmo.camera.take() {
            commands.entity(camera).despawn_recursive();
        }
        return;
    }
    let Ok((camera_transform, camera_projection)) = camera_query.get_single() else {
        return;
    };

    let origin = centroid_sum / selected_count as f32;
    let scale = (camera_transform.translation.distance(origin) * TRANSLATE_GIZMO_SCREEN_FACTOR)
        .max(f32::EPSILON);
    gizmo.origin = origin;
    gizmo.scale = scale;

    if let Some(root) = gizmo.root {
        // The root spawned via commands last frame may not be queryable for
        // one frame; skip the update rather than spawning a duplicate.
        if let Ok(mut transform) = roots.get_mut(root) {
            transform.translation = origin;
            transform.scale = Vec3::splat(scale);
        }
    } else {
        gizmo.root = Some(spawn_translate_gizmo(
            &mut commands,
            &mut meshes,
            &mut materials,
            origin,
            scale,
        ));
    }

    if let Some(camera) = gizmo.camera {
        if let Ok((mut transform, mut projection)) = overlay_cameras.get_mut(camera) {
            *transform = *camera_transform;
            // Only the far plane of the fly camera's projection ever changes
            // (scene reloads); the aspect ratio tracks the shared render
            // target automatically on both cameras.
            if let (Projection::Perspective(fly), Projection::Perspective(overlay)) =
                (camera_projection, &mut *projection)
            {
                if overlay.far != fly.far || overlay.fov != fly.fov || overlay.near != fly.near {
                    overlay.far = fly.far;
                    overlay.fov = fly.fov;
                    overlay.near = fly.near;
                }
            }
        }
    } else {
        gizmo.camera = Some(spawn_translate_gizmo_camera(
            &mut commands,
            render_target.image.clone(),
            *camera_transform,
            camera_projection.clone(),
        ));
    }
}

/// Spawns the overlay camera: same pose and projection as the fly camera,
/// rendering only the gizmo's layer into the scene render target after the
/// main camera, with its own depth buffer and no clear — so the gizmo draws
/// on top of everything the scene already drew.
fn spawn_translate_gizmo_camera(
    commands: &mut Commands,
    render_target_image: Handle<Image>,
    camera_transform: Transform,
    projection: Projection,
) -> Entity {
    commands
        .spawn((
            Camera3d::default(),
            Camera {
                order: -1,
                target: RenderTarget::Image(render_target_image),
                clear_color: ClearColorConfig::None,
                ..default()
            },
            projection,
            Msaa::Off,
            camera_transform,
            RenderLayers::layer(TRANSLATE_GIZMO_RENDER_LAYER),
            TranslateGizmoCamera,
            Name::new("translate gizmo overlay camera"),
        ))
        .id()
}

/// Spawns the translate gizmo: three unit-length world-axis arrows (shaft
/// plus cone head) under one root that carries the world position and the
/// distance-dependent scale.
fn spawn_translate_gizmo(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    origin: Vec3,
    scale: f32,
) -> Entity {
    let shaft_mesh = meshes.add(Cuboid::new(
        TRANSLATE_GIZMO_SHAFT_THICKNESS,
        TRANSLATE_GIZMO_SHAFT_LENGTH,
        TRANSLATE_GIZMO_SHAFT_THICKNESS,
    ));
    let head_mesh = meshes.add(Cone {
        radius: TRANSLATE_GIZMO_HEAD_RADIUS,
        height: TRANSLATE_GIZMO_HEAD_LENGTH,
    });
    // Arrows are authored pointing +Y, then rotated onto their world axis.
    let axis_rotations = [
        Quat::from_rotation_z(-FRAC_PI_2),
        Quat::IDENTITY,
        Quat::from_rotation_x(FRAC_PI_2),
    ];

    commands
        .spawn((
            Transform {
                translation: origin,
                scale: Vec3::splat(scale),
                ..default()
            },
            Visibility::default(),
            TranslateGizmoRoot,
            Name::new("translate gizmo"),
        ))
        .with_children(|root| {
            for (axis_index, rotation) in axis_rotations.into_iter().enumerate() {
                let material = materials.add(StandardMaterial {
                    base_color: TRANSLATE_GIZMO_AXIS_COLORS[axis_index],
                    unlit: true,
                    ..default()
                });
                root.spawn((
                    Transform::from_rotation(rotation),
                    Visibility::default(),
                    Name::new(format!(
                        "translate gizmo {} arrow",
                        ['x', 'y', 'z'][axis_index]
                    )),
                ))
                .with_children(|arrow| {
                    // `RenderLayers` does not inherit, so every mesh entity
                    // carries the overlay layer itself.
                    arrow.spawn((
                        Mesh3d(shaft_mesh.clone()),
                        MeshMaterial3d(material.clone()),
                        Transform::from_xyz(0.0, TRANSLATE_GIZMO_SHAFT_LENGTH * 0.5, 0.0),
                        RenderLayers::layer(TRANSLATE_GIZMO_RENDER_LAYER),
                    ));
                    arrow.spawn((
                        Mesh3d(head_mesh.clone()),
                        MeshMaterial3d(material),
                        Transform::from_xyz(
                            0.0,
                            TRANSLATE_GIZMO_SHAFT_LENGTH + TRANSLATE_GIZMO_HEAD_LENGTH * 0.5,
                            0.0,
                        ),
                        RenderLayers::layer(TRANSLATE_GIZMO_RENDER_LAYER),
                    ));
                });
            }
        })
        .id()
}

/// This frame's rigid rotation of the dragged selection around the player:
/// the delta between the total arcs from the drag anchor to the current and
/// previous cursor rays. It still carries the previous ray exactly onto the
/// current one (1:1 cursor sync, no sensitivity constant, immune to FOV,
/// window size, and render-target scale), but because the per-frame deltas
/// telescope to `from_rotation_arc(anchor, current)`, the accumulated
/// rotation is path-independent and twist-free — raw frame-to-frame arcs
/// pick up roll along curved cursor paths, which orbited off-center items
/// around the grabbed one on diagonal drags. A pure rotation, so every
/// item's depth is preserved exactly.
fn cursor_drag_rotation(
    anchor_direction: Vec3,
    previous_direction: Vec3,
    current_direction: Vec3,
) -> Quat {
    Quat::from_rotation_arc(anchor_direction, current_direction)
        * Quat::from_rotation_arc(anchor_direction, previous_direction).inverse()
}

/// Rescales a camera-relative offset's length by `distance_scale` for the
/// scroll-wheel distance control. The depth never drops below `min_distance`
/// — and an item already inside it only holds or grows — so the selection
/// cannot be scrolled into the player.
fn scale_offset_depth(offset: Vec3, distance_scale: f32, min_distance: f32) -> Vec3 {
    if distance_scale == 1.0 {
        return offset;
    }
    let depth = offset.length();
    if depth <= f32::EPSILON {
        return offset;
    }
    let scaled_depth = (depth * distance_scale).max(min_distance.min(depth));
    offset * (scaled_depth / depth)
}

/// A cursor ray landing on a billboard.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BillboardHit {
    pub image_id: usize,
    /// World distance from the ray origin, for occlusion against other
    /// hit-testable geometry.
    pub distance: f32,
}

/// Nearest visible billboard (image or video) whose picture is under the
/// cursor ray. The see-through letterbox padding around a picture is not
/// part of it, so it neither takes the pointer nor shields what lies behind.
pub(crate) fn nearest_billboard_hit(
    ray_origin: Vec3,
    ray_direction: Vec3,
    billboard_size: f32,
    billboards: &Query<(&MediaBillboard, &GlobalTransform, &Visibility)>,
) -> Option<BillboardHit> {
    billboards
        .iter()
        .filter(|(_, _, visibility)| **visibility != Visibility::Hidden)
        .filter_map(|(billboard, transform, _)| {
            let half_extents = billboard.content_half_extents(billboard_size);
            let (distance, _) = ray_rect_hit(ray_origin, ray_direction, transform, half_extents)?;
            Some(BillboardHit {
                image_id: billboard.image_id,
                distance,
            })
        })
        .min_by(|left, right| left.distance.total_cmp(&right.distance))
}

/// World-space cursor ray through `camera_pose`. Hit tests pass the camera's
/// propagated `GlobalTransform` so the ray lives in the same (one frame
/// lagged) space as the billboards' own global transforms; the drag anchor
/// instead passes a pose built from the camera's fresh `Transform`, because
/// anchoring to the lagged pose made the selection visibly stutter one frame
/// behind any view movement during a drag.
pub(crate) fn cursor_world_ray(
    window: &Window,
    camera: &Camera,
    camera_pose: &GlobalTransform,
    render_resolution: &RenderResolutionSettings,
) -> Option<(Vec3, Vec3)> {
    let render_position = cursor_render_position(window, render_resolution)?;
    let ray = camera
        .viewport_to_world(camera_pose, render_position)
        .ok()?;
    Some((ray.origin, *ray.direction))
}

/// Cursor position in render-target pixels, `None` while the cursor is
/// outside the window.
fn cursor_render_position(
    window: &Window,
    render_resolution: &RenderResolutionSettings,
) -> Option<Vec2> {
    cursor_to_render_target_position(
        window,
        window.cursor_position()?,
        render_resolution.target_size(window.resolution.physical_size()),
    )
}

/// On-screen direction (render-target pixels, unit length) of `axis` seen
/// from the camera at `anchor`, and how many world units one pixel of cursor
/// travel along it is worth. `None` when the axis projects to almost nothing
/// on screen — pointing (nearly) straight at or away from the viewer — where
/// no meaningful drag direction exists.
fn axis_screen_mapping(
    camera: &Camera,
    camera_pose: &GlobalTransform,
    anchor: Vec3,
    axis: Vec3,
    probe_length: f32,
) -> Option<(Vec2, f32)> {
    let anchor_screen = camera.world_to_viewport(camera_pose, anchor).ok()?;
    let probe_screen = camera
        .world_to_viewport(camera_pose, anchor + axis * probe_length)
        .ok()?;
    let screen_step = probe_screen - anchor_screen;
    let screen_length = screen_step.length();
    if screen_length < TRANSLATE_GIZMO_MIN_SCREEN_LENGTH {
        return None;
    }
    Some((screen_step / screen_length, probe_length / screen_length))
}

/// Keeps one highlight quad parented under every selected, loaded billboard:
/// spawns quads for newly selected or freshly (re)loaded billboards, drops
/// them on deselection, and forgets entries whose quad died with an evicted
/// billboard so a later reload gets its highlight back.
#[allow(clippy::too_many_arguments)]
pub fn sync_selection_highlights(
    mut commands: Commands,
    selection: Res<SelectionState>,
    billboard_size: Res<BillboardWorldSize>,
    mut highlights: ResMut<SelectionHighlights>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    billboards: Query<(Entity, &MediaBillboard)>,
    live_highlights: Query<(), With<SelectionHighlight>>,
) {
    if selection.selected.is_empty() && highlights.by_image_id.is_empty() {
        return;
    }

    highlights
        .by_image_id
        .retain(|_, entity| live_highlights.get(*entity).is_ok());

    let deselected: Vec<usize> = highlights
        .by_image_id
        .keys()
        .filter(|image_id| !selection.selected.contains(image_id))
        .copied()
        .collect();
    for image_id in deselected {
        if let Some(entity) = highlights.by_image_id.remove(&image_id) {
            commands.entity(entity).despawn_recursive();
        }
    }

    if selection.selected.is_empty() {
        return;
    }
    for (entity, billboard) in &billboards {
        if !selection.selected.contains(&billboard.image_id)
            || highlights.by_image_id.contains_key(&billboard.image_id)
        {
            continue;
        }
        let (mesh, material) = highlights.shared_assets(&mut meshes, &mut materials);
        // Rims the picture, not the letterboxed square: the see-through
        // padding is not part of what was selected.
        let size = billboard.content_half_extents(billboard_size.0) * 2.0
            + Vec2::splat(billboard_size.0 * HIGHLIGHT_RIM_FRACTION);
        let highlight = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(material),
                Transform::from_xyz(0.0, 0.0, -billboard_size.0 * HIGHLIGHT_BEHIND_OFFSET_FACTOR)
                    .with_scale(size.extend(1.0)),
                SelectionHighlight,
                Name::new("selection highlight"),
            ))
            .set_parent(entity)
            .id();
        highlights.by_image_id.insert(billboard.image_id, highlight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_drag_rotation_keeps_grabbed_item_under_cursor() {
        // An item sitting on the previous cursor ray lands exactly on the
        // current one, at unchanged depth — the 1:1 sync guarantee — even
        // when the drag anchor lies elsewhere.
        let anchor = Vec3::NEG_Z;
        let previous = Vec3::new(0.1, -0.05, -1.0).normalize();
        let current = Vec3::new(-0.2, 0.15, -1.0).normalize();
        let grabbed = previous * 12.5;

        let moved = cursor_drag_rotation(anchor, previous, current) * grabbed;

        assert!(moved.normalize().dot(current) > 0.9999);
        assert!((moved.length() - grabbed.length()).abs() < 0.0001);
    }

    #[test]
    fn cursor_drag_rotation_preserves_depth_and_relative_spacing() {
        // Rigid rotation: every item keeps its distance from the player and
        // from each other, so a dragged group can never stack.
        let anchor = Vec3::NEG_Z;
        let current = Vec3::new(0.3, 0.2, -1.0).normalize();
        let rotation = cursor_drag_rotation(anchor, anchor, current);
        let first = Vec3::new(0.0, 0.0, -10.0);
        let second = Vec3::new(2.0, 1.0, -9.0);

        assert!(((rotation * first).length() - first.length()).abs() < 0.0001);
        let separation = first.distance(second);
        assert!(((rotation * first).distance(rotation * second) - separation).abs() < 0.0001);
    }

    fn selection_of(ids: &[usize]) -> SelectionState {
        SelectionState {
            selected: ids.iter().copied().collect(),
            press: None,
        }
    }

    fn click(selection: &mut SelectionState, image_id: usize, extend: bool) {
        let release = selection.press_billboard(image_id, extend);
        selection.release_click(image_id, release);
    }

    fn selected_ids(selection: &SelectionState) -> Vec<usize> {
        let mut ids: Vec<usize> = selection.selected.iter().copied().collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn a_plain_click_selects_only_the_clicked_item() {
        let mut selection = selection_of(&[1, 2]);
        click(&mut selection, 3, false);
        assert_eq!(selected_ids(&selection), [3]);
    }

    #[test]
    fn a_click_on_a_selected_item_deselects_it() {
        let mut selection = selection_of(&[1]);
        click(&mut selection, 1, false);
        assert!(selected_ids(&selection).is_empty());

        let mut selection = selection_of(&[1, 2]);
        click(&mut selection, 2, false);
        assert_eq!(selected_ids(&selection), [1]);
    }

    #[test]
    fn pressing_a_selected_item_keeps_the_group_for_a_drag() {
        let mut selection = selection_of(&[1, 2]);
        selection.press_billboard(2, false);
        assert_eq!(selected_ids(&selection), [1, 2]);
        let mut selection = selection_of(&[1, 2]);
        selection.press_billboard(2, true);
        assert_eq!(selected_ids(&selection), [1, 2]);
    }

    #[test]
    fn shift_click_toggles_one_item() {
        let mut selection = selection_of(&[1]);
        click(&mut selection, 2, true);
        assert_eq!(selected_ids(&selection), [1, 2]);
        click(&mut selection, 1, true);
        assert_eq!(selected_ids(&selection), [2]);
    }

    #[test]
    fn empty_clicks_clear_unless_shift_is_held() {
        let mut selection = selection_of(&[1, 2]);
        selection.press_empty(true);
        assert_eq!(selected_ids(&selection), [1, 2]);
        selection.press_empty(false);
        assert!(selected_ids(&selection).is_empty());
    }

    #[test]
    fn cursor_drag_rotation_is_path_independent() {
        // Stepping through an intermediate direction composes to exactly the
        // direct anchor-to-final arc, so the accumulated rotation depends
        // only on where the cursor is, not on the path it took.
        let anchor = Vec3::NEG_Z;
        let via = Vec3::new(0.5, 0.0, -1.0).normalize();
        let final_direction = Vec3::new(0.5, 0.4, -1.0).normalize();
        let offset = Vec3::new(2.0, 1.0, -9.0);

        let stepped = cursor_drag_rotation(anchor, via, final_direction)
            * cursor_drag_rotation(anchor, anchor, via);
        let direct = cursor_drag_rotation(anchor, anchor, final_direction);

        assert!((stepped * offset - direct * offset).length() < 0.0001);
    }

    #[test]
    fn cursor_drag_rotation_accumulates_no_twist_on_closed_path() {
        // A round trip along a curved cursor path returns every item exactly
        // home; chaining raw frame-to-frame arcs instead leaves a residual
        // roll here (spherical holonomy) that visibly orbited off-center
        // items around the grabbed one.
        let anchor = Vec3::NEG_Z;
        let first_stop = Vec3::new(0.4, 0.0, -1.0).normalize();
        let second_stop = Vec3::new(0.4, 0.4, -1.0).normalize();
        let offset = Vec3::new(2.0, 1.0, -9.0);

        let round_trip = cursor_drag_rotation(anchor, second_stop, anchor)
            * cursor_drag_rotation(anchor, first_stop, second_stop)
            * cursor_drag_rotation(anchor, anchor, first_stop);

        assert!((round_trip * offset - offset).length() < 0.0001);
    }

    #[test]
    fn closest_axis_parameter_finds_point_under_ray() {
        // Ray dropping straight down from (5, 5, 0) meets the X axis at 5.
        let parameter =
            closest_axis_parameter(Vec3::ZERO, Vec3::X, Vec3::new(5.0, 5.0, 0.0), Vec3::NEG_Y);
        assert!((parameter.unwrap() - 5.0).abs() < 0.0001);

        // Anchoring the axis line elsewhere shifts the parameter.
        let parameter = closest_axis_parameter(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::X,
            Vec3::new(5.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        assert!((parameter.unwrap() - 3.0).abs() < 0.0001);

        // A ray parallel to the axis has no unique closest point.
        assert_eq!(
            closest_axis_parameter(Vec3::ZERO, Vec3::X, Vec3::new(0.0, 5.0, 0.0), Vec3::X),
            None
        );
    }

    #[test]
    fn axis_drag_offset_steps_onto_image_size_grid() {
        // An item on the 6.0 coordinate lattice dragged toward a neighbor at
        // the origin lands at exactly one image width (4.68): flush.
        let offset = axis_drag_offset(-0.9, 6.0, Some(4.68));
        assert!((6.0 + offset - 4.68).abs() < 0.0001);

        // Without snapping the raw offset passes through.
        assert_eq!(axis_drag_offset(-0.9, 6.0, None), -0.9);
        // A degenerate cell disables snapping instead of dividing by zero.
        assert_eq!(axis_drag_offset(-0.9, 6.0, Some(0.0)), -0.9);
    }

    #[test]
    fn gizmo_hit_grabs_the_pointed_at_arrow() {
        // Gizmo at the origin, scale 2 (arrow reach 2.0): a ray aimed at the
        // middle of the X arrow from the front hits it.
        let hit = hit_translate_gizmo_axis(Vec3::ZERO, 2.0, Vec3::new(1.2, 0.0, 10.0), Vec3::NEG_Z);
        assert_eq!(hit, Some(0));

        // Aimed above the Y arrow's reach: no grab.
        let miss =
            hit_translate_gizmo_axis(Vec3::ZERO, 2.0, Vec3::new(0.0, 3.5, 10.0), Vec3::NEG_Z);
        assert_eq!(miss, None);

        // Between arrows but outside every grab radius: no grab.
        let miss =
            hit_translate_gizmo_axis(Vec3::ZERO, 2.0, Vec3::new(1.2, 1.2, 10.0), Vec3::NEG_Z);
        assert_eq!(miss, None);
    }

    #[test]
    fn snap_position_to_grid_rounds_to_nearest_lattice_point() {
        assert_eq!(
            snap_position_to_grid(Vec3::new(7.1, -4.0, 0.4), 6.0),
            Vec3::new(6.0, -6.0, 0.0)
        );
        assert_eq!(
            snap_position_to_grid(Vec3::new(9.1, 3.1, -9.1), 6.0),
            Vec3::new(12.0, 6.0, -12.0)
        );
        // Degenerate spacing leaves positions untouched.
        let position = Vec3::new(1.2, 3.4, 5.6);
        assert_eq!(snap_position_to_grid(position, 0.0), position);
    }

    #[test]
    fn scale_offset_depth_scales_and_clamps() {
        let offset = Vec3::X * 10.0;
        assert!((scale_offset_depth(offset, 1.2, 2.0).length() - 12.0).abs() < 0.0001);
        assert!((scale_offset_depth(offset, 0.05, 2.0).length() - 2.0).abs() < 0.0001);
        assert_eq!(scale_offset_depth(offset, 1.0, 2.0), offset);
        assert_eq!(scale_offset_depth(Vec3::ZERO, 0.5, 2.0), Vec3::ZERO);
    }

    #[test]
    fn scale_offset_depth_inside_minimum_holds_or_grows() {
        let near = Vec3::X;
        assert_eq!(scale_offset_depth(near, 0.5, 2.0), near);
        assert!((scale_offset_depth(near, 3.0, 2.0).length() - 3.0).abs() < 0.0001);
    }

    #[test]
    fn image_point_index_maps_ids_to_indices() {
        let points = vec![
            BillboardPoint {
                image_id: 42,
                path: "".into(),
                position: Vec3::ZERO,
                is_video: false,
                duration_seconds: None,
                source_size: None,
                coordinate_labels: [None, None, None],
            },
            BillboardPoint {
                image_id: 7,
                path: "".into(),
                position: Vec3::ONE,
                is_video: false,
                duration_seconds: None,
                source_size: None,
                coordinate_labels: [None, None, None],
            },
        ];
        let index = ImagePointIndex::new(&points);
        assert_eq!(index.get(42), Some(0));
        assert_eq!(index.get(7), Some(1));
        assert_eq!(index.get(999), None);
    }
}
