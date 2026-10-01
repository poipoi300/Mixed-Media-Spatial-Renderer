//! Data-oriented point-cloud rendering for the projection markers.
//!
//! The previous implementation spawned one ECS entity (transform, visibility,
//! frustum culling, render extraction) per projected point, which collapsed
//! once catalogs reached tens of thousands of points. Here all point data
//! lives in flat arrays inside the [`PointCloud`] resource and the points are
//! baked into a small number of chunked meshes (one entity per spatial
//! chunk, vertex colors carry the per-point palette), so the per-frame ECS
//! and draw-call cost is proportional to the number of chunks, not points.
//!
//! On top of the chunks sits a fixed view boundary around the camera: chunks
//! beyond the budgeted radius are hidden entirely and opaque boundary walls
//! labelled "More points beyond this point" close the box, so the renderer
//! never pays for far geometry and the user still knows the data continues.

use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI};

use ab_glyph::FontArc;
use bevy::{
    math::primitives::Rectangle,
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
};

use crate::axis_gizmo::{render_label_text, LabelTextLine};
use crate::FlyCamera;

/// Rough number of points baked into one chunk mesh. Large enough that a
/// 100k-point catalog stays under ~100 draw calls, small enough that
/// rebuilding a chunk (hide/show on billboard load/evict) is cheap.
const POINT_CHUNK_TARGET_POINTS: usize = 1024;
/// Upper bound on chunk mesh rebuilds applied per frame so bursts of
/// billboard loads/evictions cannot monopolize a frame.
const POINT_CHUNK_REBUILDS_PER_FRAME: usize = 8;
/// Points baked into chunk meshes per frame. A chunk costs roughly 24
/// vertices per point on the CPU plus the GPU upload of the result, so this
/// keeps a freshly loaded 50k-point catalog spread over a dozen frames
/// instead of one.
const POINT_BAKES_PER_FRAME: usize = 4 * POINT_CHUNK_TARGET_POINTS;
/// Maximum number of points allowed inside the view boundary. The boundary
/// radius adapts so the visible chunk set stays under this budget.
const VISIBLE_POINT_BUDGET: usize = 60_000;
/// Camera movement (as a fraction of the chunk cell size) that triggers a
/// visibility recompute.
const VIEW_BOUNDS_RECOMPUTE_MOVE_FACTOR: f32 = 0.25;
/// The boundary never shrinks below this many chunk cells so the immediate
/// neighborhood of the camera is always rendered.
const VIEW_BOUNDS_MIN_CELL_RADIUS: f32 = 1.5;
/// Fraction of a boundary face's width the label text spans.
const WALL_TEXT_WIDTH_FRACTION: f32 = 0.55;
const WALL_TEXT_FONT_SIZE: f32 = 96.0;
const WALL_TEXT: &str = "More points beyond this point";
const WALL_FACE_COUNT: usize = 6;

/// One projected point in the cloud.
pub struct PointCloudPoint {
    pub image_id: usize,
    pub position: Vec3,
    pub color: [f32; 4],
}

/// Chunk partition of a point set, computed off the main thread so a scene
/// reload only has to spawn chunk entities and queue their meshes.
pub struct PointCloudLayout {
    positions: Vec<Vec3>,
    colors: Vec<[f32; 4]>,
    id_to_index: HashMap<usize, u32>,
    chunk_of_point: Vec<u32>,
    chunks: Vec<LayoutChunk>,
    cell_size: f32,
    point_size: f32,
}

struct LayoutChunk {
    point_indices: Vec<u32>,
    min: Vec3,
    max: Vec3,
}

impl PointCloudLayout {
    /// Partitions `points` into spatial chunks. Pure CPU work with no asset
    /// or ECS access, so it can run on the catalog load thread.
    pub fn new(points: Vec<PointCloudPoint>, point_size: f32) -> Self {
        let mut positions = Vec::with_capacity(points.len());
        let mut colors = Vec::with_capacity(points.len());
        let mut id_to_index = HashMap::with_capacity(points.len());
        for point in points {
            let index = positions.len() as u32;
            positions.push(point.position);
            colors.push(point.color);
            id_to_index.entry(point.image_id).or_insert(index);
        }
        let mut chunk_of_point = vec![0; positions.len()];

        let cell_size = chunk_cell_size(&positions, POINT_CHUNK_TARGET_POINTS);
        let mut cells: HashMap<IVec3, Vec<u32>> = HashMap::new();
        for (index, position) in positions.iter().enumerate() {
            let cell = (*position / cell_size).floor().as_ivec3();
            cells.entry(cell).or_default().push(index as u32);
        }

        let half_point = Vec3::splat(point_size * 0.5);
        let mut chunks = Vec::with_capacity(cells.len());
        for point_indices in cells.into_values() {
            let chunk_index = chunks.len() as u32;
            let mut min = Vec3::INFINITY;
            let mut max = Vec3::NEG_INFINITY;
            for &point in &point_indices {
                let position = positions[point as usize];
                min = min.min(position);
                max = max.max(position);
                chunk_of_point[point as usize] = chunk_index;
            }
            chunks.push(LayoutChunk {
                point_indices,
                min: min - half_point,
                max: max + half_point,
            });
        }
        Self {
            positions,
            colors,
            id_to_index,
            chunk_of_point,
            chunks,
            cell_size,
            point_size,
        }
    }
}

/// Marker for the per-chunk mesh entities owned by [`PointCloud`].
#[derive(Component)]
pub struct PointCloudChunkEntity;

struct PointChunk {
    entity: Entity,
    mesh: Handle<Mesh>,
    point_indices: Vec<u32>,
    min: Vec3,
    max: Vec3,
    visible_points: usize,
}

/// Flat, data-oriented storage for every projected point plus the chunked
/// meshes they are baked into.
#[derive(Resource, Default)]
pub struct PointCloud {
    positions: Vec<Vec3>,
    colors: Vec<[f32; 4]>,
    hidden: Vec<bool>,
    /// Points inside the dims-menu cutaway slice around the camera. Kept
    /// separate from `hidden` (owned by billboard load/evict) so the two
    /// hiding mechanisms restore independently; a point renders only when
    /// both flags are clear.
    sliced: Vec<bool>,
    /// Points held back for a moment while an animation makes room for
    /// them; restored independently of the other two flags too.
    withheld: Vec<bool>,
    id_to_index: HashMap<usize, u32>,
    chunk_of_point: Vec<u32>,
    chunks: Vec<PointChunk>,
    chunk_dirty: Vec<bool>,
    dirty_queue: Vec<u32>,
    point_size: f32,
    cell_size: f32,
    material: Option<Handle<StandardMaterial>>,
    /// Bumped whenever point visibility or the chunk set changes so the view
    /// bounds pass knows to recompute.
    epoch: u64,
}

impl PointCloud {
    pub fn point_count(&self) -> usize {
        self.positions.len()
    }

    /// Cell size of the chunk grid; the view boundary uses it as its unit.
    fn chunk_cell_size(&self) -> f32 {
        self.cell_size.max(1.0)
    }

    /// Hides or shows a single point (identified by its image id) by marking
    /// its chunk for a mesh rebuild. O(1) apart from the deferred rebuild.
    pub fn set_point_visible(&mut self, image_id: usize, visible: bool) {
        self.set_point_flag(image_id, |cloud| &mut cloud.hidden, !visible);
    }

    /// Holds a single point back, or lets it show again, without touching
    /// whether its billboard hides it.
    pub fn set_point_withheld(&mut self, image_id: usize, withheld: bool) {
        self.set_point_flag(image_id, |cloud| &mut cloud.withheld, withheld);
    }

    /// Sets one of a point's hiding flags, queueing its chunk for a rebake
    /// only when that changes whether the point renders.
    fn set_point_flag(
        &mut self,
        image_id: usize,
        flags: fn(&mut Self) -> &mut Vec<bool>,
        value: bool,
    ) {
        let Some(&index) = self.id_to_index.get(&image_id) else {
            return;
        };
        let index = index as usize;
        if flags(self)[index] == value {
            return;
        }
        let rendered = self.renders(index);
        flags(self)[index] = value;
        if self.renders(index) != rendered {
            self.point_rendering_changed(index);
        }
    }

    /// A point renders only while none of its hiding flags is set.
    fn renders(&self, index: usize) -> bool {
        !self.hidden[index] && !self.sliced[index] && !self.withheld[index]
    }

    /// Counts a point that just started or stopped rendering and queues its
    /// chunk for a rebake.
    fn point_rendering_changed(&mut self, index: usize) {
        let chunk_index = self.chunk_of_point[index] as usize;
        let renders = self.renders(index);
        let chunk = &mut self.chunks[chunk_index];
        if renders {
            chunk.visible_points += 1;
        } else {
            chunk.visible_points = chunk.visible_points.saturating_sub(1);
        }
        if !self.chunk_dirty[chunk_index] {
            self.chunk_dirty[chunk_index] = true;
            self.dirty_queue.push(chunk_index as u32);
        }
        self.epoch += 1;
    }

    /// Moves a single point (identified by its image id) to a new position.
    /// The point keeps its chunk membership; the chunk's AABB grows to keep
    /// covering it. A mesh rebake is queued (and the epoch bumped) only when
    /// the point currently renders — a hidden or sliced point contributes no
    /// geometry, so its move costs nothing until it is shown again.
    pub fn set_point_position(&mut self, image_id: usize, position: Vec3) {
        let Some(&index) = self.id_to_index.get(&image_id) else {
            return;
        };
        let index = index as usize;
        if self.positions[index] == position {
            return;
        }
        self.positions[index] = position;
        let chunk_index = self.chunk_of_point[index] as usize;
        let half_point = Vec3::splat(self.point_size * 0.5);
        let chunk = &mut self.chunks[chunk_index];
        chunk.min = chunk.min.min(position - half_point);
        chunk.max = chunk.max.max(position + half_point);
        if !self.renders(index) {
            return;
        }
        if !self.chunk_dirty[chunk_index] {
            self.chunk_dirty[chunk_index] = true;
            self.dirty_queue.push(chunk_index as u32);
        }
        self.epoch += 1;
    }

    /// Reapplies the cutaway slice for the current camera position: points
    /// closer than `radius` (or all points when `radius` is zero, restoring
    /// them) get their `sliced` flag updated and affected chunks are queued
    /// for a mesh rebake. One O(points) pass; cost beyond that is
    /// proportional to how many points actually flipped.
    pub fn apply_slice(&mut self, camera: Vec3, radius: f32) {
        let radius_squared = radius * radius;
        for index in 0..self.positions.len() {
            let sliced =
                radius > 0.0 && self.positions[index].distance_squared(camera) < radius_squared;
            if self.sliced[index] == sliced {
                continue;
            }
            let rendered = self.renders(index);
            self.sliced[index] = sliced;
            if self.renders(index) != rendered {
                self.point_rendering_changed(index);
            }
        }
    }

    /// Monotonic counter bumped on any visibility or chunk-set change; lets
    /// callers detect rebuilds/edits that require reapplying derived state.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Despawns all chunk entities and releases their meshes, then bakes a
    /// fresh chunk set for `points`. Used at startup and on scene reloads.
    pub fn rebuild(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        points: Vec<PointCloudPoint>,
        point_size: f32,
    ) {
        self.rebuild_from_layout(
            commands,
            meshes,
            materials,
            PointCloudLayout::new(points, point_size),
        );
    }

    /// Like [`Self::rebuild`] but from a layout partitioned ahead of time.
    /// Chunk entities are spawned with empty meshes and queued for baking, so
    /// the main-thread cost here is proportional to the chunk count and the
    /// per-point mesh work is spread over the following frames by
    /// [`apply_point_cloud_edits`], nearest chunks first.
    pub fn rebuild_from_layout(
        &mut self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        layout: PointCloudLayout,
    ) {
        for chunk in &self.chunks {
            commands.entity(chunk.entity).despawn_recursive();
            meshes.remove(&chunk.mesh);
        }
        self.chunks.clear();
        self.chunk_dirty.clear();
        self.dirty_queue.clear();
        self.epoch += 1;

        let material = self
            .material
            .get_or_insert_with(|| {
                materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    perceptual_roughness: 0.72,
                    ..default()
                })
            })
            .clone();

        let point_count = layout.positions.len();
        self.positions = layout.positions;
        self.colors = layout.colors;
        self.hidden = vec![false; point_count];
        self.sliced = vec![false; point_count];
        self.withheld = vec![false; point_count];
        self.id_to_index = layout.id_to_index;
        self.chunk_of_point = layout.chunk_of_point;
        self.cell_size = layout.cell_size;
        self.point_size = layout.point_size;

        self.chunks.reserve(layout.chunks.len());
        for (chunk_index, chunk) in layout.chunks.into_iter().enumerate() {
            let mesh = meshes.add(empty_chunk_mesh());
            let entity = commands
                .spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material.clone()),
                    Transform::IDENTITY,
                    PointCloudChunkEntity,
                    Name::new(format!("point cloud chunk {chunk_index}")),
                ))
                .id();
            self.chunks.push(PointChunk {
                entity,
                mesh,
                visible_points: chunk.point_indices.len(),
                point_indices: chunk.point_indices,
                min: chunk.min,
                max: chunk.max,
            });
            self.chunk_dirty.push(true);
            self.dirty_queue.push(chunk_index as u32);
        }
    }

    /// Reorders the pending bake queue so chunks nearest `camera` bake first
    /// (the queue pops from the back).
    fn prioritize_bakes_around(&mut self, camera: Vec3) {
        if self.dirty_queue.len() < 2 {
            return;
        }
        let chunks = &self.chunks;
        self.dirty_queue.sort_by(|left, right| {
            let left_distance = chunk_chebyshev_distance(camera, &chunks[*left as usize]);
            let right_distance = chunk_chebyshev_distance(camera, &chunks[*right as usize]);
            right_distance.total_cmp(&left_distance)
        });
    }
}

/// Placeholder mesh for a chunk whose geometry has not been baked yet; the
/// entity exists so view-bounds visibility can already be assigned to it.
fn empty_chunk_mesh() -> Mesh {
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new())
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, Vec::<[f32; 4]>::new())
    .with_inserted_indices(Indices::U32(Vec::new()))
}

/// Applies queued chunk bakes (initial layout and point hide/show edits) by
/// rebuilding the affected chunk meshes, bounded per frame both by count and
/// by the number of points baked so a burst of large chunks cannot
/// monopolize a frame. Chunks nearest the camera bake first.
pub fn apply_point_cloud_edits(
    mut cloud: ResMut<PointCloud>,
    mut meshes: ResMut<Assets<Mesh>>,
    camera_query: Query<&Transform, With<FlyCamera>>,
) {
    if cloud.dirty_queue.is_empty() {
        return;
    }
    if let Ok(camera) = camera_query.get_single() {
        cloud.prioritize_bakes_around(camera.translation);
    }
    let mut baked_points = 0;
    for _ in 0..POINT_CHUNK_REBUILDS_PER_FRAME {
        if baked_points >= POINT_BAKES_PER_FRAME {
            break;
        }
        let Some(chunk_index) = cloud.dirty_queue.pop() else {
            break;
        };
        let chunk_index = chunk_index as usize;
        cloud.chunk_dirty[chunk_index] = false;
        let chunk = &cloud.chunks[chunk_index];
        baked_points += chunk.point_indices.len();
        let mesh = build_chunk_mesh(
            &cloud.positions,
            &cloud.colors,
            &cloud.hidden,
            &cloud.sliced,
            &cloud.withheld,
            &chunk.point_indices,
            cloud.point_size,
        );
        meshes.insert(&chunk.mesh, mesh);
    }
}

/// Palette color a point gets from its image id; matches the old per-entity
/// marker materials (converted to linear space for mesh vertex colors).
pub fn point_color(image_id: usize) -> [f32; 4] {
    const PALETTE: [Color; 5] = [
        Color::srgb(0.22, 0.68, 1.0),
        Color::srgb(0.18, 0.88, 0.54),
        Color::srgb(1.0, 0.33, 0.42),
        Color::srgb(0.78, 0.48, 1.0),
        Color::srgb(1.0, 0.62, 0.18),
    ];
    const SELECTED: Color = Color::srgb(1.0, 0.86, 0.24);
    let color = if image_id == 0 {
        SELECTED
    } else {
        PALETTE[image_id % PALETTE.len()]
    };
    color.to_linear().to_f32_array()
}

/// Cell edge length that spreads `len` points over cells of roughly
/// `target_points` each.
fn chunk_cell_size(positions: &[Vec3], target_points: usize) -> f32 {
    if positions.len() <= target_points {
        return f32::MAX.sqrt();
    }
    let mut min = positions[0];
    let mut max = positions[0];
    for position in positions.iter().skip(1) {
        min = min.min(*position);
        max = max.max(*position);
    }
    let extent = max - min;
    let max_extent = extent.x.max(extent.y).max(extent.z).max(1.0);
    let cells_per_axis = (positions.len() as f32 / target_points as f32)
        .cbrt()
        .ceil()
        .max(1.0);
    (max_extent / cells_per_axis).max(1.0)
}

/// Bakes the visible points of one chunk into a triangle mesh of unit cubes
/// with per-vertex palette colors.
fn build_chunk_mesh(
    positions: &[Vec3],
    colors: &[[f32; 4]],
    hidden: &[bool],
    sliced: &[bool],
    withheld: &[bool],
    point_indices: &[u32],
    point_size: f32,
) -> Mesh {
    // Bevy's cuboid layout: 4 corners per face, CCW from outside.
    const FACES: [([f32; 3], [[f32; 3]; 4]); 6] = [
        (
            [0.0, 0.0, 1.0],
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            [
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
            ],
        ),
        (
            [1.0, 0.0, 0.0],
            [
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
            ],
        ),
        (
            [-1.0, 0.0, 0.0],
            [
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, -0.5],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            [
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            [
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
            ],
        ),
    ];

    let renders = |point: usize| !hidden[point] && !sliced[point] && !withheld[point];
    let visible_count = point_indices
        .iter()
        .filter(|&&point| renders(point as usize))
        .count();
    let mut mesh_positions = Vec::with_capacity(visible_count * 24);
    let mut mesh_normals = Vec::with_capacity(visible_count * 24);
    let mut mesh_colors = Vec::with_capacity(visible_count * 24);
    let mut mesh_indices = Vec::with_capacity(visible_count * 36);

    for &point in point_indices {
        let point = point as usize;
        if !renders(point) {
            continue;
        }
        let center = positions[point];
        let color = colors[point];
        for (normal, corners) in &FACES {
            let base = mesh_positions.len() as u32;
            for corner in corners {
                mesh_positions.push([
                    center.x + corner[0] * point_size,
                    center.y + corner[1] * point_size,
                    center.z + corner[2] * point_size,
                ]);
                mesh_normals.push(*normal);
                mesh_colors.push(color);
            }
            mesh_indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, mesh_positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, mesh_normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, mesh_colors)
    .with_inserted_indices(Indices::U32(mesh_indices))
}

/// Adaptive fixed boundary around the camera. `radius` is the half-extent of
/// the axis-aligned cube of world the renderer currently shows;
/// `f32::INFINITY` while everything fits inside the point budget.
#[derive(Resource)]
pub struct ViewBounds {
    pub radius: f32,
    faces_active: [bool; WALL_FACE_COUNT],
    last_camera_position: Option<Vec3>,
    seen_epoch: Option<u64>,
    pub visible_points: usize,
    pub culled_points: usize,
}

impl Default for ViewBounds {
    fn default() -> Self {
        Self {
            radius: f32::INFINITY,
            faces_active: [false; WALL_FACE_COUNT],
            last_camera_position: None,
            seen_epoch: None,
            visible_points: 0,
            culled_points: 0,
        }
    }
}

impl ViewBounds {
    /// Whether `position` lies outside the boundary cube around `camera`.
    pub fn is_beyond(&self, camera: Vec3, position: Vec3) -> bool {
        if !self.radius.is_finite() {
            return false;
        }
        let delta = (position - camera).abs();
        delta.x.max(delta.y).max(delta.z) > self.radius
    }
}

/// Boundary wall root: one per cube face. `axis`/`sign` name the face
/// (e.g. axis 0, sign +1 is the +X wall).
#[derive(Component)]
pub struct BoundaryWall {
    axis: usize,
    sign: f32,
}

#[derive(Component)]
pub enum BoundaryWallPart {
    Backdrop,
    /// Text plane; carries its unscaled world size so it can be resized to
    /// the current boundary radius while keeping the texture aspect ratio.
    Text {
        base_size: Vec2,
    },
}

/// Spawns the six hidden boundary walls (opaque backdrop plus a large
/// "More points beyond this point" text plane each).
pub fn spawn_boundary_walls(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    font: Option<&FontArc>,
) {
    let quad = meshes.add(Rectangle::new(1.0, 1.0));
    let backdrop_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.032, 0.036, 0.045),
        unlit: true,
        ..default()
    });
    let text = font.map(|font| {
        let rendered = render_label_text(
            font,
            &[LabelTextLine {
                text: WALL_TEXT,
                font_size: WALL_TEXT_FONT_SIZE,
                color: [214, 222, 234, 235],
            }],
            1.0,
            f32::MAX,
        );
        let texture = images.add(rendered.image);
        let material = materials.add(StandardMaterial {
            base_color_texture: Some(texture),
            // Texture is premultiplied (see `render_label_text`/`blend_label_pixel`).
            alpha_mode: AlphaMode::Premultiplied,
            unlit: true,
            ..default()
        });
        (material, rendered.world_size)
    });

    // Rotations that point each face's quad normal back at the camera.
    let faces: [(usize, f32, Quat); WALL_FACE_COUNT] = [
        (0, 1.0, Quat::from_rotation_y(-FRAC_PI_2)),
        (0, -1.0, Quat::from_rotation_y(FRAC_PI_2)),
        (1, 1.0, Quat::from_rotation_x(FRAC_PI_2)),
        (1, -1.0, Quat::from_rotation_x(-FRAC_PI_2)),
        (2, 1.0, Quat::from_rotation_y(PI)),
        (2, -1.0, Quat::IDENTITY),
    ];
    for (axis, sign, rotation) in faces {
        commands
            .spawn((
                Transform::from_rotation(rotation),
                Visibility::Hidden,
                BoundaryWall { axis, sign },
                Name::new(format!(
                    "view boundary wall {}{}",
                    if sign > 0.0 { '+' } else { '-' },
                    ['x', 'y', 'z'][axis]
                )),
            ))
            .with_children(|wall| {
                wall.spawn((
                    Mesh3d(quad.clone()),
                    MeshMaterial3d(backdrop_material.clone()),
                    Transform::IDENTITY,
                    BoundaryWallPart::Backdrop,
                ));
                if let Some((material, base_size)) = &text {
                    wall.spawn((
                        Mesh3d(quad.clone()),
                        MeshMaterial3d(material.clone()),
                        Transform::IDENTITY,
                        BoundaryWallPart::Text {
                            base_size: *base_size,
                        },
                    ));
                }
            });
    }
}

type WallQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static BoundaryWall,
        &'static mut Transform,
        &'static mut Visibility,
    ),
    (Without<PointCloudChunkEntity>, Without<FlyCamera>),
>;
type WallPartQuery<'w, 's> = Query<
    'w,
    's,
    (&'static BoundaryWallPart, &'static mut Transform),
    (
        Without<BoundaryWall>,
        Without<PointCloudChunkEntity>,
        Without<FlyCamera>,
    ),
>;

/// Keeps the view boundary centered on the camera: budgets which chunks are
/// visible, hides everything beyond the resulting radius and dresses the six
/// boundary walls.
pub fn update_view_bounds(
    mut bounds: ResMut<ViewBounds>,
    cloud: Res<PointCloud>,
    camera_query: Query<&Transform, With<FlyCamera>>,
    mut chunk_visibility: Query<
        &mut Visibility,
        (With<PointCloudChunkEntity>, Without<BoundaryWall>),
    >,
    mut walls: WallQuery,
    mut wall_parts: WallPartQuery,
) {
    let Ok(camera_transform) = camera_query.get_single() else {
        return;
    };
    let camera = camera_transform.translation;

    let moved_enough = bounds.last_camera_position.is_none_or(|previous| {
        camera.distance_squared(previous)
            > (cloud.chunk_cell_size() * VIEW_BOUNDS_RECOMPUTE_MOVE_FACTOR).powi(2)
    });
    let recompute = moved_enough || bounds.seen_epoch != Some(cloud.epoch);

    if recompute {
        bounds.last_camera_position = Some(camera);

        let mut ordered: Vec<(f32, usize)> = cloud
            .chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| (chunk_chebyshev_distance(camera, chunk), index))
            .collect();
        ordered.sort_by(|left, right| left.0.total_cmp(&right.0));

        let min_radius = cloud.chunk_cell_size() * VIEW_BOUNDS_MIN_CELL_RADIUS;
        let mut visible_points = 0;
        let mut radius = f32::INFINITY;
        for &(distance, index) in &ordered {
            let chunk = &cloud.chunks[index];
            if distance > min_radius && visible_points + chunk.visible_points > VISIBLE_POINT_BUDGET
            {
                radius = distance.max(min_radius);
                break;
            }
            visible_points += chunk.visible_points;
        }
        bounds.radius = radius;
        bounds.visible_points = visible_points;
        bounds.culled_points = cloud
            .point_count()
            .saturating_sub(
                cloud
                    .hidden
                    .iter()
                    .zip(cloud.sliced.iter())
                    .filter(|(hidden, sliced)| **hidden || **sliced)
                    .count(),
            )
            .saturating_sub(visible_points);

        // Chunk entities spawned this frame (scene reload) are not yet
        // queryable; keep the epoch un-consumed so the pass reruns next
        // frame instead of leaving their visibility unassigned.
        let mut all_chunks_present = true;
        let mut faces_active = [false; WALL_FACE_COUNT];
        for &(distance, index) in &ordered {
            let chunk = &cloud.chunks[index];
            let included = distance <= radius;
            if let Ok(mut visibility) = chunk_visibility.get_mut(chunk.entity) {
                *visibility = if included {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                };
            } else {
                all_chunks_present = false;
            }
            if !radius.is_finite() || chunk.visible_points == 0 {
                continue;
            }
            // A wall face lights up when any populated chunk has geometry
            // beyond that face's plane (excluded chunks and included chunks
            // that spill past the wall alike — the opaque wall occludes the
            // spill either way).
            for axis in 0..3 {
                if chunk.max[axis] > camera[axis] + radius {
                    faces_active[axis * 2] = true;
                }
                if chunk.min[axis] < camera[axis] - radius {
                    faces_active[axis * 2 + 1] = true;
                }
            }
        }
        bounds.faces_active = faces_active;
        if all_chunks_present {
            bounds.seen_epoch = Some(cloud.epoch);
        }

        // Resize wall backdrop/text quads to the new radius.
        if radius.is_finite() {
            let face_extent = radius * 2.0;
            for (part, mut transform) in &mut wall_parts {
                match part {
                    BoundaryWallPart::Backdrop => {
                        transform.scale = Vec3::new(face_extent, face_extent, 1.0);
                        transform.translation = Vec3::ZERO;
                    }
                    BoundaryWallPart::Text { base_size } => {
                        let width = face_extent * WALL_TEXT_WIDTH_FRACTION;
                        let scale = width / base_size.x.max(f32::EPSILON);
                        transform.scale = Vec3::new(base_size.x * scale, base_size.y * scale, 1.0);
                        // Nudged off the backdrop toward the camera so the
                        // text never z-fights its own wall.
                        transform.translation = Vec3::new(0.0, 0.0, radius * 0.01);
                    }
                }
            }
        }
    }

    // Walls track the camera every frame so the boundary feels fixed to the
    // player rather than to the world.
    let radius = bounds.radius;
    for (wall, mut transform, mut visibility) in &mut walls {
        let face_index = wall.axis * 2 + usize::from(wall.sign < 0.0);
        let active = radius.is_finite() && bounds.faces_active[face_index];
        *visibility = if active {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if active {
            let mut translation = camera;
            translation[wall.axis] += wall.sign * radius;
            transform.translation = translation;
        }
    }
}

/// Chebyshev (max-axis) distance from `camera` to the chunk's AABB; the
/// natural metric for testing against the axis-aligned boundary cube.
fn chunk_chebyshev_distance(camera: Vec3, chunk: &PointChunk) -> f32 {
    let mut distance = 0.0_f32;
    for axis in 0..3 {
        let below = chunk.min[axis] - camera[axis];
        let above = camera[axis] - chunk.max[axis];
        distance = distance.max(below).max(above);
    }
    distance.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_points(count: usize, spacing: f32) -> Vec<PointCloudPoint> {
        (0..count)
            .map(|index| PointCloudPoint {
                image_id: index,
                position: Vec3::new(index as f32 * spacing, 0.0, 0.0),
                color: point_color(index),
            })
            .collect()
    }

    fn build_cloud(points: Vec<PointCloudPoint>) -> (World, PointCloud) {
        let mut world = World::new();
        world.init_resource::<Assets<Mesh>>();
        world.init_resource::<Assets<StandardMaterial>>();
        let mut cloud = PointCloud::default();
        world.resource_scope(|world, mut meshes: Mut<Assets<Mesh>>| {
            world.resource_scope(|world, mut materials: Mut<Assets<StandardMaterial>>| {
                let mut queue = bevy::ecs::world::CommandQueue::default();
                let mut commands = Commands::new(&mut queue, world);
                cloud.rebuild(&mut commands, &mut meshes, &mut materials, points, 0.2);
                queue.apply(world);
                // Drain the initial bake queue so tests observe only the
                // edits they make themselves.
                bake_all_pending(&mut cloud, &mut meshes);
            });
        });
        (world, cloud)
    }

    fn bake_all_pending(cloud: &mut PointCloud, meshes: &mut Assets<Mesh>) {
        while let Some(chunk_index) = cloud.dirty_queue.pop() {
            let chunk_index = chunk_index as usize;
            cloud.chunk_dirty[chunk_index] = false;
            let chunk = &cloud.chunks[chunk_index];
            let mesh = build_chunk_mesh(
                &cloud.positions,
                &cloud.colors,
                &cloud.hidden,
                &cloud.sliced,
                &cloud.withheld,
                &chunk.point_indices,
                cloud.point_size,
            );
            meshes.insert(&chunk.mesh, mesh);
        }
    }

    #[test]
    fn rebuild_defers_chunk_baking_to_the_edit_queue() {
        let mut world = World::new();
        world.init_resource::<Assets<Mesh>>();
        world.init_resource::<Assets<StandardMaterial>>();
        let mut cloud = PointCloud::default();
        world.resource_scope(|world, mut meshes: Mut<Assets<Mesh>>| {
            world.resource_scope(|world, mut materials: Mut<Assets<StandardMaterial>>| {
                let mut queue = bevy::ecs::world::CommandQueue::default();
                let mut commands = Commands::new(&mut queue, world);
                cloud.rebuild(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    test_points(4000, 1.0),
                    0.2,
                );
                queue.apply(world);
                // Every chunk starts empty and queued; nothing was baked on
                // the rebuild path itself.
                assert_eq!(cloud.dirty_queue.len(), cloud.chunks.len());
                assert!(cloud.chunk_dirty.iter().all(|dirty| *dirty));
                for chunk in &cloud.chunks {
                    assert_eq!(meshes.get(&chunk.mesh).unwrap().count_vertices(), 0);
                }
                bake_all_pending(&mut cloud, &mut meshes);
                let baked_vertices: usize = cloud
                    .chunks
                    .iter()
                    .map(|chunk| meshes.get(&chunk.mesh).unwrap().count_vertices())
                    .sum();
                assert_eq!(baked_vertices, 4000 * 24);
            });
        });
    }

    #[test]
    fn pending_bakes_are_ordered_nearest_camera_first() {
        let (_, mut cloud) = build_cloud(test_points(4000, 1.0));
        assert!(cloud.chunks.len() >= 2, "expected multiple chunks");
        for chunk_index in 0..cloud.chunks.len() {
            cloud.chunk_dirty[chunk_index] = true;
            cloud.dirty_queue.push(chunk_index as u32);
        }
        let nearest = cloud.chunks.len() - 1;
        let camera = cloud.chunks[nearest].min;
        cloud.prioritize_bakes_around(camera);
        let distances = cloud
            .dirty_queue
            .iter()
            .map(|index| chunk_chebyshev_distance(camera, &cloud.chunks[*index as usize]))
            .collect::<Vec<_>>();
        // The queue pops from the back, so distances must decrease toward it.
        assert!(distances.windows(2).all(|pair| pair[0] >= pair[1]));
        assert_eq!(*distances.last().unwrap(), 0.0);
    }

    #[test]
    fn rebuild_partitions_all_points_into_chunks() {
        let (_, cloud) = build_cloud(test_points(4000, 1.0));
        assert!(cloud.chunks.len() > 1, "expected multiple chunks");
        let chunk_points: usize = cloud
            .chunks
            .iter()
            .map(|chunk| chunk.point_indices.len())
            .sum();
        assert_eq!(chunk_points, 4000);
        assert_eq!(cloud.point_count(), 4000);
    }

    #[test]
    fn hiding_a_point_marks_only_its_chunk_dirty() {
        let (_, mut cloud) = build_cloud(test_points(4000, 1.0));
        let before_epoch = cloud.epoch;

        cloud.set_point_visible(17, false);

        assert_eq!(cloud.dirty_queue.len(), 1);
        let chunk = cloud.dirty_queue[0] as usize;
        assert_eq!(
            cloud.chunks[chunk].visible_points,
            cloud.chunks[chunk].point_indices.len() - 1
        );
        assert!(cloud.epoch > before_epoch);

        // Hiding the same point again is a no-op.
        cloud.set_point_visible(17, false);
        assert_eq!(cloud.dirty_queue.len(), 1);

        cloud.set_point_visible(17, true);
        assert_eq!(
            cloud.chunks[chunk].visible_points,
            cloud.chunks[chunk].point_indices.len()
        );
    }

    #[test]
    fn moving_a_visible_point_queues_rebake_and_grows_chunk_bounds() {
        let (_, mut cloud) = build_cloud(test_points(100, 1.0));
        let epoch = cloud.epoch;

        cloud.set_point_position(4, Vec3::new(0.0, 250.0, 0.0));

        assert!(!cloud.dirty_queue.is_empty());
        assert!(cloud.epoch > epoch);
        let index = cloud.id_to_index[&4] as usize;
        assert_eq!(cloud.positions[index], Vec3::new(0.0, 250.0, 0.0));
        let chunk = cloud.chunk_of_point[index] as usize;
        assert!(cloud.chunks[chunk].max.y >= 250.0);
    }

    #[test]
    fn moving_a_hidden_point_skips_rebake() {
        let (_, mut cloud) = build_cloud(test_points(100, 1.0));
        cloud.set_point_visible(7, false);
        let queue_length = cloud.dirty_queue.len();
        let epoch = cloud.epoch;

        cloud.set_point_position(7, Vec3::new(0.0, 0.0, 99.0));

        assert_eq!(cloud.dirty_queue.len(), queue_length);
        assert_eq!(cloud.epoch, epoch);
        let index = cloud.id_to_index[&7] as usize;
        assert_eq!(cloud.positions[index], Vec3::new(0.0, 0.0, 99.0));
    }

    #[test]
    fn unknown_image_id_is_ignored() {
        let (_, mut cloud) = build_cloud(test_points(10, 1.0));
        cloud.set_point_visible(usize::MAX, false);
        assert!(cloud.dirty_queue.is_empty());
    }

    #[test]
    fn chunk_mesh_skips_hidden_points() {
        let positions = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let colors = vec![[1.0, 1.0, 1.0, 1.0]; 3];
        let hidden = vec![false, true, false];
        let sliced = vec![false; 3];
        let withheld = vec![false; 3];
        let mesh = build_chunk_mesh(
            &positions,
            &colors,
            &hidden,
            &sliced,
            &withheld,
            &[0, 1, 2],
            0.2,
        );
        assert_eq!(mesh.count_vertices(), 2 * 24);

        let sliced = vec![false, false, true];
        let mesh = build_chunk_mesh(
            &positions,
            &colors,
            &hidden,
            &sliced,
            &withheld,
            &[0, 1, 2],
            0.2,
        );
        assert_eq!(mesh.count_vertices(), 24);

        let withheld = vec![true, false, false];
        let mesh = build_chunk_mesh(
            &positions,
            &colors,
            &hidden,
            &sliced,
            &withheld,
            &[0, 1, 2],
            0.2,
        );
        assert_eq!(mesh.count_vertices(), 0);
    }

    #[test]
    fn a_withheld_point_stops_rendering_until_released_whatever_else_hides_it() {
        let (_, mut cloud) = build_cloud(test_points(10, 1.0));
        let visible = |cloud: &PointCloud| -> usize {
            cloud.chunks.iter().map(|chunk| chunk.visible_points).sum()
        };
        cloud.set_point_withheld(3, true);
        assert_eq!(visible(&cloud), 9);
        // Its billboard loading and unloading meanwhile must not show it.
        cloud.set_point_visible(3, false);
        cloud.set_point_visible(3, true);
        assert_eq!(visible(&cloud), 9);
        cloud.set_point_withheld(3, false);
        assert_eq!(visible(&cloud), 10);
        // Released while its billboard hides it, it stays hidden.
        cloud.set_point_visible(4, false);
        cloud.set_point_withheld(4, true);
        cloud.set_point_withheld(4, false);
        assert_eq!(visible(&cloud), 9);
    }

    #[test]
    fn apply_slice_flips_points_and_restores_on_zero_radius() {
        let (_, mut cloud) = build_cloud(test_points(100, 1.0));
        let total_visible: usize = cloud.chunks.iter().map(|chunk| chunk.visible_points).sum();
        assert_eq!(total_visible, 100);

        // Points 0..=5 sit within 5.5 units of the origin along +X.
        cloud.apply_slice(Vec3::ZERO, 5.5);
        assert_eq!(cloud.sliced.iter().filter(|sliced| **sliced).count(), 6);
        let total_visible: usize = cloud.chunks.iter().map(|chunk| chunk.visible_points).sum();
        assert_eq!(total_visible, 94);
        assert!(!cloud.dirty_queue.is_empty());

        // A billboard loading for an already-sliced point must not distort
        // the visible-point accounting.
        cloud.set_point_visible(2, false);
        let total_visible: usize = cloud.chunks.iter().map(|chunk| chunk.visible_points).sum();
        assert_eq!(total_visible, 94);

        cloud.apply_slice(Vec3::ZERO, 0.0);
        assert!(cloud.sliced.iter().all(|sliced| !sliced));
        let total_visible: usize = cloud.chunks.iter().map(|chunk| chunk.visible_points).sum();
        assert_eq!(total_visible, 99);
    }

    #[test]
    fn chebyshev_distance_is_zero_inside_and_axis_gap_outside() {
        let chunk = PointChunk {
            entity: Entity::PLACEHOLDER,
            mesh: Handle::default(),
            point_indices: Vec::new(),
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
            visible_points: 0,
        };
        assert_eq!(chunk_chebyshev_distance(Vec3::ZERO, &chunk), 0.0);
        assert_eq!(
            chunk_chebyshev_distance(Vec3::new(3.0, 0.0, 0.0), &chunk),
            2.0
        );
        assert_eq!(
            chunk_chebyshev_distance(Vec3::new(3.0, -4.0, 0.0), &chunk),
            3.0
        );
    }

    #[test]
    fn view_bounds_beyond_uses_cube_metric() {
        let bounds = ViewBounds {
            radius: 10.0,
            ..default()
        };
        assert!(!bounds.is_beyond(Vec3::ZERO, Vec3::new(9.0, 9.0, 9.0)));
        assert!(bounds.is_beyond(Vec3::ZERO, Vec3::new(0.0, 11.0, 0.0)));

        let unbounded = ViewBounds::default();
        assert!(!unbounded.is_beyond(Vec3::ZERO, Vec3::splat(1.0e9)));
    }
}
