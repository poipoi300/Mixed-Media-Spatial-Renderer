use std::{env, fs, path::PathBuf};

use ab_glyph::{point, Font, FontArc, GlyphId, PxScale, ScaleFont};
use bevy::{
    input::mouse::MouseButton,
    math::primitives::{Cuboid, Cylinder, Rectangle},
    prelude::*,
    render::{
        camera::RenderTarget,
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        view::RenderLayers,
    },
    sprite::Anchor,
    window::PrimaryWindow,
};
use image::{Rgba, RgbaImage};
use spatial_viewer_ui::{AxisGizmoFace, AxisGizmoState, PauseMenuState, UiInputCapture};

use super::{BillboardLabelFont, FlyCamera, PRESENTATION_RENDER_LAYER};

const GIZMO_RENDER_LAYER: usize = 1;
// Camera pulled back far enough that the axis labels at
// `GIZMO_LABEL_DISTANCE` stay inside the square render target at any
// orientation instead of clipping at the canvas edge.
const GIZMO_CAMERA_DISTANCE: f32 = 6.4;
const GIZMO_CAMERA_FOV_DEGREES: f32 = 44.0;
const GIZMO_TEXTURE_SIZE: u32 = 512;
const GIZMO_CANVAS_MIN: f32 = 220.0;
const GIZMO_CANVAS_MAX: f32 = 340.0;
const GIZMO_CANVAS_FRACTION: f32 = 0.28;
/// Matches the 18px inset the UI pills use inside the 16:9 frame.
const GIZMO_CANVAS_MARGIN: f32 = 18.0;
const GIZMO_CANVAS_Z: f32 = 10.0;
/// Half-extent of the invisible cube that the clickable face/edge/corner
/// hit regions are laid out on. The rendered core is a sphere (see
/// `GIZMO_SPHERE_RADIUS`)
const GIZMO_BOUNDING_HALF_EXTENT: f32 = 0.5;
/// Slightly smaller than `GIZMO_BOUNDING_HALF_EXTENT` so the sphere reads as
/// a rounded body sitting inside the bounding cube. The whole visible sphere
/// is tiled by 8 colored triangles (see `spherical_octant_mesh`) that meet
/// the 6 axis directions exactly at this radius, so there is no separate
/// plain "core" mesh underneath to show through.
const GIZMO_SPHERE_RADIUS: f32 = GIZMO_BOUNDING_HALF_EXTENT * 0.9;
/// Tessellation level for each of the 8 sphere tiles; higher reads as more
/// curved (and costs `subdivisions^2` triangles per tile).
const GIZMO_SPHERE_SUBDIVISIONS: u32 = 6;
/// How much to scale up a tile's saturation after averaging 3 axis colors
/// (which pulls the result toward gray), capped below full saturation so
/// tiles read as bright rather than neon.
const GIZMO_TILE_SATURATION_BOOST: f32 = 1.6;
const GIZMO_TILE_MAX_SATURATION: f32 = 0.85;
const GIZMO_TILE_MIN_LIGHTNESS: f32 = 0.5;
const GIZMO_TILE_MAX_LIGHTNESS: f32 = 0.66;
const GIZMO_AXIS_CAP_DISTANCE: f32 = 1.34;
const GIZMO_AXIS_CAP_HALF_EXTENT: f32 = 0.18;
/// Cross-section radius of the cylinder connecting the sphere to each
/// extremity cube.
const GIZMO_ROD_RADIUS: f32 = 0.045;
const GIZMO_LABEL_DISTANCE: f32 = 1.56;
const GIZMO_LABEL_WORLD_HEIGHT: f32 = 0.58;
/// Generic "large but sane" width cap for labels rendered outside the gizmo
/// (e.g. the video controls time label), which aren't bound by the gizmo
/// camera's view frustum. Axis gizmo labels use `gizmo_label_safe_half_extent`
/// instead, since a fixed constant can't account for the gizmo orbiting.
pub(crate) const GIZMO_LABEL_MAX_WORLD_WIDTH: f32 = 2.3;
/// Scale applied to a label's world size while the cursor hovers it. A
/// hovered label's font size always matches this scale exactly — long text
/// wraps onto more lines (see `wrap_text_to_pixel_width`) rather than
/// shrinking the font or being truncated to fit.
const GIZMO_LABEL_HOVER_SCALE: f32 = 2.2;
// Label textures render at double the previous pixel size (same world size)
// so the text stays crisp on the up-scaled gizmo canvas.
const GIZMO_AXIS_LABEL_FONT_SIZE: f32 = 72.0;
const GIZMO_DIMENSION_LABEL_FONT_SIZE: f32 = 52.0;
const GIZMO_LABEL_PADDING: u32 = 16;
const GIZMO_LABEL_DEPTH_OFFSET: f32 = 0.018;
const GIZMO_FACE_PLATE_ALPHA: f32 = 0.88;
const GIZMO_NEGATIVE_FACE_PLATE_ALPHA: f32 = 0.50;

/// Half-width (and, since the render target is square and the camera FOV is
/// symmetric, half-height too) that a label anchored `anchor_distance` from
/// the gizmo's center can extend to without ever being clipped by the gizmo
/// camera's view frustum, no matter which way the gizmo currently faces.
///
/// The gizmo camera orbits at a fixed `GIZMO_CAMERA_DISTANCE` looking at the
/// origin (see `sync_axis_gizmo_camera`), so as it orbits, a label's depth
/// from the camera and its screen-space offset trade off against each other.
/// Parameterizing the anchor's signed offset along the view direction as
/// `t` (`t` ranges over `[-anchor_distance, anchor_distance]`), the visible
/// half-extent at that orientation is
/// `(GIZMO_CAMERA_DISTANCE + t) * tan(half_fov) - sqrt(anchor_distance^2 - t^2)`.
/// Minimizing that over `t` (calculus: derivative zero at
/// `t = -anchor_distance * sin(half_fov)`) collapses to the closed form
/// below; `gizmo_label_safe_half_extent_matches_brute_force_worst_case`
/// checks this against a numeric search over `t`.
fn gizmo_label_safe_half_extent(anchor_distance: f32) -> f32 {
    let half_fov = (GIZMO_CAMERA_FOV_DEGREES * 0.5).to_radians();
    (GIZMO_CAMERA_DISTANCE * half_fov.sin() - anchor_distance) / half_fov.cos()
}

#[derive(Component)]
pub struct AxisGizmoCamera;

#[derive(Component)]
pub(crate) struct AxisGizmoCanvas;

#[derive(Component)]
pub struct AxisGizmoEntity;

/// One of the six extremity cube caps; its material is swapped by
/// `update_axis_gizmo_face_highlight` for hover/selection feedback.
#[derive(Component)]
pub(crate) struct AxisGizmoFacePlate {
    face: AxisGizmoFace,
}

#[derive(Component)]
pub(crate) struct AxisGizmoLabel {
    face: AxisGizmoFace,
    anchor: Vec3,
    camera_depth_offset: f32,
    /// Dimension text the current `variants` were rendered for; `None`
    /// until `refresh_axis_gizmo_labels` has rendered this label.
    rendered_dimension_label: Option<String>,
    variants: Option<AxisGizmoLabelVariants>,
    hovered: bool,
}

/// A pre-rendered mesh/texture pair for one display state (normal or
/// hovered) of a gizmo label, swapped in wholesale on hover change so no
/// text is re-rasterized at runtime.
struct AxisGizmoLabelVariant {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    texture: Handle<Image>,
    world_size: Vec2,
}

impl AxisGizmoLabelVariant {
    fn remove(
        &self,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        images: &mut Assets<Image>,
    ) {
        materials.remove(&self.material);
        meshes.remove(&self.mesh);
        images.remove(&self.texture);
    }
}

pub(crate) struct LabelTextLine<'a> {
    pub(crate) text: &'a str,
    pub(crate) font_size: f32,
    pub(crate) color: [u8; 4],
}

#[derive(Clone, Copy)]
struct AxisGizmoHitTarget {
    face: AxisGizmoFace,
    minimum: Vec3,
    maximum: Vec3,
}

#[derive(Clone, Copy)]
struct AxisGizmoCanvasLayout {
    top_left: Vec2,
    size: f32,
}

fn create_axis_gizmo_render_target_image() -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: GIZMO_TEXTURE_SIZE,
            height: GIZMO_TEXTURE_SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Bgra8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image
}

pub fn spawn_axis_gizmo_3d(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    axis_labels: &[String; 3],
) {
    let layer = RenderLayers::layer(GIZMO_RENDER_LAYER);
    let gizmo_target = images.add(create_axis_gizmo_render_target_image());
    let cube_mesh = meshes.add(Mesh::from(Cuboid::new(1.0, 1.0, 1.0)));
    let cap_mesh = cube_mesh.clone();
    let tick_mesh = cube_mesh;
    // Rod is a real cylinder, rotated per-axis rather than a cube stretched
    // anisotropically, so its circular cross-section is never squashed.
    let rod_mesh = meshes.add(Mesh::from(Cylinder::new(1.0, 1.0)));

    let tick_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.92, 0.94, 0.98, 0.72),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    let axis_materials = [
        materials.add(axis_material(axis_color(AxisGizmoFace::PositiveX), 0.88)),
        materials.add(axis_material(axis_color(AxisGizmoFace::PositiveY), 0.88)),
        materials.add(axis_material(axis_color(AxisGizmoFace::PositiveZ), 0.88)),
    ];
    let negative_axis_materials = [
        materials.add(axis_material(axis_color(AxisGizmoFace::NegativeX), 0.50)),
        materials.add(axis_material(axis_color(AxisGizmoFace::NegativeY), 0.50)),
        materials.add(axis_material(axis_color(AxisGizmoFace::NegativeZ), 0.50)),
    ];

    commands.spawn((
        Sprite {
            image: gizmo_target.clone(),
            custom_size: Some(Vec2::splat(GIZMO_CANVAS_MAX)),
            anchor: Anchor::BottomLeft,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, GIZMO_CANVAS_Z),
        Visibility::Visible,
        RenderLayers::layer(PRESENTATION_RENDER_LAYER),
        AxisGizmoCanvas,
        AxisGizmoEntity,
        Name::new("axis gizmo canvas"),
    ));

    // The sphere is approximated as an octahedron (vertices at the 6 axis
    // directions, one tile per sign combination of X/Y/Z) so its 8 curved
    // triangular tiles meet edge-to-edge and vertex-to-vertex with no gaps,
    // and each tile's 3 corners land exactly on 3 of the 6 axis directions.
    // Each tile is a single flat color (rather than blending its 3 corner
    // colors) so the 8 triangular facets stay visually distinct.
    for signs in octant_signs() {
        let corners = octant_corners(signs);
        let mesh = meshes.add(spherical_octant_mesh(
            corners,
            GIZMO_SPHERE_RADIUS,
            GIZMO_SPHERE_SUBDIVISIONS,
        ));
        let material = materials.add(StandardMaterial {
            base_color: octant_average_color(signs),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::IDENTITY,
            layer.clone(),
            AxisGizmoEntity,
            Name::new("axis gizmo sphere tile"),
        ));
    }

    for face in signed_faces() {
        let axis_index = axis_index(face);
        let rod_material = if is_positive_face(face) {
            axis_materials[axis_index].clone()
        } else {
            negative_axis_materials[axis_index].clone()
        };
        commands.spawn((
            Mesh3d(rod_mesh.clone()),
            MeshMaterial3d(rod_material),
            axis_rod_transform(face),
            layer.clone(),
            AxisGizmoEntity,
            Name::new(format!("axis gizmo {} rod", face_label(face))),
        ));
        // Each cap gets its own material instance so hover/selection
        // highlighting can restyle one extremity without touching its
        // siblings (or the rod it caps, which keeps its own material).
        let cap_material = materials.add(face_plate_material(face, false, false));
        commands.spawn((
            Mesh3d(cap_mesh.clone()),
            MeshMaterial3d(cap_material),
            Transform::from_translation(face_direction(face) * GIZMO_AXIS_CAP_DISTANCE)
                .with_scale(Vec3::splat(GIZMO_AXIS_CAP_HALF_EXTENT * 2.0)),
            layer.clone(),
            AxisGizmoEntity,
            AxisGizmoFacePlate { face },
            Name::new(format!("axis gizmo {} cap", face_label(face))),
        ));
    }

    for face in signed_faces() {
        for distance in [0.82, 1.10] {
            commands.spawn((
                Mesh3d(tick_mesh.clone()),
                MeshMaterial3d(tick_material.clone()),
                tick_transform(face, distance),
                layer.clone(),
                AxisGizmoEntity,
                Name::new(format!("axis gizmo {} tick", face_label(face))),
            ));
        }
    }

    // Labels are rasterized lazily by `refresh_axis_gizmo_labels`, one per
    // frame, so a projection change never pays for all six at once.
    for face in signed_faces() {
        let anchor = face_direction(face) * GIZMO_LABEL_DISTANCE;
        commands.spawn((
            Transform::from_translation(anchor),
            Visibility::Hidden,
            layer.clone(),
            AxisGizmoEntity,
            AxisGizmoLabel {
                face,
                anchor,
                camera_depth_offset: GIZMO_LABEL_DEPTH_OFFSET,
                rendered_dimension_label: None,
                variants: None,
                hovered: false,
            },
            Name::new(format!("axis gizmo {} label", face_label(face))),
        ));
    }
    commands.insert_resource(AxisGizmoLabels {
        dimension_labels: axis_labels.clone(),
    });

    commands.spawn((
        Camera3d::default(),
        Camera {
            order: -1,
            clear_color: ClearColorConfig::Custom(Color::NONE),
            target: RenderTarget::Image(gizmo_target),
            ..default()
        },
        Projection::from(PerspectiveProjection {
            fov: GIZMO_CAMERA_FOV_DEGREES.to_radians(),
            ..default()
        }),
        Msaa::Off,
        Transform::from_xyz(2.4, 1.8, 2.4).looking_at(Vec3::ZERO, Vec3::Y),
        RenderLayers::layer(GIZMO_RENDER_LAYER),
        AxisGizmoCamera,
        AxisGizmoEntity,
        Name::new("axis gizmo camera"),
    ));
}

/// Dimension label text per projection axis, as the gizmo should show it.
/// Changing this (a projection reload) re-renders only the labels whose
/// text differs from what they last rendered.
#[derive(Resource, Default)]
pub struct AxisGizmoLabels {
    pub dimension_labels: [String; 3],
}

/// Rasterizes at most one stale gizmo label per frame. A label is stale
/// when it has never been rendered or its axis dimension text changed.
/// Rendering both display variants of a label costs several milliseconds
/// (two glyph rasterizations, texture uploads), so spreading the six over
/// six frames keeps a projection change from stalling a frame.
pub fn refresh_axis_gizmo_labels(
    labels_text: Res<AxisGizmoLabels>,
    font: Res<BillboardLabelFont>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut labels: Query<(Entity, &mut AxisGizmoLabel)>,
    mut commands: Commands,
) {
    let Some(font) = font.0.as_ref() else {
        return;
    };
    let Some((entity, mut label)) = labels.iter_mut().find(|(_, label)| {
        label.rendered_dimension_label.as_deref()
            != Some(labels_text.dimension_labels[axis_index(label.face)].as_str())
    }) else {
        return;
    };
    let full_dimension_label = labels_text.dimension_labels[axis_index(label.face)].clone();
    let variants = build_gizmo_label_variants(
        &mut meshes,
        &mut materials,
        &mut images,
        font,
        label.face,
        &full_dimension_label,
    );
    if let Some(previous) = label.variants.take() {
        previous.remove(&mut meshes, &mut materials, &mut images);
    }
    let shown = if label.hovered {
        &variants.hovered
    } else {
        &variants.normal
    };
    commands.entity(entity).insert((
        Mesh3d(shown.mesh.clone()),
        MeshMaterial3d(shown.material.clone()),
        Visibility::Inherited,
    ));
    label.variants = Some(variants);
    label.rendered_dimension_label = Some(full_dimension_label);
}

/// Both display states of one gizmo label rendered for `dimension_label`.
struct AxisGizmoLabelVariants {
    normal: AxisGizmoLabelVariant,
    hovered: AxisGizmoLabelVariant,
}

impl AxisGizmoLabelVariants {
    fn remove(
        &self,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        images: &mut Assets<Image>,
    ) {
        for variant in [&self.normal, &self.hovered] {
            variant.remove(meshes, materials, images);
        }
    }
}

fn build_gizmo_label_variants(
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    font: &FontArc,
    face: AxisGizmoFace,
    full_dimension_label: &str,
) -> AxisGizmoLabelVariants {
    // Same anchor distance for every face and for both label variants, so
    // the safe extent (and thus the frustum-clipping guarantee) is identical
    // whether or not the label is currently hovered.
    let axis_label_max_width = gizmo_label_safe_half_extent(GIZMO_LABEL_DISTANCE) * 2.0;
    let axis_label = face_label(face);
    let dimension_label = truncate_label(full_dimension_label, 16);
    let normal = build_gizmo_label_variant(
        meshes,
        materials,
        images,
        font,
        &[
            LabelTextLine {
                text: axis_label,
                font_size: GIZMO_AXIS_LABEL_FONT_SIZE,
                color: axis_color_rgba(face),
            },
            LabelTextLine {
                text: &dimension_label,
                font_size: GIZMO_DIMENSION_LABEL_FONT_SIZE,
                color: [246, 248, 252, 255],
            },
        ],
        GIZMO_LABEL_WORLD_HEIGHT,
        axis_label_max_width,
    );
    let hovered = build_gizmo_label_variant(
        meshes,
        materials,
        images,
        font,
        &[
            LabelTextLine {
                text: axis_label,
                font_size: GIZMO_AXIS_LABEL_FONT_SIZE * GIZMO_LABEL_HOVER_SCALE,
                color: axis_color_rgba(face),
            },
            LabelTextLine {
                text: full_dimension_label,
                font_size: GIZMO_DIMENSION_LABEL_FONT_SIZE * GIZMO_LABEL_HOVER_SCALE,
                color: [246, 248, 252, 255],
            },
        ],
        GIZMO_LABEL_WORLD_HEIGHT * GIZMO_LABEL_HOVER_SCALE,
        axis_label_max_width,
    );
    AxisGizmoLabelVariants { normal, hovered }
}
pub fn update_axis_gizmo_viewport(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<&mut Camera, With<AxisGizmoCamera>>,
    mut canvases: Query<(&mut Sprite, &mut Transform, &mut Visibility), With<AxisGizmoCanvas>>,
) {
    let layout = windows
        .get_single()
        .ok()
        .and_then(|window| axis_gizmo_canvas_layout(window.resolution.size()));
    let Some(layout) = layout else {
        for mut camera in &mut cameras {
            camera.is_active = false;
        }
        for (_, _, mut visibility) in &mut canvases {
            *visibility = Visibility::Hidden;
        }
        return;
    };

    for mut camera in &mut cameras {
        camera.is_active = true;
    }
    let window_size = windows
        .get_single()
        .map(|window| window.resolution.size())
        .unwrap_or(Vec2::ZERO);
    for (mut sprite, mut transform, mut visibility) in &mut canvases {
        sprite.custom_size = Some(Vec2::splat(layout.size));
        transform.translation = axis_gizmo_canvas_translation(window_size, layout);
        *visibility = Visibility::Visible;
    }
}

pub fn sync_axis_gizmo_camera(
    main_camera: Query<&Transform, (With<FlyCamera>, Without<AxisGizmoCamera>)>,
    mut gizmo_camera: Query<&mut Transform, With<AxisGizmoCamera>>,
) {
    let Ok(main_transform) = main_camera.get_single() else {
        return;
    };
    let Ok(mut gizmo_transform) = gizmo_camera.get_single_mut() else {
        return;
    };

    let forward = main_transform.rotation.mul_vec3(Vec3::NEG_Z);
    gizmo_transform.translation = -forward * GIZMO_CAMERA_DISTANCE;
    gizmo_transform.rotation = main_transform.rotation;
}

pub fn sync_axis_gizmo_labels(
    camera: Query<&Transform, (With<AxisGizmoCamera>, Without<AxisGizmoLabel>)>,
    mut labels: Query<(&AxisGizmoLabel, &mut Transform)>,
) {
    let Ok(camera_transform) = camera.get_single() else {
        return;
    };
    let camera_backward = camera_transform.rotation.mul_vec3(Vec3::Z);
    for (label, mut label_transform) in &mut labels {
        label_transform.translation = label.anchor + camera_backward * label.camera_depth_offset;
        label_transform.rotation = camera_transform.rotation;
    }
}

/// Swaps each gizmo label to its pre-rendered hover variant (larger, with
/// the full untruncated dimension name) while the cursor is over it, and
/// back to the normal variant otherwise.
pub fn update_axis_gizmo_label_hover(
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<AxisGizmoCamera>>,
    pause_menu: Res<PauseMenuState>,
    capture: Res<UiInputCapture>,
    mut labels: Query<(
        Entity,
        &Transform,
        &mut AxisGizmoLabel,
        &mut Mesh3d,
        &mut MeshMaterial3d<StandardMaterial>,
    )>,
) {
    let ray = if pause_menu.paused || capture.pointer_over_ui {
        None
    } else {
        gizmo_cursor_ray(&windows, &camera)
    };

    // Hit-tested against the *normal* size even while hovered: if the hover
    // box itself were used, growing it could sweep over a neighboring
    // label's hit region (or keep re-qualifying itself right at its own
    // edge), and whichever label currently "wins" would flip from frame to
    // frame as their box sizes change in response to that same win/lose
    // state — a feedback loop that shows up as rapid flashing between
    // normal and expanded. Keeping the hit region fixed regardless of hover
    // state removes the feedback entirely.
    let hovered_entity = ray.and_then(|ray| {
        labels
            .iter()
            .filter_map(|(entity, transform, label, _, _)| {
                let variants = label.variants.as_ref()?;
                ray_quad_hit(
                    ray.origin,
                    ray.direction.as_vec3(),
                    transform,
                    variants.normal.world_size,
                )
                .map(|distance| (distance, entity))
            })
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, entity)| entity)
    });

    for (entity, _, mut label, mut mesh, mut material) in &mut labels {
        let should_hover = hovered_entity == Some(entity);
        if should_hover == label.hovered {
            continue;
        }
        label.hovered = should_hover;
        let Some(variants) = label.variants.as_ref() else {
            continue;
        };
        let variant = if should_hover {
            &variants.hovered
        } else {
            &variants.normal
        };
        mesh.0 = variant.mesh.clone();
        material.0 = variant.material.clone();
    }
}

pub fn handle_axis_gizmo_clicks(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<AxisGizmoCamera>>,
    pause_menu: Res<PauseMenuState>,
    capture: Res<UiInputCapture>,
    mut axis_gizmo: ResMut<AxisGizmoState>,
) {
    if pause_menu.paused || capture.blocks_world_clicks() {
        return;
    }
    if !mouse_buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(ray) = gizmo_cursor_ray(&windows, &camera) else {
        return;
    };
    let origin = ray.origin;
    let direction = ray.direction.as_vec3();
    // Cap hits take priority: they are the smaller, more specific target,
    // and a ray aimed at a cap can also graze the sphere en route to it.
    if let Some(face) = axis_gizmo_face_from_ray(origin, direction) {
        axis_gizmo.request_face(face);
    } else if let Some(octant_direction) = octant_direction_from_ray(origin, direction) {
        axis_gizmo.request_direction(octant_direction);
    }
}

/// Tracks which gizmo face is under the cursor so plates can light up before
/// a click, making the clickable regions discoverable.
pub fn update_axis_gizmo_hover(
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<AxisGizmoCamera>>,
    pause_menu: Res<PauseMenuState>,
    capture: Res<UiInputCapture>,
    mut axis_gizmo: ResMut<AxisGizmoState>,
) {
    let hovered = if pause_menu.paused || capture.pointer_over_ui {
        None
    } else {
        hovered_gizmo_face(&windows, &camera)
    };
    if axis_gizmo.hovered() != hovered {
        axis_gizmo.set_hovered(hovered);
    }
}

fn hovered_gizmo_face(
    windows: &Query<&Window, With<PrimaryWindow>>,
    camera: &Query<(&Camera, &GlobalTransform), With<AxisGizmoCamera>>,
) -> Option<AxisGizmoFace> {
    let ray = gizmo_cursor_ray(windows, camera)?;
    axis_gizmo_face_from_ray(ray.origin, ray.direction.as_vec3())
}

/// Whether the cursor currently sits inside the axis gizmo's screen canvas.
/// World click handlers use this so a click meant for the gizmo can never
/// also hit billboards or video controls rendered behind it.
pub(crate) fn cursor_over_axis_gizmo(window: &Window) -> bool {
    window
        .cursor_position()
        .and_then(|cursor| cursor_in_canvas(window, cursor))
        .is_some()
}

fn gizmo_cursor_ray(
    windows: &Query<&Window, With<PrimaryWindow>>,
    camera: &Query<(&Camera, &GlobalTransform), With<AxisGizmoCamera>>,
) -> Option<Ray3d> {
    let window = windows.get_single().ok()?;
    let cursor_position = window.cursor_position()?;
    let (camera, camera_transform) = camera.get_single().ok()?;
    if !camera.is_active {
        return None;
    }
    let local_cursor = cursor_in_canvas(window, cursor_position)?;
    camera
        .viewport_to_world(camera_transform, local_cursor)
        .ok()
}

/// Restyles face plates when hover or selection changes (or plates were just
/// respawned by a scene reload).
pub fn update_axis_gizmo_face_highlight(
    axis_gizmo: Res<AxisGizmoState>,
    plates: Query<(&AxisGizmoFacePlate, &MeshMaterial3d<StandardMaterial>)>,
    added_plates: Query<(), Added<AxisGizmoFacePlate>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut last_state: Local<Option<(Option<AxisGizmoFace>, Option<AxisGizmoFace>)>>,
) {
    let state = (axis_gizmo.hovered(), axis_gizmo.selected());
    if *last_state == Some(state) && added_plates.is_empty() {
        return;
    }
    *last_state = Some(state);
    for (plate, material_handle) in &plates {
        let Some(material) = materials.get_mut(&material_handle.0) else {
            continue;
        };
        *material = face_plate_material(
            plate.face,
            state.0 == Some(plate.face),
            state.1 == Some(plate.face),
        );
    }
}

fn face_plate_material(face: AxisGizmoFace, hovered: bool, selected: bool) -> StandardMaterial {
    let base_alpha = if is_positive_face(face) {
        GIZMO_FACE_PLATE_ALPHA
    } else {
        GIZMO_NEGATIVE_FACE_PLATE_ALPHA
    };
    let (alpha, lighten) = if selected {
        (1.0, 0.35)
    } else if hovered {
        ((base_alpha + 0.25).min(1.0), 0.20)
    } else {
        (base_alpha, 0.0)
    };
    axis_material(lighten_color(axis_color(face), lighten), alpha)
}

fn lighten_color(color: Color, amount: f32) -> Color {
    let srgba = color.to_srgba();
    Color::srgb(
        srgba.red + (1.0 - srgba.red) * amount,
        srgba.green + (1.0 - srgba.green) * amount,
        srgba.blue + (1.0 - srgba.blue) * amount,
    )
}

pub fn axis_gizmo_view_direction(face: AxisGizmoFace) -> Vec3 {
    face_direction(face)
}

pub fn axis_gizmo_up_vector(face: AxisGizmoFace) -> Vec3 {
    match face {
        AxisGizmoFace::PositiveY | AxisGizmoFace::NegativeY => Vec3::Z,
        AxisGizmoFace::PositiveX
        | AxisGizmoFace::NegativeX
        | AxisGizmoFace::PositiveZ
        | AxisGizmoFace::NegativeZ => Vec3::Y,
    }
}

/// A reasonable "up" reference for orienting the camera toward an arbitrary
/// (non-axis-aligned) direction, such as a gizmo sphere tile's corner
/// direction. Prefers world Y, falling back to Z when `direction` is nearly
/// parallel to Y (looking almost straight up/down would otherwise produce a
/// degenerate or unstable orientation).
pub fn axis_gizmo_generic_up_vector(direction: Vec3) -> Vec3 {
    let reference = if direction.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    (reference - direction * direction.dot(reference)).normalize()
}

fn build_gizmo_label_variant(
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    font: &FontArc,
    lines: &[LabelTextLine<'_>],
    world_height: f32,
    max_world_width: f32,
) -> AxisGizmoLabelVariant {
    let rendered = render_label_text(font, lines, world_height, max_world_width);
    let texture = images.add(rendered.image);
    let mesh = meshes.add(Rectangle::new(rendered.world_size.x, rendered.world_size.y));
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(texture.clone()),
        // Texture is premultiplied (see `render_label_text`/`blend_label_pixel`).
        alpha_mode: AlphaMode::Premultiplied,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    AxisGizmoLabelVariant {
        mesh,
        material,
        texture,
        world_size: rendered.world_size,
    }
}

pub(crate) struct RenderedLabelText {
    pub(crate) image: Image,
    pub(crate) world_size: Vec2,
}

pub(crate) fn render_label_text(
    font: &FontArc,
    lines: &[LabelTextLine<'_>],
    world_height: f32,
    max_world_width: f32,
) -> RenderedLabelText {
    let fallback_line = [LabelTextLine {
        text: " ",
        font_size: GIZMO_DIMENSION_LABEL_FONT_SIZE,
        color: [255, 255, 255, 255],
    }];
    let lines = if lines.is_empty() {
        fallback_line.as_slice()
    } else {
        lines
    };
    // Line height depends only on font size, not on the text itself, so the
    // world-units-per-pixel scale below is fixed by the *unwrapped* spec:
    // wrapping a long line into more sub-lines grows the label's total
    // height instead of shrinking every line's font size to compensate.
    let line_scales = lines
        .iter()
        .map(|line| {
            let scale = PxScale::from(line.font_size);
            let scaled_font = font.as_scaled(scale);
            let line_height = (scaled_font.height() + scaled_font.line_gap())
                .ceil()
                .max(1.0);
            (scale, line_height)
        })
        .collect::<Vec<_>>();
    let reference_text_height_px = line_scales
        .iter()
        .map(|(_, height)| *height)
        .sum::<f32>()
        .max(1.0);
    let per_pixel_scale = world_height / reference_text_height_px;
    let max_line_pixel_width = (max_world_width / per_pixel_scale).max(1.0);

    // Wrap (never truncate) each line's text so hover labels stay fully
    // readable regardless of length; a line that already fits comes back as
    // a single unchanged "sub-line".
    let mut render_lines: Vec<(String, PxScale, f32, [u8; 4])> = Vec::new();
    for (line, &(scale, line_height)) in lines.iter().zip(line_scales.iter()) {
        for sub_line in wrap_text_to_pixel_width(line.text, max_line_pixel_width, |candidate| {
            line_width(font, scale, candidate)
        }) {
            render_lines.push((sub_line, scale, line_height, line.color));
        }
    }

    let total_pixel_height = render_lines
        .iter()
        .map(|(_, _, line_height, _)| *line_height)
        .sum::<f32>()
        .max(1.0);
    // Wrapping alone keeps the font size fixed, but pathologically long text
    // could still wrap into more lines than fit the gizmo's safe vertical
    // extent. Shrink uniformly (keeping every line intact) rather than clip
    // or truncate in that rare case; ordinary labels never hit this.
    let scale_correction = (max_world_width / (total_pixel_height * per_pixel_scale)).min(1.0);
    let per_pixel_scale = per_pixel_scale * scale_correction;

    let text_width = render_lines
        .iter()
        .map(|(text, scale, _, _)| line_width(font, *scale, text))
        .fold(1.0, f32::max)
        .ceil() as u32;
    let text_height = total_pixel_height.ceil() as u32;
    let width = text_width + GIZMO_LABEL_PADDING * 2;
    let height = text_height + GIZMO_LABEL_PADDING * 2;
    let mut rgba = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));
    let mut baseline_cursor = GIZMO_LABEL_PADDING as f32;

    for (text, scale, line_height, color) in &render_lines {
        let scaled_font = font.as_scaled(*scale);
        let baseline = baseline_cursor + scaled_font.ascent();
        let pixel_width = line_width(font, *scale, text);
        let x = (rgba.width() as f32 - pixel_width) * 0.5;
        draw_shadowed_text_line(&mut rgba, font, *scale, text, x, baseline, *color);
        baseline_cursor += *line_height;
    }

    let image = Image::new(
        Extent3d {
            width: rgba.width(),
            height: rgba.height(),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );

    RenderedLabelText {
        world_size: Vec2::new(width as f32, height as f32) * per_pixel_scale,
        image,
    }
}

fn draw_shadowed_text_line(
    image: &mut RgbaImage,
    font: &FontArc,
    scale: PxScale,
    text: &str,
    x: f32,
    baseline: f32,
    color: [u8; 4],
) {
    for (offset_x, offset_y, alpha) in [(3.0, 3.0, 155), (1.5, 1.5, 220)] {
        draw_text_line(
            image,
            font,
            scale,
            text,
            x + offset_x,
            baseline + offset_y,
            [0, 0, 0, alpha],
        );
    }
    draw_text_line(image, font, scale, text, x, baseline, color);
}

fn draw_text_line(
    image: &mut RgbaImage,
    font: &FontArc,
    scale: PxScale,
    text: &str,
    mut x: f32,
    baseline: f32,
    color: [u8; 4],
) {
    let scaled_font = font.as_scaled(scale);
    let mut previous = None;
    for character in text.chars() {
        let glyph_id = scaled_font.glyph_id(character);
        if let Some(previous_id) = previous {
            x += scaled_font.kern(previous_id, glyph_id);
        }
        let glyph = glyph_id.with_scale_and_position(scale, point(x, baseline));
        x += scaled_font.h_advance(glyph_id);
        previous = Some(glyph_id);

        let Some(outlined) = font.outline_glyph(glyph) else {
            continue;
        };
        let bounds = outlined.px_bounds();
        outlined.draw(|glyph_x, glyph_y, coverage| {
            let x = bounds.min.x as i32 + glyph_x as i32;
            let y = bounds.min.y as i32 + glyph_y as i32;
            blend_label_pixel(image, x, y, coverage, color);
        });
    }
}

/// Composites in premultiplied alpha, and the resulting buffer is stored
/// premultiplied (see `render_label_text`). Straight alpha would let the GPU's
/// bilinear filter blend an opaque glyph texel's color against a fully
/// transparent neighbor's `rgb = 0`, darkening the edge into a visible fringe
/// whenever the label is viewed up close; premultiplied edge texels already
/// carry correctly-scaled color so filtering between them stays correct.
fn blend_label_pixel(
    image: &mut RgbaImage,
    glyph_x: i32,
    glyph_y: i32,
    coverage: f32,
    color: [u8; 4],
) {
    if glyph_x < 0 || glyph_y < 0 {
        return;
    }
    let glyph_x = glyph_x as u32;
    let glyph_y = glyph_y as u32;
    if glyph_x >= image.width() || glyph_y >= image.height() {
        return;
    }
    let source_alpha = (coverage.clamp(0.0, 1.0) * f32::from(color[3]) / 255.0).clamp(0.0, 1.0);
    if source_alpha <= 0.0 {
        return;
    }
    let pixel = image.get_pixel_mut(glyph_x, glyph_y);
    let destination_alpha = f32::from(pixel[3]) / 255.0;
    let output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha);

    for channel in 0..3 {
        let source_premultiplied = f32::from(color[channel]) / 255.0 * source_alpha;
        let destination_premultiplied = f32::from(pixel[channel]) / 255.0;
        let output_premultiplied =
            source_premultiplied + destination_premultiplied * (1.0 - source_alpha);
        pixel[channel] = (output_premultiplied * 255.0).round() as u8;
    }
    pixel[3] = (output_alpha * 255.0).round() as u8;
}

fn line_width(font: &FontArc, scale: PxScale, text: &str) -> f32 {
    let scaled_font = font.as_scaled(scale);
    let mut width = 0.0;
    let mut previous: Option<GlyphId> = None;
    for character in text.chars() {
        let glyph_id = scaled_font.glyph_id(character);
        if let Some(previous_id) = previous {
            width += scaled_font.kern(previous_id, glyph_id);
        }
        width += scaled_font.h_advance(glyph_id);
        previous = Some(glyph_id);
    }
    width.max(1.0)
}

/// Splits `text` at whitespace/`_`/`-` boundaries, each piece keeping its
/// trailing separator so pieces can be rejoined with no extra logic (e.g.
/// `"elevation_offset"` -> `["elevation_", "offset"]`).
fn wrap_word_boundaries(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let mut start = 0;
    for (index, character) in text.char_indices() {
        if character == ' ' || character == '_' || character == '-' {
            let end = index + character.len_utf8();
            words.push(&text[start..end]);
            start = end;
        }
    }
    if start < text.len() {
        words.push(&text[start..]);
    }
    words
}

/// Greedily wraps `text` into lines that each measure no wider than
/// `max_pixel_width` under `measure`, breaking at word boundaries where
/// possible. A single word wider than the budget on its own is hard-wrapped
/// character by character instead of being truncated, so text is never
/// elided with an ellipsis purely for being long — see `render_label_text`,
/// which instead shrinks the whole label as a last resort if the wrapped
/// result still doesn't fit the gizmo's safe vertical extent.
fn wrap_text_to_pixel_width(
    text: &str,
    max_pixel_width: f32,
    measure: impl Fn(&str) -> f32,
) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in wrap_word_boundaries(text) {
        let mut candidate = current.clone();
        candidate.push_str(word);
        if current.is_empty() || measure(&candidate) <= max_pixel_width {
            current = candidate;
        } else {
            let finished = current.trim_end().to_owned();
            if !finished.is_empty() {
                lines.push(finished);
            }
            current = word.trim_start().to_owned();
        }
        while measure(&current) > max_pixel_width && current.chars().count() > 1 {
            let mut split_at = current.len();
            for (index, _) in current.char_indices().rev() {
                if index == 0 {
                    break;
                }
                if measure(&current[..index]) <= max_pixel_width {
                    split_at = index;
                    break;
                }
            }
            if split_at == current.len() {
                // Even a single character's prefix doesn't fit; peel off
                // one character anyway so this loop always makes progress.
                split_at = current
                    .char_indices()
                    .nth(1)
                    .map(|(index, _)| index)
                    .unwrap_or(current.len());
            }
            let finished = current[..split_at].trim_end().to_owned();
            if !finished.is_empty() {
                lines.push(finished);
            }
            current = current[split_at..].trim_start().to_owned();
        }
    }
    let finished = current.trim_end().to_owned();
    if !finished.is_empty() || lines.is_empty() {
        lines.push(finished);
    }
    lines
}

pub(crate) fn load_label_font() -> Option<FontArc> {
    for path in gizmo_font_candidates() {
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        match FontArc::try_from_vec(bytes) {
            Ok(font) => return Some(font),
            Err(error) => eprintln!("Failed to load gizmo font {}: {error}", path.display()),
        }
    }
    eprintln!(
        "No gizmo font found. Set SPATIAL_VIEWER_GIZMO_FONT to a TrueType/OpenType font path."
    );
    None
}

fn gizmo_font_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = env::var_os("SPATIAL_VIEWER_GIZMO_FONT") {
        paths.push(PathBuf::from(path));
    }
    paths.extend([
        PathBuf::from(r"C:\Windows\Fonts\segoeui.ttf"),
        PathBuf::from(r"C:\Windows\Fonts\arial.ttf"),
        PathBuf::from("/System/Library/Fonts/Supplemental/Arial.ttf"),
        PathBuf::from("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        PathBuf::from("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        PathBuf::from("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]);
    paths
}

fn axis_gizmo_canvas_layout(window_size: Vec2) -> Option<AxisGizmoCanvasLayout> {
    if window_size.x <= f32::EPSILON || window_size.y <= f32::EPSILON {
        return None;
    }
    // Anchor inside the largest 16:9 rect centered in the window — the same
    // frame the UI pills use — so the gizmo respects the 16:9 convention
    // instead of drifting to the raw window corner.
    let frame_height = window_size.y.min(window_size.x * 9.0 / 16.0);
    let frame_width = frame_height * 16.0 / 9.0;
    let frame_top_left = (window_size - Vec2::new(frame_width, frame_height)) * 0.5;
    let size = (frame_height * GIZMO_CANVAS_FRACTION)
        .round()
        .clamp(GIZMO_CANVAS_MIN, GIZMO_CANVAS_MAX);
    let margin = GIZMO_CANVAS_MARGIN;
    if frame_width <= size + margin * 2.0 || frame_height <= size + margin * 2.0 {
        return None;
    }
    Some(AxisGizmoCanvasLayout {
        top_left: Vec2::new(
            frame_top_left.x + margin,
            frame_top_left.y + frame_height - size - margin,
        ),
        size,
    })
}

fn axis_gizmo_canvas_translation(window_size: Vec2, layout: AxisGizmoCanvasLayout) -> Vec3 {
    Vec3::new(
        -window_size.x * 0.5 + layout.top_left.x,
        window_size.y * 0.5 - layout.top_left.y - layout.size,
        GIZMO_CANVAS_Z,
    )
}

fn cursor_in_canvas(window: &Window, cursor_position: Vec2) -> Option<Vec2> {
    let layout = axis_gizmo_canvas_layout(window.resolution.size())?;
    let viewport_min = layout.top_left;
    let viewport_size = Vec2::splat(layout.size);
    let viewport_max = viewport_min + viewport_size;
    if cursor_position.x < viewport_min.x
        || cursor_position.x > viewport_max.x
        || cursor_position.y < viewport_min.y
        || cursor_position.y > viewport_max.y
    {
        return None;
    }
    Some((cursor_position - viewport_min) / viewport_size * Vec2::splat(GIZMO_TEXTURE_SIZE as f32))
}

/// Hit-tests only the 6 extremity cube caps. The sphere tiles have their own
/// hit test (`octant_direction_from_ray`) bound to a different action
/// (orient to a corner direction rather than an axis face), so the two
/// clickable surfaces are kept geometrically disjoint on purpose rather than
/// having one large region cover both.
fn axis_gizmo_face_from_ray(origin: Vec3, direction: Vec3) -> Option<AxisGizmoFace> {
    gizmo_cap_hit_targets()
        .into_iter()
        .filter_map(|target| {
            ray_aabb_hit(origin, direction, target.minimum, target.maximum)
                .map(|distance| (distance, target.face))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, face)| face)
}

fn gizmo_cap_hit_targets() -> Vec<AxisGizmoHitTarget> {
    signed_faces()
        .into_iter()
        .map(|face| {
            let center = face_direction(face) * GIZMO_AXIS_CAP_DISTANCE;
            AxisGizmoHitTarget {
                face,
                minimum: center - Vec3::splat(GIZMO_AXIS_CAP_HALF_EXTENT),
                maximum: center + Vec3::splat(GIZMO_AXIS_CAP_HALF_EXTENT),
            }
        })
        .collect()
}

/// Ray/sphere intersection used to hit-test the gizmo's sphere tiles; on a
/// hit, the intersection point's per-axis signs identify which of the 8
/// octant tiles was clicked, and the canonical (exactly diagonal) direction
/// for that octant is returned rather than the literal (off-center) hit
/// point, so every click on a given tile snaps to the same view.
fn octant_direction_from_ray(origin: Vec3, direction: Vec3) -> Option<Vec3> {
    let distance = ray_sphere_hit(origin, direction, GIZMO_SPHERE_RADIUS)?;
    let point = origin + direction * distance;
    // `f32::signum` never returns exactly 0.0 (0.0_f32.signum() == 1.0), so
    // this always yields one of the 8 unit diagonal directions.
    let octant_direction =
        Vec3::new(point.x.signum(), point.y.signum(), point.z.signum()).normalize();
    Some(octant_direction)
}

fn ray_sphere_hit(origin: Vec3, direction: Vec3, radius: f32) -> Option<f32> {
    // Solve |origin + t * direction|^2 = radius^2 for the smallest t >= 0.
    let b = origin.dot(direction);
    let c = origin.length_squared() - radius * radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_discriminant = discriminant.sqrt();
    let near = -b - sqrt_discriminant;
    let far = -b + sqrt_discriminant;
    if near >= 0.0 {
        Some(near)
    } else if far >= 0.0 {
        Some(far)
    } else {
        None
    }
}

fn ray_aabb_hit(origin: Vec3, direction: Vec3, minimum: Vec3, maximum: Vec3) -> Option<f32> {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    for axis in 0..3 {
        let origin_axis = origin[axis];
        let direction_axis = direction[axis];
        let min_axis = minimum[axis];
        let max_axis = maximum[axis];
        if direction_axis.abs() <= f32::EPSILON {
            if origin_axis < min_axis || origin_axis > max_axis {
                return None;
            }
            continue;
        }
        let inv_direction = 1.0 / direction_axis;
        let mut near = (min_axis - origin_axis) * inv_direction;
        let mut far = (max_axis - origin_axis) * inv_direction;
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }
        t_min = t_min.max(near);
        t_max = t_max.min(far);
        if t_min > t_max {
            return None;
        }
    }
    if t_max < 0.0 {
        None
    } else {
        Some(t_min.max(0.0))
    }
}

/// Ray/plane intersection against a camera-facing label quad, bounded to its
/// `size` rectangle in the quad's local X/Y plane.
fn ray_quad_hit(origin: Vec3, direction: Vec3, transform: &Transform, size: Vec2) -> Option<f32> {
    let normal = transform.rotation.mul_vec3(Vec3::Z);
    let denominator = direction.dot(normal);
    if denominator.abs() <= f32::EPSILON {
        return None;
    }
    let distance = (transform.translation - origin).dot(normal) / denominator;
    if distance < 0.0 {
        return None;
    }
    let relative = (origin + direction * distance) - transform.translation;
    let right = transform.rotation.mul_vec3(Vec3::X);
    let up = transform.rotation.mul_vec3(Vec3::Y);
    let half_extent = size * 0.5;
    if relative.dot(right).abs() <= half_extent.x && relative.dot(up).abs() <= half_extent.y {
        Some(distance)
    } else {
        None
    }
}

fn signed_faces() -> [AxisGizmoFace; 6] {
    [
        AxisGizmoFace::PositiveX,
        AxisGizmoFace::NegativeX,
        AxisGizmoFace::PositiveY,
        AxisGizmoFace::NegativeY,
        AxisGizmoFace::PositiveZ,
        AxisGizmoFace::NegativeZ,
    ]
}

fn is_positive_face(face: AxisGizmoFace) -> bool {
    matches!(
        face,
        AxisGizmoFace::PositiveX | AxisGizmoFace::PositiveY | AxisGizmoFace::PositiveZ
    )
}

fn axis_index(face: AxisGizmoFace) -> usize {
    match face {
        AxisGizmoFace::PositiveX | AxisGizmoFace::NegativeX => 0,
        AxisGizmoFace::PositiveY | AxisGizmoFace::NegativeY => 1,
        AxisGizmoFace::PositiveZ | AxisGizmoFace::NegativeZ => 2,
    }
}

fn face_direction(face: AxisGizmoFace) -> Vec3 {
    match face {
        AxisGizmoFace::PositiveX => Vec3::X,
        AxisGizmoFace::NegativeX => Vec3::NEG_X,
        AxisGizmoFace::PositiveY => Vec3::Y,
        AxisGizmoFace::NegativeY => Vec3::NEG_Y,
        AxisGizmoFace::PositiveZ => Vec3::Z,
        AxisGizmoFace::NegativeZ => Vec3::NEG_Z,
    }
}

fn axis_color(face: AxisGizmoFace) -> Color {
    let base = match axis_index(face) {
        0 => Vec3::new(0.95, 0.18, 0.24),
        1 => Vec3::new(0.20, 0.92, 0.36),
        _ => Vec3::new(0.20, 0.48, 1.0),
    };
    let multiplier = if is_positive_face(face) { 1.0 } else { 0.72 };
    Color::srgb(
        base.x * multiplier,
        base.y * multiplier,
        base.z * multiplier,
    )
}

fn axis_color_rgba(face: AxisGizmoFace) -> [u8; 4] {
    let (red, green, blue): (f32, f32, f32) = match axis_index(face) {
        0 => (242.0, 46.0, 61.0),
        1 => (51.0, 235.0, 92.0),
        _ => (51.0, 122.0, 255.0),
    };
    let multiplier = if is_positive_face(face) { 1.0 } else { 0.72 };
    [
        (red * multiplier).round() as u8,
        (green * multiplier).round() as u8,
        (blue * multiplier).round() as u8,
        255,
    ]
}

fn axis_material(color: Color, alpha: f32) -> StandardMaterial {
    StandardMaterial {
        base_color: color.with_alpha(alpha),
        alpha_mode: AlphaMode::Blend,
        emissive: color.into(),
        unlit: true,
        ..default()
    }
}

/// The rod mesh is a unit cylinder (radius 1, height 1) built along local
/// +Y, so it is rotated to the axis direction and non-uniformly scaled
/// (radius in X/Z, length in Y) rather than relying on an axis-aligned cube
/// whose anisotropic scale would otherwise squash a circular cross-section.
/// It starts exactly at the sphere's surface (where the tiles meet at that
/// axis direction) and runs out to the extremity cube.
fn axis_rod_transform(face: AxisGizmoFace) -> Transform {
    let direction = face_direction(face);
    let start = GIZMO_SPHERE_RADIUS;
    let end = GIZMO_AXIS_CAP_DISTANCE;
    let length = end - start;
    let center = direction * (start + length * 0.5);
    let rotation = Quat::from_rotation_arc(Vec3::Y, direction);
    Transform::from_translation(center)
        .with_rotation(rotation)
        .with_scale(Vec3::new(GIZMO_ROD_RADIUS, length, GIZMO_ROD_RADIUS))
}

fn tick_transform(face: AxisGizmoFace, distance: f32) -> Transform {
    let direction = face_direction(face);
    let signed_distance = if is_positive_face(face) {
        distance
    } else {
        -distance
    };
    let (translation, scale) = match axis_index(face) {
        0 => (
            Vec3::new(signed_distance, 0.0, 0.0),
            Vec3::new(0.020, 0.18, 0.020),
        ),
        1 => (
            Vec3::new(0.0, signed_distance, 0.0),
            Vec3::new(0.18, 0.020, 0.020),
        ),
        _ => (
            Vec3::new(0.0, 0.0, signed_distance),
            Vec3::new(0.020, 0.18, 0.020),
        ),
    };
    Transform::from_translation(translation + direction * 0.0).with_scale(scale)
}

/// The 8 sign combinations of (X, Y, Z), one per octahedron face/tile.
fn octant_signs() -> [[f32; 3]; 8] {
    let mut signs = [[0.0; 3]; 8];
    let mut index = 0;
    for x in [1.0f32, -1.0] {
        for y in [1.0f32, -1.0] {
            for z in [1.0f32, -1.0] {
                signs[index] = [x, y, z];
                index += 1;
            }
        }
    }
    signs
}

/// The AxisGizmoFace for a given axis (0 = X, 1 = Y, 2 = Z) and sign.
fn axis_face(axis_index: usize, positive: bool) -> AxisGizmoFace {
    match (axis_index, positive) {
        (0, true) => AxisGizmoFace::PositiveX,
        (0, false) => AxisGizmoFace::NegativeX,
        (1, true) => AxisGizmoFace::PositiveY,
        (1, false) => AxisGizmoFace::NegativeY,
        (2, true) => AxisGizmoFace::PositiveZ,
        _ => AxisGizmoFace::NegativeZ,
    }
}

/// The 3 corner directions for one octant tile, already reordered (if
/// needed) so the natural corner winding faces outward. A tile's corners
/// are the unit vectors along each axis with the signs in `signs` (e.g.
/// `[1.0, -1.0, 1.0]` is the tile touching +X, -Y, +Z), which is exactly
/// where the 3 neighboring rods/cubes attach.
///
/// Connecting corners `[X, Y, Z]` directly (no subdivision) faces outward
/// when `sign_x * sign_y * sign_z > 0.0` (an even number of negative signs,
/// verified in `spherical_octant_mesh_all_positive_octant_winds_outward`);
/// an odd number of negative signs is a reflection, so swapping 2 corners
/// restores outward winding (verified in the sibling
/// `..._negative_octant_winds_outward` test).
fn octant_corners(signs: [f32; 3]) -> [Vec3; 3] {
    let [sx, sy, sz] = signs;
    let poles = [Vec3::X * sx, Vec3::Y * sy, Vec3::Z * sz];
    if sx * sy * sz > 0.0 {
        poles
    } else {
        [poles[0], poles[2], poles[1]]
    }
}

/// A single flat color for one octant tile: the linear-space average of the
/// 3 axis colors it touches. Tiles are solid rather than blended across
/// their 3 corners so each triangular facet stays visually distinct.
fn octant_average_color(signs: [f32; 3]) -> Color {
    let [sx, sy, sz] = signs;
    let corners = [
        axis_color(axis_face(0, sx > 0.0)).to_linear(),
        axis_color(axis_face(1, sy > 0.0)).to_linear(),
        axis_color(axis_face(2, sz > 0.0)).to_linear(),
    ];
    let averaged = LinearRgba::rgb(
        (corners[0].red + corners[1].red + corners[2].red) / 3.0,
        (corners[0].green + corners[1].green + corners[2].green) / 3.0,
        (corners[0].blue + corners[1].blue + corners[2].blue) / 3.0,
    );
    // Averaging 3 saturated axis colors pulls the result toward gray; push
    // saturation and lightness back into a bright-but-not-neon band.
    let mut hsl = Hsla::from(averaged);
    hsl.saturation = (hsl.saturation * GIZMO_TILE_SATURATION_BOOST).min(GIZMO_TILE_MAX_SATURATION);
    hsl.lightness = hsl
        .lightness
        .clamp(GIZMO_TILE_MIN_LIGHTNESS, GIZMO_TILE_MAX_LIGHTNESS);
    Color::LinearRgba(LinearRgba::from(hsl))
}

/// Builds one curved triangular tile of a sphere approximated as an
/// octahedron (6 vertices at the signed axis directions, 8 triangular
/// faces), so the 8 tiles built from every sign combination of X/Y/Z meet
/// edge-to-edge and vertex-to-vertex with no gaps, and each tile's 3
/// corners land exactly on 3 of the gizmo's 6 axis directions where the
/// connecting rods attach. `corners` are interpolated directly in 3D and
/// projected onto the sphere (`point.normalize() * radius`).
fn spherical_octant_mesh(corners: [Vec3; 3], radius: f32, subdivisions: u32) -> Mesh {
    let subdivisions = subdivisions.max(1);

    // Row `i` (0..=subdivisions) sweeps from the corners[1]-corners[2] edge
    // (i = 0) to the single point corners[0] (i = subdivisions), shrinking
    // by one vertex each step; `row_starts[i]` is that row's first index.
    let mut row_starts = Vec::with_capacity(subdivisions as usize + 1);
    let mut next_index = 0u32;
    for i in 0..=subdivisions {
        row_starts.push(next_index);
        next_index += subdivisions - i + 1;
    }

    let vertex_count = next_index as usize;
    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(vertex_count);
    for i in 0..=subdivisions {
        for j in 0..=(subdivisions - i) {
            let k = subdivisions - i - j;
            let a = i as f32 / subdivisions as f32;
            let b = j as f32 / subdivisions as f32;
            let c = k as f32 / subdivisions as f32;
            let flat = corners[0] * a + corners[1] * b + corners[2] * c;
            let position = flat.normalize() * radius;
            positions.push(position.to_array());
            normals.push(position.normalize().to_array());
            uvs.push([a, b]);
        }
    }

    let mut indices = Vec::new();
    for i in 0..subdivisions {
        let row_start = row_starts[i as usize];
        let next_row_start = row_starts[i as usize + 1];
        let next_row_len = subdivisions - i;
        for j in 0..next_row_len {
            let a = row_start + j;
            let b = row_start + j + 1;
            let c = next_row_start + j;
            indices.extend_from_slice(&[a, c, b]);
            if j + 1 < next_row_len {
                let d = next_row_start + j + 1;
                indices.extend_from_slice(&[b, c, d]);
            }
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
}

fn face_label(face: AxisGizmoFace) -> &'static str {
    match face {
        AxisGizmoFace::PositiveX => "+X",
        AxisGizmoFace::NegativeX => "-X",
        AxisGizmoFace::PositiveY => "+Y",
        AxisGizmoFace::NegativeY => "-Y",
        AxisGizmoFace::PositiveZ => "+Z",
        AxisGizmoFace::NegativeZ => "-Z",
    }
}

fn truncate_label(label: &str, max_chars: usize) -> String {
    let mut chars = label.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_is_bottom_left_inside_sixteen_nine_frame() {
        // 1600x1000 window holds a 1600x900 frame centered vertically, so
        // the canvas hugs the frame's bottom-left corner, not the window's.
        let layout = axis_gizmo_canvas_layout(Vec2::new(1600.0, 1000.0)).expect("layout");

        assert_eq!(layout.top_left.x, 18.0);
        assert_eq!(layout.top_left.y, 680.0);
        assert_eq!(layout.size, 252.0);
    }

    #[test]
    fn cursor_inside_canvas_maps_to_gizmo_texture() {
        let window = Window {
            resolution: (1600.0, 1000.0).into(),
            ..default()
        };

        assert_eq!(
            cursor_in_canvas(&window, Vec2::new(144.0, 806.0)),
            Some(Vec2::splat(256.0))
        );
        assert_eq!(cursor_in_canvas(&window, Vec2::new(17.0, 806.0)), None);
    }

    fn mesh_positions(mesh: &Mesh) -> Vec<Vec3> {
        let Some(bevy::render::mesh::VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("expected float32x3 position attribute");
        };
        positions
            .iter()
            .map(|position| Vec3::from(*position))
            .collect()
    }

    fn assert_mesh_winds_outward(mesh: &Mesh, subdivisions: u32) {
        let positions = mesh_positions(mesh);
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("expected u32 indices");
        };
        // (n+1)(n+2)/2 vertices and n^2 triangles for an n-subdivided grid.
        assert_eq!(
            positions.len(),
            ((subdivisions + 1) * (subdivisions + 2) / 2) as usize
        );
        assert_eq!(indices.len() / 3, (subdivisions * subdivisions) as usize);
        for triangle in indices.chunks_exact(3) {
            let [a, b, c] = [
                positions[triangle[0] as usize],
                positions[triangle[1] as usize],
                positions[triangle[2] as usize],
            ];
            let normal = (b - a).cross(c - a);
            let centroid = (a + b + c) / 3.0;
            // A triangle wound counter-clockwise as seen from outside the
            // sphere has a normal pointing away from the origin, matching
            // the centroid direction; back-face culling would otherwise hide
            // these tiles from the gizmo camera.
            assert!(normal.dot(centroid) > 0.0);
        }
    }

    #[test]
    fn spherical_octant_mesh_vertices_all_lie_on_the_sphere() {
        let radius = 0.46;
        let mesh = spherical_octant_mesh([Vec3::X, Vec3::Y, Vec3::Z], radius, 4);

        for position in mesh_positions(&mesh) {
            assert!((position.length() - radius).abs() < 0.0001);
        }
    }

    #[test]
    fn spherical_octant_mesh_all_positive_octant_winds_outward() {
        // Even number (zero) of negative signs: natural corner order faces
        // outward without swapping.
        let corners = octant_corners([1.0, 1.0, 1.0]);
        let mesh = spherical_octant_mesh(corners, 0.46, 3);

        assert_mesh_winds_outward(&mesh, 3);
    }

    #[test]
    fn spherical_octant_mesh_single_flip_octant_winds_outward() {
        // Odd number (one) of negative signs: this only winds outward
        // because `octant_corners` swaps 2 corners to correct the
        // reflection's handedness.
        let corners = octant_corners([1.0, -1.0, 1.0]);
        let mesh = spherical_octant_mesh(corners, 0.46, 3);

        assert_mesh_winds_outward(&mesh, 3);
    }

    #[test]
    fn octant_signs_cover_every_sign_combination_once() {
        let mut signs = octant_signs();
        signs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut expected = [[0.0; 3]; 8];
        let mut index = 0;
        for x in [-1.0f32, 1.0] {
            for y in [-1.0f32, 1.0] {
                for z in [-1.0f32, 1.0] {
                    expected[index] = [x, y, z];
                    index += 1;
                }
            }
        }
        assert_eq!(signs, expected);
    }

    #[test]
    fn octant_average_color_is_bright_but_not_neon() {
        for signs in octant_signs() {
            let hsl = Hsla::from(octant_average_color(signs).to_linear());
            assert!(
                hsl.saturation <= GIZMO_TILE_MAX_SATURATION + 0.0001,
                "saturation {} exceeds the not-quite-neon cap",
                hsl.saturation
            );
            assert!(
                hsl.lightness >= GIZMO_TILE_MIN_LIGHTNESS - 0.0001
                    && hsl.lightness <= GIZMO_TILE_MAX_LIGHTNESS + 0.0001,
                "lightness {} outside the bright-but-not-muted band",
                hsl.lightness
            );
        }
    }

    #[test]
    fn ray_toward_positive_octant_hits_matching_diagonal() {
        let origin = Vec3::new(2.0, 2.0, 2.0);
        let direction = octant_direction_from_ray(origin, -origin.normalize()).expect(
            "a ray aimed straight at the origin along the +X+Y+Z diagonal should hit the sphere",
        );

        assert!((direction - Vec3::new(1.0, 1.0, 1.0).normalize()).length() < 0.01);
    }

    #[test]
    fn ray_missing_the_sphere_has_no_octant_direction() {
        let direction = octant_direction_from_ray(Vec3::new(10.0, 10.0, 10.0), Vec3::X);

        assert_eq!(direction, None);
    }

    #[test]
    fn generic_up_vector_is_perpendicular_to_direction() {
        let direction = Vec3::new(1.0, 1.0, 1.0).normalize();
        let up = axis_gizmo_generic_up_vector(direction);

        assert!(up.dot(direction).abs() < 0.0001);
        assert!((up.length() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn generic_up_vector_stays_stable_looking_straight_up() {
        let up = axis_gizmo_generic_up_vector(Vec3::Y);

        assert!(up.dot(Vec3::Y).abs() < 0.0001);
        assert!((up.length() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn wrap_text_to_pixel_width_keeps_short_text_on_one_line() {
        let lines = wrap_text_to_pixel_width("hello", 100.0, |text| text.chars().count() as f32);

        assert_eq!(lines, vec!["hello".to_owned()]);
    }

    #[test]
    fn wrap_text_to_pixel_width_breaks_on_word_boundaries_without_truncating() {
        let measure = |text: &str| text.chars().count() as f32;
        let lines = wrap_text_to_pixel_width("hello world foo", 5.0, measure);

        assert_eq!(
            lines,
            vec!["hello".to_owned(), "world".to_owned(), "foo".to_owned()]
        );
        assert!(!lines.iter().any(|line| line.contains('.')));
    }

    #[test]
    fn wrap_text_to_pixel_width_hard_wraps_a_single_overlong_word() {
        let measure = |text: &str| text.chars().count() as f32;
        let lines = wrap_text_to_pixel_width("abcdefghij", 4.0, measure);

        assert_eq!(
            lines,
            vec!["abcd".to_owned(), "efgh".to_owned(), "ij".to_owned()]
        );
        assert!(!lines.iter().any(|line| line.contains('.')));
    }

    #[test]
    fn gizmo_label_safe_half_extent_matches_brute_force_worst_case() {
        let anchor_distance = GIZMO_LABEL_DISTANCE;
        let half_fov = (GIZMO_CAMERA_FOV_DEGREES * 0.5).to_radians();
        let samples = 200_000;
        let mut worst_margin = f32::INFINITY;
        for i in 0..=samples {
            let t = -anchor_distance + 2.0 * anchor_distance * (i as f32 / samples as f32);
            let depth = GIZMO_CAMERA_DISTANCE + t;
            let perpendicular = (anchor_distance * anchor_distance - t * t).max(0.0).sqrt();
            let margin = depth * half_fov.tan() - perpendicular;
            worst_margin = worst_margin.min(margin);
        }

        let safe_extent = gizmo_label_safe_half_extent(anchor_distance);
        assert!((worst_margin - safe_extent).abs() < 0.001);
    }

    /// Regression guard for the clipping bug this module used to have: the
    /// old hardcoded `GIZMO_LABEL_MAX_WORLD_WIDTH`/hover-width constants
    /// allowed labels wider than the gizmo camera could actually show at
    /// some orientations, so part of the text was silently clipped by the
    /// render target's edge instead of being scaled down to fit.
    #[test]
    fn gizmo_label_safe_half_extent_is_smaller_than_old_unsafe_caps() {
        let axis_extent = gizmo_label_safe_half_extent(GIZMO_LABEL_DISTANCE) * 2.0;

        assert!(axis_extent > 0.0 && axis_extent < GIZMO_LABEL_MAX_WORLD_WIDTH);
    }

    #[test]
    fn ray_hits_positive_z_cap_before_cube() {
        let face = axis_gizmo_face_from_ray(Vec3::new(0.0, 0.0, 4.0), Vec3::NEG_Z)
            .expect("face should be hit");

        assert_eq!(face, AxisGizmoFace::PositiveZ);
    }

    #[test]
    fn signed_faces_map_to_expected_view_directions() {
        assert_eq!(
            axis_gizmo_view_direction(AxisGizmoFace::NegativeX),
            Vec3::NEG_X
        );
        assert_eq!(axis_gizmo_up_vector(AxisGizmoFace::PositiveY), Vec3::Z);
    }
}
