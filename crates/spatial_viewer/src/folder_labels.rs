//! What a folder shows around its cube: a tag at its lower left corner,
//! with how many images it holds on a badge and its name beside it, and,
//! while it is open, a close icon at its upper right corner. The corners are
//! picked as the view sees the cube, so both follow the camera round it.
//!
//! A folder's own name is the value of each axis everything in it shares,
//! each in its axis's color; a name the user gave it is white.
//!
//! Both face the camera and follow the cube as it springs. A press on the
//! tag selects its folder, and one on the close icon closes it
//! ([`FolderHandles`]): an open folder takes no other press, so everything
//! inside or behind it stays reachable.
//!
//! Tags show while the setting for every folder says so, unless a folder
//! says otherwise, and close icons while their setting says so (see
//! [`FolderDisplay`](spatial_viewer_ui::FolderDisplay)); a handle that does not show takes no press. Only folders near the camera get tags, a few more each
//! frame, so a scene of thousands of folders pays for the ones the user can
//! read.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use image::{Rgba, RgbaImage};
use spatial_viewer_ui::FolderControls;

use crate::axis_gizmo::{blend_label_pixel, render_tag, AXIS_TEXT_COLORS};
use crate::folders::{Folder, FolderKey, FolderShell};
use crate::image_loading::BillboardLabelFont;
use crate::video_controls::ray_rect_hit;
use crate::{ExplorerScene, FlyCamera};

const TAG_FONT_SIZE: f32 = 44.0;
/// A tag's height, in cubes.
const TAG_HEIGHT: f32 = 0.15;
const NAME_COLOR: [u8; 4] = [245, 247, 250, 255];
/// Between the values of a folder's own name.
const NAME_GAP: &str = "   ";
const BADGE_BACKGROUND: [u8; 4] = [52, 124, 226, 240];
const BADGE_COLOR: [u8; 4] = [255, 255, 255, 255];
/// A close icon's size as a fraction of its distance from the camera, so it
/// reads alike near and far, within the range below, in cubes.
const CLOSE_ICON_SCREEN_FRACTION: f32 = 0.04;
const CLOSE_ICON_MIN_SIZE: f32 = 0.1;
const CLOSE_ICON_MAX_SIZE: f32 = 0.5;
const CLOSE_ICON_TEXTURE_SIDE: u32 = 128;
/// How far toward the camera tags and icons stand off what they are drawn
/// over, in cubes.
const TOWARD_CAMERA: f32 = 0.03;
/// Folders nearer the camera than this, in cubes from their cube's surface,
/// keep their tags.
const TAG_RANGE: f32 = 36.0;
const TAG_SPAWNS_PER_FRAME: usize = 12;

/// What a tag says: the count on its badge, and its name in colored spans.
#[derive(Clone, Debug, PartialEq)]
struct TagContent {
    count: usize,
    name: Vec<(String, [u8; 4])>,
}

impl TagContent {
    fn of(folder: &Folder) -> Self {
        let shared: Vec<(String, [u8; 4])> = folder
            .shared_coordinates
            .iter()
            .enumerate()
            .filter_map(|(axis, value)| {
                Some((value.as_deref()?.to_owned(), AXIS_TEXT_COLORS[axis]))
            })
            .collect();
        let name = if folder.custom_name.is_some() || shared.is_empty() {
            vec![(folder.name(), NAME_COLOR)]
        } else {
            let mut spans = Vec::with_capacity(shared.len() * 2);
            for (index, span) in shared.into_iter().enumerate() {
                if index > 0 {
                    spans.push((NAME_GAP.to_owned(), NAME_COLOR));
                }
                spans.push(span);
            }
            spans
        };
        Self {
            count: folder.image_count,
            name,
        }
    }
}

/// A folder's tag, at its lower left corner.
#[derive(Component)]
pub(crate) struct FolderTag {
    key: FolderKey,
    content: TagContent,
    size: Vec2,
    image: Handle<Image>,
    material: Handle<StandardMaterial>,
}

/// An open folder's close icon, at its upper right corner.
#[derive(Component)]
pub(crate) struct FolderCloseIcon {
    key: FolderKey,
}

/// The quad every tag and icon is drawn on, and the close icon's material.
#[derive(Resource)]
pub(crate) struct FolderHandleAssets {
    quad: Handle<Mesh>,
    close_icon: Handle<StandardMaterial>,
}

impl FolderHandleAssets {
    pub(crate) fn new(
        meshes: &mut Assets<Mesh>,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
    ) -> Self {
        Self {
            quad: meshes.add(Rectangle::new(1.0, 1.0)),
            close_icon: materials.add(label_material(images.add(close_icon_image()))),
        }
    }
}

fn label_material(texture: Handle<Image>) -> StandardMaterial {
    StandardMaterial {
        base_color_texture: Some(texture),
        // Label textures are premultiplied; see `blend_label_pixel`.
        alpha_mode: AlphaMode::Premultiplied,
        cull_mode: None,
        unlit: true,
        ..default()
    }
}

/// A dark disc with a light ring and a bar across it: minimize.
fn close_icon_image() -> Image {
    let side = CLOSE_ICON_TEXTURE_SIDE;
    let mut rgba = RgbaImage::from_pixel(side, side, Rgba([0, 0, 0, 0]));
    let center = side as f32 * 0.5;
    let radius = center - 2.0;
    let ring = side as f32 * 0.05;
    let bar = Vec2::new(side as f32 * 0.24, side as f32 * 0.045);
    for y in 0..side {
        for x in 0..side {
            let point = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - Vec2::splat(center);
            let from_edge = radius - point.length();
            blend_label_pixel(
                &mut rgba,
                x as i32,
                y as i32,
                from_edge + 0.5,
                [22, 26, 34, 225],
            );
            let on_ring = (ring - (from_edge - ring).abs()).clamp(-0.5, 0.5) + 0.5;
            blend_label_pixel(&mut rgba, x as i32, y as i32, on_ring, [232, 238, 246, 255]);
            let outside_bar = (point.abs() - bar).max_element();
            blend_label_pixel(
                &mut rgba,
                x as i32,
                y as i32,
                0.5 - outside_bar,
                [240, 244, 250, 255],
            );
        }
    }
    Image::new(
        bevy::render::render_resource::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        rgba.into_raw(),
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        bevy::render::render_asset::RenderAssetUsages::MAIN_WORLD
            | bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD,
    )
}

/// What a press on a folder's handle does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FolderHandleKind {
    /// Its tag: selects the folder.
    Label,
    /// Its close icon: closes the folder.
    Close,
}

/// A handle a ray crossed.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FolderHandleHit {
    pub key: FolderKey,
    pub kind: FolderHandleKind,
    pub distance: f32,
}

/// Where every folder handle stood this frame, for presses to test.
#[derive(Resource, Default)]
pub(crate) struct FolderHandles {
    placed: Vec<PlacedHandle>,
}

struct PlacedHandle {
    key: FolderKey,
    kind: FolderHandleKind,
    transform: GlobalTransform,
    half_size: Vec2,
}

impl FolderHandles {
    /// The nearest handle along the ray. A close icon is round.
    pub(crate) fn hit(&self, ray_origin: Vec3, ray_direction: Vec3) -> Option<FolderHandleHit> {
        self.placed
            .iter()
            .filter_map(|handle| {
                let (distance, local) = ray_rect_hit(
                    ray_origin,
                    ray_direction,
                    &handle.transform,
                    handle.half_size,
                )?;
                let round = handle.kind == FolderHandleKind::Close;
                (!round || local.length() <= handle.half_size.x).then(|| FolderHandleHit {
                    key: handle.key.clone(),
                    kind: handle.kind,
                    distance,
                })
            })
            .min_by(|left, right| left.distance.total_cmp(&right.distance))
    }
}

/// How far the camera is from a folder's cube, in world units: zero inside.
fn distance_to_box(point: Vec3, center: Vec3, size: Vec3) -> f32 {
    ((point - center).abs() - size * 0.5)
        .max(Vec3::ZERO)
        .length()
}

/// Which way corner `corner` (0 to 7) of a cube lies from its center.
fn corner_sign(corner: usize) -> Vec3 {
    let sign = |bit: usize| if corner & bit == 0 { -1.0 } else { 1.0 };
    Vec3::new(sign(1), sign(2), sign(4))
}

/// The corner of the cube at `center`, `size` that the camera sees furthest
/// toward screen direction `toward` (right and up components): among the
/// corners in front of the camera, the one whose projection reaches
/// furthest that way. `None` when no corner is in front.
fn screen_corner(camera: &Transform, center: Vec3, size: Vec3, toward: Vec2) -> Option<Vec3> {
    let right = camera.rotation * Vec3::X;
    let up = camera.rotation * Vec3::Y;
    let forward = camera.rotation * Vec3::NEG_Z;
    (0..8)
        .map(|corner| center + size * 0.5 * corner_sign(corner))
        .filter_map(|corner| {
            let offset = corner - camera.translation;
            let depth = offset.dot(forward);
            (depth > f32::EPSILON).then(|| {
                let on_screen = Vec2::new(offset.dot(right), offset.dot(up)) / depth;
                (on_screen.dot(toward), corner)
            })
        })
        .max_by(|(left, _), (right, _)| left.total_cmp(right))
        .map(|(_, corner)| corner)
}

type TagQuery<'w, 's> = Query<'w, 's, (Entity, &'static FolderTag)>;
type CloseIconQuery<'w, 's> = Query<'w, 's, (Entity, &'static FolderCloseIcon)>;

/// Gives the folders near the camera their tags and, while open, close
/// icons; re-renders the tags whose text changed, and takes both from
/// folders that went out of range or away.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sync_folder_labels(
    mut commands: Commands,
    scene: Res<ExplorerScene>,
    controls: Res<FolderControls>,
    font: Res<BillboardLabelFont>,
    assets: Res<FolderHandleAssets>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    camera: Query<&Transform, With<FlyCamera>>,
    tags: TagQuery,
    icons: CloseIconQuery,
) {
    let (Some(font), Ok(camera)) = (font.0.as_ref(), camera.get_single()) else {
        return;
    };
    let cube_size = scene.billboard_world_size;
    let range = TAG_RANGE * cube_size;
    let folders = &scene.folders;
    let distance =
        |folder: &Folder| distance_to_box(camera.translation, folder.center, folder.size);
    let in_range = |folder: &Folder| folder.visible && distance(folder) <= range;
    let wanted = |key: &FolderKey| folders.get(key).filter(|folder| in_range(folder));
    // A folder's own tag setting wins over the one for every folder.
    let tag_shown = |folder: &Folder| folder.tag_override.unwrap_or(controls.display.tags);
    let icon_shown = |folder: &Folder| folder.open && controls.display.close_icons;

    let mut tagged = HashSet::new();
    for (entity, tag) in &tags {
        if wanted(&tag.key)
            .is_some_and(|folder| tag_shown(folder) && TagContent::of(folder) == tag.content)
        {
            tagged.insert(tag.key.clone());
        } else {
            images.remove(&tag.image);
            materials.remove(&tag.material);
            commands.entity(entity).despawn();
        }
    }
    let mut iconed = HashSet::new();
    for (entity, icon) in &icons {
        if wanted(&icon.key).is_some_and(icon_shown) {
            iconed.insert(icon.key.clone());
        } else {
            commands.entity(entity).despawn();
        }
    }

    let mut missing: Vec<&Folder> = folders
        .folders
        .iter()
        .filter(|folder| {
            in_range(folder)
                && ((tag_shown(folder) && !tagged.contains(&folder.key))
                    || (icon_shown(folder) && !iconed.contains(&folder.key)))
        })
        .collect();
    missing.sort_by(|left, right| distance(left).total_cmp(&distance(right)));
    for folder in missing.into_iter().take(TAG_SPAWNS_PER_FRAME) {
        if tag_shown(folder) && !tagged.contains(&folder.key) {
            let content = TagContent::of(folder);
            let spans: Vec<(&str, [u8; 4])> = content
                .name
                .iter()
                .map(|(text, color)| (text.as_str(), *color))
                .collect();
            let rendered = render_tag(
                font,
                &content.count.to_string(),
                &spans,
                TAG_FONT_SIZE,
                TAG_HEIGHT * cube_size,
                BADGE_BACKGROUND,
                BADGE_COLOR,
            );
            let image = images.add(rendered.image);
            let material = materials.add(label_material(image.clone()));
            commands.spawn((
                Mesh3d(assets.quad.clone()),
                MeshMaterial3d(material.clone()),
                Transform::from_translation(folder.center),
                Visibility::Hidden,
                FolderTag {
                    key: folder.key.clone(),
                    content,
                    size: rendered.world_size,
                    image,
                    material,
                },
                Name::new("folder tag"),
            ));
        }
        if icon_shown(folder) && !iconed.contains(&folder.key) {
            commands.spawn((
                Mesh3d(assets.quad.clone()),
                MeshMaterial3d(assets.close_icon.clone()),
                Transform::from_translation(folder.center),
                Visibility::Hidden,
                FolderCloseIcon {
                    key: folder.key.clone(),
                },
                Name::new("folder close icon"),
            ));
        }
    }
}

type ShellTransforms<'w, 's> = Query<
    'w,
    's,
    (&'static FolderShell, &'static Transform),
    (Without<FolderTag>, Without<FolderCloseIcon>),
>;
type CameraTransform<'w, 's> = Query<
    'w,
    's,
    &'static Transform,
    (
        With<FlyCamera>,
        Without<FolderTag>,
        Without<FolderCloseIcon>,
    ),
>;
type PlacedTags<'w, 's> = Query<
    'w,
    's,
    (
        &'static FolderTag,
        &'static mut Transform,
        &'static mut Visibility,
    ),
    Without<FolderCloseIcon>,
>;
type PlacedIcons<'w, 's> = Query<
    'w,
    's,
    (
        &'static FolderCloseIcon,
        &'static mut Transform,
        &'static mut Visibility,
    ),
    Without<FolderTag>,
>;

/// Stands every tag and close icon at its folder's cube as the cube is drawn
/// this frame, facing the camera, and records where each handle stands for
/// presses.
pub(crate) fn place_folder_handles(
    scene: Res<ExplorerScene>,
    mut handles: ResMut<FolderHandles>,
    camera: CameraTransform,
    shells: ShellTransforms,
    mut tags: PlacedTags,
    mut icons: PlacedIcons,
) {
    handles.placed.clear();
    let Ok(camera) = camera.get_single() else {
        return;
    };
    let cube_size = scene.billboard_world_size;
    let cubes: HashMap<&FolderKey, (Vec3, Vec3)> = shells
        .iter()
        .map(|(shell, transform)| (shell.key(), (transform.translation, transform.scale)))
        .collect();
    let right = camera.rotation * Vec3::X;
    let up = camera.rotation * Vec3::Y;
    let toward_camera = |point: Vec3| {
        point + (camera.translation - point).normalize_or_zero() * TOWARD_CAMERA * cube_size
    };
    // The cube of a folder that shows, as drawn this frame.
    let cube = |key: &FolderKey| {
        let &(center, size) = cubes.get(key)?;
        let shows = size.min_element() > cube_size * 0.05
            && scene.folders.get(key).is_some_and(|folder| folder.visible);
        shows.then_some((center, size))
    };
    let mut place = |transform: &mut Mut<Transform>,
                     visibility: &mut Mut<Visibility>,
                     key: &FolderKey,
                     kind: FolderHandleKind,
                     center: Option<Vec3>,
                     size: Vec2| {
        let Some(center) = center else {
            visibility.set_if_neq(Visibility::Hidden);
            return;
        };
        let placed = Transform {
            translation: center,
            rotation: camera.rotation,
            scale: size.extend(1.0),
        };
        transform.set_if_neq(placed);
        visibility.set_if_neq(Visibility::Inherited);
        handles.placed.push(PlacedHandle {
            key: key.clone(),
            kind,
            transform: GlobalTransform::from(Transform {
                scale: Vec3::ONE,
                ..placed
            }),
            half_size: size * 0.5,
        });
    };

    for (tag, mut transform, mut visibility) in &mut tags {
        // Hangs from the corner, along the cube's bottom edge.
        let center = cube(&tag.key)
            .and_then(|(center, size)| screen_corner(camera, center, size, Vec2::new(-1.0, -1.0)))
            .map(|corner| toward_camera(corner + right * tag.size.x * 0.5 - up * tag.size.y * 0.5));
        place(
            &mut transform,
            &mut visibility,
            &tag.key,
            FolderHandleKind::Label,
            center,
            tag.size,
        );
    }
    for (icon, mut transform, mut visibility) in &mut icons {
        let corner = cube(&icon.key)
            .and_then(|(center, size)| screen_corner(camera, center, size, Vec2::new(1.0, 1.0)));
        let size = corner.map_or(0.0, |corner| {
            (corner.distance(camera.translation) * CLOSE_ICON_SCREEN_FRACTION).clamp(
                CLOSE_ICON_MIN_SIZE * cube_size,
                CLOSE_ICON_MAX_SIZE * cube_size,
            )
        });
        place(
            &mut transform,
            &mut visibility,
            &icon.key,
            FolderHandleKind::Close,
            corner.map(toward_camera),
            Vec2::splat(size),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_cover_every_sign_once() {
        let mut signs: Vec<[i32; 3]> = (0..8)
            .map(|corner| corner_sign(corner).as_ivec3().to_array())
            .collect();
        signs.sort_unstable();
        signs.dedup();
        assert_eq!(signs.len(), 8);
    }

    #[test]
    fn screen_corners_are_the_ones_the_view_sees_there() {
        // Looking down -Z at a unit cube 10 away, slightly from its left.
        let camera = Transform::from_xyz(-0.2, 0.0, 10.0);
        let upper_right = screen_corner(&camera, Vec3::ZERO, Vec3::ONE, Vec2::ONE);
        // The near face's corner reaches further out on screen.
        assert_eq!(upper_right, Some(Vec3::new(0.5, 0.5, 0.5)));
        let lower_left = screen_corner(&camera, Vec3::ZERO, Vec3::ONE, Vec2::NEG_ONE);
        assert_eq!(lower_left, Some(Vec3::new(-0.5, -0.5, 0.5)));
        // Behind the camera, nothing.
        assert_eq!(
            screen_corner(&camera, Vec3::new(0.0, 0.0, 20.0), Vec3::ONE, Vec2::ONE),
            None
        );
    }

    #[test]
    fn a_close_icon_takes_presses_on_its_disc_only() {
        let mut handles = FolderHandles::default();
        handles.placed.push(PlacedHandle {
            key: FolderKey::Manual(0),
            kind: FolderHandleKind::Close,
            transform: GlobalTransform::from_translation(Vec3::new(0.0, 0.0, -5.0)),
            half_size: Vec2::splat(1.0),
        });
        let hit = handles
            .hit(Vec3::new(0.5, 0.5, 0.0), Vec3::NEG_Z)
            .expect("inside the disc");
        assert_eq!(hit.kind, FolderHandleKind::Close);
        assert!((hit.distance - 5.0).abs() < 1e-4);
        // The square's corner lies outside the disc.
        assert!(handles.hit(Vec3::new(0.9, 0.9, 0.0), Vec3::NEG_Z).is_none());
    }

    #[test]
    fn the_camera_inside_a_folder_is_no_distance_from_it() {
        assert_eq!(
            distance_to_box(Vec3::ZERO, Vec3::ZERO, Vec3::splat(4.0)),
            0.0
        );
        assert!(
            (distance_to_box(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, Vec3::splat(4.0)) - 3.0).abs()
                < 1e-5
        );
    }
}
