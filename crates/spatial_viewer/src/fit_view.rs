//! Fitting the view to the selection, or to everything: the camera flies,
//! without turning, to where what it fits reaches the edges of the screen.

use bevy::prelude::*;
use spatial_viewer_ui::{Action, Animation, AnimationSettings, ControlInput, PauseMenuState};

use crate::folders::BillboardHover;
use crate::image_loading::{ImageLoadingState, MediaBillboard};
use crate::manual_spacing::SelectionState;
use crate::{ExplorerScene, FlyCamera};

const FLIGHT_SECONDS: f32 = 0.35;
/// Flying by hand ends a flight.
const FLIGHT_ENDING_ACTIONS: [Action; 7] = [
    Action::MoveForward,
    Action::MoveBack,
    Action::MoveLeft,
    Action::MoveRight,
    Action::MoveUp,
    Action::MoveDown,
    Action::Look,
];
/// Closest a fit brings the camera to what it frames, in cubes.
const MIN_FIT_DISTANCE_CUBES: f32 = 0.6;

/// The camera's flight to a fitted view, while one is under way.
#[derive(Resource, Default)]
pub(crate) struct CameraFlight {
    flight: Option<Flight>,
}

struct Flight {
    from: Vec3,
    to: Vec3,
    elapsed: f32,
}

/// Starts a flight that fits the selection to the view when the fit key is
/// pressed, or everything the scene shows when the fit-all key is.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_selection_to_view(
    input: ControlInput,
    pause_menu: Res<PauseMenuState>,
    selection: Res<SelectionState>,
    scene: Res<ExplorerScene>,
    loading: Res<ImageLoadingState>,
    hover: Res<BillboardHover>,
    billboards: Query<&MediaBillboard>,
    camera: Query<(&Transform, &Projection), With<FlyCamera>>,
    mut flight: ResMut<CameraFlight>,
) {
    let everything = input.just_pressed(Action::FitAll);
    if pause_menu.paused || !(everything || input.just_pressed(Action::FitSelection)) {
        return;
    }
    let Ok((camera_transform, Projection::Perspective(perspective))) = camera.get_single() else {
        return;
    };
    let cube_size = scene.billboard_world_size;
    // Each picture where it rests, square to the view, as it turns to face
    // the camera once it is centered in it.
    let right = camera_transform.rotation * Vec3::X;
    let up = camera_transform.rotation * Vec3::Y;
    let picture = |center: Vec3, half: Vec2| {
        [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .map(|(x, y)| center + right * (x * half.x) + up * (y * half.y))
    };
    let picture_corners: Vec<Vec3> = if everything {
        // Every picture the scene shows full size, loaded or not, as large
        // as its cube.
        scene
            .image_points
            .iter()
            .filter(|point| point.scale >= 1.0)
            .flat_map(|point| picture(point.position, Vec2::splat(cube_size * 0.5 * point.scale)))
            .collect()
    } else {
        // At the size it settles at, grown while under the pointer.
        billboards
            .iter()
            .filter(|billboard| selection.is_selected(billboard.image_id))
            .filter_map(|billboard| {
                let point = loading.loaded_point(billboard.image_id)?;
                let half = billboard.content_half_extents(cube_size)
                    * point.scale
                    * hover.growth(billboard.image_id);
                Some(picture(point.position, half))
            })
            .flatten()
            .collect()
    };
    let folders: Vec<_> = if everything {
        scene.folders.folders.iter().collect()
    } else {
        selection
            .selected_folders()
            .filter_map(|key| scene.folders.get(key))
            .collect()
    };
    let folder_corners = folders
        .into_iter()
        .filter(|folder| folder.visible)
        .flat_map(|folder| {
            let half = folder.size * 0.5;
            (0..8).map(move |corner| {
                let sign = Vec3::new(
                    if corner & 1 == 0 { -1.0 } else { 1.0 },
                    if corner & 2 == 0 { -1.0 } else { 1.0 },
                    if corner & 4 == 0 { -1.0 } else { 1.0 },
                );
                folder.center + half * sign
            })
        });
    let corners: Vec<Vec3> = picture_corners.into_iter().chain(folder_corners).collect();
    fly_to_fit(
        &mut flight,
        camera_transform,
        perspective,
        &corners,
        cube_size,
    );
}

/// Starts a flight, without turning, to where `corners` fill the view.
pub(crate) fn fly_to_fit(
    flight: &mut CameraFlight,
    camera_transform: &Transform,
    perspective: &PerspectiveProjection,
    corners: &[Vec3],
    cube_size: f32,
) {
    let Some((center, distance)) = fit_view(
        corners,
        camera_transform.rotation,
        (perspective.fov * 0.5).tan(),
        perspective.aspect_ratio,
    ) else {
        return;
    };
    let distance = distance.max(MIN_FIT_DISTANCE_CUBES * cube_size);
    flight.flight = Some(Flight {
        from: camera_transform.translation,
        to: center - camera_transform.forward() * distance,
        elapsed: 0.0,
    });
}

/// Carries the camera along its flight, stilling whatever speed it had, or
/// straight to its end while [`Animation::CameraFlights`] does not play.
/// Flying by hand ends it.
pub(crate) fn fly_camera_to_fit(
    real_time: Res<Time<Real>>,
    animations: Res<AnimationSettings>,
    input: ControlInput,
    mut flight: ResMut<CameraFlight>,
    mut camera: Query<(&mut Transform, &mut FlyCamera)>,
) {
    let Some(active) = flight.flight.as_mut() else {
        return;
    };
    let Ok((mut transform, mut fly_camera)) = camera.get_single_mut() else {
        return;
    };
    if FLIGHT_ENDING_ACTIONS
        .iter()
        .any(|&action| input.pressed(action))
    {
        flight.flight = None;
        return;
    }
    fly_camera.velocity = Vec3::ZERO;
    active.elapsed += animations.animation_seconds(real_time.delta_secs());
    let progress = if animations.plays(Animation::CameraFlights) {
        (active.elapsed / FLIGHT_SECONDS).min(1.0)
    } else {
        1.0
    };
    // Eases out, so the camera settles into the fitted view.
    let eased = 1.0 - (1.0 - progress).powi(3);
    transform.translation = active.from.lerp(active.to, eased);
    if progress >= 1.0 {
        flight.flight = None;
    }
}

/// Where to center a view turned by `rotation` on `corners`, and how far
/// back from that center the camera stands so every corner shows, the
/// outermost at the edge of the view. `tan_half_fov` is the tangent of half
/// the vertical field of view; `aspect` is width over height. `None`
/// without corners.
fn fit_view(
    corners: &[Vec3],
    rotation: Quat,
    tan_half_fov: f32,
    aspect: f32,
) -> Option<(Vec3, f32)> {
    let right = rotation * Vec3::X;
    let up = rotation * Vec3::Y;
    let forward = rotation * Vec3::NEG_Z;
    let origin = *corners.first()?;
    let local = |corner: Vec3| {
        let offset = corner - origin;
        Vec3::new(offset.dot(right), offset.dot(up), offset.dot(forward))
    };
    let (low, high) = corners.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(low, high), &corner| (low.min(local(corner)), high.max(local(corner))),
    );
    let middle = (low + high) * 0.5;
    let center = origin + right * middle.x + up * middle.y + forward * middle.z;
    let tan_half_width = tan_half_fov * aspect;
    let distance = corners
        .iter()
        .map(|&corner| {
            let offset = local(corner) - middle;
            // Nearer corners need the camera further back to fit.
            let across = (offset.x.abs() / tan_half_width).max(offset.y.abs() / tan_half_fov);
            across - offset.z
        })
        .fold(0.0, f32::max);
    Some((center, distance))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_facing_the_camera_fills_the_narrower_side() {
        let tan_half_fov = (30f32).to_radians().tan();
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .map(|(x, y)| Vec3::new(x + 5.0, y, -20.0));
        let (center, distance) = fit_view(&corners, Quat::IDENTITY, tan_half_fov, 16.0 / 9.0)
            .expect("corners were given");
        assert!(center.distance(Vec3::new(5.0, 0.0, -20.0)) < 1e-4);
        assert!((distance - 1.0 / tan_half_fov).abs() < 1e-4);
    }

    #[test]
    fn a_deep_selection_backs_off_for_its_nearest_corner() {
        let tan_half_fov = 1.0;
        // Two unit squares, one 10 behind the other along the view.
        let corners: Vec<Vec3> = [0.0, -10.0]
            .into_iter()
            .flat_map(|z| [(-1.0, -1.0), (1.0, 1.0)].map(move |(x, y)| Vec3::new(x, y, z)))
            .collect();
        let (center, distance) =
            fit_view(&corners, Quat::IDENTITY, tan_half_fov, 1.0).expect("corners were given");
        assert!(center.distance(Vec3::new(0.0, 0.0, -5.0)) < 1e-4);
        // The near square sits 5 in front of the center.
        assert!((distance - 6.0).abs() < 1e-4);
        assert!(fit_view(&[], Quat::IDENTITY, 1.0, 1.0).is_none());
    }
}
