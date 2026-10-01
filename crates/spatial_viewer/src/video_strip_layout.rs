//! Geometry for video control strips.
//!
//! A strip is sized and positioned in screen space — overlaid along the
//! bottom edge of the part of its video's picture inside the window, never
//! reaching past that part, a constant height in pixels unless the picture
//! is too small for it, where it shrinks to fit — but lives in the world:
//! it is placed on a
//! camera-facing plane just in front of the nearest visible point of its
//! video, scaled so one local unit is one window pixel. Other content in
//! front of the video therefore occludes the strip like any other geometry,
//! while the strip keeps a readable, screen-derived size at any distance.
//!
//! Strip-local space: x right, y up, origin at the strip's center, one unit
//! per window pixel; larger z is closer to the camera.
//!
//! While strips do not follow the view, a strip is instead fixed on its
//! video's picture ([`place_strip_on_picture`]): laid out as if the picture
//! showed at a set size on screen, it moves, turns and scales with the
//! picture alone.

use bevy::prelude::*;

/// Strip height, in window pixels.
pub(crate) const STRIP_HEIGHT: f32 = 40.0;
const STRIP_MIN_WIDTH: f32 = 340.0;
const STRIP_MAX_WIDTH: f32 = 720.0;
/// Closest a strip may come to a window edge.
const STRIP_WINDOW_MARGIN: f32 = 12.0;
/// Clip depth as a multiple of the camera's near plane; a hair beyond it so
/// clipped vertices are guaranteed to project.
const NEAR_PLANE_CLIP_MARGIN: f32 = 1.001;
/// Fraction of its video's nearest depth a strip is pulled toward the
/// camera, so the video itself never cuts into it.
const STRIP_DEPTH_PULL: f32 = 0.02;
/// Closest a strip may come to the camera, as a multiple of the near plane.
const STRIP_MIN_DEPTH_NEAR_PLANES: f32 = 1.05;
/// A strip fixed on its picture is laid out as if the picture's longer side
/// showed this many pixels long.
const FIXED_STRIP_PICTURE_SIDE: f32 = 800.0;
/// How far in front of its picture a fixed strip stands, as a fraction of
/// the picture's longer half side, so the picture never cuts into it.
const FIXED_STRIP_LIFT: f32 = 0.02;

const STRIP_PADDING: f32 = 8.0;
const STRIP_GAP: f32 = 10.0;
pub(crate) const BUTTON_SIZE: f32 = 28.0;
/// Play and pause glyph height.
pub(crate) const ICON_SIZE: f32 = 12.0;
/// Speaker glyph height; its sound waves or cross make it a little wider.
pub(crate) const SPEAKER_SIZE: f32 = 14.0;
/// Sliders are pressable over a thicker band than their drawn rail.
const SLIDER_HIT_THICKNESS: f32 = 24.0;
const RAIL_THICKNESS: f32 = 6.0;
/// Knob size along and across its slider.
const KNOB_ALONG: f32 = 8.0;
const KNOB_ACROSS: f32 = 18.0;
/// The volume popup: a vertical slider in a panel standing on the strip's
/// edge directly over the speaker button, so the pointer can move from the
/// button onto it without leaving the controls.
const VOLUME_POPUP_WIDTH: f32 = 32.0;
const VOLUME_POPUP_HEIGHT: f32 = 112.0;
const VOLUME_POPUP_PADDING: f32 = 14.0;
/// Time label glyph height; its slot is reserved at the widest text the
/// video's duration produces, so the timeline never jitters.
pub(crate) const TIME_LABEL_HEIGHT: f32 = 14.0;

/// Where a strip sits in the world and how wide it is on screen.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StripPlacement {
    /// Camera-facing, scaled so one local unit is one window pixel.
    pub transform: Transform,
    pub width: f32,
    /// The volume popup opens above the strip; below when the window's top
    /// edge leaves it no room there.
    pub volume_popup_upward: bool,
}

/// Places the strip for a video whose picture is the quad with
/// `half_extents` in `picture`'s local XY plane. `None` when no part of the
/// picture is both in front of the camera and inside the window, or when the
/// picture comes so close that no depth in front of it clears the near
/// plane — there the picture would cut through its own strip.
pub(crate) fn place_strip(
    camera: &Camera,
    camera_transform: &Transform,
    near: f32,
    picture: &Transform,
    half_extents: Vec2,
    window_size: Vec2,
) -> Option<StripPlacement> {
    let viewport_size = camera.logical_viewport_size()?;
    let camera_pose = GlobalTransform::from(*camera_transform);
    let eye = camera_transform.translation;
    let forward = *camera_transform.forward();
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(x, y)| {
        picture.transform_point(Vec3::new(x * half_extents.x, y * half_extents.y, 0.0))
    });
    let visible = clip_polygon_to_depth(&corners, eye, forward, near * NEAR_PLANE_CLIP_MARGIN);
    // The scene renders to an offscreen target stretched over the window.
    let viewport_per_window = viewport_size / window_size;
    let picture_rect = visible
        .iter()
        .filter_map(|point| camera.world_to_viewport(&camera_pose, *point).ok())
        .map(|point| point / viewport_per_window)
        .map(|point| Rect::from_corners(point, point))
        .reduce(|bounds, point| bounds.union(point))?;
    let strip = screen_strip(picture_rect, window_size)?;

    let nearest_depth = visible
        .iter()
        .map(|point| (*point - eye).dot(forward))
        .fold(f32::INFINITY, f32::min);
    let depth = nearest_depth * (1.0 - STRIP_DEPTH_PULL);
    if depth < near * STRIP_MIN_DEPTH_NEAR_PLANES {
        return None;
    }
    let point_at_depth = |window_point: Vec2| {
        let ray = camera
            .viewport_to_world(&camera_pose, window_point * viewport_per_window)
            .ok()?;
        let along = ray.direction.dot(forward);
        (along > f32::EPSILON).then(|| {
            let distance = (depth - (ray.origin - eye).dot(forward)) / along;
            ray.origin + *ray.direction * distance
        })
    };
    let center = strip.rect.center();
    let left = point_at_depth(Vec2::new(strip.rect.min.x, center.y))?;
    let right = point_at_depth(Vec2::new(strip.rect.max.x, center.y))?;
    let units_per_strip_unit = left.distance(right) / strip.width;
    Some(StripPlacement {
        transform: Transform {
            translation: point_at_depth(center)?,
            rotation: camera_transform.rotation,
            scale: Vec3::splat(units_per_strip_unit),
        },
        width: strip.width,
        volume_popup_upward: strip.rect.min.y
            >= VOLUME_POPUP_HEIGHT * strip.scale + STRIP_WINDOW_MARGIN,
    })
}

/// Places the strip for a video whose picture is the quad with
/// `half_extents` in `picture`'s local XY plane, fixed on the picture:
/// along its bottom edge as [`place_strip`] would put it were the picture's
/// longer side [`FIXED_STRIP_PICTURE_SIDE`] pixels on screen, whatever the
/// view.
pub(crate) fn place_strip_on_picture(picture: &Transform, half_extents: Vec2) -> StripPlacement {
    let longer_half_side = half_extents.max_element().max(f32::EPSILON);
    let pixels_per_unit = FIXED_STRIP_PICTURE_SIDE / (2.0 * longer_half_side);
    let picture_pixels = 2.0 * half_extents * pixels_per_unit;
    let strip = strip_in_area(Rect::from_corners(Vec2::ZERO, picture_pixels));
    // Picture pixels run y down from its top left corner, picture-local
    // units y up from its center.
    let center = strip.rect.center();
    let local = Vec2::new(
        center.x - picture_pixels.x * 0.5,
        picture_pixels.y * 0.5 - center.y,
    ) / pixels_per_unit;
    StripPlacement {
        transform: Transform {
            translation: picture.transform_point(local.extend(longer_half_side * FIXED_STRIP_LIFT)),
            rotation: picture.rotation,
            scale: Vec3::splat(picture.scale.x * strip.scale / pixels_per_unit),
        },
        width: strip.width,
        volume_popup_upward: true,
    }
}

/// A strip as the window shows it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ScreenStrip {
    /// Window rectangle, in logical pixels, y down.
    rect: Rect,
    /// Window pixels per strip-local unit: 1 at full size, less where the
    /// picture is too small for the strip.
    scale: f32,
    /// Width in strip-local units.
    width: f32,
}

/// The strip of a picture with on-screen bounds `picture_rect`: overlaid
/// along the bottom edge of the part of the picture inside the window (short
/// of its margin), centered on it, flush with that part's bottom edge and
/// never reaching past it. The strip keeps its full size while that part
/// holds it, and shrinks evenly to fit a smaller one, so its size and place
/// follow the picture's edges continuously: a picture that grows, shrinks
/// or recedes, as a hovered one does, moves its strip only with its edges.
/// `None` when no part of the picture is inside the window's margin.
fn screen_strip(picture_rect: Rect, window_size: Vec2) -> Option<ScreenStrip> {
    let window = Rect::from_corners(Vec2::ZERO, window_size).inflate(-STRIP_WINDOW_MARGIN);
    let area = picture_rect.intersect(window);
    (!area.is_empty()).then(|| strip_in_area(area))
}

/// The strip overlaid along the bottom of `area`, a picture's part on
/// screen; see [`screen_strip`].
fn strip_in_area(area: Rect) -> ScreenStrip {
    let scale = (area.width() / STRIP_MIN_WIDTH)
        .min(area.height() / STRIP_HEIGHT)
        .min(1.0);
    let width = (area.width() / scale).clamp(STRIP_MIN_WIDTH, STRIP_MAX_WIDTH);
    let size = Vec2::new(width, STRIP_HEIGHT) * scale;
    let bottom_center = Vec2::new(area.center().x, area.max.y);
    ScreenStrip {
        rect: Rect::from_corners(
            bottom_center - Vec2::new(size.x * 0.5, size.y),
            bottom_center + Vec2::new(size.x * 0.5, 0.0),
        ),
        scale,
        width,
    }
}

/// Clips a convex polygon to the half-space at least `min_depth` in front
/// of `eye` along `forward` (Sutherland–Hodgman against one plane).
fn clip_polygon_to_depth(polygon: &[Vec3], eye: Vec3, forward: Vec3, min_depth: f32) -> Vec<Vec3> {
    let depth = |point: Vec3| (point - eye).dot(forward) - min_depth;
    let mut clipped = Vec::with_capacity(polygon.len() + 1);
    for (index, &current) in polygon.iter().enumerate() {
        let next = polygon[(index + 1) % polygon.len()];
        let (current_depth, next_depth) = (depth(current), depth(next));
        if current_depth >= 0.0 {
            clipped.push(current);
        }
        if (current_depth >= 0.0) != (next_depth >= 0.0) {
            let crossing = current_depth / (current_depth - next_depth);
            clipped.push(current.lerp(next, crossing));
        }
    }
    clipped
}

/// Which way a slider runs. Its parts are quads scaled, never rotated, so a
/// slider only runs along a strip-local axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SliderAxis {
    /// Left to right.
    Horizontal,
    /// Bottom to top.
    Vertical,
}

/// A slider in strip-local space: it runs `length` along `axis` from
/// `start`, where it reads 0, to where it reads 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SliderSpan {
    pub start: Vec2,
    pub axis: SliderAxis,
    pub length: f32,
}

impl SliderSpan {
    pub(crate) fn center(&self) -> Vec2 {
        self.at(0.5)
    }

    pub(crate) fn at(&self, normalized: f32) -> Vec2 {
        let along = normalized.clamp(0.0, 1.0) * self.length;
        self.start + self.part_size(along, 0.0)
    }

    /// Size of a part measuring `along` the slider and `across` it.
    fn part_size(&self, along: f32, across: f32) -> Vec2 {
        match self.axis {
            SliderAxis::Horizontal => Vec2::new(along, across),
            SliderAxis::Vertical => Vec2::new(across, along),
        }
    }

    /// Size of the pressable area.
    pub(crate) fn hit_size(&self) -> Vec2 {
        self.part_size(self.length, SLIDER_HIT_THICKNESS)
    }

    /// Size of the rail drawn from the start up to `normalized`.
    pub(crate) fn rail_size(&self, normalized: f32) -> Vec2 {
        self.part_size(self.length * normalized.clamp(0.0, 1.0), RAIL_THICKNESS)
    }

    pub(crate) fn knob_size(&self) -> Vec2 {
        self.part_size(KNOB_ALONG, KNOB_ACROSS)
    }
}

/// Left to right: play/pause, timeline, time label, speaker button. The
/// timeline takes whatever width the fixed parts leave. The volume popup
/// stands on the strip over the speaker button.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub(crate) struct StripLayout {
    pub width: f32,
    /// Slot reserved for the time label.
    pub label_width: f32,
    pub volume_popup_upward: bool,
}

impl StripLayout {
    fn left_edge(&self) -> f32 {
        -self.width * 0.5 + STRIP_PADDING
    }

    fn right_edge(&self) -> f32 {
        self.width * 0.5 - STRIP_PADDING
    }

    pub(crate) fn button_center_x(&self) -> f32 {
        self.left_edge() + BUTTON_SIZE * 0.5
    }

    pub(crate) fn speaker_center_x(&self) -> f32 {
        self.right_edge() - BUTTON_SIZE * 0.5
    }

    /// Pressable area of the speaker button. It spans the full strip height
    /// so it meets the volume popup standing on the strip's edge.
    pub(crate) fn speaker_button_size(&self) -> Vec2 {
        Vec2::new(BUTTON_SIZE, STRIP_HEIGHT)
    }

    /// Right edge of the label slot; the label is right-aligned in it.
    pub(crate) fn label_right_x(&self) -> f32 {
        self.speaker_center_x() - BUTTON_SIZE * 0.5 - STRIP_GAP
    }

    pub(crate) fn timeline(&self) -> SliderSpan {
        let start = self.left_edge() + BUTTON_SIZE + STRIP_GAP;
        let end = self.label_right_x() - self.label_width - STRIP_GAP;
        SliderSpan {
            start: Vec2::new(start, 0.0),
            axis: SliderAxis::Horizontal,
            length: (end - start).max(0.0),
        }
    }

    pub(crate) fn volume_popup_center(&self) -> Vec2 {
        let offset = (STRIP_HEIGHT + VOLUME_POPUP_HEIGHT) * 0.5;
        let direction = if self.volume_popup_upward { 1.0 } else { -1.0 };
        Vec2::new(self.speaker_center_x(), direction * offset)
    }

    pub(crate) fn volume_popup_size(&self) -> Vec2 {
        Vec2::new(VOLUME_POPUP_WIDTH, VOLUME_POPUP_HEIGHT)
    }

    /// The volume slider, silent at the bottom whichever way the popup
    /// opens.
    pub(crate) fn volume(&self) -> SliderSpan {
        let length = VOLUME_POPUP_HEIGHT - 2.0 * VOLUME_POPUP_PADDING;
        SliderSpan {
            start: self.volume_popup_center() - Vec2::Y * (length * 0.5),
            axis: SliderAxis::Vertical,
            length,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: Vec2 = Vec2::new(1600.0, 1000.0);

    fn strip_of(video: Rect) -> ScreenStrip {
        screen_strip(video, WINDOW).expect("video is on screen")
    }

    fn assert_inside(strip: Rect, area: Rect) {
        let slack = 1e-3;
        let area = area.inflate(slack);
        assert!(
            area.contains(strip.min) && area.contains(strip.max),
            "{strip:?} in {area:?}"
        );
    }

    #[test]
    fn a_large_video_hosts_its_full_size_strip_along_its_bottom_edge() {
        let video = Rect::new(400.0, 200.0, 1000.0, 700.0);
        let strip = strip_of(video);
        assert_eq!(strip.scale, 1.0);
        assert_eq!(strip.rect.max.y, video.max.y);
        assert_eq!(strip.rect.center().x, video.center().x);
        assert_eq!(strip.rect.width(), video.width());
    }

    #[test]
    fn a_video_overflowing_the_window_keeps_its_strip_on_screen() {
        let video = Rect::new(-400.0, -300.0, 2000.0, 1800.0);
        let strip = strip_of(video);
        let window = Rect::from_corners(Vec2::ZERO, WINDOW).inflate(-STRIP_WINDOW_MARGIN);
        assert_inside(strip.rect, window);
        assert_eq!(strip.rect.max.y, window.max.y);
        assert_eq!(strip.width, STRIP_MAX_WIDTH);
    }

    #[test]
    fn a_fixed_strip_sits_on_its_picture_bottom_edge_and_moves_with_it() {
        let half_extents = Vec2::new(1.0, 0.5625);
        let picture = Transform::from_translation(Vec3::new(3.0, 1.0, -2.0))
            .with_rotation(Quat::from_rotation_y(0.7))
            .with_scale(Vec3::splat(2.0));
        let placement = place_strip_on_picture(&picture, half_extents);
        // In picture-local space: flush with the bottom edge, centered,
        // no wider than the picture and just in front of it.
        let local = picture
            .compute_matrix()
            .inverse()
            .transform_point3(placement.transform.translation);
        let local_units_per_strip_unit = placement.transform.scale.x / picture.scale.x;
        let strip_bottom = local.y - STRIP_HEIGHT * 0.5 * local_units_per_strip_unit;
        assert!((strip_bottom + half_extents.y).abs() < 1e-4);
        assert!(local.x.abs() < 1e-4);
        assert!(local.z > 0.0);
        assert!(placement.width * local_units_per_strip_unit <= 2.0 * half_extents.x + 1e-4);
        assert_eq!(placement.transform.rotation, picture.rotation);

        let moved = picture.with_translation(Vec3::new(-5.0, 0.0, 4.0));
        let moved_placement = place_strip_on_picture(&moved, half_extents);
        let offset = moved.translation - picture.translation;
        assert!(
            (moved_placement.transform.translation - placement.transform.translation - offset)
                .length()
                < 1e-4
        );
    }

    #[test]
    fn a_small_video_shrinks_its_strip_to_fit_inside_it() {
        for video in [
            Rect::new(700.0, 300.0, 760.0, 360.0),
            Rect::new(700.0, 100.0, 700.0 + STRIP_MIN_WIDTH - 1.0, 700.0),
            Rect::new(100.0, 400.0, 1200.0, 420.0),
        ] {
            let strip = strip_of(video);
            assert!(strip.scale < 1.0);
            assert_inside(strip.rect, video);
            assert!((strip.rect.width() - strip.width * strip.scale).abs() < 1e-3);
            assert!((strip.rect.height() - STRIP_HEIGHT * strip.scale).abs() < 1e-3);
        }
    }

    #[test]
    fn a_video_shrinking_past_the_full_size_strip_moves_it_continuously() {
        // Just wide enough for the full-size strip, then a hair narrower.
        let video = Rect::new(600.0, 300.0, 600.0 + STRIP_MIN_WIDTH, 600.0);
        let narrower = Rect::new(600.0, 300.0, 600.0 + STRIP_MIN_WIDTH - 1.0, 600.0);
        let (strip, shrunk) = (strip_of(video), strip_of(narrower));
        assert_eq!(strip.scale, 1.0);
        assert!(shrunk.scale < 1.0);
        // Both stay flush with the bottom and sides, and barely differ.
        for (strip, video) in [(strip, video), (shrunk, narrower)] {
            assert_eq!(strip.rect.max.y, video.max.y);
            assert!((strip.rect.width() - video.width()).abs() < 1e-3);
        }
        assert!((strip.rect.min.y - shrunk.rect.min.y).abs() < 0.2);
    }

    #[test]
    fn a_growing_video_moves_its_strip_only_with_its_edges() {
        let video = Rect::new(700.0, 300.0, 1000.0, 500.0);
        let grown = video.inflate(20.0);
        let (strip, grown_strip) = (strip_of(video), strip_of(grown));
        assert_inside(strip.rect, video);
        assert_inside(grown_strip.rect, grown);
        assert_eq!(strip.rect.center().x, grown_strip.rect.center().x);
        assert!(grown_strip.rect.max.y > strip.rect.max.y);
    }

    #[test]
    fn a_partly_visible_video_centers_its_strip_on_the_visible_part() {
        let video = Rect::new(-2000.0, 200.0, 1000.0, 800.0);
        let strip = strip_of(video);
        assert_eq!(strip.rect.center().x, (STRIP_WINDOW_MARGIN + 1000.0) * 0.5);
    }

    #[test]
    fn an_off_screen_video_has_no_strip() {
        let video = Rect::new(-500.0, 100.0, -100.0, 400.0);
        assert!(screen_strip(video, WINDOW).is_none());
    }

    #[test]
    fn clipping_keeps_only_the_part_in_front_of_the_near_plane() {
        // A quad straddling the eye, spanning depth -1..3 along -Z.
        let quad = [
            Vec3::new(-1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, -3.0),
            Vec3::new(-1.0, 0.0, -3.0),
        ];
        let clipped = clip_polygon_to_depth(&quad, Vec3::ZERO, Vec3::NEG_Z, 0.5);
        assert_eq!(clipped.len(), 4);
        assert!(clipped.iter().all(|point| -point.z >= 0.5 - 1e-5));
        assert!(clip_polygon_to_depth(&quad, Vec3::ZERO, Vec3::Z, 2.0).is_empty());
    }

    #[test]
    fn strip_parts_fit_inside_the_strip_in_order() {
        let layout = StripLayout {
            width: STRIP_MIN_WIDTH,
            label_width: 80.0,
            volume_popup_upward: true,
        };
        let timeline = layout.timeline();
        assert!(layout.button_center_x() + BUTTON_SIZE * 0.5 < timeline.start.x);
        assert!(timeline.length > 0.0);
        assert!(timeline.at(1.0).x < layout.label_right_x() - layout.label_width);
        assert!(layout.label_right_x() < layout.speaker_center_x() - BUTTON_SIZE * 0.5);
        assert!(layout.speaker_center_x() + BUTTON_SIZE * 0.5 <= layout.width * 0.5);
        assert_eq!(timeline.at(1.0).x, timeline.start.x + timeline.length);
    }

    #[test]
    fn the_volume_slider_stands_on_the_strip_over_the_speaker_and_rises() {
        for volume_popup_upward in [true, false] {
            let layout = StripLayout {
                width: STRIP_MIN_WIDTH,
                label_width: 80.0,
                volume_popup_upward,
            };
            let center = layout.volume_popup_center();
            assert_eq!(center.x, layout.speaker_center_x());
            let popup_near_edge = center.y.abs() - layout.volume_popup_size().y * 0.5;
            assert_eq!(popup_near_edge, layout.speaker_button_size().y * 0.5);
            let volume = layout.volume();
            assert!(volume.at(1.0).y > volume.at(0.0).y);
            assert_eq!(volume.center(), center);
            assert_eq!(volume.hit_size().x, SLIDER_HIT_THICKNESS);
            assert!(volume.hit_size().y <= layout.volume_popup_size().y);
        }
    }
}
