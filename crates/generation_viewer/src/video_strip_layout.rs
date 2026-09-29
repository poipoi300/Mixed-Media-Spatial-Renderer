//! Geometry for video control strips.
//!
//! A strip is sized and positioned in screen space — overlaid along the
//! bottom edge of its video's picture when it fits inside it, beside the
//! picture otherwise, a constant height in pixels, never leaving the
//! window — but lives in the world: it is placed on a
//! camera-facing plane just in front of the nearest visible point of its
//! video, scaled so one local unit is one window pixel. Other content in
//! front of the video therefore occludes the strip like any other geometry,
//! while the strip keeps a readable, screen-derived size at any distance.
//!
//! Strip-local space: x right, y up, origin at the strip's center, one unit
//! per window pixel; larger z is closer to the camera.

use bevy::prelude::*;

/// Strip height, in window pixels.
pub(crate) const STRIP_HEIGHT: f32 = 40.0;
const STRIP_MIN_WIDTH: f32 = 340.0;
const STRIP_MAX_WIDTH: f32 = 720.0;
/// Closest a strip may come to a window edge.
const STRIP_WINDOW_MARGIN: f32 = 12.0;
/// Gap between a strip and the edges of the picture it sits on (or under).
const STRIP_VIDEO_INSET: f32 = 8.0;
/// A picture must be at least this many strip heights tall on screen (and
/// wide enough for the whole strip) to have its strip overlaid on it; a
/// smaller one gets the strip just outside it rather than mostly hidden
/// behind it.
const STRIP_OVERLAY_MIN_VIDEO_HEIGHTS: f32 = 3.0;
/// Clip depth as a multiple of the camera's near plane; a hair beyond it so
/// clipped vertices are guaranteed to project.
const NEAR_PLANE_CLIP_MARGIN: f32 = 1.001;
/// Fraction of its video's nearest depth a strip is pulled toward the
/// camera, so the video itself never cuts into it.
const STRIP_DEPTH_PULL: f32 = 0.02;
/// Closest a strip may come to the camera, as a multiple of the near plane.
const STRIP_MIN_DEPTH_NEAR_PLANES: f32 = 1.05;

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
    let strip_rect = screen_strip_rect(picture_rect, window_size)?;

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
    let center = strip_rect.center();
    let left = point_at_depth(Vec2::new(strip_rect.min.x, center.y))?;
    let right = point_at_depth(Vec2::new(strip_rect.max.x, center.y))?;
    let units_per_pixel = left.distance(right) / strip_rect.width();
    Some(StripPlacement {
        transform: Transform {
            translation: point_at_depth(center)?,
            rotation: camera_transform.rotation,
            scale: Vec3::splat(units_per_pixel),
        },
        width: strip_rect.width(),
        volume_popup_upward: strip_rect.min.y >= VOLUME_POPUP_HEIGHT + STRIP_WINDOW_MARGIN,
    })
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

/// Window rectangle (logical pixels, y down) for the strip of a picture
/// with on-screen bounds `picture_rect`. Sized and centered on the part of
/// the picture inside the window, and always fully inside the window. It is
/// overlaid along the picture's bottom edge while it fits inside the
/// picture; otherwise it goes just below the picture, rising over it as far
/// as its top when the window's bottom edge leaves no room below. `None`
/// when the picture is entirely off screen.
fn screen_strip_rect(picture_rect: Rect, window_size: Vec2) -> Option<Rect> {
    let visible = picture_rect.intersect(Rect::from_corners(Vec2::ZERO, window_size));
    if visible.is_empty() {
        return None;
    }
    let max_width = (window_size.x - 2.0 * STRIP_WINDOW_MARGIN).max(0.0);
    let width = (visible.width() - 2.0 * STRIP_VIDEO_INSET)
        .clamp(STRIP_MIN_WIDTH, STRIP_MAX_WIDTH)
        .min(max_width);
    let left = (visible.center().x - width * 0.5).clamp(
        STRIP_WINDOW_MARGIN,
        (window_size.x - STRIP_WINDOW_MARGIN - width).max(STRIP_WINDOW_MARGIN),
    );
    let overlaid = visible.width() >= width + 2.0 * STRIP_VIDEO_INSET
        && visible.height() >= STRIP_HEIGHT * STRIP_OVERLAY_MIN_VIDEO_HEIGHTS;
    let lowest_top = (window_size.y - STRIP_WINDOW_MARGIN - STRIP_HEIGHT).max(STRIP_WINDOW_MARGIN);
    let top = if overlaid {
        visible.max.y - STRIP_VIDEO_INSET - STRIP_HEIGHT
    } else {
        visible.max.y + STRIP_VIDEO_INSET
    }
    // Short of room below, the strip rises over the picture, never past its
    // top — unless staying in the window demands it.
    .min(lowest_top)
    .max(visible.min.y)
    .clamp(STRIP_WINDOW_MARGIN, lowest_top);
    Some(Rect::new(left, top, left + width, top + STRIP_HEIGHT))
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

    fn assert_inside_window(strip: Rect) {
        assert!(strip.min.x >= STRIP_WINDOW_MARGIN && strip.min.y >= STRIP_WINDOW_MARGIN);
        assert!(strip.max.x <= WINDOW.x - STRIP_WINDOW_MARGIN);
        assert!(strip.max.y <= WINDOW.y - STRIP_WINDOW_MARGIN);
    }

    #[test]
    fn a_video_overflowing_the_window_keeps_its_strip_on_screen() {
        let video = Rect::new(-400.0, -300.0, 2000.0, 1800.0);
        let strip = screen_strip_rect(video, WINDOW).expect("video is on screen");
        assert_inside_window(strip);
        assert_eq!(strip.max.y, WINDOW.y - STRIP_WINDOW_MARGIN);
        assert_eq!(strip.width(), STRIP_MAX_WIDTH);
    }

    #[test]
    fn a_large_video_hosts_its_strip_along_its_bottom_edge() {
        let video = Rect::new(400.0, 200.0, 1000.0, 700.0);
        let strip = screen_strip_rect(video, WINDOW).expect("video is on screen");
        assert_eq!(strip.max.y, video.max.y - STRIP_VIDEO_INSET);
        assert_eq!(strip.center().x, video.center().x);
    }

    #[test]
    fn a_video_narrower_than_the_strip_gets_it_below_however_tall_it_is() {
        let narrow = Rect::new(700.0, 100.0, 700.0 + STRIP_MIN_WIDTH, 700.0);
        let strip = screen_strip_rect(narrow, WINDOW).expect("video is on screen");
        assert_eq!(strip.min.y, narrow.max.y + STRIP_VIDEO_INSET);
        let wide_enough = Rect::new(
            700.0,
            100.0,
            700.0 + STRIP_MIN_WIDTH + 2.0 * STRIP_VIDEO_INSET,
            700.0,
        );
        let strip = screen_strip_rect(wide_enough, WINDOW).expect("video is on screen");
        assert_eq!(strip.max.y, wide_enough.max.y - STRIP_VIDEO_INSET);
    }

    #[test]
    fn a_small_video_at_the_window_bottom_gets_its_strip_over_it_never_above() {
        let video = Rect::new(700.0, 900.0, 800.0, 980.0);
        let strip = screen_strip_rect(video, WINDOW).expect("video is on screen");
        assert_inside_window(strip);
        assert!(strip.min.y >= video.min.y);
    }

    #[test]
    fn a_small_video_gets_its_strip_just_below_it() {
        let video = Rect::new(700.0, 300.0, 760.0, 360.0);
        let strip = screen_strip_rect(video, WINDOW).expect("video is on screen");
        assert_eq!(strip.min.y, video.max.y + STRIP_VIDEO_INSET);
        assert_eq!(strip.width(), STRIP_MIN_WIDTH);
    }

    #[test]
    fn a_video_at_the_window_edge_is_clamped_inside() {
        let video = Rect::new(-50.0, 900.0, 100.0, 1100.0);
        let strip = screen_strip_rect(video, WINDOW).expect("video is on screen");
        assert_inside_window(strip);
    }

    #[test]
    fn a_partly_visible_video_centers_its_strip_on_the_visible_part() {
        // Centered on the unclipped picture, the strip would pin to the
        // left window edge instead.
        let video = Rect::new(-2000.0, 200.0, 1000.0, 800.0);
        let strip = screen_strip_rect(video, WINDOW).expect("video is on screen");
        assert_eq!(strip.center().x, 500.0);
    }

    #[test]
    fn an_off_screen_video_has_no_strip() {
        let video = Rect::new(-500.0, 100.0, -100.0, 400.0);
        assert!(screen_strip_rect(video, WINDOW).is_none());
    }

    #[test]
    fn a_narrow_window_narrows_the_strip() {
        let window = Vec2::new(200.0, 400.0);
        let video = Rect::new(0.0, 0.0, 200.0, 400.0);
        let strip = screen_strip_rect(video, window).expect("video is on screen");
        assert_eq!(strip.width(), window.x - 2.0 * STRIP_WINDOW_MARGIN);
        assert_eq!(strip.min.x, STRIP_WINDOW_MARGIN);
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
