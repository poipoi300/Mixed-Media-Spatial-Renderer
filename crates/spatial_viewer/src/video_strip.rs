//! Video control strips: play/pause, a timeline, the elapsed/total time and
//! a speaker button (mute, with a vertical volume slider popping up over it)
//! for one video, drawn as world-space quads so nearer content occludes
//! them, but sized and positioned in screen space (see
//! `video_strip_layout`). Which strips exist and what pressing them does is
//! decided in `video_controls`; this module draws them and hit-tests them.
//!
//! Every quad-shaped part is a unit quad scaled to its size in strip-local
//! pixels, so hit tests only ever check a local `[-0.5, 0.5]` square and the
//! meshes and materials are shared by every strip. Only the time label is
//! per strip.

use std::collections::{HashMap, HashSet};

use ab_glyph::FontArc;
use bevy::ecs::system::SystemParam;
use bevy::math::primitives::Rectangle;
use bevy::prelude::*;
use bevy::render::{mesh::PrimitiveTopology, render_asset::RenderAssetUsages};
use bevy::window::PrimaryWindow;
use spatial_viewer_ui::{PauseMenuState, RenderResolutionSettings, UiInputCapture};

use crate::axis_gizmo::{cursor_over_axis_gizmo, render_label_text, LabelTextLine};
use crate::image_loading::MediaBillboard;
use crate::manual_spacing::{
    cursor_world_ray, nearest_billboard_hit, SelectionState, TranslateGizmo,
};
use crate::media_settings::{MediaSettings, VideoFileSettings};
use crate::video_controls::{
    format_video_time_label, ray_plane_hit, ray_rect_hit, video_duration_seconds,
    video_strip_wanted, ScrubGrab, StripSlider, VideoControlsOwner, VideoControlsState,
    VideoPlaybackControl, VideoStripStatus,
};
use crate::video_strip_layout::{
    place_strip, SliderSpan, StripLayout, StripPlacement, BUTTON_SIZE, ICON_SIZE, SPEAKER_SIZE,
    STRIP_HEIGHT, TIME_LABEL_HEIGHT,
};
use crate::{ExplorerScene, FlyCamera};

const PANEL_COLOR: Color = Color::srgba(0.05, 0.06, 0.08, 0.82);
const BUTTON_COLOR: Color = Color::srgb(0.22, 0.68, 1.0);
const RAIL_COLOR: Color = Color::srgb(0.18, 0.21, 0.26);
const KNOB_COLOR: Color = Color::srgb(1.0, 0.86, 0.24);
const ICON_COLOR: Color = Color::srgb(0.96, 0.98, 1.0);
const TIME_LABEL_FONT_SIZE: f32 = 30.0;
const TIME_LABEL_COLOR: [u8; 4] = [232, 236, 244, 235];
/// Wider than any time label, so labels never wrap.
const TIME_LABEL_MAX_WIDTH: f32 = 400.0;

/// Strip-local depth of each layer; larger is closer to the camera, so a
/// knob is hit (and drawn) over its rail, and every control over the panel.
const PANEL_Z: f32 = 0.0;
const CONTROL_Z: f32 = 1.0;
const SLIDER_HIT_Z: f32 = 1.5;
const FILL_Z: f32 = 2.0;
const KNOB_Z: f32 = 3.0;

#[derive(Resource, Default)]
pub(crate) struct VideoControlsFont(pub Option<FontArc>);

/// Root of one video's strip; its `StripLayout` sits alongside.
#[derive(Component)]
pub(crate) struct VideoStripRoot {
    image_id: usize,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VideoStripPart {
    Panel,
    PlayPauseButton,
    PlayIcon,
    PauseIcon,
    /// Pressable area of the timeline; drawn by its rail, fill and knob.
    TimelineTrack,
    TimelineRail,
    TimelineFill,
    TimelineKnob,
    TimeLabel,
    /// Pressable area of the speaker button; drawn by its icons.
    SpeakerButton,
    SpeakerIcon,
    MutedIcon,
    /// Background of the volume popup; a press anywhere on it sets the
    /// volume.
    VolumePopupPanel,
    /// The span the volume slider reads the pointer along; drawn by its
    /// rail, fill and knob.
    VolumeTrack,
    VolumeRail,
    VolumeFill,
    VolumeKnob,
}

impl VideoStripPart {
    /// Parts a press can land on. The panels are among them so a press
    /// between controls is still the strip's, never the video's behind it.
    fn hit_testable(self) -> bool {
        matches!(
            self,
            Self::Panel
                | Self::PlayPauseButton
                | Self::TimelineTrack
                | Self::TimelineKnob
                | Self::SpeakerButton
                | Self::VolumePopupPanel
        )
    }

    /// Parts that open the volume popup, or keep it open, while hovered.
    fn opens_volume_popup(self) -> bool {
        matches!(self, Self::SpeakerButton | Self::VolumePopupPanel)
    }

    /// Whether the part is drawn — and, if hit testable, pressable — on a
    /// strip showing `status`.
    fn shown(self, status: &VideoStripStatus) -> bool {
        match self {
            Self::PlayIcon => !status.playing,
            Self::PauseIcon => status.playing,
            Self::SpeakerIcon => !status.muted,
            Self::MutedIcon => status.muted,
            Self::VolumePopupPanel
            | Self::VolumeTrack
            | Self::VolumeRail
            | Self::VolumeFill
            | Self::VolumeKnob => status.volume_popup_open,
            _ => true,
        }
    }

    fn visibility(self, status: &VideoStripStatus) -> Visibility {
        if self.shown(status) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        }
    }
}

/// The part a slider is pressed and dragged on.
fn slider_track(slider: StripSlider) -> VideoStripPart {
    match slider {
        StripSlider::Timeline => VideoStripPart::TimelineTrack,
        StripSlider::Volume => VideoStripPart::VolumeTrack,
    }
}

/// Position along a slider from a normalized point on its track: the
/// timeline runs left to right, the volume slider bottom to top.
fn along_slider(slider: StripSlider, normalized: Vec2) -> f32 {
    match slider {
        StripSlider::Timeline => normalized.x,
        StripSlider::Volume => normalized.y,
    }
}

#[derive(Component)]
pub(crate) struct VideoStripLabel {
    displayed_text: String,
    /// Rendered size in strip-local pixels; the label is right-aligned in
    /// its reserved slot.
    size: Vec2,
}

/// Meshes and materials every strip shares.
#[derive(Resource)]
pub(crate) struct VideoStripAssets {
    quad: Handle<Mesh>,
    play: Handle<Mesh>,
    pause: Handle<Mesh>,
    speaker: Handle<Mesh>,
    muted: Handle<Mesh>,
    panel: Handle<StandardMaterial>,
    button: Handle<StandardMaterial>,
    rail: Handle<StandardMaterial>,
    fill: Handle<StandardMaterial>,
    knob: Handle<StandardMaterial>,
    icon: Handle<StandardMaterial>,
}

impl VideoStripAssets {
    pub(crate) fn new(meshes: &mut Assets<Mesh>, materials: &mut Assets<StandardMaterial>) -> Self {
        let mut unlit = |color: Color| {
            materials.add(StandardMaterial {
                base_color: color,
                unlit: true,
                cull_mode: None,
                ..default()
            })
        };
        let button = unlit(BUTTON_COLOR);
        let rail = unlit(RAIL_COLOR);
        let knob = unlit(KNOB_COLOR);
        let icon = unlit(ICON_COLOR);
        let panel = materials.add(StandardMaterial {
            base_color: PANEL_COLOR,
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            cull_mode: None,
            ..default()
        });
        Self {
            quad: meshes.add(Rectangle::new(1.0, 1.0)),
            play: meshes.add(flat_mesh(&play_icon_triangles())),
            pause: meshes.add(flat_mesh(&pause_icon_triangles())),
            speaker: meshes.add(flat_mesh(&speaker_icon_triangles())),
            muted: meshes.add(flat_mesh(&muted_icon_triangles())),
            panel,
            fill: button.clone(),
            button,
            rail,
            knob,
            icon,
        }
    }
}

/// Every hit-testable strip part, with the video it belongs to. Parts of a
/// hidden strip still have transforms, so visibility is part of the test.
pub(crate) type VideoStripHitQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static VideoStripPart,
        &'static VideoControlsOwner,
        &'static GlobalTransform,
        &'static InheritedVisibility,
    ),
>;

type StripCameraQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Camera, &'static Transform, &'static Projection),
    (With<FlyCamera>, Without<VideoStripRoot>),
>;

type StripRootQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static VideoStripRoot,
        &'static mut Transform,
        &'static mut Visibility,
        &'static mut StripLayout,
    ),
    (Without<FlyCamera>, Without<MediaBillboard>),
>;

type StripPartQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static VideoStripPart,
        &'static VideoControlsOwner,
        &'static mut Transform,
        &'static mut Visibility,
        Option<&'static VideoStripLabel>,
    ),
    Without<VideoStripRoot>,
>;

/// A cursor ray landing on a strip part.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StripHit {
    pub image_id: usize,
    pub part: VideoStripPart,
    /// Normalized position on the hit part, x rightward and y upward.
    pub normalized: Vec2,
}

#[derive(SystemParam)]
pub(crate) struct StripLabelAssets<'w> {
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    images: ResMut<'w, Assets<Image>>,
}

/// Pointer input on strips: hovering one keeps it revealed, a press on a
/// control acts on its video, and a live slider drag follows the pointer
/// along that slider's plane, even past its ends. Only a strip that is
/// actually visible under the pointer counts: the translate gizmo (drawn
/// over everything) and any billboard in front of the strip take the
/// pointer instead. Presses on a strip never reach selection
/// (`handle_selection_and_drag` runs the same arbitration and stands down).
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_video_strip_input(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    pause_menu: Res<PauseMenuState>,
    ui_capture: Res<UiInputCapture>,
    render_resolution: Res<RenderResolutionSettings>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<FlyCamera>>,
    scene: Res<ExplorerScene>,
    gizmo: Res<TranslateGizmo>,
    parts: VideoStripHitQuery,
    billboards: Query<(&MediaBillboard, &GlobalTransform, &Visibility)>,
    mut playback: VideoPlaybackControl,
) {
    if pause_menu.paused {
        return;
    }
    let Ok(window) = window_query.get_single() else {
        return;
    };
    let Ok((camera, camera_global)) = camera_query.get_single() else {
        return;
    };
    let Some((ray_origin, ray_direction)) =
        cursor_world_ray(window, camera, camera_global, &render_resolution)
    else {
        return;
    };
    let slider_position = |image_id: usize, slider: StripSlider| {
        let track = slider_track(slider);
        parts
            .iter()
            .find_map(|(part, owner, transform, visibility)| {
                (*part == track && owner.image_id == image_id && visibility.get())
                    .then_some(transform)
            })
            .and_then(|transform| slider_normalized_from_ray(ray_origin, ray_direction, transform))
            .map(|normalized| along_slider(slider, normalized))
    };

    if let Some((image_id, slider)) = playback.state().dragged_slider() {
        if let Some(position) = slider_position(image_id, slider) {
            let (controls, settings) = playback.state_and_settings();
            controls.drag_slider_to(position, settings);
        }
        return;
    }

    // UI panels, the axis gizmo canvas and the translate gizmo own the
    // pointer over them.
    if ui_capture.blocks_world_clicks()
        || cursor_over_axis_gizmo(window)
        || gizmo.axis_under_ray(ray_origin, ray_direction).is_some()
    {
        return;
    }
    let occluder_distance = nearest_billboard_hit(
        ray_origin,
        ray_direction,
        scene.billboard_world_size,
        &billboards,
    )
    .map(|hit| hit.distance);
    let Some(hit) = nearest_strip_hit(ray_origin, ray_direction, &parts, occluder_distance) else {
        return;
    };
    // A right-drag look sweeps strips under a still pointer; that is not
    // hovering them.
    if !mouse_buttons.pressed(MouseButton::Right) {
        let controls = playback.state_mut();
        controls.reveal(hit.image_id);
        if hit.part.opens_volume_popup() {
            controls.hover_volume(hit.image_id);
        }
    }
    if !mouse_buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(billboard) = billboards
        .iter()
        .map(|(billboard, _, _)| billboard)
        .find(|billboard| billboard.image_id == hit.image_id)
    else {
        return;
    };
    match hit.part {
        VideoStripPart::PlayPauseButton => playback.started(billboard).toggle_playing(hit.image_id),
        VideoStripPart::TimelineTrack => playback.started(billboard).begin_scrub(
            hit.image_id,
            ScrubGrab::Track(along_slider(StripSlider::Timeline, hit.normalized)),
        ),
        VideoStripPart::TimelineKnob => {
            // A knob grab is measured along the track, not across the knob.
            if let Some(grabbed_at) = slider_position(hit.image_id, StripSlider::Timeline) {
                playback
                    .started(billboard)
                    .begin_scrub(hit.image_id, ScrubGrab::Knob(grabbed_at));
            }
        }
        VideoStripPart::SpeakerButton => playback
            .media_settings_mut()
            .update(&billboard.path, VideoFileSettings::toggle_mute),
        VideoStripPart::VolumePopupPanel => {
            // The whole popup takes presses, read along the slider and
            // clamped, so a press past either end sets silence or full.
            if let Some(position) = slider_position(hit.image_id, StripSlider::Volume) {
                let (controls, settings) = playback.state_and_settings();
                controls.begin_volume_drag(hit.image_id, &billboard.path, position, settings);
            }
        }
        _ => {}
    }
}

/// Spawns a strip for every video that should show one, despawns the rest,
/// and places each against its video. Runs after billboards are faced and
/// hidden for the frame and uses the camera's fresh `Transform`, so strips
/// never trail a moving view or a dragged video.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sync_video_strips(
    mut commands: Commands,
    controls_state: Res<VideoControlsState>,
    media_settings: Res<MediaSettings>,
    selection: Res<SelectionState>,
    scene: Res<ExplorerScene>,
    video_font: Res<VideoControlsFont>,
    strip_assets: Res<VideoStripAssets>,
    mut label_assets: StripLabelAssets,
    window_query: Query<&Window, With<PrimaryWindow>>,
    camera_query: StripCameraQuery,
    billboards: Query<(&MediaBillboard, &Transform, &Visibility), Without<VideoStripRoot>>,
    mut roots: StripRootQuery,
) {
    let (Ok(window), Ok((camera, camera_transform, projection))) =
        (window_query.get_single(), camera_query.get_single())
    else {
        return;
    };
    let near = match projection {
        Projection::Perspective(perspective) => perspective.near,
        Projection::Orthographic(orthographic) => orthographic.near,
    };
    let mut wanted: HashMap<usize, _> = billboards
        .iter()
        .filter(|(billboard, _, visibility)| {
            video_strip_wanted(&controls_state, &selection, billboard, visibility)
        })
        .map(|(billboard, transform, _)| {
            let placement = place_strip(
                camera,
                camera_transform,
                near,
                transform,
                billboard.content_half_extents(scene.billboard_world_size),
                window.size(),
            );
            (billboard.image_id, (billboard, placement))
        })
        .collect();

    for (entity, root, mut transform, mut visibility, mut layout) in &mut roots {
        let Some((_, placement)) = wanted.remove(&root.image_id) else {
            commands.entity(entity).despawn_recursive();
            continue;
        };
        match placement {
            Some(placement) => {
                transform.set_if_neq(placement.transform);
                visibility.set_if_neq(Visibility::Inherited);
                layout.set_if_neq(StripLayout {
                    width: placement.width,
                    volume_popup_upward: placement.volume_popup_upward,
                    ..*layout
                });
            }
            None => {
                visibility.set_if_neq(Visibility::Hidden);
            }
        }
    }
    for (image_id, (billboard, placement)) in wanted {
        let duration_seconds = video_duration_seconds(billboard);
        let status = controls_state.strip_status(
            image_id,
            duration_seconds,
            &media_settings.video(&billboard.path),
        );
        spawn_video_strip(
            &mut commands,
            &strip_assets,
            &mut label_assets,
            video_font.0.as_ref(),
            image_id,
            duration_seconds,
            placement,
            &status,
        );
    }
}

/// Lays out every strip's parts for its current width and its video's clock
/// and volume.
pub(crate) fn layout_video_strip_parts(
    controls_state: Res<VideoControlsState>,
    media_settings: Res<MediaSettings>,
    billboards: Query<&MediaBillboard>,
    roots: Query<(&VideoStripRoot, &StripLayout)>,
    mut parts: StripPartQuery,
) {
    if roots.is_empty() {
        return;
    }
    let layouts: HashMap<usize, StripLayout> = roots
        .iter()
        .map(|(root, layout)| (root.image_id, *layout))
        .collect();
    let statuses = strip_statuses(
        &controls_state,
        &media_settings,
        &billboards,
        layouts.keys().copied(),
    );
    for (part, owner, mut transform, mut visibility, label) in &mut parts {
        let (Some(layout), Some(status)) =
            (layouts.get(&owner.image_id), statuses.get(&owner.image_id))
        else {
            continue;
        };
        let label_size = label.map_or(Vec2::ZERO, |label| label.size);
        transform.set_if_neq(part_transform(*part, layout, status, label_size));
        visibility.set_if_neq(part.visibility(status));
    }
}

/// Re-renders a strip's time label when its text changes.
pub(crate) fn update_video_strip_labels(
    controls_state: Res<VideoControlsState>,
    media_settings: Res<MediaSettings>,
    video_font: Res<VideoControlsFont>,
    mut label_assets: StripLabelAssets,
    billboards: Query<&MediaBillboard>,
    mut labels: Query<(
        &VideoControlsOwner,
        &mut VideoStripLabel,
        &mut Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
    )>,
) {
    let Some(font) = video_font.0.as_ref() else {
        return;
    };
    if labels.is_empty() {
        return;
    }
    let statuses = strip_statuses(
        &controls_state,
        &media_settings,
        &billboards,
        labels.iter().map(|(owner, ..)| owner.image_id),
    );
    for (owner, mut label, mut mesh, material) in &mut labels {
        let Some(text) = statuses
            .get(&owner.image_id)
            .map(|status| &status.time_label)
        else {
            continue;
        };
        if label.displayed_text == *text {
            continue;
        }
        let text = text.clone();
        let rendered = render_time_label(font, &text);
        label.size = rendered.world_size;
        label.displayed_text = text;
        mesh.0 = label_assets
            .meshes
            .add(Rectangle::new(rendered.world_size.x, rendered.world_size.y));
        if let Some(material) = label_assets.materials.get_mut(&material.0) {
            material.base_color_texture = Some(label_assets.images.add(rendered.image));
        }
    }
}

/// What the strips of the videos in `image_ids` show this frame. A video
/// whose billboard is gone has none; its strip is despawned this frame
/// anyway.
fn strip_statuses(
    controls_state: &VideoControlsState,
    media_settings: &MediaSettings,
    billboards: &Query<&MediaBillboard>,
    image_ids: impl IntoIterator<Item = usize>,
) -> HashMap<usize, VideoStripStatus> {
    let image_ids: HashSet<usize> = image_ids.into_iter().collect();
    billboards
        .iter()
        .filter(|billboard| billboard.is_video && image_ids.contains(&billboard.image_id))
        .map(|billboard| {
            let status = controls_state.strip_status(
                billboard.image_id,
                video_duration_seconds(billboard),
                &media_settings.video(&billboard.path),
            );
            (billboard.image_id, status)
        })
        .collect()
}

/// Nearest strip part under the ray, unless something at
/// `occluder_distance` sits in front of it — strips are depth-tested, so a
/// strip hidden behind a billboard must not take the pointer either.
pub(crate) fn nearest_strip_hit(
    ray_origin: Vec3,
    ray_direction: Vec3,
    parts: &VideoStripHitQuery,
    occluder_distance: Option<f32>,
) -> Option<StripHit> {
    parts
        .iter()
        .filter(|(part, _, _, visibility)| part.hit_testable() && visibility.get())
        .filter_map(|(part, owner, transform, _)| {
            let (distance, local) =
                ray_rect_hit(ray_origin, ray_direction, transform, Vec2::splat(0.5))?;
            if occluder_distance.is_some_and(|occluder| occluder < distance) {
                return None;
            }
            Some((
                StripHit {
                    image_id: owner.image_id,
                    part: *part,
                    normalized: (local + 0.5).clamp(Vec2::ZERO, Vec2::ONE),
                },
                distance,
            ))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(hit, _)| hit)
}

/// Normalized position on a slider's unit quad where the ray crosses its
/// plane, unbounded by the quad itself so a drag keeps tracking past either
/// end (clamped to [0, 1] instead of being lost).
fn slider_normalized_from_ray(
    ray_origin: Vec3,
    ray_direction: Vec3,
    track: &GlobalTransform,
) -> Option<Vec2> {
    ray_plane_hit(ray_origin, ray_direction, track)
        .map(|local_hit| (local_hit + 0.5).clamp(Vec2::ZERO, Vec2::ONE))
}

/// Where a part sits in its strip. Quad parts are unit quads scaled to size;
/// mesh parts (icons, label) are authored at their pixel size.
fn part_transform(
    part: VideoStripPart,
    layout: &StripLayout,
    status: &VideoStripStatus,
    label_size: Vec2,
) -> Transform {
    let quad = |center: Vec2, size: Vec2, z: f32| {
        Transform::from_translation(center.extend(z)).with_scale(size.extend(1.0))
    };
    let fill = |span: SliderSpan, normalized: f32| {
        let normalized = normalized.clamp(0.0, 1.0);
        quad(
            span.at(normalized * 0.5),
            span.rail_size(normalized),
            FILL_Z,
        )
    };
    let timeline = layout.timeline();
    let volume = layout.volume();
    let button_center = Vec2::new(layout.button_center_x(), 0.0);
    let speaker_center = Vec2::new(layout.speaker_center_x(), 0.0);
    match part {
        VideoStripPart::Panel => quad(Vec2::ZERO, Vec2::new(layout.width, STRIP_HEIGHT), PANEL_Z),
        VideoStripPart::PlayPauseButton => quad(button_center, Vec2::splat(BUTTON_SIZE), CONTROL_Z),
        VideoStripPart::PlayIcon | VideoStripPart::PauseIcon => {
            Transform::from_translation(button_center.extend(FILL_Z))
        }
        VideoStripPart::TimelineTrack => quad(timeline.center(), timeline.hit_size(), SLIDER_HIT_Z),
        VideoStripPart::TimelineRail => quad(timeline.center(), timeline.rail_size(1.0), CONTROL_Z),
        VideoStripPart::TimelineFill => fill(timeline, status.normalized_time),
        VideoStripPart::TimelineKnob => quad(
            timeline.at(status.normalized_time),
            timeline.knob_size(),
            KNOB_Z,
        ),
        VideoStripPart::TimeLabel => {
            Transform::from_xyz(layout.label_right_x() - label_size.x * 0.5, 0.0, CONTROL_Z)
        }
        VideoStripPart::SpeakerButton => {
            quad(speaker_center, layout.speaker_button_size(), CONTROL_Z)
        }
        VideoStripPart::SpeakerIcon | VideoStripPart::MutedIcon => {
            Transform::from_translation(speaker_center.extend(FILL_Z))
        }
        VideoStripPart::VolumePopupPanel => quad(
            layout.volume_popup_center(),
            layout.volume_popup_size(),
            PANEL_Z,
        ),
        VideoStripPart::VolumeTrack => quad(volume.center(), volume.hit_size(), SLIDER_HIT_Z),
        VideoStripPart::VolumeRail => quad(volume.center(), volume.rail_size(1.0), CONTROL_Z),
        VideoStripPart::VolumeFill => fill(volume, status.volume),
        VideoStripPart::VolumeKnob => quad(volume.at(status.volume), volume.knob_size(), KNOB_Z),
    }
}

fn render_time_label(font: &FontArc, text: &str) -> crate::axis_gizmo::RenderedLabelText {
    render_label_text(
        font,
        &[LabelTextLine {
            text,
            font_size: TIME_LABEL_FONT_SIZE,
            color: TIME_LABEL_COLOR,
        }],
        TIME_LABEL_HEIGHT,
        TIME_LABEL_MAX_WIDTH,
    )
}

#[allow(clippy::too_many_arguments)]
fn spawn_video_strip(
    commands: &mut Commands,
    strip_assets: &VideoStripAssets,
    label_assets: &mut StripLabelAssets,
    font: Option<&FontArc>,
    image_id: usize,
    duration_seconds: f32,
    placement: Option<StripPlacement>,
    status: &VideoStripStatus,
) {
    let owner = VideoControlsOwner { image_id };
    // The label slot fits the widest text this video's duration produces.
    let label_width = font.map_or(0.0, |font| {
        render_time_label(
            font,
            &format_video_time_label(duration_seconds, duration_seconds),
        )
        .world_size
        .x
    });
    let rendered_label = font.map(|font| render_time_label(font, &status.time_label));
    let label_size = rendered_label
        .as_ref()
        .map_or(Vec2::ZERO, |rendered| rendered.world_size);
    let label_mesh = label_assets.meshes.add(Rectangle::new(
        label_size.x.max(f32::EPSILON),
        label_size.y.max(f32::EPSILON),
    ));
    let label_material = label_assets.materials.add(StandardMaterial {
        // Texture is premultiplied (see `render_label_text`/`blend_label_pixel`).
        base_color_texture: rendered_label.map(|rendered| label_assets.images.add(rendered.image)),
        alpha_mode: AlphaMode::Premultiplied,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let layout = StripLayout {
        width: placement.map_or(0.0, |placement| placement.width),
        label_width,
        volume_popup_upward: placement.is_none_or(|placement| placement.volume_popup_upward),
    };
    let (transform, visibility) = match placement {
        Some(placement) => (placement.transform, Visibility::Inherited),
        None => (Transform::default(), Visibility::Hidden),
    };
    let mesh_part =
        |part: VideoStripPart, mesh: &Handle<Mesh>, material: &Handle<StandardMaterial>| {
            (
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                part_transform(part, &layout, status, label_size),
                part.visibility(status),
                part,
                owner,
            )
        };
    let quad_part = |part: VideoStripPart, material: &Handle<StandardMaterial>| {
        mesh_part(part, &strip_assets.quad, material)
    };
    let hit_area = |part: VideoStripPart| {
        (
            part_transform(part, &layout, status, label_size),
            part.visibility(status),
            part,
            owner,
        )
    };
    commands
        .spawn((
            transform,
            visibility,
            layout,
            VideoStripRoot { image_id },
            Name::new(format!("video controls {image_id}")),
        ))
        .with_children(|strip| {
            strip.spawn(quad_part(VideoStripPart::Panel, &strip_assets.panel));
            strip.spawn(quad_part(
                VideoStripPart::PlayPauseButton,
                &strip_assets.button,
            ));
            strip.spawn(mesh_part(
                VideoStripPart::PlayIcon,
                &strip_assets.play,
                &strip_assets.icon,
            ));
            strip.spawn(mesh_part(
                VideoStripPart::PauseIcon,
                &strip_assets.pause,
                &strip_assets.icon,
            ));
            strip.spawn(hit_area(VideoStripPart::TimelineTrack));
            strip.spawn(quad_part(VideoStripPart::TimelineRail, &strip_assets.rail));
            strip.spawn(quad_part(VideoStripPart::TimelineFill, &strip_assets.fill));
            strip.spawn(quad_part(VideoStripPart::TimelineKnob, &strip_assets.knob));
            strip.spawn((
                Mesh3d(label_mesh),
                MeshMaterial3d(label_material),
                part_transform(VideoStripPart::TimeLabel, &layout, status, label_size),
                VideoStripPart::TimeLabel,
                VideoStripLabel {
                    displayed_text: status.time_label.clone(),
                    size: label_size,
                },
                owner,
            ));
            strip.spawn(hit_area(VideoStripPart::SpeakerButton));
            strip.spawn(mesh_part(
                VideoStripPart::SpeakerIcon,
                &strip_assets.speaker,
                &strip_assets.icon,
            ));
            strip.spawn(mesh_part(
                VideoStripPart::MutedIcon,
                &strip_assets.muted,
                &strip_assets.icon,
            ));
            strip.spawn(quad_part(
                VideoStripPart::VolumePopupPanel,
                &strip_assets.panel,
            ));
            strip.spawn(hit_area(VideoStripPart::VolumeTrack));
            strip.spawn(quad_part(VideoStripPart::VolumeRail, &strip_assets.rail));
            strip.spawn(quad_part(VideoStripPart::VolumeFill, &strip_assets.fill));
            strip.spawn(quad_part(VideoStripPart::VolumeKnob, &strip_assets.knob));
        });
}

/// A flat triangle-list mesh in the local XY plane, facing +Z.
fn flat_mesh(triangles: &[[Vec2; 3]]) -> Mesh {
    let positions: Vec<[f32; 3]> = triangles
        .iter()
        .flatten()
        .map(|vertex| [vertex.x, vertex.y, 0.0])
        .collect();
    let count = positions.len();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; count]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; count]);
    mesh
}

/// Two triangles covering a convex quad given counter-clockwise.
fn quad_triangles([a, b, c, d]: [Vec2; 4]) -> [[Vec2; 3]; 2] {
    [[a, b, c], [a, c, d]]
}

/// Two triangles covering an axis-aligned rectangle.
fn rect_triangles(min: Vec2, max: Vec2) -> [[Vec2; 3]; 2] {
    quad_triangles([min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)])
}

/// A straight bar `thickness` wide from `start` to `end`.
fn bar_triangles(start: Vec2, end: Vec2, thickness: f32) -> [[Vec2; 3]; 2] {
    let across = (end - start).normalize_or_zero().perp() * (thickness * 0.5);
    quad_triangles([start - across, end - across, end + across, start + across])
}

/// A band `thickness` wide along the arc of `radius` around `center`,
/// between two angles (radians, counter-clockwise from +x).
fn arc_band_triangles(
    center: Vec2,
    radius: f32,
    thickness: f32,
    (start_angle, end_angle): (f32, f32),
) -> Vec<[Vec2; 3]> {
    const SEGMENTS: usize = 8;
    let (inner, outer) = (radius - thickness * 0.5, radius + thickness * 0.5);
    let point = |segment: usize, arc_radius: f32| {
        let angle = start_angle + (end_angle - start_angle) * segment as f32 / SEGMENTS as f32;
        center + Vec2::from_angle(angle) * arc_radius
    };
    (0..SEGMENTS)
        .flat_map(|segment| {
            quad_triangles([
                point(segment, inner),
                point(segment, outer),
                point(segment + 1, outer),
                point(segment + 1, inner),
            ])
        })
        .collect()
}

/// Right-pointing triangle, nudged right so its mass sits centered.
fn play_icon_triangles() -> Vec<[Vec2; 3]> {
    let half = ICON_SIZE * 0.5;
    vec![[
        Vec2::new(half * 1.1, 0.0),
        Vec2::new(-half * 0.8, half),
        Vec2::new(-half * 0.8, -half),
    ]]
}

fn pause_icon_triangles() -> Vec<[Vec2; 3]> {
    let half = ICON_SIZE * 0.5;
    let bar_width = ICON_SIZE / 3.0;
    [-1.0, 1.0]
        .into_iter()
        .flat_map(|side: f32| {
            let center = side * bar_width;
            rect_triangles(
                Vec2::new(center - bar_width * 0.5, -half),
                Vec2::new(center + bar_width * 0.5, half),
            )
        })
        .collect()
}

/// A speaker: a small box with a flared cone, on the left of its icon so
/// the waves or cross of `speaker_icon_triangles`/`muted_icon_triangles`
/// fit on the right.
fn speaker_triangles() -> Vec<[Vec2; 3]> {
    let half = SPEAKER_SIZE * 0.5;
    let body_right = -half * 0.45;
    let body_half_height = half * 0.4;
    let mut triangles = rect_triangles(
        Vec2::new(-half, -body_half_height),
        Vec2::new(body_right, body_half_height),
    )
    .to_vec();
    let (body_top, body_bottom) = (
        Vec2::new(body_right, body_half_height),
        Vec2::new(body_right, -body_half_height),
    );
    let (cone_top, cone_bottom) = (Vec2::new(0.0, half), Vec2::new(0.0, -half));
    triangles.push([body_bottom, cone_bottom, cone_top]);
    triangles.push([body_bottom, cone_top, body_top]);
    triangles
}

/// A speaker sending out two sound waves.
fn speaker_icon_triangles() -> Vec<[Vec2; 3]> {
    let half = SPEAKER_SIZE * 0.5;
    let wave_center = Vec2::new(-half * 0.15, 0.0);
    let wave_angles = (-50f32.to_radians(), 50f32.to_radians());
    let mut triangles = speaker_triangles();
    for radius in [half * 0.6, half * 1.05] {
        triangles.extend(arc_band_triangles(
            wave_center,
            radius,
            half * 0.2,
            wave_angles,
        ));
    }
    triangles
}

/// A speaker crossed out.
fn muted_icon_triangles() -> Vec<[Vec2; 3]> {
    let half = SPEAKER_SIZE * 0.5;
    let cross_center = Vec2::new(half * 0.6, 0.0);
    let arm = half * 0.32;
    let mut triangles = speaker_triangles();
    for diagonal in [Vec2::new(arm, arm), Vec2::new(arm, -arm)] {
        triangles.extend(bar_triangles(
            cross_center - diagonal,
            cross_center + diagonal,
            half * 0.2,
        ));
    }
    triangles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(normalized_time: f32, volume: f32) -> VideoStripStatus {
        VideoStripStatus {
            normalized_time,
            playing: false,
            time_label: String::new(),
            volume,
            muted: false,
            volume_popup_open: true,
        }
    }

    fn layout() -> StripLayout {
        StripLayout {
            width: 500.0,
            label_width: 80.0,
            volume_popup_upward: true,
        }
    }

    #[test]
    fn knobs_and_fills_follow_time_and_volume() {
        let layout = layout();
        let status = status(0.25, 0.75);
        let timeline = layout.timeline();
        let knob = part_transform(VideoStripPart::TimelineKnob, &layout, &status, Vec2::ZERO);
        assert!((knob.translation.truncate() - timeline.at(0.25)).length() < 1e-4);
        let fill = part_transform(VideoStripPart::TimelineFill, &layout, &status, Vec2::ZERO);
        assert!((fill.scale.x - timeline.length * 0.25).abs() < 1e-4);
        assert!((fill.translation.x - fill.scale.x * 0.5 - timeline.start.x).abs() < 1e-4);

        // The volume slider is vertical: its fill grows upward from the
        // bottom of the popup.
        let volume = layout.volume();
        let knob = part_transform(VideoStripPart::VolumeKnob, &layout, &status, Vec2::ZERO);
        assert!((knob.translation.truncate() - volume.at(0.75)).length() < 1e-4);
        let fill = part_transform(VideoStripPart::VolumeFill, &layout, &status, Vec2::ZERO);
        assert!((fill.scale.y - volume.length * 0.75).abs() < 1e-4);
        assert!((fill.translation.y - fill.scale.y * 0.5 - volume.start.y).abs() < 1e-4);
    }

    #[test]
    fn the_icons_and_volume_popup_follow_the_status() {
        let mut status = status(0.0, 0.5);
        assert!(VideoStripPart::SpeakerIcon.shown(&status));
        assert!(!VideoStripPart::MutedIcon.shown(&status));
        assert!(VideoStripPart::VolumeTrack.shown(&status));
        status.muted = true;
        status.volume_popup_open = false;
        assert!(!VideoStripPart::SpeakerIcon.shown(&status));
        assert!(VideoStripPart::MutedIcon.shown(&status));
        assert!(!VideoStripPart::VolumeTrack.shown(&status));
        assert!(!VideoStripPart::VolumePopupPanel.shown(&status));
        assert!(VideoStripPart::SpeakerButton.shown(&status));
    }

    #[test]
    fn knobs_sit_over_their_tracks_and_every_control_over_the_panel() {
        let layout = layout();
        let status = status(0.5, 0.5);
        let z = |part| {
            part_transform(part, &layout, &status, Vec2::ZERO)
                .translation
                .z
        };
        assert!(z(VideoStripPart::TimelineKnob) > z(VideoStripPart::TimelineTrack));
        assert!(z(VideoStripPart::PlayPauseButton) > z(VideoStripPart::Panel));
        assert!(z(VideoStripPart::VolumeKnob) > z(VideoStripPart::VolumeRail));
        assert!(z(VideoStripPart::VolumeRail) > z(VideoStripPart::VolumePopupPanel));
        assert!(z(VideoStripPart::SpeakerButton) > z(VideoStripPart::Panel));
    }

    #[test]
    fn icon_meshes_face_the_camera() {
        for triangles in [
            play_icon_triangles(),
            pause_icon_triangles(),
            speaker_icon_triangles(),
            muted_icon_triangles(),
        ] {
            for [a, b, c] in triangles {
                assert!((b - a).perp_dot(c - a) > 0.0, "clockwise triangle");
            }
        }
    }
}
