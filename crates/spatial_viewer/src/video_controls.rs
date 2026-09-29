//! Video playback clocks, per-video volume and mute, and the rules for when
//! a video's control strip — play/pause, a timeline, the elapsed/total time
//! and a speaker button with a volume popup — is shown. The strip itself
//! (drawing, placement, where a press landed) lives in `video_strip`.
//!
//! A video's strip is shown while any of these hold:
//! - the video is selected;
//! - it has been started (owns a clock) and is paused;
//! - one of its sliders is being dragged;
//! - the pointer moved over the video within the last
//!   `VIDEO_CONTROLS_AUTO_HIDE_SECONDS` (hovering the strip itself keeps it
//!   revealed).
//!
//! Merely showing a strip starts nothing: a video gets its clock and decode
//! pipelines on the first play or seek. Clocks outlive their strips, so a
//! video keeps playing after its strip hides and any number of videos can
//! play at once. Play state is the user's intent and only play/pause changes
//! it — a seek keeps it, and a slider drag merely holds the clock still until
//! release.
//!
//! A playing clock runs on wall-clock time and follows its video's sound: it
//! waits while the sound starts, then eases toward where the sound has got
//! to, so picture and sound stay together through frame hitches and output
//! clock drift. A video without sound runs on wall-clock time alone.
//!
//! How a file plays — volume, mute, tracks, delays, speed, looping — lives
//! in [`MediaSettings`], keyed by file, and is read here every frame. The
//! speaker button toggles mute, and hovering it opens the volume popup,
//! which lingers briefly after the pointer leaves both so a diagonal move
//! onto it does not close it.
//!
//! While the app remembers playback positions, each started video's
//! position is recorded as it plays (see [`remember_playback_positions`])
//! and the video opens there next time, unless it was left near its end.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::window::{CursorMoved, PrimaryWindow};
use spatial_viewer_ui::{
    AudioSettings, ControlPanelState, PauseMenuState, PlaybackSettings, RenderResolutionSettings,
    UiInputCapture,
};

use crate::audio_stream::{
    AudioClock, AudioClockSample, AudioPlaybackState, SoundChoice, SoundTrack,
};
use crate::axis_gizmo::cursor_over_axis_gizmo;
use crate::image_loading::{MediaBillboard, BILLBOARD_VIDEO_TEXTURE_SIDE};
use crate::manual_spacing::{cursor_world_ray, nearest_billboard_hit, SelectionState};
use crate::media_probe::{MediaProbe, MediaProbes};
use crate::media_settings::{MediaSettings, VideoFileSettings};
use crate::video_stream::{PlaybackPosition, VideoPlaybackState};
use crate::video_strip::{nearest_strip_hit, VideoStripHitQuery};
use crate::{ExplorerScene, FlyCamera};

/// How long a strip stays revealed after the pointer last moved over its
/// video. Only pointer motion reveals, so flying past videos with a still
/// cursor never flashes strips up.
pub(crate) const VIDEO_CONTROLS_AUTO_HIDE_SECONDS: f32 = 2.0;

/// Seconds one J/L press seeks the selected videos.
pub(crate) const VIDEO_KEYBOARD_SEEK_SECONDS: f32 = 5.0;

/// Fraction of the slider track width the pointer must cross, measured from
/// where it grabbed the knob, before a knob drag starts moving playback.
/// Keeps a click on the knob from being read as a scrub.
pub(crate) const VIDEO_CONTROLS_DRAG_DEADZONE_NORMALIZED: f32 = 0.015;

/// Shortest length a video plays as, so its timeline stays usable.
const MIN_VIDEO_DURATION_SECONDS: f32 = 1.0;

/// Positions this close to a video's start or end are not remembered: a
/// video left there opens at the start next time.
const VIDEO_RESUME_START_MARGIN_SECONDS: f32 = 1.0;
const VIDEO_RESUME_END_MARGIN_SECONDS: f32 = 5.0;
/// A playing video's remembered position is refreshed this often, so a
/// crash loses at most this much; pausing, unloading and exiting record it
/// exactly.
const VIDEO_RESUME_REFRESH_SECONDS: f32 = 5.0;

/// How long the volume popup stays open after the pointer leaves both it
/// and the speaker button.
const VIDEO_VOLUME_POPUP_LINGER_SECONDS: f32 = 0.35;

/// Seconds over which a playing clock closes the gap to its sound. Long
/// enough to smooth the steps in which the output device consumes sound,
/// short enough that drift never builds up.
const VIDEO_AUDIO_FOLLOW_SECONDS: f32 = 0.3;
/// A gap to the sound larger than this is closed at once instead of eased.
const VIDEO_AUDIO_RESYNC_SECONDS: f32 = 0.25;

/// One video's playback clock. Lives in `VideoControlsState` from the first
/// play or seek until the video's billboard unloads.
pub(crate) struct VideoPlaybackClock {
    /// The video's file, which its settings are kept under.
    path: Arc<str>,
    time_seconds: f32,
    duration_seconds: f32,
    /// Whether `duration_seconds` is the file's, or a stand-in until its
    /// probe tells (see [`VideoControlsState::learn_durations`]). A clock
    /// of unknown length records no position: it cannot tell a position
    /// worth reopening at from one near the end.
    duration_known: bool,
    /// Play intent. Only play/pause changes it: seeks keep it, and a slider
    /// drag holds the clock still without touching it (see
    /// [`VideoControlsState::clock_advancing`]).
    playing: bool,
    /// Bumped on every discontinuous time change — a seek, a scrub step or a
    /// loop wrap — so the sound restarts at the new time.
    seek_generation: u64,
}

impl VideoPlaybackClock {
    /// A paused clock at `start_seconds`. A video of unknown length plays as
    /// one second until its length is learned, and keeps its start time
    /// until then rather than clamping it to that stand-in.
    fn new(path: Arc<str>, duration_seconds: Option<f32>, start_seconds: f32) -> Self {
        let start_seconds = start_seconds.max(0.0);
        Self {
            path,
            time_seconds: duration_seconds
                .map_or(start_seconds, |duration| start_seconds.min(duration)),
            duration_seconds: duration_seconds.unwrap_or(1.0),
            duration_known: duration_seconds.is_some(),
            playing: false,
            seek_generation: 0,
        }
    }

    fn learn_duration(&mut self, duration_seconds: f32) {
        self.duration_seconds = duration_seconds;
        self.duration_known = true;
        self.time_seconds = self.time_seconds.min(duration_seconds);
    }

    pub(crate) fn time_seconds(&self) -> f32 {
        self.time_seconds
    }

    pub(crate) fn normalized_time(&self) -> f32 {
        if self.duration_seconds > 0.0 {
            (self.time_seconds / self.duration_seconds).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// The video's end, once its length is known; the stand-in length of
    /// an unknown one must not clamp, stop or wrap its clock.
    fn end_seconds(&self) -> Option<f32> {
        self.duration_known.then_some(self.duration_seconds)
    }

    fn seek_to(&mut self, time_seconds: f32) {
        let end = self.end_seconds().unwrap_or(f32::INFINITY);
        self.time_seconds = time_seconds.clamp(0.0, end);
        self.seek_generation += 1;
    }

    fn at_end(&self) -> bool {
        self.end_seconds()
            .is_some_and(|end| self.time_seconds >= end)
    }

    /// Moves the clock `delta_seconds` of video time on. Past the end a
    /// looping video starts over, and any other stops at the end.
    fn advance(&mut self, delta_seconds: f32, looping: bool) {
        self.time_seconds += delta_seconds;
        let Some(end) = self.end_seconds() else {
            return;
        };
        if self.time_seconds <= end {
            return;
        }
        if looping {
            self.time_seconds %= end.max(0.001);
            self.seek_generation += 1;
        } else {
            self.time_seconds = end;
            self.playing = false;
        }
    }

    /// Advances a playing clock by `delta_seconds` of wall-clock time, at
    /// the file's speed, as its sound directs.
    fn follow(&mut self, delta_seconds: f32, settings: &VideoFileSettings, audio: AudioClock) {
        let video_delta = delta_seconds * settings.speed;
        match audio {
            AudioClock::Prerolling => {}
            AudioClock::Free => self.advance(video_delta, settings.looping),
            AudioClock::Playing(audio_seconds) => {
                let gap = audio_seconds - (self.time_seconds + video_delta);
                let correction = if gap.abs() > VIDEO_AUDIO_RESYNC_SECONDS {
                    1.0
                } else {
                    (delta_seconds / VIDEO_AUDIO_FOLLOW_SECONDS).min(1.0)
                };
                self.advance(video_delta + gap * correction, settings.looping);
            }
        }
    }

    /// The position worth reopening the video at; `None` near either end.
    fn resume_point(&self) -> Option<f32> {
        (self.time_seconds >= VIDEO_RESUME_START_MARGIN_SECONDS
            && self.time_seconds <= self.duration_seconds - VIDEO_RESUME_END_MARGIN_SECONDS)
            .then_some(self.time_seconds)
    }

    /// Records where the video is, exactly, as where to reopen it.
    fn remember_position(&self, settings: &mut MediaSettings) {
        if self.duration_known {
            let resume_seconds = self.resume_point();
            settings.update(&self.path, |video| video.resume_seconds = resume_seconds);
        }
    }
}

/// Where a slider press landed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ScrubGrab {
    /// On the track, at this normalized position: playback jumps there at
    /// once and follows the pointer from then on.
    Track(f32),
    /// On the knob, at this normalized track position: once the pointer
    /// leaves the deadzone around the grab point, playback follows its
    /// motion with the knob keeping its offset under the pointer, so neither
    /// a click nor the start of a drag jumps playback to the pointer.
    Knob(f32),
}

/// The two sliders on a video's strip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StripSlider {
    Timeline,
    Volume,
}

/// A live slider drag. At most one exists, since it follows the held left
/// button.
#[derive(Clone, Debug)]
struct SliderDrag {
    image_id: usize,
    /// The video's file, whose volume a volume drag sets.
    path: Arc<str>,
    slider: StripSlider,
    /// Normalized knob position minus pointer position at the grab.
    pointer_offset: f32,
    /// Where a knob grab happened, until the pointer leaves the deadzone.
    deadzone_origin: Option<f32>,
}

/// What a control strip displays for one video.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VideoStripStatus {
    pub normalized_time: f32,
    pub playing: bool,
    pub time_label: String,
    /// Volume slider position; 0 while muted.
    pub volume: f32,
    pub muted: bool,
    pub volume_popup_open: bool,
}

/// The video whose volume popup the pointer last hovered, and how long the
/// popup has left before it closes. At most one exists: it follows the
/// pointer.
#[derive(Clone, Copy, Debug)]
struct VolumePopupHover {
    image_id: usize,
    remaining_seconds: f32,
}

#[derive(Resource, Default)]
pub(crate) struct VideoControlsState {
    clocks: HashMap<usize, VideoPlaybackClock>,
    /// Seconds each hover-revealed strip has left before it hides.
    reveal_seconds: HashMap<usize, f32>,
    volume_popup: Option<VolumePopupHover>,
    drag: Option<SliderDrag>,
}

impl VideoControlsState {
    pub(crate) fn clock(&self, image_id: usize) -> Option<&VideoPlaybackClock> {
        self.clocks.get(&image_id)
    }

    /// The video's clock, created paused at `start_seconds` the first time.
    /// Only [`start_video`] calls this, alongside starting the decode
    /// pipelines.
    fn start_clock(
        &mut self,
        image_id: usize,
        path: &Arc<str>,
        duration_seconds: Option<f32>,
        start_seconds: f32,
    ) -> &mut VideoPlaybackClock {
        self.clocks.entry(image_id).or_insert_with(|| {
            VideoPlaybackClock::new(path.clone(), duration_seconds, start_seconds)
        })
    }

    /// Videos with a live clock, whether or not their strip is shown.
    pub(crate) fn clock_count(&self) -> usize {
        self.clocks.len()
    }

    /// Video whose timeline the held left button is dragging, if any.
    pub(crate) fn scrubbing_video(&self) -> Option<usize> {
        self.drag
            .as_ref()
            .filter(|drag| drag.slider == StripSlider::Timeline)
            .map(|drag| drag.image_id)
    }

    /// The slider the held left button is dragging, if any.
    pub(crate) fn dragged_slider(&self) -> Option<(usize, StripSlider)> {
        self.drag.as_ref().map(|drag| (drag.image_id, drag.slider))
    }

    /// Opens the video's volume popup, or keeps it open, for another
    /// `VIDEO_VOLUME_POPUP_LINGER_SECONDS`.
    pub(crate) fn hover_volume(&mut self, image_id: usize) {
        self.volume_popup = Some(VolumePopupHover {
            image_id,
            remaining_seconds: VIDEO_VOLUME_POPUP_LINGER_SECONDS,
        });
    }

    /// The popup is hovered (or recently was), or its slider is dragged.
    pub(crate) fn volume_popup_open(&self, image_id: usize) -> bool {
        self.volume_popup
            .is_some_and(|popup| popup.image_id == image_id)
            || self.dragged_slider() == Some((image_id, StripSlider::Volume))
    }

    /// The clock moves at playback rate: it is playing and no slider drag is
    /// holding it.
    pub(crate) fn clock_advancing(&self, image_id: usize) -> bool {
        self.scrubbing_video() != Some(image_id)
            && self.clock(image_id).is_some_and(|clock| clock.playing)
    }

    /// Whether the video's control strip belongs on screen; see the module
    /// docs for the rules.
    pub(crate) fn strip_shown(&self, image_id: usize, selected: bool) -> bool {
        selected
            || self.reveal_seconds.contains_key(&image_id)
            || self
                .drag
                .as_ref()
                .is_some_and(|drag| drag.image_id == image_id)
            || self.clock(image_id).is_some_and(|clock| !clock.playing)
    }

    /// Reveals the video's strip for another `VIDEO_CONTROLS_AUTO_HIDE_SECONDS`.
    pub(crate) fn reveal(&mut self, image_id: usize) {
        self.reveal_seconds
            .insert(image_id, VIDEO_CONTROLS_AUTO_HIDE_SECONDS);
    }

    /// What the video's strip shows, given how its file plays.
    pub(crate) fn strip_status(
        &self,
        image_id: usize,
        unstarted_duration_seconds: f32,
        settings: &VideoFileSettings,
    ) -> VideoStripStatus {
        let muted = settings.silent();
        let (normalized_time, playing, time_label) = match self.clock(image_id) {
            Some(clock) => (
                clock.normalized_time(),
                clock.playing,
                format_video_time_label(clock.time_seconds, clock.duration_seconds),
            ),
            None => (
                0.0,
                false,
                format_video_time_label(0.0, unstarted_duration_seconds),
            ),
        };
        VideoStripStatus {
            normalized_time,
            playing,
            time_label,
            volume: if muted { 0.0 } else { settings.volume },
            muted,
            volume_popup_open: self.volume_popup_open(image_id),
        }
    }

    pub(crate) fn toggle_playing(&mut self, image_id: usize) {
        if let Some(playing) = self.clock(image_id).map(|clock| !clock.playing) {
            self.set_playing(image_id, playing);
        }
    }

    /// Playing a video that stopped at its end starts it over.
    pub(crate) fn set_playing(&mut self, image_id: usize, playing: bool) {
        if let Some(clock) = self.clocks.get_mut(&image_id) {
            if playing && !clock.playing && clock.at_end() {
                clock.seek_to(0.0);
            }
            clock.playing = playing;
        }
    }

    pub(crate) fn seek_by(&mut self, image_id: usize, delta_seconds: f32) {
        if let Some(clock) = self.clocks.get_mut(&image_id) {
            let time_seconds = clock.time_seconds + delta_seconds;
            clock.seek_to(time_seconds);
        }
    }

    /// Starts a timeline drag on a started video. Playback intent is
    /// untouched: the clock just holds still until [`Self::end_slider_drag`].
    pub(crate) fn begin_scrub(&mut self, image_id: usize, grab: ScrubGrab) {
        let Some(clock) = self.clocks.get_mut(&image_id) else {
            return;
        };
        let (pointer_offset, deadzone_origin) = match grab {
            ScrubGrab::Track(normalized) => {
                clock.seek_to(clock.duration_seconds * normalized.clamp(0.0, 1.0));
                (0.0, None)
            }
            ScrubGrab::Knob(normalized) => {
                let normalized = normalized.clamp(0.0, 1.0);
                (clock.normalized_time() - normalized, Some(normalized))
            }
        };
        self.drag = Some(SliderDrag {
            image_id,
            path: clock.path.clone(),
            slider: StripSlider::Timeline,
            pointer_offset,
            deadzone_origin,
        });
    }

    /// Sets the volume of the video in file `path` from a press on its
    /// volume slider, unmuting it, and keeps following the pointer. Needs
    /// no clock: a volume set before the video starts applies once it does.
    pub(crate) fn begin_volume_drag(
        &mut self,
        image_id: usize,
        path: &Arc<str>,
        normalized: f32,
        settings: &mut MediaSettings,
    ) {
        settings.update(path, |video| video.set_volume(normalized));
        self.drag = Some(SliderDrag {
            image_id,
            path: path.clone(),
            slider: StripSlider::Volume,
            pointer_offset: 0.0,
            deadzone_origin: None,
        });
    }

    /// Follows the dragging pointer's normalized position along the dragged
    /// slider.
    pub(crate) fn drag_slider_to(&mut self, normalized: f32, settings: &mut MediaSettings) {
        let Some(drag) = self.drag.as_mut() else {
            return;
        };
        if let Some(origin) = drag.deadzone_origin {
            if (normalized - origin).abs() < VIDEO_CONTROLS_DRAG_DEADZONE_NORMALIZED {
                return;
            }
            drag.deadzone_origin = None;
        }
        let position = (normalized + drag.pointer_offset).clamp(0.0, 1.0);
        match drag.slider {
            StripSlider::Timeline => {
                let Some(clock) = self.clocks.get_mut(&drag.image_id) else {
                    return;
                };
                let time_seconds = clock.duration_seconds * position;
                if (time_seconds - clock.time_seconds).abs() > f32::EPSILON {
                    clock.seek_to(time_seconds);
                }
            }
            StripSlider::Volume => settings.update(&drag.path, |video| video.set_volume(position)),
        }
    }

    /// Releases the slider; a playing video resumes from where it was left.
    /// A released volume slider's popup lingers as if just hovered, rather
    /// than closing under the pointer.
    pub(crate) fn end_slider_drag(&mut self) {
        if let Some(drag) = self.drag.take() {
            if drag.slider == StripSlider::Volume {
                self.hover_volume(drag.image_id);
            }
        }
    }

    /// Moves every playing clock `delta_seconds` of wall-clock time forward,
    /// at its file's speed, as the sound `audio_clock` reports for it
    /// directs.
    fn advance_clocks(
        &mut self,
        delta_seconds: f32,
        settings: &MediaSettings,
        audio_clock: impl Fn(usize, u64) -> AudioClock,
    ) {
        let scrubbing = self.scrubbing_video();
        for (&image_id, clock) in &mut self.clocks {
            if clock.playing && scrubbing != Some(image_id) {
                let audio = audio_clock(image_id, clock.seek_generation);
                clock.follow(delta_seconds, &settings.video(&clock.path), audio);
            }
        }
    }

    /// Records where every started video is, exactly, as where to reopen it.
    fn remember_positions(&self, settings: &mut MediaSettings) {
        for (_, clock) in self.clock_per_file() {
            clock.remember_position(settings);
        }
    }

    /// One started clock per file, the one with the lowest image id: a file
    /// a catalog shows twice has one remembered position, and the two
    /// clocks must not take turns overwriting it.
    fn clock_per_file(&self) -> impl Iterator<Item = (usize, &VideoPlaybackClock)> {
        let mut per_file: HashMap<&str, (usize, &VideoPlaybackClock)> = HashMap::new();
        for (&image_id, clock) in &self.clocks {
            per_file
                .entry(&clock.path)
                .and_modify(|kept| {
                    if image_id < kept.0 {
                        *kept = (image_id, clock);
                    }
                })
                .or_insert((image_id, clock));
        }
        per_file.into_values()
    }

    /// Gives clocks started before their length was known the length their
    /// file's probe found.
    fn learn_durations(&mut self, probes: &MediaProbes) {
        for clock in self.clocks.values_mut() {
            if clock.duration_known {
                continue;
            }
            let probed = probes
                .info(Path::new(&*clock.path))
                .and_then(|info| info.duration_seconds);
            if let Some(duration_seconds) = probed {
                clock.learn_duration(duration_seconds.max(MIN_VIDEO_DURATION_SECONDS));
            }
        }
    }

    fn tick_reveals(&mut self, delta_seconds: f32) {
        self.reveal_seconds.retain(|_, remaining| {
            *remaining -= delta_seconds;
            *remaining > 0.0
        });
        if let Some(popup) = self.volume_popup.as_mut() {
            popup.remaining_seconds -= delta_seconds;
            if popup.remaining_seconds <= 0.0 {
                self.volume_popup = None;
            }
        }
    }

    /// Drops the playback state held for a video whose billboard unloaded;
    /// how its file plays is kept in [`MediaSettings`].
    fn forget(&mut self, image_id: usize) {
        self.clocks.remove(&image_id);
        self.reveal_seconds.remove(&image_id);
        if self
            .drag
            .as_ref()
            .is_some_and(|drag| drag.image_id == image_id)
        {
            self.drag = None;
        }
        if self
            .volume_popup
            .is_some_and(|popup| popup.image_id == image_id)
        {
            self.volume_popup = None;
        }
    }
}

/// Marks every part of a control strip that is hit tested or updated per
/// video, naming the video it belongs to.
#[derive(Component, Clone, Copy)]
pub(crate) struct VideoControlsOwner {
    pub image_id: usize,
}

/// Starting playback touches the clock and both decode pipelines at once;
/// bundled so every entry point (strip presses, the keyboard) starts a video
/// the same way.
#[derive(SystemParam)]
pub(crate) struct VideoPlaybackControl<'w> {
    controls: ResMut<'w, VideoControlsState>,
    video_playback: ResMut<'w, VideoPlaybackState>,
    audio_playback: ResMut<'w, AudioPlaybackState>,
    probes: ResMut<'w, MediaProbes>,
    media_settings: ResMut<'w, MediaSettings>,
    playback_settings: Res<'w, PlaybackSettings>,
    scene: Res<'w, ExplorerScene>,
}

impl VideoPlaybackControl<'_> {
    pub(crate) fn state(&self) -> &VideoControlsState {
        &self.controls
    }

    pub(crate) fn state_mut(&mut self) -> &mut VideoControlsState {
        &mut self.controls
    }

    /// The controls together with the per-file settings a volume change
    /// writes.
    pub(crate) fn state_and_settings(&mut self) -> (&mut VideoControlsState, &mut MediaSettings) {
        (&mut self.controls, &mut self.media_settings)
    }

    pub(crate) fn media_settings_mut(&mut self) -> &mut MediaSettings {
        &mut self.media_settings
    }

    /// Starts the video (see [`start_video`]), then hands back the state so
    /// the caller can act on its clock.
    pub(crate) fn started(&mut self, billboard: &MediaBillboard) -> &mut VideoControlsState {
        start_video(
            VideoStart {
                controls: &mut self.controls,
                video_playback: &mut self.video_playback,
                audio_playback: Some(&mut *self.audio_playback),
                probes: &mut self.probes,
                media_settings: &self.media_settings,
                remember_position: self.playback_settings.remember_position,
            },
            billboard,
            self.scene.max_texture_side,
        );
        &mut self.controls
    }
}

/// Everything starting a video touches.
pub(crate) struct VideoStart<'a> {
    pub controls: &'a mut VideoControlsState,
    pub video_playback: &'a mut VideoPlaybackState,
    pub audio_playback: Option<&'a mut AudioPlaybackState>,
    pub probes: &'a mut MediaProbes,
    pub media_settings: &'a MediaSettings,
    pub remember_position: bool,
}

/// Ensures the video has a clock — paused where it was left when positions
/// are remembered, else at 0:00 — a probe of its file, and running decode
/// pipelines. Every entry point that starts a video goes through here.
pub(crate) fn start_video(start: VideoStart, billboard: &MediaBillboard, max_texture_side: u32) {
    let image_id = billboard.image_id;
    if start.controls.clock(image_id).is_some() {
        return;
    }
    start.probes.request(Path::new(&*billboard.path));
    // The catalog's length, else the probe's if the file was probed before.
    let duration_seconds = billboard
        .duration_seconds
        .or_else(|| {
            start
                .probes
                .info(Path::new(&*billboard.path))
                .and_then(|info| info.duration_seconds)
        })
        .map(|duration| duration.max(MIN_VIDEO_DURATION_SECONDS));
    start.video_playback.activate(
        image_id,
        &*billboard.path,
        max_texture_side.min(BILLBOARD_VIDEO_TEXTURE_SIDE),
    );
    if let Some(audio_playback) = start.audio_playback {
        audio_playback.activate(image_id, &*billboard.path);
    }
    let start_seconds = start
        .remember_position
        .then(|| start.media_settings.video(&billboard.path).resume_seconds)
        .flatten()
        .unwrap_or(0.0);
    start
        .controls
        .start_clock(image_id, &billboard.path, duration_seconds, start_seconds);
}

/// Playback length of a video billboard; an unknown duration plays as one
/// second rather than breaking the slider.
pub(crate) fn video_duration_seconds(billboard: &MediaBillboard) -> f32 {
    billboard
        .duration_seconds
        .unwrap_or(1.0)
        .max(MIN_VIDEO_DURATION_SECONDS)
}

/// Whether a billboard gets a control strip this frame. Hidden billboards
/// (sliced away or behind the camera) never do.
pub(crate) fn video_strip_wanted(
    controls: &VideoControlsState,
    selection: &SelectionState,
    billboard: &MediaBillboard,
    visibility: &Visibility,
) -> bool {
    billboard.is_video
        && *visibility != Visibility::Hidden
        && controls.strip_shown(
            billboard.image_id,
            selection.is_selected(billboard.image_id),
        )
}

/// Stops the videos whose billboards unloaded, recording where they were
/// left first.
pub(crate) fn clear_stale_video_controls(
    mut controls_state: ResMut<VideoControlsState>,
    mut video_playback: ResMut<VideoPlaybackState>,
    mut audio_playback: ResMut<AudioPlaybackState>,
    playback_settings: Res<PlaybackSettings>,
    mut media_settings: ResMut<MediaSettings>,
    billboard_query: Query<&MediaBillboard>,
) {
    if controls_state.clocks.is_empty() && controls_state.reveal_seconds.is_empty() {
        return;
    }
    let existing: HashSet<usize> = billboard_query
        .iter()
        .map(|billboard| billboard.image_id)
        .collect();
    let stale: Vec<usize> = controls_state
        .clocks
        .keys()
        .chain(controls_state.reveal_seconds.keys())
        .filter(|image_id| !existing.contains(image_id))
        .copied()
        .collect();
    for image_id in stale {
        if let (true, Some(clock)) = (
            playback_settings.remember_position,
            controls_state.clock(image_id),
        ) {
            clock.remember_position(&mut media_settings);
        }
        controls_state.forget(image_id);
        video_playback.deactivate(image_id);
        audio_playback.deactivate(image_id);
    }
}

/// Ends a slider drag once the left button is released (or the pause menu
/// opens), before the strip reads this frame's pointer.
pub(crate) fn release_slider_drag(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    pause_menu: Res<PauseMenuState>,
    mut controls_state: ResMut<VideoControlsState>,
) {
    if controls_state.dragged_slider().is_some()
        && (pause_menu.paused || !mouse_buttons.pressed(MouseButton::Left))
    {
        controls_state.end_slider_drag();
    }
}

/// Counts down hover reveals and reveals the video under a moving pointer.
/// Only the nearest billboard under the pointer counts, so an image in front
/// of a video shields it, and a strip in front of a video shields it too:
/// hovering a strip is handled by the strip.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reveal_hovered_videos(
    time: Res<Time>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut cursor_moved: EventReader<CursorMoved>,
    pause_menu: Res<PauseMenuState>,
    ui_capture: Res<UiInputCapture>,
    render_resolution: Res<RenderResolutionSettings>,
    scene: Res<ExplorerScene>,
    mut controls_state: ResMut<VideoControlsState>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<FlyCamera>>,
    billboards: Query<(&MediaBillboard, &GlobalTransform, &Visibility)>,
    strip_parts: VideoStripHitQuery,
) {
    controls_state.tick_reveals(time.delta_secs());
    let pointer_moved = cursor_moved.read().count() > 0;
    // A right-drag look moves the view, not the pointer over the scene.
    if !pointer_moved
        || pause_menu.paused
        || ui_capture.blocks_world_clicks()
        || mouse_buttons.pressed(MouseButton::Right)
    {
        return;
    }
    let Ok(window) = window_query.get_single() else {
        return;
    };
    if cursor_over_axis_gizmo(window) {
        return;
    }
    let Ok((camera, camera_global)) = camera_query.get_single() else {
        return;
    };
    let Some((ray_origin, ray_direction)) =
        cursor_world_ray(window, camera, camera_global, &render_resolution)
    else {
        return;
    };
    let Some(hit) = nearest_billboard_hit(
        ray_origin,
        ray_direction,
        scene.billboard_world_size,
        &billboards,
    ) else {
        return;
    };
    if nearest_strip_hit(ray_origin, ray_direction, &strip_parts, Some(hit.distance)).is_some() {
        return;
    }
    let image_id = hit.image_id;
    let is_video = billboards
        .iter()
        .any(|(billboard, _, _)| billboard.image_id == image_id && billboard.is_video);
    if is_video {
        controls_state.reveal(image_id);
    }
}

/// YouTube-style keys for the selected videos: K plays/pauses them (pausing
/// all when any is playing), J and L seek back and forward. Space is taken
/// by flying up.
pub(crate) fn handle_video_keyboard(
    keyboard: Res<ButtonInput<KeyCode>>,
    pause_menu: Res<PauseMenuState>,
    control_panel: Res<ControlPanelState>,
    selection: Res<SelectionState>,
    billboards: Query<&MediaBillboard>,
    mut playback: VideoPlaybackControl,
) {
    if pause_menu.paused || control_panel.input_focused() {
        return;
    }
    let toggle = keyboard.just_pressed(KeyCode::KeyK);
    let seek_steps = i32::from(keyboard.just_pressed(KeyCode::KeyL))
        - i32::from(keyboard.just_pressed(KeyCode::KeyJ));
    if !toggle && seek_steps == 0 {
        return;
    }
    let selected_videos: Vec<&MediaBillboard> = billboards
        .iter()
        .filter(|billboard| billboard.is_video && selection.is_selected(billboard.image_id))
        .collect();
    if toggle {
        let any_playing = selected_videos.iter().any(|billboard| {
            playback
                .state()
                .clock(billboard.image_id)
                .is_some_and(|clock| clock.playing)
        });
        for billboard in &selected_videos {
            playback
                .started(billboard)
                .set_playing(billboard.image_id, !any_playing);
        }
    }
    if seek_steps != 0 {
        let delta_seconds = seek_steps as f32 * VIDEO_KEYBOARD_SEEK_SECONDS;
        for billboard in &selected_videos {
            playback
                .started(billboard)
                .seek_by(billboard.image_id, delta_seconds);
        }
    }
}

/// Runs on real time: virtual time drops whatever a long frame exceeds its
/// maximum delta by, which the sound playing on regardless never does.
pub(crate) fn advance_video_clocks(
    real_time: Res<Time<Real>>,
    pause_menu: Res<PauseMenuState>,
    audio_playback: Res<AudioPlaybackState>,
    probes: Res<MediaProbes>,
    media_settings: Res<MediaSettings>,
    mut controls_state: ResMut<VideoControlsState>,
) {
    controls_state.learn_durations(&probes);
    if !pause_menu.paused {
        controls_state.advance_clocks(
            real_time.delta_secs(),
            &media_settings,
            |image_id, seek_generation| audio_playback.clock(image_id, seek_generation),
        );
    }
}

/// Keeps every video's audio sink following its clock: advancing starts
/// sound at the current position, pausing (a slider drag and the pause menu
/// included) silences it, and a seek, a loop wrap or new track settings
/// restart it at the new time.
pub(crate) fn sync_video_audio(
    controls_state: Res<VideoControlsState>,
    pause_menu: Res<PauseMenuState>,
    audio_settings: Res<AudioSettings>,
    probes: Res<MediaProbes>,
    media_settings: Res<MediaSettings>,
    mut audio_playback: ResMut<AudioPlaybackState>,
) {
    audio_playback.set_master_volume(audio_settings.volume());
    let controls = &*controls_state;
    audio_playback.sync(controls.clocks.iter().map(|(&image_id, clock)| {
        let settings = media_settings.video(&clock.path);
        AudioClockSample {
            image_id,
            time_seconds: clock.time_seconds,
            playing: controls.clock_advancing(image_id) && !pause_menu.paused,
            seek_generation: clock.seek_generation,
            gain: settings.output_gain(),
            sound: sound_choice(probes.get(Path::new(&*clock.path)), &settings),
        }
    }));
}

/// The sound a file's settings pick, once its probe says what it holds.
fn sound_choice(probe: Option<&MediaProbe>, settings: &VideoFileSettings) -> SoundChoice {
    let track = |stream_index| {
        SoundChoice::Track(SoundTrack {
            stream_index,
            delay_seconds: settings.audio_delay_seconds(),
            speed: settings.speed,
        })
    };
    match probe {
        None | Some(MediaProbe::Pending) => SoundChoice::Pending,
        // Unprobed, ffmpeg picks a track, and finds none if there is none.
        Some(MediaProbe::Failed) => track(None),
        Some(MediaProbe::Ready(info)) => match settings.audio_track(info) {
            Some(audio) => track(Some(audio.stream_index)),
            None => SoundChoice::Silent,
        },
    }
}

/// Keeps each started video's remembered position current while the app
/// remembers positions: exactly once it stops moving, and every
/// `VIDEO_RESUME_REFRESH_SECONDS` of playback while it plays.
pub(crate) fn remember_playback_positions(
    controls_state: Res<VideoControlsState>,
    playback_settings: Res<PlaybackSettings>,
    mut media_settings: ResMut<MediaSettings>,
) {
    if !playback_settings.remember_position {
        return;
    }
    for (image_id, clock) in controls_state.clock_per_file() {
        if !clock.duration_known {
            continue;
        }
        let resume_seconds = clock.resume_point();
        let remembered = media_settings.video(&clock.path).resume_seconds;
        let stale = match (remembered, resume_seconds) {
            (Some(remembered), Some(current)) if controls_state.clock_advancing(image_id) => {
                (current - remembered).abs() >= VIDEO_RESUME_REFRESH_SECONDS
            }
            (remembered, current) => remembered != current,
        };
        if stale {
            media_settings.update(&clock.path, |video| video.resume_seconds = resume_seconds);
        }
    }
}

/// Records every started video's exact position as the app closes, before
/// the settings are written (see `save_media_settings_on_exit`).
pub(crate) fn remember_positions_on_exit(
    mut exit: EventReader<AppExit>,
    controls_state: Res<VideoControlsState>,
    playback_settings: Res<PlaybackSettings>,
    mut media_settings: ResMut<MediaSettings>,
) {
    if exit.read().count() > 0 && playback_settings.remember_position {
        controls_state.remember_positions(&mut media_settings);
    }
}

/// Billboard textures are `RENDER_WORLD`-only, so the main-world copy is
/// gone once extracted and cannot be modified in place; re-inserting under
/// the same handle re-uploads the image but the material's bind group keeps
/// pointing at the previous GPU texture. Each frame therefore gets a fresh
/// handle and the material is re-pointed at it, which rebuilds the bind
/// group; dropping the previous handle frees its texture.
pub(crate) fn apply_video_playback_frame(
    controls: Res<VideoControlsState>,
    probes: Res<MediaProbes>,
    media_settings: Res<MediaSettings>,
    mut video_playback: ResMut<VideoPlaybackState>,
    mut billboard_query: Query<&mut MediaBillboard>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    if controls.clocks.is_empty() {
        return;
    }
    for mut billboard in &mut billboard_query {
        let image_id = billboard.image_id;
        let Some(clock) = controls.clocks.get(&image_id) else {
            continue;
        };
        let Some(probe) = probes.get(Path::new(&*clock.path)) else {
            continue;
        };
        let subtitles = probe
            .info()
            .and_then(|info| media_settings.video(&clock.path).subtitle_burn(info));
        let Some(image) = video_playback.take_frame_for_time(
            image_id,
            PlaybackPosition {
                time_seconds: clock.time_seconds,
                advancing: controls.clock_advancing(image_id),
            },
            probe,
            subtitles,
        ) else {
            continue;
        };
        let assets = &mut billboard.surface_assets;
        if let (Some(texture_handle), Some(material_handle)) = (
            assets.texture_handle.as_mut(),
            assets.material_handle.as_ref(),
        ) {
            if let Some(material) = materials.get_mut(material_handle) {
                let frame_handle = images.add(image);
                material.base_color_texture = Some(frame_handle.clone());
                *texture_handle = frame_handle;
            }
        }
    }
}

/// Where the ray crosses `transform`'s local XY plane, in local coordinates.
/// `None` when the ray runs along the plane or away from it.
pub(crate) fn ray_plane_hit(
    ray_origin: Vec3,
    ray_direction: Vec3,
    transform: &GlobalTransform,
) -> Option<Vec2> {
    let inverse = transform.compute_matrix().inverse();
    let local_origin = inverse.transform_point3(ray_origin);
    let local_direction = inverse.transform_vector3(ray_direction);
    if local_direction.z.abs() <= f32::EPSILON {
        return None;
    }
    let local_distance = -local_origin.z / local_direction.z;
    if local_distance < 0.0 {
        return None;
    }
    let local_hit = (local_origin + local_direction * local_distance).truncate();
    // A degenerate (zero-scale) transform has no inverse; its NaNs would
    // slip through every comparison a caller makes.
    local_hit.is_finite().then_some(local_hit)
}

/// Where the ray crosses the rectangle with `half_extents` in `transform`'s
/// local XY plane: the world distance along the ray and the local hit point.
pub(crate) fn ray_rect_hit(
    ray_origin: Vec3,
    ray_direction: Vec3,
    transform: &GlobalTransform,
    half_extents: Vec2,
) -> Option<(f32, Vec2)> {
    let local_hit = ray_plane_hit(ray_origin, ray_direction, transform)?;
    if local_hit.x.abs() > half_extents.x || local_hit.y.abs() > half_extents.y {
        return None;
    }
    let world_hit = transform.transform_point(local_hit.extend(0.0));
    Some(((world_hit - ray_origin).length(), local_hit))
}

pub(crate) fn format_video_time_label(elapsed_seconds: f32, duration_seconds: f32) -> String {
    format!(
        "{} / {}",
        format_seconds_mmss(elapsed_seconds),
        format_seconds_mmss(duration_seconds)
    )
}

pub(crate) fn format_seconds_mmss(seconds: f32) -> String {
    let total_seconds = seconds.max(0.0).round() as u64;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let secs = total_seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes}:{secs:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media_probe::{MediaInfo, MediaTrack};

    const VIDEO: usize = 7;
    const PATH: &str = "clip.mp4";

    fn path() -> Arc<str> {
        PATH.into()
    }

    fn state_with_playing_video() -> VideoControlsState {
        let mut state = VideoControlsState::default();
        state.start_clock(VIDEO, &path(), Some(10.0), 0.0).playing = true;
        state
    }

    fn free_run(state: &mut VideoControlsState, settings: &MediaSettings, seconds: f32) {
        state.advance_clocks(seconds, settings, |_, _| AudioClock::Free);
    }

    fn time(state: &VideoControlsState) -> f32 {
        state.clock(VIDEO).expect("started").time_seconds
    }

    #[test]
    fn format_seconds_mmss_pads_and_rounds() {
        assert_eq!(format_seconds_mmss(0.0), "0:00");
        assert_eq!(format_seconds_mmss(5.4), "0:05");
        assert_eq!(format_seconds_mmss(65.6), "1:06");
        assert_eq!(format_seconds_mmss(-3.0), "0:00");
    }

    #[test]
    fn format_seconds_mmss_includes_hours_when_needed() {
        assert_eq!(format_seconds_mmss(3661.0), "1:01:01");
    }

    #[test]
    fn format_video_time_label_joins_elapsed_and_total() {
        assert_eq!(format_video_time_label(12.0, 65.0), "0:12 / 1:05");
    }

    #[test]
    fn seeking_keeps_playing_and_bumps_the_seek_generation() {
        let mut state = state_with_playing_video();
        state.seek_by(VIDEO, 4.0);
        let clock = state.clock(VIDEO).expect("started");
        assert!(clock.playing);
        assert_eq!(clock.time_seconds, 4.0);
        assert_eq!(clock.seek_generation, 1);

        state.seek_by(VIDEO, -10.0);
        assert_eq!(time(&state), 0.0);
    }

    #[test]
    fn a_track_scrub_holds_the_clock_and_resumes_on_release() {
        let mut settings = MediaSettings::in_memory();
        let mut state = state_with_playing_video();
        state.begin_scrub(VIDEO, ScrubGrab::Track(0.5));
        assert_eq!(time(&state), 5.0);
        assert!(!state.clock_advancing(VIDEO));

        free_run(&mut state, &settings, 1.0);
        state.drag_slider_to(0.25, &mut settings);
        let clock = state.clock(VIDEO).expect("started");
        assert_eq!(clock.time_seconds, 2.5);
        assert!(clock.playing);

        state.end_slider_drag();
        assert!(state.clock_advancing(VIDEO));
        free_run(&mut state, &settings, 1.0);
        assert_eq!(time(&state), 3.5);
    }

    #[test]
    fn a_knob_grab_keeps_its_offset_and_only_seeks_past_the_deadzone() {
        let mut settings = MediaSettings::in_memory();
        let mut state = state_with_playing_video();
        state.seek_by(VIDEO, 5.0);
        // Grabbed off-center, to the right of the knob at 0.5.
        let grab = 0.53;
        state.begin_scrub(VIDEO, ScrubGrab::Knob(grab));
        state.drag_slider_to(grab, &mut settings);
        state.drag_slider_to(
            grab + VIDEO_CONTROLS_DRAG_DEADZONE_NORMALIZED * 0.5,
            &mut settings,
        );
        assert_eq!(time(&state), 5.0);

        state.drag_slider_to(grab + 0.1, &mut settings);
        assert!((time(&state) - 6.0).abs() < 1e-4);
        // Past the deadzone, even tiny motions follow the pointer.
        state.drag_slider_to(grab + 0.101, &mut settings);
        assert!((time(&state) - 6.01).abs() < 1e-4);
    }

    #[test]
    fn a_loop_wrap_counts_as_a_seek() {
        let settings = MediaSettings::in_memory();
        let mut state = state_with_playing_video();
        free_run(&mut state, &settings, 10.5);
        let clock = state.clock(VIDEO).expect("started");
        assert!((clock.time_seconds - 0.5).abs() < 1e-4);
        assert_eq!(clock.seek_generation, 1);
    }

    #[test]
    fn without_looping_a_video_stops_at_its_end_and_play_starts_it_over() {
        let mut settings = MediaSettings::in_memory();
        settings.update(&path(), |video| video.looping = false);
        let mut state = state_with_playing_video();
        free_run(&mut state, &settings, 10.5);
        let clock = state.clock(VIDEO).expect("started");
        assert_eq!(clock.time_seconds, 10.0);
        assert!(!clock.playing);
        assert_eq!(clock.seek_generation, 0);

        state.toggle_playing(VIDEO);
        let clock = state.clock(VIDEO).expect("started");
        assert!(clock.playing);
        assert_eq!(clock.time_seconds, 0.0);
        assert_eq!(clock.seek_generation, 1);
    }

    #[test]
    fn a_faster_video_runs_its_clock_faster() {
        let mut settings = MediaSettings::in_memory();
        settings.update(&path(), |video| video.speed = 1.5);
        let mut state = state_with_playing_video();
        free_run(&mut state, &settings, 2.0);
        assert!((time(&state) - 3.0).abs() < 1e-4);
    }

    #[test]
    fn a_playing_clock_waits_for_its_sound_then_eases_toward_it() {
        let settings = MediaSettings::in_memory();
        let mut state = state_with_playing_video();
        state.advance_clocks(1.0, &settings, |_, _| AudioClock::Prerolling);
        assert_eq!(time(&state), 0.0);

        // A small gap closes over the follow period, not in one frame.
        let frame = 0.016;
        state.advance_clocks(frame, &settings, |_, _| AudioClock::Playing(0.1));
        let eased = time(&state);
        assert!(eased > frame && eased < 0.1);

        // A gap past the resync limit closes at once, without a seek.
        state.advance_clocks(frame, &settings, |_, _| AudioClock::Playing(4.0));
        let clock = state.clock(VIDEO).expect("started");
        assert!((clock.time_seconds - 4.0).abs() < 1e-4);
        assert_eq!(clock.seek_generation, 0);
    }

    #[test]
    fn strips_show_for_selected_paused_scrubbed_or_recently_hovered_videos() {
        let mut state = VideoControlsState::default();
        // An untouched poster only shows when selected.
        assert!(!state.strip_shown(VIDEO, false));
        assert!(state.strip_shown(VIDEO, true));

        // A started, paused video always shows.
        state.start_clock(VIDEO, &path(), Some(10.0), 0.0);
        assert!(state.strip_shown(VIDEO, false));

        // A playing video shows while revealed or scrubbed.
        state.set_playing(VIDEO, true);
        assert!(!state.strip_shown(VIDEO, false));
        state.reveal(VIDEO);
        assert!(state.strip_shown(VIDEO, false));
        state.tick_reveals(VIDEO_CONTROLS_AUTO_HIDE_SECONDS);
        assert!(!state.strip_shown(VIDEO, false));
        state.begin_scrub(VIDEO, ScrubGrab::Knob(0.0));
        assert!(state.strip_shown(VIDEO, false));
    }

    #[test]
    fn a_volume_drag_follows_the_pointer_without_holding_the_clock() {
        let mut settings = MediaSettings::in_memory();
        let mut state = state_with_playing_video();
        state.begin_volume_drag(VIDEO, &path(), 0.4, &mut settings);
        assert_eq!(settings.video(PATH).volume, 0.4);
        assert!(state.clock_advancing(VIDEO));
        assert!(state.strip_shown(VIDEO, false));
        state.drag_slider_to(1.3, &mut settings);
        assert_eq!(settings.video(PATH).volume, 1.0);
        state.drag_slider_to(0.8, &mut settings);
        state.end_slider_drag();

        // The volume belongs to the file, whatever id a catalog reload gives
        // its video.
        state.forget(VIDEO);
        let renumbered = VIDEO + 1;
        state.start_clock(renumbered, &path(), Some(10.0), 0.0);
        let status = state.strip_status(renumbered, 10.0, &settings.video(PATH));
        assert_eq!(status.volume, 0.8);
    }

    #[test]
    fn a_muted_strip_shows_the_level_at_zero() {
        let state = state_with_playing_video();
        let mut video = VideoFileSettings::default();
        video.set_volume(0.6);
        video.toggle_mute();
        let status = state.strip_status(VIDEO, 10.0, &video);
        assert!(status.muted);
        assert_eq!(status.volume, 0.0);
    }

    #[test]
    fn the_volume_popup_lingers_after_hover_and_stays_open_while_dragged() {
        let mut settings = MediaSettings::in_memory();
        let mut state = state_with_playing_video();
        state.hover_volume(VIDEO);
        let status = state.strip_status(VIDEO, 10.0, &VideoFileSettings::default());
        assert!(status.volume_popup_open);
        state.tick_reveals(VIDEO_VOLUME_POPUP_LINGER_SECONDS * 0.5);
        assert!(state.volume_popup_open(VIDEO));
        state.tick_reveals(VIDEO_VOLUME_POPUP_LINGER_SECONDS);
        assert!(!state.volume_popup_open(VIDEO));

        state.begin_volume_drag(VIDEO, &path(), 0.5, &mut settings);
        state.tick_reveals(VIDEO_VOLUME_POPUP_LINGER_SECONDS * 2.0);
        assert!(state.volume_popup_open(VIDEO));
        assert!(!state.volume_popup_open(VIDEO + 1));
        // Released, it lingers like a hover instead of closing at once.
        state.end_slider_drag();
        assert!(state.volume_popup_open(VIDEO));
        state.tick_reveals(VIDEO_VOLUME_POPUP_LINGER_SECONDS);
        assert!(!state.volume_popup_open(VIDEO));
    }

    #[test]
    fn positions_near_either_end_are_not_worth_resuming() {
        let mut state = state_with_playing_video();
        let resume_point =
            |state: &VideoControlsState| state.clock(VIDEO).expect("started").resume_point();
        assert_eq!(resume_point(&state), None);
        state.seek_by(VIDEO, 3.0);
        assert_eq!(resume_point(&state), Some(3.0));
        state.seek_by(VIDEO, 4.0);
        assert_eq!(resume_point(&state), None);
    }

    #[test]
    fn remembered_positions_are_recorded_and_cleared() {
        let mut settings = MediaSettings::in_memory();
        let mut state = state_with_playing_video();
        state.seek_by(VIDEO, 3.0);
        state.remember_positions(&mut settings);
        assert_eq!(settings.video(PATH).resume_seconds, Some(3.0));

        state.seek_by(VIDEO, -3.0);
        state.remember_positions(&mut settings);
        assert_eq!(settings.video(PATH).resume_seconds, None);

        // A reopened video starts where it was remembered.
        let mut reopened = VideoControlsState::default();
        reopened.start_clock(VIDEO, &path(), Some(10.0), 4.5);
        assert_eq!(time(&reopened), 4.5);
    }

    #[test]
    fn a_video_of_unknown_length_keeps_its_position_until_the_length_is_known() {
        let mut settings = MediaSettings::in_memory();
        settings.update(&path(), |video| video.resume_seconds = Some(300.0));
        let mut state = VideoControlsState::default();
        state.start_clock(VIDEO, &path(), None, 300.0);
        assert_eq!(time(&state), 300.0);
        // Playing and seeking before the length is known keep the position.
        state.toggle_playing(VIDEO);
        state.seek_by(VIDEO, 2.0);
        free_run(&mut state, &settings, 0.5);
        assert_eq!(time(&state), 302.5);
        state.seek_by(VIDEO, -2.5);
        // Unknown length: nothing is recorded over the remembered position.
        state.remember_positions(&mut settings);
        assert_eq!(settings.video(PATH).resume_seconds, Some(300.0));

        state
            .clocks
            .get_mut(&VIDEO)
            .expect("started")
            .learn_duration(1440.0);
        assert_eq!(time(&state), 300.0);
        state.seek_by(VIDEO, 12.0);
        state.remember_positions(&mut settings);
        assert_eq!(settings.video(PATH).resume_seconds, Some(312.0));
    }

    #[test]
    fn a_file_shown_twice_is_remembered_by_its_first_clock() {
        let mut settings = MediaSettings::in_memory();
        let mut state = VideoControlsState::default();
        state.start_clock(VIDEO + 1, &path(), Some(10.0), 6.0);
        state.start_clock(VIDEO, &path(), Some(10.0), 3.0);
        state.remember_positions(&mut settings);
        assert_eq!(settings.video(PATH).resume_seconds, Some(3.0));
    }

    #[test]
    fn sound_follows_the_chosen_track_once_the_file_is_probed() {
        let track = |stream_index, default| MediaTrack {
            stream_index,
            kind_index: stream_index - 1,
            codec: "aac".to_owned(),
            language: None,
            title: None,
            default,
        };
        let info = MediaInfo {
            video: None,
            audio_tracks: vec![track(1, false), track(2, true)],
            subtitle_tracks: Vec::new(),
            duration_seconds: None,
            size_bytes: None,
            start_seconds: 0.0,
        };
        let probe = MediaProbe::Ready(Arc::new(info));
        let mut settings = VideoFileSettings::default();
        assert_eq!(sound_choice(None, &settings), SoundChoice::Pending);
        let stream_index = |choice| match choice {
            SoundChoice::Track(track) => track.stream_index,
            _ => panic!("expected a track, got {choice:?}"),
        };
        assert_eq!(stream_index(sound_choice(Some(&probe), &settings)), Some(2));
        settings.audio_track = Some(1);
        settings.audio_delay_ms = 250;
        let SoundChoice::Track(chosen) = sound_choice(Some(&probe), &settings) else {
            panic!("expected a track");
        };
        assert_eq!(chosen.stream_index, Some(1));
        assert_eq!(chosen.delay_seconds, 0.25);
        assert_eq!(
            stream_index(sound_choice(Some(&MediaProbe::Failed), &settings)),
            None
        );

        let silent = MediaProbe::Ready(Arc::new(MediaInfo {
            video: None,
            audio_tracks: Vec::new(),
            subtitle_tracks: Vec::new(),
            duration_seconds: None,
            size_bytes: None,
            start_seconds: 0.0,
        }));
        assert_eq!(sound_choice(Some(&silent), &settings), SoundChoice::Silent);
    }

    #[test]
    fn a_degenerate_rect_is_never_hit() {
        let collapsed = GlobalTransform::from(Transform::from_scale(Vec3::new(0.0, 1.0, 1.0)));
        assert!(ray_rect_hit(Vec3::Z, Vec3::NEG_Z, &collapsed, Vec2::splat(0.5)).is_none());
        let unit = GlobalTransform::IDENTITY;
        assert!(ray_rect_hit(Vec3::Z, Vec3::NEG_Z, &unit, Vec2::splat(0.5)).is_some());
    }

    #[test]
    fn toggling_play_needs_a_started_clock() {
        let mut state = VideoControlsState::default();
        state.toggle_playing(VIDEO);
        assert!(state.clock(VIDEO).is_none());

        state.start_clock(VIDEO, &path(), Some(10.0), 0.0);
        state.toggle_playing(VIDEO);
        assert!(state.clock(VIDEO).expect("started").playing);
    }

    #[test]
    fn forgetting_a_video_drops_its_clock_reveal_and_scrub() {
        let mut state = state_with_playing_video();
        state.reveal(VIDEO);
        state.hover_volume(VIDEO);
        state.begin_scrub(VIDEO, ScrubGrab::Knob(0.0));
        state.forget(VIDEO);
        assert!(state.clock(VIDEO).is_none());
        assert!(!state.strip_shown(VIDEO, false));
        assert!(!state.volume_popup_open(VIDEO));
        assert_eq!(state.scrubbing_video(), None);
    }
}
