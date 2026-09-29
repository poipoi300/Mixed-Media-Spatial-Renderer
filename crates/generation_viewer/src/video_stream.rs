use std::{collections::HashMap, path::PathBuf};

use bevy::prelude::*;
use crossbeam_channel::{Receiver, TryRecvError};

use crate::ffmpeg_pipe::{spawn_ffmpeg_stream, StreamEnd, StreamItem};
use crate::media_decode::{
    decoded_image_to_bevy_image, video_decode_side, video_stream_arguments, DecodedImage,
    SubtitleBurn, VideoSource, VideoTiming, DEFAULT_MAX_VIDEO_FPS,
};
use crate::media_probe::MediaProbe;

/// Frames a stream decodes ahead of the playhead, in seconds of playback.
/// Each frame is uncompressed RGBA (4 MiB at the 1024 px video side), so this
/// is kept to what rides out a decoder briefly falling behind.
const VIDEO_READ_AHEAD_SECONDS: f32 = 0.5;
const VIDEO_MIN_READ_AHEAD_FRAMES: usize = 2;
const VIDEO_MAX_READ_AHEAD_FRAMES: usize = 30;
/// A playhead this far past the last frame a stream decoded is reached
/// sooner by restarting ffmpeg at it (one keyframe seek) than by decoding
/// every frame up to it.
const VIDEO_RESTART_AHEAD_SECONDS: f32 = 3.0;
/// Frames a playing clock may step back (easing toward its sound) while the
/// shown frame stays up, instead of restarting the stream to go back.
const VIDEO_BACKWARD_TOLERANCE_FRAMES: u64 = 2;

/// Frame streams for every video that has been started, keyed by image id so
/// any number of videos can play at once. Each video decodes through one
/// ffmpeg process that runs from its playhead onward (see
/// [`crate::ffmpeg_pipe`]); it is restarted only when playback jumps
/// somewhere the stream cannot reach by decoding forward, and dropped while
/// the video is paused on the frame it shows. Subtitles are drawn into the
/// frames as they decode, so choosing others restarts the stream too.
#[derive(Resource)]
pub struct VideoPlaybackState {
    /// Ceiling on any video's playback rate; slower sources play natively.
    max_fps: f32,
    active: HashMap<usize, ActiveVideoPlayback>,
}

impl Default for VideoPlaybackState {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_VIDEO_FPS)
    }
}

#[derive(Debug)]
struct ActiveVideoPlayback {
    path: PathBuf,
    max_texture_side: u32,
    /// `None` until the file's probe answers; nothing is decoded or shown
    /// before then, because every frame index depends on the rate.
    timing: Option<VideoTiming>,
    /// The picture stream the probe found, once it answered.
    video_stream_index: Option<usize>,
    /// Subtitles the running stream draws, and the shown frame carries.
    subtitles: Option<SubtitleBurn>,
    stream: Option<FrameStream>,
    applied_frame_index: Option<u64>,
    /// The file's last frame at the playback rate, once a stream reached
    /// the end. The container's duration can overshoot it.
    last_frame_index: Option<u64>,
    /// How far back the next end probe starts after a stream started past
    /// the end found no frames; doubles on every miss, so an overshooting
    /// duration costs a handful of restarts rather than one per frame.
    end_probe_frames: u64,
    /// Set when a stream failed without a frame; the video then stays on
    /// the frame it shows instead of respawning ffmpeg every frame.
    broken: bool,
}

/// One running ffmpeg frame stream.
#[derive(Debug)]
struct FrameStream {
    records: Receiver<StreamItem<Vec<u8>>>,
    start_frame_index: u64,
    /// Index the next frame read off the stream carries: frames come out one
    /// per playback frame from the start frame on.
    next_frame_index: u64,
    /// A frame read off the stream that is not due yet.
    pending: Option<(u64, Vec<u8>)>,
    end: Option<StreamEnd>,
}

impl FrameStream {
    fn new(records: Receiver<StreamItem<Vec<u8>>>, start_frame_index: u64) -> Self {
        Self {
            records,
            start_frame_index,
            next_frame_index: start_frame_index,
            pending: None,
            end: None,
        }
    }

    fn delivered_any(&self) -> bool {
        self.next_frame_index > self.start_frame_index
    }

    fn failed_mid_video(&self) -> bool {
        self.delivered_any() && matches!(self.end, Some(StreamEnd::Failed(_)))
    }

    /// The earliest frame this stream can still show.
    fn earliest_frame_index(&self) -> u64 {
        self.pending
            .as_ref()
            .map_or(self.next_frame_index, |(index, _)| *index)
    }

    /// Reads frames up to `target`, keeping the latest one due and holding
    /// the first one past it. Returns whether the stream's end was read.
    fn read_to(&mut self, target: u64, due: &mut Option<(u64, Vec<u8>)>) -> bool {
        if self.end.is_some() {
            return false;
        }
        loop {
            let frame = match self.pending.take() {
                Some(frame) => frame,
                None => match self.records.try_recv() {
                    Ok(StreamItem::Record(rgba)) => {
                        let index = self.next_frame_index;
                        self.next_frame_index += 1;
                        (index, rgba)
                    }
                    Ok(StreamItem::End(end)) => {
                        self.end = Some(end);
                        return true;
                    }
                    Err(TryRecvError::Empty) => return false,
                    // The reader always sends an end before it hangs up,
                    // unless it panicked; either way no more frames come.
                    Err(TryRecvError::Disconnected) => {
                        self.end = Some(StreamEnd::Failed("frame reader stopped".to_owned()));
                        return true;
                    }
                },
            };
            if frame.0 > target {
                self.pending = Some(frame);
                return false;
            }
            *due = Some(frame);
        }
    }
}

/// Where a video's clock stands this frame, as the frame lookup needs it.
#[derive(Clone, Copy, Debug)]
pub struct PlaybackPosition {
    pub time_seconds: f32,
    /// The clock is moving at playback rate: playing and not held by a
    /// slider drag.
    pub advancing: bool,
}

impl VideoPlaybackState {
    pub fn new(max_fps: f32) -> Self {
        Self {
            max_fps,
            active: HashMap::new(),
        }
    }

    /// Registers a video for frame decoding; its file must be probed (see
    /// [`crate::media_probe::MediaProbes`]) before frames flow. A video
    /// already registered with the same path and texture size keeps its
    /// stream and playback progress.
    pub fn activate(&mut self, image_id: usize, path: impl Into<PathBuf>, max_texture_side: u32) {
        let path = path.into();
        let known = match self.active.get(&image_id) {
            Some(existing) if existing.path == path => {
                if existing.max_texture_side == max_texture_side {
                    return;
                }
                // Only the frame size changed; the rest is the file's own.
                Some((existing.timing, existing.video_stream_index))
            }
            _ => None,
        };
        let (timing, video_stream_index) = known.unwrap_or_default();
        self.active.insert(
            image_id,
            ActiveVideoPlayback {
                path,
                max_texture_side,
                timing,
                video_stream_index,
                subtitles: None,
                stream: None,
                applied_frame_index: None,
                last_frame_index: None,
                end_probe_frames: 1,
                broken: false,
            },
        );
    }

    pub fn deactivate(&mut self, image_id: usize) {
        // Dropping the stream's receiver stops its ffmpeg process.
        self.active.remove(&image_id);
    }

    /// Where to seek to decode again the frame a video shows, once it shows
    /// a decoded one (see [`VideoTiming::frame_seek_seconds`]).
    pub fn shown_frame_seek_seconds(&self, image_id: usize) -> Option<f32> {
        let active = self.active.get(&image_id)?;
        Some(
            active
                .timing?
                .frame_seek_seconds(active.applied_frame_index?),
        )
    }

    /// ffmpeg frame streams currently running across all videos.
    pub fn stream_count(&self) -> usize {
        self.active
            .values()
            .filter(|active| active.stream.is_some())
            .count()
    }

    /// Bytes one decoded frame of every streaming video occupies on the GPU;
    /// video frames are uploaded on top of the image tile budget so a frame
    /// that lands in the same frame as a tile upload is never skipped.
    pub fn frame_upload_bytes(&self) -> usize {
        self.active
            .values()
            .filter(|active| active.stream.is_some())
            .map(|active| {
                let side = video_decode_side(active.max_texture_side) as usize;
                side * side * 4
            })
            .sum()
    }

    /// The frame to show next for a video's clock, when it changes. Starts,
    /// restarts or stops the video's stream as the clock and the chosen
    /// `subtitles` require. `probe` is the video file's.
    pub fn take_frame_for_time(
        &mut self,
        image_id: usize,
        position: PlaybackPosition,
        probe: &MediaProbe,
        subtitles: Option<SubtitleBurn>,
    ) -> Option<Image> {
        let max_fps = self.max_fps;
        let active = self.active.get_mut(&image_id)?;
        let timing = active.resolve_timing(probe, max_fps)?;
        if active.broken {
            return None;
        }
        active.choose_subtitles(subtitles);
        let mut target = timing.frame_index(position.time_seconds);
        if let Some(last_frame_index) = active.last_frame_index {
            target = target.min(last_frame_index);
        }
        if active.applied_frame_index == Some(target)
            && (!position.advancing || active.last_frame_index == Some(target))
        {
            // Paused on the frame it shows, or held on the last one: nothing
            // more to decode.
            active.stream = None;
            return None;
        }
        if active.stream_must_restart(timing, target, position.advancing) {
            active.start_stream(image_id, timing, target);
        }
        let image = active.read_stream_to(image_id, target);
        if active.broken
            || active
                .stream
                .as_ref()
                .is_some_and(FrameStream::failed_mid_video)
        {
            // A mid-video failure restarts from the target next frame; one
            // that fails again before a frame marks the video broken.
            active.stream = None;
        }
        image
    }
}

impl ActiveVideoPlayback {
    /// The playback timing, settled from the file's probe the first time it
    /// has answered.
    fn resolve_timing(&mut self, probe: &MediaProbe, max_fps: f32) -> Option<VideoTiming> {
        if self.timing.is_none() {
            let video = match probe {
                MediaProbe::Pending => return None,
                MediaProbe::Ready(info) => info.video.as_ref(),
                // The rate is unknown: play at the cap, which ffmpeg
                // resamples to, wasting only duplicate frames.
                MediaProbe::Failed => None,
            };
            self.video_stream_index = video.map(|video| video.stream_index);
            self.timing = Some(VideoTiming::new(
                video.and_then(|video| video.frame_rate),
                max_fps,
            ));
        }
        self.timing
    }

    /// Switches the subtitles drawn into the frames. The shown frame carries
    /// the old ones, so it is decoded again.
    fn choose_subtitles(&mut self, subtitles: Option<SubtitleBurn>) {
        if self.subtitles != subtitles {
            self.subtitles = subtitles;
            self.stream = None;
            self.applied_frame_index = None;
        }
    }

    /// Whether the running stream cannot bring up `target` by decoding
    /// forward: there is none, the target is behind what it still holds, or
    /// it is so far ahead that a fresh seek gets there sooner. A stream that
    /// has not delivered its first frame is left to land first, so a scrub
    /// keeps one decode in flight instead of respawning ffmpeg every frame.
    fn stream_must_restart(&self, timing: VideoTiming, target: u64, advancing: bool) -> bool {
        let Some(stream) = self.stream.as_ref() else {
            return true;
        };
        if !stream.delivered_any() && stream.end.is_none() {
            return false;
        }
        let earliest = stream.earliest_frame_index();
        if target >= earliest {
            let restart_ahead_frames = timing.frame_index(VIDEO_RESTART_AHEAD_SECONDS).max(1);
            return stream.end.is_none() && target >= earliest + restart_ahead_frames;
        }
        match self.applied_frame_index {
            Some(applied) if applied == target => false,
            Some(applied) => !advancing || target + VIDEO_BACKWARD_TOLERANCE_FRAMES < applied,
            None => true,
        }
    }

    fn start_stream(&mut self, image_id: usize, timing: VideoTiming, start_frame_index: u64) {
        self.stream = None;
        let side = video_decode_side(self.max_texture_side);
        let read_ahead_frames = ((timing.fps() * VIDEO_READ_AHEAD_SECONDS).ceil() as usize)
            .clamp(VIDEO_MIN_READ_AHEAD_FRAMES, VIDEO_MAX_READ_AHEAD_FRAMES);
        match spawn_ffmpeg_stream(
            format!("video-{image_id}"),
            video_stream_arguments(
                &VideoSource {
                    path: self.path.clone(),
                    video_stream_index: self.video_stream_index,
                    subtitles: self.subtitles,
                },
                timing,
                start_frame_index,
                side,
            ),
            side as usize * side as usize * 4,
            read_ahead_frames,
            |rgba| rgba,
        ) {
            Ok(records) => self.stream = Some(FrameStream::new(records, start_frame_index)),
            Err(error) => {
                eprintln!("Failed to stream video {image_id}: {error:#}");
                self.broken = true;
            }
        }
    }

    /// Advances the stream to `target` and returns the frame to show, if it
    /// changed. A stream that lands just past the target (the clock eased
    /// back) still shows its first frame rather than nothing.
    fn read_stream_to(&mut self, image_id: usize, target: u64) -> Option<Image> {
        let stream = self.stream.as_mut()?;
        let mut due = None;
        let ended = stream.read_to(target, &mut due);
        if due.is_none() && self.applied_frame_index.is_none() {
            due = stream.pending.take();
        }
        if ended {
            self.learn_from_stream_end(image_id);
        }
        let (frame_index, rgba) = due?;
        if self.applied_frame_index == Some(frame_index) {
            return None;
        }
        self.applied_frame_index = Some(frame_index);
        let side = video_decode_side(self.max_texture_side);
        Some(decoded_image_to_bevy_image(DecodedImage::full(side, rgba)))
    }

    /// Records what a finished stream says about the file: where its frames
    /// end, or that it cannot be decoded.
    fn learn_from_stream_end(&mut self, image_id: usize) {
        let Some(stream) = self.stream.as_ref() else {
            return;
        };
        let Some(end) = stream.end.as_ref() else {
            return;
        };
        if stream.delivered_any() {
            match end {
                StreamEnd::Failed(error) => {
                    eprintln!("Video {image_id} stopped decoding, restarting: {error}");
                }
                StreamEnd::Finished | StreamEnd::MissingStream => {
                    self.last_frame_index = Some(stream.next_frame_index - 1);
                    self.end_probe_frames = 1;
                }
            }
            return;
        }
        match end {
            StreamEnd::Finished if stream.start_frame_index > 0 => {
                // Started past the last frame; look again further back.
                self.last_frame_index = Some(
                    stream
                        .start_frame_index
                        .saturating_sub(self.end_probe_frames),
                );
                self.end_probe_frames = self.end_probe_frames.saturating_mul(2);
            }
            StreamEnd::Finished => {
                eprintln!("Video {image_id} has no frames to play");
                self.broken = true;
            }
            StreamEnd::MissingStream => {
                eprintln!("Video {image_id} has no video stream");
                self.broken = true;
            }
            StreamEnd::Failed(error) => {
                eprintln!("Failed to decode video {image_id}: {error}");
                self.broken = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crossbeam_channel::{bounded, Sender};

    use super::*;
    use crate::media_probe::{MediaInfo, VideoStreamInfo};

    const TEST_PATH: &str = "test.mp4";
    const VIDEO: usize = 7;

    fn at(time_seconds: f32, advancing: bool) -> PlaybackPosition {
        PlaybackPosition {
            time_seconds,
            advancing,
        }
    }

    /// A probe that found a picture stream at `frame_rate`.
    fn probe_at(frame_rate: Option<f32>) -> MediaProbe {
        MediaProbe::Ready(Arc::new(MediaInfo {
            video: Some(VideoStreamInfo {
                stream_index: 0,
                codec: "h264".to_owned(),
                width: 16,
                height: 16,
                frame_rate,
            }),
            audio_tracks: Vec::new(),
            subtitle_tracks: Vec::new(),
            duration_seconds: None,
            size_bytes: None,
            start_seconds: 0.0,
        }))
    }

    /// Registers a video and settles its timing from `probe`, without
    /// spawning ffprobe.
    fn state_with_probe(probe: &MediaProbe) -> VideoPlaybackState {
        let mut state = VideoPlaybackState::new(60.0);
        state.activate(VIDEO, TEST_PATH, 1);
        let active = state.active.get_mut(&VIDEO).expect("registered");
        active.resolve_timing(probe, 60.0);
        state
    }

    fn state_with_probed_video(native_fps: f32) -> VideoPlaybackState {
        state_with_probe(&probe_at(Some(native_fps)))
    }

    /// The frame lookup for a video whose timing is already settled.
    fn take(state: &mut VideoPlaybackState, position: PlaybackPosition) -> Option<Image> {
        state.take_frame_for_time(VIDEO, position, &MediaProbe::Pending, None)
    }

    /// Gives the video a stream fed by the returned sender instead of
    /// ffmpeg.
    fn attach_stream(
        state: &mut VideoPlaybackState,
        start_frame_index: u64,
    ) -> Sender<StreamItem<Vec<u8>>> {
        let (sender, records) = bounded(16);
        state.active.get_mut(&VIDEO).expect("registered").stream =
            Some(FrameStream::new(records, start_frame_index));
        sender
    }

    fn send_frames(sender: &Sender<StreamItem<Vec<u8>>>, count: usize) {
        for _ in 0..count {
            sender
                .send(StreamItem::Record(vec![0; 4]))
                .expect("stream listening");
        }
    }

    fn timing(state: &VideoPlaybackState) -> VideoTiming {
        state.active[&VIDEO].timing.expect("probed")
    }

    #[test]
    fn probed_rate_sets_playback_timing_capped_and_with_a_fallback() {
        let native = state_with_probed_video(24.0);
        assert_eq!(timing(&native).fps(), 24.0);

        let capped = state_with_probed_video(144.0);
        assert_eq!(timing(&capped).fps(), 60.0);

        let unknown = state_with_probe(&probe_at(None));
        assert_eq!(timing(&unknown).fps(), 60.0);

        let failed = state_with_probe(&MediaProbe::Failed);
        assert_eq!(timing(&failed).fps(), 60.0);
    }

    #[test]
    fn nothing_decodes_or_shows_before_the_probe_answers() {
        let mut state = VideoPlaybackState::new(60.0);
        state.activate(VIDEO, TEST_PATH, 1);
        assert!(take(&mut state, at(0.0, true)).is_none());
        assert_eq!(state.active[&VIDEO].timing, None);
        assert_eq!(state.stream_count(), 0);
    }

    #[test]
    fn a_texture_size_change_keeps_the_probed_rate() {
        let mut state = state_with_probed_video(24.0);
        state.activate(VIDEO, TEST_PATH, 2);
        assert_eq!(timing(&state).fps(), 24.0);
    }

    #[test]
    fn choosing_other_subtitles_decodes_the_shown_frame_again() {
        let mut state = state_with_probed_video(24.0);
        let sender = attach_stream(&mut state, 0);
        send_frames(&sender, 1);
        assert!(take(&mut state, at(0.0, true)).is_some());
        let active = state.active.get_mut(&VIDEO).expect("registered");

        active.choose_subtitles(None);
        assert!(active.stream.is_some() && active.applied_frame_index.is_some());

        let subtitles = SubtitleBurn {
            track_kind_index: 0,
            delay_seconds: 0.5,
            file_start_seconds: 0.0,
        };
        active.choose_subtitles(Some(subtitles));
        assert_eq!(active.subtitles, Some(subtitles));
        assert!(active.stream.is_none() && active.applied_frame_index.is_none());
    }

    #[test]
    fn playback_shows_the_latest_due_frame_and_holds_the_next() {
        let mut state = state_with_probed_video(24.0);
        let sender = attach_stream(&mut state, 0);
        send_frames(&sender, 5);
        // Mid-frame, clear of the rounding at frame boundaries.
        let time = 2.5 / timing(&state).fps();

        assert!(take(&mut state, at(time, true)).is_some());
        let active = &state.active[&VIDEO];
        assert_eq!(active.applied_frame_index, Some(2));
        assert_eq!(
            active
                .stream
                .as_ref()
                .expect("streaming")
                .earliest_frame_index(),
            3
        );
    }

    #[test]
    fn a_stream_restarts_only_when_it_cannot_decode_forward_to_the_target() {
        let mut state = state_with_probed_video(24.0);
        let video_timing = timing(&state);
        let sender = attach_stream(&mut state, 10);
        let active = state.active.get_mut(&VIDEO).expect("registered");
        // Not landed yet: left alone wherever the target moves.
        assert!(!active.stream_must_restart(video_timing, 0, false));
        assert!(!active.stream_must_restart(video_timing, 500, false));

        send_frames(&sender, 2);
        let mut due = None;
        let stream = active.stream.as_mut().expect("streaming");
        stream.read_to(10, &mut due);
        active.applied_frame_index = due.map(|(index, _)| index);

        let ahead = video_timing.frame_index(VIDEO_RESTART_AHEAD_SECONDS);
        assert!(!active.stream_must_restart(video_timing, 11, true));
        assert!(!active.stream_must_restart(video_timing, 11 + ahead - 1, true));
        assert!(active.stream_must_restart(video_timing, 11 + ahead, true));
        // Easing back a frame while playing keeps the shown frame; going back
        // further, or any step back while paused, restarts.
        assert!(!active.stream_must_restart(video_timing, 9, true));
        assert!(active.stream_must_restart(video_timing, 9, false));
        assert!(active.stream_must_restart(video_timing, 5, true));
    }

    #[test]
    fn a_paused_video_drops_its_stream_once_its_frame_shows() {
        let mut state = state_with_probed_video(24.0);
        let sender = attach_stream(&mut state, 0);
        send_frames(&sender, 3);
        let paused = at(0.0, false);

        assert!(take(&mut state, paused).is_some());
        assert_eq!(state.active[&VIDEO].applied_frame_index, Some(0));
        assert_eq!(state.stream_count(), 1);

        assert!(take(&mut state, paused).is_none());
        assert_eq!(state.stream_count(), 0);
    }

    #[test]
    fn the_stream_end_caps_the_target_at_the_last_frame() {
        let mut state = state_with_probed_video(24.0);
        let video_timing = timing(&state);
        let sender = attach_stream(&mut state, 0);
        send_frames(&sender, 3);
        sender
            .send(StreamItem::End(StreamEnd::Finished))
            .expect("stream listening");

        let past_end = at(video_timing.frame_seconds(10), true);
        assert!(take(&mut state, past_end).is_some());
        let active = &state.active[&VIDEO];
        assert_eq!(active.last_frame_index, Some(2));
        assert_eq!(active.applied_frame_index, Some(2));
        // Playing on past the end holds the last frame and drops the
        // finished stream without restarting it.
        assert!(take(&mut state, past_end).is_none());
        assert_eq!(state.stream_count(), 0);
        assert!(take(&mut state, past_end).is_none());
        assert_eq!(state.stream_count(), 0);
    }

    #[test]
    fn a_stream_started_past_the_end_probes_further_back_each_miss() {
        let mut state = state_with_probed_video(24.0);
        for (start, expected_last) in [(100, 99), (99, 97), (97, 93)] {
            let sender = attach_stream(&mut state, start);
            sender
                .send(StreamItem::End(StreamEnd::Finished))
                .expect("stream listening");
            let active = state.active.get_mut(&VIDEO).expect("registered");
            active.read_stream_to(VIDEO, start);
            assert_eq!(active.last_frame_index, Some(expected_last));
            assert!(!active.broken);
        }
    }

    #[test]
    fn a_stream_failing_mid_video_restarts_without_capping_the_video() {
        let mut state = state_with_probed_video(24.0);
        let sender = attach_stream(&mut state, 0);
        send_frames(&sender, 2);
        sender
            .send(StreamItem::End(StreamEnd::Failed("read error".to_owned())))
            .expect("stream listening");
        assert!(take(&mut state, at(0.5, true)).is_some());
        let active = &state.active[&VIDEO];
        assert_eq!(active.last_frame_index, None);
        assert!(!active.broken);
        assert_eq!(state.stream_count(), 0);
    }

    #[test]
    fn a_stream_that_fails_without_frames_marks_the_video_broken() {
        let mut state = state_with_probed_video(24.0);
        let sender = attach_stream(&mut state, 0);
        sender
            .send(StreamItem::End(StreamEnd::Failed("corrupt".to_owned())))
            .expect("stream listening");
        take(&mut state, at(0.0, true));
        assert!(state.active[&VIDEO].broken);

        // A broken video keeps what it shows without respawning anything.
        assert!(take(&mut state, at(1.0, true)).is_none());
        assert_eq!(state.stream_count(), 0);
    }
}
