use std::{
    collections::HashMap,
    ffi::OsString,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use bevy::prelude::*;
use crossbeam_channel::{Receiver, TryRecvError};
use rodio::{OutputStream, OutputStreamHandle, Sink, Source};

use crate::ffmpeg_pipe::{spawn_ffmpeg_stream, StreamEnd, StreamItem};

/// Audio is resampled by ffmpeg to one fixed layout so a sample count
/// converts straight to playback time.
const AUDIO_SAMPLE_RATE: u32 = 44_100;
const AUDIO_CHANNELS: u16 = 2;
/// Interleaved samples per record read off ffmpeg: about 46 ms of sound.
const AUDIO_BLOCK_SAMPLES: usize = 4096;
/// Blocks decoded ahead of the output, about 1.5 s: enough to ride out a
/// disk shared with many other streams.
const AUDIO_READ_AHEAD_BLOCKS: usize = 32;

/// How a video clock should move this frame, as its sound sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AudioClock {
    /// Sound is starting at the clock's time but has not reached the output
    /// yet; the clock waits so picture and sound start together.
    Prerolling,
    /// Sound is playing and has reached this time in the video.
    Playing(f32),
    /// No sound follows this clock (no track, no output device, or the
    /// track ended before the video); the clock runs on its own.
    Free,
}

/// One video clock as the audio follows it this frame.
#[derive(Clone, Copy, Debug)]
pub struct AudioClockSample {
    pub image_id: usize,
    pub time_seconds: f32,
    /// Sound should be coming out: the clock is advancing and the pause menu
    /// is closed.
    pub playing: bool,
    /// Bumped by the clock on every discontinuous jump (seek, loop wrap).
    pub seek_generation: u64,
    /// The picture shows the clock's frame since the clock last started.
    /// Sound that restarts or resumes is held until then, so picture and
    /// sound begin together.
    pub picture_ready: bool,
    /// The video's own gain (0 while muted), which the master volume scales.
    pub gain: f32,
    pub sound: SoundChoice,
}

/// What a video's sound should play, as far as its file is known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SoundChoice {
    /// The file's probe has not answered; no sound starts yet, and the
    /// clock waits for it.
    Pending,
    /// The file has no sound track.
    Silent,
    Track(SoundTrack),
}

/// How a video's sound track plays. A change restarts the sound, like a
/// seek does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoundTrack {
    /// The sound stream's index in the file; `None` leaves the choice to
    /// ffmpeg (the file could not be probed).
    pub stream_index: Option<usize>,
    /// Seconds the sound plays later than the picture.
    pub delay_seconds: f32,
    /// Playback rate; the sound keeps its pitch.
    pub speed: f32,
}

/// Plays the sound of every started video through its own rodio sink, each
/// following the per-video clock owned by `VideoControlsState`, and reports
/// back how far the sound got so the clock can follow the output device (see
/// [`AudioPlaybackState::clock`]). Sound streams from one ffmpeg process per
/// playing video, started at the clock's time: play resumes it, pause pauses
/// it, and a seek (or loop wrap) replaces it with one starting at the new
/// time. A started or resumed stream stays paused, decoding ahead, until the
/// video's picture is ready; the clock waits for the sound meanwhile.
#[derive(Resource)]
pub struct AudioPlaybackState {
    output: Option<OutputStreamHandle>,
    entries: HashMap<usize, AudioEntry>,
    master_volume: f32,
}

struct AudioEntry {
    path: PathBuf,
    /// Whether the chosen sound can play: the file has a track, and the
    /// chosen one is not `unplayable`. Settled by each sync.
    has_sound: bool,
    /// A track whose stream found no sound to play (none there, or a decode
    /// failure before any), so later seeks, delays and speeds do not retry
    /// it; choosing another track does.
    unplayable: Option<SoundTrack>,
    stream: Option<ActiveAudioStream>,
    was_playing: bool,
    /// The stream was restarted or resumed and is held paused until the
    /// picture is ready.
    awaiting_picture: bool,
    /// Seek generation of the clock the stream was started from. Left behind
    /// by a seek until the sound restarts at it.
    seek_generation: u64,
    /// The video's own gain (0 while muted), which the master volume scales.
    gain: f32,
}

struct ActiveAudioStream {
    sink: Sink,
    progress: Arc<AudioStreamProgress>,
    /// The file time the stream's first sound plays; before the file's
    /// start when a delay leads it with silence.
    start_seconds: f32,
    track: SoundTrack,
}

impl AudioEntry {
    fn apply_volume(&self, master_volume: f32) {
        if let Some(stream) = self.stream.as_ref() {
            stream.sink.set_volume(master_volume * self.gain);
        }
    }
}

impl ActiveAudioStream {
    /// Where the picture should be for the sound playing now: the file time
    /// of that sound, plus the delay the sound runs behind by.
    fn clock(&self) -> AudioClock {
        if self.progress.end().is_some() {
            return AudioClock::Free;
        }
        match self.progress.played_samples() {
            None => AudioClock::Prerolling,
            Some(samples) => {
                let played_seconds =
                    samples as f32 / (AUDIO_SAMPLE_RATE as f32 * AUDIO_CHANNELS as f32);
                AudioClock::Playing(
                    self.start_seconds
                        + played_seconds * self.track.speed
                        + self.track.delay_seconds,
                )
            }
        }
    }
}

impl AudioPlaybackState {
    pub fn new() -> Self {
        let output = match OutputStream::try_default() {
            Ok((stream, handle)) => {
                // The device stream must outlive every sink but is neither
                // Send nor Sync, so it is leaked for the app's lifetime and
                // only the handle is kept (the same pattern bevy_audio uses).
                std::mem::forget(stream);
                Some(handle)
            }
            Err(error) => {
                eprintln!("Audio output unavailable, videos will play without sound: {error}");
                None
            }
        };
        Self::with_output(output)
    }

    /// Playback state for tests, which must not open the audio device.
    #[cfg(test)]
    pub fn without_output() -> Self {
        Self::with_output(None)
    }

    fn with_output(output: Option<OutputStreamHandle>) -> Self {
        Self {
            output,
            entries: HashMap::new(),
            master_volume: 1.0,
        }
    }

    /// Applies the settings-menu volume to every current and future sink, on
    /// top of each video's own gain.
    pub fn set_master_volume(&mut self, volume: f32) {
        let volume = volume.clamp(0.0, 1.0);
        if (self.master_volume - volume).abs() < f32::EPSILON {
            return;
        }
        self.master_volume = volume;
        for entry in self.entries.values() {
            entry.apply_volume(volume);
        }
    }

    /// Registers a started video's sound; nothing decodes until it plays. A
    /// video already registered keeps its stream.
    pub fn activate(&mut self, image_id: usize, path: impl Into<PathBuf>) {
        if self.output.is_none() || self.entries.contains_key(&image_id) {
            return;
        }
        self.entries.insert(
            image_id,
            AudioEntry {
                path: path.into(),
                has_sound: true,
                unplayable: None,
                stream: None,
                was_playing: false,
                awaiting_picture: false,
                seek_generation: 0,
                gain: 1.0,
            },
        );
    }

    pub fn deactivate(&mut self, image_id: usize) {
        // Dropping the sink drops its source, which ends the ffmpeg stream.
        self.entries.remove(&image_id);
    }

    /// Where the sound of a playing video says its clock should be. Read
    /// before [`Self::sync`] catches up with this frame's clock changes, so
    /// a clock whose seek generation moved on is reported as prerolling: the
    /// sync restarts its sound at the new time this same frame.
    pub fn clock(&self, image_id: usize, seek_generation: u64) -> AudioClock {
        let Some(entry) = self.entries.get(&image_id) else {
            return AudioClock::Free;
        };
        if !entry.has_sound {
            return AudioClock::Free;
        }
        match entry.stream.as_ref() {
            Some(stream) if entry.seek_generation == seek_generation => stream.clock(),
            _ => AudioClock::Prerolling,
        }
    }

    /// Follows every started video's clock each frame.
    pub fn sync(&mut self, clocks: impl IntoIterator<Item = AudioClockSample>) {
        for clock in clocks {
            let Some(entry) = self.entries.get_mut(&clock.image_id) else {
                continue;
            };
            if (entry.gain - clock.gain).abs() > f32::EPSILON {
                entry.gain = clock.gain;
                entry.apply_volume(self.master_volume);
            }
            handle_stream_end(clock.image_id, entry);
            let track = match clock.sound {
                // Until the file is probed the sound is taken to exist, so
                // the clock waits for it rather than running ahead.
                SoundChoice::Pending => {
                    entry.has_sound = true;
                    None
                }
                SoundChoice::Silent => {
                    entry.has_sound = false;
                    entry.stream = None;
                    None
                }
                SoundChoice::Track(track) => {
                    entry.has_sound = entry
                        .unplayable
                        .is_none_or(|failed| failed.stream_index != track.stream_index);
                    entry.has_sound.then_some(track)
                }
            };
            if let Some(track) = track {
                match audio_transition(entry, clock, track) {
                    AudioTransition::Restart => {
                        restart_entry(
                            self.output.as_ref(),
                            clock.image_id,
                            entry,
                            clock.time_seconds,
                            track,
                            self.master_volume,
                        );
                        // Only a restart catches the sound up with a seek: a
                        // seek made while paused or scrubbing stays pending
                        // until playback resumes.
                        entry.seek_generation = clock.seek_generation;
                        entry.awaiting_picture = true;
                    }
                    AudioTransition::Resume => entry.awaiting_picture = true,
                    AudioTransition::Pause => {
                        if let Some(stream) = entry.stream.as_ref() {
                            stream.sink.pause();
                        }
                    }
                    AudioTransition::Keep => {}
                }
                if entry.awaiting_picture && clock.playing && clock.picture_ready {
                    if let Some(stream) = entry.stream.as_ref() {
                        stream.sink.play();
                    }
                    entry.awaiting_picture = false;
                }
            }
            entry.was_playing = clock.playing;
        }
    }
}

/// Acts on a stream that stopped: one that had played sound before failing
/// is dropped so the next sync restarts it where the clock is; one that never
/// played any (no such track, or an undecodable one) marks its track
/// unplayable.
fn handle_stream_end(image_id: usize, entry: &mut AudioEntry) {
    let Some(stream) = entry.stream.as_ref() else {
        return;
    };
    let Some(end) = stream.progress.end() else {
        return;
    };
    let played_sound = stream.progress.played_samples().is_some();
    let unplayable = match end {
        StreamEnd::Finished => return,
        StreamEnd::MissingStream => true,
        StreamEnd::Failed(error) => {
            eprintln!("Failed to decode audio for video {image_id}: {error}");
            !played_sound
        }
    };
    if unplayable {
        entry.unplayable = Some(stream.track);
        entry.has_sound = false;
    }
    entry.stream = None;
}

#[derive(Debug, PartialEq, Eq)]
enum AudioTransition {
    Restart,
    Resume,
    Pause,
    Keep,
}

/// What a sink must do to follow its clock: start a stream when playback
/// begins without one, or when the clock jumped or the track settings
/// changed since it started; resume or pause it when playback starts or
/// stops.
fn audio_transition(
    entry: &AudioEntry,
    clock: AudioClockSample,
    track: SoundTrack,
) -> AudioTransition {
    if !clock.playing {
        return if entry.was_playing {
            AudioTransition::Pause
        } else {
            AudioTransition::Keep
        };
    }
    let stream_current = entry
        .stream
        .as_ref()
        .is_some_and(|stream| stream.track == track);
    if !stream_current || entry.seek_generation != clock.seek_generation {
        return AudioTransition::Restart;
    }
    if entry.was_playing {
        AudioTransition::Keep
    } else {
        AudioTransition::Resume
    }
}

/// Replaces the entry's stream with one playing `track` for the picture at
/// `time_seconds`, paused until the picture is ready.
fn restart_entry(
    output: Option<&OutputStreamHandle>,
    image_id: usize,
    entry: &mut AudioEntry,
    time_seconds: f32,
    track: SoundTrack,
    master_volume: f32,
) {
    entry.stream = None;
    let Some(output) = output else {
        return;
    };
    let sink = match Sink::try_new(output) {
        Ok(sink) => sink,
        Err(error) => {
            eprintln!("Failed to open audio sink: {error}");
            return;
        }
    };
    // A delayed sound plays what the file had `delay` earlier; before the
    // file's start that is silence.
    let start_seconds = time_seconds - track.delay_seconds;
    let records = match spawn_ffmpeg_stream(
        format!("audio-{image_id}"),
        audio_stream_arguments(&entry.path, start_seconds, track),
        AUDIO_BLOCK_SAMPLES * 2,
        AUDIO_READ_AHEAD_BLOCKS,
        samples_from_le_bytes,
    ) {
        Ok(records) => records,
        Err(error) => {
            eprintln!("Failed to stream audio for video {image_id}: {error:#}");
            entry.unplayable = Some(track);
            entry.has_sound = false;
            return;
        }
    };
    let progress = Arc::new(AudioStreamProgress::default());
    sink.set_volume(master_volume * entry.gain);
    queue_held(&sink, StreamedAudioSource::new(records, progress.clone()));
    entry.stream = Some(ActiveAudioStream {
        sink,
        progress,
        start_seconds,
        track,
    });
}

/// Queues `source` on `sink` held until the sink plays. Paused before the
/// source is appended, the sink never pulls from it, so the source's clock
/// holds at its start while ffmpeg decodes ahead.
fn queue_held(sink: &Sink, source: StreamedAudioSource) {
    sink.pause();
    sink.append(source);
}

/// Decodes `track` from file time `start_seconds` on, as interleaved stereo
/// PCM at the fixed rate. The seek is an input seek, which ffmpeg makes
/// sample-accurate when it decodes. A start before the file's leads with
/// that much silence; a speed other than 1 is applied by `atempo`, which
/// keeps the pitch.
///
/// The seek point is the file's time, but a sound track can begin after the
/// file does. The seek trims every track to a point inside it, while a
/// start before the track's first sample would begin the sound there, early
/// by the gap; `aresample` pads the gap with silence so the first sample out
/// always plays the seek point, as the picture's first frame does. With
/// `async=1` it only fills and trims, stretching by at most a sample a
/// second, so it treats a gap or overlap in the track's timestamps later on
/// the same way: the sound stays on the file's timeline, which the picture
/// follows too.
fn audio_stream_arguments(
    path: &std::path::Path,
    start_seconds: f32,
    track: SoundTrack,
) -> Vec<OsString> {
    let mut arguments: Vec<OsString> = vec![
        "-ss".into(),
        format!("{:.6}", start_seconds.max(0.0)).into(),
        "-i".into(),
    ];
    arguments.push(path.as_os_str().to_owned());
    if let Some(stream_index) = track.stream_index {
        arguments.extend(["-map".into(), format!("0:{stream_index}").into()]);
    }
    let mut filters = vec!["aresample=async=1:first_pts=0".to_owned()];
    if start_seconds < 0.0 {
        filters.push(format!(
            "adelay=delays={:.3}:all=1",
            -start_seconds * 1000.0
        ));
    }
    if track.speed != 1.0 {
        filters.push(format!("atempo={}", track.speed));
    }
    arguments.extend(["-af".into(), filters.join(",").into()]);
    arguments.extend(
        [
            "-vn",
            "-sn",
            "-dn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            &AUDIO_SAMPLE_RATE.to_string(),
            "-ac",
            &AUDIO_CHANNELS.to_string(),
            "-f",
            "s16le",
            "pipe:1",
        ]
        .map(OsString::from),
    );
    arguments
}

fn samples_from_le_bytes(bytes: Vec<u8>) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

/// What a stream's source reports back from the audio thread.
#[derive(Default)]
struct AudioStreamProgress {
    started: AtomicBool,
    /// Samples handed to the output since the first decoded one arrived,
    /// including silence played in place of late sound.
    played_samples: AtomicU64,
    end: Mutex<Option<StreamEnd>>,
}

impl AudioStreamProgress {
    /// `None` until sound first reaches the output.
    fn played_samples(&self) -> Option<u64> {
        self.started
            .load(Ordering::Acquire)
            .then(|| self.played_samples.load(Ordering::Relaxed))
    }

    fn end(&self) -> Option<StreamEnd> {
        self.end.lock().ok().and_then(|end| end.clone())
    }

    fn finish(&self, end: StreamEnd) {
        if let Ok(mut slot) = self.end.lock() {
            *slot = Some(end);
        }
    }
}

/// A rodio source playing one ffmpeg stream. It runs on the output thread
/// and never blocks: until the first block arrives it plays silence without
/// counting it, so the clock holds at the start time; once started, a block
/// that is late is covered with silence that is counted, and as much of the
/// late sound is skipped when it arrives. Sound therefore stays on the
/// output device's clock, and the video clock following it never stalls.
struct StreamedAudioSource {
    records: Receiver<StreamItem<Vec<i16>>>,
    block: Vec<i16>,
    block_position: usize,
    progress: Arc<AudioStreamProgress>,
    started: bool,
    played_samples: u64,
    late_samples: u64,
}

impl StreamedAudioSource {
    fn new(records: Receiver<StreamItem<Vec<i16>>>, progress: Arc<AudioStreamProgress>) -> Self {
        Self {
            records,
            block: Vec::new(),
            block_position: 0,
            progress,
            started: false,
            played_samples: 0,
            late_samples: 0,
        }
    }

    fn count_played_sample(&mut self) {
        self.played_samples += 1;
        self.progress
            .played_samples
            .store(self.played_samples, Ordering::Relaxed);
    }

    fn take_block(&mut self, block: Vec<i16>) {
        let skipped = self.late_samples.min(block.len() as u64);
        self.late_samples -= skipped;
        self.block = block;
        self.block_position = skipped as usize;
        if !self.started {
            self.started = true;
            self.progress.started.store(true, Ordering::Release);
        }
    }
}

impl Iterator for StreamedAudioSource {
    type Item = i16;

    fn next(&mut self) -> Option<i16> {
        loop {
            if let Some(&sample) = self.block.get(self.block_position) {
                self.block_position += 1;
                self.count_played_sample();
                return Some(sample);
            }
            match self.records.try_recv() {
                Ok(StreamItem::Record(block)) => self.take_block(block),
                Ok(StreamItem::End(end)) => {
                    self.progress.finish(end);
                    return None;
                }
                Err(TryRecvError::Empty) => {
                    if self.started {
                        self.late_samples += 1;
                        self.count_played_sample();
                    }
                    return Some(0);
                }
                // The reader always sends an end before it hangs up, unless
                // it panicked; either way no more sound will come.
                Err(TryRecvError::Disconnected) => {
                    self.progress.finish(StreamEnd::Finished);
                    return None;
                }
            }
        }
    }
}

impl Source for StreamedAudioSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        AUDIO_CHANNELS
    }

    fn sample_rate(&self) -> u32 {
        AUDIO_SAMPLE_RATE
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::bounded;

    use super::*;

    fn entry(was_playing: bool, seek_generation: u64, streaming: bool) -> AudioEntry {
        let (sink, _output) = Sink::new_idle();
        AudioEntry {
            path: PathBuf::from("test.mp4"),
            has_sound: true,
            unplayable: None,
            stream: streaming.then(|| ActiveAudioStream {
                sink,
                progress: Arc::new(AudioStreamProgress::default()),
                start_seconds: 0.0,
                track: TRACK,
            }),
            was_playing,
            awaiting_picture: false,
            seek_generation,
            gain: 1.0,
        }
    }

    const TRACK: SoundTrack = SoundTrack {
        stream_index: Some(1),
        delay_seconds: 0.0,
        speed: 1.0,
    };

    fn sample(playing: bool, seek_generation: u64) -> AudioClockSample {
        AudioClockSample {
            image_id: 1,
            time_seconds: 2.0,
            playing,
            seek_generation,
            picture_ready: true,
            gain: 1.0,
            sound: SoundChoice::Track(TRACK),
        }
    }

    #[test]
    fn a_seek_while_playing_restarts_and_a_pause_keeps_the_stream() {
        let playing = entry(true, 3, true);
        assert_eq!(
            audio_transition(&playing, sample(true, 4), TRACK),
            AudioTransition::Restart
        );
        assert_eq!(
            audio_transition(&playing, sample(true, 3), TRACK),
            AudioTransition::Keep
        );
        assert_eq!(
            audio_transition(&playing, sample(false, 3), TRACK),
            AudioTransition::Pause
        );
        assert_eq!(
            audio_transition(&entry(false, 3, true), sample(false, 4), TRACK),
            AudioTransition::Keep
        );
        assert_eq!(
            audio_transition(&entry(false, 3, true), sample(true, 3), TRACK),
            AudioTransition::Resume
        );
        assert_eq!(
            audio_transition(&entry(false, 3, false), sample(true, 3), TRACK),
            AudioTransition::Restart
        );
    }

    #[test]
    fn a_track_change_while_playing_restarts_and_one_while_paused_waits() {
        let delayed = SoundTrack {
            delay_seconds: 0.2,
            ..TRACK
        };
        assert_eq!(
            audio_transition(&entry(true, 3, true), sample(true, 3), delayed),
            AudioTransition::Restart
        );
        assert_eq!(
            audio_transition(&entry(false, 3, true), sample(false, 3), delayed),
            AudioTransition::Keep
        );
    }

    #[test]
    fn delayed_and_faster_sound_reports_where_the_picture_belongs() {
        let (sink, _output) = Sink::new_idle();
        let stream = ActiveAudioStream {
            sink,
            progress: Arc::new(AudioStreamProgress::default()),
            // The picture was at 10 s with the sound 0.5 s behind.
            start_seconds: 9.5,
            track: SoundTrack {
                stream_index: Some(1),
                delay_seconds: 0.5,
                speed: 2.0,
            },
        };
        stream.progress.started.store(true, Ordering::Release);
        stream.progress.played_samples.store(
            AUDIO_SAMPLE_RATE as u64 * AUDIO_CHANNELS as u64,
            Ordering::Relaxed,
        );
        // One second of output at double speed is two seconds of file.
        assert_eq!(stream.clock(), AudioClock::Playing(12.0));
    }

    #[test]
    fn stream_arguments_pick_the_track_and_lead_a_delay_with_silence() {
        let arguments = |start_seconds: f32, track: SoundTrack| -> Vec<String> {
            audio_stream_arguments(std::path::Path::new("a.mkv"), start_seconds, track)
                .into_iter()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect()
        };
        let plain = arguments(3.0, TRACK);
        assert!(plain.windows(2).any(|pair| pair == ["-map", "0:1"]));
        // A track starting after the file is padded to the seek point.
        assert!(plain
            .windows(2)
            .any(|pair| pair == ["-af", "aresample=async=1:first_pts=0"]));

        let early = arguments(
            -0.25,
            SoundTrack {
                speed: 1.5,
                ..TRACK
            },
        );
        assert!(early.windows(2).any(|pair| pair == ["-ss", "0.000000"]));
        assert!(early.windows(2).any(|pair| pair
            == [
                "-af",
                "aresample=async=1:first_pts=0,adelay=delays=250.000:all=1,atempo=1.5"
            ]));
    }

    #[test]
    fn a_file_without_sound_frees_its_clock() {
        let mut state = AudioPlaybackState::without_output();
        state.entries.insert(1, entry(false, 0, false));
        state.sync([AudioClockSample {
            sound: SoundChoice::Silent,
            ..sample(true, 0)
        }]);
        assert_eq!(state.clock(1, 0), AudioClock::Free);
    }

    #[test]
    fn the_source_holds_the_clock_until_sound_arrives_then_skips_late_sound() {
        let (sender, records) = bounded(4);
        let progress = Arc::new(AudioStreamProgress::default());
        let mut source = StreamedAudioSource::new(records, progress.clone());

        // Nothing decoded yet: silence that does not count as played.
        assert_eq!(source.next(), Some(0));
        assert_eq!(progress.played_samples(), None);

        sender
            .send(StreamItem::Record(vec![1, 2]))
            .expect("source listening");
        assert_eq!(source.next(), Some(1));
        assert_eq!(source.next(), Some(2));
        assert_eq!(progress.played_samples(), Some(2));

        // The next block is late by two samples: they play as counted
        // silence, and the same two samples are skipped when it arrives.
        assert_eq!(source.next(), Some(0));
        assert_eq!(source.next(), Some(0));
        sender
            .send(StreamItem::Record(vec![3, 4, 5]))
            .expect("source listening");
        assert_eq!(source.next(), Some(5));
        assert_eq!(progress.played_samples(), Some(5));

        sender
            .send(StreamItem::End(StreamEnd::Finished))
            .expect("source listening");
        assert_eq!(source.next(), None);
        assert_eq!(progress.end(), Some(StreamEnd::Finished));
    }

    #[test]
    fn a_held_stream_is_not_read_until_its_sink_plays() {
        let (sender, records) = bounded(4);
        sender
            .send(StreamItem::Record(vec![1; 64]))
            .expect("source listening");
        let progress = Arc::new(AudioStreamProgress::default());
        let (sink, mut output) = Sink::new_idle();
        queue_held(&sink, StreamedAudioSource::new(records, progress.clone()));

        // The output device keeps pulling; a held sink hands it silence.
        for _ in 0..AUDIO_SAMPLE_RATE {
            assert_eq!(output.next(), Some(0.0));
        }
        assert_eq!(progress.played_samples(), None);

        sink.play();
        // Unpausing reaches the source at the sink's next periodic check.
        let played = (0..AUDIO_SAMPLE_RATE as usize).any(|_| {
            output.next();
            progress.played_samples().is_some()
        });
        assert!(played);
    }

    #[test]
    fn resumed_sound_waits_for_the_picture() {
        let mut state = AudioPlaybackState::without_output();
        let paused = entry(false, 3, true);
        paused.stream.as_ref().expect("streaming").sink.pause();
        state.entries.insert(1, paused);
        let sink_paused = |state: &AudioPlaybackState| {
            state.entries[&1]
                .stream
                .as_ref()
                .expect("streaming")
                .sink
                .is_paused()
        };

        state.sync([AudioClockSample {
            picture_ready: false,
            ..sample(true, 3)
        }]);
        assert!(sink_paused(&state));
        assert_eq!(state.clock(1, 3), AudioClock::Prerolling);

        state.sync([sample(true, 3)]);
        assert!(!sink_paused(&state));
        assert!(!state.entries[&1].awaiting_picture);
    }

    #[test]
    fn a_seek_made_while_paused_restarts_the_sound_on_resume() {
        let mut state = AudioPlaybackState::without_output();
        state.entries.insert(1, entry(false, 3, true));
        state.sync([sample(false, 4)]);
        assert_eq!(state.entries[&1].seek_generation, 3);
        assert!(state.entries[&1].stream.is_some());

        state.sync([sample(true, 4)]);
        let resumed = &state.entries[&1];
        assert_eq!(resumed.seek_generation, 4);
        // Without an output device the restart leaves no stream, which is
        // what shows it replaced the pre-seek one instead of resuming it.
        assert!(resumed.stream.is_none());
    }

    #[test]
    fn a_stream_failing_after_sound_is_retried_but_one_without_sound_is_not() {
        let mut failed_mid_track = entry(true, 0, true);
        let progress = &failed_mid_track
            .stream
            .as_ref()
            .expect("streaming")
            .progress;
        progress.started.store(true, Ordering::Release);
        progress.finish(StreamEnd::Failed("read error".to_owned()));
        handle_stream_end(1, &mut failed_mid_track);
        assert!(failed_mid_track.has_sound && failed_mid_track.stream.is_none());

        let mut silent = entry(true, 0, true);
        let progress = &silent.stream.as_ref().expect("streaming").progress;
        progress.finish(StreamEnd::MissingStream);
        handle_stream_end(1, &mut silent);
        assert!(!silent.has_sound && silent.stream.is_none());
        assert_eq!(silent.unplayable, Some(TRACK));
    }

    #[test]
    fn an_unplayable_track_is_not_retried_but_another_one_is() {
        let mut state = AudioPlaybackState::without_output();
        let mut unplayable = entry(true, 0, false);
        unplayable.unplayable = Some(TRACK);
        state.entries.insert(1, unplayable);
        state.sync([sample(true, 0)]);
        assert_eq!(state.clock(1, 0), AudioClock::Free);

        let other = SoundTrack {
            stream_index: Some(2),
            ..TRACK
        };
        state.sync([AudioClockSample {
            sound: SoundChoice::Track(other),
            ..sample(true, 0)
        }]);
        // Without an output device the restart leaves no stream, so the
        // clock waits for the sound it asked for.
        assert_eq!(state.clock(1, 0), AudioClock::Prerolling);
    }

    #[test]
    fn a_clock_whose_generation_moved_on_waits_for_the_restart() {
        let mut state = AudioPlaybackState::without_output();
        assert_eq!(state.clock(1, 0), AudioClock::Free);

        state.entries.insert(1, entry(true, 3, true));
        assert_eq!(state.clock(1, 3), AudioClock::Prerolling);
        assert_eq!(state.clock(1, 4), AudioClock::Prerolling);

        let stream = state.entries[&1].stream.as_ref().expect("streaming");
        stream.progress.started.store(true, Ordering::Release);
        stream.progress.played_samples.store(
            AUDIO_SAMPLE_RATE as u64 * AUDIO_CHANNELS as u64,
            Ordering::Relaxed,
        );
        assert_eq!(state.clock(1, 3), AudioClock::Playing(1.0));

        stream.progress.finish(StreamEnd::Finished);
        assert_eq!(state.clock(1, 3), AudioClock::Free);

        state.entries.get_mut(&1).expect("registered").has_sound = false;
        assert_eq!(state.clock(1, 3), AudioClock::Free);
    }
}
