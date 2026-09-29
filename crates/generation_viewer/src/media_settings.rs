//! Everything the viewer remembers about how to play a file — volume, mute,
//! the chosen sound and subtitle tracks, their delays, speed, looping and
//! where playback was left — plus the app-wide playback settings, kept in one
//! JSON file in the per-user state directory.
//!
//! Settings are keyed by file path, so they follow a file wherever a catalog
//! puts it. A file whose settings are all defaults has no entry. Changes are
//! saved in the background shortly after they happen, and once more on exit.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use bevy::prelude::*;
use generation_viewer_ui::{AudioSettings, PlaybackSettings};
use serde::{Deserialize, Serialize};

use crate::catalog_session::user_state_directory;
use crate::media_decode::SubtitleBurn;
use crate::media_probe::{MediaInfo, MediaTrack};

const SETTINGS_FILE_NAME: &str = "media_settings.json";

/// How long after the first unsaved change the settings are written, so a
/// slider drag or a playing video costs one write rather than one per frame.
const SAVE_DELAY: Duration = Duration::from_secs(1);

/// Loudness range the volume slider spans, from just above silent to full.
const VOLUME_RANGE_DB: f32 = 30.0;

/// Volume slider position of a video nobody has adjusted.
pub const DEFAULT_VOLUME: f32 = 1.0;

/// Which subtitles a file shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubtitleChoice {
    /// The track the file flags as default, if any.
    #[default]
    FileDefault,
    Off,
    /// The subtitle stream at this index in the file.
    Track(usize),
}

/// How one file plays. Missing fields load as their defaults, so settings
/// saved by an older viewer still load.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VideoFileSettings {
    /// Volume slider position; the sound follows it exponentially (see
    /// [`volume_gain`]).
    pub volume: f32,
    /// Separate from the level, so unmuting brings the level back.
    pub muted: bool,
    /// The sound stream at this index in the file; `None` follows the
    /// file's default.
    pub audio_track: Option<usize>,
    pub subtitles: SubtitleChoice,
    /// Milliseconds the sound plays later than the picture.
    pub audio_delay_ms: i32,
    /// Milliseconds the subtitles show later than the file times them.
    pub subtitle_delay_ms: i32,
    /// Playback rate; sound keeps its pitch.
    pub speed: f32,
    /// Plays again from the start at the end, instead of stopping there.
    pub looping: bool,
    /// Where playback was left, while the app remembers positions.
    pub resume_seconds: Option<f32>,
}

impl Default for VideoFileSettings {
    fn default() -> Self {
        Self {
            volume: DEFAULT_VOLUME,
            muted: false,
            audio_track: None,
            subtitles: SubtitleChoice::FileDefault,
            audio_delay_ms: 0,
            subtitle_delay_ms: 0,
            speed: 1.0,
            looping: true,
            resume_seconds: None,
        }
    }
}

impl VideoFileSettings {
    /// Muted by the user, or turned all the way down.
    pub fn silent(&self) -> bool {
        self.muted || self.volume <= 0.0
    }

    /// Gain the sound plays at, before the master volume applies.
    pub fn output_gain(&self) -> f32 {
        if self.silent() {
            0.0
        } else {
            volume_gain(self.volume)
        }
    }

    /// Mutes an audible video; otherwise unmutes it, bringing one that was
    /// turned all the way down back to the default volume.
    pub fn toggle_mute(&mut self) {
        if !self.silent() {
            self.muted = true;
            return;
        }
        self.muted = false;
        if self.volume <= 0.0 {
            self.volume = DEFAULT_VOLUME;
        }
    }

    /// Moving the volume slider is an explicit choice of level, so it also
    /// lifts a mute.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
        self.muted = false;
    }

    /// The sound track that plays: the chosen one while the file still has
    /// it, else the file's default.
    pub fn audio_track<'info>(&self, info: &'info MediaInfo) -> Option<&'info MediaTrack> {
        self.audio_track
            .and_then(|stream_index| info.audio_track(stream_index))
            .or_else(|| info.default_audio_track())
    }

    /// The subtitle track shown, if any and if it can be drawn.
    pub fn subtitle_track<'info>(&self, info: &'info MediaInfo) -> Option<&'info MediaTrack> {
        match self.subtitles {
            SubtitleChoice::FileDefault => info.default_subtitle_track(),
            SubtitleChoice::Off => None,
            SubtitleChoice::Track(stream_index) => info
                .subtitle_track(stream_index)
                .filter(|track| track.is_text_subtitle()),
        }
    }

    /// The subtitles a decode of this file draws.
    pub fn subtitle_burn(&self, info: &MediaInfo) -> Option<SubtitleBurn> {
        self.subtitle_track(info).map(|track| SubtitleBurn {
            track_kind_index: track.kind_index,
            delay_seconds: self.subtitle_delay_ms as f32 / 1000.0,
            file_start_seconds: info.start_seconds,
        })
    }

    pub fn audio_delay_seconds(&self) -> f32 {
        self.audio_delay_ms as f32 / 1000.0
    }

    /// Everything but the remembered position back to defaults.
    pub fn reset_playback(&mut self) {
        *self = Self {
            resume_seconds: self.resume_seconds,
            ..Self::default()
        };
    }
}

/// Gain for a volume slider position: exponential across `VOLUME_RANGE_DB`,
/// since loudness is heard logarithmically, and shifted to reach exact
/// silence at 0 while still reaching 1 at 1.
pub fn volume_gain(volume: f32) -> f32 {
    let volume = volume.clamp(0.0, 1.0);
    let full_range = 10f32.powf(VOLUME_RANGE_DB / 20.0);
    (full_range.powf(volume) - 1.0) / (full_range - 1.0)
}

/// What the settings file holds. Files are sorted by path, so a save
/// changes only the lines whose settings changed.
#[derive(Serialize, Deserialize)]
#[serde(default)]
struct SettingsFile {
    remember_playback_position: bool,
    master_volume: f32,
    videos: BTreeMap<String, VideoFileSettings>,
}

impl Default for SettingsFile {
    fn default() -> Self {
        Self {
            remember_playback_position: PlaybackSettings::default().remember_position,
            master_volume: AudioSettings::default().volume(),
            videos: BTreeMap::new(),
        }
    }
}

/// The app-wide settings saved alongside the per-file ones. They live in
/// the UI's [`PlaybackSettings`] and [`AudioSettings`], and are read from
/// there when saving.
#[derive(Clone, Copy)]
struct AppSettings {
    remember_playback_position: bool,
    master_volume: f32,
}

impl AppSettings {
    fn of(playback: &PlaybackSettings, audio: &AudioSettings) -> Self {
        Self {
            remember_playback_position: playback.remember_position,
            master_volume: audio.volume(),
        }
    }
}

/// Everything read back from the settings file at startup.
pub struct LoadedSettings {
    pub media: MediaSettings,
    pub playback: PlaybackSettings,
    pub audio: AudioSettings,
}

#[derive(Resource)]
pub struct MediaSettings {
    videos: HashMap<Arc<str>, VideoFileSettings>,
    /// Where the settings are saved; `None` keeps them in memory only.
    file: Option<PathBuf>,
    /// When the first change since the last save happened.
    unsaved_since: Option<Instant>,
    writer: SettingsWriter,
}

impl MediaSettings {
    /// The saved settings, or defaults when there are none. A file that
    /// cannot be read is moved aside rather than overwritten by the next
    /// save, so nothing in it is lost for good.
    pub fn load() -> LoadedSettings {
        let file = user_state_directory().map(|directory| directory.join(SETTINGS_FILE_NAME));
        let (saved, file) = match file {
            Some(file) => match read_settings(&file) {
                Ok(saved) => (saved.unwrap_or_default(), Some(file)),
                // Saving would overwrite whatever could not be read.
                Err(()) => (SettingsFile::default(), None),
            },
            None => (SettingsFile::default(), None),
        };
        LoadedSettings {
            playback: PlaybackSettings {
                remember_position: saved.remember_playback_position,
            },
            audio: AudioSettings::with_volume(saved.master_volume),
            media: Self::from_file(saved, file),
        }
    }

    /// Settings kept in memory only, for runs (benchmarks, the performance
    /// harness, tests) that must neither follow nor change the user's.
    pub fn in_memory() -> Self {
        Self::from_file(SettingsFile::default(), None)
    }

    fn from_file(saved: SettingsFile, file: Option<PathBuf>) -> Self {
        Self {
            videos: saved
                .videos
                .into_iter()
                .map(|(path, settings)| (Arc::from(path), settings))
                .collect(),
            file,
            unsaved_since: None,
            writer: SettingsWriter::default(),
        }
    }

    /// How the file at `path` plays.
    pub fn video(&self, path: &str) -> VideoFileSettings {
        self.videos.get(path).copied().unwrap_or_default()
    }

    /// Changes how the file at `path` plays.
    pub fn update(&mut self, path: &Arc<str>, edit: impl FnOnce(&mut VideoFileSettings)) {
        let before = self.video(path);
        let mut after = before;
        edit(&mut after);
        if after == before {
            return;
        }
        if after == VideoFileSettings::default() {
            self.videos.remove(path);
        } else {
            self.videos.insert(path.clone(), after);
        }
        self.mark_changed();
    }

    fn mark_changed(&mut self) {
        self.unsaved_since.get_or_insert_with(Instant::now);
    }

    fn snapshot(&self, app: AppSettings) -> SettingsFile {
        SettingsFile {
            remember_playback_position: app.remember_playback_position,
            master_volume: app.master_volume,
            videos: self
                .videos
                .iter()
                .map(|(path, settings)| (path.to_string(), *settings))
                .collect(),
        }
    }

    /// Writes the settings on a background thread once they have been
    /// unsaved for `SAVE_DELAY`.
    fn save_when_due(&mut self, app: AppSettings) {
        if self.writer.take_failure() {
            self.mark_changed();
        }
        if self
            .unsaved_since
            .is_some_and(|since| since.elapsed() >= SAVE_DELAY)
        {
            self.save(app, false);
        }
    }

    /// Writes any unsaved settings, waiting for the write when `blocking`.
    /// A failed earlier write leaves them unsaved.
    fn save(&mut self, app: AppSettings, blocking: bool) {
        if self.writer.take_failure() {
            self.mark_changed();
        }
        if self.unsaved_since.take().is_none() {
            return;
        }
        let Some(file) = self.file.clone() else {
            return;
        };
        let contents = match serde_json::to_string_pretty(&self.snapshot(app)) {
            Ok(contents) => contents,
            Err(error) => {
                eprintln!("Failed to encode media settings: {error}");
                return;
            }
        };
        self.writer.write(file, contents, blocking);
    }
}

/// The settings in `file`: `Ok(None)` when there are none yet, and `Err`
/// when some are there but cannot be read and are still in the way. A file
/// that reads but does not parse is moved aside, so saving does not lose it.
fn read_settings(file: &Path) -> Result<Option<SettingsFile>, ()> {
    let contents = match fs::read_to_string(file) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            eprintln!(
                "Cannot read media settings ({error}); leaving {} alone this session",
                file.display()
            );
            return Err(());
        }
    };
    match serde_json::from_str(&contents) {
        Ok(saved) => Ok(Some(saved)),
        Err(error) => {
            let aside = file.with_extension("json.unreadable");
            eprintln!(
                "Media settings are unreadable ({error}); moving them to {}",
                aside.display()
            );
            match fs::rename(file, &aside) {
                Ok(()) => Ok(None),
                Err(error) => {
                    eprintln!("Failed to move the unreadable media settings: {error}");
                    Err(())
                }
            }
        }
    }
}

/// Orders the settings writes: each takes a sequence number, and a write
/// that finds a later one already on disk is dropped, so a slow background
/// write can never replace newer settings (such as the ones written on exit).
#[derive(Clone, Default)]
struct SettingsWriter {
    next_sequence: u64,
    written_sequence: Arc<Mutex<u64>>,
    /// Set by a write that failed, so the settings are saved again.
    failed: Arc<AtomicBool>,
}

impl SettingsWriter {
    fn write(&mut self, file: PathBuf, contents: String, blocking: bool) {
        self.next_sequence += 1;
        let sequence = self.next_sequence;
        let written_sequence = self.written_sequence.clone();
        let failed = self.failed.clone();
        let write = move || {
            let Ok(mut written) = written_sequence.lock() else {
                return;
            };
            if *written > sequence {
                return;
            }
            match write_replacing(&file, &contents) {
                Ok(()) => *written = sequence,
                Err(error) => {
                    eprintln!("Failed to save media settings: {error}");
                    failed.store(true, Ordering::Relaxed);
                }
            }
        };
        if blocking {
            write();
        } else {
            thread::spawn(write);
        }
    }

    fn take_failure(&self) -> bool {
        self.failed.swap(false, Ordering::Relaxed)
    }
}

/// Writes beside the file, then moves the result over it, so a crash mid
/// write never leaves a truncated settings file.
fn write_replacing(file: &Path, contents: &str) -> std::io::Result<()> {
    if let Some(directory) = file.parent() {
        fs::create_dir_all(directory)?;
    }
    let staging = file.with_extension("json.saving");
    fs::write(&staging, contents)?;
    fs::rename(&staging, file)
}

/// Saves the settings shortly after they change, including the app-wide
/// ones the pause menu edits.
pub fn save_media_settings(
    playback_settings: Res<PlaybackSettings>,
    audio_settings: Res<AudioSettings>,
    mut media_settings: ResMut<MediaSettings>,
) {
    let edited = |changed: bool, added: bool| changed && !added;
    if edited(playback_settings.is_changed(), playback_settings.is_added())
        || edited(audio_settings.is_changed(), audio_settings.is_added())
    {
        media_settings.mark_changed();
    }
    media_settings.save_when_due(AppSettings::of(&playback_settings, &audio_settings));
}

/// Writes whatever is unsaved before the app closes. Runs after whatever
/// records last-moment state (see `remember_positions_on_exit`).
pub fn save_media_settings_on_exit(
    mut exit: EventReader<AppExit>,
    playback_settings: Res<PlaybackSettings>,
    audio_settings: Res<AudioSettings>,
    mut media_settings: ResMut<MediaSettings>,
) {
    if exit.read().count() > 0 {
        media_settings.save(AppSettings::of(&playback_settings, &audio_settings), true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_back_at_defaults_leaves_no_entry() {
        let mut settings = MediaSettings::in_memory();
        let path: Arc<str> = "clip.mp4".into();
        settings.update(&path, |video| video.speed = 1.5);
        assert_eq!(settings.video(&path).speed, 1.5);
        assert!(settings.unsaved_since.is_some());

        settings.update(&path, |video| video.speed = 1.0);
        assert!(settings.videos.is_empty());
        assert_eq!(settings.video("other.mp4"), VideoFileSettings::default());
    }

    #[test]
    fn muting_keeps_the_level_and_moving_the_slider_unmutes() {
        let mut video = VideoFileSettings::default();
        video.set_volume(0.6);
        let audible_gain = video.output_gain();
        assert!(audible_gain > 0.0);

        video.toggle_mute();
        assert!(video.silent());
        assert_eq!(video.output_gain(), 0.0);
        video.toggle_mute();
        assert_eq!(video.output_gain(), audible_gain);

        video.toggle_mute();
        video.set_volume(0.3);
        assert!(!video.silent());

        video.set_volume(0.0);
        assert!(video.silent());
        video.toggle_mute();
        assert_eq!(video.volume, DEFAULT_VOLUME);
    }

    #[test]
    fn volume_gain_is_exponential_from_silence_to_full() {
        assert_eq!(volume_gain(0.0), 0.0);
        assert!((volume_gain(1.0) - 1.0).abs() < 1e-6);
        // Half way up the slider is about half the loudness range down; the
        // shift that reaches silence at 0 takes it a little lower.
        let half_db = 20.0 * volume_gain(0.5).log10();
        let expected_db = -VOLUME_RANGE_DB * 0.5;
        assert!(half_db < expected_db && half_db > expected_db - 2.0);
        let steps: Vec<f32> = (0..=10)
            .map(|step| volume_gain(step as f32 / 10.0))
            .collect();
        assert!(steps.windows(2).all(|pair| pair[1] > pair[0]));
    }

    #[test]
    fn settings_round_trip_through_their_file_format() {
        let mut settings = MediaSettings::in_memory();
        let path: Arc<str> = r"E:\Anime\episode.mkv".into();
        settings.update(&path, |video| {
            video.subtitles = SubtitleChoice::Track(4);
            video.audio_delay_ms = -150;
            video.resume_seconds = Some(312.5);
        });
        let app = AppSettings {
            remember_playback_position: false,
            master_volume: 0.4,
        };
        let encoded = serde_json::to_string(&settings.snapshot(app)).expect("encodes");
        let decoded: SettingsFile = serde_json::from_str(&encoded).expect("decodes");
        assert!(!decoded.remember_playback_position);
        assert_eq!(decoded.master_volume, 0.4);
        let reloaded = MediaSettings::from_file(decoded, None);
        assert_eq!(reloaded.video(&path), settings.video(&path));

        // A file from an older viewer, missing fields, loads with defaults.
        let old: SettingsFile =
            serde_json::from_str(r#"{"videos": {"a.mp4": {"volume": 0.5}}}"#).expect("decodes");
        assert!(old.remember_playback_position);
        let old = MediaSettings::from_file(old, None);
        assert_eq!(old.video("a.mp4").volume, 0.5);
        assert!(old.video("a.mp4").looping);
    }

    #[test]
    fn unsaved_changes_are_written_and_read_back() {
        let directory =
            std::env::temp_dir().join(format!("media_settings_save_{}", std::process::id()));
        let file = directory.join(SETTINGS_FILE_NAME);
        let mut settings = MediaSettings::from_file(SettingsFile::default(), Some(file.clone()));
        let path: Arc<str> = "clip.mp4".into();
        settings.update(&path, |video| video.looping = false);
        let app = AppSettings::of(&PlaybackSettings::default(), &AudioSettings::default());
        settings.save(app, true);
        assert!(settings.unsaved_since.is_none());

        let saved: SettingsFile =
            serde_json::from_str(&fs::read_to_string(&file).expect("written")).expect("decodes");
        let reloaded = MediaSettings::from_file(saved, None);
        assert!(!reloaded.video(&path).looping);
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_stale_background_write_never_replaces_a_newer_one() {
        let directory =
            std::env::temp_dir().join(format!("media_settings_test_{}", std::process::id()));
        let file = directory.join(SETTINGS_FILE_NAME);
        let mut writer = SettingsWriter::default();
        writer.write(file.clone(), "first".to_owned(), true);
        writer.write(file.clone(), "second".to_owned(), true);
        // A write numbered before the one on disk arrives late.
        let mut late = SettingsWriter {
            next_sequence: 0,
            ..writer.clone()
        };
        late.write(file.clone(), "late".to_owned(), true);
        assert_eq!(fs::read_to_string(&file).expect("written"), "second");
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_failed_write_leaves_the_settings_unsaved() {
        let directory =
            std::env::temp_dir().join(format!("media_settings_blocked_{}", std::process::id()));
        // A directory where the file should be makes every write fail.
        let file = directory.join(SETTINGS_FILE_NAME);
        fs::create_dir_all(file.join("occupied")).expect("blocking directory");
        let mut settings = MediaSettings::from_file(SettingsFile::default(), Some(file));
        let path: Arc<str> = "clip.mp4".into();
        settings.update(&path, |video| video.speed = 2.0);
        let app = AppSettings::of(&PlaybackSettings::default(), &AudioSettings::default());
        settings.save(app, true);
        assert!(settings.unsaved_since.is_none());
        // The next save sees the failure and tries again.
        settings.save(app, true);
        assert!(settings.writer.take_failure());
        let _ = fs::remove_dir_all(&directory);
    }
}
