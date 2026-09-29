//! The right-click menu of a billboard.
//!
//! A right press released before it turns into a look-drag is a right-click
//! ([`WorldRightClick`]); on a video (its picture or its control strip) it
//! opens this menu, which offers what the control strip does not: looping,
//! speed, sound and subtitle tracks and their delays, the file's details,
//! and a few file actions. Everything it changes is a per-file setting in
//! [`MediaSettings`], so it applies whether or not the video is playing.
//!
//! The commands are [`BillboardMenuCommand`]s rather than video-only ones so
//! images can get a menu of their own (sharing the file actions) later.

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    thread,
};

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use generation_viewer_ui::{
    ContextMenu, ContextMenuItem, ContextMenuModel, ContextMenuOption, ContextMenuSection,
    PauseMenuState, RenderResolutionSettings,
};

use crate::image_loading::GenerationBillboard;
use crate::manual_spacing::{cursor_world_ray, nearest_billboard_hit};
use crate::media_decode::{still_frame_arguments, VideoSource};
use crate::media_probe::{MediaInfo, MediaProbe, MediaProbes};
use crate::media_settings::{MediaSettings, SubtitleChoice, VideoFileSettings};
use crate::video_controls::{format_seconds_mmss, VideoControlsState};
use crate::video_stream::VideoPlaybackState;
use crate::video_strip::{nearest_strip_hit, VideoStripHitQuery};
use crate::{ExplorerScene, FlyCamera};

const VIDEO_SPEEDS: [f32; 7] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
const AUDIO_DELAY_STEP_MS: i32 = 50;
const SUBTITLE_DELAY_STEP_MS: i32 = 100;
/// Delays past this are a wrong file, not a sync fix.
const MAX_DELAY_MS: i32 = 10_000;
const SCREENSHOT_DIRECTORY_NAME: &str = "Mixed Media Spatial Renderer";

/// A right press released over the world before it became a look-drag.
#[derive(Event, Clone, Copy, Debug)]
pub(crate) struct WorldRightClick;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum BillboardMenuCommand {
    SetLooping(bool),
    SetSpeed(f32),
    /// The sound stream at this index in the file.
    SetAudioTrack(usize),
    ShiftAudioDelay(i32),
    ResetAudioDelay,
    SetSubtitles(SubtitleChoice),
    ShiftSubtitleDelay(i32),
    ResetSubtitleDelay,
    ResetVideoSettings,
    SaveFrame,
    RevealInFileBrowser,
    OpenInDefaultApp,
    CopyPath,
}

/// The billboard the menu was last opened for; the commands it hands over
/// act on it.
#[derive(Resource, Default)]
pub(crate) struct BillboardMenuTarget(Option<MenuTarget>);

/// The system clipboard, kept open for the app's lifetime: on X11 and
/// Wayland, copied text disappears when the handle that copied it closes.
/// Not `Send` on every platform, so a main-thread resource.
#[derive(Default)]
pub(crate) struct ClipboardHandle(Option<arboard::Clipboard>);

/// The menu's systems, in the order they must run within the frame: the
/// commands picked this frame act on the menu's target before a right-click
/// may retarget it, and before the refresh redraws the menu with their
/// effect.
pub(crate) fn billboard_menu_systems() -> impl IntoSystemConfigs<()> {
    (
        apply_billboard_menu_commands,
        open_billboard_menu,
        refresh_billboard_menu,
    )
        .chain()
}

#[derive(Clone)]
struct MenuTarget {
    image_id: usize,
    path: Arc<str>,
}

impl BillboardMenuTarget {
    /// A target as a right-click on the video would set it.
    #[cfg(test)]
    pub(crate) fn video(image_id: usize, path: &str) -> Self {
        Self(Some(MenuTarget {
            image_id,
            path: path.into(),
        }))
    }
}

/// Opens the menu of the video under a right-click: its control strip, or
/// failing that the nearest picture under the pointer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn open_billboard_menu(
    mut right_clicks: EventReader<WorldRightClick>,
    render_resolution: Res<RenderResolutionSettings>,
    scene: Res<ExplorerScene>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<FlyCamera>>,
    strip_parts: VideoStripHitQuery,
    billboards: Query<(&GenerationBillboard, &GlobalTransform, &Visibility)>,
    mut probes: ResMut<MediaProbes>,
    media_settings: Res<MediaSettings>,
    mut target: ResMut<BillboardMenuTarget>,
    mut menu: ResMut<ContextMenu<BillboardMenuCommand>>,
) {
    if right_clicks.read().count() == 0 {
        return;
    }
    let Ok(window) = window_query.get_single() else {
        return;
    };
    let (Some(cursor), Ok((camera, camera_global))) =
        (window.cursor_position(), camera_query.get_single())
    else {
        return;
    };
    let Some((ray_origin, ray_direction)) =
        cursor_world_ray(window, camera, camera_global, &render_resolution)
    else {
        return;
    };
    let billboard_hit = nearest_billboard_hit(
        ray_origin,
        ray_direction,
        scene.billboard_world_size,
        &billboards,
    );
    let image_id = nearest_strip_hit(
        ray_origin,
        ray_direction,
        &strip_parts,
        billboard_hit.map(|hit| hit.distance),
    )
    .map(|hit| hit.image_id)
    .or(billboard_hit.map(|hit| hit.image_id));
    let Some(billboard) = image_id.and_then(|image_id| {
        billboards
            .iter()
            .map(|(billboard, _, _)| billboard)
            .find(|billboard| billboard.image_id == image_id && billboard.is_video)
    }) else {
        return;
    };
    let path = billboard.path.clone();
    probes.request(Path::new(&*path));
    let model = video_menu_model(
        &path,
        probes.get(Path::new(&*path)),
        &media_settings.video(&path),
    );
    target.0 = Some(MenuTarget {
        image_id: billboard.image_id,
        path,
    });
    menu.open(cursor, model);
}

/// Keeps the open menu showing its video's current settings and tracks, and
/// closes it once the video unloads or the pause menu opens.
pub(crate) fn refresh_billboard_menu(
    pause_menu: Res<PauseMenuState>,
    probes: Res<MediaProbes>,
    media_settings: Res<MediaSettings>,
    billboards: Query<&GenerationBillboard>,
    target: Res<BillboardMenuTarget>,
    mut menu: ResMut<ContextMenu<BillboardMenuCommand>>,
) {
    if !menu.is_open() {
        return;
    }
    let Some(current) = target.0.as_ref() else {
        menu.close();
        return;
    };
    let loaded = billboards
        .iter()
        .any(|billboard| billboard.image_id == current.image_id && billboard.path == current.path);
    if !loaded || pause_menu.paused {
        menu.close();
        return;
    }
    menu.update(video_menu_model(
        &current.path,
        probes.get(Path::new(&*current.path)),
        &media_settings.video(&current.path),
    ));
}

pub(crate) fn apply_billboard_menu_commands(
    controls: Res<VideoControlsState>,
    video_playback: Res<VideoPlaybackState>,
    probes: Res<MediaProbes>,
    mut media_settings: ResMut<MediaSettings>,
    mut clipboard: NonSendMut<ClipboardHandle>,
    target: Res<BillboardMenuTarget>,
    mut menu: ResMut<ContextMenu<BillboardMenuCommand>>,
) {
    let commands = menu.take_activated();
    let Some(target) = target.0.as_ref() else {
        return;
    };
    for command in commands {
        let path = &target.path;
        let shift_delay =
            |delay_ms: i32, step_ms: i32| (delay_ms + step_ms).clamp(-MAX_DELAY_MS, MAX_DELAY_MS);
        match command {
            BillboardMenuCommand::SetLooping(looping) => {
                media_settings.update(path, |video| video.looping = looping);
            }
            BillboardMenuCommand::SetSpeed(speed) => {
                media_settings.update(path, |video| video.speed = speed);
            }
            BillboardMenuCommand::SetAudioTrack(stream_index) => {
                media_settings.update(path, |video| video.audio_track = Some(stream_index));
            }
            BillboardMenuCommand::ShiftAudioDelay(step_ms) => {
                media_settings.update(path, |video| {
                    video.audio_delay_ms = shift_delay(video.audio_delay_ms, step_ms);
                })
            }
            BillboardMenuCommand::ResetAudioDelay => {
                media_settings.update(path, |video| video.audio_delay_ms = 0);
            }
            BillboardMenuCommand::SetSubtitles(choice) => {
                media_settings.update(path, |video| video.subtitles = choice);
            }
            BillboardMenuCommand::ShiftSubtitleDelay(step_ms) => {
                media_settings.update(path, |video| {
                    video.subtitle_delay_ms = shift_delay(video.subtitle_delay_ms, step_ms);
                });
            }
            BillboardMenuCommand::ResetSubtitleDelay => {
                media_settings.update(path, |video| video.subtitle_delay_ms = 0);
            }
            BillboardMenuCommand::ResetVideoSettings => {
                media_settings.update(path, VideoFileSettings::reset_playback);
            }
            BillboardMenuCommand::SaveFrame => {
                // The frame on screen, not the clock's exact time, which
                // is usually part way to the next frame.
                let time_seconds = video_playback
                    .shown_frame_seek_seconds(target.image_id)
                    .or_else(|| {
                        controls
                            .clock(target.image_id)
                            .map(|clock| clock.time_seconds())
                    })
                    .unwrap_or(0.0);
                let info = probes.info(Path::new(&**path));
                let source = VideoSource {
                    path: PathBuf::from(&**path),
                    video_stream_index: info
                        .and_then(|info| info.video.as_ref())
                        .map(|video| video.stream_index),
                    subtitles: info.and_then(|info| media_settings.video(path).subtitle_burn(info)),
                };
                save_frame(source, time_seconds);
            }
            BillboardMenuCommand::RevealInFileBrowser => reveal_in_file_browser(path),
            BillboardMenuCommand::OpenInDefaultApp => open_in_default_app(path),
            BillboardMenuCommand::CopyPath => clipboard.copy(path),
        }
    }
}

/// The menu of the video in file `path`.
fn video_menu_model(
    path: &str,
    probe: Option<&MediaProbe>,
    settings: &VideoFileSettings,
) -> ContextMenuModel<BillboardMenuCommand> {
    let info = probe.and_then(MediaProbe::info);
    let reading = matches!(probe, None | Some(MediaProbe::Pending));
    let mut sections = vec![
        ContextMenuSection {
            heading: Some("Playback".to_owned()),
            items: vec![
                ContextMenuItem::Toggle {
                    label: "Loop".to_owned(),
                    on: settings.looping,
                    command: BillboardMenuCommand::SetLooping(!settings.looping),
                },
                ContextMenuItem::Choice {
                    label: "Speed".to_owned(),
                    value: speed_label(settings.speed),
                    options: VIDEO_SPEEDS
                        .iter()
                        .map(|&speed| ContextMenuOption {
                            label: speed_label(speed),
                            selected: speed == settings.speed,
                            command: Some(BillboardMenuCommand::SetSpeed(speed)),
                        })
                        .collect(),
                },
            ],
        },
        ContextMenuSection {
            heading: Some("Audio".to_owned()),
            items: vec![
                track_item(reading, info.map(|info| audio_track_item(info, settings))),
                ContextMenuItem::Stepper {
                    label: "Delay".to_owned(),
                    value: delay_label(settings.audio_delay_ms),
                    decrease: BillboardMenuCommand::ShiftAudioDelay(-AUDIO_DELAY_STEP_MS),
                    increase: BillboardMenuCommand::ShiftAudioDelay(AUDIO_DELAY_STEP_MS),
                    reset: (settings.audio_delay_ms != 0)
                        .then_some(BillboardMenuCommand::ResetAudioDelay),
                },
            ],
        },
        ContextMenuSection {
            heading: Some("Subtitles".to_owned()),
            items: vec![
                track_item(
                    reading,
                    info.map(|info| subtitle_track_item(info, settings)),
                ),
                ContextMenuItem::Stepper {
                    label: "Delay".to_owned(),
                    value: delay_label(settings.subtitle_delay_ms),
                    decrease: BillboardMenuCommand::ShiftSubtitleDelay(-SUBTITLE_DELAY_STEP_MS),
                    increase: BillboardMenuCommand::ShiftSubtitleDelay(SUBTITLE_DELAY_STEP_MS),
                    reset: (settings.subtitle_delay_ms != 0)
                        .then_some(BillboardMenuCommand::ResetSubtitleDelay),
                },
            ],
        },
    ];
    let mut file_items = info.map(file_details).unwrap_or_default();
    file_items.extend(
        [
            ("Save frame as PNG", BillboardMenuCommand::SaveFrame),
            (
                "Show in file browser",
                BillboardMenuCommand::RevealInFileBrowser,
            ),
            (
                "Open in default app",
                BillboardMenuCommand::OpenInDefaultApp,
            ),
            ("Copy path", BillboardMenuCommand::CopyPath),
        ]
        .map(|(label, command)| ContextMenuItem::Action {
            label: label.to_owned(),
            command,
        }),
    );
    sections.push(ContextMenuSection {
        heading: Some("File".to_owned()),
        items: file_items,
    });
    sections.push(ContextMenuSection {
        heading: None,
        items: vec![ContextMenuItem::Action {
            label: "Reset video settings".to_owned(),
            command: BillboardMenuCommand::ResetVideoSettings,
        }],
    });
    ContextMenuModel {
        title: file_name(path),
        sections,
    }
}

/// A track choice once the file is probed; until then, or when it could not
/// be read, a note in its place.
fn track_item(
    reading: bool,
    item: Option<ContextMenuItem<BillboardMenuCommand>>,
) -> ContextMenuItem<BillboardMenuCommand> {
    item.unwrap_or_else(|| ContextMenuItem::Info {
        label: "Track".to_owned(),
        value: if reading {
            "Reading file...".to_owned()
        } else {
            "Unknown".to_owned()
        },
    })
}

fn audio_track_item(
    info: &MediaInfo,
    settings: &VideoFileSettings,
) -> ContextMenuItem<BillboardMenuCommand> {
    let Some(playing) = settings.audio_track(info) else {
        return ContextMenuItem::Info {
            label: "Track".to_owned(),
            value: "None in this file".to_owned(),
        };
    };
    ContextMenuItem::Choice {
        label: "Track".to_owned(),
        value: playing.label(),
        options: info
            .audio_tracks
            .iter()
            .map(|track| ContextMenuOption {
                label: track.label(),
                selected: track.stream_index == playing.stream_index,
                command: Some(BillboardMenuCommand::SetAudioTrack(track.stream_index)),
            })
            .collect(),
    }
}

fn subtitle_track_item(
    info: &MediaInfo,
    settings: &VideoFileSettings,
) -> ContextMenuItem<BillboardMenuCommand> {
    let shown_track = settings.subtitle_track(info);
    let shown = shown_track.map(|track| track.stream_index);
    let off = ContextMenuOption {
        label: "Off".to_owned(),
        selected: shown.is_none(),
        command: Some(BillboardMenuCommand::SetSubtitles(SubtitleChoice::Off)),
    };
    let tracks = info.subtitle_tracks.iter().map(|track| {
        if track.is_text_subtitle() {
            ContextMenuOption {
                label: track.label(),
                selected: shown == Some(track.stream_index),
                command: Some(BillboardMenuCommand::SetSubtitles(SubtitleChoice::Track(
                    track.stream_index,
                ))),
            }
        } else {
            ContextMenuOption {
                label: format!("{} (image subtitles, unsupported)", track.label()),
                selected: false,
                command: None,
            }
        }
    });
    ContextMenuItem::Choice {
        label: "Track".to_owned(),
        value: shown_track.map_or_else(|| "Off".to_owned(), |track| track.label()),
        options: std::iter::once(off).chain(tracks).collect(),
    }
}

fn file_details(info: &MediaInfo) -> Vec<ContextMenuItem<BillboardMenuCommand>> {
    let info_item = |label: &str, value: String| ContextMenuItem::Info {
        label: label.to_owned(),
        value,
    };
    let mut items = Vec::new();
    if let Some(video) = info.video.as_ref() {
        let rate = video
            .frame_rate
            .map_or_else(String::new, |fps| format!(", {} fps", trimmed(fps, 3)));
        items.push(info_item(
            "Picture",
            format!(
                "{}x{} {}{rate}",
                video.width,
                video.height,
                video.codec.to_uppercase()
            ),
        ));
    }
    if let Some(duration) = info.duration_seconds {
        items.push(info_item("Length", format_seconds_mmss(duration)));
    }
    if let Some(size) = info.size_bytes {
        items.push(info_item("Size", file_size_label(size)));
    }
    items
}

fn file_name(path: &str) -> String {
    Path::new(path).file_name().map_or_else(
        || path.to_owned(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// `1x`, `0.75x`.
fn speed_label(speed: f32) -> String {
    format!("{}x", trimmed(speed, 2))
}

/// `0 ms`, `+150 ms`, `-50 ms`.
fn delay_label(delay_ms: i32) -> String {
    if delay_ms == 0 {
        "0 ms".to_owned()
    } else {
        format!("{delay_ms:+} ms")
    }
}

/// `value` with at most `decimals` decimals and no trailing zeros.
fn trimmed(value: f32, decimals: usize) -> String {
    let fixed = format!("{value:.decimals$}");
    if fixed.contains('.') {
        fixed.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        fixed
    }
}

fn file_size_label(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    if size < 1024.0 {
        return format!("{bytes} B");
    }
    let mut unit = 0;
    size /= 1024.0;
    while size >= 1024.0 && unit + 1 < UNITS.len() {
        size /= 1024.0;
        unit += 1;
    }
    format!("{size:.1} {}", UNITS[unit])
}

/// Writes the frame at `time_seconds`, with the subtitles playback shows,
/// to the screenshot directory, off the main thread.
fn save_frame(source: VideoSource, time_seconds: f32) {
    let Some(directory) = screenshot_directory() else {
        eprintln!("No pictures directory to save the frame to");
        return;
    };
    let stem = source.path.file_stem().map_or_else(
        || "frame".to_owned(),
        |stem| stem.to_string_lossy().into_owned(),
    );
    let output = directory.join(format!("{stem} {}.png", timestamp_label(time_seconds)));
    thread::spawn(move || {
        if let Err(error) = std::fs::create_dir_all(&directory) {
            eprintln!("Failed to create {}: {error}", directory.display());
            return;
        }
        let result = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
            .args(still_frame_arguments(&source, time_seconds, &output))
            .output();
        match result {
            Ok(done) if done.status.success() => println!("Saved frame to {}", output.display()),
            Ok(done) => eprintln!(
                "Failed to save frame: {}",
                String::from_utf8_lossy(&done.stderr)
            ),
            Err(error) => eprintln!("Failed to start ffmpeg to save the frame: {error}"),
        }
    });
}

/// `12-34.567`, or `1-02-03.456` past an hour: sortable and valid in file
/// names.
fn timestamp_label(seconds: f32) -> String {
    let milliseconds = (seconds.max(0.0) * 1000.0).round() as u64;
    let (hours, minutes, seconds, millis) = (
        milliseconds / 3_600_000,
        milliseconds / 60_000 % 60,
        milliseconds / 1000 % 60,
        milliseconds % 1000,
    );
    if hours > 0 {
        format!("{hours}-{minutes:02}-{seconds:02}.{millis:03}")
    } else {
        format!("{minutes:02}-{seconds:02}.{millis:03}")
    }
}

/// The user's pictures directory, in a folder of the viewer's own.
fn screenshot_directory() -> Option<PathBuf> {
    dirs::picture_dir().map(|pictures| pictures.join(SCREENSHOT_DIRECTORY_NAME))
}

fn reveal_in_file_browser(path: &str) {
    #[cfg(target_os = "windows")]
    let result = {
        use std::os::windows::process::CommandExt;
        // Explorer parses its own command line: the path after `/select,`
        // must be quoted as-is, not as one quoted argument, and with
        // backslashes, or it opens the default folder instead.
        Command::new("explorer")
            .raw_arg(format!("/select,\"{}\"", path.replace('/', "\\")))
            .spawn()
    };
    #[cfg(target_os = "macos")]
    let result = Command::new("open").arg("-R").arg(path).spawn();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let result = Command::new("xdg-open")
        .arg(Path::new(path).parent().unwrap_or(Path::new(path)))
        .spawn();
    if let Err(error) = result {
        eprintln!("Failed to show {path} in the file browser: {error}");
    }
}

fn open_in_default_app(path: &str) {
    #[cfg(target_os = "windows")]
    let result = {
        use std::os::windows::process::CommandExt;
        Command::new("explorer")
            .raw_arg(format!("\"{}\"", path.replace('/', "\\")))
            .spawn()
    };
    #[cfg(target_os = "macos")]
    let result = Command::new("open").arg(path).spawn();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let result = Command::new("xdg-open").arg(path).spawn();
    if let Err(error) = result {
        eprintln!("Failed to open {path}: {error}");
    }
}

impl ClipboardHandle {
    fn copy(&mut self, text: &str) {
        let clipboard = match self.0.as_mut() {
            Some(clipboard) => clipboard,
            None => match arboard::Clipboard::new() {
                Ok(clipboard) => self.0.insert(clipboard),
                Err(error) => {
                    eprintln!("Failed to open the clipboard: {error}");
                    return;
                }
            },
        };
        if let Err(error) = clipboard.set_text(text) {
            eprintln!("Failed to copy to the clipboard: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::media_probe::{MediaTrack, VideoStreamInfo};

    fn track(stream_index: usize, kind_index: usize, codec: &str, default: bool) -> MediaTrack {
        MediaTrack {
            stream_index,
            kind_index,
            codec: codec.to_owned(),
            language: Some("eng".to_owned()),
            title: Some(format!("Track {stream_index}")),
            default,
        }
    }

    fn probe() -> MediaProbe {
        MediaProbe::Ready(Arc::new(MediaInfo {
            video: Some(VideoStreamInfo {
                stream_index: 0,
                codec: "hevc".to_owned(),
                width: 1920,
                height: 1080,
                frame_rate: Some(24000.0 / 1001.0),
            }),
            audio_tracks: vec![track(1, 0, "aac", true), track(2, 1, "ac3", false)],
            subtitle_tracks: vec![
                track(3, 0, "ass", true),
                track(4, 1, "hdmv_pgs_subtitle", false),
            ],
            duration_seconds: Some(1441.5),
            size_bytes: Some(851_673_088),
            start_seconds: 0.0,
        }))
    }

    fn section<'model>(
        model: &'model ContextMenuModel<BillboardMenuCommand>,
        heading: &str,
    ) -> &'model [ContextMenuItem<BillboardMenuCommand>] {
        &model
            .sections
            .iter()
            .find(|section| section.heading.as_deref() == Some(heading))
            .expect("section present")
            .items
    }

    #[test]
    fn the_menu_marks_what_plays_and_dims_what_cannot() {
        let model = video_menu_model(
            r"E:\a\ep.mkv",
            Some(&probe()),
            &VideoFileSettings::default(),
        );
        assert_eq!(model.title, "ep.mkv");
        let ContextMenuItem::Choice { options, .. } = &section(&model, "Audio")[0] else {
            panic!("audio track is a choice");
        };
        assert!(options[0].selected && !options[1].selected);

        let ContextMenuItem::Choice { value, options, .. } = &section(&model, "Subtitles")[0]
        else {
            panic!("subtitle track is a choice");
        };
        assert_eq!(value, "English (Track 3)");
        assert_eq!(options.len(), 3);
        assert!(!options[0].selected, "Off is not what shows");
        assert!(options[1].selected);
        assert_eq!(options[2].command, None);
    }

    #[test]
    fn until_the_file_is_read_tracks_show_a_note() {
        let model = video_menu_model(
            "ep.mkv",
            Some(&MediaProbe::Pending),
            &VideoFileSettings::default(),
        );
        assert!(matches!(
            &section(&model, "Audio")[0],
            ContextMenuItem::Info { value, .. } if value == "Reading file..."
        ));
    }

    #[test]
    fn a_file_without_sound_says_so_instead_of_an_empty_choice() {
        let MediaProbe::Ready(info) = probe() else {
            unreachable!()
        };
        let silent = MediaProbe::Ready(Arc::new(MediaInfo {
            audio_tracks: Vec::new(),
            ..(*info).clone()
        }));
        let model = video_menu_model("ep.mkv", Some(&silent), &VideoFileSettings::default());
        assert!(matches!(
            &section(&model, "Audio")[0],
            ContextMenuItem::Info { value, .. } if value == "None in this file"
        ));
    }

    #[test]
    fn a_delay_offers_its_reset_only_once_moved() {
        let mut settings = VideoFileSettings::default();
        let delay_reset = |settings: &VideoFileSettings| {
            let model = video_menu_model("ep.mkv", Some(&probe()), settings);
            match &section(&model, "Audio")[1] {
                ContextMenuItem::Stepper { value, reset, .. } => (value.clone(), reset.clone()),
                _ => panic!("delay is a stepper"),
            }
        };
        assert_eq!(delay_reset(&settings), ("0 ms".to_owned(), None));
        settings.audio_delay_ms = -150;
        assert_eq!(
            delay_reset(&settings),
            (
                "-150 ms".to_owned(),
                Some(BillboardMenuCommand::ResetAudioDelay)
            )
        );
    }

    #[test]
    fn labels_read_naturally() {
        assert_eq!(speed_label(1.0), "1x");
        assert_eq!(speed_label(0.75), "0.75x");
        assert_eq!(delay_label(150), "+150 ms");
        assert_eq!(file_size_label(851_673_088), "812.2 MB");
        assert_eq!(file_size_label(512), "512 B");
        assert_eq!(timestamp_label(754.5), "12-34.500");
        assert_eq!(timestamp_label(3723.456), "1-02-03.456");
        assert_eq!(trimmed(23.976_025, 3), "23.976");
    }
}
