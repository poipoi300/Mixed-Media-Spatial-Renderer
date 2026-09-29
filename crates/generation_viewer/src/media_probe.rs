//! What a video file contains — its picture stream, sound and subtitle
//! tracks, length and size — read once per file with ffprobe, off the main
//! thread. Playback needs it before anything decodes (the frame rate times
//! every frame, and the default tracks pick what plays), and the context
//! menu lists its tracks and file details.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use bevy::prelude::*;
use serde::Deserialize;

use crate::background_work::{spawn_background, CompletedWork};

/// Subtitle codecs libass can draw from their text. Image-based subtitles
/// (Blu-ray PGS, DVD, DVB) need a different renderer.
const TEXT_SUBTITLE_CODECS: [&str; 7] =
    ["ass", "ssa", "subrip", "srt", "webvtt", "mov_text", "text"];

#[derive(Clone, Debug, PartialEq)]
pub struct MediaInfo {
    pub video: Option<VideoStreamInfo>,
    pub audio_tracks: Vec<MediaTrack>,
    pub subtitle_tracks: Vec<MediaTrack>,
    pub duration_seconds: Option<f32>,
    pub size_bytes: Option<u64>,
    /// The file's first timestamp. Seeks count from it, while subtitle
    /// events are timed on the file's own clock, which starts here.
    pub start_seconds: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VideoStreamInfo {
    /// Index of the stream in the file, as ffmpeg's `-map 0:<index>` takes.
    pub stream_index: usize,
    pub codec: String,
    pub width: u32,
    pub height: u32,
    /// Average rate, falling back to the nominal one; `None` when the file
    /// reports neither.
    pub frame_rate: Option<f32>,
}

/// One sound or subtitle track.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaTrack {
    /// Index of the stream in the file, as ffmpeg's `-map 0:<index>` takes.
    pub stream_index: usize,
    /// Position among the file's tracks of the same kind, as the subtitles
    /// filter's `si` option takes.
    pub kind_index: usize,
    pub codec: String,
    pub language: Option<String>,
    pub title: Option<String>,
    /// The file flags it to play when nobody chose otherwise.
    pub default: bool,
}

impl MediaTrack {
    /// Whether libass can draw it (see `TEXT_SUBTITLE_CODECS`).
    pub fn is_text_subtitle(&self) -> bool {
        TEXT_SUBTITLE_CODECS.contains(&self.codec.as_str())
    }

    /// "Japanese", "English (Dialogue@MTBB)", or "Track 2" when the file
    /// names neither.
    pub fn label(&self) -> String {
        let language = self.language.as_deref().map(language_name);
        match (language, self.title.as_deref()) {
            (Some(language), Some(title)) => format!("{language} ({title})"),
            (Some(language), None) => language,
            (None, Some(title)) => title.to_owned(),
            (None, None) => format!("Track {}", self.kind_index + 1),
        }
    }
}

impl MediaInfo {
    /// The sound track that plays when nobody chose one: the file's default,
    /// else its first.
    pub fn default_audio_track(&self) -> Option<&MediaTrack> {
        self.audio_tracks
            .iter()
            .find(|track| track.default)
            .or_else(|| self.audio_tracks.first())
    }

    /// The subtitles shown when nobody chose any: the file's default track,
    /// if it can be drawn. A file flagging none shows none.
    pub fn default_subtitle_track(&self) -> Option<&MediaTrack> {
        self.subtitle_tracks
            .iter()
            .find(|track| track.default && track.is_text_subtitle())
    }

    pub fn audio_track(&self, stream_index: usize) -> Option<&MediaTrack> {
        self.audio_tracks
            .iter()
            .find(|track| track.stream_index == stream_index)
    }

    pub fn subtitle_track(&self, stream_index: usize) -> Option<&MediaTrack> {
        self.subtitle_tracks
            .iter()
            .find(|track| track.stream_index == stream_index)
    }
}

/// Where a file's probe stands.
#[derive(Clone, Debug)]
pub enum MediaProbe {
    Pending,
    Ready(Arc<MediaInfo>),
    /// ffprobe could not read the file; playback falls back to defaults.
    Failed,
}

impl MediaProbe {
    pub fn info(&self) -> Option<&MediaInfo> {
        match self {
            Self::Ready(info) => Some(info),
            Self::Pending | Self::Failed => None,
        }
    }
}

/// Probes by file, each run once per session: a file's streams do not
/// change while the viewer runs.
#[derive(Resource, Default)]
pub struct MediaProbes {
    probes: HashMap<PathBuf, MediaProbe>,
    completed: CompletedWork<(PathBuf, Result<MediaInfo, String>)>,
}

impl MediaProbes {
    /// Starts probing `path` unless it already was.
    pub fn request(&mut self, path: &Path) {
        if self.probes.contains_key(path) {
            return;
        }
        self.probes.insert(path.to_owned(), MediaProbe::Pending);
        let completed = self.completed.clone();
        let path = path.to_owned();
        spawn_background(move || {
            let result = probe_media(&path).map_err(|error| format!("{error:#}"));
            completed.push((path, result));
        });
    }

    /// `None` for a file nobody asked to probe.
    pub fn get(&self, path: &Path) -> Option<&MediaProbe> {
        self.probes.get(path)
    }

    pub fn info(&self, path: &Path) -> Option<&MediaInfo> {
        self.get(path).and_then(MediaProbe::info)
    }

    pub fn receive(&mut self) {
        for (path, result) in self.completed.drain() {
            let probe = match result {
                Ok(info) => MediaProbe::Ready(Arc::new(info)),
                Err(error) => {
                    eprintln!("Failed to probe {}: {error}", path.display());
                    MediaProbe::Failed
                }
            };
            self.probes.insert(path, probe);
        }
    }
}

pub fn receive_media_probes(mut probes: ResMut<MediaProbes>) {
    probes.receive();
}

fn probe_media(path: &Path) -> Result<MediaInfo> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate\
             :stream_tags=language,title\
             :stream_disposition=default,attached_pic\
             :format=duration,size,start_time",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("failed to start ffprobe from PATH")?;
    if !output.status.success() {
        bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let report: ProbeReport =
        serde_json::from_slice(&output.stdout).context("ffprobe wrote unreadable JSON")?;
    Ok(media_info_from_report(report))
}

fn media_info_from_report(report: ProbeReport) -> MediaInfo {
    let mut audio_tracks = Vec::new();
    let mut subtitle_tracks = Vec::new();
    let mut video = None;
    for stream in report.streams {
        let tracks = match stream.codec_type.as_str() {
            "audio" => &mut audio_tracks,
            "subtitle" => &mut subtitle_tracks,
            // Cover art rides along as a one-frame video stream.
            "video" if video.is_none() && stream.disposition.attached_pic == 0 => {
                let frame_rate = stream
                    .avg_frame_rate
                    .as_deref()
                    .and_then(parse_frame_rate)
                    .or_else(|| stream.r_frame_rate.as_deref().and_then(parse_frame_rate));
                video = Some(VideoStreamInfo {
                    stream_index: stream.index,
                    codec: stream.codec_name,
                    width: stream.width.unwrap_or(0),
                    height: stream.height.unwrap_or(0),
                    frame_rate,
                });
                continue;
            }
            _ => continue,
        };
        tracks.push(MediaTrack {
            stream_index: stream.index,
            kind_index: tracks.len(),
            codec: stream.codec_name,
            language: stream.tags.language.filter(|language| language != "und"),
            title: stream.tags.title,
            default: stream.disposition.default != 0,
        });
    }
    MediaInfo {
        video,
        audio_tracks,
        subtitle_tracks,
        duration_seconds: report
            .format
            .duration
            .and_then(|duration| duration.parse().ok()),
        size_bytes: report.format.size.and_then(|size| size.parse().ok()),
        start_seconds: report
            .format
            .start_time
            .and_then(|start| start.parse().ok())
            .unwrap_or(0.0),
    }
}

/// Parses ffprobe's rational rate (`30000/1001`), rejecting the `0/0` it
/// reports for an unknown rate.
fn parse_frame_rate(rate: &str) -> Option<f32> {
    let (numerator, denominator) = rate.trim().split_once('/')?;
    let fps = numerator.parse::<f64>().ok()? / denominator.parse::<f64>().ok()?;
    (fps.is_finite() && fps > 0.0).then_some(fps as f32)
}

/// English name of an ISO 639-2 code for the languages releases commonly
/// carry; any other code shows as itself.
fn language_name(code: &str) -> String {
    let name = match code {
        "eng" | "en" => "English",
        "jpn" | "ja" => "Japanese",
        "fre" | "fra" | "fr" => "French",
        "ger" | "deu" | "de" => "German",
        "spa" | "es" => "Spanish",
        "ita" | "it" => "Italian",
        "por" | "pt" => "Portuguese",
        "rus" | "ru" => "Russian",
        "chi" | "zho" | "zh" => "Chinese",
        "kor" | "ko" => "Korean",
        "ara" | "ar" => "Arabic",
        "hin" | "hi" => "Hindi",
        "pol" | "pl" => "Polish",
        "dut" | "nld" | "nl" => "Dutch",
        "swe" | "sv" => "Swedish",
        "tur" | "tr" => "Turkish",
        "vie" | "vi" => "Vietnamese",
        "tha" | "th" => "Thai",
        "ind" | "id" => "Indonesian",
        _ => return code.to_owned(),
    };
    name.to_owned()
}

#[derive(Deserialize)]
struct ProbeReport {
    #[serde(default)]
    streams: Vec<ProbeStream>,
    #[serde(default)]
    format: ProbeFormat,
}

#[derive(Deserialize)]
struct ProbeStream {
    index: usize,
    #[serde(default)]
    codec_type: String,
    #[serde(default)]
    codec_name: String,
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    #[serde(default)]
    tags: ProbeTags,
    #[serde(default)]
    disposition: ProbeDisposition,
}

#[derive(Deserialize, Default)]
struct ProbeTags {
    language: Option<String>,
    title: Option<String>,
}

#[derive(Deserialize, Default)]
struct ProbeDisposition {
    #[serde(default)]
    default: u8,
    #[serde(default)]
    attached_pic: u8,
}

/// ffprobe reports these numbers as strings.
#[derive(Deserialize, Default)]
struct ProbeFormat {
    duration: Option<String>,
    size: Option<String>,
    start_time: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed from ffprobe's report on a dual-audio episode with three
    /// subtitle tracks and cover art.
    const EPISODE_REPORT: &str = r#"{
        "streams": [
            {"index": 0, "codec_name": "hevc", "codec_type": "video", "width": 1920, "height": 1080,
             "r_frame_rate": "24000/1001", "avg_frame_rate": "24000/1001",
             "disposition": {"default": 1, "attached_pic": 0}, "tags": {"title": "Presented By EMBER"}},
            {"index": 1, "codec_name": "aac", "codec_type": "audio",
             "disposition": {"default": 1, "attached_pic": 0}, "tags": {"language": "eng", "title": "Golumpa@FUNI"}},
            {"index": 2, "codec_name": "ac3", "codec_type": "audio",
             "disposition": {"default": 0, "attached_pic": 0}, "tags": {"language": "jpn"}},
            {"index": 3, "codec_name": "ass", "codec_type": "subtitle",
             "disposition": {"default": 1, "attached_pic": 0}, "tags": {"language": "eng", "title": "Signs & Song@EMBER"}},
            {"index": 4, "codec_name": "ass", "codec_type": "subtitle",
             "disposition": {"default": 0, "attached_pic": 0}, "tags": {"language": "eng", "title": "Dialogue@MTBB"}},
            {"index": 5, "codec_name": "hdmv_pgs_subtitle", "codec_type": "subtitle",
             "disposition": {"default": 0, "attached_pic": 0}, "tags": {"language": "jpn"}},
            {"index": 11, "codec_name": "mjpeg", "codec_type": "video", "width": 600, "height": 800,
             "r_frame_rate": "90000/1", "avg_frame_rate": "0/0",
             "disposition": {"default": 0, "attached_pic": 1}}
        ],
        "format": {"duration": "1441.500000", "size": "851673088", "start_time": "0.042000"}
    }"#;

    fn episode() -> MediaInfo {
        media_info_from_report(serde_json::from_str(EPISODE_REPORT).expect("valid report"))
    }

    #[test]
    fn a_report_splits_into_picture_sound_and_subtitle_tracks() {
        let info = episode();
        let video = info.video.as_ref().expect("has a picture");
        assert_eq!(
            (video.stream_index, video.width, video.height),
            (0, 1920, 1080)
        );
        assert!((video.frame_rate.expect("has a rate") - 23.976).abs() < 1e-3);
        assert_eq!(info.audio_tracks.len(), 2);
        assert_eq!(info.subtitle_tracks.len(), 3);
        assert_eq!(info.subtitle_tracks[1].kind_index, 1);
        assert_eq!(info.duration_seconds, Some(1441.5));
        assert_eq!(info.size_bytes, Some(851_673_088));
        assert_eq!(info.start_seconds, 0.042);
    }

    #[test]
    fn default_tracks_follow_the_file_flags() {
        let info = episode();
        assert_eq!(
            info.default_audio_track().map(|track| track.stream_index),
            Some(1)
        );
        assert_eq!(
            info.default_subtitle_track()
                .map(|track| track.stream_index),
            Some(3)
        );

        let mut unflagged = episode();
        for track in unflagged
            .audio_tracks
            .iter_mut()
            .chain(&mut unflagged.subtitle_tracks)
        {
            track.default = false;
        }
        assert_eq!(
            unflagged
                .default_audio_track()
                .map(|track| track.stream_index),
            Some(1)
        );
        assert_eq!(unflagged.default_subtitle_track(), None);
    }

    #[test]
    fn tracks_are_labelled_by_language_and_title() {
        let info = episode();
        assert_eq!(info.audio_tracks[0].label(), "English (Golumpa@FUNI)");
        assert_eq!(info.audio_tracks[1].label(), "Japanese");
        assert!(!info.subtitle_tracks[2].is_text_subtitle());
        assert!(info.subtitle_tracks[0].is_text_subtitle());
    }

    #[test]
    fn frame_rates_reject_the_unknown_rate() {
        assert_eq!(parse_frame_rate("0/0"), None);
        assert_eq!(parse_frame_rate("30/1"), Some(30.0));
    }
}
