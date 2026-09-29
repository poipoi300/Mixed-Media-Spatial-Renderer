use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use anyhow::{bail, Context, Result};
use bevy::{
    math::URect,
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
};
use image::{imageops::FilterType, Rgba, RgbaImage};
use intel_tex_2::{bc7, RgbaSurface};

/// Default ceiling on the rate a video plays at; a source slower than this
/// plays at its native rate.
pub const DEFAULT_MAX_VIDEO_FPS: f32 = 60.0;

#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub side: u32,
    pub rgba: Vec<u8>,
    /// Where the image itself sits on the square canvas.
    ///
    /// [`fit_rgba_to_square`] centres a non-square source on a transparent
    /// square, so everything outside this rectangle is padding *this code*
    /// added — not content. Encoding covers only this rectangle, which is
    /// what keeps the surface free of transparency.
    ///
    /// Block-aligned on all four edges, so it is directly encodable.
    pub content: URect,
}

impl DecodedImage {
    /// A decode that fills its whole square, with no padding around it.
    pub fn full(side: u32, rgba: Vec<u8>) -> Self {
        Self {
            side,
            rgba,
            content: URect::new(0, 0, side, side),
        }
    }
}

/// Texel side of a block-compressed format's block. BC7 encodes 4x4 texels
/// into 16 bytes, so every compressed surface's dimensions must be a
/// multiple of this.
pub const TEXTURE_BLOCK_SIDE: u32 = 4;

/// Rounds a square side up to something a block-compressed format can encode.
pub fn block_aligned_side(side: u32) -> u32 {
    side.div_ceil(TEXTURE_BLOCK_SIDE) * TEXTURE_BLOCK_SIDE
}

/// Rounds a coordinate *down* to a block boundary.
///
/// Paired with [`block_aligned_side`] this grows the content rectangle
/// outward from the image on both axes, so block alignment can only ever
/// pull in padding and never crop an image pixel.
fn block_aligned_down(coordinate: u32) -> u32 {
    coordinate / TEXTURE_BLOCK_SIDE * TEXTURE_BLOCK_SIDE
}

/// Pixel format a billboard surface is uploaded in.
///
/// BC7 is what the GPU samples natively at 1 byte per texel instead of 4;
/// `Rgba8` is the fallback for an adapter without `TEXTURE_COMPRESSION_BC`,
/// and the format video frames always use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BillboardTextureFormat {
    #[default]
    Rgba8,
    Bc7,
}

/// What the negotiated render device will accept for billboard textures,
/// decided once at startup.
#[derive(Resource, Debug, Clone, Copy)]
pub struct BillboardTextureEncoding {
    pub format: BillboardTextureFormat,
    /// The device's `max_texture_dimension_2d`.
    ///
    /// A billboard's texture is one contiguous surface, so an image is
    /// decoded at source resolution up to this and scaled down past it.
    /// Exceeding it is not a soft failure — wgpu rejects the texture — and
    /// the catalog's largest side is 11656 px against a typical desktop
    /// limit of 16384, so the clamp is a guard rather than a routine path.
    pub max_texture_side: u32,
}

/// The lowest `max_texture_dimension_2d` any wgpu backend guarantees.
///
/// Used only when no render device is available (headless tests), where
/// assuming a desktop-class limit would let a test pass against a texture
/// no real adapter had agreed to.
const FALLBACK_MAX_TEXTURE_SIDE: u32 = 2048;

impl Default for BillboardTextureEncoding {
    fn default() -> Self {
        Self {
            format: BillboardTextureFormat::default(),
            max_texture_side: FALLBACK_MAX_TEXTURE_SIDE,
        }
    }
}

impl BillboardTextureFormat {
    pub fn wgpu_format(self) -> TextureFormat {
        match self {
            // Both are the sRGB variants: billboard pixels are sRGB-encoded,
            // and a linear format here would show as washed-out images.
            Self::Rgba8 => TextureFormat::Rgba8UnormSrgb,
            Self::Bc7 => TextureFormat::Bc7RgbaUnormSrgb,
        }
    }

    /// Bytes a `width` x `height` surface occupies in this format.
    pub fn surface_bytes(self, width: u32, height: u32) -> usize {
        let texels = width as usize * height as usize;
        match self {
            Self::Rgba8 => texels * 4,
            // 16 bytes per 4x4 block: exactly one byte per texel.
            Self::Bc7 => texels,
        }
    }
}

/// One billboard's whole texture: the image rectangle, encoded, in a single
/// contiguous buffer.
///
/// This used to be a grid of independently uploaded tiles. Each tile was its
/// own mesh with its own model matrix, and a rasterizer only guarantees
/// watertight adjacency between triangles whose transformed vertices are
/// bit-identical along the shared edge. Two neighbouring tiles reach that
/// edge through different matrices, so they differ by an ULP or two once the
/// billboard's rotation and world position are applied — and with MSAA off a
/// pixel on the boundary is then claimed by both quads or by neither.
/// Neither shows the clear colour: a one-pixel dark seam, appearing only at
/// the distances and orientations where the error crosses a pixel centre.
///
/// A single surface has no interior edge, so the seam cannot exist.
#[derive(Debug)]
pub struct BillboardSurface {
    /// Placement on the square, in its texel coordinates.
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Encoded texels in `format` — not necessarily RGBA bytes.
    pub pixels: Vec<u8>,
    pub format: BillboardTextureFormat,
}

/// Whether a surface's alpha channel has to survive its encoding.
///
/// The surface is cropped to the image rectangle, so the padding
/// [`fit_rgba_to_square`] adds is never part of it. What remains is decided
/// entirely by the source image: only one that carries its own transparency
/// reaches [`SurfaceAlpha::Mixed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SurfaceAlpha {
    /// Every texel is fully opaque.
    Opaque,
    /// Some texel is not, so alpha has to survive the encoding.
    Mixed,
}

/// Classifies a surface by a single pass over its alpha bytes.
///
/// Measured at ~0.3% of the surface's encode cost, and it decides which BC7
/// preset applies. Assuming opaque because the crop excludes the padding
/// would be cheaper still, but would silently erase the transparency of any
/// source image that carries its own — 6.8% of this catalog's PNGs.
fn classify_surface_alpha(rgba: &[u8]) -> SurfaceAlpha {
    if rgba.chunks_exact(4).all(|pixel| pixel[3] == 255) {
        SurfaceAlpha::Opaque
    } else {
        SurfaceAlpha::Mixed
    }
}

#[derive(Debug)]
pub struct DecodedBillboardImage {
    /// Side of the square the image was fitted to. The surface's placement
    /// is expressed in this square's texel coordinates, so the quad can be
    /// sized against the billboard mesh without knowing the source aspect.
    pub side: u32,
    /// `None` only for a degenerate decode with no image pixels at all.
    pub surface: Option<BillboardSurface>,
    pub encode_tally: SurfaceEncodeTally,
}

/// Per-image record of how alpha classification decided to encode a surface.
///
/// The point of the classification is that an opaque surface encodes ~5x
/// faster. That does not show up in a throughput number when some later
/// stage is the binding constraint, so the split is measured directly here
/// and carried into the benchmark report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SurfaceEncodeTally {
    /// Surfaces with no transparency at all: the fast opaque preset.
    pub opaque_surfaces: usize,
    /// Surfaces carrying the source image's own transparency: the
    /// alpha-aware preset.
    pub mixed_surfaces: usize,
    pub opaque_texels: u64,
    pub mixed_texels: u64,
    pub opaque_encode_nanos: u64,
    pub mixed_encode_nanos: u64,
}

impl SurfaceEncodeTally {
    pub fn add(&mut self, other: &Self) {
        self.opaque_surfaces += other.opaque_surfaces;
        self.mixed_surfaces += other.mixed_surfaces;
        self.opaque_texels += other.opaque_texels;
        self.mixed_texels += other.mixed_texels;
        self.opaque_encode_nanos += other.opaque_encode_nanos;
        self.mixed_encode_nanos += other.mixed_encode_nanos;
    }
}

/// The constant rate a video's frames are resampled to for playback. Frame
/// indices count frames at `fps`, not the source's own frames, so a
/// variable-rate source still maps every playback time to one stable index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoTiming {
    fps: f32,
    /// The file's own rate, when known; capped playback shows the source
    /// frame nearest each playback frame.
    source_fps: Option<f32>,
}

impl VideoTiming {
    /// Plays at the native rate, capped at `max_fps`. An unknown native rate
    /// plays at the cap: ffmpeg resamples to whatever rate it is given, so
    /// the timing stays correct and only duplicate frames are wasted.
    pub fn new(native_fps: Option<f32>, max_fps: f32) -> Self {
        Self {
            fps: native_fps.map_or(max_fps, |native| native.min(max_fps)),
            source_fps: native_fps,
        }
    }

    pub fn fps(&self) -> f32 {
        self.fps
    }

    pub fn frame_index(&self, time_seconds: f32) -> u64 {
        (time_seconds.max(0.0) * self.fps).floor() as u64
    }

    pub fn frame_seconds(&self, frame_index: u64) -> f32 {
        frame_index as f32 / self.fps
    }

    /// A time an accurate seek lands on the source frame shown as
    /// `frame_index` from: half a source frame before it, since the seek
    /// keeps the first frame at or after the time, and the source frame
    /// shown can sit up to half a source frame before the playback frame.
    pub fn frame_seek_seconds(&self, frame_index: u64) -> f32 {
        let source_fps = self.source_fps.unwrap_or(self.fps);
        (self.frame_seconds(frame_index) - 0.5 / source_fps).max(0.0)
    }
}

/// Splits a decode into its two expensive halves so a caller can check for
/// cancellation between them. Reading and parsing the file dominates for
/// large sources; the resize dominates for large targets.
pub fn open_image_file(path: &Path) -> Result<RgbaImage> {
    let dynamic =
        image::open(path).with_context(|| format!("failed to open image {}", path.display()))?;
    Ok(dynamic.to_rgba8())
}

pub fn fit_image_to_square(source: RgbaImage, max_texture_side: u32) -> Result<DecodedImage> {
    fit_rgba_to_square(source, max_texture_side)
}

/// The poster frame a video billboard shows before it plays. Needs no frame
/// rate: it is simply the first frame the stream decodes to.
pub fn decode_first_video_frame(path: &Path, max_texture_side: u32) -> Result<DecodedImage> {
    let side = video_decode_side(max_texture_side);
    let mut frames = decode_rgba_video_frames(
        path,
        side,
        None,
        &["-frames:v".to_owned(), "1".to_owned()],
        &letterbox_filter(side),
    )
    .context("failed to decode the first video frame")?;
    Ok(DecodedImage::full(side, frames.swap_remove(0)))
}

/// ffmpeg arguments that stream a video's frames from `start_frame_index` on
/// as raw RGBA squares of `side`, one frame per playback frame at `timing`'s
/// rate. The seek is an input seek, so ffmpeg jumps to the preceding keyframe
/// and decodes forward only from there; the first frame out is the start
/// frame.
///
/// Measured on 1080p 10-bit HEVC, a hardware decoder with two decode threads
/// and a single-threaded filter graph spends about 0.1 CPU-seconds per
/// second of video, against about 0.16 for the best software setting and 0.35
/// for ffmpeg's default thread counts, whose threading overhead dominates
/// once many videos play at once. `auto` picks whatever decoder the machine
/// has and falls back to software for codecs it cannot handle.
///
/// Subtitles are drawn by libass at the source resolution, before the frame
/// is scaled down, so they look as they do in a player (see
/// [`subtitle_filters`]).
pub fn video_stream_arguments(
    source: &VideoSource,
    timing: VideoTiming,
    start_frame_index: u64,
    side: u32,
) -> Vec<OsString> {
    let start_seconds = timing.frame_seconds(start_frame_index);
    let mut arguments: Vec<OsString> = [
        "-hwaccel".to_owned(),
        "auto".to_owned(),
        "-threads".to_owned(),
        "2".to_owned(),
        "-ss".to_owned(),
        format!("{start_seconds:.6}"),
        "-i".to_owned(),
    ]
    .map(OsString::from)
    .into();
    arguments.push(source.path.as_os_str().to_owned());
    if let Some(stream_index) = source.video_stream_index {
        arguments.extend(["-map".into(), format!("0:{stream_index}").into()]);
    }
    // Resampling to the playback rate before anything else means frames the
    // playback rate drops are never drawn on or scaled.
    let mut filters = vec![format!("fps={}", timing.fps())];
    filters.extend(subtitle_filters(source, start_seconds));
    filters.push(letterbox_filter(side));
    arguments.extend(
        [
            "-an".to_owned(),
            "-sn".to_owned(),
            "-dn".to_owned(),
            "-filter_threads".to_owned(),
            "1".to_owned(),
            "-vf".to_owned(),
            filters.join(","),
            "-f".to_owned(),
            "rawvideo".to_owned(),
            "-pix_fmt".to_owned(),
            "rgba".to_owned(),
            "pipe:1".to_owned(),
        ]
        .map(OsString::from),
    );
    arguments
}

/// ffmpeg arguments that write the frame at `time_seconds` to `output` as
/// an image at the source resolution, with the subtitles playback draws.
pub fn still_frame_arguments(
    source: &VideoSource,
    time_seconds: f32,
    output: &Path,
) -> Vec<OsString> {
    let mut arguments: Vec<OsString> = vec![
        "-ss".into(),
        format!("{:.6}", time_seconds.max(0.0)).into(),
        "-i".into(),
        source.path.as_os_str().to_owned(),
    ];
    if let Some(stream_index) = source.video_stream_index {
        arguments.extend(["-map".into(), format!("0:{stream_index}").into()]);
    }
    let filters = subtitle_filters(source, time_seconds.max(0.0));
    if !filters.is_empty() {
        arguments.extend(["-vf".into(), filters.join(",").into()]);
    }
    arguments.extend(["-frames:v", "1", "-y"].map(OsString::from));
    arguments.push(output.as_os_str().to_owned());
    arguments
}

/// Filters drawing `source`'s subtitles onto frames whose timestamps start
/// at zero for playback time `start_seconds`, as an input seek leaves them:
/// the subtitles filter times its events by the file's own clock, which
/// starts at the file's first timestamp, so timestamps are shifted to that
/// clock (less the subtitle delay) around it.
fn subtitle_filters(source: &VideoSource, start_seconds: f32) -> Vec<String> {
    let Some(subtitles) = source.subtitles else {
        return Vec::new();
    };
    let file_offset = start_seconds + subtitles.file_start_seconds - subtitles.delay_seconds;
    vec![
        format!("setpts=PTS{}/TB", signed_seconds(file_offset)),
        format!(
            "subtitles=filename={}:si={}",
            filter_graph_path(&source.path),
            subtitles.track_kind_index
        ),
        format!("setpts=PTS{}/TB", signed_seconds(-file_offset)),
    ]
}

/// Which file, picture stream and subtitles a video decode reads.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoSource {
    pub path: PathBuf,
    /// The picture stream to decode, as ffmpeg's `-map 0:<index>` takes;
    /// `None` leaves the choice to ffmpeg.
    pub video_stream_index: Option<usize>,
    pub subtitles: Option<SubtitleBurn>,
}

/// Subtitles drawn into a video's frames as it decodes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubtitleBurn {
    /// Position of the track among the file's subtitle tracks.
    pub track_kind_index: usize,
    /// Seconds the subtitles show later than the file times them.
    pub delay_seconds: f32,
    /// The file's first timestamp (see `MediaInfo::start_seconds`).
    pub file_start_seconds: f32,
}

/// `+1.500000` or `-1.500000`: an expression term that stays valid whatever
/// the sign.
fn signed_seconds(seconds: f32) -> String {
    let sign = if seconds < 0.0 { '-' } else { '+' };
    format!("{sign}{:.6}", seconds.abs())
}

/// `path` as a filter option value inside a filter graph. The option parser
/// and the graph parser each unescape once: the option level needs `\`, `'`
/// and `:` escaped, and the graph level takes the result single-quoted, where
/// only a quote itself has to step outside the quotes.
fn filter_graph_path(path: &Path) -> String {
    let mut option_escaped = String::new();
    for character in path.to_string_lossy().chars() {
        if matches!(character, '\\' | '\'' | ':') {
            option_escaped.push('\\');
        }
        option_escaped.push(character);
    }
    format!("'{}'", option_escaped.replace('\'', r"'\''"))
}

fn letterbox_filter(side: u32) -> String {
    format!(
        "scale={side}:{side}:force_original_aspect_ratio=decrease,pad={side}:{side}:(ow-iw)/2:(oh-ih)/2:color=black@0,format=rgba"
    )
}

/// Runs ffmpeg over `path` and splits its raw RGBA output into square frames
/// of `side`. The seek is placed before `-i`, making it an input seek that
/// jumps to the preceding keyframe instead of decoding from the start of the
/// file; `output_selection` limits which frames are emitted from there.
fn decode_rgba_video_frames(
    path: &Path,
    side: u32,
    input_seek_seconds: Option<f32>,
    output_selection: &[String],
    filter: &str,
) -> Result<Vec<Vec<u8>>> {
    let frame_size = side as usize * side as usize * 4;
    if frame_size == 0 {
        bail!("invalid video decode side {side}");
    }
    let input_seek = input_seek_seconds
        .map(|seconds| ["-ss".to_owned(), format!("{seconds:.6}")])
        .into_iter()
        .flatten();
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin"])
        .args(input_seek)
        .arg("-i")
        .arg(path)
        .args(output_selection)
        .arg("-vf")
        .arg(filter)
        .args(["-an", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"])
        .output()
        .with_context(|| "failed to start ffmpeg from PATH")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("ffmpeg failed for {}: {stderr}", path.display());
    }
    let frames: Vec<Vec<u8>> = output
        .stdout
        .chunks_exact(frame_size)
        .map(<[u8]>::to_vec)
        .collect();
    if frames.is_empty() {
        bail!("ffmpeg returned no frames for {}", path.display());
    }
    Ok(frames)
}

pub fn decoded_image_to_bevy_image(decoded: DecodedImage) -> Image {
    Image::new(
        Extent3d {
            width: decoded.side,
            height: decoded.side,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        decoded.rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

pub fn billboard_surface_to_bevy_image(surface: BillboardSurface) -> Image {
    billboard_surface_image(
        surface.width,
        surface.height,
        surface.pixels,
        surface.format,
    )
}

/// Builds a Bevy [`Image`] for one billboard surface.
///
/// Deliberately does *not* go through [`Image::new`]: that constructor
/// asserts `size.volume() * format.pixel_size() == data.len()`, and
/// `pixel_size()` panics outright for block-compressed formats ("Using
/// pixel_size for compressed textures is invalid"). Since the assert is a
/// `debug_assert`, routing BC7 through it would panic in every debug build
/// while release builds happened to work — the worst possible failure mode.
fn billboard_surface_image(
    width: u32,
    height: u32,
    data: Vec<u8>,
    format: BillboardTextureFormat,
) -> Image {
    let mut image = Image {
        data,
        ..Default::default()
    };
    image.texture_descriptor.dimension = TextureDimension::D2;
    image.texture_descriptor.size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    image.texture_descriptor.format = format.wgpu_format();
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image
}

/// Bytes a `source` image will occupy once fitted to a block-aligned square
/// and encoded, covering only the image rectangle.
///
/// The VRAM budget is charged before a decode starts, so this has to model
/// the same cropping [`encode_billboard_surface`] performs. Charging the
/// whole square would over-charge every non-square image by the padding it
/// never allocates.
pub fn billboard_surface_bytes(source: UVec2, format: BillboardTextureFormat) -> usize {
    let side = block_aligned_side(source.x.max(source.y));
    let width = source.x.min(side);
    let height = source.y.min(side);
    if width == 0 || height == 0 {
        return 0;
    }
    // The image is centred and block-snapped, matching `fit_rgba_to_square`.
    let offset_x = block_aligned_down((side - width) / 2);
    let offset_y = block_aligned_down((side - height) / 2);
    let surface_width = block_aligned_side(offset_x + width).min(side) - offset_x;
    let surface_height = block_aligned_side(offset_y + height).min(side) - offset_y;
    format.surface_bytes(surface_width, surface_height)
}

/// Encodes a decoded square's image rectangle into one contiguous surface.
///
/// This runs on the decode worker, keeping the large pixel copies and the
/// BC7 encode off the render thread.
///
/// The surface covers the *content* rectangle, not the square. Everything
/// outside it is padding [`fit_rgba_to_square`] added to make a non-square
/// source square; including any of it would put transparency in the surface
/// and force the alpha-aware BC7 preset, which measures 5.2x slower, to
/// shade fragments the alpha mask then throws away. Only 1.1% of a sampled
/// catalog is square, so the crop is what keeps nearly every encode on the
/// fast path.
pub fn encode_billboard_surface(
    decoded: &DecodedImage,
    format: BillboardTextureFormat,
) -> DecodedBillboardImage {
    let side = decoded.side;
    let content = decoded.content;
    let mut encode_tally = SurfaceEncodeTally::default();
    if content.is_empty() {
        return DecodedBillboardImage {
            side,
            surface: None,
            encode_tally,
        };
    }

    // `fit_rgba_to_square` already snapped the content rectangle to block
    // boundaries on both axes, so the crop is directly encodable and no
    // outward growth — which could only pull in padding — is needed here.
    let x = content.min.x;
    let y = content.min.y;
    let width = content.width();
    let height = content.height();
    debug_assert_eq!(
        (x % TEXTURE_BLOCK_SIDE, y % TEXTURE_BLOCK_SIDE),
        (0, 0),
        "content origin must be block-aligned"
    );
    debug_assert_eq!(
        (width % TEXTURE_BLOCK_SIDE, height % TEXTURE_BLOCK_SIDE),
        (0, 0),
        "content extent must be block-aligned"
    );

    let source_stride = side as usize * 4;
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for row in y..(y + height) {
        let start = row as usize * source_stride + x as usize * 4;
        rgba.extend_from_slice(&decoded.rgba[start..(start + width as usize * 4)]);
    }

    let alpha = classify_surface_alpha(&rgba);
    let texels = u64::from(width) * u64::from(height);
    let encode_start = Instant::now();
    let pixels = match format {
        BillboardTextureFormat::Rgba8 => rgba,
        BillboardTextureFormat::Bc7 => compress_surface_to_bc7(&rgba, width, height, alpha),
    };
    let encode_nanos = encode_start.elapsed().as_nanos() as u64;
    match alpha {
        SurfaceAlpha::Opaque => {
            encode_tally.opaque_surfaces += 1;
            encode_tally.opaque_texels += texels;
            encode_tally.opaque_encode_nanos += encode_nanos;
        }
        SurfaceAlpha::Mixed => {
            encode_tally.mixed_surfaces += 1;
            encode_tally.mixed_texels += texels;
            encode_tally.mixed_encode_nanos += encode_nanos;
        }
    }

    DecodedBillboardImage {
        side,
        surface: Some(BillboardSurface {
            x,
            y,
            width,
            height,
            pixels,
            format,
        }),
        encode_tally,
    }
}

/// Compresses a surface's RGBA texels to BC7.
///
/// The `ultra_fast` presets measured ~48 dB PSNR (mean error under 1/255) on
/// a sample of this catalog — below what the eye resolves — while the slower
/// presets cost 3x to 12x more for error that is already invisible.
///
/// An opaque surface encodes **5.15x faster** at very slightly *better*
/// quality, since the opaque modes spend their whole bit budget on colour
/// instead of reserving some for a channel that never varies.
fn compress_surface_to_bc7(rgba: &[u8], width: u32, height: u32, alpha: SurfaceAlpha) -> Vec<u8> {
    let surface = RgbaSurface {
        width,
        height,
        stride: width * 4,
        data: rgba,
    };
    let settings = match alpha {
        SurfaceAlpha::Opaque => bc7::opaque_ultra_fast_settings(),
        SurfaceAlpha::Mixed => bc7::alpha_ultra_fast_settings(),
    };
    bc7::compress_blocks(&settings, &surface)
}

fn fit_rgba_to_square(source: RgbaImage, max_texture_side: u32) -> Result<DecodedImage> {
    let (source_width, source_height) = source.dimensions();
    if source_width == 0 || source_height == 0 {
        bail!("image has zero width or height");
    }

    let source_side = source_width.max(source_height);
    let target_side_limit = if max_texture_side == 0 {
        source_side
    } else {
        max_texture_side.max(1)
    };
    // Block compression encodes 4x4 texel blocks, so a square whose side is
    // not a multiple of 4 has no valid encoding for its last row/column. 21%
    // of a sampled catalog is in that state. Rounding *up* is free here: the
    // image is centred on a transparent canvas below, so the at-most-3px of
    // extra padding is invisible, and quad geometry is derived from the side
    // rather than assuming it.
    let side = block_aligned_side(source_side.min(target_side_limit));
    // Scale to the *unaligned* side so alignment padding never enlarges the
    // image itself — it only widens the transparent margin around it.
    let unaligned_side = source_side.min(target_side_limit);
    let scale = unaligned_side as f32 / source_side as f32;
    let target_width = ((source_width as f32 * scale).round() as u32).clamp(1, side);
    let target_height = ((source_height as f32 * scale).round() as u32).clamp(1, side);

    let fitted = if target_width == source_width && target_height == source_height {
        source
    } else {
        image::imageops::resize(&source, target_width, target_height, FilterType::Triangle)
    };
    let mut canvas = RgbaImage::from_pixel(side, side, Rgba([0, 0, 0, 0]));
    // Snap the centring offset to a block boundary. The surface is cropped
    // to the content rectangle and must start on a block boundary to be
    // encodable; an image starting mid-block would force the crop outward
    // into the padding, carrying transparency into the surface and costing
    // the 5.2x slower alpha-aware preset. Shifting by up to 3px within the
    // image's own margin is invisible.
    let offset_x = block_aligned_down((side - target_width) / 2);
    let offset_y = block_aligned_down((side - target_height) / 2);
    image::imageops::overlay(
        &mut canvas,
        &fitted,
        i64::from(offset_x),
        i64::from(offset_y),
    );

    // The image's far edge can still end mid-block when its own dimensions
    // are not multiples of 4 — 41% of a sampled catalog. Rather than let the
    // surface extend into transparent padding (which would force the 5.2x
    // slower alpha-aware BC7 preset for the whole image), extend the image's
    // own edge texels across that at-most-3px sliver. The surface stays
    // fully opaque, and the visible cost is a 3px edge repeat on a >=900px
    // image.
    let content_end_x = block_aligned_side(offset_x + target_width).min(side);
    let content_end_y = block_aligned_side(offset_y + target_height).min(side);
    replicate_edges_into(
        &mut canvas,
        URect::new(
            offset_x,
            offset_y,
            offset_x + target_width,
            offset_y + target_height,
        ),
        UVec2::new(content_end_x, content_end_y),
    );

    Ok(DecodedImage {
        side,
        rgba: canvas.into_raw(),
        content: URect::new(offset_x, offset_y, content_end_x, content_end_y),
    })
}

/// Extends `image`'s right and bottom edge texels out to `extended`.
///
/// Block alignment can only round the surface's extent *up*, so when an
/// image ends mid-block the last block holds transparent padding and the
/// whole surface has to be encoded alpha-aware. Repeating the edge texel
/// across that sliver keeps it opaque. The sliver is at most
/// `TEXTURE_BLOCK_SIDE - 1` texels wide on each axis.
fn replicate_edges_into(canvas: &mut RgbaImage, image: URect, extended: UVec2) {
    for y in image.min.y..extended.y.min(canvas.height()) {
        let source_y = y.min(image.max.y - 1);
        for x in image.min.x..extended.x.min(canvas.width()) {
            if x < image.max.x && y < image.max.y {
                continue;
            }
            let source_x = x.min(image.max.x - 1);
            let texel = *canvas.get_pixel(source_x, source_y);
            canvas.put_pixel(x, y, texel);
        }
    }
}

pub fn video_decode_side(max_texture_side: u32) -> u32 {
    if max_texture_side == 0 {
        1024
    } else {
        max_texture_side.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_rgba_to_square_pads_mismatched_aspect_with_transparency() {
        // 8x3 on a 8px square: the image is 3 rows tall, so row 3 is inside
        // the aligned block and gets the edge repeated into it, while rows
        // 4-7 are true padding and stay transparent.
        let source = RgbaImage::from_pixel(8, 3, Rgba([10, 20, 30, 255]));

        let decoded = fit_rgba_to_square(source, 0).expect("image should fit");

        assert_eq!(decoded.side, 8);
        assert_eq!(decoded.rgba.len(), 8 * 8 * 4);
        let opaque = decoded
            .rgba
            .chunks_exact(4)
            .filter(|pixel| pixel[3] == 255)
            .count();
        // 8x3 image plus the 8x1 replicated alignment row.
        assert_eq!(opaque, 8 * 4, "image rows plus the block-alignment row");
        let transparent = decoded
            .rgba
            .chunks_exact(4)
            .filter(|pixel| pixel[3] == 0)
            .count();
        assert_eq!(transparent, 8 * 4, "the remaining rows stay padding");
    }

    /// Edge replication exists only to fill a block-alignment sliver. It must
    /// never spill further than that, or a small image would be smeared
    /// across its own margin instead of being letterboxed.
    #[test]
    fn edge_replication_never_exceeds_one_block() {
        for (width, height) in [(4031u32, 2887u32), (999, 501), (2, 1), (1000, 1000)] {
            let decoded = padded_gradient(width, height);
            let content = decoded.content;
            let image_width = width.min(decoded.side);
            let image_height = height.min(decoded.side);
            assert!(
                content.width() - image_width < TEXTURE_BLOCK_SIDE,
                "{width}x{height}: replicated {} columns",
                content.width() - image_width
            );
            assert!(
                content.height() - image_height < TEXTURE_BLOCK_SIDE,
                "{width}x{height}: replicated {} rows",
                content.height() - image_height
            );
        }
    }

    #[test]
    fn fit_rgba_to_square_scales_to_maximum_side() {
        let source = RgbaImage::from_pixel(4, 2, Rgba([10, 20, 30, 255]));

        let decoded = fit_rgba_to_square(source, 2).expect("image should fit");

        // Requested 2px, block-aligned up to 4px; the image is still scaled
        // to the 2px limit, so alignment costs margin and not detail.
        assert_eq!(decoded.side, 4);
        assert_eq!(decoded.rgba.len(), 4 * 4 * 4);
    }

    #[test]
    fn a_square_side_is_always_encodable_as_blocks() {
        // 4031 is a real catalog size, and one of the 21% that are not a
        // multiple of 4. A square that missed alignment would have no valid
        // BC7 encoding for its last row and column.
        for source_side in [1u32, 2, 3, 5, 215, 1023, 4031, 5375] {
            let source = RgbaImage::from_pixel(source_side, source_side, Rgba([9, 9, 9, 255]));
            let decoded = fit_rgba_to_square(source, 0).expect("image should fit");
            assert_eq!(
                decoded.side % TEXTURE_BLOCK_SIDE,
                0,
                "side {} produced unaligned square {}",
                source_side,
                decoded.side
            );
            assert!(decoded.side >= source_side, "alignment never shrinks");
            assert!(
                decoded.side - source_side < TEXTURE_BLOCK_SIDE,
                "alignment adds less than one block of padding"
            );
        }
    }

    #[test]
    fn bc7_surfaces_are_a_quarter_the_size_and_visually_lossless() {
        // A gradient rather than a flat fill: a constant image compresses
        // perfectly in any format and would prove nothing about quality.
        let mut source = RgbaImage::new(64, 64);
        for (x, y, pixel) in source.enumerate_pixels_mut() {
            *pixel = Rgba([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8, 255]);
        }
        let decoded = fit_rgba_to_square(source.clone(), 0).expect("image should fit");

        let rgba = encode_billboard_surface(&decoded, BillboardTextureFormat::Rgba8);
        let bc7 = encode_billboard_surface(&decoded, BillboardTextureFormat::Bc7);
        let rgba_bytes = rgba.surface.expect("a surface").pixels.len();
        let bc7_bytes = bc7.surface.as_ref().expect("a surface").pixels.len();
        assert_eq!(bc7_bytes * 4, rgba_bytes, "BC7 is one byte per texel");

        // Decode the blocks back and measure real error, so a change that
        // silently produced garbage blocks of the right length still fails.
        let surface = bc7.surface.as_ref().expect("a surface");
        let mut restored = vec![0u8; (surface.width * surface.height * 4) as usize];
        let blocks_x = (surface.width / TEXTURE_BLOCK_SIDE) as usize;
        for block_y in 0..(surface.height / TEXTURE_BLOCK_SIDE) as usize {
            for block_x in 0..blocks_x {
                let block = &surface.pixels[(block_y * blocks_x + block_x) * 16..][..16];
                let mut texels = [0u8; 64];
                bcdec_rs::bc7(block, &mut texels, 16);
                for row in 0..4 {
                    let start = ((block_y * 4 + row) * surface.width as usize + block_x * 4) * 4;
                    restored[start..start + 16].copy_from_slice(&texels[row * 16..row * 16 + 16]);
                }
            }
        }
        let squared_error: f64 = decoded
            .rgba
            .iter()
            .zip(restored.iter())
            .map(|(source, restored)| {
                let error = f64::from(source.abs_diff(*restored));
                error * error
            })
            .sum();
        let mse = squared_error / decoded.rgba.len() as f64;
        let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(f64::EPSILON)).log10();
        assert!(psnr > 40.0, "BC7 round trip only reached {psnr:.1} dB");
    }

    /// Builds a `width` x `height` opaque gradient centred on a transparent
    /// square, the way a real catalog image arrives.
    fn padded_gradient(width: u32, height: u32) -> DecodedImage {
        let mut source = RgbaImage::new(width, height);
        for (x, y, pixel) in source.enumerate_pixels_mut() {
            *pixel = Rgba([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8, 255]);
        }
        fit_rgba_to_square(source, 0).expect("image should fit")
    }

    #[test]
    fn alpha_classification_finds_a_source_image_s_own_transparency() {
        assert_eq!(
            classify_surface_alpha(&[7, 7, 7, 255]),
            SurfaceAlpha::Opaque
        );
        assert_eq!(classify_surface_alpha(&[0, 0, 0, 0]), SurfaceAlpha::Mixed);
        assert_eq!(
            classify_surface_alpha(&[7, 7, 7, 255, 0, 0, 0, 0]),
            SurfaceAlpha::Mixed
        );
        // A soft edge is neither fully on nor fully off, and must not be
        // encoded as opaque — that would erase it.
        assert_eq!(classify_surface_alpha(&[7, 7, 7, 128]), SurfaceAlpha::Mixed);
    }

    /// The surface must cover every image pixel and no padding at all.
    ///
    /// Missing part of the content would render the billboard with a hole in
    /// it; including padding would put transparency in the surface and force
    /// the 5.2x slower alpha-aware BC7 preset.
    #[test]
    fn the_surface_covers_the_image_and_nothing_but_the_image() {
        for (width, height) in [
            (2600u32, 3900u32),
            (1024, 3072),
            (3000, 1952),
            (1000, 1000),
            (4031, 2887),
            (512, 4096),
        ] {
            let decoded = padded_gradient(width, height);
            let encoded = encode_billboard_surface(&decoded, BillboardTextureFormat::Bc7);
            let surface = encoded.surface.as_ref().expect("a surface");
            let content = decoded.content;

            let rect = URect::new(
                surface.x,
                surface.y,
                surface.x + surface.width,
                surface.y + surface.height,
            );
            assert_eq!(
                rect, content,
                "{width}x{height}: surface {rect:?} does not match content {content:?}"
            );
            assert!(
                rect.max.x <= decoded.side && rect.max.y <= decoded.side,
                "{width}x{height}: surface left the canvas"
            );
        }
    }

    /// Cropping to the image is the whole point of the content rectangle: no
    /// surface should contain padding, so none should need the 5.2x slower
    /// alpha-aware preset.
    ///
    /// This holds even for sides that are not multiples of 4 (41% of the
    /// catalog) because `fit_rgba_to_square` replicates the image's own edge
    /// texels across the block-alignment sliver.
    #[test]
    fn cropping_to_the_image_leaves_the_surface_opaque() {
        for (width, height) in [
            (2600u32, 3900u32),
            (1024, 3072),
            (3000, 1952),
            (512, 4096),
            (4031, 2887),
        ] {
            let decoded = padded_gradient(width, height);
            let encoded = encode_billboard_surface(&decoded, BillboardTextureFormat::Bc7);
            assert_eq!(
                encoded.encode_tally.mixed_surfaces, 0,
                "{width}x{height} needed the alpha-aware preset"
            );
            assert_eq!(encoded.encode_tally.opaque_surfaces, 1);
        }
    }

    /// A source image that carries its own transparency must keep it. The
    /// crop excludes the padding, so this is the only way alpha can reach a
    /// surface — and encoding it opaque would erase it.
    #[test]
    fn a_transparent_source_image_still_encodes_alpha_aware() {
        let mut source = RgbaImage::new(64, 64);
        for (x, y, pixel) in source.enumerate_pixels_mut() {
            let alpha = if x < 32 { 255 } else { 0 };
            *pixel = Rgba([(x * 4) as u8, (y * 4) as u8, 0, alpha]);
        }
        let decoded = fit_rgba_to_square(source, 0).expect("image should fit");
        let encoded = encode_billboard_surface(&decoded, BillboardTextureFormat::Bc7);
        assert_eq!(encoded.encode_tally.mixed_surfaces, 1);
        assert_eq!(encoded.encode_tally.opaque_surfaces, 0);
    }

    /// Block alignment has to grow the surface *outward* from the image.
    /// Growing inward would silently crop a strip of pixels off every edge.
    #[test]
    fn the_surface_is_block_aligned_without_cropping_image_pixels() {
        // 4031x2887: both sides are odd and neither is a multiple of 4, so
        // the centred image starts and ends off a block boundary on both
        // axes before `fit_rgba_to_square` snaps and extends it.
        let decoded = padded_gradient(4031, 2887);
        let encoded = encode_billboard_surface(&decoded, BillboardTextureFormat::Bc7);
        let surface = encoded.surface.as_ref().expect("a surface");

        assert_eq!(surface.x % TEXTURE_BLOCK_SIDE, 0, "unaligned origin");
        assert_eq!(surface.y % TEXTURE_BLOCK_SIDE, 0, "unaligned origin");
        assert_eq!(surface.width % TEXTURE_BLOCK_SIDE, 0, "unaligned width");
        assert_eq!(surface.height % TEXTURE_BLOCK_SIDE, 0, "unaligned height");
        assert!(
            surface.width >= 4031 && surface.height >= 2887,
            "alignment cropped image pixels"
        );
        assert!(
            surface.width - 4031 < TEXTURE_BLOCK_SIDE && surface.height - 2887 < TEXTURE_BLOCK_SIDE,
            "alignment grew by more than one block"
        );
    }

    /// The VRAM budget charges a point before its decode starts, using
    /// `billboard_surface_bytes`. If that estimate and the encoder disagree,
    /// the budget either overshoots VRAM or leaves part of it permanently
    /// unusable — so they are pinned to each other here across the aspect
    /// ratios that dominate the catalog (2:3 and 3:4 portrait, 3:2
    /// landscape, and square).
    #[test]
    fn the_vram_estimate_matches_what_encoding_actually_produces() {
        for (width, height) in [
            (1024u32, 1536u32),
            (1536, 2048),
            (3000, 2000),
            (2048, 2048),
            (512, 3072),
            (1000, 1000),
            (4031, 2887),
        ] {
            let decoded = padded_gradient(width, height);
            for format in [BillboardTextureFormat::Bc7, BillboardTextureFormat::Rgba8] {
                let encoded = encode_billboard_surface(&decoded, format);
                let actual = encoded.surface.expect("a surface").pixels.len();
                let estimated = billboard_surface_bytes(UVec2::new(width, height), format);
                assert_eq!(
                    estimated, actual,
                    "{width}x{height} {format:?}: estimate {estimated} != actual {actual}"
                );
            }
        }
    }

    #[test]
    fn video_timing_plays_native_rates_up_to_the_cap() {
        let native = VideoTiming::new(Some(24.0), DEFAULT_MAX_VIDEO_FPS);
        assert_eq!(native.fps(), 24.0);

        let capped = VideoTiming::new(Some(120.0), DEFAULT_MAX_VIDEO_FPS);
        assert_eq!(capped.fps(), DEFAULT_MAX_VIDEO_FPS);

        let unknown = VideoTiming::new(None, 30.0);
        assert_eq!(unknown.fps(), 30.0);
    }

    #[test]
    fn video_time_maps_to_stable_frame_indices() {
        let timing = VideoTiming::new(Some(30.0), DEFAULT_MAX_VIDEO_FPS);
        assert_eq!(timing.frame_index(0.0), 0);
        assert_eq!(timing.frame_index(0.05), 1);
    }

    #[test]
    fn filter_paths_survive_both_unescaping_levels() {
        assert_eq!(
            filter_graph_path(Path::new(r"E:\Shows\A [1].mkv")),
            r"'E\:\\Shows\\A [1].mkv'"
        );
        // The quote steps outside the graph-level quotes; the option level
        // then unescapes `\'`.
        assert_eq!(
            filter_graph_path(Path::new("Frieren's Journey.mkv")),
            r"'Frieren\'\''s Journey.mkv'"
        );
    }

    #[test]
    fn subtitles_are_drawn_on_the_file_clock_less_their_delay() {
        let timing = VideoTiming::new(Some(10.0), DEFAULT_MAX_VIDEO_FPS);
        let source = VideoSource {
            path: PathBuf::from("a.mkv"),
            video_stream_index: Some(0),
            subtitles: Some(SubtitleBurn {
                track_kind_index: 2,
                delay_seconds: 0.5,
                file_start_seconds: 0.25,
            }),
        };
        let arguments: Vec<String> = video_stream_arguments(&source, timing, 40, 64)
            .into_iter()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect();
        assert!(arguments.windows(2).any(|pair| pair == ["-map", "0:0"]));
        let filters = &arguments[arguments.iter().position(|a| a == "-vf").expect("filters") + 1];
        assert!(filters.starts_with(
            "fps=10,setpts=PTS+3.750000/TB,subtitles=filename='a.mkv':si=2,setpts=PTS-3.750000/TB,"
        ));
    }
}
