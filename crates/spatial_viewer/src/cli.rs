//! Command-line configuration for the viewer binary.
//!
//! Parsing is hand-rolled rather than derived: the surface is a dozen flags
//! with no subcommands, so staying dependency-free keeps `--help` and
//! argument errors identical whether or not a render backend or the API is
//! reachable.

use anyhow::{bail, Context, Result};
use spatial_api::ControlValues;
use spatial_viewer_ui::control_values_from_assignments;

use crate::media_decode::DEFAULT_MAX_VIDEO_FPS;

#[derive(Debug, Clone)]
pub struct ViewerArgs {
    pub api: String,
    /// Initial values for server-described controls, from `--control
    /// id=value`. Applied over the first panel a server sends, so a scripted
    /// run starts where it asked to; ids the server never defines are
    /// reported once the panel arrives.
    pub controls: ControlValues,
    pub limit: usize,
    /// VRAM ceiling in MiB for resident billboard textures. Replaces a count
    /// cap: catalog images differ ~100x in size, so a count cannot bound the
    /// memory that actually runs out.
    pub texture_budget_mib: u32,
    pub image_concurrency: usize,
    pub max_texture_side: u32,
    /// Ceiling on video playback rate; a slower source plays at its native
    /// rate. Decoded frames are uncompressed, so memory and upload cost grow
    /// linearly with it.
    pub max_video_fps: f32,
    pub debug_probe_billboard: bool,
    /// Run a catalog benchmark for this many seconds as soon as the catalog
    /// finishes loading, then exit. `None` leaves benchmarking to the Debug
    /// pill button.
    pub benchmark_seconds: Option<u64>,
}

impl ViewerArgs {
    pub fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut control_assignments: Vec<String> = Vec::new();
        let mut parsed = Self {
            api: "http://127.0.0.1:8765".to_owned(),
            controls: ControlValues::new(),
            limit: 10_000,
            texture_budget_mib: spatial_viewer_ui::DEFAULT_TEXTURE_BUDGET_MIB,
            image_concurrency: 12,
            // Source resolution. Sizing every image down cost a visible
            // quality drop on 78% of on-screen billboards while saving only
            // 12% of decode time, because reading and parsing the file
            // dominates a decode regardless of the target size.
            max_texture_side: 0,
            max_video_fps: DEFAULT_MAX_VIDEO_FPS,
            debug_probe_billboard: false,
            benchmark_seconds: None,
        };

        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--api" => parsed.api = required_value(&flag, args.next())?,
                "--control" => control_assignments.push(required_value(&flag, args.next())?),
                "--limit" => parsed.limit = parse_value(&flag, args.next())?,
                "--texture-budget-mib" => {
                    parsed.texture_budget_mib = parse_value(&flag, args.next())?
                }
                "--image-concurrency" => {
                    parsed.image_concurrency = parse_value(&flag, args.next())?
                }
                "--max-texture-side" => parsed.max_texture_side = parse_value(&flag, args.next())?,
                "--max-video-fps" => parsed.max_video_fps = parse_value(&flag, args.next())?,
                "--debug-probe-billboard" => parsed.debug_probe_billboard = true,
                "--benchmark" => parsed.benchmark_seconds = Some(parse_value(&flag, args.next())?),
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => bail!("Unknown argument: {unknown}"),
            }
        }
        if !(parsed.max_video_fps.is_finite() && parsed.max_video_fps > 0.0) {
            bail!("--max-video-fps must be a positive number");
        }
        parsed.controls =
            control_values_from_assignments(control_assignments.iter().map(String::as_str))
                .map_err(|error| anyhow::anyhow!("Invalid value for --control: {error}"))?;
        Ok(parsed)
    }
}

fn required_value(flag: &str, value: Option<String>) -> Result<String> {
    value.with_context(|| format!("Missing value for {flag}"))
}

fn parse_value<T>(flag: &str, value: Option<String>) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = required_value(flag, value)?;
    raw.parse::<T>()
        .map_err(|error| anyhow::anyhow!("Invalid value for {flag}: {error}"))
}

fn print_help() {
    println!(
        "\
Mixed Media Spatial Renderer

USAGE:
    spatial_viewer [OPTIONS]

OPTIONS:
    --api <URL>           Viewer API base URL [default: http://127.0.0.1:8765]
    --control <ID=VALUE>  Initial value for a server-described control, repeatable
                          (e.g. --control x=0 --control shape=sphere). Which controls
                          exist is up to the API; open the View pill to see them
    --limit <COUNT>       Maximum projected points to load [default: 10000]
    --texture-budget-mib <MIB>
                          VRAM ceiling for resident billboard textures [default: 6144].
                          Adjustable at runtime from the Billboards panel
    --image-concurrency <COUNT>
                          Concurrent local media decode workers [default: 12]. Decode
                          threads contend with rendering: higher values raise decode
                          throughput but lengthen worst-case frames. Block compression
                          made decoding CPU-bound, so more workers now help where
                          they used to hurt
    --max-texture-side <PX>
                          Cap on the decoded texture side; 0 keeps source resolution
                          [default: 0]
    --max-video-fps <FPS> Cap on video playback rate; slower videos play at their native
                          rate [default: 60]. Each playing video buffers uncompressed
                          frames, so memory grows with this (~240 MiB per second of
                          buffered 60 fps video)
    --benchmark <SECONDS> Benchmark the catalog for SECONDS once it loads, write the
                          JSON report, print its path, and exit
    -h, --help            Print help
"
    );
}
