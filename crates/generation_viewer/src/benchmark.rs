//! In-app catalog benchmark: "why is this catalog slow, on this machine?"
//!
//! The `tests/performance` harness answers that for synthetic fixtures, from
//! outside the process, against a scripted timeline. This module answers it
//! for the catalog the user actually has open, from the Debug pill, while
//! they fly through it — the case where the bottleneck depends on their real
//! image sizes, disk and GPU.
//!
//! It records three things per frame and reduces them at the end:
//!
//! * **Where main-thread time goes.** The [`FrameStageProbe`] stamps already
//!   bracket every scene system; the benchmark accumulates them per stage so
//!   the report ranks systems by total and worst-frame cost. This is the
//!   flamegraph-equivalent: attribution by stage rather than by call stack,
//!   which is the granularity the schedule actually has.
//! * **Where the pipeline is starved.** Decode workers busy, GPU upload bytes
//!   and queue depth, pending/loaded counts, and how often the loader could
//!   have used a worker but had none. Together these separate "the disk is
//!   the limit" from "the upload budget is the limit" from "we are not even
//!   asking for work".
//! * **Hardware utilization.** CPU and memory come from Bevy's
//!   `SystemInformationDiagnosticsPlugin`; GPU pass timings come from
//!   `RenderDiagnosticsPlugin` where the backend supports timestamp queries
//!   (Vulkan and DX12 — elsewhere those read as CPU-side pass time). Both are
//!   sampled through `DiagnosticsStore`, so nothing here polls the OS itself.
//!
//! The result is written next to the viewer's session state as JSON and
//! summarized in the Debug pill. Recording costs one resource write per
//! frame and allocates only when a run is active.

use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use bevy::{
    diagnostic::{DiagnosticPath, DiagnosticsStore, SystemInformationDiagnosticsPlugin},
    prelude::*,
    render::renderer::RenderAdapterInfo,
};
use generation_viewer_ui::{
    BenchmarkControls, BenchmarkPhase, BenchmarkReport, BenchmarkStageRow, BillboardControls,
    BillboardStats, NavigationSettings,
};
use serde_json::{json, Value};

use crate::{
    catalog_session::user_state_directory, performance::FrameStageProbe, ExplorerScene, FlyCamera,
};

/// Frames dropped from the front of a run. Starting a benchmark toggles UI,
/// and the first frames after that are dominated by the panel relayout rather
/// than by the catalog.
const BENCHMARK_WARMUP_FRAMES: usize = 30;
/// Slowest frames retained with their full stage breakdown.
const BENCHMARK_WORST_FRAMES: usize = 12;
/// Stages listed in the on-screen summary; the JSON report keeps all of them.
const BENCHMARK_SUMMARY_STAGES: usize = 5;
/// Bevy reports render-pass timings under this diagnostic prefix.
const RENDER_PASS_DIAGNOSTIC_PREFIX: &str = "render";

#[derive(Resource, Default)]
pub struct BenchmarkRun {
    active: Option<ActiveRun>,
}

/// Set by `--benchmark <seconds>`: start one run as soon as the catalog has
/// finished loading, then exit with the report path on stdout. Makes the same
/// measurement the Debug pill button makes scriptable.
#[derive(Resource)]
pub struct HeadlessBenchmark {
    duration: Duration,
    /// Set once the catalog has settled and the measured run has begun.
    /// Also gates the camera sweep, so the two stay in step.
    started: bool,
    quiet_frames: usize,
    /// When the wait for a quiet loader began, so it can be given up on.
    waiting_since: Option<Instant>,
    /// Whether the run began on the timeout rather than on a quiet loader,
    /// which the report has to disclose: those numbers include the tail of
    /// the initial fill.
    settled_before_start: bool,
    /// Which way along the sweep axis the camera is currently travelling;
    /// flipped each time it reaches the far side.
    sweep_direction: f32,
}

impl HeadlessBenchmark {
    pub fn new(seconds: u64) -> Self {
        Self {
            duration: Duration::from_secs(seconds.max(1)),
            started: false,
            quiet_frames: 0,
            waiting_since: None,
            settled_before_start: true,
            sweep_direction: 1.0,
        }
    }
}

/// Consecutive frames with no decode in flight and nothing waiting to upload
/// before the headless run treats the catalog as settled. A single quiet
/// frame happens routinely between batches.
const HEADLESS_SETTLE_FRAMES: usize = 30;
/// How long to wait for that quiet loader before starting anyway.
///
/// A configuration whose decodes are expensive enough never goes quiet at
/// all — every worker stays busy indefinitely. Waiting forever there means
/// the one configuration most worth measuring is the one that produces no
/// measurement, which reads as a crash rather than as a result. Starting
/// late is a worse measurement; starting never is no measurement.
const HEADLESS_SETTLE_TIMEOUT: Duration = Duration::from_secs(90);
/// Fraction of the catalog's extent the headless sweep crosses per second.
/// Fast enough that loading, quality refresh and eviction all stay busy for
/// the whole run; slow enough that the camera does not outrun the loader so
/// far that it only ever measures an empty view.
const HEADLESS_SWEEP_EXTENTS_PER_SECOND: f32 = 0.25;

/// Flies the camera back and forth *through* the catalog for the duration of
/// a headless run.
///
/// A benchmark taken from a stationary camera measures rendering a resident
/// set and nothing else — exactly the case where the loader cannot be the
/// bottleneck. Equally, a sweep that simply flies along the camera's initial
/// heading leaves the catalog within seconds and then measures empty space,
/// which reads as a permanently starved loader. So the sweep tracks a target
/// on the far side of the catalog centre, turns around when it arrives, and
/// always keeps the camera pointed at where the images are.
pub fn sweep_camera_for_headless_benchmark(
    mut headless: ResMut<HeadlessBenchmark>,
    controls: Res<BenchmarkControls>,
    scene: Res<ExplorerScene>,
    time: Res<Time>,
    mut camera: Query<(&mut Transform, &mut FlyCamera)>,
) {
    if !headless.started || controls.phase != BenchmarkPhase::Recording {
        return;
    }
    let Ok((mut transform, mut fly_camera)) = camera.get_single_mut() else {
        return;
    };

    let centre = scene.bounds.center;
    let extent = scene.bounds.max_extent.max(1.0);
    // Traverse along the catalog's longest axis, from one side to the other
    // through the middle, so the camera spends the run inside the point
    // cloud rather than approaching it from outside.
    let axis = sweep_axis(scene.bounds.maximum - scene.bounds.minimum);
    let reach = extent * 0.5;
    let target = centre + axis * reach * headless.sweep_direction;
    let to_target = target - transform.translation;
    if to_target.length() <= extent * 0.05 {
        headless.sweep_direction = -headless.sweep_direction;
    }

    // Look along the direction of travel: billboard orientation, LOD and
    // scheduling all key off the view vector, so a sweep that flew sideways
    // would exercise none of them the way real navigation does.
    let heading = normalized_or_forward(to_target);
    transform.look_to(heading, Vec3::Y);
    let (yaw, pitch) = crate::yaw_pitch_from_rotation(transform.rotation);
    fly_camera.yaw = yaw;
    fly_camera.pitch = pitch;

    let speed = extent * HEADLESS_SWEEP_EXTENTS_PER_SECOND;
    // The loader reads the camera's velocity for its lookahead, so it must
    // reflect the sweep rather than stay at the zero the fly controls leave.
    fly_camera.velocity = heading * speed;
    transform.translation += heading * speed * time.delta_secs();
}

/// Unit vector along the longest axis of the catalog's bounding box.
fn sweep_axis(extent: Vec3) -> Vec3 {
    if extent.x >= extent.y && extent.x >= extent.z {
        Vec3::X
    } else if extent.y >= extent.z {
        Vec3::Y
    } else {
        Vec3::Z
    }
}

fn normalized_or_forward(vector: Vec3) -> Vec3 {
    vector.try_normalize().unwrap_or(Vec3::NEG_Z)
}

/// Starts the `--benchmark` run once the catalog has actually settled, and
/// exits when its report is ready. Waiting for a quiet loader keeps the
/// measurement about steady-state behaviour rather than about the initial
/// fill, which is dominated by catalog streaming.
pub fn drive_headless_benchmark(
    mut headless: ResMut<HeadlessBenchmark>,
    mut controls: ResMut<BenchmarkControls>,
    stats: Res<BillboardStats>,
    mut exit: EventWriter<AppExit>,
) {
    if !headless.started {
        // `pending` never reaches zero on a real catalog: points behind the
        // camera stay deferred in the pending index by design. A quiet
        // loader — nothing decoding, nothing waiting to upload — is the
        // observable that actually means "the initial fill is done".
        // Measuring from here means the run reflects steady-state flight
        // rather than the one-off initial fill.
        let waiting_since = *headless.waiting_since.get_or_insert_with(Instant::now);
        let loader_quiet = stats.loaded > 0 && stats.in_flight == 0 && stats.upload_queue == 0;
        headless.quiet_frames = if loader_quiet {
            headless.quiet_frames + 1
        } else {
            0
        };
        let settled = headless.quiet_frames >= HEADLESS_SETTLE_FRAMES;
        let gave_up = stats.loaded > 0 && waiting_since.elapsed() >= HEADLESS_SETTLE_TIMEOUT;
        if settled || gave_up {
            if gave_up && !settled {
                headless.settled_before_start = false;
                eprintln!(
                    "benchmark: loader still busy after {}s; measuring anyway",
                    HEADLESS_SETTLE_TIMEOUT.as_secs()
                );
            }
            controls.request_start_for(headless.duration);
            headless.started = true;
        }
        return;
    }
    if controls.phase != BenchmarkPhase::Complete {
        return;
    }
    if let Some(report) = controls.report.as_ref() {
        if !headless.settled_before_start {
            println!(
                "note: loader had not settled after {}s, so these numbers include the initial fill",
                HEADLESS_SETTLE_TIMEOUT.as_secs()
            );
        }
        println!("{}", report.summary);
        match &report.report_path {
            Ok(path) => println!("benchmark report: {}", path.display()),
            Err(error) => eprintln!("benchmark report not written: {error}"),
        }
    }
    exit.send(AppExit::Success);
}

struct ActiveRun {
    started: Instant,
    duration: Duration,
    frames_seen: usize,
    /// Per-stage totals and worst observed frame, keyed by stage name.
    stages: HashMap<&'static str, StageAccumulator>,
    frame_times_ms: Vec<f64>,
    worst_frames: Vec<WorstFrame>,
    pipeline: PipelineAccumulator,
    hardware: HardwareAccumulator,
}

#[derive(Default, Clone, Copy)]
struct StageAccumulator {
    total_ms: f64,
    worst_ms: f64,
    samples: usize,
}

/// Loader/upload counters, accumulated so the report can say which stage of
/// the pipeline ran out of capacity first.
#[derive(Default)]
struct PipelineAccumulator {
    decode_worker_frames: usize,
    decode_worker_busy_frames: usize,
    /// Frames where work was pending but no decode was in flight — the
    /// loader wanted to progress and could not.
    starved_frames: usize,
    uploaded_textures: usize,
    uploaded_bytes: usize,
    upload_queue_total: usize,
    /// Frames with nothing decoding and nothing schedulable: the catalog is
    /// fully resident for this view and the loader has no work to do.
    idle_frames: usize,
    peak_upload_queue: usize,
    peak_in_flight: usize,
    peak_loaded: usize,
    peak_pending: usize,
    /// Peak VRAM the resident textures reached over the run.
    peak_texture_bytes: usize,
    orientation_updated: usize,
    /// Frames on which at least one billboard was close enough and in front
    /// of the camera to be oriented — i.e. frames where something was
    /// actually in view.
    frames_with_visible_billboards: usize,
    /// Summed per frame: points in view, and how many of those had a
    /// texture. Their ratio is the coverage the user actually perceives
    /// while flying, which residency counts cannot express — they include
    /// every texture in memory, most of it behind the camera.
    visible_billboard_samples: u64,
    visible_textured_samples: u64,
    /// Final reading of the loader's cumulative tile-encode split. Cumulative
    /// since the catalog loaded, so the run's total is the last value seen
    /// rather than a sum over frames.
    encode_opaque_surfaces: usize,
    encode_mixed_surfaces: usize,
    encode_opaque_texels: u64,
    encode_mixed_texels: u64,
    encode_opaque_nanos: u64,
    encode_mixed_nanos: u64,
}

#[derive(Default)]
struct HardwareAccumulator {
    cpu_samples: Vec<f64>,
    memory_samples: Vec<f64>,
    /// Render pass name to accumulated GPU (or CPU-fallback) milliseconds.
    render_passes: HashMap<String, (f64, usize)>,
    /// Timestamp of the newest measurement already recorded, per diagnostic.
    ///
    /// These diagnostics refresh on their own cadence (CPU sampling is
    /// throttled to sysinfo's minimum interval, GPU timestamps arrive with
    /// the render world), which is far slower than the frame rate. Reading
    /// the latest value every frame would re-record one measurement hundreds
    /// of times and turn the summary into "whatever was current longest".
    last_sample_times: HashMap<String, Instant>,
}

impl HardwareAccumulator {
    /// Returns a diagnostic's value only when it is one this accumulator has
    /// not already counted.
    fn take_new_measurement(
        &mut self,
        diagnostics: &DiagnosticsStore,
        path: &DiagnosticPath,
    ) -> Option<f64> {
        let measurement = diagnostics.get_measurement(path)?;
        let seen = self.last_sample_times.get(path.as_str());
        if seen.is_some_and(|last| *last >= measurement.time) {
            return None;
        }
        self.last_sample_times
            .insert(path.as_str().to_owned(), measurement.time);
        Some(measurement.value)
    }
}

struct WorstFrame {
    frame_ms: f64,
    elapsed_seconds: f64,
    loaded: usize,
    in_flight: usize,
    pending: usize,
    upload_queue: usize,
    uploaded_bytes: usize,
    stages: Vec<(&'static str, f64)>,
}

impl BenchmarkRun {
    fn start(&mut self, duration: Duration) {
        self.active = Some(ActiveRun {
            started: Instant::now(),
            duration,
            frames_seen: 0,
            stages: HashMap::new(),
            frame_times_ms: Vec::with_capacity(4096),
            worst_frames: Vec::with_capacity(BENCHMARK_WORST_FRAMES + 1),
            pipeline: PipelineAccumulator::default(),
            hardware: HardwareAccumulator::default(),
        });
    }
}

/// Drives an active benchmark: samples this frame, and finishes the run when
/// its duration elapses. Does nothing at all while no run is active.
#[allow(clippy::too_many_arguments)]
pub fn run_benchmark(
    mut run: ResMut<BenchmarkRun>,
    mut controls: ResMut<BenchmarkControls>,
    probe: Res<FrameStageProbe>,
    time: Res<Time<Real>>,
    diagnostics: Res<DiagnosticsStore>,
    stats: Res<BillboardStats>,
    billboard_controls: Res<BillboardControls>,
    navigation: Res<NavigationSettings>,
    scene: Res<ExplorerScene>,
    adapter: Option<Res<RenderAdapterInfo>>,
    camera: Query<&Transform, With<FlyCamera>>,
) {
    if let Some(requested) = controls.take_start_request() {
        run.start(requested);
        controls.phase = BenchmarkPhase::Recording;
    }
    let Some(active) = run.active.as_mut() else {
        return;
    };

    sample_frame(active, &probe, &time, &diagnostics, &stats);

    let elapsed = active.started.elapsed();
    controls.progress = (elapsed.as_secs_f32() / active.duration.as_secs_f32()).clamp(0.0, 1.0);
    if elapsed < active.duration {
        return;
    }

    let active = run.active.take().expect("run was active above");
    let report = finish_run(
        active,
        &scene,
        &billboard_controls,
        &navigation,
        adapter.as_deref(),
        camera.get_single().ok(),
    );
    controls.phase = BenchmarkPhase::Complete;
    controls.progress = 1.0;
    controls.report = Some(report);
}

fn sample_frame(
    active: &mut ActiveRun,
    probe: &FrameStageProbe,
    time: &Time<Real>,
    diagnostics: &DiagnosticsStore,
    stats: &BillboardStats,
) {
    active.frames_seen += 1;
    // Hardware diagnostics refresh on their own cadence and are cheap to
    // read, so they are sampled even during warmup.
    sample_hardware(&mut active.hardware, diagnostics);
    if active.frames_seen <= BENCHMARK_WARMUP_FRAMES {
        return;
    }

    let frame_ms = time.delta_secs_f64().max(0.0) * 1_000.0;
    active.frame_times_ms.push(frame_ms);

    let stages = probe.frame_stage_durations_ms();
    for (stage, stage_ms) in &stages {
        let entry = active.stages.entry(stage).or_default();
        entry.total_ms += stage_ms;
        entry.worst_ms = entry.worst_ms.max(*stage_ms);
        entry.samples += 1;
    }

    let pipeline = &mut active.pipeline;
    pipeline.decode_worker_frames += 1;
    if stats.in_flight > 0 {
        pipeline.decode_worker_busy_frames += 1;
    } else if stats.schedulable_pending > 0 {
        // Only *schedulable* pending counts as a stall. Deferred points
        // (behind the camera) sit in `pending` forever at a fixed pose, and
        // counting them made every idle catalog look starved.
        pipeline.starved_frames += 1;
    } else {
        pipeline.idle_frames += 1;
    }
    pipeline.uploaded_textures += stats.uploaded_textures;
    pipeline.uploaded_bytes += stats.uploaded_bytes;
    pipeline.upload_queue_total += stats.upload_queue;
    pipeline.peak_upload_queue = pipeline.peak_upload_queue.max(stats.upload_queue);
    pipeline.peak_in_flight = pipeline.peak_in_flight.max(stats.in_flight);
    pipeline.peak_loaded = pipeline.peak_loaded.max(stats.loaded);
    pipeline.peak_pending = pipeline.peak_pending.max(stats.pending);
    pipeline.peak_texture_bytes = pipeline.peak_texture_bytes.max(stats.texture_bytes_used);
    // Counters are cumulative since the catalog loaded, so the run's totals
    // are the final reading rather than a sum over frames.
    pipeline.orientation_updated += stats.orientation_updated;
    if stats.orientation_updated > 0 {
        pipeline.frames_with_visible_billboards += 1;
    }
    pipeline.visible_billboard_samples += stats.visible_billboards as u64;
    pipeline.visible_textured_samples += stats.visible_textured as u64;
    pipeline.encode_opaque_surfaces = stats.encode_opaque_surfaces;
    pipeline.encode_mixed_surfaces = stats.encode_mixed_surfaces;
    pipeline.encode_opaque_texels = stats.encode_opaque_texels;
    pipeline.encode_mixed_texels = stats.encode_mixed_texels;
    pipeline.encode_opaque_nanos = stats.encode_opaque_nanos;
    pipeline.encode_mixed_nanos = stats.encode_mixed_nanos;

    record_worst_frame(
        &mut active.worst_frames,
        WorstFrame {
            frame_ms,
            elapsed_seconds: active.started.elapsed().as_secs_f64(),
            loaded: stats.loaded,
            in_flight: stats.in_flight,
            pending: stats.pending,
            upload_queue: stats.upload_queue,
            uploaded_bytes: stats.uploaded_bytes,
            stages,
        },
    );
}

fn sample_hardware(hardware: &mut HardwareAccumulator, diagnostics: &DiagnosticsStore) {
    if let Some(cpu) =
        hardware.take_new_measurement(diagnostics, &SystemInformationDiagnosticsPlugin::CPU_USAGE)
    {
        hardware.cpu_samples.push(cpu);
    }
    if let Some(memory) =
        hardware.take_new_measurement(diagnostics, &SystemInformationDiagnosticsPlugin::MEM_USAGE)
    {
        hardware.memory_samples.push(memory);
    }
    // Render-pass timings arrive as one diagnostic per pass. Their names are
    // backend- and feature-dependent, so they are discovered rather than
    // enumerated; on backends without timestamp queries these carry CPU-side
    // pass time instead, which the report labels accordingly.
    let render_pass_paths = diagnostics
        .iter()
        .map(|diagnostic| diagnostic.path().clone())
        .filter(|path| path.as_str().starts_with(RENDER_PASS_DIAGNOSTIC_PREFIX))
        .collect::<Vec<_>>();
    for path in render_pass_paths {
        let Some(value) = hardware.take_new_measurement(diagnostics, &path) else {
            continue;
        };
        let entry = hardware
            .render_passes
            .entry(path.as_str().to_owned())
            .or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    }
}

fn record_worst_frame(worst: &mut Vec<WorstFrame>, frame: WorstFrame) {
    let position = worst
        .iter()
        .position(|existing| existing.frame_ms < frame.frame_ms)
        .unwrap_or(worst.len());
    if position >= BENCHMARK_WORST_FRAMES {
        return;
    }
    worst.insert(position, frame);
    worst.truncate(BENCHMARK_WORST_FRAMES);
}

fn finish_run(
    active: ActiveRun,
    scene: &ExplorerScene,
    controls: &BillboardControls,
    navigation: &NavigationSettings,
    adapter: Option<&RenderAdapterInfo>,
    camera: Option<&Transform>,
) -> BenchmarkReport {
    let mut frame_times = active.frame_times_ms.clone();
    frame_times.sort_by(f64::total_cmp);
    let frames = frame_times.len();
    let median = percentile(&frame_times, 0.50);
    let p95 = percentile(&frame_times, 0.95);
    let p99 = percentile(&frame_times, 0.99);
    let mean = if frames == 0 {
        0.0
    } else {
        frame_times.iter().sum::<f64>() / frames as f64
    };
    let fps = if mean > 0.0 { 1_000.0 / mean } else { 0.0 };

    let mut stages = active
        .stages
        .iter()
        .map(|(stage, accumulator)| BenchmarkStageRow {
            stage: (*stage).to_owned(),
            mean_ms: if accumulator.samples == 0 {
                0.0
            } else {
                accumulator.total_ms / accumulator.samples as f64
            },
            worst_ms: accumulator.worst_ms,
            share: if mean > 0.0 && accumulator.samples > 0 {
                (accumulator.total_ms / accumulator.samples as f64) / mean
            } else {
                0.0
            },
        })
        .collect::<Vec<_>>();
    stages.sort_by(|left, right| right.mean_ms.total_cmp(&left.mean_ms));

    let pipeline = &active.pipeline;
    let sampled_frames = pipeline.decode_worker_frames.max(1);
    let bottleneck = diagnose_bottleneck(&stages, pipeline, sampled_frames, controls);
    let visible_quality = visible_coverage_fraction(pipeline);
    let mean_visible = pipeline.visible_billboard_samples as f64 / sampled_frames as f64;
    let encoded_surfaces = pipeline.encode_opaque_surfaces + pipeline.encode_mixed_surfaces;
    let encode_line = if encoded_surfaces == 0 {
        String::new()
    } else {
        format!(
            "
         surfaces: {:.0}% opaque of {encoded_surfaces}, {:.1}s encoding",
            fraction(pipeline.encode_opaque_surfaces, encoded_surfaces) * 100.0,
            (pipeline.encode_opaque_nanos + pipeline.encode_mixed_nanos) as f64 / 1.0e9,
        )
    };
    let summary = format!(
        "{fps:.0} fps  p95 {p95:.1} ms  p99 {p99:.1} ms
\n         {:.0}% of {mean_visible:.0} in-view points textured{encode_line}
\n         {bottleneck}",
        visible_quality * 100.0
    );

    let json = json!({
        "schema": "generation_viewer.benchmark/1",
        "catalog": {
            "projected_points": scene.projection.points.len(),
            "coordinate_spacing": scene.coordinate_spacing,
            "texture_budget_mib": controls.texture_budget_mib,
            "max_texture_side": controls.max_texture_side,
            "reference_distance": navigation.reference_distance,
            "camera": camera.map(|transform| json!({
                "position": transform.translation.to_array(),
            })),
        },
        "hardware": {
            "gpu": adapter.map(|info| json!({
                "name": info.name,
                "backend": format!("{:?}", info.backend),
                "device_type": format!("{:?}", info.device_type),
                "driver": info.driver_info,
            })),
            "cpu_usage_percent": summarize(&active.hardware.cpu_samples),
            // Bevy's system CPU diagnostic reports 0 on some platforms
            // (observed on Windows 11 with sysinfo 0.32). Say so in the
            // report rather than presenting a flat zero as a measurement.
            "cpu_usage_available": active.hardware.cpu_samples.iter().any(|value| *value > 0.0),
            "memory_usage_percent": summarize(&active.hardware.memory_samples),
            "render_passes_ms": render_pass_json(&active.hardware),
        },
        "frames": {
            "sampled": frames,
            "warmup_skipped": BENCHMARK_WARMUP_FRAMES.min(active.frames_seen),
            "fps": fps,
            "mean_ms": mean,
            "median_ms": median,
            "p95_ms": p95,
            "p99_ms": p99,
        },
        "stages_ms": stages.iter().map(|row| json!({
            "stage": row.stage,
            "mean_ms": row.mean_ms,
            "worst_ms": row.worst_ms,
            "share_of_frame": row.share,
        })).collect::<Vec<_>>(),
        "pipeline": {
            "decode_busy_fraction": pipeline.decode_worker_busy_frames as f64 / sampled_frames as f64,
            "starved_fraction": pipeline.starved_frames as f64 / sampled_frames as f64,
            "idle_fraction": pipeline.idle_frames as f64 / sampled_frames as f64,
            "frames_with_visible_billboards_fraction":
                pipeline.frames_with_visible_billboards as f64 / sampled_frames as f64,
            "uploaded_textures": pipeline.uploaded_textures,
            "uploaded_mib": pipeline.uploaded_bytes as f64 / (1024.0 * 1024.0),
            "upload_mib_per_second": upload_rate_mib(pipeline, active.duration),
            "mean_upload_queue": pipeline.upload_queue_total as f64 / sampled_frames as f64,
            "peak_upload_queue": pipeline.peak_upload_queue,
            "peak_in_flight": pipeline.peak_in_flight,
            "peak_loaded": pipeline.peak_loaded,
            "peak_pending": pipeline.peak_pending,
            "peak_texture_mib": pipeline.peak_texture_bytes as f64 / (1024.0 * 1024.0),
            "visible_textured_fraction": visible_coverage_fraction(pipeline),
            "mean_visible_billboards": pipeline.visible_billboard_samples as f64
                / pipeline.decode_worker_frames.max(1) as f64,
            "orientation_updates": pipeline.orientation_updated,
        },
        "surface_encoding": surface_encoding_json(pipeline),
        "worst_frames": active.worst_frames.iter().map(|frame| json!({
            "frame_ms": frame.frame_ms,
            "elapsed_seconds": frame.elapsed_seconds,
            "loaded": frame.loaded,
            "in_flight": frame.in_flight,
            "pending": frame.pending,
            "upload_queue": frame.upload_queue,
            "uploaded_bytes": frame.uploaded_bytes,
            "stage_ms": frame.stages.iter()
                .map(|(stage, ms)| (stage.to_string(), json!(ms)))
                .collect::<serde_json::Map<_, _>>(),
        })).collect::<Vec<_>>(),
        "bottleneck": bottleneck,
    });

    let report_path = write_report(&json);
    BenchmarkReport {
        summary,
        bottleneck,
        stages: stages.into_iter().take(BENCHMARK_SUMMARY_STAGES).collect(),
        fps: fps as f32,
        p95_ms: p95 as f32,
        p99_ms: p99 as f32,
        sampled_frames: frames,
        decode_busy_fraction: (pipeline.decode_worker_busy_frames as f64 / sampled_frames as f64)
            as f32,
        starved_fraction: (pipeline.starved_frames as f64 / sampled_frames as f64) as f32,
        upload_mib_per_second: upload_rate_mib(pipeline, active.duration) as f32,
        report_path,
    }
}

/// Names the first limit the run actually hit, in the order the pipeline
/// would hit them. This is a heuristic summary of the numbers below it in the
/// report, not a substitute for reading them.
fn diagnose_bottleneck(
    stages: &[BenchmarkStageRow],
    pipeline: &PipelineAccumulator,
    sampled_frames: usize,
    controls: &BillboardControls,
) -> &'static str {
    let starved = pipeline.starved_frames as f64 / sampled_frames as f64;
    let busy = pipeline.decode_worker_busy_frames as f64 / sampled_frames as f64;
    let idle = pipeline.idle_frames as f64 / sampled_frames as f64;
    let visible = pipeline.frames_with_visible_billboards as f64 / sampled_frames as f64;
    let mean_queue = pipeline.upload_queue_total as f64 / sampled_frames as f64;
    // A camera that is not looking at anything produces a plausible-looking
    // report in which every loader number is an artefact: nothing is near
    // enough to schedule, so the loader reads as permanently starved. Say the
    // run is invalid rather than diagnose the pipeline from an empty view.
    if visible < 0.25 {
        return "INVALID: the camera saw almost no billboards; re-run pointing into the catalog";
    }
    // Nothing to load means nothing here can be the limit; frame time is
    // whatever rendering the resident set costs.
    if idle >= 0.9 {
        return "none: the catalog was fully resident, so this measures render cost only";
    }
    // A persistently deep upload queue means decodes are finishing faster
    // than the per-frame GPU upload budget drains them.
    if mean_queue >= 4.0 {
        return "GPU upload budget: decoded tiles are queueing faster than they upload";
    }
    if starved >= 0.25 {
        return "scheduler: work was pending with no decode running";
    }
    if busy >= 0.9 {
        return "decode workers: saturated, raise --image-concurrency or lower texture size";
    }
    if pipeline.peak_texture_bytes >= controls.texture_budget_bytes() {
        return "VRAM budget: the resident set is full, so loads evict each other";
    }
    if let Some(dominant) = stages.first() {
        if dominant.share >= 0.30 {
            return "main thread: one update stage dominates the frame";
        }
    }
    "GPU or present: frame time is not explained by loading or update work"
}

/// How alpha classification split this run's encode work.
///
/// The opaque BC7 preset is ~5x faster than the alpha-aware one. That is not
/// visible in a throughput number when a later stage is the binding
/// constraint, so the split and the time each group actually cost are
/// reported directly.
fn surface_encoding_json(pipeline: &PipelineAccumulator) -> serde_json::Value {
    let opaque_surfaces = pipeline.encode_opaque_surfaces;
    let mixed_surfaces = pipeline.encode_mixed_surfaces;
    let encoded_surfaces = opaque_surfaces + mixed_surfaces;
    let opaque_ms = pipeline.encode_opaque_nanos as f64 / 1.0e6;
    let mixed_ms = pipeline.encode_mixed_nanos as f64 / 1.0e6;
    let opaque_texels = pipeline.encode_opaque_texels;
    let mixed_texels = pipeline.encode_mixed_texels;
    let total_texels = opaque_texels + mixed_texels;

    // Per-megatexel rates make the two presets directly comparable even
    // though they encoded different images at different sizes.
    let rate = |ms: f64, texels: u64| {
        if texels == 0 {
            None
        } else {
            Some(ms / (texels as f64 / 1.0e6))
        }
    };
    let opaque_rate = rate(opaque_ms, opaque_texels);
    let mixed_rate = rate(mixed_ms, mixed_texels);

    json!({
        "opaque_surfaces": opaque_surfaces,
        "mixed_surfaces": mixed_surfaces,
        "opaque_surface_fraction": fraction(opaque_surfaces, encoded_surfaces),
        "opaque_encode_ms": opaque_ms,
        "mixed_encode_ms": mixed_ms,
        "opaque_ms_per_megatexel": opaque_rate,
        "mixed_ms_per_megatexel": mixed_rate,
        // How much faster the opaque preset ran, measured rather than assumed.
        "opaque_speedup": match (opaque_rate, mixed_rate) {
            (Some(opaque), Some(mixed)) if opaque > 0.0 => Some(mixed / opaque),
            _ => None,
        },
        // What the same texels would have cost with everything encoded
        // alpha-aware — the pipeline before the per-surface preset.
        "estimated_encode_ms_without_policy": mixed_rate
            .map(|rate| rate * (total_texels as f64 / 1.0e6)),
        "actual_encode_ms": opaque_ms + mixed_ms,
    })
}

fn fraction(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// Fraction of in-view point-frames that had a texture. This is the
/// perceptual measure: 1.0 means the user never looked at a hole.
fn visible_coverage_fraction(pipeline: &PipelineAccumulator) -> f64 {
    if pipeline.visible_billboard_samples == 0 {
        return 0.0;
    }
    pipeline.visible_textured_samples as f64 / pipeline.visible_billboard_samples as f64
}

fn upload_rate_mib(pipeline: &PipelineAccumulator, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64().max(f64::EPSILON);
    pipeline.uploaded_bytes as f64 / (1024.0 * 1024.0) / seconds
}

fn render_pass_json(hardware: &HardwareAccumulator) -> Value {
    hardware
        .render_passes
        .iter()
        .map(|(pass, (total, samples))| {
            (
                pass.clone(),
                json!(if *samples == 0 {
                    0.0
                } else {
                    total / *samples as f64
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into()
}

fn summarize(samples: &[f64]) -> Value {
    if samples.is_empty() {
        return Value::Null;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({
        "mean": samples.iter().sum::<f64>() / samples.len() as f64,
        "median": percentile(&sorted, 0.50),
        "peak": sorted.last().copied().unwrap_or(0.0),
        "samples": samples.len(),
    })
}

/// Linear-interpolated percentile of an already-sorted slice.
fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = quantile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

/// Writes the full report beside the viewer's other per-user state. Returns
/// the path, or a message explaining why nothing was written — a failed write
/// must not discard the on-screen summary the run just produced.
fn write_report(report: &Value) -> Result<PathBuf, String> {
    let Some(directory) = user_state_directory() else {
        return Err("no per-user state directory available".to_owned());
    };
    let directory = directory.join("benchmarks");
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = directory.join(format!("benchmark-{}.json", report_timestamp()));
    let contents = serde_json::to_string_pretty(report).map_err(|error| error.to_string())?;
    fs::write(&path, contents).map_err(|error| error.to_string())?;
    Ok(path)
}

/// Seconds since the Unix epoch, as a filename-safe run identifier. A
/// wall-clock date would need a calendar dependency the viewer does not
/// otherwise carry, and ordering is all the filename has to convey.
fn report_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(stage: &str, mean_ms: f64, share: f64) -> BenchmarkStageRow {
        BenchmarkStageRow {
            stage: stage.to_owned(),
            mean_ms,
            worst_ms: mean_ms * 3.0,
            share,
        }
    }

    fn pipeline(busy: usize, starved: usize, queue_total: usize) -> PipelineAccumulator {
        PipelineAccumulator {
            decode_worker_frames: 100,
            decode_worker_busy_frames: busy,
            starved_frames: starved,
            upload_queue_total: queue_total,
            // Every diagnosis below assumes a camera that was actually
            // looking at the catalog; the empty-view case has its own test.
            frames_with_visible_billboards: 100,
            ..Default::default()
        }
    }

    #[test]
    fn percentiles_interpolate_between_samples() {
        let sorted = [10.0, 20.0, 30.0, 40.0];

        assert_eq!(percentile(&sorted, 0.0), 10.0);
        assert_eq!(percentile(&sorted, 1.0), 40.0);
        assert_eq!(percentile(&sorted, 0.5), 25.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(percentile(&[7.0], 0.9), 7.0);
    }

    #[test]
    fn a_run_that_saw_no_billboards_is_reported_as_invalid() {
        let controls = BillboardControls::new(2048, 1024);
        // The exact shape of the first broken sweep: workers idle, work
        // "pending", and nothing whatsoever on screen.
        let blind = PipelineAccumulator {
            decode_worker_frames: 100,
            starved_frames: 88,
            frames_with_visible_billboards: 0,
            ..Default::default()
        };

        assert_eq!(
            diagnose_bottleneck(&[], &blind, 100, &controls),
            "INVALID: the camera saw almost no billboards; re-run pointing into the catalog",
            "an empty view must not be diagnosed as a starved scheduler"
        );
    }

    #[test]
    fn sweep_axis_follows_the_catalog_s_longest_dimension() {
        assert_eq!(sweep_axis(Vec3::new(10.0, 2.0, 3.0)), Vec3::X);
        assert_eq!(sweep_axis(Vec3::new(1.0, 9.0, 3.0)), Vec3::Y);
        assert_eq!(sweep_axis(Vec3::new(1.0, 2.0, 8.0)), Vec3::Z);
        // A degenerate (zero-extent) catalog still yields a unit axis rather
        // than a zero vector the sweep could not travel along.
        assert_eq!(sweep_axis(Vec3::ZERO), Vec3::X);
    }

    #[test]
    fn a_deep_upload_queue_is_reported_before_worker_saturation() {
        let controls = BillboardControls::new(2048, 1024);
        // Workers are saturated too, but the queue behind them is the
        // limit that is actually holding frames back.
        let accumulator = pipeline(95, 0, 800);

        assert_eq!(
            diagnose_bottleneck(&[], &accumulator, 100, &controls),
            "GPU upload budget: decoded tiles are queueing faster than they upload"
        );
    }

    #[test]
    fn idle_workers_with_pending_work_are_reported_as_a_scheduler_stall() {
        let controls = BillboardControls::new(2048, 1024);

        assert_eq!(
            diagnose_bottleneck(&[], &pipeline(10, 40, 0), 100, &controls),
            "scheduler: work was pending with no decode running"
        );
    }

    #[test]
    fn saturated_workers_are_reported_when_nothing_upstream_is_stalled() {
        let controls = BillboardControls::new(2048, 1024);

        assert_eq!(
            diagnose_bottleneck(&[], &pipeline(95, 0, 0), 100, &controls),
            "decode workers: saturated, raise --image-concurrency or lower texture size"
        );
    }

    #[test]
    fn a_dominant_update_stage_is_named_once_the_pipeline_is_healthy() {
        let controls = BillboardControls::new(2048, 1024);
        let stages = [
            stage("schedule_image_loads", 6.0, 0.45),
            stage("ui", 1.0, 0.1),
        ];

        assert_eq!(
            diagnose_bottleneck(&stages, &pipeline(10, 0, 0), 100, &controls),
            "main thread: one update stage dominates the frame"
        );
    }

    #[test]
    fn an_unexplained_frame_time_falls_through_to_gpu_or_present() {
        let controls = BillboardControls::new(2048, 1024);
        let stages = [stage("schedule_image_loads", 0.4, 0.05)];

        assert_eq!(
            diagnose_bottleneck(&stages, &pipeline(10, 0, 0), 100, &controls),
            "GPU or present: frame time is not explained by loading or update work"
        );
    }

    #[test]
    fn worst_frames_are_kept_slowest_first_and_bounded() {
        let mut worst = Vec::new();
        for index in 0..(BENCHMARK_WORST_FRAMES * 2) {
            record_worst_frame(
                &mut worst,
                WorstFrame {
                    frame_ms: index as f64,
                    elapsed_seconds: 0.0,
                    loaded: 0,
                    in_flight: 0,
                    pending: 0,
                    upload_queue: 0,
                    uploaded_bytes: 0,
                    stages: Vec::new(),
                },
            );
        }

        assert_eq!(worst.len(), BENCHMARK_WORST_FRAMES);
        assert!(worst
            .windows(2)
            .all(|pair| pair[0].frame_ms >= pair[1].frame_ms));
        assert_eq!(
            worst[0].frame_ms,
            (BENCHMARK_WORST_FRAMES * 2 - 1) as f64,
            "the slowest frame seen must be kept"
        );
    }

    #[test]
    fn summarize_reports_nothing_when_a_diagnostic_never_arrived() {
        assert_eq!(summarize(&[]), Value::Null);
        let summary = summarize(&[10.0, 20.0, 30.0]);
        assert_eq!(summary["peak"], json!(30.0));
        assert_eq!(summary["median"], json!(20.0));
        assert_eq!(summary["samples"], json!(3));
    }
}
