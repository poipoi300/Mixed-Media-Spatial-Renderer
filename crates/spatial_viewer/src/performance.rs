//! Opt-in, rendered performance emulation used by `tests/performance/run.py`.
//!
//! The harness is absent from normal viewer runs.  Setting
//! `SPATIAL_VIEWER_PERF_CONFIG` to a JSON config file enables a seeded
//! action timeline, per-second JSONL telemetry, state assertions, and a clean
//! exit at the requested duration.  Process-level CPU and memory are sampled
//! by the Python parent so child ffmpeg processes are included as well.

use std::{
    collections::HashSet,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use bevy::{
    app::AppExit, ecs::system::SystemParam, input::InputSystem, prelude::*,
    transform::TransformSystem, ui::UiSystem,
};
use spatial_geometry::Bounds3;
use spatial_viewer_ui::{
    Action, BillboardControls, BillboardStats, ControlBindings, ControlPanelState, Input,
    NavigationSettings, PauseMenuState, StartMenuState, MIN_TEXTURE_BUDGET_MIB,
};

/// Budget the harness restores after squeezing the cache, large enough that
/// the resident set is bounded by the catalog rather than by memory.
const PERF_RESTORE_TEXTURE_BUDGET_MIB: u32 = 4096;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{
    audio_stream::AudioPlaybackState,
    image_loading::MediaBillboard,
    media_probe::MediaProbes,
    media_settings::MediaSettings,
    video_controls::{start_video, VideoStart},
    video_stream::VideoPlaybackState,
    ExplorerScene, FlyCamera, VideoControlsState,
};

pub const PERF_CONFIG_ENV: &str = "SPATIAL_VIEWER_PERF_CONFIG";
pub const PERF_ROOTS_ENV: &str = "SPATIAL_VIEWER_PERF_ROOTS";

#[derive(Debug, Clone, Deserialize)]
struct HarnessConfig {
    scenario: String,
    duration_seconds: f64,
    seed: u64,
    output_path: PathBuf,
    expected_points: usize,
    expect_video: bool,
    #[serde(default = "default_heartbeat_seconds")]
    heartbeat_seconds: f64,
    #[serde(default = "default_assertion_grace_seconds")]
    assertion_grace_seconds: f64,
    /// Frames before this many seconds are excluded from the steady-state
    /// uniformity summary so pipeline compilation and window creation do not
    /// dominate the spike statistics.
    #[serde(default = "default_warmup_seconds")]
    warmup_seconds: f64,
    #[serde(default)]
    timeline: TimelineKind,
}

/// Which seeded action timeline the harness drives.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TimelineKind {
    /// Discrete interactions: teleport, flight bursts, pause, cache and
    /// projection changes.
    #[default]
    Interactive,
    /// One continuous camera sweep through the whole catalog so billboards
    /// load, refresh quality, and evict for the entire run.
    Patrol,
}

fn default_heartbeat_seconds() -> f64 {
    1.0
}

fn default_assertion_grace_seconds() -> f64 {
    5.0
}

fn default_warmup_seconds() -> f64 {
    3.0
}

/// Frames slower than this multiple of the interval median count as spikes.
/// On a vsync-locked display a doubled frame time is exactly one missed
/// refresh.
const SPIKE_MEDIAN_MULTIPLIER: f64 = 2.0;
/// How many of the slowest steady-state frames the completion event lists
/// together with the loading state they were observed in.
const WORST_FRAME_CONTEXT_COUNT: usize = 8;
/// Distinct billboards a patrol must have shown before it counts as having
/// run through the catalog rather than idling in one spot.
const PATROL_MIN_DISTINCT_LOADED: usize = 64;
/// Control values the reprojection action submits, and the axis labels the
/// fixture server is expected to answer with. Both are fixture contract, not
/// viewer behavior: the viewer only knows it sent values and got a layout.
const REPROJECT_AXIS_VALUES: &[(&str, &str)] = &[("x", "1"), ("y", "0")];
const REPROJECT_EXPECTED_AXIS_LABELS: [&str; 3] = ["Row", "Column", "Layer"];
/// Patrol row spacing in multiples of the navigation reference distance
/// (roughly one coordinate spacing); rows this far apart keep every image
/// inside the full or high quality tier of some pass.
const PATROL_ROW_SPACING_FACTOR: f32 = 2.0;
const PATROL_MAX_ROWS: usize = 24;

/// Wall-clock stamps taken at the boundaries of the main-thread update so a
/// slow frame can be attributed to the update systems (and which stage of
/// them) rather than to rendering or presentation. Stamps are taken by
/// [`stamp_stage`] systems ordered around the scene update chain.
#[derive(Resource, Default)]
pub struct FrameStageProbe {
    stamps: Vec<(&'static str, Instant)>,
}

impl FrameStageProbe {
    fn stamp(&mut self, stage: &'static str) {
        self.stamps.push((stage, Instant::now()));
    }

    /// Discards the previous frame's stamps so a frame always measures
    /// itself.
    ///
    /// The `stamp_stage` systems are installed unconditionally, but the
    /// consumers that drain them (`collect_telemetry` and the in-app
    /// benchmark) are both opt-in. Without an unconditional reset the vector
    /// grew by one entry per stage per frame for the life of the process.
    pub(crate) fn begin_frame(&mut self) {
        self.stamps.clear();
    }

    /// Stage durations for the frame being measured, without draining them,
    /// so several consumers can read the same frame.
    pub(crate) fn frame_stage_durations_ms(&self) -> Vec<(&'static str, f64)> {
        self.stage_durations_ms()
    }

    /// Milliseconds spent between each consecutive pair of stamps, keyed by
    /// the stage the interval ended on.
    fn stage_durations_ms(&self) -> Vec<(&'static str, f64)> {
        self.stamps
            .windows(2)
            .map(|pair| {
                (
                    pair[1].0,
                    pair[1].1.duration_since(pair[0].1).as_secs_f64() * 1_000.0,
                )
            })
            .collect()
    }

    /// Drains the stamps, returning the stage durations and the instant of
    /// the last stamp.
    fn take(&mut self) -> (Vec<(&'static str, f64)>, Option<Instant>) {
        let durations = self.stage_durations_ms();
        let ended = self.stamps.last().map(|(_, instant)| *instant);
        self.stamps.clear();
        (durations, ended)
    }

    fn first_stamp(&self) -> Option<Instant> {
        self.stamps.first().map(|(_, instant)| *instant)
    }
}

/// Builds a system that records a probe stamp named `stage`. Install one in
/// the schedule before and after any group of systems whose cost should be
/// attributable.
pub fn stamp_stage(stage: &'static str) -> impl FnMut(ResMut<FrameStageProbe>) {
    move |mut probe: ResMut<FrameStageProbe>| probe.stamp(stage)
}

/// Opens a new frame's stage timings. Install exactly once, in `First`,
/// ahead of every [`stamp_stage`].
pub fn begin_frame_stages(mut probe: ResMut<FrameStageProbe>) {
    probe.begin_frame();
    probe.stamp("frame_start");
}

/// Always safe to install: without [`PERF_CONFIG_ENV`] this plugin is a no-op.
pub struct PerformanceHarnessPlugin;

impl Plugin for PerformanceHarnessPlugin {
    fn build(&self, app: &mut App) {
        let Some(config_path) = std::env::var_os(PERF_CONFIG_ENV) else {
            return;
        };
        let config = load_config(Path::new(&config_path)).unwrap_or_else(|error| {
            panic!(
                "failed to initialize viewer performance harness from {}: {error}",
                Path::new(&config_path).display()
            )
        });
        let harness = PerformanceHarness::new(config)
            .unwrap_or_else(|error| panic!("failed to create performance artifact: {error}"));
        app.insert_resource(harness)
            .init_resource::<FrameStageProbe>()
            .add_systems(PreUpdate, drive_timeline.after(InputSystem))
            // Telemetry runs at the end of PostUpdate so the stage probe has
            // seen the whole main-thread update, including transform
            // propagation and UI layout.
            .add_systems(
                PostUpdate,
                (stamp_stage("post_update"), collect_telemetry)
                    .chain()
                    .after(TransformSystem::TransformPropagate)
                    .after(UiSystem::Layout),
            );
    }
}

fn load_config(path: &Path) -> Result<HarnessConfig, String> {
    let contents = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let config: HarnessConfig =
        serde_json::from_str(&contents).map_err(|error| error.to_string())?;
    if !config.duration_seconds.is_finite() || config.duration_seconds <= 0.0 {
        return Err("duration_seconds must be finite and positive".to_owned());
    }
    if !config.heartbeat_seconds.is_finite() || config.heartbeat_seconds <= 0.0 {
        return Err("heartbeat_seconds must be finite and positive".to_owned());
    }
    if !config.warmup_seconds.is_finite() || config.warmup_seconds < 0.0 {
        return Err("warmup_seconds must be finite and non-negative".to_owned());
    }
    Ok(config)
}

#[derive(Resource)]
struct PerformanceHarness {
    config: HarnessConfig,
    output: BufWriter<File>,
    started: Instant,
    last_heartbeat: f64,
    frame_count: u64,
    interval_frame_times_ms: Vec<f64>,
    all_frame_times_ms: Vec<f64>,
    /// Frame times observed after the warmup period, in frame order.
    steady_frame_times_ms: Vec<f64>,
    interval_worst_frame: Option<FrameContext>,
    steady_worst_frames: Vec<FrameContext>,
    actions: Vec<ScheduledAction>,
    failures: usize,
    emitted_start: bool,
    finished: bool,
    first_point_seen_while_streaming: bool,
    camera_targeted_first_point: bool,
    patrol_distinct_loaded: HashSet<usize>,
    patrol_peak_loaded: usize,
    /// Completed update probe records, newest first, plus the pending
    /// current update's stages and the instant its probing ended.
    recent_updates: Vec<UpdateProbeRecord>,
    pending_update: Option<(Vec<(&'static str, f64)>, Instant)>,
}

/// Loading state captured alongside an unusually slow frame so a spike can
/// be attributed to uploads, evictions, or catalog work after the fact.
#[derive(Debug, Clone)]
struct FrameContext {
    frame_ms: f64,
    elapsed_seconds: f64,
    projected_points: usize,
    loaded: usize,
    in_flight: usize,
    upload_queue: usize,
    uploaded_textures: usize,
    uploaded_bytes: usize,
    orientation_updated: usize,
    video_decode_streams: usize,
    /// Probe records of the most recent main-thread updates, newest first.
    /// The frame delta Bevy reports is measured on the render thread and
    /// reaches the main world a frame late, so the update responsible for a
    /// spike is usually one or two records back.
    recent_updates: Vec<UpdateProbeRecord>,
}

/// Main-thread cost of one update: time per probed stage in stamp order,
/// then the wall-clock gap from its last stamp to the next update's first
/// stamp (render extraction, waiting on the render thread, presentation).
#[derive(Debug, Clone)]
struct UpdateProbeRecord {
    stage_ms: Vec<(&'static str, f64)>,
    outside_update_ms: f64,
}

impl UpdateProbeRecord {
    fn to_json(&self) -> Value {
        let stages = self
            .stage_ms
            .iter()
            .map(|(stage, ms)| (stage.to_string(), json!(ms)))
            .collect::<serde_json::Map<_, _>>();
        json!({
            "stage_ms": stages,
            "outside_update_ms": self.outside_update_ms,
            "total_ms": self.stage_ms.iter().map(|(_, ms)| ms).sum::<f64>() + self.outside_update_ms,
        })
    }
}

/// Updates whose probe records stay attached to a slow frame.
const RECENT_UPDATE_RECORDS: usize = 3;

impl FrameContext {
    fn to_json(&self) -> Value {
        json!({
            "recent_updates": self.recent_updates.iter().map(UpdateProbeRecord::to_json).collect::<Vec<_>>(),
            "frame_ms": self.frame_ms,
            "elapsed_seconds": self.elapsed_seconds,
            "projected_points": self.projected_points,
            "loaded": self.loaded,
            "in_flight": self.in_flight,
            "upload_queue": self.upload_queue,
            "uploaded_textures": self.uploaded_textures,
            "uploaded_bytes": self.uploaded_bytes,
            "orientation_updated": self.orientation_updated,
            "video_decode_streams": self.video_decode_streams,
        })
    }
}

/// Keeps the slowest `WORST_FRAME_CONTEXT_COUNT` frames, slowest first.
fn record_worst_frame(worst: &mut Vec<FrameContext>, context: FrameContext) {
    let position = worst
        .iter()
        .position(|existing| existing.frame_ms < context.frame_ms)
        .unwrap_or(worst.len());
    if position >= WORST_FRAME_CONTEXT_COUNT {
        return;
    }
    worst.insert(position, context);
    worst.truncate(WORST_FRAME_CONTEXT_COUNT);
}

impl PerformanceHarness {
    fn new(config: HarnessConfig) -> Result<Self, String> {
        if let Some(parent) = config.output_path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let file = File::create(&config.output_path).map_err(|error| error.to_string())?;
        let actions = build_timeline(&config);
        Ok(Self {
            config,
            output: BufWriter::new(file),
            started: Instant::now(),
            last_heartbeat: f64::NEG_INFINITY,
            frame_count: 0,
            interval_frame_times_ms: Vec::with_capacity(256),
            all_frame_times_ms: Vec::with_capacity(36_000),
            steady_frame_times_ms: Vec::with_capacity(36_000),
            interval_worst_frame: None,
            steady_worst_frames: Vec::with_capacity(WORST_FRAME_CONTEXT_COUNT + 1),
            actions,
            failures: 0,
            emitted_start: false,
            finished: false,
            first_point_seen_while_streaming: false,
            camera_targeted_first_point: false,
            patrol_distinct_loaded: HashSet::new(),
            patrol_peak_loaded: 0,
            recent_updates: Vec::with_capacity(RECENT_UPDATE_RECORDS + 1),
            pending_update: None,
        })
    }

    fn elapsed(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    fn emit(&mut self, value: Value) {
        if let Err(error) = serde_json::to_writer(&mut self.output, &value) {
            panic!("failed writing performance artifact: {error}");
        }
        self.output
            .write_all(b"\n")
            .and_then(|_| self.output.flush())
            .unwrap_or_else(|error| panic!("failed flushing performance artifact: {error}"));
    }
}

#[derive(Debug, Clone, Copy)]
enum ActionKind {
    CatalogFirstPoint,
    CatalogComplete,
    FirstTextureLoaded,
    /// Holds the first key bound to `action`.
    HoldAction {
        action: Action,
        seconds: f64,
    },
    TapPause {
        expected: bool,
    },
    TeleportRight,
    /// Set the VRAM ceiling for resident billboard textures, in MiB.
    SetTextureBudget {
        value: u32,
    },
    StartFirstVideo,
    ReprojectAxes,
    /// Sweep the camera through the catalog for `seconds`.
    Patrol {
        seconds: f64,
    },
}

impl ActionKind {
    fn name(self) -> &'static str {
        match self {
            Self::CatalogFirstPoint => "catalog_first_point",
            Self::CatalogComplete => "catalog_complete",
            Self::FirstTextureLoaded => "first_texture_loaded",
            Self::HoldAction {
                action: Action::MoveForward,
                ..
            } => "move_forward",
            Self::HoldAction {
                action: Action::MoveRight,
                ..
            } => "strafe_right",
            Self::HoldAction { .. } => "hold_key",
            Self::TapPause { expected: true } => "pause",
            Self::TapPause { expected: false } => "resume",
            Self::TeleportRight => "teleport_right",
            Self::SetTextureBudget { value } if value <= MIN_TEXTURE_BUDGET_MIB => {
                "texture_budget_low"
            }
            Self::SetTextureBudget { .. } => "texture_budget_restore",
            Self::StartFirstVideo => "start_first_video",
            Self::ReprojectAxes => "reproject_axes",
            Self::Patrol { .. } => "patrol_catalog",
        }
    }
}

/// The clock a `StartFirstVideo` action started, and where it stood then.
#[derive(Clone, Copy, Debug)]
struct VideoClockBaseline {
    image_id: usize,
    time_seconds: f32,
}

#[derive(Debug, Clone, Copy)]
enum ActionStatus {
    Waiting,
    Holding {
        until: f64,
        key: KeyCode,
        baseline: Vec3,
    },
    Checking {
        deadline: f64,
        baseline: Option<Vec3>,
        baseline_clock: Option<VideoClockBaseline>,
    },
    Patrolling {
        started_at: f64,
        until: f64,
    },
    Done,
}

#[derive(Debug, Clone, Copy)]
struct ScheduledAction {
    scheduled_at: f64,
    kind: ActionKind,
    status: ActionStatus,
}

fn build_timeline(config: &HarnessConfig) -> Vec<ScheduledAction> {
    let scale = (config.duration_seconds / 300.0).clamp(0.1, 1.0);
    let mut rng = SeededFuzz::new(config.seed);
    let mut at = |base: f64| {
        let scaled = base * scale;
        (scaled + rng.range(-0.35 * scale, 0.35 * scale)).max(0.1)
    };
    if config.timeline == TimelineKind::Patrol {
        // The sweep begins once the streamed catalog has settled and ends
        // just before the run so its assertion can still be evaluated.
        let patrol_start = at(20.0).max(1.5);
        let patrol_end = (config.duration_seconds - 1.0).max(patrol_start + 1.0);
        let mut actions = vec![
            scheduled(at(1.0), ActionKind::CatalogFirstPoint),
            scheduled(at(12.0), ActionKind::CatalogComplete),
            scheduled(at(20.0), ActionKind::FirstTextureLoaded),
            scheduled(
                patrol_start,
                ActionKind::Patrol {
                    seconds: patrol_end - patrol_start,
                },
            ),
        ];
        actions.sort_by(|left, right| left.scheduled_at.total_cmp(&right.scheduled_at));
        return actions;
    }
    let hold = |seconds: f64| (seconds * scale).max(0.45);
    let mut actions = vec![
        scheduled(at(1.0), ActionKind::CatalogFirstPoint),
        scheduled(at(12.0), ActionKind::CatalogComplete),
        scheduled(at(20.0), ActionKind::FirstTextureLoaded),
        scheduled(at(28.0), ActionKind::TeleportRight),
        scheduled(
            at(40.0),
            ActionKind::HoldAction {
                action: Action::MoveForward,
                seconds: hold(6.0),
            },
        ),
        scheduled(at(52.0), ActionKind::TapPause { expected: true }),
        scheduled(at(58.0), ActionKind::TapPause { expected: false }),
        scheduled(
            at(108.0),
            ActionKind::SetTextureBudget {
                value: MIN_TEXTURE_BUDGET_MIB,
            },
        ),
        scheduled(
            at(134.0),
            ActionKind::HoldAction {
                action: Action::MoveRight,
                seconds: hold(5.0),
            },
        ),
    ];
    if config.expect_video {
        actions.push(scheduled(at(24.0), ActionKind::StartFirstVideo));
    }
    actions.extend([
        scheduled(at(188.0), ActionKind::ReprojectAxes),
        scheduled(at(205.0), ActionKind::TapPause { expected: true }),
        scheduled(at(212.0), ActionKind::TapPause { expected: false }),
        scheduled(
            at(238.0),
            ActionKind::SetTextureBudget {
                value: PERF_RESTORE_TEXTURE_BUDGET_MIB,
            },
        ),
        scheduled(
            at(266.0),
            ActionKind::HoldAction {
                action: Action::MoveForward,
                seconds: hold(6.0),
            },
        ),
    ]);
    actions.sort_by(|left, right| left.scheduled_at.total_cmp(&right.scheduled_at));
    actions
}

/// The first key `action` is bound to alone (no modifiers, one tap), which
/// the harness presses for it.
fn bound_key(bindings: &ControlBindings, action: Action) -> Option<KeyCode> {
    bindings
        .chords(action)
        .filter(|chord| chord.modifiers.is_empty() && !chord.double_tap)
        .find_map(|chord| match chord.input {
            Input::Key(key) => Some(key),
            Input::Modifier(_) | Input::Mouse(_) | Input::Wheel => None,
        })
}

fn scheduled(scheduled_at: f64, kind: ActionKind) -> ScheduledAction {
    ScheduledAction {
        scheduled_at,
        kind,
        status: ActionStatus::Waiting,
    }
}

struct SeededFuzz(u64);

impl SeededFuzz {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn range(&mut self, low: f64, high: f64) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = self.0 as f64 / u64::MAX as f64;
        low + (high - low) * unit
    }
}

/// What starting a video touches, each absent until the scene is set up.
#[derive(SystemParam)]
struct HarnessVideo<'w> {
    controls: Option<ResMut<'w, VideoControlsState>>,
    playback: Option<ResMut<'w, VideoPlaybackState>>,
    audio: Option<ResMut<'w, AudioPlaybackState>>,
    probes: Option<ResMut<'w, MediaProbes>>,
    media_settings: Option<Res<'w, MediaSettings>>,
}

#[allow(clippy::too_many_arguments)]
fn drive_timeline(
    mut harness: ResMut<PerformanceHarness>,
    mut keyboard: ResMut<ButtonInput<KeyCode>>,
    bindings: Res<ControlBindings>,
    scene: Res<ExplorerScene>,
    pause_menu: Option<Res<PauseMenuState>>,
    start_menu: Option<Res<StartMenuState>>,
    mut controls: Option<ResMut<ControlPanelState>>,
    stats: Option<Res<BillboardStats>>,
    mut billboard_controls: Option<ResMut<BillboardControls>>,
    mut video: HarnessVideo,
    navigation: Option<Res<NavigationSettings>>,
    mut camera: Query<(&mut Transform, &mut FlyCamera)>,
    billboards: Query<&MediaBillboard>,
) {
    let elapsed = harness.elapsed();
    let grace = harness.config.assertion_grace_seconds.max(0.5);
    let expected_points = harness.config.expected_points;
    let camera_position = camera
        .get_single()
        .ok()
        .map(|(transform, _)| transform.translation);
    let paused = pause_menu.as_ref().map(|value| value.paused);
    if !harness.first_point_seen_while_streaming
        && start_menu.as_ref().is_some_and(|menu| menu.loading)
        && !scene.image_points.is_empty()
    {
        harness.first_point_seen_while_streaming = true;
        harness.camera_targeted_first_point = camera.get_single().is_ok_and(|(transform, _)| {
            let direction =
                (scene.image_points[0].position - transform.translation).normalize_or_zero();
            transform.forward().as_vec3().dot(direction) > 0.999
        });
    }
    let first_point_seen_while_streaming = harness.first_point_seen_while_streaming;
    let camera_targeted_first_point = harness.camera_targeted_first_point;
    let patrol_active = harness
        .actions
        .iter()
        .any(|action| matches!(action.status, ActionStatus::Patrolling { .. }));
    if patrol_active {
        let mut loaded_now = 0;
        for billboard in &billboards {
            loaded_now += 1;
            harness.patrol_distinct_loaded.insert(billboard.image_id);
        }
        harness.patrol_peak_loaded = harness.patrol_peak_loaded.max(loaded_now);
    }
    let patrol = PatrolProgress {
        distinct_loaded: harness.patrol_distinct_loaded.len(),
        peak_loaded: harness.patrol_peak_loaded,
    };
    let patrol_row_spacing = navigation
        .as_ref()
        .map_or(1.0, |settings| settings.reference_distance)
        .max(1.0)
        * PATROL_ROW_SPACING_FACTOR;
    let mut events = Vec::new();
    let mut failure_count = 0;

    for action in &mut harness.actions {
        match action.status {
            ActionStatus::Waiting if elapsed >= action.scheduled_at => match action.kind {
                ActionKind::CatalogFirstPoint
                | ActionKind::CatalogComplete
                | ActionKind::FirstTextureLoaded => {
                    action.status = ActionStatus::Checking {
                        deadline: elapsed + grace.max(12.0),
                        baseline: None,
                        baseline_clock: None,
                    };
                }
                ActionKind::HoldAction {
                    action: held,
                    seconds,
                } => {
                    let (Some(baseline), Some(key)) = (camera_position, bound_key(&bindings, held))
                    else {
                        continue;
                    };
                    keyboard.press(key);
                    action.status = ActionStatus::Holding {
                        until: elapsed + seconds,
                        key,
                        baseline,
                    };
                    events.push(action_event(
                        action,
                        elapsed,
                        "started",
                        json!({"key": format!("{key:?}")}),
                    ));
                }
                ActionKind::TapPause { .. } | ActionKind::TeleportRight => {
                    let tapped = match action.kind {
                        ActionKind::TapPause { .. } => Action::PauseMenu,
                        _ => Action::StepRight,
                    };
                    let Some(key) = bound_key(&bindings, tapped) else {
                        continue;
                    };
                    keyboard.press(key);
                    action.status = ActionStatus::Holding {
                        until: elapsed + 0.05,
                        key,
                        baseline: camera_position.unwrap_or(Vec3::ZERO),
                    };
                    events.push(action_event(
                        action,
                        elapsed,
                        "started",
                        json!({"key": format!("{key:?}")}),
                    ));
                }
                ActionKind::SetTextureBudget { value } => {
                    if let Some(controls) = billboard_controls.as_mut() {
                        controls.texture_budget_mib = value;
                        action.status = ActionStatus::Checking {
                            deadline: elapsed + grace,
                            baseline: None,
                            baseline_clock: None,
                        };
                        events.push(action_event(
                            action,
                            elapsed,
                            "started",
                            json!({"texture_budget_mib": value}),
                        ));
                    }
                }
                ActionKind::StartFirstVideo => {
                    let loaded_video = billboards.iter().find(|billboard| billboard.is_video);
                    let (
                        Some(billboard),
                        Some(controls),
                        Some(playback),
                        Some(probes),
                        Some(media_settings),
                    ) = (
                        loaded_video,
                        video.controls.as_mut(),
                        video.playback.as_mut(),
                        video.probes.as_mut(),
                        video.media_settings.as_ref(),
                    )
                    else {
                        // Loading is deliberately asynchronous. Keep trying
                        // through the action's grace window before failing.
                        if elapsed > action.scheduled_at + grace.max(15.0) {
                            events.push(action_event(
                                action,
                                elapsed,
                                "failed",
                                json!({"reason": "no loaded video billboard"}),
                            ));
                            action.status = ActionStatus::Done;
                            failure_count += 1;
                        }
                        continue;
                    };
                    start_video(
                        VideoStart {
                            controls,
                            video_playback: playback,
                            audio_playback: video.audio.as_deref_mut(),
                            probes,
                            media_settings,
                            // Every run starts its video from the top.
                            remember_position: false,
                        },
                        billboard,
                        scene.max_texture_side,
                    );
                    controls.set_playing(billboard.image_id, true);
                    let baseline_clock =
                        controls
                            .clock(billboard.image_id)
                            .map(|clock| VideoClockBaseline {
                                image_id: billboard.image_id,
                                time_seconds: clock.time_seconds(),
                            });
                    action.status = ActionStatus::Checking {
                        deadline: elapsed + grace.max(4.0),
                        baseline: None,
                        baseline_clock,
                    };
                    events.push(action_event(
                        action,
                        elapsed,
                        "started",
                        json!({"image_id": billboard.image_id}),
                    ));
                }
                ActionKind::ReprojectAxes => {
                    // Swaps the fixture's first two axis controls, which is a
                    // full reprojection round-trip through the API whatever
                    // those controls happen to mean to that server.
                    if let Some(controls) = controls.as_mut() {
                        for (control_id, value) in REPROJECT_AXIS_VALUES {
                            controls.set_value(
                                control_id,
                                spatial_api::ControlValue::Text((*value).to_owned()),
                            );
                        }
                        action.status = ActionStatus::Checking {
                            deadline: elapsed + grace.max(15.0),
                            baseline: None,
                            baseline_clock: None,
                        };
                        events.push(action_event(
                            action,
                            elapsed,
                            "started",
                            json!({"requested_axes": REPROJECT_AXIS_VALUES
                                .iter()
                                .map(|(control_id, value)| format!("{control_id}={value}"))
                                .collect::<Vec<_>>()}),
                        ));
                    }
                }
                ActionKind::Patrol { seconds } => {
                    let Ok((mut transform, mut fly_camera)) = camera.get_single_mut() else {
                        continue;
                    };
                    let path = PatrolPath::new(&scene.bounds, patrol_row_spacing);
                    place_patrol_camera(&mut transform, &mut fly_camera, &path, 0.0);
                    action.status = ActionStatus::Patrolling {
                        started_at: elapsed,
                        until: elapsed + seconds,
                    };
                    events.push(action_event(
                        action,
                        elapsed,
                        "started",
                        json!({
                            "seconds": seconds,
                            "rows": path.rows,
                            "path_length": path.length(),
                            "speed_per_second": path.length() / seconds.max(f64::EPSILON) as f32,
                        }),
                    ));
                }
            },
            ActionStatus::Patrolling { started_at, until } => {
                let Ok((mut transform, mut fly_camera)) = camera.get_single_mut() else {
                    continue;
                };
                // Re-derive the path from the live bounds so a catalog that
                // is still streaming widens the sweep instead of stranding
                // the camera outside it.
                let path = PatrolPath::new(&scene.bounds, patrol_row_spacing);
                let progress = ((elapsed - started_at) / (until - started_at).max(f64::EPSILON))
                    .clamp(0.0, 1.0) as f32;
                place_patrol_camera(&mut transform, &mut fly_camera, &path, progress);
                if elapsed >= until {
                    action.status = ActionStatus::Checking {
                        deadline: elapsed + grace,
                        baseline: None,
                        baseline_clock: None,
                    };
                }
            }
            ActionStatus::Holding {
                until,
                key,
                baseline,
            } if elapsed >= until => {
                keyboard.release(key);
                action.status = ActionStatus::Checking {
                    deadline: elapsed + grace,
                    baseline: Some(baseline),
                    baseline_clock: None,
                };
            }
            ActionStatus::Checking {
                deadline,
                baseline,
                baseline_clock,
            } => {
                let evaluation = evaluate_action(
                    action.kind,
                    expected_points,
                    &scene,
                    paused,
                    first_point_seen_while_streaming,
                    camera_targeted_first_point,
                    camera_position,
                    baseline,
                    billboard_controls.as_deref(),
                    stats.as_deref(),
                    video.controls.as_deref(),
                    baseline_clock,
                    patrol,
                );
                match evaluation {
                    AssertionResult::Passed(actual) => {
                        events.push(action_event(action, elapsed, "passed", actual));
                        action.status = ActionStatus::Done;
                    }
                    AssertionResult::Waiting(actual) if elapsed < deadline => {
                        let _ = actual;
                    }
                    AssertionResult::Waiting(actual) | AssertionResult::Failed(actual) => {
                        events.push(action_event(action, elapsed, "failed", actual));
                        action.status = ActionStatus::Done;
                        failure_count += 1;
                    }
                }
            }
            _ => {}
        }
    }
    harness.failures += failure_count;
    for mut event in events {
        event["scenario"] = Value::String(harness.config.scenario.clone());
        harness.emit(event);
    }
}

enum AssertionResult {
    Passed(Value),
    Waiting(Value),
    Failed(Value),
}

#[derive(Debug, Clone, Copy)]
struct PatrolProgress {
    distinct_loaded: usize,
    peak_loaded: usize,
}

/// Boustrophedon sweep through the catalog volume: rows along X spaced
/// along Y, in the plane through the bounds centre so images sit on every
/// side of the camera and cycle through all quality tiers as it passes.
#[derive(Debug, Clone, Copy)]
struct PatrolPath {
    minimum: Vec3,
    maximum: Vec3,
    rows: usize,
}

impl PatrolPath {
    fn new(bounds: &Bounds3, row_spacing: f32) -> Self {
        let minimum = Vec3::new(bounds.minimum.x, bounds.minimum.y, bounds.minimum.z);
        let maximum = Vec3::new(bounds.maximum.x, bounds.maximum.y, bounds.maximum.z);
        let extent_y = (maximum.y - minimum.y).max(0.0);
        let rows = ((extent_y / row_spacing.max(f32::EPSILON)).ceil() as usize + 1)
            .clamp(2, PATROL_MAX_ROWS);
        Self {
            minimum,
            maximum,
            rows,
        }
    }

    fn length(&self) -> f32 {
        let extent = self.maximum - self.minimum;
        self.rows as f32 * extent.x.max(0.0) + extent.y.max(0.0)
    }

    /// Camera position for `progress` in `0..=1` along the sweep: a
    /// continuous serpentine that zig-zags along X `rows` times while
    /// climbing Y, so consecutive frames are always a short step apart.
    fn position(&self, progress: f32) -> Vec3 {
        let extent = self.maximum - self.minimum;
        let progress = progress.clamp(0.0, 1.0);
        let row_progress = progress * self.rows as f32;
        let row = (row_progress.floor() as usize).min(self.rows - 1);
        let along = (row_progress - row as f32).min(1.0);
        let x_fraction = if row % 2 == 0 { along } else { 1.0 - along };
        Vec3::new(
            self.minimum.x + extent.x * x_fraction,
            self.minimum.y + extent.y * progress,
            self.minimum.z + extent.z * 0.5,
        )
    }
}

fn place_patrol_camera(
    transform: &mut Transform,
    fly_camera: &mut FlyCamera,
    path: &PatrolPath,
    progress: f32,
) {
    fly_camera.yaw = 0.0;
    fly_camera.pitch = 0.0;
    fly_camera.velocity = Vec3::ZERO;
    transform.translation = path.position(progress);
    transform.rotation = Quat::IDENTITY;
}

#[allow(clippy::too_many_arguments)]
fn evaluate_action(
    kind: ActionKind,
    expected_points: usize,
    scene: &ExplorerScene,
    paused: Option<bool>,
    first_point_seen_while_streaming: bool,
    camera_targeted_first_point: bool,
    camera_position: Option<Vec3>,
    baseline: Option<Vec3>,
    billboard_controls: Option<&BillboardControls>,
    stats: Option<&BillboardStats>,
    video_controls: Option<&VideoControlsState>,
    baseline_clock: Option<VideoClockBaseline>,
    patrol: PatrolProgress,
) -> AssertionResult {
    match kind {
        ActionKind::CatalogFirstPoint => compare_wait(
            first_point_seen_while_streaming && camera_targeted_first_point,
            json!({
                "expected_minimum": 1,
                "projected_points": scene.projection.points.len(),
                "first_point_seen_while_streaming": first_point_seen_while_streaming,
                "camera_targets_first_coordinate": camera_targeted_first_point,
            }),
        ),
        ActionKind::CatalogComplete => compare_wait(
            scene.projection.points.len() >= expected_points,
            json!({
                "expected_minimum": expected_points,
                "projected_points": scene.projection.points.len(),
                "projected_total": scene.projection.total,
            }),
        ),
        ActionKind::FirstTextureLoaded => compare_wait(
            stats.is_some_and(|value| value.loaded >= 1),
            json!({
                "expected_loaded_minimum": 1,
                "loaded": stats.map_or(0, |value| value.loaded),
                "in_flight": stats.map_or(0, |value| value.in_flight),
                "failed": stats.map_or(0, |value| value.failed),
            }),
        ),
        ActionKind::HoldAction { .. } => {
            let distance = camera_position
                .zip(baseline)
                .map(|(current, start)| current.distance(start));
            compare_wait(
                distance.is_some_and(|value| value >= 0.05),
                json!({"expected_distance_minimum": 0.05, "distance": distance}),
            )
        }
        ActionKind::TapPause { expected } => match paused {
            Some(actual) if actual == expected => AssertionResult::Passed(json!({
                "expected_paused": expected,
                "paused": actual,
            })),
            Some(actual) => AssertionResult::Waiting(json!({
                "expected_paused": expected,
                "paused": actual,
            })),
            None => AssertionResult::Failed(json!({"reason": "PauseMenuState missing"})),
        },
        ActionKind::TeleportRight => {
            if scene.projection.points.len() < 2 {
                return AssertionResult::Passed(json!({
                    "skipped": true,
                    "reason": "catalog has fewer than two points",
                }));
            }
            let distance = camera_position
                .zip(baseline)
                .map(|(current, start)| current.distance(start));
            compare_wait(
                distance.is_some_and(|value| value >= 0.01),
                json!({"expected_distance_minimum": 0.01, "distance": distance}),
            )
        }
        ActionKind::SetTextureBudget { value } => match (billboard_controls, stats) {
            // The budget is met in BYTES, not in image count: a lowered
            // slider has taken effect once the resident textures fit under
            // it, however many images that turns out to be.
            (Some(controls), Some(stats))
                if controls.texture_budget_mib == value
                    && stats.texture_bytes_used <= controls.texture_budget_bytes() =>
            {
                AssertionResult::Passed(json!({
                    "expected_mib": value,
                    "texture_budget_mib": controls.texture_budget_mib,
                    "texture_mib_used": stats.texture_bytes_used / (1024 * 1024),
                    "loaded": stats.loaded,
                }))
            }
            (Some(controls), Some(stats)) => AssertionResult::Waiting(json!({
                "expected_mib": value,
                "texture_budget_mib": controls.texture_budget_mib,
                "texture_mib_used": stats.texture_bytes_used / (1024 * 1024),
                "loaded": stats.loaded,
            })),
            _ => AssertionResult::Failed(json!({"reason": "billboard state missing"})),
        },
        ActionKind::StartFirstVideo => {
            let current = video_controls
                .zip(baseline_clock)
                .and_then(|(controls, baseline)| controls.clock(baseline.image_id))
                .map(|clock| clock.time_seconds());
            let baseline_seconds = baseline_clock.map(|baseline| baseline.time_seconds);
            let advanced = current
                .zip(baseline_seconds)
                .is_some_and(|(current, baseline)| current >= baseline + 0.05);
            compare_wait(
                advanced,
                json!({
                    "expected_clock_advance_minimum": 0.05,
                    "baseline_seconds": baseline_seconds,
                    "clock_seconds": current,
                }),
            )
        }
        ActionKind::ReprojectAxes => compare_wait(
            scene.projection.axis_labels
                == REPROJECT_EXPECTED_AXIS_LABELS.map(|label| Some(label.to_owned())),
            json!({
                "expected_axis_labels": REPROJECT_EXPECTED_AXIS_LABELS,
                "axis_labels": scene.projection.axis_labels,
            }),
        ),
        ActionKind::Patrol { .. } => {
            let expected_minimum = expected_points.min(PATROL_MIN_DISTINCT_LOADED);
            let actual = json!({
                "expected_distinct_loaded_minimum": expected_minimum,
                "distinct_loaded": patrol.distinct_loaded,
                "peak_loaded": patrol.peak_loaded,
                "loaded": stats.map_or(0, |value| value.loaded),
                "failed": stats.map_or(0, |value| value.failed),
            });
            // The sweep is over, so the count can no longer grow.
            if patrol.distinct_loaded >= expected_minimum {
                AssertionResult::Passed(actual)
            } else {
                AssertionResult::Failed(actual)
            }
        }
    }
}

fn compare_wait(passed: bool, actual: Value) -> AssertionResult {
    if passed {
        AssertionResult::Passed(actual)
    } else {
        AssertionResult::Waiting(actual)
    }
}

fn action_event(action: &ScheduledAction, elapsed: f64, phase: &str, details: Value) -> Value {
    json!({
        "type": "action",
        "scenario": null,
        "elapsed_seconds": elapsed,
        "scheduled_at_seconds": action.scheduled_at,
        "action": action.kind.name(),
        "phase": phase,
        "details": details,
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_telemetry(
    mut harness: ResMut<PerformanceHarness>,
    mut probe: ResMut<FrameStageProbe>,
    time: Res<Time<Real>>,
    scene: Res<ExplorerScene>,
    pause_menu: Option<Res<PauseMenuState>>,
    stats: Option<Res<BillboardStats>>,
    video_playback: Option<Res<VideoPlaybackState>>,
    video_controls: Option<Res<VideoControlsState>>,
    camera: Query<&Transform, With<FlyCamera>>,
    mut exit: EventWriter<AppExit>,
) {
    if harness.finished {
        return;
    }
    let elapsed = harness.elapsed();
    let frame_ms = time.delta_secs_f64().max(0.0) * 1_000.0;
    harness.frame_count += 1;
    harness.interval_frame_times_ms.push(frame_ms);
    harness.all_frame_times_ms.push(frame_ms);
    let update_started = probe.first_stamp();
    if let Some((stage_ms, previous_ended)) = harness.pending_update.take() {
        let outside_update_ms = update_started.map_or(0.0, |started| {
            started.duration_since(previous_ended).as_secs_f64() * 1_000.0
        });
        harness.recent_updates.insert(
            0,
            UpdateProbeRecord {
                stage_ms,
                outside_update_ms,
            },
        );
        harness.recent_updates.truncate(RECENT_UPDATE_RECORDS);
    }
    let (current_stages, current_ended) = probe.take();
    harness.pending_update = current_ended.map(|ended| (current_stages, ended));
    let context = FrameContext {
        frame_ms,
        elapsed_seconds: elapsed,
        projected_points: scene.projection.points.len(),
        loaded: stats.as_ref().map_or(0, |value| value.loaded),
        in_flight: stats.as_ref().map_or(0, |value| value.in_flight),
        upload_queue: stats.as_ref().map_or(0, |value| value.upload_queue),
        uploaded_textures: stats.as_ref().map_or(0, |value| value.uploaded_textures),
        uploaded_bytes: stats.as_ref().map_or(0, |value| value.uploaded_bytes),
        orientation_updated: stats.as_ref().map_or(0, |value| value.orientation_updated),
        video_decode_streams: video_playback
            .as_ref()
            .map_or(0, |value| value.stream_count()),
        recent_updates: harness.recent_updates.clone(),
    };
    if harness
        .interval_worst_frame
        .as_ref()
        .is_none_or(|worst| worst.frame_ms < frame_ms)
    {
        harness.interval_worst_frame = Some(context.clone());
    }
    if elapsed >= harness.config.warmup_seconds {
        harness.steady_frame_times_ms.push(frame_ms);
        record_worst_frame(&mut harness.steady_worst_frames, context);
    }

    if !harness.emitted_start {
        let timeline = harness
            .actions
            .iter()
            .map(|action| {
                json!({
                    "action": action.kind.name(),
                    "scheduled_at_seconds": action.scheduled_at,
                })
            })
            .collect::<Vec<_>>();
        let start_event = json!({
            "type": "start",
            "scenario": harness.config.scenario,
            "duration_seconds": harness.config.duration_seconds,
            "seed": harness.config.seed,
            "expected_points": harness.config.expected_points,
            "timeline": timeline,
        });
        harness.emit(start_event);
        harness.emitted_start = true;
    }

    if elapsed - harness.last_heartbeat >= harness.config.heartbeat_seconds {
        let interval = std::mem::take(&mut harness.interval_frame_times_ms);
        let sample_count = interval.len();
        let summary = frame_summary(&interval);
        let worst_frame = harness.interval_worst_frame.take();
        let camera_position = camera
            .get_single()
            .map(|transform| transform.translation.to_array())
            .ok();
        let heartbeat = json!({
            "type": "heartbeat",
            "scenario": harness.config.scenario,
            "elapsed_seconds": elapsed,
            "frame_count": harness.frame_count,
            "sample_count": sample_count,
            "fps": summary.fps,
            "frame_time_ms": summary.to_json(),
            "worst_frame": worst_frame.as_ref().map(FrameContext::to_json),
            "catalog": {
                "projected_points": scene.projection.points.len(),
                "projected_total": scene.projection.total,
            },
            "billboards": stats.as_ref().map(|value| json!({
                "entities": value.entity_count,
                "loaded": value.loaded,
                "in_flight": value.in_flight,
                "pending": value.pending,
                "failed": value.failed,
                "uploaded_textures": value.uploaded_textures,
                "uploaded_bytes": value.uploaded_bytes,
                "upload_queue": value.upload_queue,
            })),
            "video": {
                "clocks": video_controls.as_ref().map_or(0, |value| value.clock_count()),
                "decode_streams": video_playback.as_ref().map_or(0, |value| value.stream_count()),
            },
            "paused": pause_menu.as_ref().is_some_and(|value| value.paused),
            "camera_position": camera_position,
        });
        harness.emit(heartbeat);
        harness.last_heartbeat = elapsed;
    }

    if elapsed < harness.config.duration_seconds {
        return;
    }

    let unfinished = harness
        .actions
        .iter()
        .filter(|action| !matches!(action.status, ActionStatus::Done))
        .map(|action| action.kind.name())
        .collect::<Vec<_>>();
    harness.failures += unfinished.len();
    let all_frames = std::mem::take(&mut harness.all_frame_times_ms);
    let summary = frame_summary(&all_frames);
    let steady_frames = std::mem::take(&mut harness.steady_frame_times_ms);
    let steady_summary = frame_summary(&steady_frames);
    let worst_frames = std::mem::take(&mut harness.steady_worst_frames);
    let failures = harness.failures;
    let complete = json!({
        "type": "complete",
        "scenario": harness.config.scenario,
        "elapsed_seconds": elapsed,
        "frame_count": harness.frame_count,
        "failures": failures,
        "unfinished_actions": unfinished,
        "fps": summary.fps,
        "frame_time_ms": summary.to_json(),
        "warmup_seconds": harness.config.warmup_seconds,
        "steady_state": {
            "sample_count": steady_frames.len(),
            "fps": steady_summary.fps,
            "frame_time_ms": steady_summary.to_json(),
            "worst_frames": worst_frames.iter().map(FrameContext::to_json).collect::<Vec<_>>(),
        },
    });
    harness.emit(complete);
    harness.finished = true;
    exit.send(if failures == 0 {
        AppExit::Success
    } else {
        AppExit::error()
    });
}

/// Frame-time distribution plus the uniformity measures that percentiles
/// alone hide: how spread the times are, how often a frame doubles, and how
/// jerky consecutive frames feel.
#[derive(Default, Debug, PartialEq)]
struct FrameSummary {
    fps: f64,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    std_dev_ms: f64,
    /// Standard deviation over the mean; dimensionless so runs at different
    /// refresh rates compare directly.
    coefficient_of_variation: f64,
    /// Fraction of frames slower than `SPIKE_MEDIAN_MULTIPLIER` times the
    /// median.
    spike_fraction: f64,
    spike_count: usize,
    /// Mean absolute difference between consecutive frame times, in ms.
    /// Uniform stutter-free rendering keeps this near zero even when the
    /// overall mean is high.
    mean_consecutive_delta_ms: f64,
    /// Longest wall-clock run of consecutive spike frames, in ms.
    longest_spike_burst_ms: f64,
}

impl FrameSummary {
    fn to_json(&self) -> Value {
        json!({
            "mean": self.mean_ms,
            "p50": self.p50_ms,
            "p95": self.p95_ms,
            "p99": self.p99_ms,
            "max": self.max_ms,
            "std_dev": self.std_dev_ms,
            "coefficient_of_variation": self.coefficient_of_variation,
            "spike_fraction": self.spike_fraction,
            "spike_count": self.spike_count,
            "spike_threshold_multiplier": SPIKE_MEDIAN_MULTIPLIER,
            "mean_consecutive_delta": self.mean_consecutive_delta_ms,
            "longest_spike_burst": self.longest_spike_burst_ms,
        })
    }
}

/// Summarizes frame times given in frame order (consecutive-delta and burst
/// measures depend on the ordering).
fn frame_summary(samples: &[f64]) -> FrameSummary {
    if samples.is_empty() {
        return FrameSummary::default();
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len() as f64;
    let mean_ms = sorted.iter().sum::<f64>() / count;
    let variance = sorted
        .iter()
        .map(|value| (value - mean_ms).powi(2))
        .sum::<f64>()
        / count;
    let std_dev_ms = variance.sqrt();
    let p50_ms = percentile(&sorted, 0.50);
    let spike_threshold_ms = p50_ms * SPIKE_MEDIAN_MULTIPLIER;
    let spike_count = sorted
        .iter()
        .filter(|value| **value > spike_threshold_ms)
        .count();
    let mean_consecutive_delta_ms = samples
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .sum::<f64>()
        / (samples.len() - 1).max(1) as f64;
    let mut longest_spike_burst_ms: f64 = 0.0;
    let mut current_burst_ms = 0.0;
    for value in samples {
        if *value > spike_threshold_ms {
            current_burst_ms += value;
            longest_spike_burst_ms = longest_spike_burst_ms.max(current_burst_ms);
        } else {
            current_burst_ms = 0.0;
        }
    }
    FrameSummary {
        fps: if mean_ms > f64::EPSILON {
            1_000.0 / mean_ms
        } else {
            0.0
        },
        mean_ms,
        p50_ms,
        p95_ms: percentile(&sorted, 0.95),
        p99_ms: percentile(&sorted, 0.99),
        max_ms: *sorted.last().unwrap_or(&0.0),
        std_dev_ms,
        coefficient_of_variation: if mean_ms > f64::EPSILON {
            std_dev_ms / mean_ms
        } else {
            0.0
        },
        spike_fraction: spike_count as f64 / count,
        spike_count,
        mean_consecutive_delta_ms,
        longest_spike_burst_ms,
    }
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(seed: u64) -> HarnessConfig {
        HarnessConfig {
            scenario: "test".to_owned(),
            duration_seconds: 300.0,
            seed,
            output_path: PathBuf::from("unused.jsonl"),
            expected_points: 1,
            expect_video: false,
            heartbeat_seconds: 1.0,
            assertion_grace_seconds: 5.0,
            warmup_seconds: 3.0,
            timeline: TimelineKind::Interactive,
        }
    }

    fn patrol_config(duration_seconds: f64) -> HarnessConfig {
        HarnessConfig {
            timeline: TimelineKind::Patrol,
            duration_seconds,
            ..config(3)
        }
    }

    fn bounds(minimum: [f32; 3], maximum: [f32; 3]) -> Bounds3 {
        let minimum = Vec3::from_array(minimum);
        let maximum = Vec3::from_array(maximum);
        Bounds3 {
            minimum,
            maximum,
            center: (minimum + maximum) * 0.5,
            max_extent: 1.0,
        }
    }

    #[test]
    fn timeline_fuzz_is_repeatable_and_seeded() {
        let first = build_timeline(&config(7));
        let again = build_timeline(&config(7));
        let other = build_timeline(&config(8));
        let times = |timeline: &[ScheduledAction]| {
            timeline
                .iter()
                .map(|action| action.scheduled_at)
                .collect::<Vec<_>>()
        };
        assert_eq!(times(&first), times(&again));
        assert_ne!(times(&first), times(&other));
    }

    #[test]
    fn frame_summary_reports_expected_percentiles() {
        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let summary = frame_summary(&samples);
        assert_eq!(summary.mean_ms, 30.0);
        assert_eq!(summary.p50_ms, 30.0);
        assert_eq!(summary.p95_ms, 50.0);
        assert_eq!(summary.max_ms, 50.0);
    }

    #[test]
    fn frame_summary_measures_uniformity_in_frame_order() {
        let uniform = frame_summary(&[4.0; 100]);
        assert_eq!(uniform.std_dev_ms, 0.0);
        assert_eq!(uniform.coefficient_of_variation, 0.0);
        assert_eq!(uniform.spike_count, 0);
        assert_eq!(uniform.mean_consecutive_delta_ms, 0.0);
        assert_eq!(uniform.longest_spike_burst_ms, 0.0);

        // Two consecutive spikes and one isolated spike among steady frames.
        let mut samples = vec![4.0; 20];
        samples[5] = 12.0;
        samples[6] = 10.0;
        samples[15] = 9.0;
        let stuttering = frame_summary(&samples);
        assert_eq!(stuttering.p50_ms, 4.0);
        assert_eq!(stuttering.spike_count, 3);
        assert!((stuttering.spike_fraction - 0.15).abs() < 1e-12);
        assert_eq!(stuttering.longest_spike_burst_ms, 22.0);
        assert!(stuttering.std_dev_ms > 0.0);
        assert!(stuttering.mean_consecutive_delta_ms > 0.0);

        // The same samples in a different order keep the distribution but
        // change the ordering-sensitive measures.
        let mut clustered = samples.clone();
        clustered.sort_by(f64::total_cmp);
        let sorted = frame_summary(&clustered);
        assert_eq!(sorted.spike_count, stuttering.spike_count);
        assert_eq!(sorted.std_dev_ms, stuttering.std_dev_ms);
        assert_eq!(sorted.longest_spike_burst_ms, 31.0);
        assert!(sorted.mean_consecutive_delta_ms < stuttering.mean_consecutive_delta_ms);
    }

    #[test]
    fn worst_frames_keep_slowest_first_and_bounded() {
        let mut worst = Vec::new();
        for frame_ms in [3.0, 9.0, 1.0, 7.0, 5.0, 11.0, 2.0, 8.0, 6.0, 4.0, 10.0] {
            record_worst_frame(
                &mut worst,
                FrameContext {
                    frame_ms,
                    elapsed_seconds: 0.0,
                    projected_points: 0,
                    loaded: 0,
                    in_flight: 0,
                    upload_queue: 0,
                    uploaded_textures: 0,
                    uploaded_bytes: 0,
                    orientation_updated: 0,
                    video_decode_streams: 0,
                    recent_updates: Vec::new(),
                },
            );
        }
        let times = worst.iter().map(|frame| frame.frame_ms).collect::<Vec<_>>();
        assert_eq!(times, vec![11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn patrol_timeline_sweeps_until_just_before_the_end() {
        let timeline = build_timeline(&patrol_config(60.0));
        let patrol = timeline
            .iter()
            .find(|action| matches!(action.kind, ActionKind::Patrol { .. }))
            .expect("patrol action");
        let ActionKind::Patrol { seconds } = patrol.kind else {
            unreachable!()
        };
        assert!(patrol.scheduled_at >= 1.5);
        assert!((patrol.scheduled_at + seconds - 59.0).abs() < 1e-9);
        assert!(!timeline
            .iter()
            .any(|action| matches!(action.kind, ActionKind::HoldAction { .. })));
    }

    #[test]
    fn patrol_path_covers_bounds_without_jumping_between_rows() {
        let path = PatrolPath::new(&bounds([-10.0, -4.0, -2.0], [10.0, 4.0, 2.0]), 2.0);
        assert_eq!(path.rows, 5);
        assert_eq!(path.position(0.0), Vec3::new(-10.0, -4.0, 0.0));
        assert_eq!(path.position(1.0), Vec3::new(10.0, 4.0, 0.0));
        // Row ends meet: the end of row 0 is the start of row 1.
        let end_of_first_row = path.position(0.2 - 1e-6);
        let start_of_second_row = path.position(0.2);
        assert!(end_of_first_row.distance(start_of_second_row) < 1e-3);
        assert!((end_of_first_row.x - 10.0).abs() < 1e-3);
        // Row 1 runs back toward the minimum while Y keeps climbing.
        let middle_of_second_row = path.position(0.3);
        assert!((middle_of_second_row.x - 0.0).abs() < 1e-3);
        assert!((middle_of_second_row.y - (-1.6)).abs() < 1e-3);
        // Consecutive samples are always close: no jumps anywhere along the
        // sweep.
        let mut previous = path.position(0.0);
        for step in 1..=1000 {
            let current = path.position(step as f32 / 1000.0);
            assert!(current.distance(previous) < 0.2, "jump at step {step}");
            previous = current;
        }
        assert_eq!(path.length(), 5.0 * 20.0 + 8.0);

        let single_point = PatrolPath::new(&bounds([0.0; 3], [0.0; 3]), 2.0);
        assert_eq!(single_point.rows, 2);
        assert_eq!(single_point.position(0.5), Vec3::ZERO);
    }
}
