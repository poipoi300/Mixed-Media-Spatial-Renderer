mod animations_panel;
mod context_menu;
mod control_panel;
mod controls_sheet;
mod edit_panel;
mod folder_panel;
mod input_bindings;
mod look_crosshair;
mod search_panel;
mod selection_box;
mod text_entry;

pub use animations_panel::{
    handle_animation_buttons, update_animation_button_colors, update_animation_panels,
    update_animation_text, Animation, AnimationSettings,
};
pub use context_menu::{
    ContextMenu, ContextMenuItem, ContextMenuModel, ContextMenuOption, ContextMenuPlugin,
    ContextMenuSection, ContextMenuSystems,
};

pub use control_panel::{
    control_values_from_assignments, handle_control_buttons, handle_control_dropdown_scroll,
    handle_control_keyboard, parse_control_assignment, rebuild_control_widgets,
    update_control_button_colors, update_control_panels, update_control_text, ControlAction,
    ControlPanelState, ControlWidgetButton, ControlWidgetPanel, ControlWidgetRoot,
    ControlWidgetText, FocusOwner, PendingSubmit,
};

pub use folder_panel::{
    handle_folder_buttons, update_folder_button_colors, update_folder_panels, update_folder_text,
    FolderControls, FolderDisplay, FolderRequest,
};

pub use controls_sheet::{
    handle_controls_sheet_buttons, scroll_controls_sheet, update_controls_sheet,
    update_controls_sheet_colors, ControlsSheetState,
};
pub use edit_panel::{handle_edit_buttons, update_edit_button_colors, EditControls, EditRequest};
pub use input_bindings::{
    capture_binding, update_control_input_state, update_typing_focus, Action, ActionCategory,
    ActionContext, BindingConflict, BindingEditor, BindingRefusal, BindingSlot, BindingSource,
    Chord, ControlBindings, ControlInput, ControlInputState, DefaultBinding, Gesture, HoldMode,
    Input, Modifier, Modifiers, Slots, TypingFocus, SLOT_COUNT,
};
pub use look_crosshair::{spawn_look_crosshair, LookCrosshair};
pub use search_panel::{
    apply_search_typing, handle_search_buttons, update_search_button_colors, update_search_panels,
    update_search_text, SearchControls, SearchRequest,
};
pub use selection_box::{
    spawn_selection_box, update_selection_box, SelectionBox, SelectionBoxNode,
};
pub use text_entry::{
    handle_text_entry_keyboard, update_rename_prompt, RenamePromptPart, TextEntry, TextEntryBox,
    TextEntryTarget,
};

use std::collections::{HashMap, VecDeque};

use bevy::prelude::*;
use bevy::ui::{FocusPolicy, RelativeCursorPosition};

use animations_panel::spawn_animations_screen;
use control_panel::spawn_control_panel_pill;
use controls_sheet::spawn_controls_sheet;
use edit_panel::spawn_edit_pill;
use folder_panel::spawn_folder_pill;
use search_panel::spawn_search_pill;
use text_entry::spawn_rename_prompt;

const MIN_BASE_SPEED: f32 = 0.05;
const MAX_BASE_SPEED: f32 = 5_000.0;
const SPEED_BUTTON_FACTOR: f32 = 1.25;
const SCROLL_SPEED_FACTOR: f32 = 0.08;
const SHIFT_SPEED_MULTIPLIER: f32 = 3.0;
const CRUISE_RISE_PER_SECOND: f32 = 0.15;
const CRUISE_DECAY_PER_SECOND: f32 = 1.8;
const CRUISE_ALIGNMENT_THRESHOLD: f32 = -0.25;
/// Peak speed boost added at full cruise pressure: effective speed is
/// `base_speed * (1 + cruise_pressure * CRUISE_SPEED_GAIN)`.
const CRUISE_SPEED_GAIN: f32 = 21.0;
const CLUSTER_GRID_FACTOR: f32 = 4.0;
const PERF_SAMPLE_COUNT: usize = 120;
const PERF_GRAPH_BARS: usize = 40;
const PERF_HISTOGRAM_BINS: usize = 12;
/// Weight applied to off-axis distance when picking the nearest position in
/// an axis direction, so a same-row neighbor beats a nearer diagonal one.
const DIRECTIONAL_ORTHOGONAL_WEIGHT: f32 = 2.0;
/// VRAM budget bounds for resident billboard textures, in MiB.
///
/// This is what bounds the cache now that every image is decoded at source
/// resolution: sizes differ ~100x across a catalog, so a count of images
/// cannot bound the memory that actually runs out. The floor holds a few
/// large textures; the ceiling exceeds current consumer VRAM so the slider
/// is never the binding constraint on a big card.
pub const MIN_TEXTURE_BUDGET_MIB: u32 = 512;
pub const MAX_TEXTURE_BUDGET_MIB: u32 = 24576;
/// ~350 average-sized BC7 textures, a fraction of a 32 GiB card, leaving
/// room for the render targets and everything else on the GPU. Exported so
/// the CLI default and the slider cannot drift apart.
pub const DEFAULT_TEXTURE_BUDGET_MIB: u32 = 6144;
/// Slider granularity, so a drag lands on round values.
const TEXTURE_BUDGET_STEP_MIB: u32 = 256;
const MIN_RENDER_SCALE: f32 = 0.25;
const MAX_RENDER_SCALE: f32 = 1.5;
const RENDER_SCALE_STEP: f32 = 0.05;
const AUDIO_VOLUME_STEP: f32 = 0.05;
/// Number of folder rows the start menu can display individually; additional
/// folders are summarized in an overflow line.
pub const START_MENU_FOLDER_ROWS: usize = 8;

#[derive(Resource)]
pub struct NavigationSettings {
    pub base_speed: f32,
    /// Scene-scale reference (roughly the coordinate spacing) used to size
    /// view-dependent thresholds such as billboard LOD and eviction distances.
    /// It is a static property of the loaded scene and does not affect speed.
    pub reference_distance: f32,
    pub shift_speed_multiplier: f32,
    pub cruise_pressure: f32,
    pub speed_pill_expanded: bool,
    pub status: NavigationStatus,
    navigation_request: Option<NavigationRequest>,
}

#[derive(Resource)]
pub struct NavigationTargets {
    positions: Vec<Vec3>,
    grid: SpatialGrid,
    clusters: Vec<Vec3>,
}

#[derive(Debug, Clone)]
struct SpatialGrid {
    cell_size: f32,
    cells: HashMap<IVec3, Vec<usize>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NavigationStatus {
    pub cruise_pressure: f32,
    pub velocity_multiplier: f32,
    pub effective_speed: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationRequest {
    TeleportNearest,
    SnapOrbitNearestCluster,
    FocusNearestImage,
    JumpNextVisibleCluster,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BillboardFacingAxis {
    X,
    Y,
    Z,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisGizmoFace {
    PositiveX,
    NegativeX,
    PositiveY,
    NegativeY,
    PositiveZ,
    NegativeZ,
}

#[derive(Resource)]
pub struct BillboardFacingSettings {
    pub axis: BillboardFacingAxis,
}

#[derive(Resource)]
pub struct BillboardControls {
    pub max_texture_side: u32,
    /// VRAM ceiling, in MiB, for resident billboard textures.
    pub texture_budget_mib: u32,
    pub billboards_pill_expanded: bool,
    pub show_coordinates: bool,
    /// Manually dragged billboards snap to the coordinate-spacing grid when
    /// the drag is released.
    pub snap_to_grid: bool,
    /// The billboard under the pointer grows a little.
    pub grow_on_hover: bool,
}

#[derive(Resource, Default)]
pub struct AxisGizmoState {
    selected_face: Option<AxisGizmoFace>,
    requested_face: Option<AxisGizmoFace>,
    hovered_face: Option<AxisGizmoFace>,
    requested_direction: Option<Vec3>,
}

#[derive(Resource, Default)]
pub struct BillboardStats {
    pub entity_count: usize,
    pub loaded: usize,
    pub failed: usize,
    pub in_flight: usize,
    pub pending: usize,
    /// Pending points the scheduler could actually start right now. Excludes
    /// points deferred as ineligible (behind the camera, beyond bounds),
    /// which stay counted in `pending` but will not be scheduled until the
    /// view changes — so only this number means "work is waiting on us".
    pub schedulable_pending: usize,
    pub uploaded_textures: usize,
    pub uploaded_bytes: usize,
    pub upload_queue: usize,
    /// VRAM the resident billboard textures occupy, and the ceiling they are
    /// held under. This is what bounds the cache now that every image is
    /// decoded at source resolution.
    pub texture_bytes_used: usize,
    pub texture_budget_bytes: usize,
    pub orientation_updated: usize,
    pub orientation_checked: usize,
    pub orientation_skipped: usize,
    /// Points near enough and in front of the camera to be worth showing,
    /// and how many of those actually have a texture.
    ///
    /// This is what the user perceives, and it is deliberately not `loaded`:
    /// residency counts every texture in memory including the ones behind
    /// the camera, so they rank a run that fills the view worse than one
    /// that hoards textures nobody can see.
    pub visible_billboards: usize,
    pub visible_textured: usize,
    /// How alpha classification split this run's encode work: surfaces
    /// taking the fast opaque BC7 preset versus those needing the
    /// alpha-aware one, with the encode time each group actually cost.
    /// Cumulative over the run.
    pub encode_opaque_surfaces: usize,
    pub encode_mixed_surfaces: usize,
    pub encode_opaque_texels: u64,
    pub encode_mixed_texels: u64,
    pub encode_opaque_nanos: u64,
    pub encode_mixed_nanos: u64,
    /// Cumulative over the run.
    pub cache_churn: BillboardCacheChurn,
}

/// How much work the texture cache has spent displacing its own residents.
///
/// With the budget full, residents are evicted to admit better-placed
/// images; that is the cache doing its job. Load/unload at a camera that is
/// not moving is not, and shows up here as `reloads` climbing while nothing
/// in the view changes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BillboardCacheChurn {
    /// Residents evicted to make room for an arriving image.
    pub evicted_for_arrivals: usize,
    /// Residents evicted because the cache, counting decodes still in
    /// flight, was over its budget.
    pub evicted_over_budget: usize,
    /// Successfully decoded images dropped on arrival for lack of room.
    pub discarded_decodes: usize,
    /// Billboards admitted for an image evicted earlier in the same catalog.
    pub reloads: usize,
    /// The part of `reloads` that took free room rather than displacing a
    /// resident.
    pub reloads_into_free_room: usize,
}

impl BillboardCacheChurn {
    pub fn evictions(self) -> usize {
        self.evicted_for_arrivals + self.evicted_over_budget
    }

    /// What accumulated after `earlier` was read, for a measurement window.
    pub fn since(self, earlier: Self) -> Self {
        Self {
            evicted_for_arrivals: self
                .evicted_for_arrivals
                .saturating_sub(earlier.evicted_for_arrivals),
            evicted_over_budget: self
                .evicted_over_budget
                .saturating_sub(earlier.evicted_over_budget),
            discarded_decodes: self
                .discarded_decodes
                .saturating_sub(earlier.discarded_decodes),
            reloads: self.reloads.saturating_sub(earlier.reloads),
            reloads_into_free_room: self
                .reloads_into_free_room
                .saturating_sub(earlier.reloads_into_free_room),
        }
    }
}

#[derive(Resource, Default)]
pub struct DebugSettings {
    pub debug_pill_expanded: bool,
}

/// Durations the benchmark button cycles through. A short run answers "is
/// this fast right now", the longer ones cover enough camera travel for
/// loading, refresh and eviction to all be exercised.
pub const BENCHMARK_DURATIONS_SECONDS: [u64; 3] = [15, 30, 60];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BenchmarkPhase {
    #[default]
    Idle,
    Recording,
    Complete,
}

/// One update stage's contribution to the frame, as shown in the report.
#[derive(Debug, Clone)]
pub struct BenchmarkStageRow {
    pub stage: String,
    pub mean_ms: f64,
    pub worst_ms: f64,
    /// Mean stage time as a fraction of the mean frame time.
    pub share: f64,
}

/// What a finished benchmark shows on screen. The full record — every stage,
/// the worst frames with their breakdowns, hardware counters — goes to the
/// JSON file named by `report_path`.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub summary: String,
    pub bottleneck: &'static str,
    pub stages: Vec<BenchmarkStageRow>,
    pub fps: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    pub sampled_frames: usize,
    pub decode_busy_fraction: f32,
    pub starved_fraction: f32,
    pub upload_mib_per_second: f32,
    /// Where the full report landed, or why it could not be written.
    pub report_path: Result<std::path::PathBuf, String>,
}

#[derive(Resource)]
pub struct BenchmarkControls {
    pub phase: BenchmarkPhase,
    /// 0..=1 while recording.
    pub progress: f32,
    pub report: Option<BenchmarkReport>,
    duration_index: usize,
    start_requested: bool,
    /// Overrides the cycled duration for one run.
    requested_duration: Option<std::time::Duration>,
}

impl Default for BenchmarkControls {
    fn default() -> Self {
        Self {
            phase: BenchmarkPhase::Idle,
            progress: 0.0,
            report: None,
            duration_index: 0,
            start_requested: false,
            requested_duration: None,
        }
    }
}

impl BenchmarkControls {
    pub fn duration_seconds(&self) -> u64 {
        BENCHMARK_DURATIONS_SECONDS[self.duration_index]
    }

    fn cycle_duration(&mut self) {
        self.duration_index = (self.duration_index + 1) % BENCHMARK_DURATIONS_SECONDS.len();
    }

    fn request_start(&mut self) {
        self.start_requested = true;
        self.report = None;
        self.progress = 0.0;
    }

    /// Requests a run of an explicit length, independent of the button's
    /// duration cycle. Used by `--benchmark <seconds>`.
    pub fn request_start_for(&mut self, duration: std::time::Duration) {
        self.request_start();
        self.requested_duration = Some(duration);
    }

    /// Takes a pending start request, so the benchmark system owns the
    /// transition out of `Idle` and a click cannot start two runs.
    pub fn take_start_request(&mut self) -> Option<std::time::Duration> {
        if !std::mem::take(&mut self.start_requested) {
            return None;
        }
        Some(
            self.requested_duration
                .take()
                .unwrap_or_else(|| std::time::Duration::from_secs(self.duration_seconds())),
        )
    }

    pub fn status_line(&self) -> String {
        match self.phase {
            BenchmarkPhase::Idle => {
                format!("Benchmark {} s   idle", self.duration_seconds())
            }
            BenchmarkPhase::Recording => {
                format!("Recording {:>3.0}%", self.progress * 100.0)
            }
            BenchmarkPhase::Complete => self
                .report
                .as_ref()
                .map(|report| report.summary.clone())
                .unwrap_or_else(|| "Complete".to_owned()),
        }
    }

    /// Multi-line detail for the expanded panel.
    pub fn detail_lines(&self) -> String {
        let Some(report) = self.report.as_ref() else {
            return match self.phase {
                BenchmarkPhase::Recording => {
                    "Fly through the catalog while this records.".to_owned()
                }
                _ => "Run a benchmark to profile this catalog.".to_owned(),
            };
        };
        let mut lines = format!(
            "frames {:>6}   fps {:>6.1}
p95 {:>6.2} ms   p99 {:>6.2} ms
decode busy {:>4.0}%  starved {:>4.0}%
upload {:>6.1} MiB/s

limit: {}

top stages (mean / worst ms)",
            report.sampled_frames,
            report.fps,
            report.p95_ms,
            report.p99_ms,
            report.decode_busy_fraction * 100.0,
            report.starved_fraction * 100.0,
            report.upload_mib_per_second,
            report.bottleneck,
        );
        for row in &report.stages {
            lines.push_str(&format!(
                "\n{:<22} {:>6.2} {:>6.2}",
                truncate_stage(&row.stage),
                row.mean_ms,
                row.worst_ms
            ));
        }
        match &report.report_path {
            Ok(path) => lines.push_str(&format!("\n\nsaved {}", path.display())),
            Err(error) => lines.push_str(&format!("\n\nnot saved: {error}")),
        }
        lines
    }
}

/// Keeps the stage column aligned in the fixed-width panel.
fn truncate_stage(stage: &str) -> String {
    const MAX: usize = 22;
    if stage.len() <= MAX {
        return stage.to_owned();
    }
    format!("{}...", &stage[..MAX - 3])
}

#[derive(Resource)]
pub struct PerformanceMetrics {
    frame_times_ms: VecDeque<f32>,
    max_samples: usize,
    pub last_frame_ms: f32,
    pub fps: f32,
    pub metrics_pill_expanded: bool,
}

/// A level in the pause menu hierarchy. `Escape` and the on-screen back
/// controls both step back exactly one level, so the traversal order here is
/// the single source of truth for that structure.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PauseScreen {
    #[default]
    Main,
    Settings,
    CatalogOptions,
    Controls,
    Animations,
}

#[derive(Resource, Default)]
pub struct PauseMenuState {
    pub paused: bool,
    pub screen: PauseScreen,
}

/// Snapshot of where this frame's pointer input belongs, sampled after
/// `bevy_ui` focus runs and before any gameplay click handler. World-space
/// click/drag systems consult this instead of re-deriving UI state so a click
/// consumed by the UI (buttons, panels, open menus) can never also reach the
/// 3D scene — including on the exact frame a menu closes.
#[derive(Resource, Default, Clone, Copy)]
pub struct UiInputCapture {
    /// The cursor is over an interactive or panel UI node this frame.
    pub pointer_over_ui: bool,
    /// The pause or start menu was open when this frame's input was sampled;
    /// stays true for the whole frame even if a button press closes the menu
    /// later in the same frame.
    pub menu_open: bool,
    /// A right-click menu was open when this frame's input was sampled; a
    /// press anywhere then belongs to it (outside it, the press dismisses
    /// it).
    pub context_menu_open: bool,
}

impl UiInputCapture {
    /// True when a click at the current cursor position belongs to the UI
    /// rather than the 3D world.
    pub fn blocks_world_clicks(&self) -> bool {
        self.pointer_over_ui || self.menu_open || self.context_menu_open
    }
}

/// Runs in `PreUpdate` after `UiSystem::Focus` so every `Update` system sees
/// a consistent snapshot of this frame's UI pointer ownership.
pub fn update_ui_input_capture(
    mut capture: ResMut<UiInputCapture>,
    pause_menu: Res<PauseMenuState>,
    interactions: Query<&Interaction>,
) {
    capture.pointer_over_ui = interactions
        .iter()
        .any(|interaction| *interaction != Interaction::None);
    // The catalog options screen only opens as a child of the pause menu, so
    // `paused` alone covers every menu level.
    capture.menu_open = pause_menu.paused;
    // Each open right-click menu sets this again after this system runs.
    capture.context_menu_open = false;
}

#[derive(Resource)]
pub struct RenderResolutionSettings {
    scale: f32,
    pending_resolution: bool,
}

/// Master volume for video audio, set from the pause-menu settings slider and
/// applied to the viewer's audio sink each frame.
#[derive(Resource)]
pub struct AudioSettings {
    volume: f32,
}

/// App-wide video playback settings from the pause-menu settings screen.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct PlaybackSettings {
    /// Videos open where they were left, as MPC-HC's "Remember file
    /// position" does; off, they always open at the start.
    pub remember_position: bool,
}

impl Default for PlaybackSettings {
    fn default() -> Self {
        Self {
            remember_position: true,
        }
    }
}

/// App-wide settings for the view, from the pause-menu settings screen.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct ViewSettings {
    /// Each catalog view opens with the camera where it was left there:
    /// position, direction and base speed.
    pub remember_camera: bool,
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self {
            remember_camera: true,
        }
    }
}

/// Whether the interface (panels, the axis gizmo, video control strips)
/// is hidden, for a clear view of the scene. Menus the user opens still
/// show. Not kept between sessions.
#[derive(Resource, Default)]
pub struct HudVisibility {
    pub hidden: bool,
}

/// The panels around the edge of the window, hidden with the interface.
#[derive(Component)]
pub struct HudPanels;

/// Hides or shows the interface when its key is pressed outside a menu.
pub fn toggle_hud(
    input: ControlInput,
    pause_menu: Res<PauseMenuState>,
    mut hud: ResMut<HudVisibility>,
) {
    if !pause_menu.paused && input.just_pressed(Action::ToggleHud) {
        hud.hidden = !hud.hidden;
    }
}

pub fn update_hud_panels(
    hud: Res<HudVisibility>,
    mut panels: Query<&mut Visibility, With<HudPanels>>,
) {
    let visibility = if hud.hidden {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut panel in &mut panels {
        panel.set_if_neq(visibility);
    }
}

#[derive(Resource)]
pub struct StartMenuState {
    pub roots: Vec<String>,
    pub status: String,
    /// A catalog load is running on a background thread; interactive start
    /// menu actions are disabled until it completes.
    pub loading: bool,
    /// A native folder picker dialog is open on a background thread.
    pub picking_folder: bool,
    add_folder_requested: bool,
    load_requested: bool,
    use_current_requested: bool,
    clear_requested: bool,
    reset_layout_requested: bool,
    remove_requested: Option<usize>,
}

#[derive(Component, Clone, Copy)]
pub enum ViewerUiButton {
    Slower,
    Faster,
    ToggleNavigationPill,
    FocusNearestImage,
    JumpNextVisibleCluster,
    ToggleControlPanelPill,
    SelectControlDropdownOption(usize),
    SliceDepthLower,
    SliceDepthHigher,
    ToggleBillboardsPill,
    SetBillboardAxis(BillboardFacingAxis),
    ToggleBillboardCoordinates,
    ToggleBillboardGridSnap,
    ToggleBillboardHoverGrowth,
    TextureLower,
    TextureHigher,
    SetTextureBudgetFromSlider,
    ToggleFolderPill,
    OpenAllFolders,
    CloseAllFolders,
    ToggleFolderTags,
    ToggleFolderCloseIcons,
    ToggleFolderBackgrounds,
    UndoArrangement,
    RedoArrangement,
    ToggleSearchPill,
    FocusSearch,
    SelectSearchMatches,
    NextSearchMatch,
    ToggleDebugPill,
    StartBenchmark,
    CycleBenchmarkDuration,
    TogglePerformancePill,
    PauseResume,
    PauseSettings,
    PauseBack,
    PauseCatalogOptions,
    PauseControls,
    PauseAnimations,
    PauseQuit,
    SetResolutionScaleFromSlider,
    SetAudioVolumeFromSlider,
    ToggleRememberPlaybackPosition,
    ToggleRememberCamera,
    ToggleAnimationsDisabled,
    ToggleAnimation(Animation),
    SetAnimationScaleFromSlider,
    StartAddFolder,
    StartClearFolders,
    StartLoadFolders,
    StartUseCurrentCatalog,
    StartResetLayout,
    StartRemoveFolder(usize),
}

#[derive(Component, Clone, Copy)]
pub enum ViewerUiText {
    SpeedSummary,
    SpeedDetails,
    ControlPanelSummary,
    ControlPanelStats,
    ControlDropdownTitle,
    ControlDropdownOption(usize),
    BillboardsSummary,
    BillboardsDetails,
    BillboardCoordinatesMark,
    BillboardGridSnapMark,
    BillboardHoverGrowthMark,
    BillboardTextureBudgetSummary,
    FolderSummary,
    FolderTagsMark,
    FolderCloseIconsMark,
    FolderBackgroundsMark,
    SearchSummary,
    SearchQuery,
    SearchResults,
    DebugSummary,
    BenchmarkStatus,
    BenchmarkDetails,
    PerformanceSummary,
    PerformanceDetails,
    ResolutionSummary,
    AudioVolumeSummary,
    RememberPlaybackPositionMark,
    RememberCameraMark,
    AnimationsDisabledMark,
    AnimationScaleSummary,
    AnimationMark(Animation),
    StartRoots,
    StartStatus,
    StartFolderPath(usize),
}

#[derive(Component, Clone, Copy)]
pub enum ViewerUiPanel {
    NavigationPill,
    NavigationExpanded,
    ControlPanelPill,
    ControlPanelExpanded,
    ControlDropdown,
    ControlDropdownOption(usize),
    BillboardsPill,
    BillboardsExpanded,
    BillboardTextureBudgetFill,
    FolderPill,
    FolderExpanded,
    SearchPill,
    SearchExpanded,
    EditPill,
    DebugPill,
    DebugExpanded,
    PerformancePill,
    PerformanceExpanded,
    PauseOverlay,
    PauseMain,
    PauseSettings,
    PauseControls,
    PauseAnimations,
    ResolutionScaleFill,
    AudioVolumeFill,
    AnimationScaleFill,
    StartOverlay,
    StartFolderRow(usize),
}

#[derive(Component, Clone, Copy)]
pub enum PerfGraphBar {
    Time(usize),
    Histogram(usize),
}

pub type ButtonInteractionQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Interaction, &'static ViewerUiButton),
    (Changed<Interaction>, With<Button>),
>;
pub type UiTextQuery<'w, 's> = Query<'w, 's, (&'static mut Text, &'static ViewerUiText)>;
pub type UiButtonColorQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static ViewerUiButton,
        &'static Interaction,
        &'static mut BackgroundColor,
    ),
    With<Button>,
>;
pub type UiPanelQuery<'w, 's> = Query<'w, 's, (&'static mut Node, &'static ViewerUiPanel)>;
pub type PerfGraphQuery<'w, 's> = Query<'w, 's, (&'static mut Node, &'static PerfGraphBar)>;

impl NavigationSettings {
    pub fn new(base_speed: f32, reference_distance: f32) -> Self {
        Self {
            base_speed: base_speed.clamp(MIN_BASE_SPEED, MAX_BASE_SPEED),
            reference_distance: reference_distance.max(1.0),
            shift_speed_multiplier: SHIFT_SPEED_MULTIPLIER,
            speed_pill_expanded: false,
            status: NavigationStatus {
                velocity_multiplier: 1.0,
                effective_speed: base_speed,
                ..default()
            },
            cruise_pressure: 0.0,
            navigation_request: None,
        }
    }

    pub fn apply_scroll_delta(&mut self, scroll_delta: f32) {
        let factor = 1.0 + scroll_delta * SCROLL_SPEED_FACTOR;
        if factor > 0.0 {
            self.base_speed = (self.base_speed * factor).clamp(MIN_BASE_SPEED, MAX_BASE_SPEED);
        }
    }

    pub fn retarget_reference_distance(&mut self, reference_distance: f32) {
        self.reference_distance = reference_distance.max(1.0);
    }

    /// Sets the base speed, within the range the speed buttons keep to.
    pub fn set_base_speed(&mut self, base_speed: f32) {
        self.base_speed = base_speed.clamp(MIN_BASE_SPEED, MAX_BASE_SPEED);
    }

    fn slow_down(&mut self) {
        self.base_speed =
            (self.base_speed / SPEED_BUTTON_FACTOR).clamp(MIN_BASE_SPEED, MAX_BASE_SPEED);
    }

    fn speed_up(&mut self) {
        self.base_speed =
            (self.base_speed * SPEED_BUTTON_FACTOR).clamp(MIN_BASE_SPEED, MAX_BASE_SPEED);
    }

    fn toggle_speed_pill(&mut self) {
        self.speed_pill_expanded = !self.speed_pill_expanded;
    }

    fn request_navigation(&mut self, request: NavigationRequest) {
        self.navigation_request = Some(request);
    }

    pub fn take_navigation_request(&mut self) -> Option<NavigationRequest> {
        self.navigation_request.take()
    }

    /// Cruise pressure rises while the camera moves steadily forward and
    /// decays when it stops or reverses. It depends only on the camera's own
    /// motion, never on the surrounding images.
    pub fn update_cruise_pressure(
        &mut self,
        movement: Vec3,
        previous_velocity: Vec3,
        delta_seconds: f32,
    ) {
        let dt = delta_seconds.max(0.0);
        let moving = movement.length_squared() > f32::EPSILON;
        let reversing = moving
            && previous_velocity.length_squared() > f32::EPSILON
            && movement
                .normalize_or_zero()
                .dot(previous_velocity.normalize_or_zero())
                < CRUISE_ALIGNMENT_THRESHOLD;

        if moving && !reversing {
            self.cruise_pressure += CRUISE_RISE_PER_SECOND * dt;
        } else {
            self.cruise_pressure -= CRUISE_DECAY_PER_SECOND * dt;
        }

        self.cruise_pressure = self.cruise_pressure.clamp(0.0, 1.0);
    }

    /// Speed multiplier contributed by cruise pressure, in `[1, 1 + gain]`.
    pub fn cruise_multiplier(&self) -> f32 {
        1.0 + self.cruise_pressure.clamp(0.0, 1.0) * CRUISE_SPEED_GAIN
    }
}

impl NavigationStatus {
    pub fn with_velocity_multiplier(mut self, velocity_multiplier: f32) -> Self {
        self.velocity_multiplier = velocity_multiplier.max(0.0);
        self.effective_speed *= self.velocity_multiplier;
        self
    }
}

impl NavigationTargets {
    pub fn from_positions<I>(positions: I) -> Self
    where
        I: IntoIterator<Item = Vec3>,
    {
        let positions: Vec<Vec3> = positions.into_iter().collect();
        Self {
            grid: SpatialGrid::from_positions(&positions),
            clusters: cluster_centers(&positions),
            positions,
        }
    }

    pub fn replace_positions<I>(&mut self, positions: I)
    where
        I: IntoIterator<Item = Vec3>,
    {
        self.positions = positions.into_iter().collect();
        self.grid = SpatialGrid::from_positions(&self.positions);
        self.clusters = cluster_centers(&self.positions);
    }

    pub fn nearest_to(&self, position: Vec3) -> Option<Vec3> {
        self.grid
            .nearby_positions(position, self.grid.cell_size * 3.0, &self.positions)
            .into_iter()
            .chain(self.positions.iter().copied())
            .min_by(|left, right| {
                left.distance_squared(position)
                    .total_cmp(&right.distance_squared(position))
            })
    }

    /// Nearest position lying in the `axis` direction from `origin`,
    /// preferring positions close to the axis line: a candidate must advance
    /// at least `min_advance` along `axis`, and among candidates the one with
    /// the smallest `advance + weight * off_axis_distance` wins, so stepping
    /// through a regular grid follows the row rather than cutting diagonals.
    /// Positions rejected by `permitted` (e.g. currently invisible ones) are
    /// never returned.
    pub fn nearest_in_direction(
        &self,
        origin: Vec3,
        axis: Vec3,
        min_advance: f32,
        permitted: impl Fn(Vec3) -> bool,
    ) -> Option<Vec3> {
        let axis = axis.normalize_or_zero();
        if axis.length_squared() <= f32::EPSILON {
            return None;
        }
        self.positions
            .iter()
            .copied()
            .filter_map(|position| {
                if !permitted(position) {
                    return None;
                }
                let displacement = position - origin;
                let advance = displacement.dot(axis);
                if advance <= min_advance {
                    return None;
                }
                let off_axis = (displacement - axis * advance).length();
                Some((position, advance + DIRECTIONAL_ORTHOGONAL_WEIGHT * off_axis))
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(position, _)| position)
    }

    pub fn nearest_cluster_to(&self, position: Vec3) -> Option<Vec3> {
        self.clusters.iter().copied().min_by(|left, right| {
            left.distance_squared(position)
                .total_cmp(&right.distance_squared(position))
        })
    }

    pub fn next_visible_cluster(&self, position: Vec3, forward: Vec3) -> Option<Vec3> {
        let forward = forward.normalize_or_zero();
        if forward.length_squared() <= f32::EPSILON {
            return None;
        }
        self.clusters
            .iter()
            .copied()
            .filter_map(|cluster| {
                let to_cluster = cluster - position;
                let forward_distance = to_cluster.dot(forward);
                if forward_distance <= 0.0 {
                    return None;
                }
                let alignment = to_cluster.normalize_or_zero().dot(forward).max(0.0);
                if alignment < 0.35 {
                    return None;
                }
                Some((cluster, forward_distance / alignment.max(0.001)))
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(cluster, _)| cluster)
    }
}

impl SpatialGrid {
    fn from_positions(positions: &[Vec3]) -> Self {
        let cell_size = estimate_spatial_cell_size(positions);
        let mut cells: HashMap<IVec3, Vec<usize>> = HashMap::new();
        for (index, position) in positions.iter().enumerate() {
            cells
                .entry(Self::cell_key(*position, cell_size))
                .or_default()
                .push(index);
        }
        Self { cell_size, cells }
    }

    fn cell_key(position: Vec3, cell_size: f32) -> IVec3 {
        (position / cell_size.max(0.001)).floor().as_ivec3()
    }

    fn nearby_positions(&self, center: Vec3, radius: f32, positions: &[Vec3]) -> Vec<Vec3> {
        if positions.is_empty() {
            return Vec::new();
        }
        let radius = radius.max(self.cell_size);
        let center_key = Self::cell_key(center, self.cell_size);
        let cell_radius = (radius / self.cell_size).ceil() as i32;
        let radius_squared = radius * radius;
        let mut nearby = Vec::new();
        for z in -cell_radius..=cell_radius {
            for y in -cell_radius..=cell_radius {
                for x in -cell_radius..=cell_radius {
                    let key = center_key + IVec3::new(x, y, z);
                    let Some(indices) = self.cells.get(&key) else {
                        continue;
                    };
                    nearby.extend(indices.iter().filter_map(|index| {
                        let position = positions.get(*index).copied()?;
                        (position.distance_squared(center) <= radius_squared).then_some(position)
                    }));
                }
            }
        }
        nearby
    }
}

impl Default for BillboardFacingSettings {
    fn default() -> Self {
        Self {
            axis: BillboardFacingAxis::All,
        }
    }
}

impl BillboardFacingAxis {
    pub fn label(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
            Self::All => "All",
        }
    }
}

impl AxisGizmoFace {
    pub fn opposite(self) -> Self {
        match self {
            Self::PositiveX => Self::NegativeX,
            Self::NegativeX => Self::PositiveX,
            Self::PositiveY => Self::NegativeY,
            Self::NegativeY => Self::PositiveY,
            Self::PositiveZ => Self::NegativeZ,
            Self::NegativeZ => Self::PositiveZ,
        }
    }
}

impl AxisGizmoState {
    /// Clicking the face the camera is already snapped to flips the view to
    /// the opposite face, matching common 3D viewport gizmo behavior.
    pub fn request_face(&mut self, face: AxisGizmoFace) {
        let face = if self.selected_face == Some(face) {
            face.opposite()
        } else {
            face
        };
        self.selected_face = Some(face);
        self.requested_face = Some(face);
    }

    pub fn take_requested_face(&mut self) -> Option<AxisGizmoFace> {
        self.requested_face.take()
    }

    /// Clicking a gizmo sphere tile orients the view to an arbitrary
    /// (non-axis-aligned) direction, so it is tracked separately from the
    /// discrete face requests; it also clears any snapped face selection
    /// since the view is no longer aligned with one.
    pub fn request_direction(&mut self, direction: Vec3) {
        self.selected_face = None;
        self.requested_direction = Some(direction);
    }

    pub fn take_requested_direction(&mut self) -> Option<Vec3> {
        self.requested_direction.take()
    }

    pub fn is_selected(&self, face: AxisGizmoFace) -> bool {
        self.selected_face == Some(face)
    }

    /// Manual camera rotation invalidates the snapped-face state.
    pub fn clear_selection(&mut self) {
        self.selected_face = None;
    }

    pub fn set_hovered(&mut self, face: Option<AxisGizmoFace>) {
        self.hovered_face = face;
    }

    pub fn hovered(&self) -> Option<AxisGizmoFace> {
        self.hovered_face
    }

    pub fn selected(&self) -> Option<AxisGizmoFace> {
        self.selected_face
    }
}

impl BillboardControls {
    pub fn new(texture_budget_mib: u32, max_texture_side: u32) -> Self {
        Self {
            max_texture_side,
            texture_budget_mib,
            billboards_pill_expanded: false,
            show_coordinates: false,
            snap_to_grid: false,
            grow_on_hover: true,
        }
    }

    /// The VRAM ceiling in bytes, as the loader accounts for it.
    pub fn texture_budget_bytes(&self) -> usize {
        self.texture_budget_mib as usize * 1024 * 1024
    }

    fn lower_texture_limit(&mut self) {
        if self.max_texture_side > 0 {
            self.max_texture_side = ((self.max_texture_side as f32 / 1.25).round() as u32).max(64);
        }
    }

    fn raise_texture_limit(&mut self) {
        if self.max_texture_side > 0 {
            self.max_texture_side = ((self.max_texture_side as f32 * 1.25).round() as u32).max(64);
        }
    }

    /// Maps a normalized slider position to a budget on a quadratic curve, so
    /// the low end (where a few hundred MiB matters) stays fine-grained while
    /// the top end still reaches tens of GiB within one drag.
    fn set_texture_budget_from_slider(&mut self, normalized_x: f32) {
        let fraction = normalized_x.clamp(0.0, 1.0).powi(2);
        let span = (MAX_TEXTURE_BUDGET_MIB - MIN_TEXTURE_BUDGET_MIB) as f32;
        let raw = MIN_TEXTURE_BUDGET_MIB as f32 + fraction * span;
        let stepped =
            (raw / TEXTURE_BUDGET_STEP_MIB as f32).round() as u32 * TEXTURE_BUDGET_STEP_MIB;
        self.texture_budget_mib = stepped.clamp(MIN_TEXTURE_BUDGET_MIB, MAX_TEXTURE_BUDGET_MIB);
    }

    /// Inverse of [`Self::set_texture_budget_from_slider`], for the fill bar.
    pub fn texture_budget_slider_percent(&self) -> f32 {
        let span = (MAX_TEXTURE_BUDGET_MIB - MIN_TEXTURE_BUDGET_MIB) as f32;
        let fraction = (self
            .texture_budget_mib
            .saturating_sub(MIN_TEXTURE_BUDGET_MIB)) as f32
            / span;
        fraction.clamp(0.0, 1.0).sqrt() * 100.0
    }

    fn toggle_pill(&mut self) {
        self.billboards_pill_expanded = !self.billboards_pill_expanded;
    }

    fn toggle_show_coordinates(&mut self) {
        self.show_coordinates = !self.show_coordinates;
    }

    fn toggle_snap_to_grid(&mut self) {
        self.snap_to_grid = !self.snap_to_grid;
    }

    fn toggle_grow_on_hover(&mut self) {
        self.grow_on_hover = !self.grow_on_hover;
    }
}

impl DebugSettings {
    fn toggle_pill(&mut self) {
        self.debug_pill_expanded = !self.debug_pill_expanded;
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            frame_times_ms: VecDeque::with_capacity(PERF_SAMPLE_COUNT),
            max_samples: PERF_SAMPLE_COUNT,
            last_frame_ms: 0.0,
            fps: 0.0,
            metrics_pill_expanded: false,
        }
    }
}

impl PauseMenuState {
    /// `Escape`: opens the menu at the top level if it's closed, otherwise
    /// steps back one level, same as the on-screen back controls.
    pub fn escape_pressed(&mut self) {
        if !self.paused {
            self.paused = true;
            self.screen = PauseScreen::Main;
            return;
        }
        self.back();
    }

    /// Steps back one level in the menu hierarchy; closes the menu entirely
    /// from the top level. Shared by `Escape` and every on-screen back button
    /// so all of them navigate identically.
    pub fn back(&mut self) {
        match self.screen {
            PauseScreen::Main => self.resume(),
            PauseScreen::Settings | PauseScreen::Controls | PauseScreen::Animations => {
                self.screen = PauseScreen::Main
            }
            PauseScreen::CatalogOptions => self.screen = PauseScreen::Settings,
        }
    }

    /// The controls shortcut: opens the menu on the controls sheet, or
    /// closes it from there.
    pub fn toggle_controls(&mut self) {
        if self.paused && self.screen == PauseScreen::Controls {
            self.resume();
        } else {
            self.paused = true;
            self.screen = PauseScreen::Controls;
        }
    }

    pub fn resume(&mut self) {
        self.paused = false;
        self.screen = PauseScreen::Main;
    }

    fn open_settings(&mut self) {
        self.screen = PauseScreen::Settings;
    }

    fn open_catalog_options(&mut self) {
        self.screen = PauseScreen::CatalogOptions;
    }

    /// Opens the menu directly on the catalog screen, e.g. while the last
    /// session's catalog is being restored at startup.
    pub fn open_catalog_options_screen(&mut self) {
        self.paused = true;
        self.screen = PauseScreen::CatalogOptions;
    }

    /// True while the catalog options screen (nested under Settings) is on
    /// screen; only reachable while paused.
    pub fn catalog_options_open(&self) -> bool {
        self.paused && self.screen == PauseScreen::CatalogOptions
    }
}

impl Default for RenderResolutionSettings {
    fn default() -> Self {
        Self {
            scale: 1.0,
            pending_resolution: false,
        }
    }
}

impl RenderResolutionSettings {
    /// Snaps to 5% steps so slider drags land on readable values.
    fn set_scale(&mut self, scale: f32) {
        let scale = ((scale / RENDER_SCALE_STEP).round() * RENDER_SCALE_STEP)
            .clamp(MIN_RENDER_SCALE, MAX_RENDER_SCALE);
        if (self.scale - scale).abs() < 0.001 {
            return;
        }
        self.scale = scale;
        self.pending_resolution = true;
    }

    fn set_scale_from_slider(&mut self, normalized_x: f32) {
        let amount = normalized_x.clamp(0.0, 1.0);
        self.set_scale(MIN_RENDER_SCALE + amount * (MAX_RENDER_SCALE - MIN_RENDER_SCALE));
    }

    pub fn take_resolution_changed(&mut self) -> bool {
        let changed = self.pending_resolution;
        self.pending_resolution = false;
        changed
    }

    pub fn selected_scale(&self) -> f32 {
        self.scale.clamp(MIN_RENDER_SCALE, MAX_RENDER_SCALE)
    }

    pub fn target_size(&self, window_size: UVec2) -> UVec2 {
        let scale = self.selected_scale();
        UVec2::new(
            (window_size.x as f32 * scale).round().max(1.0) as u32,
            (window_size.y as f32 * scale).round().max(1.0) as u32,
        )
    }

    fn scale_label(&self) -> String {
        format!("{:.0}%", self.selected_scale() * 100.0)
    }

    fn slider_percent(&self) -> f32 {
        ((self.selected_scale() - MIN_RENDER_SCALE) / (MAX_RENDER_SCALE - MIN_RENDER_SCALE) * 100.0)
            .clamp(0.0, 100.0)
    }
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self { volume: 1.0 }
    }
}

impl AudioSettings {
    /// Settings starting at a remembered `volume`.
    pub fn with_volume(volume: f32) -> Self {
        Self {
            volume: volume.clamp(0.0, 1.0),
        }
    }

    /// Snaps to 5% steps so slider drags land on readable values.
    fn set_volume_from_slider(&mut self, normalized_x: f32) {
        self.volume = ((normalized_x.clamp(0.0, 1.0) / AUDIO_VOLUME_STEP).round()
            * AUDIO_VOLUME_STEP)
            .clamp(0.0, 1.0);
    }

    pub fn volume(&self) -> f32 {
        self.volume
    }

    fn volume_label(&self) -> String {
        format!("{:.0}%", self.volume * 100.0)
    }

    fn slider_percent(&self) -> f32 {
        (self.volume * 100.0).clamp(0.0, 100.0)
    }
}

impl Default for StartMenuState {
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            status: "Select folders or use the current catalog.".to_owned(),
            loading: false,
            picking_folder: false,
            add_folder_requested: false,
            load_requested: false,
            use_current_requested: false,
            clear_requested: false,
            reset_layout_requested: false,
            remove_requested: None,
        }
    }
}

impl StartMenuState {
    pub fn add_root(&mut self, root: impl Into<String>) {
        let root = root.into();
        if !self.roots.iter().any(|existing| existing == &root) {
            self.roots.push(root);
        }
    }

    pub fn remove_root(&mut self, index: usize) {
        if index < self.roots.len() {
            self.roots.remove(index);
        }
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    /// Interactive actions are ignored while a background load or folder
    /// dialog is running.
    pub fn busy(&self) -> bool {
        self.loading || self.picking_folder
    }

    pub fn take_add_folder_request(&mut self) -> bool {
        let requested = self.add_folder_requested;
        self.add_folder_requested = false;
        requested
    }

    /// Queues a catalog load of the current roots, same as pressing
    /// "Load selected".
    pub fn request_load(&mut self) {
        self.load_requested = true;
    }

    pub fn take_load_request(&mut self) -> bool {
        let requested = self.load_requested;
        self.load_requested = false;
        requested
    }

    pub fn take_use_current_request(&mut self) -> bool {
        let requested = self.use_current_requested;
        self.use_current_requested = false;
        requested
    }

    pub fn take_clear_request(&mut self) -> bool {
        let requested = self.clear_requested;
        self.clear_requested = false;
        requested
    }

    /// Whether "Reset layout" was pressed since this was last called.
    pub fn take_reset_layout_request(&mut self) -> bool {
        std::mem::take(&mut self.reset_layout_requested)
    }

    pub fn take_remove_request(&mut self) -> Option<usize> {
        self.remove_requested.take()
    }
}

impl PerformanceMetrics {
    pub fn sample(&mut self, frame_seconds: f32) {
        self.last_frame_ms = frame_seconds.max(0.0) * 1_000.0;
        self.fps = if frame_seconds > f32::EPSILON {
            1.0 / frame_seconds
        } else {
            0.0
        };
        self.frame_times_ms.push_back(self.last_frame_ms);
        while self.frame_times_ms.len() > self.max_samples {
            self.frame_times_ms.pop_front();
        }
    }

    fn time_bar_value(&self, index: usize, count: usize) -> f32 {
        if self.frame_times_ms.is_empty() || count == 0 {
            return 0.0;
        }
        let sample_index = index * self.frame_times_ms.len() / count;
        let sample = self
            .frame_times_ms
            .get(sample_index.min(self.frame_times_ms.len() - 1))
            .copied()
            .unwrap_or_default();
        let max_ms = self.max_frame_ms().max(16.0);
        (sample / max_ms).clamp(0.03, 1.0)
    }

    fn histogram_value(&self, index: usize, count: usize) -> f32 {
        if self.frame_times_ms.is_empty() || count == 0 {
            return 0.0;
        }
        let max_ms = self.max_frame_ms().max(16.0);
        let bucket_width = max_ms / count as f32;
        let low = index as f32 * bucket_width;
        let high = if index + 1 == count {
            f32::MAX
        } else {
            low + bucket_width
        };
        let bucket_count = self
            .frame_times_ms
            .iter()
            .filter(|value| **value >= low && **value < high)
            .count();
        let max_bucket = (self.frame_times_ms.len() / count.max(1)).max(1) as f32;
        (bucket_count as f32 / max_bucket).clamp(0.03, 1.0)
    }

    fn max_frame_ms(&self) -> f32 {
        self.frame_times_ms.iter().copied().fold(0.0, f32::max)
    }

    fn sample_count(&self) -> usize {
        self.frame_times_ms.len()
    }

    fn toggle_pill(&mut self) {
        self.metrics_pill_expanded = !self.metrics_pill_expanded;
    }
}

/// Speed is a fixed function of the base speed and cruise pressure only; it no
/// longer varies with the images around the camera.
pub fn navigation_status(settings: &NavigationSettings) -> NavigationStatus {
    NavigationStatus {
        cruise_pressure: settings.cruise_pressure,
        velocity_multiplier: 1.0,
        effective_speed: settings.base_speed * settings.cruise_multiplier(),
    }
}

pub fn spawn_viewer_ui(commands: &mut Commands) {
    let ui_camera = commands
        .spawn((
            Camera2d,
            IsDefaultUiCamera,
            Camera {
                order: 3,
                clear_color: ClearColorConfig::None,
                ..default()
            },
        ))
        .id();
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::NONE),
            TargetCamera(ui_camera),
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Node {
                        position_type: PositionType::Relative,
                        width: Val::VMin(177.7778),
                        height: Val::VMin(100.0),
                        max_width: Val::Percent(100.0),
                        max_height: Val::Percent(100.0),
                        aspect_ratio: Some(16.0 / 9.0),
                        ..default()
                    },
                    HudPanels,
                ))
                .with_children(|bounds| {
                    // Right-hand columns align their pills to the right edge
                    // of the 16:9 frame so collapsed pills hug the frame
                    // instead of floating at a fixed column width.
                    bounds
                        .spawn(Node {
                            position_type: PositionType::Absolute,
                            top: Val::Px(18.0),
                            right: Val::Px(18.0),
                            width: Val::Px(300.0),
                            flex_direction: FlexDirection::Column,
                            align_items: AlignItems::FlexEnd,
                            row_gap: Val::Px(8.0),
                            ..default()
                        })
                        .with_children(|menus| {
                            spawn_header(menus, "Navigation");
                            spawn_speed_pill(menus);
                        });
                    bounds
                        .spawn(Node {
                            position_type: PositionType::Absolute,
                            top: Val::Px(18.0),
                            left: Val::Px(18.0),
                            width: Val::Px(300.0),
                            flex_direction: FlexDirection::Column,
                            align_items: AlignItems::FlexStart,
                            row_gap: Val::Px(8.0),
                            ..default()
                        })
                        .with_children(|view| {
                            spawn_header(view, "View");
                            spawn_control_panel_pill(view);
                            spawn_billboards_pill(view);
                            spawn_folder_pill(view);
                            spawn_search_pill(view);
                            spawn_edit_pill(view);
                        });
                    bounds
                        .spawn(Node {
                            position_type: PositionType::Absolute,
                            bottom: Val::Px(18.0),
                            right: Val::Px(18.0),
                            width: Val::Px(300.0),
                            flex_direction: FlexDirection::Column,
                            align_items: AlignItems::FlexEnd,
                            row_gap: Val::Px(8.0),
                            ..default()
                        })
                        .with_children(|developer| {
                            spawn_header(developer, "Developer");
                            spawn_debug_pill(developer);
                            spawn_performance_pill(developer);
                        });
                });
            spawn_rename_prompt(parent);
            spawn_pause_overlay(parent);
            spawn_start_overlay(parent);
        });
}

pub fn handle_navigation_buttons(
    mut navigation: ResMut<NavigationSettings>,
    interaction_query: ButtonInteractionQuery,
) {
    for (interaction, button) in &interaction_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button {
            ViewerUiButton::Slower => navigation.slow_down(),
            ViewerUiButton::Faster => navigation.speed_up(),
            ViewerUiButton::ToggleNavigationPill => navigation.toggle_speed_pill(),
            ViewerUiButton::FocusNearestImage => {
                navigation.request_navigation(NavigationRequest::FocusNearestImage)
            }
            ViewerUiButton::JumpNextVisibleCluster => {
                navigation.request_navigation(NavigationRequest::JumpNextVisibleCluster)
            }
            _ => {}
        }
    }
}

pub fn handle_billboard_buttons(
    mut billboard_facing: ResMut<BillboardFacingSettings>,
    mut billboard_controls: ResMut<BillboardControls>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    slider_query: Query<
        (
            &Interaction,
            &ViewerUiButton,
            Option<&RelativeCursorPosition>,
        ),
        With<Button>,
    >,
    interaction_query: ButtonInteractionQuery,
) {
    // A slider follows the held cursor rather than discrete presses, so it is
    // read before the press-driven buttons below.
    if mouse_buttons.pressed(MouseButton::Left) {
        for (interaction, button, cursor_position) in &slider_query {
            if *interaction != Interaction::Pressed
                || !matches!(button, ViewerUiButton::SetTextureBudgetFromSlider)
            {
                continue;
            }
            let Some(position) = cursor_position.and_then(|cursor| cursor.normalized) else {
                continue;
            };
            billboard_controls.set_texture_budget_from_slider(position.x);
        }
    }

    for (interaction, button) in &interaction_query {
        if *interaction != Interaction::Pressed || button_disabled(button, &billboard_controls) {
            continue;
        }
        match button {
            ViewerUiButton::ToggleBillboardsPill => billboard_controls.toggle_pill(),
            ViewerUiButton::SetBillboardAxis(axis) => billboard_facing.axis = *axis,
            ViewerUiButton::ToggleBillboardCoordinates => {
                billboard_controls.toggle_show_coordinates()
            }
            ViewerUiButton::ToggleBillboardGridSnap => billboard_controls.toggle_snap_to_grid(),
            ViewerUiButton::ToggleBillboardHoverGrowth => billboard_controls.toggle_grow_on_hover(),
            ViewerUiButton::TextureLower => billboard_controls.lower_texture_limit(),
            ViewerUiButton::TextureHigher => billboard_controls.raise_texture_limit(),
            _ => {}
        }
    }
}

pub fn handle_debug_buttons(
    mut debug_settings: ResMut<DebugSettings>,
    mut benchmark: ResMut<BenchmarkControls>,
    interaction_query: ButtonInteractionQuery,
) {
    for (interaction, button) in &interaction_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button {
            ViewerUiButton::ToggleDebugPill => debug_settings.toggle_pill(),
            // Both are inert mid-run: a restart would discard the samples
            // already collected, and changing the duration under a running
            // measurement would make the result unattributable.
            ViewerUiButton::StartBenchmark if benchmark.phase != BenchmarkPhase::Recording => {
                benchmark.request_start();
            }
            ViewerUiButton::CycleBenchmarkDuration
                if benchmark.phase != BenchmarkPhase::Recording =>
            {
                benchmark.cycle_duration();
            }
            _ => {}
        }
    }
}

pub fn handle_performance_buttons(
    mut performance: ResMut<PerformanceMetrics>,
    interaction_query: ButtonInteractionQuery,
) {
    for (interaction, button) in &interaction_query {
        if *interaction == Interaction::Pressed
            && matches!(button, ViewerUiButton::TogglePerformancePill)
        {
            performance.toggle_pill();
        }
    }
}

pub fn handle_pause_menu_buttons(
    mut pause_menu: ResMut<PauseMenuState>,
    mut playback_settings: ResMut<PlaybackSettings>,
    mut view_settings: ResMut<ViewSettings>,
    mut app_exit: EventWriter<AppExit>,
    interaction_query: ButtonInteractionQuery,
) {
    for (interaction, button) in &interaction_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button {
            ViewerUiButton::PauseResume => pause_menu.resume(),
            ViewerUiButton::PauseSettings => pause_menu.open_settings(),
            // Shared by the Settings panel's "Back" button and the catalog
            // options screen's back arrow: both just step back one level.
            ViewerUiButton::PauseBack => pause_menu.back(),
            ViewerUiButton::PauseCatalogOptions => pause_menu.open_catalog_options(),
            ViewerUiButton::PauseControls => pause_menu.screen = PauseScreen::Controls,
            ViewerUiButton::PauseAnimations => pause_menu.screen = PauseScreen::Animations,
            ViewerUiButton::ToggleRememberPlaybackPosition => {
                playback_settings.remember_position = !playback_settings.remember_position;
            }
            ViewerUiButton::ToggleRememberCamera => {
                view_settings.remember_camera = !view_settings.remember_camera;
            }
            ViewerUiButton::PauseQuit => {
                app_exit.send(AppExit::Success);
            }
            _ => {}
        }
    }
}

pub fn handle_resolution_buttons(
    mut render_resolution: ResMut<RenderResolutionSettings>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    slider_query: Query<
        (
            &Interaction,
            &ViewerUiButton,
            Option<&RelativeCursorPosition>,
        ),
        With<Button>,
    >,
) {
    if !mouse_buttons.pressed(MouseButton::Left) {
        return;
    }
    for (interaction, button, cursor_position) in &slider_query {
        if *interaction != Interaction::Pressed
            || !matches!(button, ViewerUiButton::SetResolutionScaleFromSlider)
        {
            continue;
        }
        let Some(position) = cursor_position.and_then(|cursor| cursor.normalized) else {
            continue;
        };
        render_resolution.set_scale_from_slider(position.x);
    }
}

pub fn handle_audio_buttons(
    mut audio_settings: ResMut<AudioSettings>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    slider_query: Query<
        (
            &Interaction,
            &ViewerUiButton,
            Option<&RelativeCursorPosition>,
        ),
        With<Button>,
    >,
) {
    if !mouse_buttons.pressed(MouseButton::Left) {
        return;
    }
    for (interaction, button, cursor_position) in &slider_query {
        if *interaction != Interaction::Pressed
            || !matches!(button, ViewerUiButton::SetAudioVolumeFromSlider)
        {
            continue;
        }
        let Some(position) = cursor_position.and_then(|cursor| cursor.normalized) else {
            continue;
        };
        audio_settings.set_volume_from_slider(position.x);
    }
}

pub fn handle_start_menu_buttons(
    mut start_menu: ResMut<StartMenuState>,
    interaction_query: ButtonInteractionQuery,
) {
    if start_menu.busy() {
        return;
    }
    for (interaction, button) in &interaction_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button {
            ViewerUiButton::StartAddFolder => start_menu.add_folder_requested = true,
            ViewerUiButton::StartClearFolders => start_menu.clear_requested = true,
            ViewerUiButton::StartLoadFolders => start_menu.load_requested = true,
            ViewerUiButton::StartUseCurrentCatalog => start_menu.use_current_requested = true,
            ViewerUiButton::StartResetLayout => start_menu.reset_layout_requested = true,
            ViewerUiButton::StartRemoveFolder(index) => start_menu.remove_requested = Some(*index),
            _ => {}
        }
    }
}

pub fn update_navigation_text(navigation: Res<NavigationSettings>, mut text_query: UiTextQuery) {
    for (mut text, text_kind) in &mut text_query {
        match text_kind {
            ViewerUiText::SpeedSummary => {
                **text = format!("Speed {:>8.2} u/s", navigation.status.effective_speed);
            }
            ViewerUiText::SpeedDetails => {
                **text = speed_details(&navigation);
            }
            _ => {}
        }
    }
}

pub fn update_billboard_text(
    billboard_facing: Res<BillboardFacingSettings>,
    billboard_controls: Res<BillboardControls>,
    billboard_stats: Res<BillboardStats>,
    mut text_query: UiTextQuery,
) {
    for (mut text, text_kind) in &mut text_query {
        match text_kind {
            ViewerUiText::BillboardsSummary => {
                **text = format!(
                    "Billboards {:>5}/{:<5}",
                    billboard_stats.loaded, billboard_stats.entity_count
                );
            }
            ViewerUiText::BillboardsDetails => {
                **text = format!(
                    "axis {:>3}   entities {:>7}
loaded {:>7}   pending {:>7}
active {:>7}   failed  {:>7}
upload {:>3} {:>5.1} MiB  queued {:>5}
orient {:>5}/{:<5} skipped {:>5}
inview {:>5}/{:<5} textured",
                    billboard_facing.axis.label(),
                    billboard_stats.entity_count,
                    billboard_stats.loaded,
                    billboard_stats.pending,
                    billboard_stats.in_flight,
                    billboard_stats.failed,
                    billboard_stats.uploaded_textures,
                    billboard_stats.uploaded_bytes as f32 / (1024.0 * 1024.0),
                    billboard_stats.upload_queue,
                    billboard_stats.orientation_updated,
                    billboard_stats.orientation_checked,
                    billboard_stats.orientation_skipped,
                    billboard_stats.visible_textured,
                    billboard_stats.visible_billboards,
                );
            }
            ViewerUiText::BillboardTextureBudgetSummary => {
                **text = format!(
                    "Texture VRAM {:>5.1}/{:<5.1} GiB",
                    billboard_stats.texture_bytes_used as f32 / (1024.0 * 1024.0 * 1024.0),
                    billboard_controls.texture_budget_mib as f32 / 1024.0,
                );
            }
            ViewerUiText::BillboardCoordinatesMark => {
                **text = if billboard_controls.show_coordinates {
                    "X".to_owned()
                } else {
                    String::new()
                };
            }
            ViewerUiText::BillboardGridSnapMark => {
                **text = if billboard_controls.snap_to_grid {
                    "X".to_owned()
                } else {
                    String::new()
                };
            }
            ViewerUiText::BillboardHoverGrowthMark => {
                **text = if billboard_controls.grow_on_hover {
                    "X".to_owned()
                } else {
                    String::new()
                };
            }
            _ => {}
        }
    }
}

pub fn update_debug_text(benchmark: Res<BenchmarkControls>, mut text_query: UiTextQuery) {
    for (mut text, text_kind) in &mut text_query {
        match text_kind {
            ViewerUiText::DebugSummary => {
                **text = match benchmark.phase {
                    BenchmarkPhase::Recording => {
                        format!("Bench {:>3.0}%", benchmark.progress * 100.0)
                    }
                    _ => "Debug tools".to_owned(),
                };
            }
            ViewerUiText::BenchmarkStatus => **text = benchmark.status_line(),
            ViewerUiText::BenchmarkDetails => **text = benchmark.detail_lines(),
            _ => {}
        }
    }
}

pub fn update_performance_text(performance: Res<PerformanceMetrics>, mut text_query: UiTextQuery) {
    for (mut text, text_kind) in &mut text_query {
        match text_kind {
            ViewerUiText::PerformanceSummary => {
                **text = format!("Perf {:>5.0} fps", performance.fps);
            }
            ViewerUiText::PerformanceDetails => {
                **text = format!(
                    "fps {:>6.1}   frame {:>7.2} ms\nsamples {:>4}",
                    performance.fps,
                    performance.last_frame_ms,
                    performance.sample_count()
                );
            }
            _ => {}
        }
    }
}

pub fn update_resolution_text(
    render_resolution: Res<RenderResolutionSettings>,
    mut text_query: UiTextQuery,
) {
    for (mut text, text_kind) in &mut text_query {
        if let ViewerUiText::ResolutionSummary = text_kind {
            **text = format!("Render scale {:>4}", render_resolution.scale_label());
        }
    }
}

pub fn update_audio_text(
    audio_settings: Res<AudioSettings>,
    playback_settings: Res<PlaybackSettings>,
    view_settings: Res<ViewSettings>,
    mut text_query: UiTextQuery,
) {
    for (mut text, text_kind) in &mut text_query {
        match text_kind {
            ViewerUiText::AudioVolumeSummary => {
                **text = format!("Volume {:>4}", audio_settings.volume_label());
            }
            ViewerUiText::RememberPlaybackPositionMark => {
                **text = if playback_settings.remember_position {
                    "X".to_owned()
                } else {
                    String::new()
                };
            }
            ViewerUiText::RememberCameraMark => {
                **text = if view_settings.remember_camera {
                    "X".to_owned()
                } else {
                    String::new()
                };
            }
            _ => {}
        }
    }
}

pub fn update_start_text(start_menu: Res<StartMenuState>, mut text_query: UiTextQuery) {
    for (mut text, text_kind) in &mut text_query {
        match text_kind {
            ViewerUiText::StartRoots => {
                **text = if start_menu.roots.is_empty() {
                    "No folders selected".to_owned()
                } else if start_menu.roots.len() > START_MENU_FOLDER_ROWS {
                    format!(
                        "+{} more folders",
                        start_menu.roots.len() - START_MENU_FOLDER_ROWS
                    )
                } else {
                    String::new()
                };
            }
            ViewerUiText::StartStatus => {
                **text = start_menu.status.clone();
            }
            ViewerUiText::StartFolderPath(row_index) => {
                **text = start_menu
                    .roots
                    .get(*row_index)
                    .map(|root| truncate_path_label(root, 48))
                    .unwrap_or_default();
            }
            _ => {}
        }
    }
}

pub fn update_navigation_panels(
    navigation: Res<NavigationSettings>,
    mut panel_query: UiPanelQuery,
) {
    for (mut node, panel) in &mut panel_query {
        match panel {
            ViewerUiPanel::NavigationPill => {
                node.width = if navigation.speed_pill_expanded {
                    Val::Px(300.0)
                } else {
                    Val::Px(164.0)
                };
            }
            ViewerUiPanel::NavigationExpanded => {
                node.display = display_if(navigation.speed_pill_expanded);
            }
            _ => {}
        }
    }
}

pub fn update_billboard_panels(
    billboard_controls: Res<BillboardControls>,
    mut panel_query: UiPanelQuery,
) {
    for (mut node, panel) in &mut panel_query {
        match panel {
            ViewerUiPanel::BillboardsPill => {
                node.width = if billboard_controls.billboards_pill_expanded {
                    Val::Px(300.0)
                } else {
                    Val::Px(190.0)
                };
            }
            ViewerUiPanel::BillboardsExpanded => {
                node.display = display_if(billboard_controls.billboards_pill_expanded);
            }
            ViewerUiPanel::BillboardTextureBudgetFill => {
                node.width = Val::Percent(billboard_controls.texture_budget_slider_percent());
            }
            _ => {}
        }
    }
}

pub fn update_debug_panels(debug_settings: Res<DebugSettings>, mut panel_query: UiPanelQuery) {
    for (mut node, panel) in &mut panel_query {
        match panel {
            ViewerUiPanel::DebugPill => {
                node.width = if debug_settings.debug_pill_expanded {
                    Val::Px(300.0)
                } else {
                    Val::Px(164.0)
                };
            }
            ViewerUiPanel::DebugExpanded => {
                node.display = display_if(debug_settings.debug_pill_expanded);
            }
            _ => {}
        }
    }
}

pub fn update_performance_panels(
    performance: Res<PerformanceMetrics>,
    mut panel_query: UiPanelQuery,
) {
    for (mut node, panel) in &mut panel_query {
        match panel {
            ViewerUiPanel::PerformancePill => {
                node.width = if performance.metrics_pill_expanded {
                    Val::Px(300.0)
                } else {
                    Val::Px(164.0)
                };
            }
            ViewerUiPanel::PerformanceExpanded => {
                node.display = display_if(performance.metrics_pill_expanded);
            }
            _ => {}
        }
    }
}

pub fn update_pause_panels(
    pause_menu: Res<PauseMenuState>,
    render_resolution: Res<RenderResolutionSettings>,
    audio_settings: Res<AudioSettings>,
    mut panel_query: UiPanelQuery,
) {
    for (mut node, panel) in &mut panel_query {
        match panel {
            ViewerUiPanel::PauseOverlay => {
                node.display = display_if(pause_menu.paused);
            }
            ViewerUiPanel::PauseMain => {
                node.display =
                    display_if(pause_menu.paused && pause_menu.screen == PauseScreen::Main);
            }
            ViewerUiPanel::PauseSettings => {
                node.display =
                    display_if(pause_menu.paused && pause_menu.screen == PauseScreen::Settings);
            }
            ViewerUiPanel::PauseControls => {
                node.display =
                    display_if(pause_menu.paused && pause_menu.screen == PauseScreen::Controls);
            }
            ViewerUiPanel::PauseAnimations => {
                node.display =
                    display_if(pause_menu.paused && pause_menu.screen == PauseScreen::Animations);
            }
            ViewerUiPanel::ResolutionScaleFill => {
                node.width = Val::Percent(render_resolution.slider_percent());
            }
            ViewerUiPanel::AudioVolumeFill => {
                node.width = Val::Percent(audio_settings.slider_percent());
            }
            _ => {}
        }
    }
}

pub fn update_start_panels(
    pause_menu: Res<PauseMenuState>,
    start_menu: Res<StartMenuState>,
    mut panel_query: UiPanelQuery,
) {
    for (mut node, panel) in &mut panel_query {
        match panel {
            ViewerUiPanel::StartOverlay => {
                node.display = display_if(pause_menu.catalog_options_open());
            }
            ViewerUiPanel::StartFolderRow(row_index) => {
                node.display = display_if(*row_index < start_menu.roots.len());
            }
            _ => {}
        }
    }
}

pub fn update_navigation_button_colors(
    _navigation: Res<NavigationSettings>,
    mut button_query: UiButtonColorQuery,
) {
    for (button, interaction, color) in &mut button_query {
        let active = match button {
            ViewerUiButton::ToggleNavigationPill => {
                set_header_color(color, interaction);
                continue;
            }
            ViewerUiButton::Slower
            | ViewerUiButton::Faster
            | ViewerUiButton::FocusNearestImage
            | ViewerUiButton::JumpNextVisibleCluster => false,
            _ => continue,
        };
        set_button_color(color, active, interaction);
    }
}

pub fn update_billboard_button_colors(
    billboard_facing: Res<BillboardFacingSettings>,
    billboard_controls: Res<BillboardControls>,
    mut button_query: UiButtonColorQuery,
) {
    for (button, interaction, color) in &mut button_query {
        if button_disabled(button, &billboard_controls) {
            set_disabled_button_color(color);
            continue;
        }
        let active = match button {
            ViewerUiButton::ToggleBillboardsPill => {
                set_header_color(color, interaction);
                continue;
            }
            ViewerUiButton::SetBillboardAxis(axis) => billboard_facing.axis == *axis,
            ViewerUiButton::ToggleBillboardCoordinates => billboard_controls.show_coordinates,
            ViewerUiButton::ToggleBillboardGridSnap => billboard_controls.snap_to_grid,
            ViewerUiButton::ToggleBillboardHoverGrowth => billboard_controls.grow_on_hover,
            ViewerUiButton::TextureLower | ViewerUiButton::TextureHigher => false,
            _ => continue,
        };
        set_button_color(color, active, interaction);
    }
}

pub fn update_debug_button_colors(
    benchmark: Res<BenchmarkControls>,
    mut button_query: UiButtonColorQuery,
) {
    let recording = benchmark.phase == BenchmarkPhase::Recording;
    for (button, interaction, color) in &mut button_query {
        match button {
            ViewerUiButton::ToggleDebugPill => set_header_color(color, interaction),
            // Held active while a run is in progress, so the button reads as
            // "recording" rather than as a control that stopped responding.
            ViewerUiButton::StartBenchmark => set_button_color(color, recording, interaction),
            ViewerUiButton::CycleBenchmarkDuration => set_button_color(color, false, interaction),
            _ => {}
        }
    }
}

pub fn update_performance_button_colors(mut button_query: UiButtonColorQuery) {
    for (button, interaction, color) in &mut button_query {
        if matches!(button, ViewerUiButton::TogglePerformancePill) {
            set_header_color(color, interaction);
        }
    }
}

pub fn update_pause_button_colors(
    pause_menu: Res<PauseMenuState>,
    playback_settings: Res<PlaybackSettings>,
    view_settings: Res<ViewSettings>,
    mut button_query: UiButtonColorQuery,
) {
    for (button, interaction, color) in &mut button_query {
        let active = match button {
            ViewerUiButton::PauseSettings => pause_menu.screen == PauseScreen::Settings,
            ViewerUiButton::ToggleRememberPlaybackPosition => playback_settings.remember_position,
            ViewerUiButton::ToggleRememberCamera => view_settings.remember_camera,
            ViewerUiButton::PauseResume
            | ViewerUiButton::PauseBack
            | ViewerUiButton::PauseCatalogOptions
            | ViewerUiButton::PauseControls
            | ViewerUiButton::PauseAnimations
            | ViewerUiButton::PauseQuit => false,
            _ => continue,
        };
        set_button_color(color, active, interaction);
    }
}

pub fn update_resolution_button_colors(mut button_query: UiButtonColorQuery) {
    for (button, interaction, color) in &mut button_query {
        if matches!(
            button,
            ViewerUiButton::SetTextureBudgetFromSlider
                | ViewerUiButton::SetResolutionScaleFromSlider
                | ViewerUiButton::SetAudioVolumeFromSlider
        ) {
            set_button_color(color, false, interaction);
        }
    }
}

pub fn update_start_button_colors(
    start_menu: Res<StartMenuState>,
    mut button_query: UiButtonColorQuery,
) {
    for (button, interaction, color) in &mut button_query {
        match button {
            ViewerUiButton::StartAddFolder
            | ViewerUiButton::StartClearFolders
            | ViewerUiButton::StartLoadFolders
            | ViewerUiButton::StartUseCurrentCatalog
            | ViewerUiButton::StartResetLayout
            | ViewerUiButton::StartRemoveFolder(_) => {
                if start_menu.busy() {
                    set_disabled_button_color(color);
                } else {
                    set_button_color(color, false, interaction);
                }
            }
            _ => {}
        }
    }
}

pub fn update_perf_graph_bars(
    performance: Res<PerformanceMetrics>,
    mut graph_query: PerfGraphQuery,
) {
    for (mut node, bar) in &mut graph_query {
        match bar {
            PerfGraphBar::Time(index) => {
                node.height = Val::Px(36.0 * performance.time_bar_value(*index, PERF_GRAPH_BARS));
            }
            PerfGraphBar::Histogram(index) => {
                node.height =
                    Val::Px(36.0 * performance.histogram_value(*index, PERF_HISTOGRAM_BINS));
            }
        }
    }
}

pub fn collect_performance_metrics(time: Res<Time>, mut performance: ResMut<PerformanceMetrics>) {
    performance.sample(time.delta_secs());
}

/// Two fixed-width lines so the buttons below never shift as the values
/// change. `eff` already folds in both the cruise and shift multipliers.
fn speed_details(settings: &NavigationSettings) -> String {
    let status = &settings.status;
    format!(
        "base   {:>8.2}  eff   {:>8.2}\ncruise  x{:>6.2}  shift  x{:>6.2}",
        settings.base_speed,
        status.effective_speed,
        settings.cruise_multiplier(),
        status.velocity_multiplier,
    )
}

fn estimate_spatial_cell_size(positions: &[Vec3]) -> f32 {
    if positions.len() < 2 {
        return 1.0;
    }
    let sample_count = positions.len().min(96);
    let mut nearest_sum = 0.0;
    let mut measured = 0;
    for index in 0..sample_count {
        let position = positions[index];
        if let Some(distance) = positions
            .iter()
            .enumerate()
            .filter_map(|(other_index, other)| {
                (index != other_index).then_some(position.distance(*other))
            })
            .filter(|distance| *distance > f32::EPSILON)
            .min_by(|left, right| left.total_cmp(right))
        {
            nearest_sum += distance;
            measured += 1;
        }
    }
    if measured == 0 {
        1.0
    } else {
        (nearest_sum / measured as f32).max(1.0)
    }
}

fn cluster_centers(positions: &[Vec3]) -> Vec<Vec3> {
    if positions.is_empty() {
        return Vec::new();
    }
    let cell_size = estimate_spatial_cell_size(positions) * CLUSTER_GRID_FACTOR;
    let mut cells: HashMap<IVec3, (Vec3, usize)> = HashMap::new();
    for position in positions {
        let key = SpatialGrid::cell_key(*position, cell_size);
        let entry = cells.entry(key).or_insert((Vec3::ZERO, 0));
        entry.0 += *position;
        entry.1 += 1;
    }
    cells
        .into_values()
        .filter_map(|(sum, count)| (count > 0).then_some(sum / count as f32))
        .collect()
}

pub(crate) fn truncate_ui_label(label: &str, max_chars: usize) -> String {
    let mut chars = label.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

/// Truncates from the front so the most specific trailing path components
/// stay visible (`...\parent\folder`).
fn truncate_path_label(path: &str, max_chars: usize) -> String {
    let count = path.chars().count();
    if count <= max_chars {
        return path.to_owned();
    }
    let tail: String = path
        .chars()
        .skip(count.saturating_sub(max_chars.saturating_sub(3)))
        .collect();
    format!("...{tail}")
}

pub(crate) fn compact_usize(value: usize) -> String {
    compact_u128(value as u128)
}

fn compact_u128(value: u128) -> String {
    if value >= 1_000_000_000 {
        format!("{:.1}B", value as f64 / 1_000_000_000.0)
    } else if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn spawn_pause_overlay(parent: &mut ChildBuilder) {
    parent
        .spawn((
            Node {
                display: Display::None,
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.015, 0.018, 0.024, 0.58)),
            ViewerUiPanel::PauseOverlay,
            // The full-screen scrim owns the pointer while the menu is open
            // so no click leaks into the 3D scene behind it.
            Interaction::default(),
            FocusPolicy::Block,
        ))
        .with_children(|overlay| {
            overlay
                .spawn(menu_panel(ViewerUiPanel::PauseMain, 340.0))
                .with_children(|menu| {
                    spawn_menu_title(menu, "Paused");
                    spawn_menu_button(menu, ViewerUiButton::PauseResume, "Resume");
                    spawn_menu_button(menu, ViewerUiButton::PauseSettings, "Settings");
                    spawn_menu_button(menu, ViewerUiButton::PauseAnimations, "Animations");
                    spawn_menu_button(menu, ViewerUiButton::PauseControls, "Controls");
                    spawn_menu_button(menu, ViewerUiButton::PauseQuit, "Quit");
                });
            spawn_controls_sheet(overlay);
            spawn_animations_screen(overlay);
            overlay
                .spawn(menu_panel(ViewerUiPanel::PauseSettings, 340.0))
                .with_children(|settings| {
                    spawn_menu_title(settings, "Settings");
                    spawn_settings_slider(
                        settings,
                        ViewerUiText::ResolutionSummary,
                        ViewerUiButton::SetResolutionScaleFromSlider,
                        ViewerUiPanel::ResolutionScaleFill,
                    );
                    spawn_settings_slider(
                        settings,
                        ViewerUiText::AudioVolumeSummary,
                        ViewerUiButton::SetAudioVolumeFromSlider,
                        ViewerUiPanel::AudioVolumeFill,
                    );
                    spawn_checkbox_row(
                        settings,
                        ViewerUiButton::ToggleRememberPlaybackPosition,
                        ViewerUiText::RememberPlaybackPositionMark,
                        "Remember playback position",
                    );
                    spawn_checkbox_row(
                        settings,
                        ViewerUiButton::ToggleRememberCamera,
                        ViewerUiText::RememberCameraMark,
                        "Remember camera position per catalog",
                    );
                    spawn_menu_button(
                        settings,
                        ViewerUiButton::PauseCatalogOptions,
                        "Catalog options",
                    );
                    spawn_menu_button(settings, ViewerUiButton::PauseBack, "Back");
                });
        });
}

/// A settings row made of a value label above a draggable slider whose fill
/// panel is resized each frame from the setting it displays.
pub(crate) fn spawn_settings_slider(
    parent: &mut ChildBuilder,
    summary: ViewerUiText,
    slider: ViewerUiButton,
    fill: ViewerUiPanel,
) {
    parent.spawn((
        Text::new(""),
        TextFont {
            font_size: 13.0,
            ..default()
        },
        TextColor(Color::srgb(0.94, 0.96, 0.98)),
        summary,
    ));
    parent
        .spawn((
            Button,
            Node {
                min_height: Val::Px(28.0),
                width: Val::Percent(100.0),
                padding: UiRect::horizontal(Val::Px(7.0)),
                justify_content: JustifyContent::FlexStart,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderRadius::all(Val::Px(8.0)),
            BorderColor(Color::srgba(0.64, 0.70, 0.78, 0.55)),
            BackgroundColor(button_color(false, false)),
            RelativeCursorPosition::default(),
            slider,
        ))
        .with_children(|slider| {
            slider
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(8.0),
                        position_type: PositionType::Relative,
                        ..default()
                    },
                    BorderRadius::MAX,
                    BackgroundColor(Color::srgba(0.06, 0.075, 0.095, 0.95)),
                ))
                .with_children(|track| {
                    track.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(0.0),
                            height: Val::Px(8.0),
                            width: Val::Percent(60.0),
                            ..default()
                        },
                        BorderRadius::MAX,
                        BackgroundColor(Color::srgb(0.22, 0.68, 1.0)),
                        fill,
                    ));
                });
        });
}

fn spawn_start_overlay(parent: &mut ChildBuilder) {
    parent
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            // Transparent: this screen only ever shows nested inside the
            // pause menu, whose own scrim (`ViewerUiPanel::PauseOverlay`)
            // already darkens the background. A second scrim here would
            // double-darken it.
            BackgroundColor(Color::NONE),
            ViewerUiPanel::StartOverlay,
            Interaction::default(),
            FocusPolicy::Block,
        ))
        .with_children(|overlay| {
            overlay
                .spawn(menu_panel(ViewerUiPanel::StartOverlay, 460.0))
                .with_children(|menu| {
                    spawn_menu_header_with_back(menu, "Mixed Media Spatial Renderer");
                    spawn_text(menu, ViewerUiText::StartStatus);
                    for row_index in 0..START_MENU_FOLDER_ROWS {
                        spawn_start_folder_row(menu, row_index);
                    }
                    spawn_text(menu, ViewerUiText::StartRoots);
                    spawn_menu_button(menu, ViewerUiButton::StartAddFolder, "Add folder");
                    spawn_menu_button(menu, ViewerUiButton::StartClearFolders, "Clear all folders");
                    spawn_menu_button(
                        menu,
                        ViewerUiButton::StartLoadFolders,
                        "Load selected folders",
                    );
                    spawn_menu_button(
                        menu,
                        ViewerUiButton::StartUseCurrentCatalog,
                        "Use current catalog",
                    );
                    spawn_menu_button(
                        menu,
                        ViewerUiButton::StartResetLayout,
                        "Reset layout (as first opened)",
                    );
                });
        });
}

/// One selectable folder line: the (tail-truncated) path plus a remove
/// button, hidden while the row has no folder to show.
fn spawn_start_folder_row(parent: &mut ChildBuilder, row_index: usize) {
    parent
        .spawn((
            Node {
                display: Display::None,
                width: Val::Percent(100.0),
                min_height: Val::Px(28.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::SpaceBetween,
                column_gap: Val::Px(8.0),
                padding: UiRect::horizontal(Val::Px(8.0)),
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderRadius::all(Val::Px(6.0)),
            BorderColor(Color::srgba(0.64, 0.70, 0.78, 0.35)),
            BackgroundColor(Color::srgba(0.06, 0.07, 0.09, 0.85)),
            ViewerUiPanel::StartFolderRow(row_index),
        ))
        .with_children(|row| {
            row.spawn((
                Text::new(""),
                TextFont {
                    font_size: 11.5,
                    ..default()
                },
                TextColor(Color::srgb(0.86, 0.90, 0.96)),
                ViewerUiText::StartFolderPath(row_index),
            ));
            row.spawn((
                Button,
                Node {
                    width: Val::Px(24.0),
                    height: Val::Px(22.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BorderRadius::all(Val::Px(5.0)),
                BackgroundColor(button_color(false, false)),
                ViewerUiButton::StartRemoveFolder(row_index),
            ))
            .with_child((
                Text::new("x"),
                TextFont {
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgb(0.94, 0.96, 0.98)),
            ));
        });
}

pub(crate) fn menu_panel(
    panel: ViewerUiPanel,
    width: f32,
) -> (
    Node,
    BorderRadius,
    BorderColor,
    BackgroundColor,
    ViewerUiPanel,
    Interaction,
    FocusPolicy,
) {
    (
        Node {
            display: Display::None,
            width: Val::Px(width),
            padding: UiRect::all(Val::Px(14.0)),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(10.0),
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BorderRadius::all(Val::Px(8.0)),
        BorderColor(Color::srgba(0.78, 0.84, 0.92, 0.60)),
        BackgroundColor(Color::srgba(0.035, 0.040, 0.052, 0.96)),
        panel,
        Interaction::default(),
        FocusPolicy::Block,
    )
}

pub(crate) fn spawn_menu_title(parent: &mut ChildBuilder, label: &str) {
    parent.spawn((
        Text::new(label),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        TextColor(Color::srgb(0.96, 0.97, 0.99)),
    ));
}

/// A menu title with a back button pinned to the top right, for screens
/// nested under another menu level.
fn spawn_menu_header_with_back(parent: &mut ChildBuilder, label: &str) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|header| {
            header.spawn((
                Text::new(label),
                TextFont {
                    font_size: 24.0,
                    ..default()
                },
                TextColor(Color::srgb(0.96, 0.97, 0.99)),
            ));
            header
                .spawn((
                    Button,
                    Node {
                        width: Val::Px(30.0),
                        height: Val::Px(30.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderRadius::all(Val::Px(8.0)),
                    BorderColor(Color::srgba(0.7, 0.76, 0.84, 0.75)),
                    BackgroundColor(button_color(false, false)),
                    ViewerUiButton::PauseBack,
                ))
                .with_child((
                    Text::new("<-"),
                    TextFont {
                        font_size: 15.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.94, 0.96, 0.98)),
                ));
        });
}

pub(crate) fn spawn_menu_button(parent: &mut ChildBuilder, action: ViewerUiButton, label: &str) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                min_height: Val::Px(34.0),
                padding: UiRect::horizontal(Val::Px(12.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderRadius::all(Val::Px(8.0)),
            BorderColor(Color::srgba(0.7, 0.76, 0.84, 0.75)),
            BackgroundColor(button_color(false, false)),
            action,
        ))
        .with_child((
            Text::new(label),
            TextFont {
                font_size: 13.0,
                ..default()
            },
            TextColor(Color::srgb(0.94, 0.96, 0.98)),
        ));
}

fn spawn_header(parent: &mut ChildBuilder, label: &str) {
    parent.spawn((
        Text::new(label),
        TextFont {
            font_size: 15.0,
            ..default()
        },
        TextColor(Color::srgba(0.72, 0.78, 0.86, 0.85)),
    ));
}

fn spawn_speed_pill(parent: &mut ChildBuilder) {
    parent
        .spawn(pill_node(164.0, ViewerUiPanel::NavigationPill))
        .with_children(|pill| {
            spawn_pill_button(
                pill,
                ViewerUiButton::ToggleNavigationPill,
                ViewerUiText::SpeedSummary,
                13.0,
            );
            pill.spawn(expanded_panel(ViewerUiPanel::NavigationExpanded))
                .with_children(|expanded| {
                    spawn_title(expanded, "Speed");
                    spawn_text(expanded, ViewerUiText::SpeedDetails);
                    spawn_button_row(
                        expanded,
                        &[(ViewerUiButton::Slower, "-"), (ViewerUiButton::Faster, "+")],
                    );
                    spawn_menu_button(
                        expanded,
                        ViewerUiButton::FocusNearestImage,
                        "Focus nearest image",
                    );
                    spawn_menu_button(
                        expanded,
                        ViewerUiButton::JumpNextVisibleCluster,
                        "Jump to next cluster",
                    );
                });
        });
}

fn spawn_billboards_pill(parent: &mut ChildBuilder) {
    parent
        .spawn(pill_node(190.0, ViewerUiPanel::BillboardsPill))
        .with_children(|pill| {
            spawn_pill_button(
                pill,
                ViewerUiButton::ToggleBillboardsPill,
                ViewerUiText::BillboardsSummary,
                13.0,
            );
            pill.spawn(expanded_panel(ViewerUiPanel::BillboardsExpanded))
                .with_children(|expanded| {
                    spawn_title(expanded, "Billboards");
                    spawn_text(expanded, ViewerUiText::BillboardsDetails);
                    spawn_checkbox_row(
                        expanded,
                        ViewerUiButton::ToggleBillboardCoordinates,
                        ViewerUiText::BillboardCoordinatesMark,
                        "Show coordinates",
                    );
                    spawn_checkbox_row(
                        expanded,
                        ViewerUiButton::ToggleBillboardGridSnap,
                        ViewerUiText::BillboardGridSnapMark,
                        "Snap drag to grid",
                    );
                    spawn_checkbox_row(
                        expanded,
                        ViewerUiButton::ToggleBillboardHoverGrowth,
                        ViewerUiText::BillboardHoverGrowthMark,
                        "Grow on hover",
                    );
                    spawn_title(expanded, "Facing axis");
                    spawn_button_row(
                        expanded,
                        &[
                            (
                                ViewerUiButton::SetBillboardAxis(BillboardFacingAxis::X),
                                "X",
                            ),
                            (
                                ViewerUiButton::SetBillboardAxis(BillboardFacingAxis::Y),
                                "Y",
                            ),
                            (
                                ViewerUiButton::SetBillboardAxis(BillboardFacingAxis::Z),
                                "Z",
                            ),
                            (
                                ViewerUiButton::SetBillboardAxis(BillboardFacingAxis::All),
                                "All",
                            ),
                        ],
                    );
                    spawn_title(expanded, "Cache and detail");
                    spawn_button_row(
                        expanded,
                        &[
                            (ViewerUiButton::TextureLower, "Texture -"),
                            (ViewerUiButton::TextureHigher, "Texture +"),
                        ],
                    );
                    spawn_settings_slider(
                        expanded,
                        ViewerUiText::BillboardTextureBudgetSummary,
                        ViewerUiButton::SetTextureBudgetFromSlider,
                        ViewerUiPanel::BillboardTextureBudgetFill,
                    );
                });
        });
}

fn spawn_debug_pill(parent: &mut ChildBuilder) {
    parent
        .spawn(pill_node(164.0, ViewerUiPanel::DebugPill))
        .with_children(|pill| {
            spawn_pill_button(
                pill,
                ViewerUiButton::ToggleDebugPill,
                ViewerUiText::DebugSummary,
                13.0,
            );
            pill.spawn(expanded_panel(ViewerUiPanel::DebugExpanded))
                .with_children(|expanded| {
                    spawn_title(expanded, "Catalog benchmark");
                    spawn_button_row(
                        expanded,
                        &[
                            (ViewerUiButton::StartBenchmark, "Run"),
                            (ViewerUiButton::CycleBenchmarkDuration, "Duration"),
                        ],
                    );
                    spawn_text(expanded, ViewerUiText::BenchmarkStatus);
                    spawn_text(expanded, ViewerUiText::BenchmarkDetails);
                });
        });
}

fn spawn_performance_pill(parent: &mut ChildBuilder) {
    parent
        .spawn(pill_node(164.0, ViewerUiPanel::PerformancePill))
        .with_children(|pill| {
            spawn_pill_button(
                pill,
                ViewerUiButton::TogglePerformancePill,
                ViewerUiText::PerformanceSummary,
                13.0,
            );
            pill.spawn(expanded_panel(ViewerUiPanel::PerformanceExpanded))
                .with_children(|expanded| {
                    spawn_title(expanded, "Performance");
                    spawn_text(expanded, ViewerUiText::PerformanceDetails);
                    spawn_caption(expanded, "Frame time (ms)");
                    spawn_graph(expanded, PERF_GRAPH_BARS, PerfGraphBar::Time);
                    spawn_caption(expanded, "Distribution");
                    spawn_graph(expanded, PERF_HISTOGRAM_BINS, PerfGraphBar::Histogram);
                });
        });
}

pub(crate) fn pill_node(
    width: f32,
    panel: ViewerUiPanel,
) -> (
    Node,
    BorderRadius,
    BorderColor,
    BackgroundColor,
    ViewerUiPanel,
    Interaction,
    FocusPolicy,
) {
    (
        Node {
            width: Val::Px(width),
            padding: UiRect::axes(Val::Px(8.0), Val::Px(7.0)),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.0),
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BorderRadius::MAX,
        BorderColor(Color::srgba(0.70, 0.76, 0.84, 0.55)),
        BackgroundColor(Color::srgba(0.03, 0.035, 0.045, 0.86)),
        panel,
        // Panels join UI focus so `UiInputCapture` sees the cursor over them
        // and world click handlers stand down, even between the buttons.
        Interaction::default(),
        FocusPolicy::Block,
    )
}

pub(crate) fn expanded_panel(panel: ViewerUiPanel) -> (Node, ViewerUiPanel) {
    (
        Node {
            display: Display::None,
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.0),
            padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
            ..default()
        },
        panel,
    )
}

// The pill container already draws the border, so the header button stays
// borderless and transparent — otherwise a collapsed pill shows two nested
// outlines.
pub(crate) fn spawn_pill_button(
    parent: &mut ChildBuilder,
    action: ViewerUiButton,
    text_kind: ViewerUiText,
    font_size: f32,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                min_height: Val::Px(30.0),
                padding: UiRect::horizontal(Val::Px(8.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BorderRadius::MAX,
            BackgroundColor(header_button_color(false)),
            action,
        ))
        .with_child((
            Text::new(""),
            TextFont {
                font_size,
                ..default()
            },
            TextColor(Color::srgb(0.94, 0.96, 0.98)),
            text_kind,
        ));
}

pub(crate) fn spawn_button_row(parent: &mut ChildBuilder, buttons: &[(ViewerUiButton, &str)]) {
    parent
        .spawn(Node {
            height: Val::Px(30.0),
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|row| {
            for (action, label) in buttons {
                spawn_button(row, *action, label);
            }
        });
}

pub(crate) fn spawn_checkbox_row(
    parent: &mut ChildBuilder,
    action: ViewerUiButton,
    mark_text: ViewerUiText,
    label: &str,
) {
    spawn_checkbox_row_with_label(parent, action, mark_text, label, ());
}

/// A checkbox row whose label text also carries `label_bundle`.
pub(crate) fn spawn_checkbox_row_with_label(
    parent: &mut ChildBuilder,
    action: ViewerUiButton,
    mark_text: ViewerUiText,
    label: &str,
    label_bundle: impl Bundle,
) {
    parent
        .spawn(Node {
            height: Val::Px(26.0),
            align_items: AlignItems::Center,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Button,
                Node {
                    width: Val::Px(20.0),
                    height: Val::Px(20.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border: UiRect::all(Val::Px(1.0)),
                    ..default()
                },
                BorderRadius::all(Val::Px(4.0)),
                BorderColor(Color::srgba(0.7, 0.76, 0.84, 0.75)),
                BackgroundColor(button_color(false, false)),
                action,
            ))
            .with_child((
                Text::new(""),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgb(0.94, 0.96, 0.98)),
                mark_text,
            ));
            row.spawn((
                Text::new(label),
                TextFont {
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgb(0.86, 0.90, 0.96)),
                label_bundle,
            ));
        });
}

pub(crate) fn spawn_button(parent: &mut ChildBuilder, action: ViewerUiButton, label: &str) {
    parent
        .spawn((
            Button,
            Node {
                min_width: Val::Px(46.0),
                height: Val::Px(30.0),
                padding: UiRect::horizontal(Val::Px(10.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            BorderRadius::MAX,
            BorderColor(Color::srgba(0.7, 0.76, 0.84, 0.75)),
            BackgroundColor(button_color(false, false)),
            action,
        ))
        .with_child((
            Text::new(label),
            TextFont {
                font_size: 12.0,
                ..default()
            },
            TextColor(Color::srgb(0.94, 0.96, 0.98)),
        ));
}

fn spawn_caption(parent: &mut ChildBuilder, label: &str) {
    parent.spawn((
        Text::new(label),
        TextFont {
            font_size: 11.0,
            ..default()
        },
        TextColor(Color::srgba(0.70, 0.76, 0.84, 0.85)),
    ));
}

pub(crate) fn spawn_title(parent: &mut ChildBuilder, label: &str) {
    parent.spawn((
        Text::new(label),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.96, 0.97, 0.99)),
    ));
}

pub(crate) fn spawn_text(parent: &mut ChildBuilder, text_kind: ViewerUiText) {
    parent.spawn((
        Text::new(""),
        TextFont {
            font_size: 12.0,
            ..default()
        },
        TextColor(Color::srgb(0.78, 0.82, 0.88)),
        text_kind,
    ));
}

fn spawn_graph(parent: &mut ChildBuilder, count: usize, bar: fn(usize) -> PerfGraphBar) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Px(40.0),
            align_items: AlignItems::FlexEnd,
            column_gap: Val::Px(2.0),
            ..default()
        })
        .with_children(|graph| {
            for index in 0..count {
                graph.spawn((
                    Node {
                        width: Val::Px(4.0),
                        height: Val::Px(2.0),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.32, 0.74, 0.93, 0.82)),
                    bar(index),
                ));
            }
        });
}

pub(crate) fn display_if(value: bool) -> Display {
    if value {
        Display::Flex
    } else {
        Display::None
    }
}

pub(crate) fn button_color(active: bool, hovered: bool) -> Color {
    match (active, hovered) {
        (true, true) => Color::srgb(0.18, 0.50, 0.42),
        (true, false) => Color::srgb(0.12, 0.40, 0.34),
        (false, true) => Color::srgb(0.18, 0.20, 0.25),
        (false, false) => Color::srgb(0.11, 0.13, 0.17),
    }
}

/// Pill header buttons are transparent so the pill container provides the
/// only visible outline; hovering shows a subtle highlight.
pub(crate) fn header_button_color(hovered: bool) -> Color {
    if hovered {
        Color::srgba(0.85, 0.90, 0.98, 0.10)
    } else {
        Color::NONE
    }
}

pub(crate) fn set_header_color(mut color: Mut<BackgroundColor>, interaction: &Interaction) {
    *color = header_button_color(*interaction == Interaction::Hovered).into();
}

pub(crate) fn set_button_color(
    mut color: Mut<BackgroundColor>,
    active: bool,
    interaction: &Interaction,
) {
    let hovered = *interaction == Interaction::Hovered;
    *color = button_color(active, hovered).into();
}

pub(crate) fn set_disabled_button_color(mut color: Mut<BackgroundColor>) {
    *color = disabled_button_color().into();
}

pub(crate) fn disabled_button_color() -> Color {
    Color::srgba(0.08, 0.09, 0.12, 0.68)
}

fn button_disabled(button: &ViewerUiButton, controls: &BillboardControls) -> bool {
    match button {
        ViewerUiButton::TextureLower | ViewerUiButton::TextureHigher => {
            controls.max_texture_side == 0
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_is_base_speed_without_cruise_pressure() {
        let settings = NavigationSettings::new(4.0, 10.0);

        let status = navigation_status(&settings);

        assert_eq!(status.cruise_pressure, 0.0);
        assert_eq!(status.effective_speed, 4.0);
    }

    #[test]
    fn cruise_pressure_scales_effective_speed() {
        let mut settings = NavigationSettings::new(4.0, 10.0);
        settings.cruise_pressure = 1.0;

        let status = navigation_status(&settings);

        assert_eq!(status.effective_speed, 4.0 * (1.0 + CRUISE_SPEED_GAIN));
        assert!(status.effective_speed > settings.base_speed);
    }

    #[test]
    fn cruise_pressure_rises_moving_and_decays_when_stopped() {
        let mut settings = NavigationSettings::new(4.0, 10.0);

        settings.update_cruise_pressure(Vec3::X, Vec3::X, 1.0);
        let cruising = settings.cruise_pressure;
        assert!(cruising > 0.0);

        settings.update_cruise_pressure(Vec3::ZERO, Vec3::X, 1.0);
        assert!(settings.cruise_pressure < cruising);
    }

    #[test]
    fn velocity_multiplier_scales_effective_speed() {
        let status = NavigationStatus {
            velocity_multiplier: 1.0,
            effective_speed: 8.0,
            ..default()
        }
        .with_velocity_multiplier(1.5);

        assert_eq!(status.velocity_multiplier, 1.5);
        assert_eq!(status.effective_speed, 12.0);
    }

    #[test]
    fn render_resolution_defaults_to_window_size() {
        let settings = RenderResolutionSettings::default();

        assert_eq!(
            settings.target_size(UVec2::new(1600, 1000)),
            UVec2::new(1600, 1000)
        );
    }

    #[test]
    fn render_resolution_slider_maps_to_scale_range() {
        let mut settings = RenderResolutionSettings::default();

        settings.set_scale_from_slider(0.0);
        assert_eq!(
            settings.target_size(UVec2::new(1600, 1000)),
            UVec2::new(400, 250)
        );

        settings.set_scale_from_slider(1.0);
        assert_eq!(
            settings.target_size(UVec2::new(1600, 1000)),
            UVec2::new(2400, 1500)
        );
    }

    #[test]
    fn axis_gizmo_request_marks_face_for_camera_snap() {
        let mut state = AxisGizmoState::default();

        state.request_face(AxisGizmoFace::PositiveZ);

        assert!(state.is_selected(AxisGizmoFace::PositiveZ));
        assert_eq!(state.take_requested_face(), Some(AxisGizmoFace::PositiveZ));
        assert_eq!(state.take_requested_face(), None);
    }

    #[test]
    fn axis_gizmo_repeat_click_snaps_to_opposite_face() {
        let mut state = AxisGizmoState::default();

        state.request_face(AxisGizmoFace::PositiveZ);
        state.take_requested_face();
        state.request_face(AxisGizmoFace::PositiveZ);

        assert!(state.is_selected(AxisGizmoFace::NegativeZ));
        assert_eq!(state.take_requested_face(), Some(AxisGizmoFace::NegativeZ));
    }

    #[test]
    fn texture_budget_slider_round_trips_and_stays_in_range() {
        let mut controls = BillboardControls::new(DEFAULT_TEXTURE_BUDGET_MIB, 0);

        controls.set_texture_budget_from_slider(0.0);
        assert_eq!(controls.texture_budget_mib, MIN_TEXTURE_BUDGET_MIB);
        assert_eq!(controls.texture_budget_slider_percent(), 0.0);

        controls.set_texture_budget_from_slider(1.0);
        assert_eq!(controls.texture_budget_mib, MAX_TEXTURE_BUDGET_MIB);
        assert!((controls.texture_budget_slider_percent() - 100.0).abs() < 0.01);

        // Out-of-range drags are clamped rather than producing a wild budget.
        controls.set_texture_budget_from_slider(-5.0);
        assert_eq!(controls.texture_budget_mib, MIN_TEXTURE_BUDGET_MIB);
        controls.set_texture_budget_from_slider(9.0);
        assert_eq!(controls.texture_budget_mib, MAX_TEXTURE_BUDGET_MIB);

        // The curve is monotonic, so dragging right never lowers the budget.
        let mut previous = 0;
        for step in 0..=20 {
            controls.set_texture_budget_from_slider(step as f32 / 20.0);
            assert!(controls.texture_budget_mib >= previous);
            previous = controls.texture_budget_mib;
        }

        // The byte view the loader reads must agree with the MiB the slider
        // shows, or the panel would report a budget the cache is not using.
        controls.set_texture_budget_from_slider(0.5);
        assert_eq!(
            controls.texture_budget_bytes(),
            controls.texture_budget_mib as usize * 1024 * 1024
        );
    }

    #[test]
    fn update_viewer_ui_systems_run_without_query_conflicts() {
        let mut app = App::new();
        app.insert_resource(NavigationSettings::new(4.0, 10.0));
        app.insert_resource(ControlPanelState::default());
        app.insert_resource(BillboardFacingSettings::default());
        app.insert_resource(BillboardControls::new(512, 1024));
        app.insert_resource(BillboardStats::default());
        app.insert_resource(DebugSettings::default());
        app.insert_resource(BenchmarkControls::default());
        app.insert_resource(PerformanceMetrics::default());
        app.insert_resource(PauseMenuState::default());
        app.insert_resource(ViewSettings::default());
        app.insert_resource(RenderResolutionSettings::default());
        app.insert_resource(AudioSettings::default());
        app.insert_resource(PlaybackSettings::default());
        app.insert_resource(StartMenuState::default());
        app.insert_resource(FolderControls::default());
        app.insert_resource(ControlBindings::default());
        app.init_resource::<AnimationSettings>();
        app.add_systems(
            Update,
            (
                update_navigation_text,
                update_control_text,
                update_billboard_text,
                update_folder_text,
                update_debug_text,
                update_performance_text,
                update_resolution_text,
                update_start_text,
                update_animation_text,
            )
                .chain(),
        );
        app.add_systems(
            Update,
            (
                update_navigation_panels,
                update_control_panels,
                update_billboard_panels,
                update_folder_panels,
                update_debug_panels,
                update_performance_panels,
                update_pause_panels,
                update_start_panels,
                update_animation_panels,
                update_perf_graph_bars,
            )
                .chain(),
        );
        app.add_systems(
            Update,
            (
                update_navigation_button_colors,
                update_control_button_colors,
                update_billboard_button_colors,
                update_folder_button_colors,
                update_debug_button_colors,
                update_performance_button_colors,
                update_pause_button_colors,
                update_resolution_button_colors,
                update_start_button_colors,
                update_animation_button_colors,
            )
                .chain(),
        );

        app.update();
    }

    #[test]
    fn escape_steps_back_one_level_at_a_time() {
        let mut pause = PauseMenuState::default();

        pause.escape_pressed();
        assert!(pause.paused);
        assert_eq!(pause.screen, PauseScreen::Main);

        pause.open_settings();
        pause.open_catalog_options();
        assert_eq!(pause.screen, PauseScreen::CatalogOptions);

        pause.escape_pressed();
        assert!(pause.paused);
        assert_eq!(pause.screen, PauseScreen::Settings);

        pause.escape_pressed();
        assert!(pause.paused);
        assert_eq!(pause.screen, PauseScreen::Main);

        pause.escape_pressed();
        assert!(!pause.paused);
    }

    #[test]
    fn back_from_catalog_options_returns_to_settings_not_closed() {
        let mut pause = PauseMenuState::default();
        pause.escape_pressed();
        pause.open_settings();
        pause.open_catalog_options();

        pause.back();

        assert!(pause.paused);
        assert_eq!(pause.screen, PauseScreen::Settings);
    }

    #[test]
    fn nearest_in_direction_prefers_same_row_over_nearer_diagonal() {
        let targets = NavigationTargets::from_positions([
            Vec3::ZERO,
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(4.0, 4.0, 0.0),
            Vec3::new(12.0, 0.0, 0.0),
            Vec3::new(-6.0, 0.0, 0.0),
        ]);

        // The diagonal at (4,4,0) is closer overall, but the same-row
        // neighbor wins because off-axis distance is penalized.
        assert_eq!(
            targets.nearest_in_direction(Vec3::ZERO, Vec3::X, 1.5, |_| true),
            Some(Vec3::new(6.0, 0.0, 0.0))
        );
        // Behind the origin along +X only (-6,0,0) and the origin itself sit
        // below the advance threshold; both are excluded.
        assert_eq!(
            targets.nearest_in_direction(Vec3::new(12.0, 0.0, 0.0), Vec3::X, 1.5, |_| true),
            None
        );
        assert_eq!(
            targets.nearest_in_direction(Vec3::ZERO, Vec3::ZERO, 1.5, |_| true),
            None
        );
        // A filtered-out best candidate falls through to the next one.
        assert_eq!(
            targets.nearest_in_direction(Vec3::ZERO, Vec3::X, 1.5, |position| position.x > 6.0),
            Some(Vec3::new(12.0, 0.0, 0.0))
        );
    }
}
