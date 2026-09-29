use anyhow::{bail, Context, Result};
use bevy::core::{TaskPoolOptions, TaskPoolPlugin, TaskPoolThreadAssignmentPolicy};
use bevy::ecs::system::SystemParam;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::math::primitives::Rectangle;
use bevy::prelude::*;
use bevy::render::{
    camera::RenderTarget,
    render_asset::{RenderAssetBytesPerFrame, RenderAssetUsages},
    render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    renderer::RenderDevice,
    settings::{Backends, WgpuSettings},
    view::RenderLayers,
    RenderPlugin,
};
use bevy::sprite::AlphaMode2d;
use bevy::window::{CursorGrabMode, PrimaryWindow, WindowResizeConstraints};
use spatial_api::{ProjectionPage, ProjectionRequest, SpatialApiClient};
use spatial_geometry::{estimate_smallest_axis_gap, median_position, projection_bounds, Bounds3};
use spatial_viewer_ui::{
    collect_performance_metrics, handle_audio_buttons, handle_billboard_buttons,
    handle_control_buttons, handle_control_dropdown_scroll, handle_control_keyboard,
    handle_debug_buttons, handle_navigation_buttons, handle_pause_menu_buttons,
    handle_performance_buttons, handle_resolution_buttons, handle_start_menu_buttons,
    navigation_status, rebuild_control_widgets, spawn_viewer_ui, update_audio_text,
    update_billboard_button_colors, update_billboard_panels, update_billboard_text,
    update_control_button_colors, update_control_panels, update_control_text,
    update_debug_button_colors, update_debug_panels, update_debug_text,
    update_navigation_button_colors, update_navigation_panels, update_navigation_text,
    update_pause_button_colors, update_pause_panels, update_perf_graph_bars,
    update_performance_button_colors, update_performance_panels, update_performance_text,
    update_resolution_button_colors, update_resolution_text, update_start_button_colors,
    update_start_panels, update_start_text, update_ui_input_capture, AudioSettings, AxisGizmoState,
    BenchmarkControls, BillboardControls, BillboardFacingAxis, BillboardFacingSettings,
    BillboardStats, ContextMenuPlugin, ContextMenuSystems, ControlPanelState, DebugSettings,
    NavigationRequest, NavigationSettings, NavigationTargets, PauseMenuState, PerformanceMetrics,
    PlaybackSettings, RenderResolutionSettings, StartMenuState, UiInputCapture,
};

mod audio_stream;
mod axis_gizmo;
mod background_work;
mod benchmark;
mod billboard_menu;
mod catalog_load;
mod catalog_session;
mod cli;
mod decode_budget;
mod ffmpeg_pipe;
mod image_loading;
mod load_sampling;
mod manual_spacing;
mod media_decode;
mod media_probe;
mod media_settings;
use media_decode::{BillboardTextureEncoding, BillboardTextureFormat};
mod performance;
mod point_cloud;
mod video_controls;
mod video_stream;
mod video_strip;
mod video_strip_layout;

use audio_stream::AudioPlaybackState;
use axis_gizmo::{
    axis_gizmo_generic_up_vector, axis_gizmo_up_vector, axis_gizmo_view_direction,
    handle_axis_gizmo_clicks, load_label_font, refresh_axis_gizmo_labels, spawn_axis_gizmo_3d,
    sync_axis_gizmo_camera, sync_axis_gizmo_labels, update_axis_gizmo_face_highlight,
    update_axis_gizmo_hover, update_axis_gizmo_label_hover, update_axis_gizmo_viewport,
};
use background_work::spawn_background;
use billboard_menu::{
    billboard_menu_systems, BillboardMenuCommand, BillboardMenuTarget, ClipboardHandle,
    WorldRightClick,
};
use catalog_load::{
    apply_control_submit_requests, handle_start_menu_requests, point_cloud_points,
    poll_catalog_load_task, poll_folder_pick_task, projection_billboard_points,
    restore_last_catalog, CatalogLoadTask, FolderPickTask, InitialPlayerPlacement,
    LastPickedFolder, RestoredCatalogRoots,
};
use catalog_session::load_last_catalog_roots;
use cli::ViewerArgs;
use decode_budget::DecodeBudget;
use image_loading::{
    billboard_rotation, billboard_world_size, create_billboard_mesh, face_billboards_to_camera,
    publish_image_loading_stats, receive_image_loads, schedule_image_loads,
    spawn_visible_coordinate_labels, update_billboard_coordinate_label_visibility,
    BillboardAxisLabels, BillboardLabelFont, BillboardPoint, BillboardWorldSize, ImageLoadingState,
    MediaBillboard, BILLBOARD_GPU_UPLOAD_BYTES_PER_FRAME,
};
use manual_spacing::{
    handle_selection_and_drag, sync_selection_highlights, sync_translate_gizmo, ImagePointIndex,
    SelectionHighlights, SelectionState, TranslateGizmo, TranslateGizmoCamera,
};
use media_probe::{receive_media_probes, MediaProbes};
use media_settings::{save_media_settings, save_media_settings_on_exit, MediaSettings};
use point_cloud::{
    apply_point_cloud_edits, spawn_boundary_walls, update_view_bounds, PointCloud, ViewBounds,
};
use video_controls::{
    advance_video_clocks, apply_video_playback_frame, clear_stale_video_controls,
    handle_video_keyboard, release_slider_drag, remember_playback_positions,
    remember_positions_on_exit, reveal_hovered_videos, sync_video_audio, VideoControlsState,
};
use video_stream::VideoPlaybackState;
use video_strip::{
    handle_video_strip_input, layout_video_strip_parts, sync_video_strips,
    update_video_strip_labels, VideoControlsFont, VideoStripAssets,
};

#[derive(Resource)]
struct ExplorerScene {
    projection: spatial_api::ProjectionPage,
    bounds: Bounds3,
    image_points: Vec<BillboardPoint>,
    client: SpatialApiClient,
    projection_limit: usize,
    coordinate_spacing: f32,
    duplicate_spacing: f32,
    texture_budget_mib: u32,
    image_concurrency: usize,
    max_texture_side: u32,
    billboard_world_size: f32,
    camera_speed: f32,
    navigation_reference_distance: f32,
    initial_view_distance: f32,
    axis_labels: [String; 3],
    debug_probe_billboard: bool,
}

impl ExplorerScene {
    /// A request carrying this scene's layout parameters. Callers add the
    /// roots and control values.
    fn projection_request(&self) -> ProjectionRequest {
        ProjectionRequest::new(
            self.coordinate_spacing,
            self.duplicate_spacing,
            self.billboard_world_size,
            self.projection_limit,
        )
    }
}

#[derive(Component)]
struct FlyCamera {
    yaw: f32,
    pitch: f32,
    sensitivity: f32,
    velocity: Vec3,
    velocity_response: f32,
}

const CAMERA_VELOCITY_RESPONSE_PER_SECOND: f32 = 6.5;
const CAMERA_STOP_EPSILON: f32 = 0.0001;
/// Raw mouse motion (device pixels) a held right button must accumulate
/// before it counts as a look-drag; below this it is a plain right click and
/// the cursor is left alone.
const RIGHT_DRAG_MOTION_THRESHOLD: f32 = 4.0;
/// Fraction of the coordinate spacing an arrow-key teleport candidate must
/// advance along the travel axis, so the currently focused image (and its
/// duplicate-spread jitter) is never picked as its own neighbor.
const TELEPORT_MIN_ADVANCE_FACTOR: f32 = 0.25;
/// Squared camera displacement (world units) beyond which the anchor from the
/// previous arrow-key teleport is considered stale; anything the fly camera,
/// gizmo, or focus buttons do moves the camera far more than this.
const TELEPORT_ANCHOR_POSITION_EPSILON_SQUARED: f32 = 0.0001;
const MIN_WINDOW_WIDTH: f32 = 960.0;
const MIN_WINDOW_HEIGHT: f32 = 540.0;
/// Threads left for everything that is not a media decode: rendering,
/// systems, and Bevy's own parallel work. The async-compute pool is sized
/// from the decode budget but never at the expense of this many cores.
///
/// Decode threads are CPU- and memory-bandwidth-heavy, and measurably
/// contend with the render thread: on a 32-thread machine, sweeping a 3000
/// point catalog, worst-frame time scaled almost linearly with the number of
/// concurrent decodes (4 workers 10 ms, 8 workers 18 ms, 12 workers 27 ms,
/// 24 workers 49 ms) while the sum of all update stages stayed under 1 ms —
/// so the cost lands outside the update, in render and present. More decode
/// concurrency buys throughput only until it starts costing frames.
const NON_DECODE_THREAD_RESERVE: usize = 8;

/// Sizes Bevy's task pools so the async-compute pool can actually run
/// `decode_budget` concurrent decodes.
///
/// Every poster and video first-frame decode is dispatched onto
/// `AsyncComputeTaskPool` by [`background_work::spawn_background`], and
/// Bevy's default policy caps that pool at 4 threads (25% of cores, hard
/// `max_threads: 4`). That cap sat below `--image-concurrency` at its default
/// of 12, so the decode budget could never be spent: the extra work simply
/// queued behind 4 threads, and raising `--image-concurrency` did nothing.
///
/// The pool is sized to the budget, bounded so the machine keeps
/// [`NON_DECODE_THREAD_RESERVE`] cores for rendering and systems, and never
/// shrinks below Bevy's own default.
fn media_decode_task_pool_options(decode_budget: usize) -> TaskPoolOptions {
    const BEVY_DEFAULT_ASYNC_COMPUTE_THREADS: usize = 4;
    let total_threads = bevy::tasks::available_parallelism();
    let spare = total_threads.saturating_sub(NON_DECODE_THREAD_RESERVE);
    let decode_threads = decode_budget
        .min(spare)
        .max(BEVY_DEFAULT_ASYNC_COMPUTE_THREADS);
    TaskPoolOptions {
        async_compute: TaskPoolThreadAssignmentPolicy {
            min_threads: decode_threads,
            max_threads: decode_threads,
            // Ignored once min and max agree, but must stay in range.
            percent: 1.0,
        },
        ..default()
    }
}
pub(crate) const PRESENTATION_RENDER_LAYER: usize = 2;

/// Tracks a potential right-button look-drag so the cursor is hidden and
/// pinned in place only once real drag motion happens — a plain right click
/// never touches the cursor.
#[derive(Resource, Default)]
struct RightDragCursorState {
    /// Cursor position (logical pixels) when the right button went down in
    /// the world; `None` while the button is up or the press began over UI.
    anchor: Option<Vec2>,
    /// Raw motion accumulated since the press, before the drag threshold.
    accumulated_motion: f32,
    /// The threshold was crossed: the cursor is hidden and pinned to
    /// `anchor` until release.
    dragging: bool,
}

/// Frees a large value off the main thread; deallocating tens of thousands
/// of small allocations is itself a frame-visible cost.
fn drop_in_background<T: Send + 'static>(value: T) {
    spawn_background(move || drop(value));
}

fn point_cloud_point_size(bounds: &Bounds3) -> f32 {
    (bounds.max_extent * 0.01).clamp(0.08, 0.35)
}

#[derive(Resource)]
struct SceneRenderTarget {
    image: Handle<Image>,
}

#[derive(Component)]
struct RenderedSceneCanvas;

#[derive(Component)]
struct PresentationCamera;

type BillboardEntitiesQuery<'w, 's> = Query<'w, 's, (Entity, &'static MediaBillboard)>;
type FlyCameraTransformQuery<'w, 's> =
    Query<'w, 's, (&'static mut Transform, &'static mut FlyCamera)>;
#[derive(SystemParam)]
struct FlyCameraInput<'w, 's> {
    time: Res<'w, Time>,
    keyboard: Res<'w, ButtonInput<KeyCode>>,
    mouse_buttons: Res<'w, ButtonInput<MouseButton>>,
    mouse_motion: EventReader<'w, 's, MouseMotion>,
    mouse_wheel: EventReader<'w, 's, MouseWheel>,
}

type FlyCameraProjectionQuery<'w, 's> = Query<'w, 's, &'static mut Projection, With<FlyCamera>>;

fn main() -> Result<()> {
    let args = ViewerArgs::parse()?;
    let client = SpatialApiClient::new(&args.api);
    let health = client
        .health()
        .with_context(|| format!("Failed to reach viewer API at {}", args.api))?;
    if health.status != "ok" {
        bail!("Viewer API returned non-ok status: {}", health.status);
    }

    let image_world_size = billboard_world_size(args.spacing, args.billboard_scale);
    // No catalog is loaded at startup: the scene starts empty and the last
    // used roots (if any) are loaded through the start menu's background
    // task once the app is running, so the window appears immediately.
    let projection = ProjectionPage {
        axis_labels: [None, None, None],
        coordinate_spacing: args.spacing,
        duplicate_spacing: args.duplicates,
        sprite_world_height: image_world_size,
        offset: 0,
        limit: args.limit,
        total: 0,
        points: Vec::new(),
    };
    let bounds = projection_bounds(&projection.points);
    let image_points = projection_billboard_points(&projection);
    let nearest_gap = estimate_smallest_axis_gap(&projection.points).unwrap_or(args.spacing);
    let camera_speed = navigation_speed(nearest_gap, args.spacing);
    let navigation_reference_distance = args.spacing.max(image_world_size).max(1.0);
    let initial_view_distance = initial_camera_distance(args.spacing, image_world_size);
    let axis_labels = projection.axis_labels.clone().map(axis_label);
    let last_catalog_roots =
        if std::env::var_os("SPATIAL_VIEWER_PERF_CONFIG").is_some() {
            serde_json::from_str(&std::env::var(performance::PERF_ROOTS_ENV).with_context(
                || format!("Performance run requires {}", performance::PERF_ROOTS_ENV),
            )?)
            .context("Invalid performance catalog roots")?
        } else {
            load_last_catalog_roots()
        };

    if last_catalog_roots.is_empty() {
        println!("No previous catalog to restore; open Catalog options to load one.");
    } else {
        println!(
            "Restoring last catalog from {} root(s):",
            last_catalog_roots.len()
        );
        for root in &last_catalog_roots {
            println!("  {root}");
        }
    }
    let texture_side = if args.max_texture_side == 0 {
        "source".to_owned()
    } else {
        format!("{} px max", args.max_texture_side)
    };
    println!(
        "Media loading: {} MiB texture VRAM budget, {} local decode workers, {texture_side} resolution, billboard size {:.2}",
        args.texture_budget_mib, args.image_concurrency, image_world_size
    );

    // Video playback streams on threads of its own (see `ffmpeg_pipe`), so
    // the pool and its budget serve image loads alone.
    let decode_budget = args.image_concurrency;
    println!(
        "Media decode: {} async-compute threads for a decode budget of {decode_budget} ({} logical cores)",
        media_decode_task_pool_options(decode_budget)
            .async_compute
            .max_threads,
        bevy::tasks::available_parallelism(),
    );

    let mut app = App::new();
    if let Some(seconds) = args.benchmark_seconds {
        app.insert_resource(benchmark::HeadlessBenchmark::new(seconds));
    }
    // Loaded before the UI starts, so the settings screen opens on the
    // remembered values. Benchmark and harness runs neither follow nor
    // change the user's settings.
    let measuring = args.benchmark_seconds.is_some()
        || std::env::var_os(performance::PERF_CONFIG_ENV).is_some();
    let settings = if measuring {
        media_settings::LoadedSettings {
            media: MediaSettings::in_memory(),
            playback: PlaybackSettings::default(),
            audio: AudioSettings::default(),
        }
    } else {
        MediaSettings::load()
    };
    app.insert_resource(ClearColor(Color::srgb(0.025, 0.027, 0.032)))
        .insert_resource(settings.audio)
        .insert_resource(settings.playback)
        .insert_resource(settings.media)
        .init_resource::<MediaProbes>()
        .insert_resource(VideoPlaybackState::new(args.max_video_fps))
        .insert_resource(ExplorerScene {
            projection,
            bounds,
            image_points,
            client,
            projection_limit: args.limit,
            coordinate_spacing: args.spacing,
            duplicate_spacing: args.duplicates,
            texture_budget_mib: args.texture_budget_mib,
            image_concurrency: args.image_concurrency,
            max_texture_side: args.max_texture_side,
            billboard_world_size: image_world_size,
            camera_speed,
            navigation_reference_distance,
            initial_view_distance,
            axis_labels,
            debug_probe_billboard: args.debug_probe_billboard,
        })
        .insert_resource(ControlPanelState::new(args.controls.clone()))
        .insert_resource(RestoredCatalogRoots(last_catalog_roots))
        .init_resource::<CatalogLoadTask>()
        .init_resource::<InitialPlayerPlacement>()
        .init_resource::<FolderPickTask>()
        .init_resource::<LastPickedFolder>()
        .init_resource::<UiInputCapture>()
        .init_resource::<RightDragCursorState>()
        .add_event::<WorldRightClick>()
        .init_resource::<BillboardMenuTarget>()
        .init_non_send_resource::<ClipboardHandle>()
        .add_plugins(ContextMenuPlugin::<BillboardMenuCommand>::default())
        .init_resource::<SelectionState>()
        .init_resource::<SelectionHighlights>()
        .init_resource::<ImagePointIndex>()
        // Overwritten by setup_scene once the render device is known; present
        // from the start so the scheduler can never miss the resource.
        .init_resource::<BillboardTextureEncoding>()
        .init_resource::<TranslateGizmo>()
        .add_plugins(
            DefaultPlugins
                .set(TaskPoolPlugin {
                    task_pool_options: media_decode_task_pool_options(decode_budget),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Mixed Media Spatial Renderer".to_owned(),
                        resolution: (1600.0, 1000.0).into(),
                        resize_constraints: WindowResizeConstraints {
                            min_width: MIN_WINDOW_WIDTH,
                            min_height: MIN_WINDOW_HEIGHT,
                            ..default()
                        },
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: WgpuSettings {
                        backends: Some(preferred_render_backends()),
                        // NOTE: TEXTURE_COMPRESSION_BC is deliberately NOT
                        // requested here even though billboard textures use
                        // it. Bevy ORs `features` into the device request
                        // unconditionally and unwraps the result, so naming
                        // a feature the adapter lacks is a hard panic at
                        // startup rather than the RGBA8 fallback we want.
                        // Bevy's default `Functionality` priority already
                        // enables every feature the adapter reports; the
                        // fallback decision is made from the negotiated
                        // device in `detect_billboard_texture_format`.
                        ..default()
                    }
                    .into(),
                    ..default()
                }),
        )
        .add_plugins(performance::PerformanceHarnessPlugin)
        // Feeds the in-app catalog benchmark's hardware section. Sampling is
        // throttled internally and runs off the main thread.
        .add_plugins((
            bevy::diagnostic::SystemInformationDiagnosticsPlugin,
            bevy::render::diagnostic::RenderDiagnosticsPlugin,
        ))
        .init_resource::<benchmark::BenchmarkRun>()
        .add_systems(Startup, (setup_scene, disable_camera_msaa).chain())
        .add_systems(PostUpdate, disable_camera_msaa)
        // Between the menu reading this frame's presses and redrawing, so a
        // command's effect shows the frame it is picked.
        .add_systems(
            Update,
            billboard_menu_systems()
                .after(manage_right_drag_cursor)
                .after(ContextMenuSystems::Input)
                .before(ContextMenuSystems::Render),
        )
        // Positions first, so the settings written on exit include them.
        .add_systems(
            Last,
            (remember_positions_on_exit, save_media_settings_on_exit).chain(),
        )
        .add_systems(Update, enable_gpu_upload_budget_after_startup)
        // Sampled after bevy_ui focus so every Update system shares one
        // consistent picture of whether the pointer belongs to the UI.
        .add_systems(
            PreUpdate,
            update_ui_input_capture.after(bevy::ui::UiSystem::Focus),
        )
        .init_resource::<performance::FrameStageProbe>()
        // Clears the previous frame's stamps before recording this one, so
        // the probe measures a single frame whether or not anything is
        // draining it.
        .add_systems(First, performance::begin_frame_stages)
        .add_systems(
            Update,
            (
                performance::stamp_stage("pre_update"),
                (collect_performance_metrics, enforce_min_window_size).chain(),
                (
                    handle_navigation_buttons,
                    handle_control_buttons,
                    handle_control_keyboard,
                    handle_control_dropdown_scroll,
                    // Rebuilds the widget subtree after this frame's clicks
                    // are applied and before anything queries the new
                    // entities, so a press is never lost to a rebuild.
                    rebuild_control_widgets,
                    handle_billboard_buttons,
                    handle_debug_buttons,
                    handle_performance_buttons,
                    handle_pause_menu_buttons,
                    handle_resolution_buttons,
                    handle_audio_buttons,
                    handle_start_menu_buttons,
                )
                    .chain(),
                performance::stamp_stage("ui_buttons"),
                (
                    (
                        restore_last_catalog,
                        handle_start_menu_requests,
                        poll_folder_pick_task,
                        apply_control_submit_requests,
                        performance::stamp_stage("menu_requests"),
                        poll_catalog_load_task,
                        performance::stamp_stage("poll_catalog_load_task"),
                    )
                        .chain(),
                    (
                        apply_render_resolution_requests,
                        sync_render_target_presentation,
                        update_axis_gizmo_viewport,
                        apply_navigation_ui_requests,
                    )
                        .chain(),
                )
                    .chain(),
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                (
                    toggle_pause_menu,
                    manage_right_drag_cursor,
                    handle_axis_gizmo_clicks,
                    update_axis_gizmo_hover,
                    update_axis_gizmo_face_highlight,
                    apply_axis_gizmo_requests,
                    fly_camera_controls,
                    teleport_on_arrow_keys,
                    sync_axis_gizmo_camera,
                    refresh_axis_gizmo_labels,
                    sync_axis_gizmo_labels,
                    update_axis_gizmo_label_hover,
                )
                    .chain(),
                (
                    (
                        clear_stale_video_controls,
                        release_slider_drag,
                        handle_video_strip_input,
                        handle_video_keyboard,
                        handle_selection_and_drag,
                        reveal_hovered_videos,
                        sync_translate_gizmo,
                        advance_video_clocks,
                        sync_video_audio,
                        remember_playback_positions,
                        save_media_settings,
                        update_decode_budget,
                    )
                        .chain(),
                    (
                        performance::stamp_stage("ui_and_camera"),
                        receive_media_probes,
                        apply_video_playback_frame,
                        performance::stamp_stage("video_frames"),
                        schedule_image_loads,
                        performance::stamp_stage("schedule_image_loads"),
                        receive_image_loads,
                        performance::stamp_stage("receive_image_loads"),
                        publish_image_loading_stats,
                        face_billboards_to_camera,
                        // Strips follow this frame's billboard poses and
                        // visibility, so they never trail a moving view.
                        (
                            apply_billboard_visibility,
                            sync_video_strips,
                            // Labels first: layout right-aligns each label
                            // by its freshly rendered width.
                            update_video_strip_labels,
                            layout_video_strip_parts,
                        )
                            .chain(),
                        performance::stamp_stage("billboard_transforms"),
                        spawn_visible_coordinate_labels,
                        update_billboard_coordinate_label_visibility,
                        sync_selection_highlights,
                        performance::stamp_stage("labels_and_selection"),
                        apply_point_cloud_slice,
                        apply_point_cloud_edits,
                        update_view_bounds,
                        performance::stamp_stage("point_cloud"),
                    )
                        .chain(),
                    // Reads the stage probe for the update that just ran, so
                    // it is ordered after every `stamp_stage` above.
                    (
                        benchmark::run_benchmark,
                        (
                            benchmark::drive_headless_benchmark,
                            benchmark::sweep_camera_for_headless_benchmark,
                        )
                            .chain()
                            .run_if(resource_exists::<benchmark::HeadlessBenchmark>),
                    )
                        .chain(),
                )
                    .chain(),
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                update_navigation_text,
                update_control_text,
                update_billboard_text,
                update_debug_text,
                update_performance_text,
                update_resolution_text,
                update_audio_text,
                update_start_text,
            )
                .chain()
                .after(face_billboards_to_camera)
                // Separately registered from the button chain, so without
                // this edge nothing orders these after the widget rebuild:
                // they would query a subtree whose despawn/spawn commands
                // had not been applied yet.
                .after(rebuild_control_widgets),
        )
        .add_systems(
            Update,
            (
                update_navigation_panels,
                update_control_panels,
                update_billboard_panels,
                update_debug_panels,
                update_performance_panels,
                update_pause_panels,
                update_start_panels,
                update_perf_graph_bars,
            )
                .chain()
                .after(face_billboards_to_camera)
                // Separately registered from the button chain, so without
                // this edge nothing orders these after the widget rebuild:
                // they would query a subtree whose despawn/spawn commands
                // had not been applied yet.
                .after(rebuild_control_widgets),
        )
        .add_systems(
            Update,
            (
                update_navigation_button_colors,
                update_control_button_colors,
                update_billboard_button_colors,
                update_debug_button_colors,
                update_performance_button_colors,
                update_pause_button_colors,
                update_resolution_button_colors,
                update_start_button_colors,
            )
                .chain()
                .after(face_billboards_to_camera)
                // Separately registered from the button chain, so without
                // this edge nothing orders these after the widget rebuild:
                // they would query a subtree whose despawn/spawn commands
                // had not been applied yet.
                .after(rebuild_control_widgets),
        )
        .add_systems(
            PostUpdate,
            performance::stamp_stage("update_end").before(TransformSystem::TransformPropagate),
        )
        .run();

    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Picks the billboard texture format from the device that was actually
/// negotiated, rather than from what was asked for.
///
/// Falls back to RGBA8 — four times the VRAM — on an adapter without
/// `TEXTURE_COMPRESSION_BC`, and says so, because silently using 4x the
/// memory would look like a capacity bug rather than a missing feature.
///
/// Also reads back the device's `max_texture_dimension_2d`, which bounds how
/// large a source image may be decoded. A billboard's texture is one
/// contiguous surface, so unlike the tile grid this replaced, that limit now
/// applies to the whole image rather than to a single 1024 px tile.
fn detect_billboard_texture_encoding(device: Option<&RenderDevice>) -> BillboardTextureEncoding {
    let Some(device) = device else {
        eprintln!("No render device available; billboard textures fall back to uncompressed RGBA8");
        return BillboardTextureEncoding {
            format: BillboardTextureFormat::Rgba8,
            ..default()
        };
    };
    let format = if device
        .features()
        .contains(bevy::render::settings::WgpuFeatures::TEXTURE_COMPRESSION_BC)
    {
        BillboardTextureFormat::Bc7
    } else {
        eprintln!(
            "GPU does not support BC texture compression;              billboard textures fall back to uncompressed RGBA8 (4x the VRAM)"
        );
        BillboardTextureFormat::Rgba8
    };
    BillboardTextureEncoding {
        format,
        max_texture_side: device.limits().max_texture_dimension_2d,
    }
}

#[allow(clippy::too_many_arguments)]
fn setup_scene(
    mut commands: Commands,
    scene: Res<ExplorerScene>,
    render_device: Option<Res<RenderDevice>>,
    mut initial_player_placement: ResMut<InitialPlayerPlacement>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut color_materials: ResMut<Assets<ColorMaterial>>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let texture_encoding = detect_billboard_texture_encoding(render_device.as_deref());
    println!(
        "Billboard textures: {:?}, up to {} px per side",
        texture_encoding.format, texture_encoding.max_texture_side
    );
    commands.insert_resource(texture_encoding);

    let bounds = scene.bounds;
    let point_size = point_cloud_point_size(&bounds);
    let billboard_mesh = create_billboard_mesh(&mut meshes, scene.billboard_world_size);
    let billboard_mesh_handle = billboard_mesh.0.clone();
    let render_resolution = RenderResolutionSettings::default();
    let window_size = windows
        .get_single()
        .map(|window| window.resolution.physical_size())
        .unwrap_or_else(|_| UVec2::new(1600, 1000));
    let window_logical_size = windows
        .get_single()
        .map(|window| window.resolution.size())
        .unwrap_or_else(|_| Vec2::new(1600.0, 1000.0));
    let render_target_image = images.add(create_scene_render_target_image(
        render_resolution.target_size(window_size),
    ));
    let scene_render_target = SceneRenderTarget {
        image: render_target_image.clone(),
    };
    commands.insert_resource(billboard_mesh);
    commands.insert_resource(BillboardWorldSize(scene.billboard_world_size));
    commands.insert_resource(billboard_axis_labels(&scene.projection.axis_labels));
    commands.insert_resource(scene_render_target);
    commands.insert_resource(ImageLoadingState::new(
        scene.image_points.clone(),
        scene.texture_budget_mib as usize * 1024 * 1024,
        scene.image_concurrency,
        scene.max_texture_side,
    ));
    commands.insert_resource(NavigationSettings::new(
        scene.camera_speed,
        scene.navigation_reference_distance,
    ));
    commands.insert_resource(NavigationTargets::from_positions(
        scene.image_points.iter().map(|point| point.position),
    ));
    commands.insert_resource(ImagePointIndex::new(&scene.image_points));
    commands.insert_resource(BillboardFacingSettings::default());
    commands.insert_resource(BillboardControls::new(
        scene.texture_budget_mib.clamp(
            spatial_viewer_ui::MIN_TEXTURE_BUDGET_MIB,
            spatial_viewer_ui::MAX_TEXTURE_BUDGET_MIB,
        ),
        scene.max_texture_side,
    ));
    commands.insert_resource(BillboardStats {
        entity_count: scene.image_points.len(),
        pending: scene.image_points.len(),
        ..default()
    });
    commands.insert_resource(DebugSettings::default());
    commands.insert_resource(BenchmarkControls::default());
    commands.insert_resource(PerformanceMetrics::default());
    commands.insert_resource(PauseMenuState::default());
    commands.insert_resource(render_resolution);
    commands.insert_resource(StartMenuState::default());
    commands.insert_resource(AxisGizmoState::default());
    commands.insert_resource(VideoControlsState::default());
    let label_font = load_label_font();
    spawn_boundary_walls(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut images,
        label_font.as_ref(),
    );
    commands.insert_resource(BillboardLabelFont(label_font.clone()));
    commands.insert_resource(VideoControlsFont(label_font));
    commands.insert_resource(VideoStripAssets::new(&mut meshes, &mut materials));
    commands.insert_resource(AudioPlaybackState::new());
    // Images keep their configured worker ceiling when no videos play; the
    // headroom is what simultaneous video playback may add on top.
    commands.insert_resource(DecodeBudget::new(scene.image_concurrency));
    commands.insert_resource(ViewBounds::default());
    let mut cloud = PointCloud::default();
    cloud.rebuild(
        &mut commands,
        &mut meshes,
        &mut materials,
        point_cloud_points(&scene.projection),
        point_size,
    );
    commands.insert_resource(cloud);
    spawn_render_target_presentation(
        &mut commands,
        &mut meshes,
        &mut color_materials,
        render_target_image.clone(),
        window_logical_size,
    );
    spawn_viewer_ui(&mut commands);
    spawn_axis_gizmo_3d(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut images,
        &scene.axis_labels,
    );

    let dense_center = median_position(&scene.projection.points);
    let distance = scene.initial_view_distance;
    let mut camera_transform = Transform::default();
    let mut fly_camera = FlyCamera {
        yaw: 0.0,
        pitch: 0.0,
        sensitivity: 0.0025,
        velocity: Vec3::ZERO,
        velocity_response: CAMERA_VELOCITY_RESPONSE_PER_SECOND,
    };
    if !scene.projection.points.is_empty() {
        place_camera_at_initial_view(
            &mut camera_transform,
            &mut fly_camera,
            dense_center,
            distance,
        );
        initial_player_placement.positioned = true;
    }
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: -2,
            target: RenderTarget::Image(render_target_image),
            ..default()
        },
        Projection::from(PerspectiveProjection {
            near: 0.1,
            far: scene_camera_far_plane(bounds.max_extent),
            ..default()
        }),
        Msaa::Off,
        camera_transform,
        fly_camera,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(
            bounds.center.x + bounds.max_extent,
            bounds.center.y + bounds.max_extent,
            bounds.center.z + bounds.max_extent,
        )
        .looking_at(dense_center, Vec3::Y),
    ));

    if scene.debug_probe_billboard {
        spawn_debug_probe_billboard(
            &mut commands,
            &mut images,
            &mut materials,
            billboard_mesh_handle,
            camera_transform,
            scene.billboard_world_size,
        );
    }
}

/// Spawns one billboard directly in front of the camera using the exact same
/// component shape as `image_loading::receive_image_loads`, with a
/// procedurally generated (no network/decode dependency) texture. Enabled via
/// `--debug-probe-billboard`; used to isolate whether the billboard rendering
/// pipeline itself works, independent of catalog data / navigation / async
/// image loading.
fn spawn_debug_probe_billboard(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
    billboard_mesh: Handle<Mesh>,
    camera_transform: Transform,
    billboard_world_size: f32,
) {
    let camera_forward = camera_transform.rotation.mul_vec3(Vec3::NEG_Z);
    let camera_up = camera_transform.rotation.mul_vec3(Vec3::Y);
    let viewport_normal = camera_transform.rotation.mul_vec3(Vec3::Z);
    let probe_distance = billboard_world_size * 3.0;
    let probe_position = camera_transform.translation + camera_forward * probe_distance;
    let to_camera = (camera_transform.translation - probe_position).normalize();
    let rotation = billboard_rotation(
        to_camera,
        viewport_normal,
        camera_up,
        BillboardFacingAxis::Y,
    );

    let texture_handle = images.add(create_probe_texture());
    let material_handle = materials.add(StandardMaterial {
        base_color_texture: Some(texture_handle.clone()),
        alpha_mode: AlphaMode::Opaque,
        unlit: false,
        ..default()
    });

    println!("Debug probe billboard spawned at {probe_position:?}");

    commands.spawn((
        Mesh3d(billboard_mesh),
        MeshMaterial3d(material_handle.clone()),
        Transform {
            translation: probe_position,
            rotation,
            ..default()
        },
        MediaBillboard {
            image_id: usize::MAX,
            path: "debug-probe".into(),
            is_video: false,
            duration_seconds: None,
            source_size: None,
            texture_side: 64,
            surface_assets: image_loading::BillboardSurfaceAssets {
                entity: None,
                texture_handle: Some(texture_handle),
                material_handle: Some(material_handle),
            },
        },
        Name::new("debug probe billboard"),
    ));
}

/// A high-contrast checkerboard so both UV mapping and basic visibility are
/// obvious in a screenshot, independent of the real image-decode pipeline.
fn create_probe_texture() -> Image {
    const SIZE: u32 = 64;
    const CELL: u32 = 8;
    let mut data = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let checker = ((x / CELL) + (y / CELL)) % 2 == 0;
            if checker {
                data.extend_from_slice(&[255, 0, 255, 255]);
            } else {
                data.extend_from_slice(&[0, 255, 0, 255]);
            }
        }
    }
    Image::new(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

/// wgpu's DX12 backend removes the device with COMMAND_ALLOCATOR_RESET after
/// sustained texture creation/destruction (streamed billboards and video
/// frames), which freezes the window; Vulkan has no such failure. The
/// `WGPU_BACKEND` variable (wgpu's own override, e.g. `dx12`) still wins so
/// the backends can be compared.
fn preferred_render_backends() -> Backends {
    std::env::var("WGPU_BACKEND")
        .ok()
        .map(|value| render_backends_from_comma_list(&value))
        .filter(|backends| !backends.is_empty())
        .unwrap_or(Backends::VULKAN | Backends::METAL | Backends::GL)
}

/// Same spelling wgpu's own `WGPU_BACKEND` parser accepts (`vulkan`, `dx12`,
/// `metal`, `gl`, `webgpu`, `primary`, `secondary`, `all`, comma-separated).
fn render_backends_from_comma_list(value: &str) -> Backends {
    value
        .split(',')
        .map(|name| match name.trim().to_ascii_lowercase().as_str() {
            "vulkan" | "vk" => Backends::VULKAN,
            "dx12" | "d3d12" => Backends::DX12,
            "metal" | "mtl" => Backends::METAL,
            "opengl" | "gles" | "gl" => Backends::GL,
            "webgpu" => Backends::BROWSER_WEBGPU,
            "primary" => Backends::PRIMARY,
            "secondary" => Backends::SECONDARY,
            "all" => Backends::all(),
            _ => Backends::empty(),
        })
        .fold(Backends::empty(), |accumulated, backend| {
            accumulated | backend
        })
}

fn disable_camera_msaa(mut cameras: Query<&mut Msaa, With<Camera>>) {
    for mut msaa in &mut cameras {
        *msaa = Msaa::Off;
    }
}

/// The per-frame GPU upload budget bounds image tile uploads; playing videos
/// add one frame's worth of bytes each on top so their frames are never
/// deferred behind a tile. Bevy drops the previous GPU texture the moment a
/// replacement is extracted, so a deferred video frame renders as a blank
/// billboard for a frame (visible flicker).
fn enable_gpu_upload_budget_after_startup(
    mut startup_frames: Local<u8>,
    video_playback: Res<VideoPlaybackState>,
    mut upload_budget: ResMut<RenderAssetBytesPerFrame>,
) {
    if upload_budget.max_bytes.is_none() && *startup_frames < 3 {
        *startup_frames += 1;
        return;
    }
    let max_bytes = BILLBOARD_GPU_UPLOAD_BYTES_PER_FRAME + video_playback.frame_upload_bytes();
    if upload_budget.max_bytes != Some(max_bytes) {
        *upload_budget = RenderAssetBytesPerFrame::new(max_bytes);
    }
}

fn create_scene_render_target_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        render_target_extent(size),
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Bgra8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image
}

fn render_target_extent(size: UVec2) -> Extent3d {
    Extent3d {
        width: size.x.max(1),
        height: size.y.max(1),
        depth_or_array_layers: 1,
    }
}

fn spawn_render_target_presentation(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    image: Handle<Image>,
    size: Vec2,
) {
    let layer = RenderLayers::layer(PRESENTATION_RENDER_LAYER);
    let mesh = meshes.add(Rectangle::new(1.0, 1.0));
    let material = materials.add(ColorMaterial {
        color: Color::WHITE,
        alpha_mode: AlphaMode2d::Opaque,
        texture: Some(image),
    });
    commands.spawn((
        Mesh2d(mesh),
        MeshMaterial2d(material),
        Transform::from_scale(size.extend(1.0)),
        layer.clone(),
        RenderedSceneCanvas,
        Name::new("rendered scene canvas"),
    ));
    commands.spawn((
        Camera2d,
        Camera {
            order: 0,
            ..default()
        },
        Msaa::Off,
        layer,
        PresentationCamera,
        Name::new("scene presentation camera"),
    ));
}

fn sync_render_target_presentation(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut canvases: Query<&mut Transform, With<RenderedSceneCanvas>>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let size = window.resolution.size();
    for mut transform in &mut canvases {
        transform.scale = size.extend(1.0);
    }
}

fn place_camera_at_initial_view(
    transform: &mut Transform,
    camera: &mut FlyCamera,
    center: Vec3,
    distance: f32,
) {
    transform.translation = center + Vec3::new(distance, distance * 0.45, distance);
    transform.look_at(center, Vec3::Y);
    let (yaw, pitch) = yaw_pitch_from_rotation(transform.rotation);
    camera.yaw = yaw;
    camera.pitch = pitch;
    camera.velocity = Vec3::ZERO;
}

fn navigation_speed(nearest_gap: f32, coordinate_spacing: f32) -> f32 {
    let coordinate_spacing = coordinate_spacing.max(1.0);
    (nearest_gap.max(0.1) * 1.4).clamp(coordinate_spacing * 0.35, coordinate_spacing * 1.35)
}

fn initial_camera_distance(coordinate_spacing: f32, billboard_world_size: f32) -> f32 {
    coordinate_spacing.max(billboard_world_size).max(1.0) * 2.5
}

fn scene_camera_far_plane(bounds_extent: f32) -> f32 {
    (bounds_extent * 4.0).max(1000.0)
}

fn set_scene_camera_far_plane(projection: &mut Projection, bounds_extent: f32) {
    if let Projection::Perspective(perspective) = projection {
        perspective.far = scene_camera_far_plane(bounds_extent);
    }
}

fn enforce_min_window_size(mut windows: Query<&mut Window, With<PrimaryWindow>>) {
    let Ok(mut window) = windows.get_single_mut() else {
        return;
    };
    let width = window.resolution.width();
    let height = window.resolution.height();
    if width >= MIN_WINDOW_WIDTH && height >= MIN_WINDOW_HEIGHT {
        return;
    }
    window
        .resolution
        .set(width.max(MIN_WINDOW_WIDTH), height.max(MIN_WINDOW_HEIGHT));
}

fn toggle_pause_menu(
    keyboard: Res<ButtonInput<KeyCode>>,
    ui_capture: Res<UiInputCapture>,
    mut pause_menu: ResMut<PauseMenuState>,
    mut controls: ResMut<ControlPanelState>,
    mut query: Query<&mut FlyCamera>,
) {
    // An open right-click menu takes the press to close itself.
    if !keyboard.just_pressed(KeyCode::Escape) || ui_capture.context_menu_open {
        return;
    }
    // Escape first releases whatever the control panel has focused — an open
    // dropdown or a text field being typed into. Only a second press reaches
    // the pause menu.
    if controls.input_focused() {
        controls.clear_focus();
        return;
    }
    pause_menu.escape_pressed();
    if let Ok(mut camera) = query.get_single_mut() {
        camera.velocity = Vec3::ZERO;
    }
}

/// Every camera rendering into the shared scene render target: the fly
/// camera and the translate gizmo's overlay camera.
type RenderTargetCameraQuery<'w, 's> =
    Query<'w, 's, &'static mut Camera, Or<(With<FlyCamera>, With<TranslateGizmoCamera>)>>;

#[allow(clippy::too_many_arguments)]
fn apply_render_resolution_requests(
    mut settings: ResMut<RenderResolutionSettings>,
    mut render_target: ResMut<SceneRenderTarget>,
    mut images: ResMut<Assets<Image>>,
    mut color_materials: ResMut<Assets<ColorMaterial>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut camera_query: RenderTargetCameraQuery,
    canvas_query: Query<&MeshMaterial2d<ColorMaterial>, With<RenderedSceneCanvas>>,
) {
    let _resolution_changed = settings.take_resolution_changed();
    let Ok(window) = windows.get_single() else {
        return;
    };
    let size = settings.target_size(window.resolution.physical_size());
    // `Assets::get_mut` unconditionally marks the asset as modified, even if
    // the caller never actually writes to it. Since this system runs every
    // frame while a camera is actively rendering into this image, calling
    // `get_mut` unconditionally forced Bevy to treat the render target as
    // changed on every frame, which broke rendering into it. Check the size
    // with a read-only `get` first so we only touch anything on the rare
    // frame a resize actually happens.
    let needs_resize = images
        .get(&render_target.image)
        .is_some_and(|image| image.size() != size);
    if !needs_resize {
        return;
    }
    // Resizing the `Image` backing an active `RenderTarget::Image` in place
    // (mutating it through `Assets::get_mut`) leaves the camera's GPU-side
    // render attachment stale, so rendering into it silently stops updating
    // afterwards (a known Bevy limitation, e.g. bevyengine/bevy#16159).
    // Instead, create a fresh image at the new size and repoint the camera
    // and the presentation quad's material at the new handle.
    let new_image = images.add(create_scene_render_target_image(size));
    render_target.image = new_image.clone();
    // Both the scene camera and the gizmo overlay camera render into the
    // shared target, so both get repointed at the fresh image.
    for mut camera in &mut camera_query {
        camera.target = RenderTarget::Image(new_image.clone());
    }
    if let Ok(material_handle) = canvas_query.get_single() {
        if let Some(material) = color_materials.get_mut(&material_handle.0) {
            material.texture = Some(new_image);
        }
    }
}

fn apply_navigation_ui_requests(
    mut navigation: ResMut<NavigationSettings>,
    targets: Res<NavigationTargets>,
    scene: Res<ExplorerScene>,
    mut query: Query<(&mut Transform, &mut FlyCamera)>,
) {
    let Some(request) = navigation.take_navigation_request() else {
        return;
    };
    let Ok((mut transform, mut camera)) = query.get_single_mut() else {
        return;
    };
    let forward = transform.rotation.mul_vec3(Vec3::NEG_Z);
    let Some(target) = navigation_request_target(request, &targets, transform.translation, forward)
    else {
        return;
    };

    let mut direction = transform.translation - target;
    if direction.length_squared() <= f32::EPSILON {
        direction = Vec3::new(1.0, 0.45, 1.0);
    }
    let distance = scene
        .initial_view_distance
        .max(scene.billboard_world_size * 1.5)
        .max(1.0);
    transform.translation = target + direction.normalize() * distance;
    transform.look_at(target, Vec3::Y);
    let (yaw, pitch) = yaw_pitch_from_rotation(transform.rotation);
    camera.yaw = yaw;
    camera.pitch = pitch;
    camera.velocity = Vec3::ZERO;
}

fn navigation_request_target(
    request: NavigationRequest,
    targets: &NavigationTargets,
    position: Vec3,
    forward: Vec3,
) -> Option<Vec3> {
    match request {
        NavigationRequest::TeleportNearest | NavigationRequest::FocusNearestImage => {
            targets.nearest_to(position)
        }
        NavigationRequest::SnapOrbitNearestCluster => targets.nearest_cluster_to(position),
        // Falls back to the nearest cluster so the button still does
        // something useful when nothing lies ahead of the camera.
        NavigationRequest::JumpNextVisibleCluster => targets
            .next_visible_cluster(position, forward)
            .or_else(|| targets.nearest_cluster_to(position)),
    }
}

#[allow(clippy::too_many_arguments)]
fn fly_camera_controls(
    mut input: FlyCameraInput,
    mut navigation_settings: ResMut<NavigationSettings>,
    right_drag: Res<RightDragCursorState>,
    pause_menu: Res<PauseMenuState>,
    controls: Res<ControlPanelState>,
    selection: Res<SelectionState>,
    mut axis_gizmo: ResMut<AxisGizmoState>,
    mut query: FlyCameraTransformQuery,
) {
    let Ok((mut transform, mut camera)) = query.get_single_mut() else {
        return;
    };

    if pause_menu.paused {
        input.mouse_motion.clear();
        input.mouse_wheel.clear();
        camera.velocity = Vec3::ZERO;
        return;
    }

    // While the control panel owns the keyboard — a dropdown open, or a text
    // field focused — the wheel scrolls that control and typed characters
    // edit it, so neither may drive the camera. While a manual-spacing drag
    // is live, the wheel adjusts the dragged selection's distance instead of
    // the fly speed.
    let panel_focused = controls.input_focused();
    if panel_focused || selection.drag_active() {
        input.mouse_wheel.clear();
    } else {
        for wheel in input.mouse_wheel.read() {
            navigation_settings.apply_scroll_delta(wheel.y);
        }
    }

    // Only a right press that began in the world looks around; one that
    // began over UI or an open menu belongs to them.
    if input.mouse_buttons.pressed(MouseButton::Right) && right_drag.anchor.is_some() {
        let mut rotated = false;
        for motion in input.mouse_motion.read() {
            camera.yaw -= motion.delta.x * camera.sensitivity;
            camera.pitch -= motion.delta.y * camera.sensitivity;
            rotated = true;
        }
        camera.pitch = camera.pitch.clamp(-1.48, 1.48);
        transform.rotation = fly_camera_rotation(camera.yaw, camera.pitch);
        if rotated {
            axis_gizmo.clear_selection();
        }
    } else {
        input.mouse_motion.clear();
    }

    let forward = transform.rotation.mul_vec3(Vec3::NEG_Z);
    let right = transform.rotation.mul_vec3(Vec3::X);
    let up = transform.rotation.mul_vec3(Vec3::Y);
    let shift_multiplier = if input.keyboard.pressed(KeyCode::ShiftLeft)
        || input.keyboard.pressed(KeyCode::ShiftRight)
    {
        navigation_settings.shift_speed_multiplier
    } else {
        1.0
    };
    let movement = if panel_focused {
        Vec3::ZERO
    } else {
        movement_from_keys(forward, right, up, |key| input.keyboard.pressed(key))
    };
    navigation_settings.update_cruise_pressure(movement, camera.velocity, input.time.delta_secs());
    let status = navigation_status(&navigation_settings).with_velocity_multiplier(shift_multiplier);
    let target_speed = status.effective_speed;
    navigation_settings.status = status;

    let target_velocity = if movement.length_squared() > 0.0 {
        movement.normalize() * target_speed
    } else {
        Vec3::ZERO
    };
    camera.velocity = inertial_velocity(
        camera.velocity,
        target_velocity,
        camera.velocity_response,
        input.time.delta_secs(),
    );
    if target_velocity == Vec3::ZERO && camera.velocity.length_squared() < CAMERA_STOP_EPSILON {
        camera.velocity = Vec3::ZERO;
    }
    transform.translation += camera.velocity * input.time.delta_secs();
}

/// Hides and pins the cursor while the right button is dragged in the world,
/// and restores it on release. A plain right click (no motion beyond
/// `RIGHT_DRAG_MOTION_THRESHOLD`) leaves the cursor untouched, as does a
/// right press that starts over UI or while a menu is open.
fn manage_right_drag_cursor(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    ui_capture: Res<UiInputCapture>,
    pause_menu: Res<PauseMenuState>,
    mut state: ResMut<RightDragCursorState>,
    mut right_clicks: EventWriter<WorldRightClick>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    let Ok(mut window) = windows.get_single_mut() else {
        return;
    };
    let menu_open = pause_menu.paused || ui_capture.menu_open || ui_capture.context_menu_open;

    if mouse_buttons.just_pressed(MouseButton::Right) && !menu_open && !ui_capture.pointer_over_ui {
        state.anchor = window.cursor_position();
        state.accumulated_motion = 0.0;
    }

    let drag_active =
        state.anchor.is_some() && mouse_buttons.pressed(MouseButton::Right) && !menu_open;
    if drag_active {
        if !state.dragging {
            state.accumulated_motion += mouse_motion
                .read()
                .map(|motion| motion.delta.length())
                .sum::<f32>();
            if state.accumulated_motion >= RIGHT_DRAG_MOTION_THRESHOLD {
                state.dragging = true;
                // Re-pin to the anchor before flipping grab/visibility: bevy's
                // window sync applies a cursor-position change before grab
                // mode changes within the same frame, so `Locked` captures
                // this position rather than wherever the threshold crossed.
                window.set_cursor_position(state.anchor);
                window.cursor_options.visible = false;
                // `Confined` on Windows has a winit-specific quirk: combined
                // with a hidden cursor it clips to the *window's center*
                // (a workaround so a hidden cursor can't drift onto the
                // taskbar), not wherever the cursor actually is — which is
                // why an earlier version of this teleported to the center
                // instead of the click point. `Locked` clips to the cursor's
                // current position instead, which is the lock-in-place
                // behavior this drag needs.
                window.cursor_options.grab_mode = CursorGrabMode::Locked;
            }
        }
        if state.dragging {
            window.set_cursor_position(state.anchor);
        }
        return;
    }

    mouse_motion.clear();
    if state.dragging {
        window.cursor_options.visible = true;
        window.cursor_options.grab_mode = CursorGrabMode::None;
        window.set_cursor_position(state.anchor);
    } else if state.anchor.is_some() && mouse_buttons.just_released(MouseButton::Right) {
        // Released before the drag threshold: a click, not a look.
        right_clicks.send(WorldRightClick);
    }
    state.anchor = None;
    state.accumulated_motion = 0.0;
    state.dragging = false;
}

fn movement_from_keys<F>(forward: Vec3, right: Vec3, up: Vec3, pressed: F) -> Vec3
where
    F: Fn(KeyCode) -> bool,
{
    let mut movement = Vec3::ZERO;
    if pressed(KeyCode::KeyW) {
        movement += forward;
    }
    if pressed(KeyCode::KeyS) {
        movement -= forward;
    }
    if pressed(KeyCode::KeyD) {
        movement += right;
    }
    if pressed(KeyCode::KeyA) {
        movement -= right;
    }
    if pressed(KeyCode::Space) {
        movement += up;
    }
    if pressed(KeyCode::ControlLeft) || pressed(KeyCode::ControlRight) {
        movement -= up;
    }
    movement
}

/// Arrow keys teleport to the nearest visible image along a world axis. The
/// travel axis is view-dependent but snapped axis-orthogonal: left/right take
/// the world axis best aligned with the camera's right vector, up/down the
/// one best aligned with its up vector — so an upright view steps up/down
/// along Y, while a view looking down -Y steps along whichever of X/Z the
/// screen-up direction is most aligned with. Holding right shift swaps the
/// up/down pair onto the axis best aligned with the view direction, stepping
/// forward and backward into the scene. The camera moves by the offset
/// between the currently framed image and the target, so viewing distance
/// and framing are preserved rather than snapping to a fixed standoff.
///
/// `last_teleport` keeps stepping anchored to the image the previous press
/// landed on for as long as the camera stays where that press put it. Without
/// it, every press would re-derive the anchor from a fixed-distance probe,
/// and on lattices whose rows sit on staggered depth planes the probe can
/// flip between planes — each flip leaking the inter-plane depth difference
/// into the translation and ratcheting the camera toward the images.
#[allow(clippy::too_many_arguments)]
fn teleport_on_arrow_keys(
    keyboard: Res<ButtonInput<KeyCode>>,
    pause_menu: Res<PauseMenuState>,
    controls: Res<ControlPanelState>,
    targets: Res<NavigationTargets>,
    scene: Res<ExplorerScene>,
    bounds: Res<ViewBounds>,
    mut query: FlyCameraTransformQuery,
    mut last_teleport: Local<Option<TeleportAnchor>>,
) {
    if pause_menu.paused || controls.input_focused() {
        return;
    }
    // Right shift retargets up/down from the screen-up axis to the view
    // direction, so the same snap-to-next-coordinate step travels forward and
    // backward through the lattice instead of climbing it.
    let depth_travel = keyboard.pressed(KeyCode::ShiftRight);
    let (up_axis, up_sign) = if depth_travel {
        (Vec3::NEG_Z, 1.0)
    } else {
        (Vec3::Y, 1.0)
    };
    let Some(axis) = [
        (KeyCode::ArrowRight, Vec3::X, 1.0),
        (KeyCode::ArrowLeft, Vec3::X, -1.0),
        (KeyCode::ArrowUp, up_axis, up_sign),
        (KeyCode::ArrowDown, up_axis, -up_sign),
    ]
    .into_iter()
    .find_map(|(key, local_axis, sign)| keyboard.just_pressed(key).then_some((local_axis, sign))) else {
        return;
    };
    let Ok((mut transform, mut camera)) = query.get_single_mut() else {
        return;
    };

    let camera_position = transform.translation;
    let travel_axis = snap_to_dominant_world_axis(transform.rotation.mul_vec3(axis.0) * axis.1);
    let forward = transform.rotation.mul_vec3(Vec3::NEG_Z);
    // While the camera is still exactly where the last teleport left it, keep
    // stepping relative to that press's target. Only once the user has moved
    // (flight, gizmo, focus buttons, scene reload) re-derive the anchor as
    // the image nearest to a point probed ahead of the camera.
    let sticky_anchor = last_teleport.as_ref().and_then(|previous| {
        (camera_position.distance_squared(previous.camera_position)
            < TELEPORT_ANCHOR_POSITION_EPSILON_SQUARED)
            .then_some(previous.target)
    });
    let anchor = match sticky_anchor {
        Some(anchor) => anchor,
        None => {
            let focus = camera_position + forward * teleport_front_distance(&scene);
            let Some(anchor) = targets.nearest_to(focus) else {
                return;
            };
            anchor
        }
    };
    // Only step to images the user can currently see: skip anything sliced
    // away around the camera or beyond the view-bounds walls.
    let slice_radius = controls.slice_depth_cells() * scene.coordinate_spacing;
    let visible = |position: Vec3| {
        !bounds.is_beyond(camera_position, position)
            && (slice_radius <= 0.0
                || position.distance_squared(camera_position) >= slice_radius * slice_radius)
    };
    let min_advance = scene.coordinate_spacing * TELEPORT_MIN_ADVANCE_FACTOR;
    let Some(target) = targets.nearest_in_direction(anchor, travel_axis, min_advance, visible)
    else {
        return;
    };

    // Pure translation: distance and framing relative to the image carry
    // over from the anchor to the target unchanged.
    transform.translation = camera_position + (target - anchor);
    camera.velocity = Vec3::ZERO;
    *last_teleport = Some(TeleportAnchor {
        camera_position: transform.translation,
        target,
    });
}

/// Where the last arrow-key teleport put the camera and which image it
/// stepped to; see `teleport_on_arrow_keys`.
struct TeleportAnchor {
    camera_position: Vec3,
    target: Vec3,
}

/// Signed world axis (±X, ±Y, or ±Z) most aligned with `direction`.
fn snap_to_dominant_world_axis(direction: Vec3) -> Vec3 {
    let absolute = direction.abs();
    if absolute.x >= absolute.y && absolute.x >= absolute.z {
        Vec3::X * direction.x.signum()
    } else if absolute.y >= absolute.z {
        Vec3::Y * direction.y.signum()
    } else {
        Vec3::Z * direction.z.signum()
    }
}

/// How far ahead of the camera the arrow-key teleport probes for the image
/// it treats as currently framed: close enough to pick the image being
/// looked at, short enough to stay inside the gap between grid cells.
fn teleport_front_distance(scene: &ExplorerScene) -> f32 {
    (scene.billboard_world_size * 1.5).max(1.0)
}

fn apply_axis_gizmo_requests(
    mut axis_gizmo: ResMut<AxisGizmoState>,
    mut query: Query<(&mut Transform, &mut FlyCamera)>,
) {
    let orientation = if let Some(face) = axis_gizmo.take_requested_face() {
        Some((axis_gizmo_view_direction(face), axis_gizmo_up_vector(face)))
    } else {
        axis_gizmo
            .take_requested_direction()
            .map(|direction| (direction, axis_gizmo_generic_up_vector(direction)))
    };
    let Some((direction, up)) = orientation else {
        return;
    };
    let Ok((mut transform, mut camera)) = query.get_single_mut() else {
        return;
    };

    // The gizmo only ever changes where the camera is looking, never where
    // it is: rotate in place to face `direction` rather than repositioning
    // the camera to orbit some scene-derived center.
    transform.look_to(-direction, up);
    let (yaw, pitch) = yaw_pitch_from_rotation(transform.rotation);
    camera.yaw = yaw;
    camera.pitch = pitch;
    camera.velocity = Vec3::ZERO;
}

/// Applies the dims-menu cutaway slice: billboards closer to the camera than
/// the slice depth (in coordinate cells) are hidden — their coordinate labels
/// hide with them as children — so the interior of a dense block of images
/// becomes visible from outside. A hidden billboard also loses interactivity:
/// clicks skip hidden billboards, and its video control strip is hidden.
fn apply_billboard_visibility(
    controls: Res<ControlPanelState>,
    scene: Res<ExplorerScene>,
    camera_query: Query<&Transform, (With<FlyCamera>, Without<MediaBillboard>)>,
    mut billboards: Query<(Ref<MediaBillboard>, &Transform, &mut Visibility)>,
    mut last_applied: Local<Option<(Vec3, Quat, f32, usize)>>,
) {
    let radius = controls.slice_depth_cells() * scene.coordinate_spacing;
    let active = radius > 0.0;
    let Ok(camera_transform) = camera_query.get_single() else {
        return;
    };
    let billboard_count = billboards.iter().len();
    let billboard_added = billboards
        .iter()
        .any(|(billboard, _, _)| billboard.is_added());
    if let Some((position, rotation, previous_radius, previous_count)) = *last_applied {
        let unchanged = position.distance_squared(camera_transform.translation) <= 0.0025
            && rotation.dot(camera_transform.rotation).abs() >= 0.999_98
            && previous_radius == radius
            && previous_count == billboard_count
            && !billboard_added;
        if unchanged {
            return;
        }
    }
    *last_applied = Some((
        camera_transform.translation,
        camera_transform.rotation,
        radius,
        billboard_count,
    ));

    let radius_squared = radius * radius;
    let camera_forward = camera_transform.rotation.mul_vec3(Vec3::NEG_Z);
    for (_, transform, mut visibility) in &mut billboards {
        let to_billboard = transform.translation - camera_transform.translation;
        let behind_camera = billboard_is_behind_camera(to_billboard, camera_forward);
        let sliced = active
            && transform
                .translation
                .distance_squared(camera_transform.translation)
                < radius_squared;
        let hidden = behind_camera || sliced;
        let desired = if hidden {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if *visibility != desired {
            *visibility = desired;
        }
    }
}

/// Keeps the point-cloud markers (the colored cubes standing in for
/// unrendered coordinates) in sync with the dims-menu cutaway slice:
/// recomputes which points fall inside the slice sphere whenever the camera
/// moves meaningfully, the slice depth changes, or the cloud is edited or
/// rebuilt (either of which invalidates previously applied flags).
fn billboard_is_behind_camera(to_billboard: Vec3, camera_forward: Vec3) -> bool {
    to_billboard.length_squared() > f32::EPSILON && to_billboard.dot(camera_forward) <= 0.0
}

fn apply_point_cloud_slice(
    controls: Res<ControlPanelState>,
    scene: Res<ExplorerScene>,
    camera_query: Query<&Transform, With<FlyCamera>>,
    mut cloud: ResMut<PointCloud>,
    mut last_applied: Local<Option<(Vec3, f32, u64)>>,
) {
    let Ok(camera_transform) = camera_query.get_single() else {
        return;
    };
    let camera = camera_transform.translation;
    let radius = (controls.slice_depth_cells() * scene.coordinate_spacing).max(0.0);
    let move_threshold_squared = (scene.coordinate_spacing * 0.1).powi(2);
    let needs_apply = match *last_applied {
        None => radius > 0.0,
        Some((previous_camera, previous_radius, previous_epoch)) => {
            let inactive = radius <= 0.0 && previous_radius <= 0.0;
            !inactive
                && (previous_radius != radius
                    || cloud.epoch() != previous_epoch
                    || camera.distance_squared(previous_camera) > move_threshold_squared)
        }
    };
    if !needs_apply {
        return;
    }
    cloud.apply_slice(camera, radius);
    *last_applied = Some((camera, radius, cloud.epoch()));
}

/// Sizes the shared decode budget to this frame's image-load demand —
/// instantly upward, slowly downward — before the scheduler draws from it.
fn update_decode_budget(
    time: Res<Time>,
    loading_state: Res<ImageLoadingState>,
    mut budget: ResMut<DecodeBudget>,
) {
    budget.update(
        loading_state.pending_demand(),
        loading_state.in_flight_count(),
        time.delta_secs(),
    );
}

pub(crate) fn cursor_to_render_target_position(
    window: &Window,
    cursor_position: Vec2,
    render_size: UVec2,
) -> Option<Vec2> {
    if render_size.x == 0 || render_size.y == 0 {
        return None;
    }
    let window_size = window.resolution.size();
    if window_size.x <= f32::EPSILON || window_size.y <= f32::EPSILON {
        return None;
    }

    if cursor_position.x < 0.0
        || cursor_position.x > window_size.x
        || cursor_position.y < 0.0
        || cursor_position.y > window_size.y
    {
        return None;
    }

    Some(cursor_position / window_size * render_size.as_vec2())
}

fn inertial_velocity(
    current_velocity: Vec3,
    target_velocity: Vec3,
    response_per_second: f32,
    delta_seconds: f32,
) -> Vec3 {
    let response = response_per_second.max(0.0);
    let delta = delta_seconds.max(0.0);
    let blend = 1.0 - (-response * delta).exp();
    current_velocity.lerp(target_velocity, blend.clamp(0.0, 1.0))
}

fn yaw_pitch_from_rotation(rotation: Quat) -> (f32, f32) {
    let forward = rotation.mul_vec3(Vec3::NEG_Z);
    let yaw = (-forward.x).atan2(-forward.z);
    let pitch = forward.y.asin();
    (yaw, pitch)
}

fn fly_camera_rotation(yaw: f32, pitch: f32) -> Quat {
    Quat::from_axis_angle(Vec3::Y, yaw) * Quat::from_axis_angle(Vec3::X, pitch)
}

/// Gizmo text for one world axis. The server names its own axes, so an axis
/// it left unnamed is simply unlabeled rather than a missing lookup.
fn axis_label(label: Option<String>) -> String {
    label.unwrap_or_else(|| "None".to_owned())
}

/// Axis names used to prefix a billboard's coordinate labels; `None` where
/// the server named no axis.
fn billboard_axis_labels(axis_labels: &[Option<String>; 3]) -> BillboardAxisLabels {
    BillboardAxisLabels(axis_labels.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use spatial_viewer_ui::AxisGizmoFace;

    #[test]
    fn initial_camera_distance_uses_local_spacing() {
        assert_eq!(initial_camera_distance(6.0, 4.68), 15.0);
        assert_eq!(initial_camera_distance(2.0, 5.0), 12.5);
    }

    #[test]
    fn billboard_visibility_culls_only_the_rear_hemisphere() {
        assert!(!billboard_is_behind_camera(Vec3::NEG_Z, Vec3::NEG_Z));
        assert!(billboard_is_behind_camera(Vec3::Z, Vec3::NEG_Z));
        assert!(billboard_is_behind_camera(Vec3::X, Vec3::NEG_Z));
        assert!(!billboard_is_behind_camera(Vec3::ZERO, Vec3::NEG_Z));
    }

    #[test]
    fn gpu_upload_budget_activates_after_static_startup_assets() {
        let mut app = App::new();
        app.init_resource::<RenderAssetBytesPerFrame>()
            .init_resource::<VideoPlaybackState>()
            .add_systems(Update, enable_gpu_upload_budget_after_startup);

        for _ in 0..3 {
            app.update();
            assert_eq!(
                app.world().resource::<RenderAssetBytesPerFrame>().max_bytes,
                None
            );
        }
        app.update();
        assert_eq!(
            app.world().resource::<RenderAssetBytesPerFrame>().max_bytes,
            Some(BILLBOARD_GPU_UPLOAD_BYTES_PER_FRAME)
        );
    }

    #[test]
    fn median_position_ignores_outlier_bounding_box_center() {
        // A high-cardinality axis (e.g. raw seed values) can stretch the
        // bounding box far past where most points actually live; the median
        // should stay inside the dense cluster instead of drifting toward an
        // empty region like `Bounds3::center` (the min/max midpoint) does.
        let make_point = |image_id: usize, position: [f32; 3]| spatial_api::ProjectionPoint {
            image_id,
            path: String::new(),
            position,
            width: None,
            height: None,
            media_type: "image".to_owned(),
            duration_seconds: None,
            coordinate_labels: [None, None, None],
        };
        let points = vec![
            make_point(0, [0.0, 0.0, 0.0]),
            make_point(1, [1.0, 1.0, 1.0]),
            make_point(2, [2.0, 2.0, 10_000.0]),
        ];
        let median = median_position(&points);
        assert_eq!(median.z, 1.0);
        let bounds = projection_bounds(&points);
        assert!(median.z < bounds.center.z);
    }

    #[test]
    fn inertial_velocity_accelerates_toward_target() {
        let velocity = inertial_velocity(Vec3::ZERO, Vec3::X * 10.0, 6.5, 0.1);

        assert!(velocity.x > 0.0);
        assert!(velocity.x < 10.0);
    }

    #[test]
    fn inertial_velocity_slows_toward_zero() {
        let velocity = inertial_velocity(Vec3::X * 10.0, Vec3::ZERO, 6.5, 0.1);

        assert!(velocity.x > 0.0);
        assert!(velocity.x < 10.0);
    }

    #[test]
    fn space_and_ctrl_move_along_view_up_axis() {
        // Looking straight down, the view's up vector points along -Z, so
        // Space must move toward -Z rather than world +Y.
        let rotation = fly_camera_rotation(0.0, -std::f32::consts::FRAC_PI_2);
        let up = rotation.mul_vec3(Vec3::Y);

        let ascend = movement_from_keys(Vec3::ZERO, Vec3::ZERO, up, |key| {
            matches!(key, KeyCode::Space)
        });
        assert!((ascend - up).length() < 0.0001);
        assert!(ascend.z < -0.99);

        let descend = movement_from_keys(Vec3::ZERO, Vec3::ZERO, up, |key| {
            matches!(key, KeyCode::ControlLeft)
        });
        assert!((descend + up).length() < 0.0001);
    }

    #[test]
    fn arrow_keys_no_longer_feed_continuous_movement() {
        let movement = movement_from_keys(Vec3::NEG_Z, Vec3::X, Vec3::Y, |key| {
            matches!(
                key,
                KeyCode::ArrowRight | KeyCode::ArrowLeft | KeyCode::ArrowUp | KeyCode::ArrowDown
            )
        });

        assert_eq!(movement, Vec3::ZERO);
    }

    #[test]
    fn depth_travel_snaps_to_the_axis_the_camera_looks_along() {
        // Upright view looking along -Z: right shift steps up/down through Z
        // rather than climbing Y.
        let upright = fly_camera_rotation(0.0, 0.0);
        assert_eq!(
            snap_to_dominant_world_axis(upright.mul_vec3(Vec3::Y)),
            Vec3::Y
        );
        assert_eq!(
            snap_to_dominant_world_axis(upright.mul_vec3(Vec3::NEG_Z)),
            Vec3::NEG_Z
        );

        // Yawed a quarter turn: the view direction now snaps to an X axis,
        // while the screen-up axis is still Y.
        let yawed = fly_camera_rotation(std::f32::consts::FRAC_PI_2, 0.0);
        assert_eq!(
            snap_to_dominant_world_axis(yawed.mul_vec3(Vec3::Y)),
            Vec3::Y
        );
        assert_eq!(
            snap_to_dominant_world_axis(yawed.mul_vec3(Vec3::NEG_Z)),
            Vec3::NEG_X
        );
    }

    #[test]
    fn dominant_axis_snap_is_view_orientation_dependent() {
        // Upright view: screen-up snaps to +Y.
        let upright = fly_camera_rotation(0.3, 0.1);
        assert_eq!(
            snap_to_dominant_world_axis(upright.mul_vec3(Vec3::Y)),
            Vec3::Y
        );

        // Looking straight down -Y with yaw 0: screen-up lies along -Z, so
        // up/down teleports travel the most-aligned axis between X and Z.
        let looking_down = fly_camera_rotation(0.0, -std::f32::consts::FRAC_PI_2);
        assert_eq!(
            snap_to_dominant_world_axis(looking_down.mul_vec3(Vec3::Y)),
            Vec3::NEG_Z
        );
        // Screen-right stays ±X regardless of pitch.
        assert_eq!(
            snap_to_dominant_world_axis(looking_down.mul_vec3(Vec3::X)),
            Vec3::X
        );
    }

    #[test]
    fn cursor_to_render_target_position_scales_full_window() {
        let window = Window {
            resolution: (1600.0, 1000.0).into(),
            ..default()
        };
        let render_size = UVec2::new(800, 500);

        assert_eq!(
            cursor_to_render_target_position(&window, Vec2::new(800.0, 500.0), render_size),
            Some(Vec2::new(400.0, 250.0))
        );
        assert_eq!(
            cursor_to_render_target_position(&window, Vec2::new(800.0, 50.0), render_size),
            Some(Vec2::new(400.0, 25.0))
        );
        assert_eq!(
            cursor_to_render_target_position(&window, Vec2::new(1601.0, 500.0), render_size),
            None
        );
    }

    #[test]
    fn yaw_pitch_round_trip_preserves_camera_forward() {
        let rotation = Transform::from_xyz(15.0, 6.75, 15.0)
            .looking_at(Vec3::ZERO, Vec3::Y)
            .rotation;
        let (yaw, pitch) = yaw_pitch_from_rotation(rotation);

        let original_forward = rotation.mul_vec3(Vec3::NEG_Z);
        let reconstructed_forward = fly_camera_rotation(yaw, pitch).mul_vec3(Vec3::NEG_Z);

        assert!((original_forward - reconstructed_forward).length() < 0.0001);
    }

    #[test]
    fn axis_gizmo_y_face_uses_non_collinear_up_vector() {
        assert_eq!(axis_gizmo_view_direction(AxisGizmoFace::PositiveY), Vec3::Y);
        assert_eq!(axis_gizmo_up_vector(AxisGizmoFace::PositiveY), Vec3::Z);
    }

    #[test]
    fn video_controls_systems_run_without_query_conflicts() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(test_scene());
        app.insert_resource(VideoControlsState::default());
        app.insert_resource(SelectionState::default());
        app.insert_resource(PauseMenuState::default());
        app.insert_resource(AudioPlaybackState::without_output());
        app.insert_resource(MediaSettings::in_memory());
        app.init_resource::<MediaProbes>();
        app.init_resource::<PlaybackSettings>();
        app.insert_resource(VideoControlsFont(None));
        app.init_resource::<Assets<Mesh>>();
        app.init_resource::<Assets<StandardMaterial>>();
        app.init_resource::<Assets<Image>>();
        app.add_systems(
            Update,
            (
                advance_video_clocks,
                update_video_strip_labels,
                layout_video_strip_parts,
            )
                .chain(),
        );

        app.update();
    }

    /// Picking a menu item closes the menu; the command must still reach
    /// the video the menu was opened for.
    #[test]
    fn a_menu_command_that_closes_the_menu_still_acts_on_its_video() {
        const PATH: &str = "clip.mp4";
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(ContextMenuPlugin::<BillboardMenuCommand>::default());
        app.insert_resource(test_scene());
        app.init_resource::<RenderResolutionSettings>();
        app.init_resource::<PauseMenuState>();
        app.init_resource::<UiInputCapture>();
        app.init_resource::<ButtonInput<KeyCode>>();
        app.init_resource::<ButtonInput<MouseButton>>();
        app.init_resource::<VideoControlsState>();
        app.init_resource::<VideoPlaybackState>();
        app.init_resource::<MediaProbes>();
        app.insert_resource(MediaSettings::in_memory());
        app.insert_resource(BillboardMenuTarget::video(7, PATH));
        app.init_non_send_resource::<ClipboardHandle>();
        app.add_event::<WorldRightClick>();
        app.add_systems(
            Update,
            billboard_menu_systems()
                .after(ContextMenuSystems::Input)
                .before(ContextMenuSystems::Render),
        );

        let mut menu = app
            .world_mut()
            .resource_mut::<spatial_viewer_ui::ContextMenu<BillboardMenuCommand>>();
        menu.open(
            Vec2::ZERO,
            spatial_viewer_ui::ContextMenuModel {
                title: PATH.to_owned(),
                sections: Vec::new(),
            },
        );
        menu.activate(BillboardMenuCommand::SetSpeed(1.5), true);
        assert!(!menu.is_open());
        app.update();

        let settings = app.world().resource::<MediaSettings>();
        assert_eq!(settings.video(PATH).speed, 1.5);
    }

    /// Pointer-driven systems need a window and input resources to run, but
    /// parameter conflicts already surface when a system is initialized.
    #[test]
    fn video_input_systems_initialize_without_param_conflicts() {
        fn initialize<Marker>(world: &mut World, system: impl IntoSystem<(), (), Marker>) {
            IntoSystem::into_system(system).initialize(world);
        }
        let mut world = World::new();
        initialize(&mut world, handle_video_strip_input);
        initialize(&mut world, handle_video_keyboard);
        initialize(&mut world, reveal_hovered_videos);
        initialize(&mut world, handle_selection_and_drag);
        initialize(&mut world, sync_video_strips);
        initialize(&mut world, billboard_menu::open_billboard_menu);
    }

    pub(crate) fn test_scene() -> ExplorerScene {
        ExplorerScene {
            projection: spatial_api::ProjectionPage {
                axis_labels: [None, None, None],
                coordinate_spacing: 6.0,
                duplicate_spacing: 0.8,
                sprite_world_height: 4.68,
                offset: 0,
                limit: 0,
                total: 0,
                points: Vec::new(),
            },
            bounds: Bounds3 {
                minimum: Vec3::ZERO,
                maximum: Vec3::ZERO,
                center: Vec3::ZERO,
                max_extent: 1.0,
            },
            image_points: Vec::new(),
            client: SpatialApiClient::new("http://127.0.0.1:8765"),
            projection_limit: 0,
            coordinate_spacing: 6.0,
            duplicate_spacing: 0.8,
            texture_budget_mib: 1024,
            image_concurrency: 1,
            max_texture_side: 0,
            billboard_world_size: 4.68,
            camera_speed: 2.0,
            navigation_reference_distance: 6.0,
            initial_view_distance: 15.0,
            axis_labels: ["None".to_owned(), "None".to_owned(), "None".to_owned()],
            debug_probe_billboard: false,
        }
    }
}
