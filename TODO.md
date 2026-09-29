# Viewer TODO

## Locomotion

- Removed the adaptive/conditional speed system (align/target/offset/field/depth/dens/adapt
  multipliers and the look-hitbox it depended on) as the source of inconsistent speed
  behavior; only `cruise` and `shift` multipliers remain. Re-evaluate whether any
  position-aware speed behavior is worth reintroducing once the fixed baseline has been
  used for a while, and if so design it as a single, well-tested multiplier rather than
  the previous stack of interacting ones.

- Investigate low framerate when only a single coordinate is present.
  - Reproduce with a projection/catalog containing exactly one visible coordinate or one projected point.
  - Check whether single-point bounds, camera distance, spatial cell size, axis/gizmo scale, billboard orientation radius, or density sampling produce degenerate values.
  - Profile the update loop with the performance panel expanded and narrowed to systems that still run every frame despite one coordinate.
  - Add regression coverage for a one-point projection that verifies stable navigation values, no degenerate bounds, no excessive spatial-grid scanning, and normal frame-time behavior.

- Validate cluster navigation assists.
  - Test `TeleportNearest`, `FocusNearestImage`, `SnapOrbitNearestCluster`, and `JumpNextVisibleCluster` on sparse and dense projections.
  - Confirm the assists do not produce disorienting camera jumps or leave non-zero velocity after snapping.
  - Add tests for empty target lists, one target, one cluster, and clusters behind the camera.

## UI Polish

- Input routing: `UiInputCapture` (sampled in `PreUpdate` after UI focus) now blocks world clicks while the pointer is over UI panels/buttons, the axis gizmo canvas, or while a menu was open at frame start, so UI clicks (including pause-menu Resume and the start menu) no longer pass through to billboards/gizmo.
- Right-drag look now hides the cursor and pins it in place for the duration of the drag (`manage_right_drag_cursor`); a plain right click without motion leaves the cursor untouched.
  - Uses `CursorGrabMode::Locked`, not `Confined`: winit's Windows backend clips a `Confined` + hidden cursor to the window's *center* (an intentional workaround to keep a hidden cursor off the taskbar), which is why an earlier version of this teleported to the center instead of the click point. `Locked` clips to wherever the cursor actually is when applied, which is genuine lock-in-place.

- Decide whether to reintroduce `viewer-ui-overhaul`'s dense HUD/settings panel as an optional layout.
  - Keep the current pill/dropdown/start-menu surface available.
  - Add an optional dense HUD/settings mode for professional repeated use.
  - Separate always-visible metrics from tuning controls so changing statistics do not move interactive controls.

- Stabilize dynamic metric layout.
  - Give changing numeric fields fixed-width slots or tabular formatting for speed, FPS, loaded counts, pending counts, texture size, LOD, and orientation counts.
  - Audit `SpeedSummary`, `BillboardsSummary`, `PerformanceSummary`, and expanded detail text for width changes that can resize or shift UI.
  - Replace long text buttons such as "Jump next visible cluster" with compact controls or a wider settings layout.
  - Add visual states for "pending", "applied", "disabled", and "requires reload/refresh" for cache and texture controls.

## Rendering And Cache

- Point rendering was reworked to a data-oriented point cloud (`point_cloud.rs`).
  - Markers are no longer one entity per point; all point data lives in flat arrays in the `PointCloud` resource and points are baked into chunked meshes (~1024 points/chunk) with vertex colors, so ECS/draw cost scales with chunks, not points.
  - Billboard load/evict hides/shows points by rebaking only the affected chunk mesh, budgeted per frame (`POINT_CHUNK_REBUILDS_PER_FRAME`).
  - An adaptive view boundary (`ViewBounds`, `VISIBLE_POINT_BUDGET`) hides chunks beyond a budgeted radius around the camera and closes the box with opaque "More points beyond this point" walls.
  - Loading skips pending points beyond the walls, and loaded billboards beyond the walls or far behind the camera are evicted proactively (`evict_out_of_view_billboards`).
  - Follow-ups: tune `VISIBLE_POINT_BUDGET` / wall styling on a real large catalog; consider surfacing `ViewBounds::culled_points` in the performance pill.

- Exercise the merged billboard cache with a large mixed image/video catalog.
  - Verify spatial scheduling prioritizes visible/near/movement-direction billboards.
  - Verify cache eviction despawns far billboards and restores marker visibility.
  - Verify stale texture refresh re-decodes already-loaded billboards after texture or LOD changes.
  - Verify GPU image/material asset counts stop growing when changing cache and texture limits repeatedly.
  - Verify pending, in-flight, completed, loaded, failed, and evicted counts remain internally consistent.

- Tighten frame-budgeted receive behavior.
  - Add a maximum processed-results-per-frame guard in addition to upload/spawn/time budgets.
  - Verify bursts of failed decodes or stale decode results cannot monopolize a frame.
  - Add tests around completed queue draining for success, failure, stale texture limit, and over-capacity cases.

- Clean up video state during billboard eviction.
  - When evicting a loaded video billboard, close any active video controls for that image immediately.
  - Stop or invalidate playback state for the evicted image so stale controls/playback cannot persist for even one frame.
  - Add coverage for video-control cleanup during cache eviction, stale texture refresh, catalog reload, and dimension changes.

- Add a runtime `ffmpeg` availability check with a user-facing status path.
  - Detect missing or unusable `ffmpeg` before the first video poster/playback decode.
  - Show a clear UI status explaining that video poster/playback is unavailable because `ffmpeg` is missing from `PATH`.
  - Avoid logging repeated per-video failures once global `ffmpeg` unavailability is known.

- Add integration coverage for catalog reloads and dimension changes.
  - Confirm billboard entities despawn.
  - Confirm their Bevy `Image` and `StandardMaterial` assets are removed.
  - Confirm marker, axis, gizmo, video controls, playback state, navigation targets, billboard stats, and loading queues reset consistently.
