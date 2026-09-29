# Rendering and loading design

Notes on how the viewer decodes, schedules, compresses and draws media, with
the measurements behind each decision. Numbers were taken on the author's
machine and catalog; treat them as relative, not absolute.

## Decoding and scheduling

Decode workers run on Bevy's async-compute pool, which the viewer sizes to the decode
budget (Bevy's default caps that pool at 4 threads regardless of core count, which
silently capped `--image-concurrency`). More decode concurrency is not automatically
better: decode threads contend with the render thread, and on a 32-thread machine
sweeping a 3000 point catalog, worst-case frame time scaled roughly linearly with
concurrency (4 workers 10 ms, 8 workers 18 ms, 12 workers 27 ms, 24 workers 49 ms)
while the sum of all update stages stayed under 1 ms. Those numbers predate BC7
compression, which made decoding CPU-bound; `--image-concurrency` now defaults to 12.
Lower it if frame consistency matters more than how fast the catalog fills.

The viewer renders through Vulkan by default. wgpu's DX12 backend removes the device
(`COMMAND_ALLOCATOR_RESET`, then a frozen window) or exhausts its descriptor heap after
sustained billboard streaming; set `WGPU_BACKEND=dx12` only to compare.

The API streams catalog snapshots beginning with the first discovered media file;
the camera starts at that coordinate while later snapshots update the coordinate
space in place without moving the player or previously placed billboards again.
Which image is decoded next is chosen by a weighted random draw rather than by a
queue. Every candidate — a point with no texture, and a resident whose texture no
longer suits its distance — is scored by one Gaussian centred on the camera (or on
where the camera is heading), narrowed by view direction, and the frame's free decode
workers are filled from a single sample over all of them. There are no per-lane worker
rations, so a free worker is filled whenever any work exists and nothing can be starved
by a lane boundary: a distant image is not forbidden, just drawn with vanishing
probability. Points outside the render boundary score exactly zero, since the boundary
walls hide them at any quality.
Every image is decoded once, at source resolution, and uploaded as one GPU texture.
There is no LOD ladder. Distance-based tiers were measured against the alternative on a
real catalog and lost on both counts: because reading and parsing the file is ~70% of a
decode, full quality costs only 1.12x the cheapest tier, while sizing images down left
only 22% of on-screen billboards at the resolution their position deserved, against
99.5% at source resolution. `--max-texture-side` remains as a global cap (default `0`,
meaning source resolution, bounded by the device's `max_texture_dimension_2d`); videos
stay capped at 1024 px, since a frame is replaced 12 times a second.

Billboards behind the camera are hidden but retained and preloaded
inside a turn-around buffer. Each playing video streams its picture and its sound through
one long-running FFmpeg process each, restarted only when playback jumps somewhere the
stream cannot reach by decoding forward; the picture decodes on the GPU where FFmpeg
supports it. A playing video's clock follows its sound, so picture and sound stay together
through frame hitches. Each file is probed once with FFprobe for its tracks and frame rate;
chosen subtitles are drawn into the frames by libass as they decode. How each file plays
(volume, mute, tracks, delays, speed, looping, where it was left) is saved per file in
`media_settings.json` in the per-user state directory.

Because a decode can no longer be superseded by a different target resolution, the only
in-flight cancellation left is for points the camera has put outside the render
boundary — those are hidden behind the boundary walls, so finishing them could not show
anything.

Billboard textures are BC7 block-compressed on the decode worker, which the GPU samples
natively at 1 byte per texel instead of 4. On a 3000 point sweep that took resident
VRAM from **27.2 GiB to 7.3 GiB** for the same content. Compression is visually
lossless in measurement — 50.7 dB PSNR, mean error 0.4/255 on a sample catalog
— and each square's side is rounded up to a multiple of 4 so every surface is encodable
(21% of a sampled catalog has a side that is not). An adapter without
`TEXTURE_COMPRESSION_BC` falls back to RGBA8 and says so at startup.

**One contiguous surface per billboard, covering the image and not the square.**
Fitting a non-square source onto a square canvas is what introduces transparency in the
first place — only 1.1% of the test catalog is square, and essentially no source image
carries real alpha (6.8% of PNGs, under 0.5% of their area). Encoding only the image
rectangle means the letterbox margin is never encoded, never uploaded, and never drawn,
and the surface stays fully opaque — which matters because any transparent texel forces
the alpha-aware BC7 preset, measured **~5x slower** than the opaque one.

Two details keep it opaque. The centring offset is snapped down to a block boundary, so
the crop starts on one without reaching into the margin. And where the image's own
dimensions are not multiples of 4 (41% of the catalog), its edge texels are repeated
across the at-most-3px alignment sliver rather than letting the surface grow into
transparent padding. Block alignment only ever grows the rectangle *outward*, so it can
pull in padding but never crop an image pixel.

The benchmark reports the split under `surface_encoding`, because a throughput number
cannot show it when a later stage binds. Measured on 25 s fills from an empty cache,
8000 points, same binary:

| | all-alpha | per-region preset | + cropped to image |
|---|---|---|---|
| encode CPU | 380.6 s | 294.3 s | **76.0 s** |
| opaque fraction | 0% | 30% | **97%** |
| images loaded | 1221 | 1291 | **1362** |
| decode busy | 0.925 | 0.832 | **0.657** |

**5.0x less encode CPU** for 1.12x the images. What remains alpha-encoded is a source
image that genuinely carries transparency. Steady-state numbers understate all of this:
once the VRAM budget is full only one replacement decode may be in flight at a time, so
most workers are idle by design and faster encoding cannot show up at all.

**Why one surface rather than a tile grid.** Tiles were a grid of separate meshes, each
with its own model matrix. A rasterizer only guarantees watertight adjacency between
triangles whose transformed vertices are bit-identical along the shared edge, and two
neighbours reach that edge through different matrices — differing by an ULP or two once
the billboard's rotation and world position are applied. With MSAA off, a pixel on the
boundary is then claimed by both quads or by neither, and neither shows the clear
colour: a one-pixel dark seam, visible only at the distances and orientations where the
error crosses a pixel centre. A 1-texel tile overlap was tried against this and did not
work. A single surface has no interior edge, so the seam cannot exist.

The grid was never a batching win — it cost 14.4 draw calls per image on average where
one now suffices. What it bought was upload pacing, splitting a large image across
frames against the 4 MiB per-frame budget. A surface cannot be split, so that budget now
admits the first upload of a frame however large it is (median catalog image ~8.9 MB of
BC7, largest ~90 MB) and stops anything else joining it. Measured on the same 25 s fill,
that traded nothing: identical VRAM, same MiB uploaded, and **229 fps against 158, p99
8.2 ms against 17.5 ms, worst frame 17.8 ms against 35.3 ms**. Per-frame partial uploads
into a pre-allocated texture remain available if a worse case turns up.

Because the whole image is one texture, the device's `max_texture_dimension_2d` now
bounds the image rather than a 1024 px tile, and source resolution is clamped to it.
The test catalog's largest side is 11656 px against 32768 on the test GPU, so the clamp is a
guard rather than a routine path.

The resident set is bounded by a **VRAM byte budget**, not an image count — catalog
images differ ~100x in size, so a count cannot bound the memory that actually runs out.
It is the "Texture VRAM" slider in the Billboards panel (also `--texture-budget-mib`,
default 6144). Eviction is unchanged: the worst-placed billboard goes first, and the
last resident is never evicted. GPU transfers are byte-budgeted per frame, and
coordinate labels are rasterized lazily only while their display option is enabled.


## Layout

Projection coordinates are laid out with image width/height metadata: coordinate
slots grow when an image footprint needs more room, and duplicate records at the
same coordinate are packed into an extent-aware 3D grid.


## Window and movement

The viewer enforces a `960 x 540` minimum window size to keep the renderer,
gizmo viewport, and UI layout inside valid bounds during resize.

Set `GENERATION_VIEWER_GIZMO_FONT` to a TrueType/OpenType font path if the
gizmo cannot find a system font for its 3D axis labels.

Movement speed is a fixed function of the base speed, an optional `Shift`
boost, and cruise pressure, which rises while flying steadily forward and
decays when you stop or reverse. Speed no longer varies with the images
around the camera.
