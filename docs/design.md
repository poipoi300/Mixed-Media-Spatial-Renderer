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
queue. Every point with no texture is scored by one utility: it falls with the fourth
power of distance from the camera (or from where the camera is heading), and a point
behind the camera counts as up to 8x farther than it is. The frame's free decode workers
are filled from a single sample over all candidates. There are no per-lane worker
rations, so a free worker is filled whenever any work exists. Points outside the render
boundary score exactly zero, since the boundary walls hide them at any quality.

Candidates are gathered by a best-first walk over a spatial grid, visiting cells in
order of the most any point inside them could be worth, so the visited region stretches
out along the view rather than forming a cube around the camera. The walk's per-frame
budget ends a few widths out in a dense catalog, so the rest of the view is sampled
instead: positions drawn in a cone ahead of the camera, log-uniform in distance, offer
the points of the cells they land in, weighted by value over the chance that cell was
probed. Across frames the draw then follows the utility all the way out — the number of
draws at each distance falls monotonically, and a billboard you are looking at is
reachable however far away it is.

The same utility decides eviction (least valuable first) and admission: once the VRAM
budget is full, a candidate is only drawn if it is worth more than the least valuable
resident, and a decode to replace residents only starts if the candidate is worth at
least 4x each one it displaces. That margin keeps a moving camera from trading
near-equals back and forth, while a turn — where the image now in view is worth
thousands of times the one behind — clears it easily. Any number of replacements may be
in flight; each reserves the residents it will displace, which stay on screen until it
lands.
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
image that genuinely carries transparency. These numbers were taken when only one
replacement decode could be in flight once the VRAM budget was full, so steady-state
workers were mostly idle and they understate what faster encoding buys.

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

A server places every image in a unit cube and lays the cubes out touching, and
may name each image's group. The viewer lays the groups out itself, on a grid of
touching slots (`SlotGrid` in `spatial_geometry`) where each slot is as wide as
the widest thing it shows: one cube for a closed folder or a single image, the
whole block plus a margin for an open folder. Only what is shown becomes a point,
so a closed folder's hidden images are never loaded.

Opening or closing a folder re-lays out the loaded scene off-thread, with no
server round trip, and shifts the whole layout so the toggled folder (through the
outermost folder around it) stays put while its neighbours move. Moved billboards
and folder cubes spring to their new places; images a closing folder hides fly
back into it; images an opening folder reveals fly out of it as their textures
load, and their placeholders wait for the cube to grow around them.

What the user arranges by hand is kept apart from what the server sent: where
each dropped image or folder went (loose in world space, or at an offset inside
a folder), the folders they made, named and deleted. Every layout applies it on
top of the server's groups, so it survives relayouts and streamed snapshots. A
folder moved out of its slot stops taking room on the grid. Folders nest, and a
nested folder shows only while every folder around it is open.

Each change to the arrangement that settles while no drag is under way is one
undo step, so a whole drag undoes at once; opening and closing folders
arranges nothing and is not undone. The arrangement is also kept per view (a
catalog's roots plus the server's control values) in a JSON file in the
per-user state directory. Images are kept there by path, since the ids a
server hands out last only one load, and loose positions relative to the
layout's origin, where a load starts; a saved placement applies to its image
whenever a snapshot brings it, and a new placement by id replaces it. The
same file keeps, per view, where the camera was left (relative to that
origin, with its direction and base speed), taken whenever the camera has
held still for a frame while the setting is on; a load of the view places
the camera there instead of at the initial view, once its first snapshot
arrives.

Everything in a folder, image or folder, takes a cell of the folder's own grid,
and the cells pack like the scene's slots: each row, column and layer is as
wide as the widest thing in it, so a folder opening inside another pushes its
neighbours aside rather than covering them. The block is centered on the
folder, and the folder just opened or closed stays where it was. Something
dropped into a folder takes the cell under it, or the nearest free one.

Every animation reads `AnimationSettings`, which the pause menu edits: each
can be switched off, and all advance by real time divided by one duration
scale. Springs step at most 1/30 s at a time, however far a frame advances
them, so they stay stable at any scale. A billboard's scale is its layout's
scale times its hover growth (`BillboardGrowth`), and the two spring apart,
so growth can animate while rearranging jumps, or the other way round. An
animation switched off lands on its end state the same frame: billboards and
cubes take their targets, retiring images vanish, placeholders show at once
and a camera flight ends where it would have.

A folder's cube draws only its far faces, from the inside, so no wall stands
between the viewer and what the folder holds. An open folder's walls are a
backdrop besides: their fragments sit at the far plane, so they show only where
nothing else was drawn, and no open folder the view looks through or stands in
hides anything. Blended meshes sort by their origin alone, which the images
inside share with the cube, so each wall material carries a sort bias that
draws the walls first.

Clicks go to what the viewer sees: pictures and closed folders take them
through any open folder in front, and an open folder takes none itself. A
closed folder's previews are pictures inside its cube: the ray enters the
cube first, but a preview it reaches takes the click, and the cube takes
only what misses them. Hovering resolves the same way, so what grows under
the pointer is what a click would take: an image, or a closed folder's cube
(by its own growth factor, kept apart from its layout size like a
billboard's). Each
folder has handles instead, camera-facing quads that follow its cube as it
springs: a tag (a count badge and the folder's name; a click selects the
folder without toggling it) hanging from the corner the view sees lowest and
leftmost, and while it
is open a close icon on the corner it sees highest and rightmost, sized by
distance so it reads alike near and far. Among the corners in front of the
camera, those are the ones whose projections reach furthest that way, so the
handles follow the camera round the cube. A folder's own name is the value of
each axis everything in it shares, each in the axis's gizmo color, as are the
lines of a billboard's coordinates. Whether folders show their tags, their
close icons and their cubes' walls is kept with the arrangements, and a
folder may override the first for itself (kept with its view's arrangement).
A handle that does not show takes no press. A cube without walls is hidden
unless it is selected or a drop target, but it still springs, grows under
the pointer and takes presses, since those come from the folder's layout, not
its mesh. Only folders near the camera get tags, a
few more each frame, so a scene of thousands of folders pays for the ones the
user can read.

A video's control strip is laid out in screen space over the part of its
picture on screen, so it moves whenever the view or the picture does. With
that switched off (an animation, as far as the settings go), the strip is
laid out once as if the picture showed at a fixed size, and fixed on the
picture: it moves, turns and scales with the picture alone.

## Controls

Every control the viewer answers to is an `Action` in one table
(`spatial_viewer_ui::input_bindings`) that carries its category, description
and default binding; an action that uses another's input with a different
gesture (a drag of the button that selects on a click) names that action. The
viewer reads input only as actions, and a test fails if any source reads a key
or mouse button directly, so the controls sheet, which lists the table, cannot
fall out of step with what the keys do. Keys stand down while text is being
typed, except the key that ends the typing.

Each action that owns its binding has two slots, each holding a chord: an
input, the modifiers held with it (either side of `Ctrl`, `Shift`, `Alt`),
and whether it is double-tapped. When a press matches chords on the same
input, the one with the most modifiers, all held, takes it, so `Ctrl+Shift+Z`
redoes without also undoing; and a second tap goes to a double-tap on the
same chord alone, so the second `F` fits everything without fitting the
selection again. Actions that only change what another does while held
(moving faster, stepping through depth) never take a press this way: they
add to it. A held action can toggle instead; its on state
is kept frame to frame, with the taps a double-tap needs. The controls sheet
rebinds a slot by capturing the next input (`Esc` cancels, `Backspace`
empties the slot; a press waits the double-tap window for a second one), and
nothing else reads input meanwhile. It flags conflicts: two actions read at
the same time on one chord, unless both only change what another action does
while held (moving faster, stepping through depth), which may share a key.
The pause-menu action always keeps a binding. Only how the bindings differ
from the defaults is saved, as action and chord names in `controls.json` in
the per-user state directory, so a later viewer's new defaults still reach
every control the user has not changed.


## Window and movement

The viewer enforces a `960 x 540` minimum window size to keep the renderer,
gizmo viewport, and UI layout inside valid bounds during resize.

Set `SPATIAL_VIEWER_GIZMO_FONT` to a TrueType/OpenType font path if the
gizmo cannot find a system font for its 3D axis labels.

Movement speed is a fixed function of the base speed, an optional `Shift`
boost, and cruise pressure, which rises while flying steadily forward and
decays when you stop or reverse. Speed no longer varies with the images
around the camera.
