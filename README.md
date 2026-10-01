# Mixed Media Spatial Renderer

A 3D viewer for large collections of images and videos, built with
[Bevy](https://bevyengine.org/). Every file becomes a billboard placed in space,
and you fly through the collection like a level in a game.

The viewer doesn't decide where anything goes. A small HTTP server tells it
where each file sits, what the axes are called, and which controls to show.
Swap the server and you get a different layout of the same files. Everything
the viewer needs from a server is described in [docs/api.md](docs/api.md).

## Features

- **Scales to large catalogs.** Tens of thousands of points. Files are decoded
  near the camera first, BC7-compressed before upload, and evicted under a
  VRAM budget you control, not an image count.
- **Images and video.** Videos play in place with sound, seeking, speed
  control, audio and subtitle track selection, and per-file remembered
  settings.
- **Server-described UI.** The server sends a panel of widgets (dropdowns,
  sliders, buttons, text fields, toggles, groups), and the viewer renders it.
  The server decides whether changes apply immediately or as a form submission.
- **Streaming loads.** The server can send the catalog as a series of
  snapshots, so the first files appear while the rest are still being scanned.
- **Built-in benchmarking.** Records frame times, decode pipeline counters and
  GPU pass timings for the catalog you have open, on your hardware.

## Requirements

- Rust 1.89. `rust-toolchain.toml` selects it automatically under `rustup`.
- A GPU with Vulkan support. The viewer uses Vulkan by default; see
  [docs/design.md](docs/design.md) for why not DX12.
- [FFmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on `PATH`), for video
  only. Images work without it.

## Quick start

`shape_api` is a small reference server included in this workspace. It
arranges every image and video under the folders you pick into a sphere, cube,
ring or spiral. Start it, then point the viewer at it:

```sh
cargo run --release -p shape_api -- --port 8766
cargo run --release -p spatial_viewer -- --api http://127.0.0.1:8766
```

On first launch the scene is empty. Press `Escape`, open **Settings >
Catalog options**, and add one or more folders. The viewer remembers them and
reloads them on the next launch.

Always use `--release`. Debug builds are unoptimized and run far slower.

## Controls

`F1` (or **Controls** in the pause menu) lists every control. The list is
built from the same table the viewer reads its input through, so it always
matches what the keys do. It is also where controls are rebound: each has two
slots, which take a key, a mouse button, a combination with `Ctrl`, `Shift`
or `Alt`, or a double-tap, and a held control can be set to toggle instead.
A search box narrows the list, and bindings that get in each other's way are
flagged. Changes are saved as they are made. The defaults:

| Input | Action |
|---|---|
| Right mouse drag | Look around; a crosshair stands in for the pointer at the center of the view |
| `W` `A` `S` `D` | Move and strafe |
| `Space` / `C` | Up / down relative to the view |
| `Shift` | Move 1.5x faster |
| Mouse wheel | Change base speed |
| Arrow keys | Jump to the nearest image along the axis closest to screen-right / screen-up |
| Right `Shift` + up / down | Jump to the nearest image forward / backward |
| Left click | Select a billboard, a closed folder's previews included (click again to deselect, click empty space to clear); open a closed folder elsewhere on its cube |
| Left drag on empty space | Box-select the pictures and closed folders inside the box |
| `Shift` + left click or box | Add to the selection |
| Left drag on a selection | Move the selected billboards and folders |
| `F` | Fit the selection to the screen; double-tap to fit everything |
| `Ctrl+A` | Select every picture and closed folder shown |
| `Ctrl+Z` / `Ctrl+Shift+Z` (or `Ctrl+Y`) | Undo / redo the last arrangement change |
| `Delete` | Delete the selected folders |
| Right click a billboard | Add it to a new folder; for a video also loop, speed, tracks, delays, save frame; open, copy path |
| Right click a folder | Rename, move or delete the folder; show or hide its tag |
| Right click empty space | Create a folder there; undo or redo |
| `K` / `J` / `L` | Play-pause / seek back 5 s / seek forward 5 s, for the video under the pointer, else the selected videos |
| `Escape` | Stop typing, then clear the selection, then the pause menu |
| `H` | Hide or show the interface: panels, axis gizmo and video controls (menus you open still show) |

The panels along the edge of the window:

- **View.** The server's controls and stats, plus a cutaway slider that hides
  everything closer than a given depth so you can see inside dense clusters.
- **Billboards.** Loaded texture counts, the **Texture VRAM** budget, whether
  billboards grow a little under the pointer, and whether they show their
  coordinates, one line per axis in that axis's color.
- **Folders.** How many folders the scene has and how many are open, with
  **Open all** and **Close all**, and which parts of a folder show: its tag
  (a folder's right-click menu can override this for that folder), the
  minimize icon of an open folder, and the cube's background. With all three
  off a folder shows only while it is selected or something is dragged over
  it, and still takes clicks where its cube stands. All three are kept
  between sessions.
- **Search.** Finds images by file name as you type. **Select shown** selects
  the matches the scene shows; **Next** (or `Enter`) flies to the next match,
  opening the folders it is in.
- **Undo** and **Redo** for everything you arrange by hand: moves, new,
  renamed and deleted folders.
- **Navigation.** Movement stats and base speed.
- **Developer > Debug tools.** Catalog benchmark; see [docs/benchmarking.md](docs/benchmarking.md).

Images a server groups together (for example, every image sharing a
coordinate) show as a **folder**: a grey cube holding small previews of a few
of them. A tag hangs from its lower left corner, as you see the cube: a badge
counting its images, and its name, which is the coordinate everything in it
shares along each axis, in that axis's color (a name you give it is white).
A closed folder grows a little under the pointer, as images do, except while
the pointer is on one of its previews: that preview is what a click selects.
Click the cube anywhere else to open it: it grows to hold all its images, turns see-through,
and pushes its neighbours aside. An open folder takes no clicks, so everything
inside and behind it stays reachable; the close icon at its upper right corner
closes it. Clicking a folder's tag selects the folder without opening or
closing it, ready to drag; `Shift` + click adds it to the selection.

Folders are yours to rearrange. Drop an image or a folder into a folder's cube
and it joins that folder; drag it out and it stays where you left it. Make new
folders from the right-click menu, inside other folders too: a folder inside
another moves with it and shrinks into it when it closes. Deleting a folder
moves what it held up a level, into the folder around it or out of every
folder; a server's folder deleted in place leaves its images where the layout
had them. The viewer remembers what you arranged for each catalog and view,
and brings it back when you load that view again. **Reset layout** in Catalog
options forgets it and lays the view out as it first opened, with every
folder closed and the camera back at its starting view; **Undo** brings the
arrangement back. It also remembers where you left the camera in each view
(position, direction and base speed) and puts it back there when the view
loads; **Remember camera position per catalog** in **Settings** turns that
off.

The axis gizmo at the bottom shows orientation and the server's axis names.
Click a face to snap the camera to that axis.

**Animations** in the pause menu turns the viewer's animations off, all at
once or one by one, and scales how long they take (above 1x is slower):
hover growth (of images and closed folders), images sliding to new places,
images sliding in and out of folders, folder cubes opening and closing, camera flights to what `F` or search fits, and a video's control
strip following the part of the video on screen. An animation turned off
jumps straight to where it ends; a strip that does not follow the view stays
put along its video's bottom edge. Moving the view, and billboards turning to
face it, are not animations. The settings are kept between sessions.

## Command-line options

```text
--api <URL>                  Server base URL [default: http://127.0.0.1:8765]
--control <ID=VALUE>         Initial value for a server control, repeatable
--limit <COUNT>              Maximum points to load [default: 10000]
--texture-budget-mib <MIB>   VRAM for resident textures [default: 6144]
--image-concurrency <COUNT>  Concurrent decode workers [default: 12]
--max-texture-side <PX>      Cap on decoded texture size; 0 = source [default: 0]
--max-video-fps <FPS>        Cap on video playback rate [default: 60]
--benchmark <SECONDS>        Benchmark the catalog once loaded, write a report, exit
```

`cargo run --release -p spatial_viewer -- --help` prints the full
descriptions.

## Writing a server

A server implements three endpoints: `GET /health`, `POST /catalog/stream`
and `POST /projection`. It keeps no per-client state; every request carries
all control values. [docs/api.md](docs/api.md) describes the contract, and
[`crates/shape_api`](crates/shape_api) is a complete implementation in under
2,000 lines, tests included. The wire types live in
[`crates/spatial_api`](crates/spatial_api) and can be reused from Rust.

## Workspace layout

| Crate | Purpose |
|---|---|
| `spatial_viewer` | The viewer binary: rendering, decoding, video, input |
| `spatial_viewer_ui` | Panels, menus and the server-described control renderer |
| `spatial_api` | HTTP client and wire schema shared by viewer and servers |
| `spatial_geometry` | Engine-free geometry helpers |
| `shape_api` | Reference server |

`tests/performance` holds a harness that runs several viewer processes
against synthetic catalogs and compares frame-time statistics to a baseline.

## Further reading

- [docs/api.md](docs/api.md): the server contract
- [docs/design.md](docs/design.md): how decoding, scheduling, compression and
  drawing work, with the measurements behind each choice
- [docs/benchmarking.md](docs/benchmarking.md): the in-app benchmark

## License

[MIT](LICENSE)
