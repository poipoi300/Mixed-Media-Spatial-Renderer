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

| Input | Action |
|---|---|
| Right mouse drag | Look around |
| `W` `A` `S` `D` | Move and strafe |
| `Space` / `Ctrl` | Up / down relative to the view |
| `Shift` | Move 1.5x faster |
| Mouse wheel | Change base speed |
| Arrow keys | Jump to the nearest image along the axis closest to screen-right / screen-up |
| Right `Shift` + up / down | Jump to the nearest image forward / backward |
| Left click | Select a billboard (click again to deselect, click empty space to clear) |
| `Shift` + left click | Add to the selection |
| Left drag on a selection | Move the selected billboards |
| Right click a video | Video menu: loop, speed, tracks, delays, save frame, open, copy path |
| `K` / `J` / `L` | Play-pause / seek back 5 s / seek forward 5 s, for selected videos |
| `Escape` | Pause menu |

The panels along the edge of the window:

- **View.** The server's controls and stats, plus a cutaway slider that hides
  everything closer than a given depth so you can see inside dense clusters.
- **Billboards.** Loaded texture counts and the **Texture VRAM** budget.
- **Navigation.** Movement stats and base speed.
- **Developer > Debug tools.** Catalog benchmark; see [docs/benchmarking.md](docs/benchmarking.md).

The axis gizmo at the bottom shows orientation and the server's axis names.
Click a face to snap the camera to that axis.

## Command-line options

```text
--api <URL>                  Server base URL [default: http://127.0.0.1:8765]
--control <ID=VALUE>         Initial value for a server control, repeatable
--limit <COUNT>              Maximum points to load [default: 10000]
--spacing <VALUE>            Coordinate spacing [default: 6.0]
--duplicates <VALUE>         Spread of points sharing a coordinate [default: 0.8]
--texture-budget-mib <MIB>   VRAM for resident textures [default: 6144]
--image-concurrency <COUNT>  Concurrent decode workers [default: 12]
--max-texture-side <PX>      Cap on decoded texture size; 0 = source [default: 0]
--billboard-scale <VALUE>    Billboard size relative to spacing [default: 0.78]
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
