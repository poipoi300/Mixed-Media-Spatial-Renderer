# The viewer API

The viewer holds no knowledge of what its data means: it renders whatever an
HTTP server describes. This is that contract. `crates/shape_api` is a complete,
small reference implementation, and `crates/spatial_api` holds the shared
wire types.

Three endpoints drive the viewer. Any server that implements them can drive it;
`crates/shape_api` is a complete one that arranges a folder of media into 3D
shapes:

```powershell
cargo run --release -p shape_api -- --port 8766
cargo run --release -p spatial_viewer -- --api http://127.0.0.1:8766
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `POST /catalog/stream` | NDJSON snapshots; a server may send exactly one |
| `POST /projection` | one snapshot, for a control change |

Both POSTs take the same body: the folder roots the start menu picked, every
control value, and the id of the control the user activated (which is what
distinguishes a button press from a value change). Nothing is kept per client,
so a reconnect cannot desynchronize.

A snapshot answers with the points **and** the panel whose controls produced
them, so the pill always describes what is on screen:

```json
{"kind": "snapshot", "complete": true, "roots": ["..."],
 "panel": {"revision": 1, "summary": "Sphere  120 pts",
           "stats": [{"label": "shown", "value": "120 / 500"}],
           "error": null, "widgets": [ ... ]},
 "projection": {"axis_labels": ["Longitude", "Latitude", "Depth"],
                "total": 500, "points": [ ... ]}}
```

`axis_labels` is what the gizmo and billboard coordinate labels display — the
server names its own axes, so the viewer needs no notion of what produced the
layout.

**Positions are in cubes.** Every image fills a 1×1×1 cube, and a server lays
cubes out touching: neighbours one unit apart.

**Groups.** A point may name the group it belongs to, for example every image
sharing a coordinate value. The viewer shows a group of two or more images as
a folder, closed until the user opens it, and lays groups out itself so a
closed folder takes one cube and an open one its whole block:

- `key` names the group, and must stay the same across snapshots even when
  its rank changes.
- `index` is the group's slot along each axis, counting from 0 at the low
  end. Groups sharing a slot line up, and each slot is as wide as the widest
  thing the viewer shows in it.
- `offset` is the point's cube center relative to the center of its group's
  block.

`position` still gives the point's place with every group laid out whole, for
a client that does not lay groups out. A server without groups omits `group`,
and the viewer then uses `position` as sent.

```json
{"image_id": 7, "path": "...", "position": [2.0, -0.5, 0.0],
 "group": {"key": "2024-05-01", "index": [1, 0, 0], "offset": [0.0, -0.5, 0.0]},
 "media_type": "image", "coordinate_labels": ["2024-05-01", null, null]}
```

**Widgets.** Six kinds: `group` (nests, optionally collapsible), `select`,
`button`, `slider`, `text` and `toggle`. Each leaf carries an `id`, a `label`,
its current `value`, and optionally `detail` and `disabled`.

**`submits` is how a server declares its apply model.** Interacting with a
`submits: true` widget sends the request; a `submits: false` widget only
updates the viewer's local values until something else submits. So apply-on-
change is every widget submitting, and a submit-style form is a run of
non-submitting fields plus one submitting button. The pill marks any control
edited since the last submission, so an unsent form is never mistaken for
applied state.

**`revision`** changes only when the widget *structure* does — one added,
removed, reordered, or its kind changed. Values, stats, labels and errors move
freely without it. The viewer rebuilds its widget entities when it changes, so
a server that bumps it needlessly makes the panel flicker.

Two error paths, both needed: `panel.error` is the server rejecting the values
it was sent (the scene stays as it was), while `{"kind": "error", "message":
"..."}` ends a stream.

