# Rendered performance emulation

This suite launches six real viewer processes concurrently against isolated
local streaming APIs:

- one image;
- one video;
- 50,000 mixed image/video records;
- `mixed-sizes-patrol`: 2,000 images cycling through 128 to 3072 px sides,
  with the camera sweeping the whole catalog for the entire run so loading,
  quality refresh, multi-tile upload and eviction never stop. This is the
  frame-time uniformity workload.
- `mixed-sizes-parked`: the same catalog with the camera never moving. Once
  the texture budget is full nothing in view changes, so the heartbeat's
  `billboards.cache_churn` counters after the fill measure cache churn alone.
- `mixed-sizes-parked-inside`: the same catalog with the camera parked at its
  centre facing +X, then turned in place to face -X. The heartbeat's
  `visible_billboards` / `visible_textured` show whether looking at
  billboards, without flying toward them, is enough to load them.

The default run lasts five minutes per process. The fixture sends the first
point immediately, then geometrically larger full snapshots whose coordinates
change as the catalog extent grows. Its 1024 px image and 512 px video exercise
preview, high-quality, and GPU-upload behavior rather than only tiny placeholder
textures. Seeded timing jitter and assertion grace periods keep the test
sensitive to freezes and state failures without assuming identical frame timing
on every machine.

Run from the repository root:

```powershell
uv run --with psutil viewer/tests/performance/run.py
```

Use a shorter local smoke run while changing the harness:

```powershell
uv run --with psutil viewer/tests/performance/run.py --duration 30 --massive-points 5000
```

Each scenario writes `viewer.jsonl` (frame-time percentiles, FPS, catalog,
billboard, video, camera and action state), `process.jsonl` (CPU, RSS, VMS,
thread and child-process measurements), plus stdout/stderr logs. A top-level
`summary.json` gives the final result. The watchdog terminates and fails a
viewer that stops producing heartbeats.

Every scenario records frame-time uniformity alongside the percentiles:
standard deviation, coefficient of variation, the fraction of frames slower
than twice the median (`spike_fraction`), the longest run of such frames,
and the mean absolute difference between consecutive frames. The completion
event also carries a `steady_state` block (frames after the 3 s warmup) and
its slowest frames with the loading state they occurred in plus per-stage
main-thread timings of the surrounding updates, so a spike can be attributed
to a specific system or to render/present time.

Run a subset with `--scenarios mixed-sizes-patrol massive-mixed`, and size
the patrol with `--patrol-points`.

Create a machine-specific regression baseline from a successful run:

```powershell
uv run --with psutil viewer/tests/performance/run.py `
  --write-baseline viewer/tests/performance/baseline.local.json
```

Compare a later run with it:

```powershell
uv run --with psutil viewer/tests/performance/run.py `
  --baseline viewer/tests/performance/baseline.local.json
```

The comparison allows a 20% FPS reduction plus 2 FPS, a 30% p95 frame-time
increase plus 2 ms, and a 25% peak-RSS increase plus 64 MiB. Steady-state
uniformity is compared too: coefficient of variation (30% + 0.05), spike
fraction (50% + 0.5 pt), longest spike burst (30% + 8 ms) and p99 (30% +
2 ms). Baselines are hardware-specific and should be stored as CI artifacts
rather than committed.
