# Benchmarking a catalog

The Debug Tools pill benchmarks the catalog you actually have open, on your
hardware — as opposed to `tests/performance`, which measures synthetic fixtures
from outside the process. Each run records, per frame: main-thread time
attributed to each update stage (the same probe stamps the performance harness
uses), decode/upload pipeline counters, and CPU, memory and per-render-pass GPU
timings read from Bevy's diagnostics. GPU pass times are real timestamp queries
on Vulkan and DX12, and CPU-side pass time elsewhere.

The full record — every stage, the twelve slowest frames with their individual
breakdowns, and the hardware counters — is written to
`%LOCALAPPDATA%\generation_viewer\benchmarks\benchmark-<unix-seconds>.json`
(`$XDG_CONFIG_HOME/generation_viewer/benchmarks/` elsewhere); the panel shows a
summary and the path.

To script a run, use `--benchmark <seconds>`. The viewer loads the catalog,
waits for the loader to go quiet, then sweeps the camera across the catalog for
that many seconds so loading, quality refresh and eviction all stay active,
writes the report, prints its path and exits:

```powershell
cargo run --release -p generation_viewer -- --limit 3000 --benchmark 20
```

`bottleneck` names the first limit the run hit, in the order the pipeline hits
them: GPU upload budget, scheduler stall, decode-worker saturation, cache limit,
a dominant update stage, or none of those (frame time is render-bound). Read it
as a pointer into the numbers below it, not as a verdict on its own — in
particular `starved_fraction` counts frames where schedulable work existed and
no decode was running, which is a scheduling admission limit rather than an I/O
limit.

Note that Bevy's system CPU diagnostic reports 0 on some platforms (observed on
Windows 11 with sysinfo 0.32); the report carries `cpu_usage_available` so a
flat zero is not mistaken for an idle CPU.

Always benchmark release builds (`cargo run --release`).
Debug builds are unoptimized and report substantially lower FPS.
