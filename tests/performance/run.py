#!/usr/bin/env python3
"""Run the rendered spatial-viewer performance scenarios in parallel."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import struct
import subprocess
import threading
import time
import zlib
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import psutil
except ImportError as error:
    raise SystemExit("psutil is required; run with `uv run --with psutil viewer/tests/performance/run.py`") from error


ROOT = Path(__file__).resolve().parents[3]
VIEWER_ROOT = ROOT / "viewer"
DEFAULT_ARTIFACT_ROOT = VIEWER_ROOT / "target" / "performance"
MIB = 1024 * 1024
#: Frames before this are excluded from the steady-state uniformity summary
#: (window creation and pipeline compilation, not rendering behaviour).
WARMUP_SECONDS = 3.0


@dataclass(frozen=True)
class Scenario:
    name: str
    point_count: int
    media_pattern: tuple[str, ...]
    expected_video: bool
    #: Harness timeline: ``interactive`` discrete actions, or ``patrol`` one
    #: continuous camera sweep that keeps loading and evicting all run long.
    timeline: str = "interactive"
    #: Image sizes cycled through the catalog; a single entry means every
    #: image point shares the one fixture image.
    image_sides: tuple[int, ...] = (1024,)


#: Square image sizes for the mixed-size patrol: small thumbnails, the
#: single-tile maximum, and images that need several GPU upload tiles.
MIXED_IMAGE_SIDES = (128, 512, 1024, 2048, 3072)


@dataclass
class RunningScenario:
    scenario: Scenario
    server: FixtureServer
    process: subprocess.Popen[bytes]
    ps_process: psutil.Process
    config_path: Path
    viewer_metrics_path: Path
    process_metrics_path: Path
    stdout_file: Any
    stderr_file: Any
    started: float = field(default_factory=time.monotonic)
    last_heartbeat_wall: float = field(default_factory=time.monotonic)
    viewer_metrics_offset: int = 0
    frozen: bool = False
    forced_stop: bool = False


class CatalogFixture:
    def __init__(
        self,
        scenario: Scenario,
        fixture_directory: Path,
        image_paths: dict[int, Path],
        video_path: Path,
    ) -> None:
        self.scenario = scenario
        self.fixture_directory = fixture_directory
        self.image_paths = image_paths
        self.video_path = video_path
        missing = set(scenario.image_sides) - set(image_paths)
        if missing:
            raise ValueError(f"scenario {scenario.name} needs fixture images of sides {sorted(missing)}")

    @property
    def roots(self) -> list[str]:
        return [str(self.fixture_directory.resolve())]

    AXIS_LABELS = ("Column", "Row", "Layer")

    def panel(self, count: int | None = None, axes: list[int] | None = None) -> dict[str, Any]:
        """Return the control panel this fixture publishes to the viewer.

        Three axis selects, matching what a dimension-shaped API offers. The
        revision is constant: the viewer rebuilds its widgets when it moves,
        and a reprojection is not a structural change.
        """
        count = count or self.scenario.point_count
        side = max(1, math.ceil(count ** (1 / 3)))
        axes = axes or [0, 1, 2]
        options = [{"value": "none", "label": "None"}] + [
            {"value": str(index), "label": label, "detail": f"{side} coords"}
            for index, label in enumerate(self.AXIS_LABELS)
        ]
        return {
            "revision": 1,
            "title": "Dimensions",
            "summary": f"Dims {'/'.join(str(axis) for axis in axes)}  {count} pts",
            "stats": [{"label": "shown", "value": f"{count} / {self.scenario.point_count}"}],
            "error": None,
            "widgets": [
                {
                    "kind": "group",
                    "id": "axes",
                    "label": "Axes",
                    "collapsible": True,
                    "children": [
                        {
                            "kind": "select",
                            "id": control_id,
                            "label": control_id.upper(),
                            "value": str(axis),
                            "options": options,
                            "submits": True,
                        }
                        for control_id, axis in zip(("x", "y", "z"), axes, strict=True)
                    ],
                }
            ],
        }

    def point(self, image_id: int, count: int) -> dict[str, Any]:
        side = max(1, math.ceil(count ** (1 / 3)))
        x = image_id % side
        y = (image_id // side) % side
        z = image_id // (side * side)
        center = (side - 1) / 2
        position = [(x - center) * 6.0, (y - center) * 6.0, (z - center) * 6.0]
        media_type = self.scenario.media_pattern[image_id % len(self.scenario.media_pattern)]
        if media_type == "video":
            path = self.video_path
            side = 512
        else:
            side = self.scenario.image_sides[image_id % len(self.scenario.image_sides)]
            path = self.image_paths[side]
        return {
            "image_id": image_id,
            "path": str(path.resolve()),
            "position": position,
            "canonical_position": position,
            "width": side,
            "height": side,
            "media_type": media_type,
            "duration_seconds": 3.0 if media_type == "video" else None,
            "footprint": [1.0, 1.0],
            "coordinate_labels": [str(x), str(y), str(z)],
        }

    def projection(self, count: int, axes: list[int] | None = None) -> dict[str, Any]:
        axes = axes or [0, 1, 2]
        return {
            "axis_labels": [self.AXIS_LABELS[axis] for axis in axes],
            "coordinate_spacing": 6.0,
            "duplicate_spacing": 0.8,
            "sprite_world_height": 3.9,
            "offset": 0,
            "limit": self.scenario.point_count,
            "total": count,
            "points": [self.point(image_id, count) for image_id in range(count)],
        }

    def catalog_summary(self) -> dict[str, Any]:
        video_count = sum(
            1
            for image_id in range(self.scenario.point_count)
            if self.scenario.media_pattern[image_id % len(self.scenario.media_pattern)] == "video"
        )
        return {
            "roots": self.roots,
            "cache_path": str(self.fixture_directory / "catalog.json"),
            "plot_count": 1,
            "cache_used": False,
            "error_count": 0,
            "point_count": self.scenario.point_count,
            "dimension_count": 3,
            "plots": [
                {
                    "index": 0,
                    "name": self.scenario.name,
                    "folder": str(self.fixture_directory),
                    "result_path": str(self.fixture_directory),
                    "cell_count": self.scenario.point_count,
                    "image_count": self.scenario.point_count,
                    "axes": [],
                }
            ],
            "workflow": {
                "roots": self.roots,
                "cache_path": str(self.fixture_directory / "workflow.json"),
                "usable_images": self.scenario.point_count - video_count,
                "usable_videos": video_count,
                "discovered_count": self.scenario.point_count,
                "cache_hits": 0,
                "parsed_count": self.scenario.point_count,
                "invalidated_count": 0,
                "pruned_count": 0,
                "error_count": 0,
            },
        }

    def stream_snapshots(self) -> Iterator[dict[str, Any]]:
        count = self.scenario.point_count
        if count == 1:
            # Keep the first one-point snapshot incomplete long enough for
            # the viewer harness to observe rendering while loading is true.
            batch_sizes = [1, 1]
        else:
            batch_sizes = [1, 64, 512, 4_096, count]
            batch_sizes = sorted({min(size, count) for size in batch_sizes})
        for index, batch_size in enumerate(batch_sizes):
            # Recompute every coordinate against the growing extent. This
            # deliberately exercises continual coordinate updates.
            yield {
                "kind": "snapshot",
                "roots": self.roots,
                "panel": self.panel(batch_size),
                "projection": self.projection(batch_size),
                "complete": index + 1 == len(batch_sizes),
            }


class FixtureRequestHandler(BaseHTTPRequestHandler):
    server: FixtureHttpServer
    protocol_version = "HTTP/1.0"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json({"status": "ok"})
        elif parsed.path == "/catalog/summary":
            self._json(self.server.fixture.catalog_summary())
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == "/projection":
            fixture = self.server.fixture
            count = min(int(body.get("limit") or fixture.scenario.point_count), fixture.scenario.point_count)
            axes = _submitted_axes(body.get("control_values") or {})
            self._json(
                {
                    "panel": fixture.panel(count, axes),
                    "projection": fixture.projection(count, axes),
                }
            )
            return
        if self.path == "/catalog/reload":
            self._json(self.server.fixture.catalog_summary())
            return
        if self.path != "/catalog/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for index, snapshot in enumerate(self.server.fixture.stream_snapshots()):
                self.wfile.write(json.dumps(snapshot, separators=(",", ":")).encode())
                self.wfile.write(b"\n")
                self.wfile.flush()
                # The first point arrives nearly immediately; later snapshots
                # overlap rendering and media decoding.
                time.sleep(0.08 if index == 0 else 0.30)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _json(self, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FixtureHttpServer(ThreadingHTTPServer):
    fixture: CatalogFixture


def _submitted_axes(control_values: dict[str, Any]) -> list[int]:
    """Return the axis indices the viewer's control values name."""
    axes = []
    for control_id, default in zip(("x", "y", "z"), (0, 1, 2), strict=True):
        raw = control_values.get(control_id, default)
        try:
            axes.append(int(raw))
        except (TypeError, ValueError):
            axes.append(default)
    return axes


class FixtureServer:
    def __init__(self, fixture: CatalogFixture) -> None:
        self.httpd = FixtureHttpServer(("127.0.0.1", 0), FixtureRequestHandler)
        self.httpd.fixture = fixture
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def write_png(path: Path, side: int = 1024) -> None:
    # The default is large enough to exercise the full/high LOD upload path
    # rather than only measuring tiny placeholder textures.
    width = height = side
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(((x * 3) % 256, (y * 3) % 256, ((x + y) * 2) % 256))

    def chunk(name: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), level=6))
        + chunk(b"IEND", b"")
    )


def write_video(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required for the video performance scenarios")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=512x512:rate=12",
        "-t",
        "3",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(path),
    ]
    subprocess.run(command, check=True)


def executable_name() -> str:
    return "spatial_viewer.exe" if os.name == "nt" else "spatial_viewer"


def resolve_viewer(args: argparse.Namespace) -> Path:
    if args.viewer:
        viewer = args.viewer.resolve()
    else:
        viewer = VIEWER_ROOT / "target" / "release" / executable_name()
        if not args.no_build:
            subprocess.run(
                ["cargo", "build", "--release", "-p", "spatial_viewer"],
                cwd=VIEWER_ROOT,
                check=True,
            )
    if not viewer.is_file():
        raise SystemExit(f"viewer executable not found: {viewer}")
    return viewer


def prepare_scenario(
    scenario: Scenario,
    artifact_dir: Path,
    viewer: Path,
    duration: float,
    seed: int,
    image_paths: dict[int, Path],
    video_path: Path,
    extra_environment: dict[str, str],
) -> RunningScenario:
    scenario_dir = artifact_dir / scenario.name
    scenario_dir.mkdir(parents=True)
    fixture_dir = scenario_dir / "catalog"
    fixture_dir.mkdir()
    fixture = CatalogFixture(scenario, fixture_dir, image_paths, video_path)
    server = FixtureServer(fixture)
    server.start()

    viewer_metrics = scenario_dir / "viewer.jsonl"
    process_metrics = scenario_dir / "process.jsonl"
    config_path = scenario_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "scenario": scenario.name,
                "duration_seconds": duration,
                "seed": seed,
                "output_path": str(viewer_metrics.resolve()),
                "expected_points": scenario.point_count,
                "expect_video": scenario.expected_video,
                "heartbeat_seconds": 1.0,
                "assertion_grace_seconds": max(5.0, duration * 0.04),
                "warmup_seconds": WARMUP_SECONDS,
                "timeline": scenario.timeline,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stdout_file = (scenario_dir / "stdout.log").open("wb")
    stderr_file = (scenario_dir / "stderr.log").open("wb")
    environment = os.environ.copy()
    environment.update(extra_environment)
    environment["SPATIAL_VIEWER_PERF_CONFIG"] = str(config_path.resolve())
    environment["SPATIAL_VIEWER_PERF_ROOTS"] = json.dumps(fixture.roots)
    command = [
        str(viewer),
        "--api",
        server.url,
        "--limit",
        str(scenario.point_count),
        # The resident-texture cap is a VRAM budget, not an image count;
        # a single-point scenario gets the floor so eviction still engages.
        "--texture-budget-mib",
        "2048" if scenario.point_count > 1 else "512",
        "--image-concurrency",
        "8",
        "--max-texture-side",
        "1024",
    ]
    process = subprocess.Popen(
        command,
        cwd=VIEWER_ROOT,
        env=environment,
        stdout=stdout_file,
        stderr=stderr_file,
    )
    ps_process = psutil.Process(process.pid)
    ps_process.cpu_percent(None)
    return RunningScenario(
        scenario=scenario,
        server=server,
        process=process,
        ps_process=ps_process,
        config_path=config_path,
        viewer_metrics_path=viewer_metrics,
        process_metrics_path=process_metrics,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
    )


def poll_viewer_heartbeats(run: RunningScenario) -> None:
    if not run.viewer_metrics_path.exists():
        return
    with run.viewer_metrics_path.open("rb") as stream:
        stream.seek(run.viewer_metrics_offset)
        for raw_line in stream:
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") in {"start", "heartbeat", "complete"}:
                run.last_heartbeat_wall = time.monotonic()
        run.viewer_metrics_offset = stream.tell()


def process_sample(run: RunningScenario) -> dict[str, Any] | None:
    try:
        processes = [run.ps_process, *run.ps_process.children(recursive=True)]
        rss = vms = threads = 0
        cpu = 0.0
        live_children = 0
        for index, process in enumerate(processes):
            try:
                memory = process.memory_info()
                rss += memory.rss
                vms += memory.vms
                threads += process.num_threads()
                cpu += process.cpu_percent(None)
                if index:
                    live_children += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {
            "type": "process",
            "scenario": run.scenario.name,
            "elapsed_seconds": time.monotonic() - run.started,
            "cpu_percent": cpu,
            "rss_bytes": rss,
            "vms_bytes": vms,
            "thread_count": threads,
            "child_process_count": live_children,
        }
    except psutil.NoSuchProcess:
        return None


def terminate_tree(run: RunningScenario) -> None:
    try:
        descendants = run.ps_process.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for process in descendants:
        with suppress(psutil.NoSuchProcess):
            process.terminate()
    if run.process.poll() is None:
        run.process.terminate()
    _, alive = psutil.wait_procs(descendants, timeout=3)
    for process in alive:
        with suppress(psutil.NoSuchProcess):
            process.kill()
    try:
        run.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        run.process.kill()
        run.process.wait(timeout=5)


def supervise(runs: list[RunningScenario], duration: float, watchdog_seconds: float) -> None:
    process_streams = {run.scenario.name: run.process_metrics_path.open("w", encoding="utf-8") for run in runs}
    deadline = time.monotonic() + duration + 90.0
    try:
        while any(run.process.poll() is None for run in runs):
            now = time.monotonic()
            for run in runs:
                if run.process.poll() is not None:
                    continue
                poll_viewer_heartbeats(run)
                sample = process_sample(run)
                if sample is not None:
                    stream = process_streams[run.scenario.name]
                    stream.write(json.dumps(sample, separators=(",", ":")) + "\n")
                    stream.flush()
                startup_grace = min(45.0, max(15.0, duration * 0.2))
                if now - run.started > startup_grace and now - run.last_heartbeat_wall > watchdog_seconds:
                    run.frozen = True
                    run.forced_stop = True
                    terminate_tree(run)
            if now > deadline:
                for run in runs:
                    if run.process.poll() is None:
                        run.forced_stop = True
                        terminate_tree(run)
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        for run in runs:
            if run.process.poll() is None:
                run.forced_stop = True
                terminate_tree(run)
        raise
    finally:
        for stream in process_streams.values():
            stream.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize(run: RunningScenario) -> dict[str, Any]:
    viewer_events = read_jsonl(run.viewer_metrics_path)
    process_events = read_jsonl(run.process_metrics_path)
    complete = next(
        (event for event in reversed(viewer_events) if event.get("type") == "complete"),
        None,
    )
    action_failures = [
        event for event in viewer_events if event.get("type") == "action" and event.get("phase") == "failed"
    ]
    rss_values = [float(event["rss_bytes"]) for event in process_events]
    cpu_values = [float(event["cpu_percent"]) for event in process_events]
    child_values = [int(event["child_process_count"]) for event in process_events]
    steady_state = complete.get("steady_state", {}) if complete else {}
    return {
        "scenario": run.scenario.name,
        "exit_code": run.process.returncode,
        "frozen": run.frozen,
        "forced_stop": run.forced_stop,
        "completed": complete is not None,
        "action_failures": action_failures,
        "frame_count": complete.get("frame_count", 0) if complete else 0,
        "fps": complete.get("fps", 0.0) if complete else 0.0,
        "frame_time_ms": complete.get("frame_time_ms", {}) if complete else {},
        # Uniformity is judged after warmup: the whole-run numbers above are
        # dominated by the first frames' pipeline compilation.
        "steady_state": {
            "sample_count": steady_state.get("sample_count", 0),
            "fps": steady_state.get("fps", 0.0),
            "frame_time_ms": steady_state.get("frame_time_ms", {}),
            "worst_frames": steady_state.get("worst_frames", []),
        },
        "process": {
            "cpu_percent_mean": statistics.fmean(cpu_values) if cpu_values else 0.0,
            "cpu_percent_p95": percentile(cpu_values, 0.95),
            "cpu_percent_max": max(cpu_values, default=0.0),
            "rss_bytes_mean": statistics.fmean(rss_values) if rss_values else 0.0,
            "rss_bytes_p95": percentile(rss_values, 0.95),
            "rss_bytes_max": max(rss_values, default=0.0),
            "child_processes_max": max(child_values, default=0),
        },
    }


def baseline_payload(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "scenarios": {
            summary["scenario"]: {
                "fps": summary["fps"],
                "frame_time_p95_ms": summary["frame_time_ms"].get("p95", 0.0),
                "rss_bytes_max": summary["process"]["rss_bytes_max"],
                "cpu_percent_p95": summary["process"]["cpu_percent_p95"],
                "steady_state": {
                    "p99_ms": summary["steady_state"]["frame_time_ms"].get("p99", 0.0),
                    "coefficient_of_variation": summary["steady_state"]["frame_time_ms"].get(
                        "coefficient_of_variation", 0.0
                    ),
                    "spike_fraction": summary["steady_state"]["frame_time_ms"].get("spike_fraction", 0.0),
                    "longest_spike_burst_ms": summary["steady_state"]["frame_time_ms"].get("longest_spike_burst", 0.0),
                },
            }
            for summary in summaries
        },
    }


def compare_baseline(summaries: list[dict[str, Any]], baseline_path: Path) -> list[str]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for current in summaries:
        name = current["scenario"]
        previous = baseline.get("scenarios", {}).get(name)
        if not previous:
            failures.append(f"{name}: scenario missing from baseline")
            continue
        current_fps = float(current["fps"])
        minimum_fps = float(previous["fps"]) * 0.80 - 2.0
        if current_fps < minimum_fps:
            failures.append(f"{name}: FPS {current_fps:.2f} below tolerated {minimum_fps:.2f}")
        current_p95 = float(current["frame_time_ms"].get("p95", 0.0))
        maximum_p95 = float(previous["frame_time_p95_ms"]) * 1.30 + 2.0
        if current_p95 > maximum_p95:
            failures.append(f"{name}: frame-time p95 {current_p95:.2f}ms above tolerated {maximum_p95:.2f}ms")
        current_rss = float(current["process"]["rss_bytes_max"])
        maximum_rss = float(previous["rss_bytes_max"]) * 1.25 + 64 * MIB
        if current_rss > maximum_rss:
            failures.append(f"{name}: peak RSS {current_rss / MIB:.1f}MiB above tolerated {maximum_rss / MIB:.1f}MiB")
        failures.extend(
            compare_uniformity(name, current["steady_state"]["frame_time_ms"], previous.get("steady_state"))
        )
    return failures


def compare_uniformity(name: str, current: dict[str, Any], previous: dict[str, Any] | None) -> list[str]:
    """Flag steady-state stutter regressions against a baseline.

    Percentiles and FPS barely move when a few frames double, so this
    compares the spread (coefficient of variation), how often frames spike,
    the longest run of spikes, and the p99 tail. Absolute floors keep tiny
    baselines from failing on measurement noise.
    """
    if previous is None:
        return [f"{name}: baseline predates steady-state uniformity (schema_version 1); rewrite it"]
    failures = []
    checks = (
        ("coefficient_of_variation", "coefficient_of_variation", 1.30, 0.05, "coefficient of variation", ""),
        ("spike_fraction", "spike_fraction", 1.50, 0.005, "spike fraction", ""),
        ("longest_spike_burst", "longest_spike_burst_ms", 1.30, 8.0, "longest spike burst", "ms"),
        ("p99", "p99_ms", 1.30, 2.0, "steady-state p99", "ms"),
    )
    for current_key, previous_key, ratio, floor, label, unit in checks:
        current_value = float(current.get(current_key, 0.0))
        maximum = float(previous.get(previous_key, 0.0)) * ratio + floor
        if current_value > maximum:
            failures.append(f"{name}: {label} {current_value:.3f}{unit} above tolerated {maximum:.3f}{unit}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0x5EED)
    parser.add_argument("--massive-points", type=int, default=50_000)
    parser.add_argument("--patrol-points", type=int, default=2_000)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Only run the named scenarios (default: all).",
    )
    parser.add_argument("--watchdog-seconds", type=float, default=12.0)
    parser.add_argument("--viewer", type=Path)
    parser.add_argument(
        "--viewer-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Extra environment variable for the viewer processes (repeatable).",
    )
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--write-baseline", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration < 10:
        raise SystemExit("--duration must be at least 10 seconds")
    if args.massive_points < 1_000:
        raise SystemExit("--massive-points must be at least 1000")
    if args.patrol_points < 100:
        raise SystemExit("--patrol-points must be at least 100")
    viewer = resolve_viewer(args)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = (
        args.artifacts.resolve()
        if args.artifacts
        else (DEFAULT_ARTIFACT_ROOT / f"{timestamp}-seed-{args.seed}").resolve()
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    fixture_media = artifact_dir / "fixture-media"
    fixture_media.mkdir()
    video_path = fixture_media / "video.mp4"
    write_video(video_path)

    scenarios = [
        Scenario("single-image", 1, ("image",), False),
        Scenario("single-video", 1, ("video",), True),
        Scenario(
            "massive-mixed",
            args.massive_points,
            ("video", "image", "image", "image"),
            True,
        ),
        # A continuous sweep through images of every size so loading, quality
        # refresh, multi-tile upload and eviction all happen throughout the
        # run: the synthetic frame-time uniformity workload.
        Scenario(
            "mixed-sizes-patrol",
            args.patrol_points,
            ("image",),
            False,
            timeline="patrol",
            image_sides=MIXED_IMAGE_SIDES,
        ),
    ]
    if args.scenarios:
        known = {scenario.name for scenario in scenarios}
        unknown = set(args.scenarios) - known
        if unknown:
            raise SystemExit(f"unknown scenarios {sorted(unknown)}; choose from {sorted(known)}")
        scenarios = [scenario for scenario in scenarios if scenario.name in args.scenarios]
    image_paths: dict[int, Path] = {}
    for side in sorted({side for scenario in scenarios for side in scenario.image_sides}):
        image_paths[side] = fixture_media / f"image-{side}.png"
        write_png(image_paths[side], side)
    extra_environment = dict(entry.split("=", 1) for entry in args.viewer_env)
    runs: list[RunningScenario] = []
    try:
        for index, scenario in enumerate(scenarios):
            runs.append(
                prepare_scenario(
                    scenario,
                    artifact_dir,
                    viewer,
                    args.duration,
                    args.seed + index,
                    image_paths,
                    video_path,
                    extra_environment,
                )
            )
        supervise(runs, args.duration, args.watchdog_seconds)
    finally:
        for run in runs:
            if run.process.poll() is None:
                terminate_tree(run)
            run.server.close()
            run.stdout_file.close()
            run.stderr_file.close()

    summaries = [summarize(run) for run in runs]
    failures = []
    for summary in summaries:
        if summary["exit_code"] != 0:
            failures.append(f"{summary['scenario']}: viewer exit code {summary['exit_code']}")
        if summary["frozen"]:
            failures.append(f"{summary['scenario']}: heartbeat watchdog detected a freeze")
        if not summary["completed"]:
            failures.append(f"{summary['scenario']}: missing completion event")
        if summary["action_failures"]:
            failures.append(f"{summary['scenario']}: {len(summary['action_failures'])} action assertion(s) failed")
    if args.baseline:
        failures.extend(compare_baseline(summaries, args.baseline.resolve()))

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "duration_seconds": args.duration,
        "seed": args.seed,
        "viewer": str(viewer),
        "summaries": summaries,
        "failures": failures,
    }
    report_path = artifact_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.write_baseline and not failures:
        baseline_path = args.write_baseline.resolve()
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline_payload(summaries), indent=2), encoding="utf-8")

    print(json.dumps({"artifacts": str(artifact_dir), "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
