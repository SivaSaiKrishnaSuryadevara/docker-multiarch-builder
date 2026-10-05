#!/usr/bin/env python3
"""Benchmark multi-architecture Docker Buildx builds: QEMU emulation vs native nodes.

Creates a throwaway Buildx builder (one local node, or several platform-pinned
nodes joined with `docker buildx create --append`), runs `docker buildx build
--progress=plain`, and records per run:

  * wall-clock time,
  * per-platform and per-stage step timings parsed from BuildKit's plain log,
  * cache reuse (CACHED vs executed Dockerfile steps),
  * CPU and memory of each BuildKit node container, sampled with `docker stats`,
  * whether each platform ran natively or under emulation,
  * a failure category when the build breaks (QEMU segfault, missing binfmt,
    OOM kill, network, generic).

The builder is always removed afterwards — on success, on error, on Ctrl-C,
and on SIGTERM (what CI sends when a job is cancelled) — unless
--keep-builder is passed.

Subcommands:
  run      run a benchmark and write a JSON report
  compare  render several JSON reports as a Markdown table

Exit codes: 0 all runs built, 1 a build failed, 2 usage/environment error,
130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel, Field

log = logging.getLogger("benchmark")

DEFAULT_PLATFORMS = ["linux/amd64", "linux/arm64"]
PLATFORM_RE = re.compile(r"^linux/(amd64|arm64|arm/v[5-7]|386|ppc64le|s390x|riscv64)$")
ENDPOINT_RE = re.compile(r"^(?:(?:ssh|tcp|unix)://\S+|[A-Za-z0-9][A-Za-z0-9_.-]*)$")
ARCH_ALIASES = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class BenchmarkError(Exception):
    """Usage or environment problem (exit code 2)."""


class Terminated(BaseException):
    """Raised from the SIGTERM handler so `finally` blocks still run."""


# =============================================================================
# Models
# =============================================================================

class NodeSpec(BaseModel):
    name: str
    platform: str
    endpoint: str | None = None  # None = the current Docker context

    def docker_target_args(self) -> list[str]:
        """Global docker CLI flags that address this node's daemon."""
        if self.endpoint is None:
            return []
        if "://" in self.endpoint:
            return ["--host", self.endpoint]
        return ["--context", self.endpoint]


class StepTiming(BaseModel):
    id: int
    platform: str | None
    stage: str | None
    position: str
    instruction: str
    status: str  # done | cached | error | incomplete
    seconds: float | None = None


class PlatformSummary(BaseModel):
    platform: str
    execution: str  # native | emulated | unknown
    steps_executed: int
    steps_cached: int
    cache_hit_ratio: float
    step_seconds_total: float
    slowest_step: str | None = None
    slowest_step_seconds: float | None = None
    stage_seconds: dict[str, float] = Field(default_factory=dict)


class NodeStats(BaseModel):
    node: str
    samples: int = 0
    cpu_peak_pct: float = 0.0
    cpu_mean_pct: float = 0.0
    mem_peak_bytes: int = 0
    ncpu: int | None = None

    @property
    def saturation(self) -> float | None:
        """Mean CPU as a fraction of the node's cores (docker stats reports 100% per core)."""
        if not self.ncpu or not self.samples:
            return None
        return round(self.cpu_mean_pct / (100 * self.ncpu), 3)


class RunResult(BaseModel):
    index: int
    cold: bool
    success: bool
    exit_code: int
    wall_seconds: float
    failure_category: str | None = None
    failure_message: str | None = None
    platforms: list[PlatformSummary]
    nodes: list[NodeStats]


class NodeInfo(BaseModel):
    name: str
    endpoint: str | None
    pinned_platform: str | None
    arch: str | None
    ncpu: int | None
    mem_total_bytes: int | None


class BenchmarkReport(BaseModel):
    label: str
    builder: str
    dockerfile: str
    platforms: list[str]
    execution: dict[str, str]
    nodes: list[NodeInfo]
    runs: list[RunResult]

    @property
    def success(self) -> bool:
        return all(r.success for r in self.runs)


# =============================================================================
# Argument validation and command builders
# =============================================================================

def validate_platform(platform: str) -> str:
    if not PLATFORM_RE.match(platform):
        raise BenchmarkError(f"Unsupported platform '{platform}' (expected e.g. linux/amd64, linux/arm64)")
    return platform


def parse_node(spec: str, index: int, builder: str) -> NodeSpec:
    """Parse --node PLATFORM[=ENDPOINT], e.g. linux/arm64=ssh://ci@graviton-1."""
    platform, sep, endpoint = spec.partition("=")
    validate_platform(platform)
    if sep and not endpoint:
        raise BenchmarkError(f"--node {spec!r}: empty endpoint after '='")
    if endpoint and (endpoint.startswith("-") or not ENDPOINT_RE.match(endpoint)):
        raise BenchmarkError(f"--node {spec!r}: endpoint must be ssh://, tcp://, unix:// or a docker context name")
    return NodeSpec(name=f"{builder}-{index}", platform=platform, endpoint=endpoint or None)


def create_commands(builder: str, nodes: list[NodeSpec]) -> list[list[str]]:
    """`docker buildx create` invocations: the first node creates, the rest --append."""
    if not nodes:
        return [["docker", "buildx", "create", "--name", builder, "--driver", "docker-container",
                 "--node", f"{builder}-0"]]
    cmds = []
    for i, node in enumerate(nodes):
        cmd = ["docker", "buildx", "create", "--name", builder, "--driver", "docker-container",
               "--node", node.name, "--platform", node.platform]
        if i > 0:
            cmd.append("--append")
        if node.endpoint:
            cmd.append(node.endpoint)
        cmds.append(cmd)
    return cmds


def validate_output(platforms: list[str], output: str, tag: str | None) -> None:
    if output == "load" and len(platforms) > 1:
        raise BenchmarkError(
            "--output load cannot be used with more than one platform: the default Docker image store "
            "holds one image per tag, not a multi-platform manifest list. Use --output cacheonly "
            "(the default) or --output push with --tag."
        )
    if output == "push" and not tag:
        raise BenchmarkError("--output push requires --tag")


def build_command(builder: str, dockerfile: Path, context: Path, platforms: list[str], *,
                  no_cache: bool, output: str = "cacheonly", tag: str | None = None,
                  build_args: list[str] | None = None) -> list[str]:
    validate_output(platforms, output, tag)
    cmd = ["docker", "buildx", "build", "--builder", builder, "--progress=plain",
           "-f", str(dockerfile), "--platform", ",".join(platforms)]
    if no_cache:
        cmd.append("--no-cache")
    if output == "cacheonly":
        cmd += ["--output", "type=cacheonly"]
    elif output == "load":
        cmd.append("--load")
    elif output == "push":
        cmd.append("--push")
    if tag:
        cmd += ["--tag", tag]
    for arg in build_args or []:
        cmd += ["--build-arg", arg]
    cmd.append(str(context))
    return cmd


# =============================================================================
# BuildKit plain-log parsing
# =============================================================================

HEADER_RE = re.compile(
    r"^#(?P<id>\d+) \[(?:(?P<platform>linux/[a-z0-9/]+) )?(?:(?P<stage>[A-Za-z0-9_.-]+) )?"
    r"(?P<pos>\d+/\d+)\] (?P<instr>.+)$"
)
DONE_RE = re.compile(r"^#(?P<id>\d+) DONE (?P<secs>\d+(?:\.\d+)?)s$")
CACHED_RE = re.compile(r"^#(?P<id>\d+) CACHED$")
STEP_ERROR_RE = re.compile(r"^#(?P<id>\d+) ERROR: (?P<msg>.*)$")


def parse_steps(log_text: str, platforms: list[str]) -> list[StepTiming]:
    """Extract Dockerfile-instruction steps (internal steps like 'load metadata' are skipped)."""
    default_platform = platforms[0] if len(platforms) == 1 else None
    steps: dict[int, StepTiming] = {}
    for raw in log_text.splitlines():
        line = ANSI_RE.sub("", raw).rstrip("\r")
        if m := HEADER_RE.match(line):
            sid = int(m.group("id"))
            if sid not in steps:
                steps[sid] = StepTiming(
                    id=sid, platform=m.group("platform") or default_platform, stage=m.group("stage"),
                    position=m.group("pos"), instruction=m.group("instr"), status="incomplete",
                )
        elif m := DONE_RE.match(line):
            if (step := steps.get(int(m.group("id")))) and step.status != "cached":
                step.status, step.seconds = "done", float(m.group("secs"))
        elif m := CACHED_RE.match(line):
            if step := steps.get(int(m.group("id"))):
                step.status, step.seconds = "cached", 0.0
        elif m := STEP_ERROR_RE.match(line):
            if step := steps.get(int(m.group("id"))):
                step.status = "error"
    return sorted(steps.values(), key=lambda s: s.id)


def summarize(steps: list[StepTiming], platform: str, execution: str) -> PlatformSummary:
    mine = [s for s in steps if s.platform == platform]
    done = [s for s in mine if s.status == "done"]
    cached = [s for s in mine if s.status == "cached"]
    counted = len(done) + len(cached)
    stage_seconds: dict[str, float] = {}
    for s in done:
        key = s.stage or "?"
        stage_seconds[key] = round(stage_seconds.get(key, 0.0) + (s.seconds or 0.0), 1)
    slowest = max(done, key=lambda s: s.seconds or 0.0, default=None)
    return PlatformSummary(
        platform=platform,
        execution=execution,
        steps_executed=len(done),
        steps_cached=len(cached),
        cache_hit_ratio=round(len(cached) / counted, 3) if counted else 0.0,
        step_seconds_total=round(sum(s.seconds or 0.0 for s in done), 1),
        slowest_step=f"[{s.stage} {s.position}] {s.instruction[:80]}" if (s := slowest) else None,
        slowest_step_seconds=slowest.seconds if slowest else None,
        stage_seconds=stage_seconds,
    )


FAILURE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("qemu_segfault", re.compile(r"qemu: uncaught target signal \d+|qemu-[\w-]+: .*(?:Segmentation fault|core dumped)", re.I)),
    ("binfmt_missing", re.compile(r"exec format error", re.I)),
    ("oom_killed", re.compile(r"signal: killed|exit code: 137|out of memory|Killed\s*$", re.I | re.M)),
    ("network", re.compile(r"Temporary failure resolving|Could not resolve host|TLS handshake timeout"
                           r"|connection reset by peer|i/o timeout", re.I)),
    ("build_failed", re.compile(r"did not complete successfully|ERROR: failed to", re.I)),
]


def classify_failure(log_text: str) -> tuple[str, str]:
    """Return (category, the log line that matched). Specific causes win over 'build_failed'."""
    clean = ANSI_RE.sub("", log_text)
    for category, pattern in FAILURE_PATTERNS:
        if m := pattern.search(clean):
            line_start = clean.rfind("\n", 0, m.start()) + 1
            line_end = clean.find("\n", m.end())
            return category, clean[line_start:line_end if line_end != -1 else None].strip()[:300]
    return "unknown", ""


# =============================================================================
# Resource sampling
# =============================================================================

UNITS = {"b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4,
         "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4}
MEM_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[KMGT]?i?B)\s*$", re.I)


def parse_mem(value: str) -> int:
    m = MEM_RE.match(value)
    if not m:
        raise ValueError(f"unparseable memory value: {value!r}")
    return int(float(m.group("num")) * UNITS[m.group("unit").lower()])


def parse_stats_line(line: str) -> tuple[str, float, int] | None:
    """One `docker stats --format '{{json .}}'` line -> (container, cpu %, mem bytes)."""
    try:
        row = json.loads(line)
        cpu = float(row["CPUPerc"].rstrip("%"))
        mem = parse_mem(row["MemUsage"].split("/")[0])
        return row["Name"], cpu, mem
    except (ValueError, KeyError, AttributeError):
        return None


class Runner(Protocol):
    def __call__(self, cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess: ...


def run_cmd(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class StatsSampler(threading.Thread):
    """Polls `docker stats` for each node's BuildKit container until stopped."""

    def __init__(self, nodes: list[NodeSpec], ncpu: dict[str, int | None], runner: Runner, interval: float = 2.0):
        super().__init__(daemon=True)
        self.nodes, self.runner, self.interval = nodes, runner, interval
        self.stats = {n.name: NodeStats(node=n.name, ncpu=ncpu.get(n.name)) for n in nodes}
        self._cpu_sum = {n.name: 0.0 for n in nodes}
        self._stop = threading.Event()

    def sample_once(self) -> None:
        for node in self.nodes:
            container = f"buildx_buildkit_{node.name}"
            try:
                proc = self.runner(["docker", *node.docker_target_args(), "stats", "--no-stream",
                                    "--format", "{{json .}}", container], timeout=15)
            except subprocess.TimeoutExpired:
                continue
            for line in proc.stdout.splitlines():
                parsed = parse_stats_line(line)
                if not parsed or parsed[0] != container:
                    continue
                _, cpu, mem = parsed
                st = self.stats[node.name]
                st.samples += 1
                self._cpu_sum[node.name] += cpu
                st.cpu_peak_pct = max(st.cpu_peak_pct, cpu)
                st.cpu_mean_pct = round(self._cpu_sum[node.name] / st.samples, 1)
                st.mem_peak_bytes = max(st.mem_peak_bytes, mem)

    def run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval)

    def stop(self) -> list[NodeStats]:
        self._stop.set()
        if self.is_alive():
            self.join(timeout=20)
        return list(self.stats.values())


# =============================================================================
# Builder lifecycle
# =============================================================================

class BuilderSession:
    """Creates the builder on enter and always removes it on exit.

    SIGTERM is converted into an exception for the duration of the session so
    a cancelled CI job still tears down BuildKit containers on remote nodes.
    """

    def __init__(self, builder: str, nodes: list[NodeSpec], runner: Runner, keep: bool = False):
        self.builder, self.nodes, self.runner, self.keep = builder, nodes, runner, keep
        self.created = False
        self._prev_sigterm: Callable | int | None = None

    def __enter__(self) -> BuilderSession:
        if threading.current_thread() is threading.main_thread():
            self._prev_sigterm = signal.signal(signal.SIGTERM, self._on_sigterm)
        try:
            for cmd in create_commands(self.builder, self.nodes):
                self.created = True  # a failed --append can still leave earlier nodes behind
                proc = self.runner(cmd, timeout=120)
                if proc.returncode != 0:
                    raise BenchmarkError(f"{' '.join(cmd[:4])} failed: {proc.stderr.strip()[:300]}")
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    @staticmethod
    def _on_sigterm(signum, frame):
        raise Terminated()

    def platforms(self) -> set[str]:
        proc = self.runner(["docker", "buildx", "inspect", "--bootstrap", self.builder], timeout=600)
        if proc.returncode != 0:
            raise BenchmarkError(f"builder bootstrap failed: {proc.stderr.strip()[:300]}")
        return parse_inspect_platforms(proc.stdout)

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.created and not self.keep:
                proc = self.runner(["docker", "buildx", "rm", self.builder], timeout=120)
                if proc.returncode != 0:
                    log.warning("could not remove builder %s: %s", self.builder, proc.stderr.strip()[:200])
                else:
                    log.info("removed builder %s", self.builder)
        finally:
            if self._prev_sigterm is not None:
                signal.signal(signal.SIGTERM, self._prev_sigterm)
                self._prev_sigterm = None


def parse_inspect_platforms(text: str) -> set[str]:
    """Union of platforms from `docker buildx inspect`; '*' marks user-pinned platforms."""
    found: set[str] = set()
    for line in text.splitlines():
        if line.strip().startswith("Platforms:"):
            for p in line.split(":", 1)[1].split(","):
                if p := p.strip().rstrip("*").strip():
                    found.add(p)
    return found


def node_info(node: NodeSpec, runner: Runner) -> NodeInfo:
    proc = runner(["docker", *node.docker_target_args(), "info", "--format",
                   "{{.Architecture}}|{{.NCPU}}|{{.MemTotal}}"], timeout=30)
    arch = ncpu = mem = None
    if proc.returncode == 0 and proc.stdout.count("|") == 2:
        a, c, m = proc.stdout.strip().split("|")
        arch = ARCH_ALIASES.get(a, a)
        ncpu = int(c) if c.isdigit() else None
        mem = int(m) if m.isdigit() else None
    pinned = node.platform if node.platform != "*" else None
    return NodeInfo(name=node.name, endpoint=node.endpoint, pinned_platform=pinned, arch=arch, ncpu=ncpu,
                    mem_total_bytes=mem)


def execution_modes(platforms: list[str], infos: list[NodeInfo]) -> dict[str, str]:
    """native if the node that serves a platform has the same CPU architecture, else emulated."""
    modes = {}
    for platform in platforms:
        target_arch = platform.split("/")[1]
        pinned = [i for i in infos if i.pinned_platform == platform]
        serving = pinned[0] if pinned else (infos[0] if infos else None)
        if serving is None or serving.arch is None:
            modes[platform] = "unknown"
        else:
            modes[platform] = "native" if serving.arch == target_arch else "emulated"
    return modes


# =============================================================================
# Orchestration
# =============================================================================

def run_benchmark(args: argparse.Namespace, runner: Runner = run_cmd) -> BenchmarkReport:
    platforms = [validate_platform(p) for p in (args.platform or DEFAULT_PLATFORMS)]
    validate_output(platforms, args.output, args.tag)  # reject --load before creating anything
    dockerfile, context = Path(args.file), Path(args.context)
    if not dockerfile.is_file():
        raise BenchmarkError(f"Dockerfile not found: {dockerfile}")
    if runner(["docker", "buildx", "version"], timeout=30).returncode != 0:
        raise BenchmarkError("docker buildx is not available")

    builder = args.builder or f"mab-{uuid.uuid4().hex[:8]}"
    nodes = [parse_node(spec, i, builder) for i, spec in enumerate(args.node or [])]
    sample_nodes = nodes or [NodeSpec(name=f"{builder}-0", platform="*")]

    with BuilderSession(builder, nodes, runner, keep=args.keep_builder) as session:
        available = session.platforms()
        missing = [p for p in platforms if p not in available]
        if missing:
            raise BenchmarkError(
                f"builder {builder} cannot build {', '.join(missing)} (available: {', '.join(sorted(available))}). "
                "For emulation, register QEMU handlers: docker run --privileged --rm tonistiigi/binfmt --install all"
            )
        infos = [node_info(n, runner) for n in sample_nodes]
        modes = execution_modes(platforms, infos)
        log.info("builder %s ready; execution: %s", builder, ", ".join(f"{p}={m}" for p, m in modes.items()))

        runs: list[RunResult] = []
        for i in range(args.runs):
            cold = i == 0 and not args.warm_only
            cmd = build_command(builder, dockerfile, context, platforms, no_cache=cold,
                                output=args.output, tag=args.tag, build_args=args.build_arg)
            log.info("run %d/%d (%s) started", i + 1, args.runs, "cold" if cold else "warm")
            sampler = StatsSampler(sample_nodes, {n.name: n.ncpu for n in infos}, runner, args.sample_interval)
            sampler.start()
            started = time.monotonic()
            try:
                proc = runner(cmd, timeout=args.timeout)
                exit_code, output = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
            except subprocess.TimeoutExpired as exc:
                exit_code = 124
                output = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                output += (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            finally:
                wall = round(time.monotonic() - started, 1)
                node_stats = sampler.stop()

            steps = parse_steps(output, platforms)
            category = message = None
            if exit_code == 124:
                category, message = "timeout", f"build exceeded --timeout {args.timeout}s"
            elif exit_code != 0:
                category, message = classify_failure(output)
            runs.append(RunResult(
                index=i, cold=cold, success=exit_code == 0, exit_code=exit_code, wall_seconds=wall,
                failure_category=category, failure_message=message,
                platforms=[summarize(steps, p, modes[p]) for p in platforms], nodes=node_stats,
            ))
            log.info("run %d/%d %s in %.1fs", i + 1, args.runs, "passed" if exit_code == 0 else f"FAILED ({category})", wall)
            if args.save_logs:
                Path(args.save_logs).mkdir(parents=True, exist_ok=True)
                (Path(args.save_logs) / f"{args.label or builder}-run{i}.log").write_text(output)
            if exit_code != 0:
                break  # a failed cold build makes warm-cache numbers meaningless

    return BenchmarkReport(
        label=args.label or builder, builder=builder, dockerfile=str(dockerfile), platforms=platforms,
        execution=modes, nodes=infos, runs=runs,
    )


# =============================================================================
# Reporting
# =============================================================================

def _fmt_bytes(n: int) -> str:
    return f"{n / 1024**3:.2f} GiB" if n >= 1024**3 else f"{n / 1024**2:.0f} MiB"


def compare_markdown(reports: list[BenchmarkReport], baseline: str | None = None) -> str:
    base = next((r for r in reports if r.label == baseline), reports[0]) if reports else None
    base_cold = next((r.wall_seconds for r in base.runs if r.cold and r.success), None) if base else None
    rows = ["| Label | Platforms (execution) | Cold build | vs baseline | Warm build | Warm cache hits | "
            "CPU mean (saturation) | Peak memory | Result |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        cold = next((x for x in r.runs if x.cold), None)
        warm = next((x for x in r.runs if not x.cold), None)
        execs = ", ".join(f"{p.split('/')[1]} ({m})" for p, m in r.execution.items())
        cold_s = f"{cold.wall_seconds:.0f}s" if cold else "-"
        speed = "-"
        if cold and cold.success and base_cold:
            speed = "baseline" if r is base else f"{base_cold / cold.wall_seconds:.2f}x"
        warm_s = f"{warm.wall_seconds:.0f}s" if warm else "-"
        hits = "-"
        if warm and warm.platforms:
            cached = sum(p.steps_cached for p in warm.platforms)
            total = cached + sum(p.steps_executed for p in warm.platforms)
            hits = f"{cached}/{total}" if total else "-"
        ref = cold or warm
        cpu = mem = "-"
        if ref and ref.nodes and any(n.samples for n in ref.nodes):
            cpu = "; ".join(f"{n.cpu_mean_pct:.0f}%" + (f" ({n.saturation:.0%})" if n.saturation is not None else "")
                            for n in ref.nodes if n.samples)
            mem = "; ".join(_fmt_bytes(n.mem_peak_bytes) for n in ref.nodes if n.samples)
        result = "ok" if r.success else next((f"FAIL: {x.failure_category}" for x in r.runs if not x.success), "FAIL")
        rows.append(f"| {r.label} | {execs} | {cold_s} | {speed} | {warm_s} | {hits} | {cpu} | {mem} | {result} |")
    return "\n".join(rows)


def print_summary(report: BenchmarkReport) -> None:
    for run in report.runs:
        status = "ok" if run.success else f"FAIL [{run.failure_category}] {run.failure_message}"
        print(f"run {run.index} ({'cold' if run.cold else 'warm'}): {run.wall_seconds:.1f}s {status}")
        for p in run.platforms:
            print(f"  {p.platform} ({p.execution}): {p.steps_executed} executed, {p.steps_cached} cached, "
                  f"slowest {p.slowest_step_seconds or 0:.1f}s {p.slowest_step or ''}")
        for n in run.nodes:
            if n.samples:
                sat = f", saturation {n.saturation:.0%}" if n.saturation is not None else ""
                print(f"  node {n.node}: cpu mean {n.cpu_mean_pct:.0f}% peak {n.cpu_peak_pct:.0f}%{sat}, "
                      f"mem peak {_fmt_bytes(n.mem_peak_bytes)}")


# =============================================================================
# CLI
# =============================================================================

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark multi-arch Docker Buildx builds")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a benchmark")
    run.add_argument("-f", "--file", default="Dockerfile.benchmark")
    run.add_argument("context", nargs="?", default=".")
    run.add_argument("--platform", action="append", help="repeatable; default linux/amd64 + linux/arm64")
    run.add_argument("--node", action="append", metavar="PLATFORM[=ENDPOINT]",
                     help="add a platform-pinned node (repeatable); omit for one local node")
    run.add_argument("--runs", type=int, default=2, help="first run is cold (--no-cache), the rest warm")
    run.add_argument("--warm-only", action="store_true", help="do not force --no-cache on the first run")
    run.add_argument("--output", choices=["cacheonly", "load", "push"], default="cacheonly")
    run.add_argument("--tag")
    run.add_argument("--build-arg", action="append")
    run.add_argument("--builder", help="builder name (default: random mab-xxxxxxxx)")
    run.add_argument("--keep-builder", action="store_true")
    run.add_argument("--label")
    run.add_argument("--timeout", type=float, default=3600, help="per-build timeout in seconds")
    run.add_argument("--sample-interval", type=float, default=2.0)
    run.add_argument("--json", dest="json_out", help="write the report to this file")
    run.add_argument("--save-logs", metavar="DIR", help="keep each run's raw BuildKit log")

    cmp_ = sub.add_parser("compare", help="render JSON reports as a Markdown table")
    cmp_.add_argument("reports", nargs="+")
    cmp_.add_argument("--baseline", help="label to compute speedups against (default: first report)")

    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None, runner: Runner = run_cmd) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s", stream=sys.stderr)
    try:
        if args.command == "compare":
            reports = [BenchmarkReport.model_validate_json(Path(p).read_text()) for p in args.reports]
            print(compare_markdown(reports, args.baseline))
            return 0
        if args.runs < 1:
            raise BenchmarkError("--runs must be at least 1")
        report = run_benchmark(args, runner)
    except BenchmarkError as exc:
        log.error("%s", exc)
        return 2
    except (KeyboardInterrupt, Terminated):
        log.error("interrupted; builder removed")
        return 130

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(report.model_dump_json(indent=2))
    print_summary(report)
    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())
