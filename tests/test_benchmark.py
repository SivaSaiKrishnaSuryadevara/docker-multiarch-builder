"""Unit tests for benchmark.py. The docker CLI is never invoked: every command
goes through a scripted FakeRunner that records what would have run."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark as bm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Fakes and fixtures
# =============================================================================

def cp(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


INSPECT_LOCAL = """\
Name:          mab-test
Driver:        docker-container

Nodes:
Name:                  mab-test-0
Endpoint:              colima
Status:                running
Platforms:             linux/arm64, linux/amd64, linux/amd64/v2, linux/386
"""

INSPECT_HYBRID = """\
Name:          mab-test
Driver:        docker-container

Nodes:
Name:                  mab-test-0
Endpoint:              ssh://ci@graviton-1
Platforms:             linux/arm64*
Name:                  mab-test-1
Endpoint:              ssh://ci@x86-1
Platforms:             linux/amd64*, linux/amd64/v2, linux/386
"""

SINGLE_PLATFORM_LOG = """\
#0 building with "mab-test" instance using docker-container driver
#1 [internal] load build definition from Dockerfile.benchmark
#1 DONE 0.0s
#4 [internal] load metadata for docker.io/library/python:3.12-slim-bookworm
#4 DONE 1.2s
#5 [toolchain 1/3] FROM docker.io/library/python:3.12-slim-bookworm@sha256:abc
#5 DONE 3.1s
#6 [toolchain 2/3] RUN apt-get update  && apt-get install -y --no-install-recommends build-essential
#6 DONE 24.5s
#7 [toolchain 3/3] RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#7 DONE 18.0s
#8 [wheels 1/2] WORKDIR /wheels
#8 CACHED
#9 [wheels 2/2] RUN pip wheel --no-cache-dir --no-deps --no-binary cryptography,pydantic-core
#9 12.40 Building wheel for cryptography (pyproject.toml): started
#9 DONE 141.7s
#10 [runtime 2/2] COPY --from=wheels /wheels /wheels
#10 DONE 0.2s
"""

MULTI_PLATFORM_LOG = """\
#5 [linux/amd64 toolchain 1/3] FROM docker.io/library/python:3.12-slim-bookworm@sha256:abc
#5 DONE 2.0s
#6 [linux/arm64 toolchain 1/3] FROM docker.io/library/python:3.12-slim-bookworm@sha256:abc
#6 DONE 2.1s
#7 [linux/arm64 wheels 2/2] RUN pip wheel --no-binary cryptography,pydantic-core
#8 [linux/amd64 wheels 2/2] RUN pip wheel --no-binary cryptography,pydantic-core
#7 DONE 150.0s
#8 1203.4 Building wheel for pydantic-core (pyproject.toml): still running...
#8 DONE 1388.9s
#9 [linux/amd64 runtime 2/2] COPY --from=wheels /wheels /wheels
#9 CACHED
"""


class FakeRunner:
    """Answers docker commands from a table of handlers; records every call."""

    def __init__(self, build=None, inspect=INSPECT_LOCAL, arch="aarch64", fail=None):
        self.calls: list[list[str]] = []
        self.lock = threading.Lock()
        self.build = build or (lambda cmd: cp(0, "", SINGLE_PLATFORM_LOG))
        self.inspect, self.arch, self.fail = inspect, arch, fail or {}

    def __call__(self, cmd, timeout=None):
        with self.lock:
            self.calls.append(cmd)
        key = " ".join(cmd[:3])
        if key in self.fail:
            result = self.fail[key]
            if isinstance(result, BaseException):
                raise result
            return result
        if cmd[:3] == ["docker", "buildx", "version"]:
            return cp(0, "github.com/docker/buildx v0.37.2")
        if cmd[:3] == ["docker", "buildx", "create"]:
            return cp(0, cmd[4])
        if cmd[:3] == ["docker", "buildx", "inspect"]:
            return cp(0, self.inspect)
        if cmd[:3] == ["docker", "buildx", "build"]:
            return self.build(cmd)
        if cmd[:3] == ["docker", "buildx", "rm"]:
            return cp(0)
        if "info" in cmd:
            arch = self.arch(cmd) if callable(self.arch) else self.arch
            return cp(0, f"{arch}|4|4102422528\n")
        if "stats" in cmd:
            name = cmd[-1]
            return cp(0, json.dumps({"Name": name, "CPUPerc": "380.50%", "MemUsage": "1.5GiB / 3.8GiB"}) + "\n")
        raise AssertionError(f"unexpected command: {cmd}")

    def commands(self, prefix):
        return [c for c in self.calls if c[:len(prefix)] == prefix]


def run_args(tmp_path, *extra):
    df = tmp_path / "Dockerfile.benchmark"
    df.write_text("FROM scratch\n")
    return bm.make_parser().parse_args(["run", "-f", str(df), str(tmp_path), "--builder", "mab-test",
                                        "--sample-interval", "0.01", *extra])


# =============================================================================
# 1. Argument builders
# =============================================================================

class TestArgumentBuilders:
    @pytest.mark.parametrize("platform", ["linux/amd64", "linux/arm64", "linux/arm/v7", "linux/riscv64"])
    def test_valid_platforms(self, platform):
        assert bm.validate_platform(platform) == platform

    @pytest.mark.parametrize("platform", ["amd64", "linux/x86_64", "windows/amd64", "linux/arm64;rm -rf /", ""])
    def test_invalid_platforms(self, platform):
        with pytest.raises(bm.BenchmarkError, match="Unsupported platform"):
            bm.validate_platform(platform)

    def test_parse_node_with_ssh_endpoint(self):
        node = bm.parse_node("linux/arm64=ssh://ci@graviton-1", 0, "mab-x")
        assert (node.name, node.platform, node.endpoint) == ("mab-x-0", "linux/arm64", "ssh://ci@graviton-1")
        assert node.docker_target_args() == ["--host", "ssh://ci@graviton-1"]

    def test_parse_node_with_context_name_and_local(self):
        assert bm.parse_node("linux/amd64=x86-runner", 1, "b").docker_target_args() == ["--context", "x86-runner"]
        local = bm.parse_node("linux/amd64", 2, "b")
        assert local.endpoint is None and local.docker_target_args() == []

    @pytest.mark.parametrize("spec", ["linux/arm64=--config=/tmp/x", "linux/arm64=", "linux/arm64=http://x",
                                      "linux/arm64=ssh://a b", "arm64=ssh://host"])
    def test_parse_node_rejects_unsafe_or_malformed(self, spec):
        with pytest.raises(bm.BenchmarkError):
            bm.parse_node(spec, 0, "b")

    def test_single_local_node_create(self):
        assert bm.create_commands("mab-x", []) == [
            ["docker", "buildx", "create", "--name", "mab-x", "--driver", "docker-container", "--node", "mab-x-0"]
        ]

    def test_hybrid_create_appends_after_first_node(self):
        nodes = [bm.parse_node("linux/arm64=ssh://ci@arm", 0, "mab-x"),
                 bm.parse_node("linux/amd64=ssh://ci@x86", 1, "mab-x")]
        first, second = bm.create_commands("mab-x", nodes)
        assert "--append" not in first and first[-1] == "ssh://ci@arm"
        assert second[-2:] == ["--append", "ssh://ci@x86"]
        assert second[second.index("--platform") + 1] == "linux/amd64"
        assert second[second.index("--node") + 1] == "mab-x-1"

    def test_build_command_cold_cacheonly(self):
        cmd = bm.build_command("mab-x", Path("Dockerfile.benchmark"), Path("."), ["linux/amd64", "linux/arm64"],
                               no_cache=True)
        assert cmd[:6] == ["docker", "buildx", "build", "--builder", "mab-x", "--progress=plain"]
        assert cmd[cmd.index("--platform") + 1] == "linux/amd64,linux/arm64"
        assert "--no-cache" in cmd
        assert cmd[cmd.index("--output") + 1] == "type=cacheonly"
        assert cmd[-1] == "."

    def test_build_command_warm_with_build_args(self):
        cmd = bm.build_command("b", Path("D"), Path("ctx"), ["linux/arm64"], no_cache=False,
                               build_args=["RUST_VERSION=1.83.0"])
        assert "--no-cache" not in cmd
        assert cmd[cmd.index("--build-arg") + 1] == "RUST_VERSION=1.83.0"

    def test_load_rejected_for_multi_platform(self):
        with pytest.raises(bm.BenchmarkError, match="manifest list"):
            bm.build_command("b", Path("D"), Path("."), ["linux/amd64", "linux/arm64"], no_cache=True, output="load")

    def test_load_allowed_for_single_platform(self):
        cmd = bm.build_command("b", Path("D"), Path("."), ["linux/arm64"], no_cache=True, output="load")
        assert "--load" in cmd and "--output" not in cmd

    def test_push_requires_tag(self):
        with pytest.raises(bm.BenchmarkError, match="--tag"):
            bm.validate_output(["linux/amd64"], "push", None)
        cmd = bm.build_command("b", Path("D"), Path("."), ["linux/amd64", "linux/arm64"], no_cache=False,
                               output="push", tag="ghcr.io/x/y:bench")
        assert "--push" in cmd and cmd[cmd.index("--tag") + 1] == "ghcr.io/x/y:bench"


# =============================================================================
# 2. Telemetry parsers
# =============================================================================

class TestLogParsing:
    def test_single_platform_steps(self):
        steps = bm.parse_steps(SINGLE_PLATFORM_LOG, ["linux/arm64"])
        assert [s.id for s in steps] == [5, 6, 7, 8, 9, 10]  # internal steps skipped
        assert all(s.platform == "linux/arm64" for s in steps)
        wheel = next(s for s in steps if s.id == 9)
        assert (wheel.stage, wheel.position, wheel.status, wheel.seconds) == ("wheels", "2/2", "done", 141.7)
        assert next(s for s in steps if s.id == 8).status == "cached"

    def test_multi_platform_prefix_and_interleaving(self):
        steps = {s.id: s for s in bm.parse_steps(MULTI_PLATFORM_LOG, ["linux/amd64", "linux/arm64"])}
        assert steps[7].platform == "linux/arm64" and steps[7].seconds == 150.0
        assert steps[8].platform == "linux/amd64" and steps[8].seconds == 1388.9
        assert steps[9].status == "cached"

    def test_error_and_incomplete_steps(self):
        log = ("#9 [wheels 2/2] RUN pip wheel x\n#9 ERROR: process did not complete successfully: exit code: 1\n"
               "#10 [runtime 2/2] COPY --from=wheels /wheels /wheels\n")
        steps = {s.id: s for s in bm.parse_steps(log, ["linux/amd64"])}
        assert steps[9].status == "error" and steps[9].seconds is None
        assert steps[10].status == "incomplete"

    def test_failed_step_records_how_long_it_ran(self):
        log = ("#10 [wheels 3/3] RUN pip wheel --no-binary cryptography cryptography==44.0.0\n"
               "#10 0.512 Collecting cryptography==44.0.0\n"
               "#10 50.05 cc: internal compiler error: Segmentation fault signal terminated program collect2\n"
               "#10 ERROR: process did not complete successfully: exit code: 1\n")
        steps = bm.parse_steps(log, ["linux/amd64"])
        assert steps[0].status == "error" and steps[0].last_output_at == 50.05
        s = bm.summarize(steps, "linux/amd64", "emulated")
        assert s.failed_step.startswith("[wheels 3/3] RUN pip wheel") and s.failed_after_seconds == 50.05
        assert s.step_seconds == {}

    def test_step_seconds_keyed_by_step(self):
        s = bm.summarize(bm.parse_steps(SINGLE_PLATFORM_LOG, ["linux/arm64"]), "linux/arm64", "native")
        apt = [v for k, v in s.step_seconds.items() if k.startswith("[toolchain 2/3] RUN apt-get update")]
        assert apt == [24.5] and len(s.step_seconds) == 5
        assert s.failed_step is None

    def test_ansi_and_crlf_tolerated(self):
        noisy = "\r\n".join("\x1b[34m" + ln + "\x1b[0m" for ln in SINGLE_PLATFORM_LOG.splitlines())
        assert len(bm.parse_steps(noisy, ["linux/arm64"])) == 6

    def test_summary_per_platform(self):
        steps = bm.parse_steps(SINGLE_PLATFORM_LOG, ["linux/arm64"])
        s = bm.summarize(steps, "linux/arm64", "native")
        assert (s.steps_executed, s.steps_cached) == (5, 1)
        assert s.cache_hit_ratio == round(1 / 6, 3)
        assert s.slowest_step_seconds == 141.7 and s.slowest_step.startswith("[wheels 2/2] RUN pip wheel")
        assert s.stage_seconds == {"toolchain": 45.6, "wheels": 141.7, "runtime": 0.2}
        assert s.step_seconds_total == 187.5

    def test_summary_separates_platforms(self):
        steps = bm.parse_steps(MULTI_PLATFORM_LOG, ["linux/amd64", "linux/arm64"])
        amd = bm.summarize(steps, "linux/amd64", "emulated")
        arm = bm.summarize(steps, "linux/arm64", "native")
        assert amd.slowest_step_seconds == 1388.9 and amd.steps_cached == 1
        assert arm.slowest_step_seconds == 150.0 and arm.steps_cached == 0

    def test_summary_with_no_steps(self):
        s = bm.summarize([], "linux/amd64", "unknown")
        assert s.cache_hit_ratio == 0.0 and s.slowest_step is None

    @pytest.mark.parametrize("log, category", [
        ("#9 0.31 qemu: uncaught target signal 11 (Segmentation fault) - core dumped\n#9 ERROR: did not complete successfully",
         "qemu_segfault"),
        ("#7 0.05 exec /bin/sh: exec format error\nERROR: failed to solve", "binfmt_missing"),
        ("#9 ERROR: process \"/bin/sh -c pip wheel\" did not complete successfully: exit code: 137", "oom_killed"),
        ("#9 280.1 error: could not compile `pydantic-core`\n#9 280.1 signal: killed\n", "oom_killed"),
        ("#6 12.0 W: Temporary failure resolving 'deb.debian.org'\n#6 ERROR: did not complete successfully",
         "network"),
        ("#9 ERROR: process \"/bin/sh -c false\" did not complete successfully: exit code: 1", "build_failed"),
        ("something else entirely", "unknown"),
    ], ids=["qemu-segv", "binfmt", "oom-137", "oom-signal", "network", "generic", "unknown"])
    def test_failure_classification(self, log, category):
        assert bm.classify_failure(log)[0] == category

    def test_failure_classification_returns_matching_line(self):
        log = "#9 0.1 ok\n#9 0.31 qemu: uncaught target signal 11 (Segmentation fault) - core dumped\n#9 ERROR: x"
        assert bm.classify_failure(log)[1] == "#9 0.31 qemu: uncaught target signal 11 (Segmentation fault) - core dumped"

    @pytest.mark.parametrize("value, expected", [
        ("1.5GiB", int(1.5 * 1024**3)), ("512MiB", 512 * 1024**2), ("12.3kB", 12300), ("0B", 0), (" 2GB ", 2 * 1000**3),
    ])
    def test_parse_mem(self, value, expected):
        assert bm.parse_mem(value) == expected

    def test_parse_mem_rejects_garbage(self):
        with pytest.raises(ValueError):
            bm.parse_mem("lots")

    def test_parse_stats_line(self):
        line = json.dumps({"Name": "buildx_buildkit_mab-x-0", "CPUPerc": "380.50%", "MemUsage": "1.5GiB / 3.8GiB"})
        assert bm.parse_stats_line(line) == ("buildx_buildkit_mab-x-0", 380.5, int(1.5 * 1024**3))
        assert bm.parse_stats_line("not json") is None
        assert bm.parse_stats_line(json.dumps({"Name": "x", "CPUPerc": "--", "MemUsage": "-- / --"})) is None

    def test_inspect_platforms_strip_pin_markers(self):
        assert bm.parse_inspect_platforms(INSPECT_HYBRID) == {"linux/arm64", "linux/amd64", "linux/amd64/v2", "linux/386"}

    def test_execution_modes(self):
        infos = [bm.NodeInfo(name="a", endpoint=None, pinned_platform=None, arch="arm64", ncpu=4, mem_total_bytes=1)]
        assert bm.execution_modes(["linux/arm64", "linux/amd64"], infos) == {"linux/arm64": "native", "linux/amd64": "emulated"}
        hybrid = [bm.NodeInfo(name="a", endpoint="ssh://a", pinned_platform="linux/arm64", arch="arm64", ncpu=4, mem_total_bytes=1),
                  bm.NodeInfo(name="b", endpoint="ssh://b", pinned_platform="linux/amd64", arch="amd64", ncpu=4, mem_total_bytes=1)]
        assert bm.execution_modes(["linux/arm64", "linux/amd64"], hybrid) == {"linux/arm64": "native", "linux/amd64": "native"}
        assert bm.execution_modes(["linux/amd64"], [bm.NodeInfo(name="a", endpoint=None, pinned_platform=None,
                                                                 arch=None, ncpu=None, mem_total_bytes=None)]) == {"linux/amd64": "unknown"}

    def test_stats_sampler_aggregates(self):
        runner = FakeRunner()
        node = bm.NodeSpec(name="mab-x-0", platform="*")
        sampler = bm.StatsSampler([node], {"mab-x-0": 4}, runner)
        sampler.sample_once()
        sampler.sample_once()
        st = sampler.stats["mab-x-0"]
        assert st.samples == 2 and st.cpu_peak_pct == 380.5 and st.cpu_mean_pct == 380.5
        assert st.saturation == round(380.5 / 400, 3)
        assert runner.calls[0][-1] == "buildx_buildkit_mab-x-0"

    def test_stats_sampler_addresses_remote_node(self):
        runner = FakeRunner()
        node = bm.parse_node("linux/arm64=ssh://ci@arm", 0, "mab-x")
        bm.StatsSampler([node], {}, runner).sample_once()
        assert runner.calls[0][:3] == ["docker", "--host", "ssh://ci@arm"]


# =============================================================================
# 3. Builder lifecycle and teardown
# =============================================================================

class TestTeardown:
    def test_builder_removed_after_success(self, tmp_path):
        runner = FakeRunner()
        report = bm.run_benchmark(run_args(tmp_path, "--platform", "linux/arm64"), runner)
        assert report.success
        assert runner.commands(["docker", "buildx", "rm"]) == [["docker", "buildx", "rm", "mab-test"]]

    def test_builder_removed_when_build_raises(self, tmp_path):
        def boom(cmd):
            raise RuntimeError("runner crashed")
        runner = FakeRunner(build=boom)
        with pytest.raises(RuntimeError):
            bm.run_benchmark(run_args(tmp_path, "--platform", "linux/arm64"), runner)
        assert runner.commands(["docker", "buildx", "rm"])

    def test_builder_removed_on_keyboard_interrupt(self, tmp_path):
        def ctrl_c(cmd):
            raise KeyboardInterrupt
        runner = FakeRunner(build=ctrl_c)
        assert bm.main(["run", "-f", str(self._df(tmp_path)), str(tmp_path), "--builder", "mab-test",
                        "--platform", "linux/arm64"], runner) == 130
        assert runner.commands(["docker", "buildx", "rm"])

    def test_sigterm_becomes_exception_and_handler_is_restored(self, tmp_path):
        before = signal.getsignal(signal.SIGTERM)

        def cancelled(cmd):
            signal.raise_signal(signal.SIGTERM)  # what a CI cancel delivers
            return cp(0)
        runner = FakeRunner(build=cancelled)
        assert bm.main(["run", "-f", str(self._df(tmp_path)), str(tmp_path), "--builder", "mab-test",
                        "--platform", "linux/arm64"], runner) == 130
        assert runner.commands(["docker", "buildx", "rm"])
        assert signal.getsignal(signal.SIGTERM) == before

    def test_partial_append_failure_still_removes_builder(self, tmp_path):
        runner = FakeRunner()
        original = runner.__call__

        def flaky(cmd, timeout=None):
            if cmd[:3] == ["docker", "buildx", "create"] and "--append" in cmd:
                runner.calls.append(cmd)
                return cp(1, "", "ssh: connect to host x86 port 22: Connection refused")
            return original(cmd, timeout)
        args = run_args(tmp_path, "--node", "linux/arm64", "--node", "linux/amd64=ssh://ci@x86")
        with pytest.raises(bm.BenchmarkError, match="Connection refused"):
            bm.run_benchmark(args, flaky)
        assert runner.commands(["docker", "buildx", "rm"])

    def test_keep_builder_still_removes_half_created_builder(self, tmp_path):
        # Real buildx error when two nodes point at the same daemon endpoint.
        runner = FakeRunner()
        original = runner.__call__

        def dup(cmd, timeout=None):
            if cmd[:3] == ["docker", "buildx", "create"] and "--append" in cmd:
                runner.calls.append(cmd)
                return cp(1, "", "ERROR: invalid duplicate endpoint colima")
            return original(cmd, timeout)
        args = run_args(tmp_path, "--node", "linux/arm64", "--node", "linux/amd64", "--keep-builder")
        with pytest.raises(bm.BenchmarkError, match="duplicate endpoint"):
            bm.run_benchmark(args, dup)
        assert runner.commands(["docker", "buildx", "rm"])

    def test_step_label_without_stage_name(self):
        steps = bm.parse_steps("#8 [linux/arm64 2/2] RUN echo hi\n#8 DONE 0.1s\n", ["linux/arm64", "linux/amd64"])
        assert steps[0].stage is None and steps[0].label == "[2/2] RUN echo hi"

    def test_keep_builder_skips_removal(self, tmp_path):
        runner = FakeRunner()
        bm.run_benchmark(run_args(tmp_path, "--platform", "linux/arm64", "--keep-builder"), runner)
        assert not runner.commands(["docker", "buildx", "rm"])

    def test_load_rejected_before_builder_is_created(self, tmp_path):
        runner = FakeRunner()
        assert bm.main(["run", "-f", str(self._df(tmp_path)), str(tmp_path), "--output", "load"], runner) == 2
        assert not runner.commands(["docker", "buildx", "create"])

    def test_missing_platform_reports_binfmt_hint(self, tmp_path):
        runner = FakeRunner(inspect="Platforms: linux/arm64\n")
        with pytest.raises(bm.BenchmarkError, match="tonistiigi/binfmt"):
            bm.run_benchmark(run_args(tmp_path), runner)
        assert not runner.commands(["docker", "buildx", "build"])
        assert runner.commands(["docker", "buildx", "rm"])

    @staticmethod
    def _df(tmp_path):
        df = tmp_path / "Dockerfile.benchmark"
        df.write_text("FROM scratch\n")
        return df


# =============================================================================
# 4. Orchestration and reporting
# =============================================================================

class TestRuns:
    def test_cold_then_warm(self, tmp_path):
        runner = FakeRunner()
        report = bm.run_benchmark(run_args(tmp_path, "--platform", "linux/arm64", "--runs", "3"), runner)
        builds = runner.commands(["docker", "buildx", "build"])
        assert ["--no-cache" in b for b in builds] == [True, False, False]
        assert [r.cold for r in report.runs] == [True, False, False]

    # Real line from a QEMU-emulated linux/amd64 build of cryptography on an Apple M4 (Colima, buildx 0.37.2).
    COLLECT2_SEGV = ("#10 50.05         = note: cc: internal compiler error: Segmentation fault signal "
                     "terminated program collect2\n#10 ERROR: process \"/bin/sh -c pip wheel\" did not complete successfully: exit code: 1\n")

    def test_plain_segfault_is_classified(self):
        category, line = bm.classify_failure(self.COLLECT2_SEGV)
        assert category == "segfault" and "collect2" in line

    def test_segfault_under_emulation_becomes_qemu_segfault(self, tmp_path):
        runner = FakeRunner(arch="aarch64", build=lambda cmd: cp(1, "", self.COLLECT2_SEGV))
        report = bm.run_benchmark(run_args(tmp_path, "--platform", "linux/amd64"), runner)
        assert report.execution == {"linux/amd64": "emulated"}
        assert report.runs[0].failure_category == "qemu_segfault"

    def test_segfault_on_native_build_stays_generic(self, tmp_path):
        runner = FakeRunner(arch="aarch64", build=lambda cmd: cp(1, "", self.COLLECT2_SEGV))
        report = bm.run_benchmark(run_args(tmp_path, "--platform", "linux/arm64"), runner)
        assert report.runs[0].failure_category == "segfault"

    def test_warm_only(self, tmp_path):
        runner = FakeRunner()
        bm.run_benchmark(run_args(tmp_path, "--platform", "linux/arm64", "--runs", "1", "--warm-only"), runner)
        assert "--no-cache" not in runner.commands(["docker", "buildx", "build"])[0]

    def test_failed_cold_build_stops_and_classifies(self, tmp_path):
        log = "#9 [linux/amd64 wheels 2/2] RUN pip wheel x\n#9 1.0 qemu: uncaught target signal 11 (Segmentation fault) - core dumped\n"
        runner = FakeRunner(build=lambda cmd: cp(1, "", log))
        report = bm.run_benchmark(run_args(tmp_path, "--runs", "2"), runner)
        assert not report.success and len(report.runs) == 1
        assert report.runs[0].failure_category == "qemu_segfault"

    def test_timeout_is_recorded(self, tmp_path):
        def slow(cmd):
            raise subprocess.TimeoutExpired(cmd, 5, output=b"#9 [wheels 2/2] RUN pip wheel x\n")
        report = bm.run_benchmark(run_args(tmp_path, "--platform", "linux/arm64"), FakeRunner(build=slow))
        assert report.runs[0].exit_code == 124 and report.runs[0].failure_category == "timeout"

    def test_execution_mode_in_report(self, tmp_path):
        report = bm.run_benchmark(run_args(tmp_path, "--runs", "1"), FakeRunner(arch="aarch64"))
        assert report.execution == {"linux/amd64": "emulated", "linux/arm64": "native"}
        assert report.nodes[0].ncpu == 4

    def test_hybrid_nodes_are_native_and_sampled_separately(self, tmp_path):
        def arch(cmd):
            return "x86_64" if "ssh://ci@x86" in cmd else "aarch64"
        runner = FakeRunner(inspect=INSPECT_HYBRID, arch=arch,
                            build=lambda cmd: cp(0, "", MULTI_PLATFORM_LOG))
        args = run_args(tmp_path, "--runs", "1", "--node", "linux/arm64", "--node", "linux/amd64=ssh://ci@x86")
        report = bm.run_benchmark(args, runner)
        assert report.execution == {"linux/amd64": "native", "linux/arm64": "native"}
        assert {n.node for n in report.runs[0].nodes} == {"mab-test-0", "mab-test-1"}
        assert any(c[:3] == ["docker", "--host", "ssh://ci@x86"] and "stats" in c for c in runner.calls)

    def test_main_writes_json_and_exit_codes(self, tmp_path, capsys):
        out = tmp_path / "out" / "r.json"
        df = TestTeardown._df(tmp_path)
        assert bm.main(["run", "-f", str(df), str(tmp_path), "--platform", "linux/arm64", "--runs", "1",
                        "--json", str(out), "--label", "native-arm64"], FakeRunner()) == 0
        report = bm.BenchmarkReport.model_validate_json(out.read_text())
        assert report.label == "native-arm64"
        assert "run 0 (cold)" in capsys.readouterr().out
        failing = FakeRunner(build=lambda cmd: cp(1, "", "ERROR: failed to solve"))
        assert bm.main(["run", "-f", str(df), str(tmp_path), "--platform", "linux/arm64", "--runs", "1"], failing) == 1

    def test_main_exit_2_without_buildx(self, tmp_path):
        runner = FakeRunner(fail={"docker buildx version": cp(1, "", "unknown command: buildx")})
        assert bm.main(["run", "-f", str(TestTeardown._df(tmp_path)), str(tmp_path)], runner) == 2

    def test_compare_table_speedup(self, tmp_path):
        def report(label, cold, warm, execution):
            run = lambda i, c, s: bm.RunResult(index=i, cold=c, success=True, exit_code=0, wall_seconds=s, platforms=[
                bm.PlatformSummary(platform="linux/amd64", execution=execution, steps_executed=0 if not c else 6,
                                   steps_cached=6 if not c else 0, cache_hit_ratio=1.0, step_seconds_total=s)],
                nodes=[bm.NodeStats(node="n", samples=3, cpu_mean_pct=350.0, cpu_peak_pct=399.0,
                                    mem_peak_bytes=2 * 1024**3, ncpu=4)])
            return bm.BenchmarkReport(label=label, builder="b", dockerfile="D", platforms=["linux/amd64"],
                                      execution={"linux/amd64": execution}, nodes=[], runs=[run(0, True, cold), run(1, False, warm)])
        paths = []
        for r in [report("qemu-amd64", 1400.0, 3.0, "emulated"), report("native-amd64", 200.0, 2.0, "native")]:
            p = tmp_path / f"{r.label}.json"
            p.write_text(r.model_dump_json())
            paths.append(str(p))
        table = bm.compare_markdown([bm.BenchmarkReport.model_validate_json(Path(p).read_text()) for p in paths])
        rows = table.splitlines()
        assert "| qemu-amd64 | amd64 (emulated) | 1400s | baseline | 3s | 6/6 | 350% (88%) | 2.00 GiB | ok |" in rows
        assert "| native-amd64 | amd64 (native) | 200s | 7.00x |" in rows[3]

    def test_compare_marks_failed_cold_runs(self):
        failed = bm.BenchmarkReport(label="qemu", builder="b", dockerfile="D", platforms=["linux/amd64"],
                                    execution={"linux/amd64": "emulated"}, nodes=[], runs=[
            bm.RunResult(index=0, cold=True, success=False, exit_code=1, wall_seconds=139.7,
                         failure_category="qemu_segfault", platforms=[], nodes=[])])
        row = bm.compare_markdown([failed]).splitlines()[2]
        assert "| 140s (crashed) |" in row and "FAIL: qemu_segfault" in row

    def test_dockerfile_benchmark_is_multi_stage_and_pinned(self):
        text = (ROOT / "Dockerfile.benchmark").read_text()
        assert text.startswith("# syntax=docker/dockerfile:1.7")
        stages = [ln.split()[-1] for ln in text.splitlines() if ln.startswith("FROM ")]
        assert stages == ["toolchain", "wheels", "runtime"]
        for pin in ("RUST_VERSION=", "CRYPTOGRAPHY_VERSION=", "PYDANTIC_CORE_VERSION=",
                    "--no-binary pydantic-core", "--no-binary cryptography"):
            assert pin in text
