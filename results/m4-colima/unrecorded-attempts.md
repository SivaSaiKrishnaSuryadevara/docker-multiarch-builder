# Unrecorded QEMU attempts (M4 / Colima, linux/amd64 emulated)

Two of the five ARM-host-emulating-amd64 attempts have **no harness JSON report**:

- **Attempt 1:** its report and BuildKit log were deleted when `results/` was cleared before re-running with a revised `Dockerfile.benchmark`.
- **Attempt 2:** it hung and was stopped with SIGTERM. At the time, the harness did not write a report for interrupted runs (fixed in a later commit).

This file preserves the raw output that survived, verbatim. These are logs, not harness reports: treat them as weaker evidence than the JSON files in this directory and in `results/gha/`.

Environment for both: Apple M4, Colima (`vz`, Rosetta off, 4 vCPU / 4 GiB), buildx 0.37.2, 5 October 2026.

## Attempt 1: linker segfault (earlier Dockerfile revision)

This revision built cryptography and pydantic-core in a single `RUN` step. The harness classified the run as `build_failed`, because the classifier did not yet recognise plain `Segmentation fault` lines.

Harness console output:

```text
INFO builder mab-2a6dac77 ready; execution: linux/amd64=emulated
INFO run 1/2 (cold) started
INFO run 1/2 FAILED (build_failed) in 429.2s
INFO removed builder mab-2a6dac77
run 0 (cold): 429.2s FAIL [build_failed] #10 332.9 ERROR: Failed to build one or more wheels
  node mab-2a6dac77-0: cpu mean 181% peak 415%, saturation 45%, mem peak 1.81 GiB
```

Process snapshot inside the BuildKit container during the wheel step (16:14:11 EDT). Every process runs under `qemu-x86_64`:

```text
buildx_buildkit_mab-2a6dac77-0 100.34% 1.76GiB / 3.813GiB
 5:08    0:08 {pip} /usr/bin/qemu-x86_64 /usr/local/bin/python3.12 /usr/local/bin/python3.12 /usr/local/bin/pip wheel ...
 4:18    0:00 {maturin} /usr/bin/qemu-x86_64 /tmp/pip-build-env-xgjwezzv/overlay/bin/maturin maturin pep517 build-wheel ...
 4:16    0:00 {cargo} /usr/bin/qemu-x86_64 /opt/rustup/toolchains/1.83.0-x86_64-unknown-linux-gnu/bin/cargo ...
 2:20    2:18 {rustc} /usr/bin/qemu-x86_64 /opt/rustup/toolchains/1.83.0-x86_64-unknown-linux-gnu/bin/rustc ...
```

Lines excerpted from the BuildKit log before it was deleted (transcribed during the session, not the original file):

```text
#10 50.05       error: linking with `cc` failed: exit status: 4
#10 50.05         = note: cc: internal compiler error: Segmentation fault signal terminated program collect2
#10 50.05       error: could not compile `pyo3-macros-backend` (build script) due to 1 previous error
#10 50.06   ERROR: Failed building wheel for cryptography
#10 50.06   Building wheel for pydantic-core (pyproject.toml): started
#10 332.2   Building wheel for pydantic-core (pyproject.toml): finished with status 'done'
#10 332.9 ERROR: Failed to build one or more wheels
```

## Attempt 2: `cargo` hang during pydantic-core

This used the current Dockerfile revision. The build stalled in the pydantic-core step and was stopped with SIGTERM. The builder was torn down cleanly:

```text
INFO builder mab-861566e5 ready; execution: linux/amd64=emulated
INFO run 1/2 (cold) started
INFO removed builder mab-861566e5
ERROR interrupted; builder removed
```

BuildKit container CPU, sampled every ~30 s by an external watcher, 16:25–16:40 EDT. It never rose above 1.23% (a running compile shows ~100%+):

```text
16:25:22 0.06%  16:27:28 0.84%  16:30:06 0.23%  16:32:43 0.49%  16:35:21 0.08%  16:37:59 1.23%  16:40:36 0.09%
(30 samples total, all < 1.3%)
```

Process listing at the end of the stall window. The newest `cargo` process had been alive 17 min 38 s with **0:00 CPU time**, and no `rustc` was running:

```text
19:39    0:07 {pip} /usr/bin/qemu-x86_64 /usr/local/bin/python3.12 /usr/local/bin/python3.12 /usr/local/bin/pip wheel ...
19:18    0:00 {maturin} /usr/bin/qemu-x86_64 /tmp/pip-build-env-whz_2c7i/overlay/bin/maturin maturin pep517 build-wheel ...
19:17    0:01 {cargo} /usr/bin/qemu-x86_64 /opt/rustup/toolchains/1.83.0-x86_64-unknown-linux-gnu/bin/cargo ...
17:38    0:00 {cargo} /usr/bin/qemu-x86_64 /opt/rustup/toolchains/1.83.0-x86_64-unknown-linux-gnu/bin/cargo ...
```
