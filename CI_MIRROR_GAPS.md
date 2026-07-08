# Fork CI Mirror Gaps

This document lists the CPU-only subset of TensorFlow's CI that is mirrored on this fork, and the permanent gaps due to unavailable GPU/self-hosted runners.

## What's Mirrored

- **PyLint (lint check)** — CPU-only static analysis on Python files using `tensorflow/tools/ci_build/pylintrc`
  - Runs on: `ubuntu-latest` (GitHub-hosted)
  - Triggered by: PR events or manual workflow dispatch or `/run-ci-mirror` comment

## Permanent Gaps (GPU/Specialized Runners Required)

These jobs require CUDA, TensorFlow's self-hosted GPU runners, or other specialized infrastructure not available on fork:

- **TensorFlow CPU build** — Full C++/CUDA build (requires self-hosted runner with build toolchain)
- **XLA compilation tests** — GPU-backed XLA/CUDA validation
- **TFLite build and tests** — Requires specific build environment
- **GPU-accelerated unit tests** — CUDA compute capability, cuDNN, TensorRT dependencies
- **Wheel builds** — Requires self-hosted infrastructure for cross-platform builds
- **ARM/architecture-specific builds** — Requires ARM runners
- **Distributed training tests** — Requires multi-GPU setup

The fork CI mirror provides a lightweight validation gate for code quality (lint) without committing to expensive build/test infrastructure. Full CI validation happens on the upstream repository.

## Usage

Trigger lint check on a fork PR:
1. Open or update a PR in this fork
2. Manually trigger with workflow dispatch, OR
3. Comment `/run-ci-mirror` on an open PR

The workflow will run PyLint on Python files and report any style violations.
