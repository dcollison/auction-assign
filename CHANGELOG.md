# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-11-23

### Added
- MkDocs Material documentation (`mkdocs.yml`, `docs/`) with `mkdocstrings` API reference and GitHub Pages deployment workflow (`.github/workflows/docs.yml`).
- Repository `.github/CODEOWNERS` and `.pre-commit-config.yaml` hook configuration.

### Changed
- Aligned Python baseline to `3.14`, `pyproject.toml` metadata/URLs/classifiers, CI matrix across Python `3.10`–`3.14` and `ubuntu`/`windows`/`macos`, and PyPI release workflow (`release: [published]`, `pypi` environment).
- Upgraded locked dependencies in `uv.lock` (`backrefs`, `certifi`, `charset-normalizer`, `filelock`, `pymdown-extensions`, `pyyaml`, `virtualenv`, `watchdog`).

## [0.1.1] - 2026-11-18

### Changed
- Eliminated square-padding ping-pong overhead in the pure-Rust solver.
- Added a CSR sparse execution path for heavily gated (`inf` / `-inf`) cost matrices.
- Added a 4-way SIMD scan in the forward/reverse auction bidding loop for improved throughput.

## [0.1.0] - 2026-11-18

### Added
- Pure-Rust Bertsekas forward–reverse auction solver with automatic $\varepsilon$-scaling (`src/solver.rs`).
- Native `cost_limit` unassigned-slack partial assignment without quadratic padding.
- Hard edge gating via `+inf` (minimization) and `-inf` (maximization), raising `InfeasibleError` when no complete assignment exists.
- PyO3 + `rust-numpy` extension module (`auction_assign._core`) releasing the GIL during solves (`py.detach`).
- Typed Python API (`auction_assign`, `solve`, `AuctionResult`, `AuctionInfo`, `InfeasibleError`, `ConvergenceError`) with `py.typed` and `_core.pyi` stubs.
- Differential Hypothesis test suite against `scipy.optimize.linear_sum_assignment` and exact partial-matching oracle, plus benchmark suite (`benchmarks/bench_assign.py`).
- Automated GitHub Actions CI and PyPI OIDC Trusted Publishing release workflows.
