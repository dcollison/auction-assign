# `auction-assign` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the `auction-assign` Rust-accelerated Python package (`import auction_assign`) implementing Bertsekas's forward–reverse auction algorithm with automatic $\varepsilon$-scaling, `uv`-managed dev tooling, and GitHub Actions CI + PyPI Trusted Publishing.

**Architecture:** A single mixed Rust/Python `maturin` project where `src/solver.rs` holds a PyO3-free pure-Rust solver (testable via `cargo test`), `src/lib.rs` exposes a thin PyO3/`rust-numpy` `_core` extension releasing the GIL during solve, and `python/auction_assign/__init__.py` provides strict-typed NumPy coercion, `auction_assign(...)`, `solve(...)`, dataclasses, and exceptions.

**Tech Stack:** Rust (`rustc >= 1.85`, `pyo3 0.30`, `numpy 0.30`), Python (`>= 3.10`, `abi3-py310`, `numpy >= 1.24`), `maturin`, `uv`, `ruff`, `mypy`, `pytest`, `hypothesis`, `scipy` (dev/test oracle), GitHub Actions (`PyO3/maturin-action`, `pypa/gh-action-pypi-publish`).

**Spec:** [2026-11-18-auction-assign-design.md](file:///d:/repo/auction-assignment/docs/superpowers/specs/2026-11-18-auction-assign-design.md)

---

## Global Constraints
- Package name: `auction-assign`; Python import name: `auction_assign`; internal Rust module: `auction_assign._core`; license: MIT.
- Python version floor: `requires-python = ">=3.10"`, built with `abi3-py310` feature; runtime dependency: `numpy>=1.24`.
- `pyo3/extension-module` MUST NOT be in default `Cargo.toml` features (only passed via `[tool.maturin] features = ["pyo3/extension-module"]`) so `cargo test` links natively without Python linker errors.
- `auction_assign(...)` always returns `(row_ind, col_ind)` (`np.ndarray` of dtype `np.intp`, sorted by `row_ind`); `solve(...)` always returns `AuctionResult`. No boolean return-type flags.
- Validation rules: `ndim != 2`, `NaN` costs, unbounded infinities (`-inf` minimizing / `+inf` maximizing), `NaN` `cost_limit`, `eps_min <= 0` or non-finite, `eps_factor <= 1` or non-finite, and `max_iterations <= 0` raise `ValueError`.
- Infeasible gated complete assignment raises `InfeasibleError(ValueError)`; exceeding `max_iterations` or ULP stagnation raises `ConvergenceError(RuntimeError)` carrying `.iterations` and `.final_eps`.
- `cost_limit` uses strict inequality (`cost < cost_limit` when minimizing; `score > cost_limit` when maximizing).

## Review Focus
1. **Non-contiguous / sliced views (`arr.T`, `arr[::2, ::-1]`) and non-`float64` inputs (`int64`, `float32`, nested lists)** — must coerce cleanly via `np.ascontiguousarray(..., dtype=np.float64)` without corrupting indices or panicking in Rust. (Pinned in Task 4)
2. **All-equal or all-zero cost matrices (`np.zeros`, constant matrix)** — `scale == 0` must terminate cleanly in one pass at `eps_min` without division by zero. (Pinned in Task 2)
3. **Invalid dimensions (`ndim != 2`) and degenerate 2D shapes (`0x0`, `0x5`, `5x0`, `1x1`)** — must raise `ValueError` for `ndim != 2` and return empty `np.intp` arrays + `0.0` cost for empty 2D shapes. (Pinned in Tasks 2 & 4)
4. **Extreme dynamic-range / tight `eps_min` causing `f64` ULP price stagnation (`p_new == p_old`)** — must raise `ConvergenceError` instead of hanging indefinitely. (Pinned in Tasks 2 & 4)
5. **`cost_limit` exact-boundary equality (`cost == cost_limit`) and mixed `inf` + `cost_limit`** — exact boundary pair must remain unassigned (never arbitrarily matched), and output pairs must have unique rows and unique columns. (Pinned in Tasks 3 & 5)

---

## File Structure
- `Cargo.toml` — Crate metadata, `cdylib` lib target `_core`, `pyo3` (`abi3-py310`) + `numpy` deps.
- `pyproject.toml` — `maturin` PEP 517 backend, `uv` `cache-keys` + `dev` dependency group, `ruff`, `mypy`, `pytest` config.
- `.python-version`, `.gitignore`, `LICENSE` (MIT) — Repo baseline.
- `src/solver.rs` — Pure-Rust solver (`SolverError`, `SolveConfig`, `SolveOutput`, `solve_dense`), bipartite feasibility check, complete forward–reverse auction, partial `cost_limit` auction, and Rust unit tests.
- `src/lib.rs` — Thin PyO3 + `rust-numpy` FFI layer exposing `solve_dense` and `InfeasibleError`/`ConvergenceError` on `auction_assign._core`.
- `python/auction_assign/__init__.py` — Public Python API (`auction_assign`, `solve`, `AuctionResult`, `AuctionInfo`, `InfeasibleError`, `ConvergenceError`, `__version__`).
- `python/auction_assign/_core.pyi` — Type stubs for `_core`.
- `python/auction_assign/py.typed` — PEP 561 typed package marker.
- `tests/test_correctness.py` — Unit, edge-case, validation, exception, and GIL-release tests.
- `tests/test_differential.py` — Hypothesis property & differential tests against `scipy.optimize.linear_sum_assignment` and brute-force partial-matching oracle.
- `README.md` — Usage, track-to-track fusion example, API summary, dev commands.
- `.github/workflows/ci.yml` — Cross-platform Rust + Python CI checks.
- `.github/workflows/release.yml` — Tag-triggered (`v*`) cross-platform `abi3` wheels + `sdist` build & PyPI OIDC Trusted Publishing.
- `benchmarks/bench_assign.py` — Reproducible timing benchmarks comparing `auction_assign` vs. `scipy.optimize.linear_sum_assignment` on dense and gated fusion matrices.

---

### Task 1: Project Scaffolding, Build Config, and Local `uv` Environment

**Files:**
- Create: `.gitignore`, `.python-version`, `LICENSE`, `Cargo.toml`, `pyproject.toml`, `src/solver.rs`, `src/lib.rs`, `python/auction_assign/__init__.py`, `python/auction_assign/_core.pyi`, `python/auction_assign/py.typed`, `tests/test_correctness.py`

**Interfaces:**
- Produces: Buildable `maturin` + `uv` project with `auction_assign._core` importing into `auction_assign`, `cargo test` green, and `uv run pytest` green.

- [ ] **Step 1: Initialize git repository and baseline repo files**
  - Run `git init` (if `.git` does not already exist).
  - Create `.gitignore` (ignoring `target/`, `.venv/`, `dist/`, `*.egg-info/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.superpowers/`, `*.pyd`, `*.so`), `.python-version` (`3.12`), and `LICENSE` (MIT license).

- [ ] **Step 2: Create `Cargo.toml` and `pyproject.toml`**
  - `Cargo.toml`:
    ```toml
    [package]
    name = "auction-assign"
    version = "0.1.0"
    edition = "2024"
    rust-version = "1.85"
    license = "MIT"

    [lib]
    name = "_core"
    crate-type = ["cdylib"]

    [dependencies]
    numpy = "0.30"
    pyo3 = { version = "0.30", features = ["abi3-py310"] }
    ```
  - `pyproject.toml`:
    - `[build-system]` requires `["maturin>=1.7,<2.0"]`, `build-backend = "maturin"`.
    - `[project]` name `"auction-assign"`, `version = "0.1.0"`, `requires-python = ">=3.10"`, `dependencies = ["numpy>=1.24"]`, MIT license, classifiers.
    - `[dependency-groups]` `dev = ["pytest>=8.0", "hypothesis>=6.100", "scipy>=1.11", "ruff>=0.8", "mypy>=1.13", "maturin>=1.7,<2.0"]`.
    - `[tool.maturin]` `python-source = "python"`, `module-name = "auction_assign._core"`, `features = ["pyo3/extension-module"]`.
    - `[tool.uv]` `cache-keys = [{ file = "pyproject.toml" }, { file = "Cargo.toml" }, { file = "Cargo.lock" }, { file = "src/**/*.rs" }]`.
    - `[tool.ruff]` (`line-length = 100`, `target-version = "py310"`, lint select `["E", "F", "I", "B", "UP", "NPY"]`), `[tool.mypy]` (`python_version = "3.10"`, `strict = true`), `[tool.pytest.ini_options]` (`testpaths = ["tests"]`).

- [ ] **Step 3: Write failing smoke test in `tests/test_correctness.py`**
  ```python
  import auction_assign


  def test_package_imports() -> None:
      assert hasattr(auction_assign, "__version__")
  ```

- [ ] **Step 4: Create minimal `src/solver.rs`, `src/lib.rs`, `python/auction_assign/{__init__.py,_core.pyi,py.typed}`**
  - `src/solver.rs`: placeholder module with one passing `#[test]` (`fn sanity() { assert_eq!(2 + 2, 4); }`).
  - `src/lib.rs`: declares `mod solver;` and minimal `#[pymodule] fn _core(_m: &Bound<'_, PyModule>) -> PyResult<()> { Ok(()) }`.
  - `python/auction_assign/__init__.py`: imports `from . import _core` and defines `__version__ = "0.1.0"`.
  - `python/auction_assign/_core.pyi`: empty stub placeholder; `py.typed` empty file.

- [ ] **Step 5: Verify Rust + Python build & tests**
  - Run: `cargo test` -> Expected: PASS (`1 passed`).
  - Run: `uv sync` -> Expected: resolves dev deps, compiles `_core` via `maturin` into `.venv`.
  - Run: `uv run pytest tests/test_correctness.py -v` -> Expected: PASS (`1 passed`).

- [ ] **Step 6: Commit**
  ```bash
  git add .gitignore .python-version LICENSE Cargo.toml Cargo.lock pyproject.toml uv.lock src/ python/ tests/
  git commit -m "chore: scaffold auction-assign maturin and uv project"
  ```

---

### Task 2: Pure-Rust Validation, Feasibility Check, and Complete Forward–Reverse Solver (`src/solver.rs`)

**Files:**
- Modify: `src/solver.rs`

**Interfaces:**
- Produces in `crate::solver`:
  - `pub enum SolverError { InvalidInput(String), Infeasible(String), Convergence { message: String, iterations: usize, final_eps: f64 } }`
  - `pub struct SolveConfig { pub maximize: bool, pub cost_limit: Option<f64>, pub eps_min: Option<f64>, pub eps_factor: f64, pub max_iterations: Option<usize> }`
  - `pub struct SolveOutput { pub row_ind: Vec<usize>, pub col_ind: Vec<usize>, pub cost: f64, pub iterations: usize, pub scaling_rounds: usize, pub final_eps: f64, pub row_prices: Vec<f64>, pub col_prices: Vec<f64> }`
  - `pub fn solve_dense(costs: &[f64], n_rows: usize, n_cols: usize, config: &SolveConfig) -> Result<SolveOutput, SolverError>`

- [ ] **Step 1: Write failing Rust unit tests in `src/solver.rs` (`#[cfg(test)] mod tests`)**
  - `test_empty_matrices`: `0x0`, `0x4`, `4x0` -> empty `row_ind`/`col_ind`, `cost == 0.0`, `iterations == 0`.
  - `test_invalid_inputs`: `NaN` in costs -> `InvalidInput`; `-inf` when `maximize=false` -> `InvalidInput`; `+inf` when `maximize=true` -> `InvalidInput`; `eps_min <= 0.0` / `NaN` / `inf` -> `InvalidInput`; `eps_factor <= 1.0` / `NaN` / `inf` -> `InvalidInput`; `max_iterations == Some(0)` -> `InvalidInput`; `costs.len() != n_rows * n_cols` -> `InvalidInput`.
  - `test_infeasible_hall_condition`: 3x3 matrix where rows 0 and 1 only allow col 0 (`inf` elsewhere) -> `Err(SolverError::Infeasible(_))`.
  - `test_square_and_rectangular_complete`:
    - 3x3 classic matrix (`[[4, 1, 3], [2, 0, 5], [3, 2, 2]]`) -> cost `5.0`, pairs `[(0, 1), (1, 0), (2, 2)]`.
    - Rectangular `2x4` (`M < N`) and transposed `4x2` (`M > N`) -> exact optimal cost and sorted `row_ind` (`[0, 1]` and subset of `[0..4]`).
    - Negative costs & `maximize=true` -> exact optimal pairs and sum of original costs.
    - All-equal / all-zero `5x5` matrix -> finishes in 1 scaling round, valid bijection, exact cost.
  - `test_max_iterations_and_ulp_stall`:
    - `max_iterations: Some(1)` on a 4x4 non-trivial matrix -> `Err(SolverError::Convergence { iterations: 1, .. })`.
    - Extreme dynamic range (`0.0`, `1e12`) with `eps_min: Some(1e-20)` -> `Err(SolverError::Convergence { .. })` (halts on ULP stall rather than spinning).

- [ ] **Step 2: Run `cargo test` to verify failures**
  - Run: `cargo test` -> Expected: FAIL (missing `solve_dense` / types).

- [ ] **Step 3: Implement complete-assignment solver in `src/solver.rs`**
  - Validate `costs.len() == n_rows * n_cols`, `eps_min`, `eps_factor`, `max_iterations`, `NaN`s, and unbounded `inf`s.
  - Handle `n_rows == 0 || n_cols == 0` fast-path.
  - Implement `has_complete_bipartite_matching(feasible: &[bool], m: usize, n: usize) -> bool` (Kuhn / Hopcroft–Karp augmenting paths, invoked whenever any forbidden `inf` exists in complete-assignment mode).
  - Normalize orientation so `m <= n` (transpose benefit matrix if `n_rows > n_cols`, swap `row_ind`/`col_ind` and `row_prices`/`col_prices` on return, re-sorting `row_ind` ascending).
  - Compute `scale` (`max(b) - min(b)`, or `max(|b|)` if constant non-zero, or `1.0` if all zero), default `eps_min = (scale * 1e-9).max(1e-12)`, `eps = (scale * 0.5).max(eps_min)`.
  - Implement Bertsekas–Castañon forward–reverse auction per scaling round:
    - Forward bids from unassigned rows (maintaining `pi[i]` and `p[j]`); reverse bids for `m == n` (when `unassigned` columns exist) and for `m < n` lowering any unassigned column with `p[j] > lambda` (`lambda = min_{k assigned} p[k]`).
    - Check `iterations >= max_iter` before each bid and check ULP stagnation (`p_new <= p_old` in forward bid / `pi_new <= pi_old` in reverse bid -> `SolverError::Convergence`).
  - Sum original `costs[r * n_cols + c]` over final pairs for exact `cost`.

- [ ] **Step 4: Run Rust tests and clippy**
  - Run: `cargo test` -> Expected: PASS.
  - Run: `cargo clippy --all-targets -- -D warnings` -> Expected: PASS (0 warnings).

- [ ] **Step 5: Commit**
  ```bash
  git add src/solver.rs
  git commit -m "feat: implement pure-rust forward-reverse auction solver for complete assignments"
  ```

---

### Task 3: Pure-Rust Partial Assignment Mode (`cost_limit`) (`src/solver.rs`)

**Files:**
- Modify: `src/solver.rs`

**Interfaces:**
- Consumes & extends: `solve_dense` in `src/solver.rs` when `config.cost_limit` is `Some(limit)`.

- [ ] **Step 1: Write failing Rust tests for `cost_limit` in `src/solver.rs`**
  - `test_cost_limit_unbounded_vs_empty`:
    - `maximize=false, cost_limit=Some(f64::INFINITY)` matches `cost_limit=None`.
    - `maximize=false, cost_limit=Some(f64::NEG_INFINITY)` returns empty matching (`cost == 0.0`).
    - `maximize=true, cost_limit=Some(f64::NEG_INFINITY)` matches `cost_limit=None`.
    - `maximize=true, cost_limit=Some(f64::INFINITY)` returns empty matching (`cost == 0.0`).
  - `test_cost_limit_strict_boundary`:
    - `2x2` matrix `[[5.0, 10.0], [10.0, 5.0]]` with `cost_limit: Some(5.0)` (`maximize=false`) -> returns empty matching (strict `<`, `5.0 == 5.0` is ineligible).
    - With `cost_limit: Some(5.01)` -> matches `[(0, 0), (1, 1)]`, `cost == 10.0`.
  - `test_cost_limit_partial_matching_beats_forced`:
    - `3x3` gated track-fusion matrix where 2 tracks match cleanly (`1.0`, `2.0`) and 3rd track's only feasible edge is `50.0` with `cost_limit = 10.0` -> returns only 2 pairs, `cost == 3.0`.
    - Same scenario with `maximize=true` (rewards `[10.0, 2.0, -5.0]` with `cost_limit = 0.0`) -> skips negative-reward match.
  - `test_cost_limit_rectangular_and_inf`:
    - `4x2` and `2x4` matrices with `inf` entries and finite `cost_limit` -> unique row/col indices, all matched costs `< cost_limit`, optimal total benefit $\sum (L - c_{ij})$ verified against brute-force maximum.

- [ ] **Step 2: Run `cargo test` to verify failures**
  - Run: `cargo test` -> Expected: FAIL (partial mode not yet implemented).

- [ ] **Step 3: Implement native unassigned-slack partial assignment in `src/solver.rs`**
  - Validate `limit` is not `NaN`; handle `+inf`/`-inf` limits per mode.
  - Build net benefits $b_{ij} = L - c_{ij}$ (minimize) or $b_{ij} = c_{ij} - L$ (maximize) only for finite entries with $b_{ij} > 0.0$ (all others `f64::NEG_INFINITY`). If no positive entry exists, return empty `SolveOutput`.
  - Run $\varepsilon$-scaled forward–reverse auction with unassigned floor `0.0` (each row can decline to bid if $\max_j (b_{ij} - p_j) \le 0.0$, staying unassigned at $\pi_i = 0.0$; in reverse pass, unassigned columns with $p_j > 0.0$ bid against rows down to $0.0$).
  - Extract all pairs with `b[i, j] > 0.0` that remain matched, compute exact `cost = sum(costs[r, c])`, sort by `row_ind`.

- [ ] **Step 4: Run Rust tests and clippy**
  - Run: `cargo test` -> Expected: PASS.
  - Run: `cargo clippy --all-targets -- -D warnings` -> Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add src/solver.rs
  git commit -m "feat: add native unassigned-slack cost_limit partial assignment"
  ```

---

### Task 4: PyO3 / `rust-numpy` Bindings and Typed Python API (`src/lib.rs`, `python/auction_assign/`)

**Files:**
- Modify: `src/lib.rs`, `python/auction_assign/__init__.py`, `python/auction_assign/_core.pyi`, `tests/test_correctness.py`

**Interfaces:**
- Consumes: `crate::solver::{solve_dense, SolveConfig, SolveOutput, SolverError}`.
- Produces in `auction_assign`:
  - `auction_assign(cost_matrix: ArrayLike, *, maximize: bool = False, cost_limit: float | None = None, eps_min: float | None = None, eps_factor: float = 5.0, max_iterations: int | None = None) -> tuple[NDArray[np.intp], NDArray[np.intp]]`
  - `solve(cost_matrix: ArrayLike, *, maximize: bool = False, cost_limit: float | None = None, eps_min: float | None = None, eps_factor: float = 5.0, max_iterations: int | None = None) -> AuctionResult`
  - `AuctionInfo(iterations: int, scaling_rounds: int, final_eps: float, row_prices: NDArray[np.float64], col_prices: NDArray[np.float64])`
  - `AuctionResult(row_ind: NDArray[np.intp], col_ind: NDArray[np.intp], cost: float, info: AuctionInfo)`
  - `InfeasibleError(ValueError)`
  - `ConvergenceError(RuntimeError)` with `.iterations: int` and `.final_eps: float`

- [ ] **Step 1: Write failing Python tests in `tests/test_correctness.py`**
  - Test `auction_assign` returns `(row_ind, col_ind)` of dtype `np.intp`, sorted by `row_ind`.
  - Test `solve` returns `AuctionResult` (`cost` matches `cost_matrix[row_ind, col_ind].sum()`, `AuctionInfo` fields populated with matching `row_prices`/`col_prices` lengths).
  - Test input coercion: nested Python list, `int64` array, `float32` array, transposed view `arr.T`, and non-contiguous strided slice `arr[::2, ::-1]`.
  - Test validation errors: 1D / 3D inputs -> `ValueError`; `NaN` -> `ValueError`; invalid `eps_min` / `eps_factor` / `max_iterations` -> `ValueError`.
  - Test `InfeasibleError` (`isinstance(err, ValueError)`) on Hall-infeasible gated matrix.
  - Test `ConvergenceError` (`isinstance(err, RuntimeError)`, `err.iterations == 1`, `err.final_eps > 0`) when `max_iterations=1` on a 4x4 non-trivial matrix.
  - Test `cost_limit`, `maximize=True`, and `0x0` / `0x3` / `3x0` / `1x1` shapes.
  - Test GIL release via `concurrent.futures.ThreadPoolExecutor` solving matrices concurrently.

- [ ] **Step 2: Run `pytest` to verify failures**
  - Run: `uv run pytest tests/test_correctness.py -v` -> Expected: FAIL (`ImportError` / `AttributeError`).

- [ ] **Step 3: Implement `src/lib.rs`, `python/auction_assign/_core.pyi`, and `python/auction_assign/__init__.py`**
  - In `src/lib.rs`:
    - Define `InfeasibleError` (subclass of `PyValueError`) and `ConvergenceError` (subclass of `PyRuntimeError`) or let Python construct `ConvergenceError(msg, iterations=..., final_eps=...)` via structured Rust error tuple/exception. (Simplest & cleanest: expose `InfeasibleError` + `ConvergenceError` in Python or `create_exception!` in Rust with `iterations`/`final_eps` set via `PyErr::from_value` / helper).
    - Expose `#[pyfunction] fn solve_dense<'py>(py: Python<'py>, costs: PyReadonlyArray2<'py, f64>, maximize: bool, cost_limit: Option<f64>, eps_min: Option<f64>, eps_factor: f64, max_iterations: Option<usize>) -> PyResult<...>` using `costs.as_slice()?` + `py.detach(|| solver::solve_dense(...))` and returning `(row_ind.into_pyarray(py), col_ind.into_pyarray(py), cost, iterations, scaling_rounds, final_eps, row_prices.into_pyarray(py), col_prices.into_pyarray(py))` with `row_ind`/`col_ind` converted to `isize` (`np.intp`).
  - In `python/auction_assign/__init__.py`:
    - Coerce `arr = np.asarray(cost_matrix, dtype=np.float64)`, validate `arr.ndim == 2`, ensure `arr = np.ascontiguousarray(arr)` before calling `_core.solve_dense(...)`.
    - Wrap result into `AuctionInfo` and `AuctionResult`; `auction_assign(...)` delegates to `solve(...)` and returns `(res.row_ind, res.col_ind)`.
  - Update `python/auction_assign/_core.pyi` with exact signatures matching `src/lib.rs`.

- [ ] **Step 4: Run sync, pytest, ruff, and mypy**
  - Run: `uv sync` -> Expected: builds `_core` cleanly.
  - Run: `uv run pytest tests/test_correctness.py -v` -> Expected: PASS.
  - Run: `uv run ruff check . && uv run ruff format --check .` -> Expected: PASS.
  - Run: `uv run mypy python tests` -> Expected: PASS (0 errors).

- [ ] **Step 5: Commit**
  ```bash
  git add src/lib.rs python/ tests/test_correctness.py
  git commit -m "feat: expose auction_assign and solve Python API over PyO3 and numpy"
  ```

---

### Task 5: Differential & Property-Based Verification vs. SciPy (`tests/test_differential.py`)

**Files:**
- Create: `tests/test_differential.py`

**Interfaces:**
- Consumes: `auction_assign`, `solve`, `InfeasibleError` from `auction_assign`.

- [ ] **Step 1: Write differential and Hypothesis tests in `tests/test_differential.py`**
  - `test_matches_scipy_dense_random` (Hypothesis over `1..=25 x 1..=25` matrices with positive, negative, zero, and mixed-scale `float64` values, both `maximize=False` and `maximize=True`):
    - Verify `row_ind` strictly increasing, `col_ind` unique, `len == min(m, n)`, `auction_assign` matches `solve` indices, and `abs(res.cost - scipy_cost) <= min(m, n) * res.info.final_eps + 1e-8`.
  - `test_matches_scipy_gated_inf` (Hypothesis over `1..=15 x 1..=15` matrices with random `+inf`/`-inf` entries):
    - Compare against `scipy.optimize.linear_sum_assignment`: when SciPy raises `ValueError`, `auction_assign` must raise `InfeasibleError`; when SciPy succeeds, total cost must match within tolerance and no matched edge may be infinite.
  - `test_cost_limit_matches_bruteforce_oracle` (Hypothesis over small `1..=5 x 1..=5` matrices with random `inf` and finite `cost_limit`, both `maximize=False` and `maximize=True`):
    - Compare against an exact Python brute-force enumeration of all valid partial matchings: assert matched costs respect strict `< cost_limit` (or `> cost_limit`), and objective sum `(L - c)` / `(c - L)` matches optimal within `len(row_ind) * res.info.final_eps + 1e-9`.

- [ ] **Step 2: Run full test & lint suite**
  - Run: `uv run pytest -v` -> Expected: PASS across all tests & Hypothesis examples.
  - Run: `uv run ruff check . && uv run mypy python tests` -> Expected: PASS.

- [ ] **Step 3: Commit**
  ```bash
  git add tests/test_differential.py
  git commit -m "test: add Hypothesis differential tests against SciPy and exact partial oracle"
  ```

---

### Task 6: Documentation and GitHub Actions CI & PyPI Trusted Publishing

**Files:**
- Create: `README.md`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`

**Interfaces:**
- Consumes: `pyproject.toml`, `Cargo.toml`, `auction_assign` public API.

- [ ] **Step 1: Write `README.md`**
  - Include overview, installation (`uv add auction-assign` / `pip install auction-assign`), track-to-track fusion quickstart showing `auction_assign` (drop-in for `linear_sum_assignment`) and `solve` (with `inf` gating, `cost_limit`, `AuctionResult`/`AuctionInfo`, `InfeasibleError`, `ConvergenceError`), and local development commands (`uv sync`, `uv run pytest`, `cargo test`, `uv run ruff check .`, `uv run mypy python tests`).

- [ ] **Step 2: Write `.github/workflows/ci.yml`**
  - Triggers: `push` (`main`), `pull_request`.
  - Jobs:
    - `rust`: `ubuntu-latest` running `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.
    - `python`: matrix `[ubuntu-latest, windows-latest, macos-latest]` with `astral-sh/setup-uv@v5` + `dtolnay/rust-toolchain@stable` running `uv sync`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy python tests`, `uv run pytest`.

- [ ] **Step 3: Write `.github/workflows/release.yml`**
  - Triggers: `push` tags `v*`, `workflow_dispatch`.
  - Build matrix using `PyO3/maturin-action@v1` for:
    - Linux: `x86_64`, `aarch64` (`manylinux: "2_28"`, `zig: true` or native)
    - macOS: `x86_64`, `aarch64`
    - Windows: `x64`
    - `sdist` (`command: sdist`)
  - Test job installing built wheels via `uv` and running `uv run --no-project --with pytest --with numpy pytest` (or smoke import + test).
  - Publish job (`needs: [build-wheels, build-sdist]`, `permissions: id-token: write`, `environment: release`) using `pypa/gh-action-pypi-publish@release/v1` for OIDC Trusted Publishing (note in `README.md` to configure the `release` environment / PyPI Trusted Publisher in GitHub + PyPI settings).

- [ ] **Step 4: Run final end-to-end verification**
  - Run: `cargo fmt --check` -> PASS.
  - Run: `cargo clippy --all-targets -- -D warnings` -> PASS.
  - Run: `cargo test` -> PASS.
  - Run: `uv sync` -> PASS.
  - Run: `uv run ruff check . && uv run ruff format --check .` -> PASS.
  - Run: `uv run mypy python tests` -> PASS.
  - Run: `uv run pytest -v` -> PASS.
  - Run: `uv build` -> builds wheel (`*cp310-abi3*.whl`) and `.tar.gz` sdist cleanly in `dist/` (clean up `dist/` afterwards or leave gitignored).

- [ ] **Step 5: Commit**
  ```bash
  git add README.md .github/workflows/ci.yml .github/workflows/release.yml
  git commit -m "docs,ci: add README, CI workflow, and PyPI trusted publishing release workflow"
  ```

---

### Task 7: Reproducible Benchmarks vs. SciPy (`benchmarks/bench_assign.py`)

**Files:**
- Create: `benchmarks/bench_assign.py`
- Modify: `README.md` (document benchmark command)

**Interfaces:**
- Consumes: `auction_assign`, `solve` from `auction_assign`; `scipy.optimize.linear_sum_assignment`.

- [ ] **Step 1: Write `benchmarks/bench_assign.py`**
  - Benchmark dense random matrices (`N = 20, 50, 100, 250, 500`, plus rectangular `50x200`) and sparse-gated track-fusion matrices (90% `inf` entries, `N = 50, 100, 250, 500`), timing `auction_assign` vs. `scipy.optimize.linear_sum_assignment` (with `1e15` sentinel for gated runs on SciPy) using `time.perf_counter()`, warmup runs, median/min ms, and relative cost gap check (`abs(auction_cost - scipy_cost) <= tol`).
  - Support `--quick` flag for fast verification and default mode for full table output.

- [ ] **Step 2: Run benchmark and verify types/lint**
  - Run: `uv run python benchmarks/bench_assign.py` -> prints timing table, all optimality gap checks pass.
  - Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy python tests benchmarks` -> PASS.

- [ ] **Step 3: Document benchmark command in `README.md` and commit**
  ```bash
  git add benchmarks/bench_assign.py README.md
  git commit -m "perf: add benchmark script comparing auction-assign with scipy"
  ```
