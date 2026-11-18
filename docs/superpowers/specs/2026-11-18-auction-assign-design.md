# Design Document: `auction-assign`

**Date:** 2026-11-18  
**Status:** Draft (pending user approval)  
**Package / Repo Name:** `auction-assign` (Python import: `auction_assign`)  
**License:** MIT  

---

## 1. Intent & Why

Build a fast, ergonomic, Rust-accelerated Python package (`auction-assign`) implementing **Bertsekas's forward–reverse auction algorithm** with automatic $\varepsilon$-scaling for the 2D linear assignment problem.

The primary driver is **track-to-track fusion**, where assignment solvers such as `scipy.optimize.linear_sum_assignment` and `lap.lapjv` are already used and an auction-based solver completes the solver lineup with matching ergonomics, validation-gate handling (`inf` / `cost_limit`), and dual price / diagnostic inspection.

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- Provide a drop-in `(row_ind, col_ind)` API (`auction_assign`) compatible with `scipy.optimize.linear_sum_assignment`, plus a structured `solve(...)` API returning `AuctionResult` (cost and solver diagnostics `AuctionInfo` with iterations, scaling rounds, final $\varepsilon$, and dual prices) without return-type-switching boolean flags.
- Implement Bertsekas's forward–reverse auction algorithm with automatic $\varepsilon$-scaling in pure Rust (`src/solver.rs`, zero PyO3 dependency), supporting:
  - Rectangular matrices (`M != N`)
  - Minimization (`maximize=False`) and maximization (`maximize=True`)
  - Negative and zero costs
  - Forbidden / gated pairs (`+inf` when `maximize=False`, `-inf` when `maximize=True`)
  - Partial / gated assignments via `cost_limit` (allowing rows/columns to remain unassigned)
  - Strict `NaN`, infeasibility (`InfeasibleError(ValueError)`), and convergence-budget (`max_iterations` / `ConvergenceError(RuntimeError)`) handling
  - Configurable `eps_min`, `eps_factor`, and `max_iterations` overrides
- Release the Python GIL during the Rust computation (`py.detach`).
- Use `uv` across Python dependency management, local builds, linting, type checking, and tests.
- Ship cross-platform `abi3` wheels (`Python >= 3.10`) and `sdist` to PyPI via GitHub Actions OIDC Trusted Publishing on `v*` tags.

### Non-Goals (v1)
- Sparse CSR/COO matrix inputs (`scipy.sparse`) — v1 targets dense 2D `numpy.ndarray` / array-like inputs.
- General network flow, min-cost max-flow, or multi-frame / MHT tracking logic.
- Publishing `auction-assign` as a standalone crate on `crates.io` (though `src/solver.rs` is kept PyO3-free so splitting later is trivial).
- Runtime GPU or multi-node distributed execution.

### Anti-Goals
- Silent wrong matches on infeasible gated matrices (must detect infeasibility deterministically and raise `ValueError`, never spin indefinitely or return invalid pairs).
- Silent `NaN` propagation.
- Breaking `cargo test` by unconditionally enabling `pyo3/extension-module` in default Cargo features.

---

## 3. Architecture (Approach 1: Mixed Rust/Python via `maturin`)

```text
auction-assign/
├── Cargo.toml
├── build.rs                       # pyo3_build_config::use_pyo3_cfgs() for abi3 + numpy
├── pyproject.toml                 # PEP 517 maturin backend + uv + ruff + mypy/pyright + pytest
├── README.md
├── LICENSE                        # MIT
├── .python-version                # 3.12 (local default; >=3.10 supported)
├── .gitignore
├── src/
│   ├── lib.rs                     # Thin PyO3 + rust-numpy bindings (`_core`)
│   └── solver.rs                  # Pure-Rust auction solver + feasibility check + unit tests
├── python/
│   └── auction_assign/
│       ├── __init__.py            # Public API: auction_assign, solve, AuctionResult, AuctionInfo, errors
│       ├── _core.pyi              # Type stubs for Rust extension module
│       └── py.typed               # PEP 561 marker
├── tests/
│   ├── test_correctness.py        # Edge cases, gating, cost_limit, validation, exceptions
│   └── test_differential.py       # Property/differential tests vs scipy.optimize.linear_sum_assignment
└── .github/
    └── workflows/
        ├── ci.yml                 # Rust fmt/clippy/test + Python ruff/typecheck/pytest
        └── release.yml            # Build abi3 wheels + sdist + publish to PyPI via OIDC
```

### Boundary Responsibilities
- **Python layer (`python/auction_assign/__init__.py`):**
  - Coerces input via `np.asarray(..., dtype=np.float64)` and ensures C-contiguous memory (`np.ascontiguousarray`), validating `ndim == 2`.
  - Validates Python-level parameters and wraps Rust outputs into clean typed objects (`AuctionResult`, `AuctionInfo`).
- **FFI layer (`src/lib.rs`):**
  - Exposes `solve_dense` in `auction_assign._core` plus `InfeasibleError` / `ConvergenceError` mapping (subclasses of `ValueError` / `RuntimeError`).
  - Borrows `PyReadonlyArray2<'_, f64>` as `&[f64]`, detaches from the Python interpreter (`py.detach(...)`), runs `solver::solve_dense(...)`, and returns NumPy arrays (`IntoPyArray`) + scalars/price vectors.
- **Pure-Rust core (`src/solver.rs`):**
  - Zero `pyo3` or `numpy` dependencies.
  - Self-contained algorithm, error enum (`SolverError`), and exhaustive Rust unit tests (`cargo test`).

---

## 4. Public Python API

### Functions
```python
def auction_assign(
    cost_matrix: ArrayLike,
    *,
    maximize: bool = False,
    cost_limit: float | None = None,
    eps_min: float | None = None,
    eps_factor: float = 5.0,
    max_iterations: int | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    ...
```
Always returns `(row_ind, col_ind)` — sorted by `row_ind`, dtype `np.intp`, drop-in compatible with `scipy.optimize.linear_sum_assignment`.

```python
def solve(
    cost_matrix: ArrayLike,
    *,
    maximize: bool = False,
    cost_limit: float | None = None,
    eps_min: float | None = None,
    eps_factor: float = 5.0,
    max_iterations: int | None = None,
) -> AuctionResult:
    ...
```
Always returns `AuctionResult` when full diagnostics, total cost, or dual prices are needed.

### Data Types & Exceptions
```python
@dataclass(frozen=True, slots=True)
class AuctionInfo:
    iterations: int
    scaling_rounds: int
    final_eps: float
    row_prices: NDArray[np.float64]
    col_prices: NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class AuctionResult:
    row_ind: NDArray[np.intp]
    col_ind: NDArray[np.intp]
    cost: float
    info: AuctionInfo

class InfeasibleError(ValueError):
    """Raised when a gated cost matrix has no valid complete assignment."""

class ConvergenceError(RuntimeError):
    """Raised when the solver hits `max_iterations` or floating-point price stagnation."""
    iterations: int
    final_eps: float
```

---

## 5. Mathematical & Algorithmic Design (`src/solver.rs`)

### 5.1 Validation Rules
- `cost_matrix` must be 2D (`ndim == 2`); otherwise raise `ValueError`.
- Any `NaN` in `cost_matrix` raises `ValueError`.
- Unbounded-benefit infinities (`-inf` when `maximize=False`, `+inf` when `maximize=True`) raise `ValueError`.
- `cost_limit`:
  - `NaN` raises `ValueError`.
  - When `maximize=False`, `cost_limit = +inf` (or `None`) means complete assignment; finite `L` keeps only pairs with `cost < L` (pairs with `cost >= L` are treated as ineligible, matching `lap.lapjv` cutoff semantics; if `cost_limit = -inf`, empty assignment is returned immediately).
  - When `maximize=True`, `cost_limit = -inf` (or `None`) means complete assignment; finite `L` keeps only pairs with `score > L` (if `cost_limit = +inf`, empty assignment is returned immediately).
- `eps_min`: if provided, must be finite and `> 0.0` (else `ValueError`).
- `eps_factor`: must be finite and `> 1.0` (else `ValueError`).
- `max_iterations`: if provided, must be `> 0` (else `ValueError`). When `None`, defaults to a generous safety cap (`max(100_000, 500 * m * n)` total bid steps across all scaling rounds).
- Empty matrix (`rows == 0` or `cols == 0`): immediately returns empty `row_ind`/`col_ind`, `cost = 0.0`, and zeroed `AuctionInfo`.

### 5.2 Failure Modes (No Solution / Non-Convergence)
1. **Graph infeasibility (before bidding):** Whenever `cost_limit` is unbounded (`None`) and any forbidden `inf` entry is present, an unweighted maximum bipartite matching check runs upfront. If maximum matching size `< min(rows, cols)`, `InfeasibleError` (a subclass of `ValueError`) is raised immediately.
2. **Iteration budget / IEEE-754 stagnation (during bidding):** If total bid steps exceed `max_iterations`, or if a price/profit update produces zero `f64` progress (`p_new == p_old` because `eps` fell below ULP at the current price magnitude), the solver halts cleanly and raises `ConvergenceError` (a subclass of `RuntimeError`, carrying `.iterations` and `.final_eps`) rather than looping or silently returning an invalid/incomplete assignment.

### 5.3 Complete Assignment Mode (`cost_limit` is unbounded / `None`)
1. **Orientation:** If `rows > cols`, transpose internally so `m <= n` (`m` rows, `n` columns), then swap `row_ind`/`col_ind` and `row_prices`/`col_prices` on return.
2. **Feasibility check (when any forbidden `inf` exists):**
   - Quick check that every row has at least one finite edge.
   - Run an unweighted maximum bipartite matching (Hopcroft–Karp / Kuhn augmenting paths) on finite edges. If maximum matching size `< m`, return `Err(SolverError::Infeasible)` (`InfeasibleError`). This guarantees the auction loop never diverges on Hall-condition failures.
3. **Benefit conversion & scaling:**
   - Convert finite entries to internal benefits $b_{ij} = c_{ij}$ (`maximize=True`) or $b_{ij} = -c_{ij}$ (`maximize=False`). Forbidden entries are `f64::NEG_INFINITY`.
   - Compute `scale`: difference `max(b) - min(b)` over finite entries, or `max(|b|)` if all finite entries are equal (fallback `1.0` if all finite entries are `0.0`).
   - Default `eps_min`: `max(1e-12, scale * 1e-9)` (prevents `f64` ULP stalling when `p_j + eps == p_j` on large costs while preserving ~9 digits of relative optimality on small costs).
   - Initial $\varepsilon$: `max(scale / 2.0, eps_min)` (if `scale == 0.0`, single round at `eps_min`).
4. **Forward–Reverse Auction Loop per $\varepsilon$-scaling round:**
   - Maintain row duals `pi[i]` (`row_prices`), column duals `p[j]` (`col_prices`), and assignments `row_to_col[i]` / `col_to_row[j]`.
   - **Forward step** (from unassigned rows): find best column $j_1 = \arg\max_j (b_{ij} - p_j)$ and second-best value $w_i = \max_{j \ne j_1} (b_{ij} - p_j)$ (if a row has only 1 feasible column, bid increment uses a large finite fallback $\Delta_{\max}$ or $\varepsilon$ directly without overflow). Set $\pi_i = w_i - \varepsilon$, $p_{j_1} = b_{ij_1} - \pi_i$, and assign $(i, j_1)$.
   - **Reverse step:**
     - **Square (`m == n`):** symmetric forward–reverse (unassigned columns bid on rows, raising $\pi_i$ and lowering $p_j$, alternating forward and reverse passes until all rows and columns are matched).
     - **Rectangular (`m < n`):** forward pass assigns all $m$ rows; if any unassigned column has $p_j > \lambda$ (where $\lambda = \min_{k \text{ assigned}} p_k$, which can happen across warm-started scaling rounds), run reverse bids on unassigned columns above $\lambda$ to equalize/lower their prices to $\le \lambda$ while maintaining $\varepsilon$-CS, ensuring the Bertsekas–Castañon asymmetric optimality condition holds.
   - Repeat scaling rounds with `eps = max(eps / eps_factor, eps_min)` until `eps == eps_min` completes.
   - **Total cost:** Computed directly by summing the original `cost_matrix[r, c]` values over the final matched pairs (exact float sum of input costs, free of any scaling drift).

### 5.4 Partial Assignment Mode (finite `cost_limit = L`)
1. Convert eligible pairs ($c_{ij} < L$ when minimizing, $c_{ij} > L$ when maximizing) to positive net benefits:
   - Minimizing: $b_{ij} = L - c_{ij} > 0$.
   - Maximizing: $b_{ij} = c_{ij} - L > 0$.
   - All ineligible or `inf` pairs have $b_{ij} = -\infty$ (never selected).
   - If no pair is eligible, immediately return empty assignment (`cost = 0.0`).
2. **Native unassigned-slack handling (no $2\times$ matrix padding):**
   - Every row $i$ and column $j$ implicitly has an unassigned option with benefit $0$ and dual floor $0$ ($\pi_i \ge 0$, $p_j \ge 0$).
   - In a forward bid from row $i$: compute best real column net value $v_1 = \max_j (b_{ij} - p_j)$ and second-best $w_i = \max(\max_{j \ne j_1} (b_{ij} - p_j), 0.0)$.
     - If $v_1 > 0.0$ (and $v_1 \ge \varepsilon$ or strictly beats $0$ — specifically, if $v_1 > 0$ and the row prefers $j_1$ over staying unassigned at $0$), bid $p_{j_1} \leftarrow b_{ij_1} - \max(w_i - \varepsilon, 0.0)$ and $\pi_i \leftarrow \max(w_i - \varepsilon, 0.0)$? Wait — careful: to keep $\pi_i \ge 0$ and $\varepsilon$-CS ($\pi_i + p_j \ge b_{ij} - \varepsilon$ for all $j$, including $j_1$'s old value), set $\text{bid} = v_1 - \max(w_i, 0.0) + \varepsilon$, $p_{j_1} \leftarrow p_{j_1} + \text{bid}$, and $\pi_i \leftarrow \max(w_i - \varepsilon, 0.0)$ (note: when $w_i \le 0$, $\pi_i$ stays $0.0$ and $p_{j_1} = b_{ij_1} - 0.0$? Wait: if $p_{j_1}$ rises by $v_1 - 0 + \varepsilon = b_{ij_1} - p_{j_1}^{\text{old}} + \varepsilon$, then $p_{j_1}^{\text{new}} = b_{ij_1} + \varepsilon$, so $\pi_i + p_{j_1}^{\text{new}} = b_{ij_1} + \varepsilon$? No: if $j_1$ was unassigned at $p_{j_1} = 0$, raising $p_{j_1}$ by $v_1 + \varepsilon$ would overshoot $b_{ij_1}$!).
     - **Exact formulation left clean for the algorithm:** Represent each row $i$'s unassigned option with a private dummy column of fixed price $0$ and benefit $0$ (so in forward bid, $0.0$ is simply another candidate in $\{b_{ij} - p_j\}$ alongside all real columns $j$):
       - If $v_1 \le 0.0$: row $i$ stays unassigned ($\pi_i = 0.0$).
       - If $v_1 > 0.0$: let $w_i = \max(0.0, \max_{j \ne j_1}(b_{ij} - p_j))$. Set $\pi_i = w_i - \varepsilon$ (Wait: if $w_i = 0.0$, $\pi_i = -\varepsilon$ or clamp? If dummy column is fixed at price $0$, $\pi_i$ can drop to $-\varepsilon$ in the round and we only require $\pi_i \ge -\varepsilon$, OR equivalently set dummy benefit $= \varepsilon$ so $\pi_i \ge 0$!).
       - Specifically, **reverse step** with `cost_limit`: unassigned columns with $p_j > 0$ run reverse bids against rows (using row unassigned floor $0.0$), lowering $p_j$ back toward $0.0$ when no row yields $b_{ij} - \pi_i > 0$.
       - The invariant $\pi_i \ge -\varepsilon$, $p_j \ge 0$, $p_j = 0$ if $j$ unassigned, $\pi_i = 0$ if $i$ unassigned, and $\pi_i + p_j \ge b_{ij} - \varepsilon$ guarantees the partial matching is within $|S^*| \cdot \varepsilon$ of optimal.

---

## 6. Build, Tooling, and CI/CD (`uv` + GitHub Actions)

### Python & Rust Packaging
- **Build backend:** `maturin` (`[build-system] requires = ["maturin>=1.7,<2.0"]`, `build-backend = "maturin"`).
- **Target Python:** `requires-python = ">=3.10"`, compiled with `pyo3` `abi3-py310` feature (single wheel per OS/arch works across Python 3.10–3.14+).
- **Runtime dependency:** `numpy>=1.24`.
- **`uv` configuration (`pyproject.toml`):**
  - `[dependency-groups]` `dev` includes `pytest`, `hypothesis`, `scipy`, `ruff`, `mypy` (or `ty`), `maturin`.
  - `[tool.uv]` `cache-keys` watches `pyproject.toml`, `Cargo.toml`, `Cargo.lock`, and `src/**/*.rs` so `uv run` / `uv sync` automatically rebuilds the Rust extension whenever Rust sources change.
- **Quality gates:**
  - `ruff check` + `ruff format --check`
  - Strict static type checking (`mypy` / `ty` against `.pyi` stubs and `py.typed`)
  - `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`
  - `uv run pytest` (unit tests, edge cases, Hypothesis property & differential tests comparing against `scipy.optimize.linear_sum_assignment` on complete matrices and brute-force / dummy-padded SciPy on gated & `cost_limit` matrices)

### GitHub Actions
1. **CI (`ci.yml`):** Triggers on `push` to `main` and `pull_request`. Runs Rust checks (`fmt`, `clippy`, `cargo test`) and Python checks (`uv sync`, `ruff`, type check, `pytest`) across Ubuntu, Windows, and macOS.
2. **Release (`release.yml`):** Triggers on `v*` tags (and `workflow_dispatch`).
   - Uses `PyO3/maturin-action` to build `abi3` wheels for:
     - Linux `x86_64` and `aarch64` (`manylinux`)
     - macOS `x86_64` and `aarch64`
     - Windows `x64`
     - `sdist`
   - Smoke-tests built wheels (`pytest`).
   - Publishes all artifacts to PyPI using OIDC Trusted Publishing (`permissions: id-token: write`, `pypa/gh-action-pypi-publish@release/v1` or `uv publish --trusted-publishing always`).

---

## 7. Left to the Builder (Implementation Details)
- Exact `pyo3` and `numpy` crate minor versions compatible with `rustc 1.87.0` (e.g., `pyo3` 0.24–0.30 / matching `numpy` crate) verified via `cargo check` / `uv sync`.
- Exact internal data structures in `src/solver.rs` (flat `Vec<f64>` row-major indexing, `Vec<usize>` queue/stack for unassigned rows/cols, and Hopcroft–Karp / Kuhn feasibility helper).
- Exact tie-breaking and the default formula for `max_iterations` when `None` is passed.
- Detailed docstrings and `README.md` usage examples (including track-to-track fusion example).
