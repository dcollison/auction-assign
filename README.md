# auction-assign

Fast Bertsekas epsilon-scaling forward-reverse auction algorithm for dense linear sum assignment in Rust with Python bindings.

## Features

- **Drop-in replacement for `scipy.optimize.linear_sum_assignment`**: `auction_assign(cost_matrix)` returns `(row_ind, col_ind)` as sorted `np.intp` NumPy arrays for square and rectangular (`m x n`) matrices.
- **Native `cost_limit` partial assignment**: Supports unassigned-slack gating (`cost < cost_limit` when minimizing, `cost > cost_limit` when maximizing) so non-beneficial pairs stay unmatched without quadratic padding.
- **Hard edge gating (`inf` / `-inf`)**: Exclude impossible associations with `+inf` (minimization) or `-inf` (maximization), raising `InfeasibleError` if no complete matching exists when `cost_limit=None`.
- **Rich solver telemetry**: `solve()` returns an immutable `AuctionResult` with exact total `cost` and `AuctionInfo` (`iterations`, `scaling_rounds`, `final_eps`, `row_prices`, `col_prices`).
- **Releases the GIL**: Core Rust computation runs without holding the Python GIL, scaling across threads with `ThreadPoolExecutor`.
- **Typed Python API**: Includes `py.typed` and passes `mypy --strict`.

## Installation

Requires Python 3.10+ and NumPy 1.24+.

```bash
uv add auction-assign
```

Or with `pip`:

```bash
pip install auction-assign
```

## Quickstart

### Drop-in `linear_sum_assignment` usage

```python
import numpy as np
from auction_assign import auction_assign

cost_matrix = np.array(
    [
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
    ]
)
row_ind, col_ind = auction_assign(cost_matrix)
total_cost = float(cost_matrix[row_ind, col_ind].sum())
print(row_ind, col_ind, total_cost)
# [0 1 2] [1 0 2] 5.0
```

### Track-to-track fusion with gating, `cost_limit`, and diagnostics

In multi-target tracking and sensor fusion, impossible associations are masked with `np.inf` and associations worse than a gate threshold `cost_limit` should remain unassigned rather than forcing bad matches:

```python
import numpy as np
from auction_assign import ConvergenceError, InfeasibleError, solve

# Pairwise track-to-measurement association costs (3 tracks x 4 measurements).
# np.inf forbids impossible pairs; track 2 has no pair below cost_limit=10.0.
costs = np.array(
    [
        [1.2, 8.5, np.inf, 9.0],
        [np.inf, 2.4, 3.1, np.inf],
        [12.0, 15.0, np.inf, 18.0],
    ]
)

try:
    result = solve(costs, cost_limit=10.0)
except InfeasibleError as exc:
    raise RuntimeError("No complete gated assignment exists") from exc
except ConvergenceError as exc:
    raise RuntimeError(
        f"Solver stopped at iteration {exc.iterations} (eps={exc.final_eps:.3e})"
    ) from exc

print("Matched tracks:", result.row_ind)  # [0 1]
print("Matched meas:  ", result.col_ind)  # [0 1]
print(f"Total cost:    {result.cost:.2f}")  # 3.60
print("Iterations:    ", result.info.iterations)
print("Scaling rounds:", result.info.scaling_rounds)
print("Final epsilon: ", result.info.final_eps)
```

For maximization problems (e.g., log-likelihood or affinity scores), pass `maximize=True`, use `-np.inf` to forbid pairs, and set `cost_limit` as a strict lower bound (`score > cost_limit`):

```python
import numpy as np
from auction_assign import solve

scores = np.array(
    [
        [10.0, -np.inf, 1.0],
        [-np.inf, 4.0, -2.0],
    ]
)
res = solve(scores, maximize=True, cost_limit=0.0)
print(res.row_ind, res.col_ind, res.cost)
# [0 1] [0 1] 14.0
```

## API Reference

- **`auction_assign(cost_matrix, *, maximize=False, cost_limit=None, eps_min=None, eps_factor=5.0, max_iterations=None) -> tuple[NDArray[np.intp], NDArray[np.intp]]`**
  Solves the 2D linear sum assignment problem and returns `(row_ind, col_ind)` with `row_ind` in strictly ascending order.
- **`solve(cost_matrix, *, maximize=False, cost_limit=None, eps_min=None, eps_factor=5.0, max_iterations=None) -> AuctionResult`**
  Same parameters as `auction_assign`, returning an `AuctionResult`.
- **`AuctionResult`** (frozen dataclass): `row_ind`, `col_ind`, `cost: float`, `info: AuctionInfo`.
- **`AuctionInfo`** (frozen dataclass): `iterations: int`, `scaling_rounds: int`, `final_eps: float`, `row_prices: NDArray[np.float64]`, `col_prices: NDArray[np.float64]`.
- **`InfeasibleError(ValueError)`**: Raised when `cost_limit is None` and a matrix gated with `inf` (or `-inf` when `maximize=True`) has no complete assignment of size `min(n_rows, n_cols)`.
- **`ConvergenceError(RuntimeError)`**: Raised when `max_iterations` is exceeded or IEEE-754 `f64` price updates stall; provides `.iterations` and `.final_eps`.

Note: Pairs with `cost == cost_limit` are excluded (strict `< cost_limit` when minimizing, strict `> cost_limit` when maximizing). `NaN`, `-inf` during minimization, and `+inf` during maximization raise `ValueError`.

## Benchmarks

Compare `auction_assign` against `scipy.optimize.linear_sum_assignment` on dense random matrices (`N = 20, 50, 100, 250, 500`, plus `50x200`) and feasible sparse-gated track-fusion matrices (90% `inf` entries, with a `1e15` sentinel for SciPy):

```bash
# Full benchmark table with median/min timings and optimality gap checks
uv run python benchmarks/bench_assign.py

# Fast verification run (fewer warmup/repeat iterations)
uv run python benchmarks/bench_assign.py --quick
```

Both modes verify solution optimality against SciPy for every benchmark case (`PASS`/`FAIL`).

## Local Development

```bash
# Create/sync virtualenv and build the Rust extension
uv sync

# Run Python test suite
uv run pytest -v

# Run Rust checks and unit tests
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test

# Run Python lint, format check, and strict type checking
uv run ruff check .
uv run ruff format --check python tests benchmarks
uv run mypy python tests benchmarks

# Build wheel and sdist into dist/
uv build
```

## Releasing (PyPI Trusted Publishing)

Releases are automated by [`.github/workflows/release.yml`](.github/workflows/release.yml) using PyPI OIDC Trusted Publishing (`id-token: write` with `pypa/gh-action-pypi-publish@release/v1`), so no long-lived PyPI API tokens are stored in repository secrets.

One-time repository and PyPI setup:

1. In your GitHub repository, go to **Settings → Environments** and create an environment named `release` (optionally configure required reviewers or branch/tag protection rules).
2. On [PyPI](https://pypi.org/), go to **Your Account → Publishing** (or your project's **Publishing** settings) and add a **GitHub** Trusted Publisher with:
   - **Owner**: `dcollison`
   - **Repository name**: `auction-assign`
   - **Workflow name**: `release.yml`
   - **Environment name**: `release`
3. Tag a release matching `v*` and push it to GitHub:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
   You can also test wheel builds and smoke tests without publishing by triggering the **Release** workflow manually via `workflow_dispatch`.

## License

MIT — see [LICENSE](LICENSE).
