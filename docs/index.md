# auction-assign

**Fast Bertsekas epsilon-scaling forward-reverse auction algorithm for linear sum assignment in Rust with Python bindings.**

`auction-assign` provides a drop-in replacement for `scipy.optimize.linear_sum_assignment` alongside native `cost_limit` partial assignment, hard `inf`/`-inf` edge gating, GIL-free Rust execution, and dual-price solver diagnostics for multi-target tracking and sensor fusion.

---

## Key Capabilities

* **Drop-in `linear_sum_assignment` API:** `auction_assign(cost_matrix)` returns `(row_ind, col_ind)` as sorted `np.intp` NumPy arrays for square and rectangular (`m x n`) matrices.
* **Native `cost_limit` Partial Assignment:** Supports unassigned-slack gating (`cost < cost_limit` when minimizing, `cost > cost_limit` when maximizing) without quadratic padding.
* **Hard Edge Gating (`inf` / `-inf`):** Excludes impossible pairs cleanly and raises `InfeasibleError` when no complete matching exists with `cost_limit=None`.
* **CSR Sparse Path & SIMD Scanning:** Automatically switches to a compressed sparse row (CSR) path for heavily gated matrices and uses a 4-way SIMD scan in the bidding loop.
* **Solver Telemetry & Dual Prices:** `solve(...)` returns an immutable `AuctionResult` with exact total `cost` and `AuctionInfo` (`iterations`, `scaling_rounds`, `final_eps`, `row_prices`, `col_prices`).
* **GIL-Free Execution:** Core Rust computation releases the Python GIL (`py.detach`), scaling cleanly across threads.
* **Strict Static Typing:** Ships `py.typed` and `_core.pyi` stubs and passes `mypy --strict`.

---

## Quick Example

```python
import numpy as np
from auction_assign import auction_assign, solve

cost_matrix = np.array(
    [
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
    ]
)

row_ind, col_ind = auction_assign(cost_matrix)
print(row_ind, col_ind)  # [0 1 2] [1 0 2]

result = solve(cost_matrix)
print(result.cost, result.info.iterations, result.info.scaling_rounds)
```
