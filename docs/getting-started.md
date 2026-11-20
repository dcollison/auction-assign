# Getting Started

## Requirements

* Python `>= 3.10` (pre-built `abi3` wheels available for Linux, macOS, and Windows)
* NumPy `>= 1.24`

## Installation

Install using `uv` (recommended):

```bash
uv add auction-assign
```

Or with `pip`:

```bash
pip install auction-assign
```

## Basic Usage

### Drop-in `linear_sum_assignment` Replacement

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

`row_ind` is always returned in strictly ascending order, matching `scipy.optimize.linear_sum_assignment`.

### Structured Result & Diagnostics

Use `solve(...)` when you want the exact objective total and solver telemetry without recomputing array sums:

```python
from auction_assign import solve

result = solve(cost_matrix)
print(result.row_ind, result.col_ind)
print("Total cost:", result.cost)
print("Iterations:", result.info.iterations)
print("Scaling rounds:", result.info.scaling_rounds)
print("Final epsilon:", result.info.final_eps)
```

### Maximization

Pass `maximize=True` to maximize total score instead of minimizing cost:

```python
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
