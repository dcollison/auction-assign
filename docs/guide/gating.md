# Gating & Partial Assignment

In multi-target tracking and sensor fusion, association matrices commonly combine hard kinematic gates (`np.inf` / `-np.inf`) with an optional maximum acceptable association cost (`cost_limit`).

## Hard Edge Gating (`inf` / `-inf`)

* **Minimization (`maximize=False`)**: set forbidden pairs to `+np.inf`. Passing `-np.inf` raises `ValueError`.
* **Maximization (`maximize=True`)**: set forbidden pairs to `-np.inf`. Passing `+np.inf` raises `ValueError`.
* **`NaN`**: always raises `ValueError`.

When `cost_limit=None`, if no complete matching of size $\min(m, n)$ exists over finite entries, `auction-assign` checks bipartite feasibility and raises `InfeasibleError` (a subclass of `ValueError`).

## Native `cost_limit` Partial Assignment

When `cost_limit` is set, tracks or measurements without an acceptable partner may remain unmatched rather than forcing high-cost associations:

```python
import numpy as np
from auction_assign import ConvergenceError, InfeasibleError, solve

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
```

!!! note "Strict Boundary Semantics"
    Pairs with `cost == cost_limit` are **excluded**: matching requires `cost < cost_limit` when minimizing and `cost > cost_limit` when maximizing.
