# Algorithm & Formulation

`auction-assign` implements **Bertsekas's forward–reverse auction algorithm** with automatic $\varepsilon$-scaling in pure Rust (`src/solver.rs`).

## Problem Formulation

Given an $m \times n$ real matrix $C$, `auction-assign` finds a set of row-column pairs $M \subseteq \{0, \dots, m-1\} \times \{0, \dots, n-1\}$ such that each row and column appears at most once:

* **Complete assignment (`cost_limit=None`)**: $|M| = \min(m, n)$, minimizing $\sum_{(i, j) \in M} C_{ij}$ (or maximizing when `maximize=True`).
* **Partial assignment (`cost_limit` provided)**: pairs may remain unassigned; each matched pair must satisfy $C_{ij} < \text{cost\_limit}$ when minimizing (or $C_{ij} > \text{cost\_limit}$ when maximizing).

## $\varepsilon$-Scaling & Convergence

To achieve strong empirical runtime across both well-conditioned and symmetric/nearly-tied cost matrices, the solver:

1. Determines a problem scale from finite entries and initializes $\varepsilon$ using `eps_factor` (default `5.0`) down toward `eps_min` (automatic by default, or user-configured).
2. Alternates forward and reverse auction rounds while tightening $\varepsilon \leftarrow \max(\varepsilon_{\min}, \varepsilon / \text{eps\_factor})$.
3. Enforces a deterministic iteration budget (`max_iterations`) and checks for IEEE-754 `f64` ULP price stagnation (`p_new == p_old`), raising `ConvergenceError` rather than looping indefinitely.

## Sparse & SIMD Execution Paths

* **Dense 4-way SIMD scan**: For dense matrices, best/second-best bid discovery uses a 4-way unrolled scan over contiguous row slices.
* **CSR sparse path**: For heavily gated matrices (high `inf` / `-inf` density), feasible edges are compacted into a Compressed Sparse Row (CSR) structure to avoid scanning forbidden cells.

## Dual Prices & GIL Release

* `solve(...)` exposes final dual variables `row_prices` and `col_prices` on `AuctionInfo`.
* Both `auction_assign(...)` and `solve(...)` detach from the Python GIL inside the `_core` PyO3 extension (`py.detach(...)`), allowing concurrent scaling with `ThreadPoolExecutor`.
