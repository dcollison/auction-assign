# Benchmarks & Performance

`auction-assign` includes a reproducible benchmark script at [`benchmarks/bench_assign.py`](https://github.com/dcollison/auction-assign/blob/main/benchmarks/bench_assign.py) comparing against `scipy.optimize.linear_sum_assignment`.

## Benchmark Scenarios

The suite evaluates:

* **Dense uniform random matrices**: square shapes (`20x20`, `50x50`, `100x100`, `250x250`, `500x500`) and rectangular (`50x200`).
* **Gated 90% sparse fusion matrices**: feasible matrices where 90% of off-assignment entries are set to `np.inf` (with a `1e15` finite sentinel for SciPy).

Every benchmark case verifies solution optimality against SciPy (`PASS`/`FAIL`) alongside median and minimum latency in milliseconds.

## Running Locally

```bash
# Full benchmark table with median/min timings and optimality checks
uv run python benchmarks/bench_assign.py

# Fast verification run (fewer warmup/repeat iterations)
uv run python benchmarks/bench_assign.py --quick
```
