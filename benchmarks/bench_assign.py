from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import auction_assign as aa
import numpy as np
import scipy  # type: ignore[import-untyped]
from auction_assign import auction_assign, solve
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

SCIPY_INF_SENTINEL = 1e15


@dataclass(frozen=True, slots=True)
class BenchCase:
    """Configuration for a single benchmark matrix shape and sparsity."""

    scenario: str
    n_rows: int
    n_cols: int
    inf_fraction: float = 0.0

    @property
    def shape_label(self) -> str:
        return f"{self.n_rows}x{self.n_cols}"


@dataclass(frozen=True, slots=True)
class BenchResult:
    """Timing statistics and optimality verification result for one benchmark case."""

    case: BenchCase
    auction_median_ms: float
    auction_min_ms: float
    scipy_median_ms: float
    scipy_min_ms: float
    auction_cost: float
    scipy_cost: float
    abs_gap: float
    rel_gap: float
    tol: float
    passed: bool
    iterations: int
    scaling_rounds: int


FULL_CASES: tuple[BenchCase, ...] = (
    BenchCase("dense", 20, 20),
    BenchCase("dense", 50, 50),
    BenchCase("dense", 100, 100),
    BenchCase("dense", 250, 250),
    BenchCase("dense", 500, 500),
    BenchCase("dense", 50, 200),
    BenchCase("gated-90%", 50, 50, inf_fraction=0.90),
    BenchCase("gated-90%", 100, 100, inf_fraction=0.90),
    BenchCase("gated-90%", 250, 250, inf_fraction=0.90),
    BenchCase("gated-90%", 500, 500, inf_fraction=0.90),
)


def _generate_case_matrices(
    case: BenchCase,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a cost matrix and a SciPy-compatible matrix (replacing inf with sentinel)."""
    matrix = rng.uniform(0.0, 100.0, size=(case.n_rows, case.n_cols)).astype(np.float64)
    if case.inf_fraction <= 0.0:
        return (matrix, matrix)

    n_min = min(case.n_rows, case.n_cols)
    perm_rows = np.arange(n_min)
    perm_cols = rng.permutation(case.n_cols)[:n_min]
    total_cells = case.n_rows * case.n_cols
    off_perm_prob = min(
        1.0,
        case.inf_fraction * total_cells / max(1, total_cells - n_min),
    )
    mask = rng.random((case.n_rows, case.n_cols)) < off_perm_prob
    mask[perm_rows, perm_cols] = False
    matrix[mask] = np.inf
    scipy_matrix = np.where(np.isinf(matrix), SCIPY_INF_SENTINEL, matrix)
    return (matrix, scipy_matrix)


def _time_callable(
    fn: Callable[[], tuple[NDArray[np.intp], NDArray[np.intp]]],
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, float, tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Time a solver callable using perf_counter and return (median_ms, min_ms, last_output)."""
    last_out: tuple[NDArray[np.intp], NDArray[np.intp]] | None = None
    for _ in range(warmup):
        last_out = fn()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        last_out = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples_ms.append(elapsed_ms)

    assert last_out is not None
    return (statistics.median(samples_ms), min(samples_ms), last_out)


def run_case(
    case: BenchCase,
    *,
    rng: np.random.Generator,
    warmup: int,
    repeats: int,
) -> BenchResult:
    """Run timing and optimality checks for a single benchmark case."""
    matrix, scipy_matrix = _generate_case_matrices(case, rng)

    a_med_ms, a_min_ms, (a_rows, a_cols) = _time_callable(
        lambda: auction_assign(matrix),
        warmup=warmup,
        repeats=repeats,
    )
    s_med_ms, s_min_ms, (s_rows, s_cols) = _time_callable(
        lambda: linear_sum_assignment(scipy_matrix),
        warmup=warmup,
        repeats=repeats,
    )

    solved = solve(matrix)
    expected_len = min(case.n_rows, case.n_cols)
    auction_cost = float(matrix[a_rows, a_cols].sum())
    scipy_cost = float(matrix[s_rows, s_cols].sum())

    abs_gap = abs(auction_cost - scipy_cost)
    scale = max(1.0, abs(scipy_cost))
    rel_gap = abs_gap / scale
    tol = max(1e-8, 1e-6 * scale, expected_len * solved.info.final_eps + 1e-8)

    valid_shapes = len(a_rows) == expected_len and len(s_rows) == expected_len
    all_finite = bool(
        np.isfinite(auction_cost)
        and np.isfinite(scipy_cost)
        and np.all(np.isfinite(matrix[a_rows, a_cols]))
        and np.all(np.isfinite(matrix[s_rows, s_cols]))
    )
    solve_consistent = abs(solved.cost - auction_cost) <= tol
    passed = valid_shapes and all_finite and solve_consistent and abs_gap <= tol

    return BenchResult(
        case=case,
        auction_median_ms=a_med_ms,
        auction_min_ms=a_min_ms,
        scipy_median_ms=s_med_ms,
        scipy_min_ms=s_min_ms,
        auction_cost=auction_cost,
        scipy_cost=scipy_cost,
        abs_gap=abs_gap,
        rel_gap=rel_gap,
        tol=tol,
        passed=passed,
        iterations=solved.info.iterations,
        scaling_rounds=solved.info.scaling_rounds,
    )


def _format_table(results: list[BenchResult]) -> str:
    headers = (
        "Scenario",
        "Shape",
        "Auc Med(ms)",
        "Auc Min(ms)",
        "SciPy Med(ms)",
        "SciPy Min(ms)",
        "SciPy/Auc",
        "Rel Gap",
        "Iters",
        "Check",
    )
    rows: list[tuple[str, ...]] = [headers]
    for r in results:
        speedup = r.scipy_median_ms / r.auction_median_ms if r.auction_median_ms > 0.0 else 0.0
        rows.append(
            (
                r.case.scenario,
                r.case.shape_label,
                f"{r.auction_median_ms:.3f}",
                f"{r.auction_min_ms:.3f}",
                f"{r.scipy_median_ms:.3f}",
                f"{r.scipy_min_ms:.3f}",
                f"{speedup:.2f}x",
                f"{r.rel_gap:.2e}",
                str(r.iterations),
                "PASS" if r.passed else "FAIL",
            )
        )

    widths = [max(len(row[idx]) for row in rows) for idx in range(len(headers))]
    lines: list[str] = []
    for row_idx, row in enumerate(rows):
        padded = [
            cell.ljust(widths[col_idx]) if col_idx < 2 else cell.rjust(widths[col_idx])
            for col_idx, cell in enumerate(row)
        ]
        lines.append("  ".join(padded))
        if row_idx == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark auction-assign against scipy.optimize.linear_sum_assignment."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast subset of warmup/repeats for quick verification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible cost matrices (default: 42).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override number of warmup iterations per case.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override number of timed iterations per case.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    warmup = args.warmup if args.warmup is not None else (1 if args.quick else 2)
    repeats = args.repeats if args.repeats is not None else (2 if args.quick else 7)
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be > 0")

    rng = np.random.default_rng(args.seed)
    mode = "quick" if args.quick else "full"
    print(
        f"auction-assign {aa.__version__} vs scipy {scipy.__version__} "
        f"(numpy {np.__version__}, mode={mode}, seed={args.seed}, "
        f"warmup={warmup}, repeats={repeats})"
    )
    print()

    results = [run_case(case, rng=rng, warmup=warmup, repeats=repeats) for case in FULL_CASES]
    print(_format_table(results))
    print()

    passed_count = sum(1 for r in results if r.passed)
    print(f"Optimality checks: {passed_count}/{len(results)} passed.")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
