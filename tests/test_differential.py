from __future__ import annotations

import math

import numpy as np
import pytest
from auction_assign import InfeasibleError, auction_assign, solve
from hypothesis import example, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]


def _finite_float_strategy(min_value: float, max_value: float) -> st.SearchStrategy[float]:
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


@st.composite
def dense_cost_matrices(draw: st.DrawFn) -> tuple[NDArray[np.float64], bool]:
    n_rows = draw(st.integers(min_value=1, max_value=25))
    n_cols = draw(st.integers(min_value=1, max_value=25))
    maximize = draw(st.booleans())
    scale_mode = draw(st.sampled_from(["mixed", "small", "medium", "large", "constant"]))

    if scale_mode == "small":
        cell_strat = _finite_float_strategy(-1e-3, 1e-3)
    elif scale_mode == "medium":
        cell_strat = _finite_float_strategy(-100.0, 100.0)
    elif scale_mode == "large":
        cell_strat = _finite_float_strategy(-1e5, 1e5)
    elif scale_mode == "constant":
        const_val = draw(st.one_of(st.just(0.0), _finite_float_strategy(-100.0, 100.0)))
        cell_strat = st.just(const_val)
    else:
        cell_strat = st.one_of(
            st.just(0.0),
            st.just(-0.0),
            _finite_float_strategy(-1e-4, 1e-4),
            _finite_float_strategy(-100.0, 100.0),
            _finite_float_strategy(-1e5, 1e5),
        )

    flat = draw(st.lists(cell_strat, min_size=n_rows * n_cols, max_size=n_rows * n_cols))
    matrix = np.array(flat, dtype=np.float64).reshape((n_rows, n_cols))
    return (matrix, maximize)


@st.composite
def gated_inf_matrices(draw: st.DrawFn) -> tuple[NDArray[np.float64], bool]:
    n_rows = draw(st.integers(min_value=1, max_value=15))
    n_cols = draw(st.integers(min_value=1, max_value=15))
    maximize = draw(st.booleans())
    inf_prob = draw(st.sampled_from([0.1, 0.3, 0.5, 0.75, 1.0]))
    inf_sentinel = float("-inf") if maximize else float("inf")

    finite_strat = st.one_of(
        st.just(0.0),
        _finite_float_strategy(-1e-3, 1e-3),
        _finite_float_strategy(-1e3, 1e3),
    )
    pair_strat = st.tuples(st.floats(min_value=0.0, max_value=1.0), finite_strat)
    raw_cells = draw(st.lists(pair_strat, min_size=n_rows * n_cols, max_size=n_rows * n_cols))
    flat = [inf_sentinel if gate < inf_prob else val for gate, val in raw_cells]
    matrix = np.array(flat, dtype=np.float64).reshape((n_rows, n_cols))
    return (matrix, maximize)


@st.composite
def partial_cost_limit_cases(draw: st.DrawFn) -> tuple[NDArray[np.float64], bool, float]:
    n_rows = draw(st.integers(min_value=1, max_value=5))
    n_cols = draw(st.integers(min_value=1, max_value=5))
    maximize = draw(st.booleans())
    cost_limit = draw(st.one_of(st.just(0.0), _finite_float_strategy(-1e3, 1e3)))
    inf_sentinel = float("-inf") if maximize else float("inf")

    finite_strat = st.one_of(
        st.just(0.0),
        _finite_float_strategy(cost_limit - 5.0, cost_limit + 5.0),
        _finite_float_strategy(-1e3, 1e3),
    )
    cell_strat = st.one_of(
        st.just(inf_sentinel),
        st.just(cost_limit),
        finite_strat,
        finite_strat,
    )
    flat = draw(st.lists(cell_strat, min_size=n_rows * n_cols, max_size=n_rows * n_cols))
    matrix = np.array(flat, dtype=np.float64).reshape((n_rows, n_cols))
    return (matrix, maximize, cost_limit)


def _assert_valid_matching(
    row_ind: NDArray[np.intp],
    col_ind: NDArray[np.intp],
    n_rows: int,
    n_cols: int,
    *,
    expected_len: int | None = None,
) -> None:
    assert row_ind.dtype == np.intp
    assert col_ind.dtype == np.intp
    assert row_ind.ndim == 1
    assert col_ind.ndim == 1
    assert len(row_ind) == len(col_ind)
    if expected_len is not None:
        assert len(row_ind) == expected_len
    else:
        assert len(row_ind) <= min(n_rows, n_cols)

    assert np.all((0 <= row_ind) & (row_ind < n_rows))
    assert np.all((0 <= col_ind) & (col_ind < n_cols))
    assert np.all(np.diff(row_ind) > 0)
    assert len(np.unique(col_ind)) == len(col_ind)


def _brute_force_best_partial_benefit(
    cost_matrix: NDArray[np.float64],
    *,
    maximize: bool,
    cost_limit: float,
) -> float:
    n_rows, n_cols = cost_matrix.shape
    eligible: list[list[tuple[int, float]]] = []
    for r in range(n_rows):
        row_edges: list[tuple[int, float]] = []
        for c in range(n_cols):
            val = float(cost_matrix[r, c])
            if not math.isfinite(val):
                continue
            if maximize and val > cost_limit:
                row_edges.append((c, val - cost_limit))
            elif not maximize and val < cost_limit:
                row_edges.append((c, cost_limit - val))
        eligible.append(row_edges)

    def search(row: int, used_cols: int, acc: float) -> float:
        if row == n_rows:
            return acc
        best = search(row + 1, used_cols, acc)
        for col, net in eligible[row]:
            bit = 1 << col
            if not (used_cols & bit):
                cand = search(row + 1, used_cols | bit, acc + net)
                if cand > best:
                    best = cand
        return best

    return search(0, 0, 0.0)


@settings(max_examples=75, deadline=None)
@given(case=dense_cost_matrices())
def test_matches_scipy_dense_random(case: tuple[NDArray[np.float64], bool]) -> None:
    cost_matrix, maximize = case
    n_rows, n_cols = cost_matrix.shape
    expected_len = min(n_rows, n_cols)

    res = solve(cost_matrix, maximize=maximize)
    row_ind, col_ind = auction_assign(cost_matrix, maximize=maximize)

    _assert_valid_matching(res.row_ind, res.col_ind, n_rows, n_cols, expected_len=expected_len)
    assert np.array_equal(row_ind, res.row_ind)
    assert np.array_equal(col_ind, res.col_ind)
    assert res.cost == pytest.approx(
        float(cost_matrix[res.row_ind, res.col_ind].sum()),
        rel=1e-12,
        abs=1e-12,
    )

    scipy_r, scipy_c = linear_sum_assignment(cost_matrix, maximize=maximize)
    scipy_cost = float(cost_matrix[scipy_r, scipy_c].sum())
    assert abs(res.cost - scipy_cost) <= expected_len * res.info.final_eps + 1e-8


@settings(max_examples=75, deadline=None)
@given(case=gated_inf_matrices())
def test_matches_scipy_gated_inf(case: tuple[NDArray[np.float64], bool]) -> None:
    cost_matrix, maximize = case
    n_rows, n_cols = cost_matrix.shape
    expected_len = min(n_rows, n_cols)

    try:
        scipy_r, scipy_c = linear_sum_assignment(cost_matrix, maximize=maximize)
    except ValueError as exc:
        assert "infeasible" in str(exc).lower()
        with pytest.raises(InfeasibleError):
            solve(cost_matrix, maximize=maximize)
        with pytest.raises(InfeasibleError):
            auction_assign(cost_matrix, maximize=maximize)
        return

    res = solve(cost_matrix, maximize=maximize)
    row_ind, col_ind = auction_assign(cost_matrix, maximize=maximize)

    _assert_valid_matching(res.row_ind, res.col_ind, n_rows, n_cols, expected_len=expected_len)
    assert np.array_equal(row_ind, res.row_ind)
    assert np.array_equal(col_ind, res.col_ind)

    matched_costs = cost_matrix[res.row_ind, res.col_ind]
    assert np.all(np.isfinite(matched_costs))
    assert res.cost == pytest.approx(float(matched_costs.sum()), rel=1e-12, abs=1e-12)

    scipy_cost = float(cost_matrix[scipy_r, scipy_c].sum())
    assert abs(res.cost - scipy_cost) <= expected_len * res.info.final_eps + 1e-8


@settings(max_examples=75, deadline=None)
@given(case=partial_cost_limit_cases())
@example((np.array([[5.0, float("inf")], [2.0, 5.0]], dtype=np.float64), False, 5.0))
@example((np.array([[5.0, float("-inf")], [8.0, 5.0]], dtype=np.float64), True, 5.0))
@example((np.array([[5.0, float("inf")], [float("inf"), 5.0]], dtype=np.float64), False, 5.0))
@example(
    (
        np.array([[1.0, 5.0, float("inf")], [5.0, 2.0, 10.0]], dtype=np.float64),
        False,
        5.0,
    )
)
def test_cost_limit_matches_bruteforce_oracle(
    case: tuple[NDArray[np.float64], bool, float],
) -> None:
    cost_matrix, maximize, cost_limit = case
    n_rows, n_cols = cost_matrix.shape

    res = solve(cost_matrix, maximize=maximize, cost_limit=cost_limit)
    row_ind, col_ind = auction_assign(cost_matrix, maximize=maximize, cost_limit=cost_limit)

    _assert_valid_matching(res.row_ind, res.col_ind, n_rows, n_cols, expected_len=None)
    assert np.array_equal(row_ind, res.row_ind)
    assert np.array_equal(col_ind, res.col_ind)

    matched_costs = cost_matrix[res.row_ind, res.col_ind]
    assert np.all(np.isfinite(matched_costs))
    if maximize:
        assert np.all(matched_costs > cost_limit)
        got_benefit = float(np.sum(matched_costs - cost_limit))
    else:
        assert np.all(matched_costs < cost_limit)
        got_benefit = float(np.sum(cost_limit - matched_costs))

    expected_cost = 0.0 if len(res.row_ind) == 0 else float(matched_costs.sum())
    assert res.cost == pytest.approx(expected_cost, rel=1e-12, abs=1e-12)

    oracle_best_benefit = _brute_force_best_partial_benefit(
        cost_matrix,
        maximize=maximize,
        cost_limit=cost_limit,
    )
    assert abs(got_benefit - oracle_best_benefit) <= len(res.row_ind) * res.info.final_eps + 1e-9
