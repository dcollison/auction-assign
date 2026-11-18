from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import auction_assign
import numpy as np
import pytest
from auction_assign import (
    AuctionInfo,
    AuctionResult,
    ConvergenceError,
    InfeasibleError,
    solve,
)
from numpy.typing import NDArray


def test_package_imports() -> None:
    assert hasattr(auction_assign, "__version__")
    assert hasattr(auction_assign, "auction_assign")
    assert hasattr(auction_assign, "solve")
    assert hasattr(auction_assign, "AuctionInfo")
    assert hasattr(auction_assign, "AuctionResult")
    assert hasattr(auction_assign, "InfeasibleError")
    assert hasattr(auction_assign, "ConvergenceError")


def test_auction_assign_basic_square_and_rectangular() -> None:
    classic = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    row_ind, col_ind = auction_assign.auction_assign(classic)

    assert isinstance(row_ind, np.ndarray)
    assert isinstance(col_ind, np.ndarray)
    assert row_ind.dtype == np.intp
    assert col_ind.dtype == np.intp
    assert np.array_equal(row_ind, np.array([0, 1, 2], dtype=np.intp))
    assert np.array_equal(col_ind, np.array([1, 0, 2], dtype=np.intp))
    assert np.all(np.diff(row_ind) > 0)

    wide = np.array([[10.0, 1.0, 5.0, 3.0], [2.0, 9.0, 4.0, 8.0]])
    rw, cw = auction_assign.auction_assign(wide)
    assert rw.dtype == np.intp
    assert cw.dtype == np.intp
    assert np.array_equal(rw, np.array([0, 1], dtype=np.intp))
    assert np.array_equal(cw, np.array([1, 0], dtype=np.intp))

    tall = np.array([[10.0, 10.0], [1.0, 9.0], [10.0, 10.0], [9.0, 2.0]])
    rt, ct = auction_assign.auction_assign(tall)
    assert rt.dtype == np.intp
    assert ct.dtype == np.intp
    assert np.array_equal(rt, np.array([1, 3], dtype=np.intp))
    assert np.array_equal(ct, np.array([0, 1], dtype=np.intp))
    assert np.all(np.diff(rt) > 0)


def test_solve_returns_structured_result_and_info() -> None:
    cost_matrix = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    res = solve(cost_matrix)

    assert isinstance(res, AuctionResult)
    assert isinstance(res.info, AuctionInfo)
    assert res.row_ind.dtype == np.intp
    assert res.col_ind.dtype == np.intp
    assert np.array_equal(res.row_ind, np.array([0, 1, 2], dtype=np.intp))
    assert np.array_equal(res.col_ind, np.array([1, 0, 2], dtype=np.intp))
    assert isinstance(res.cost, float)
    assert res.cost == pytest.approx(float(cost_matrix[res.row_ind, res.col_ind].sum()))
    assert res.cost == pytest.approx(5.0)

    assert isinstance(res.info.iterations, int)
    assert isinstance(res.info.scaling_rounds, int)
    assert isinstance(res.info.final_eps, float)
    assert res.info.iterations > 0
    assert res.info.scaling_rounds >= 1
    assert res.info.final_eps > 0.0
    assert res.info.row_prices.dtype == np.float64
    assert res.info.col_prices.dtype == np.float64
    assert res.info.row_prices.shape == (3,)
    assert res.info.col_prices.shape == (3,)

    # Frozen dataclasses cannot be mutated
    with pytest.raises(FrozenInstanceError):
        res.cost = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        res.info.iterations = 0  # type: ignore[misc]


def test_input_coercion_lists_dtypes_and_strided_views() -> None:
    nested_list = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    r_list, c_list = auction_assign.auction_assign(nested_list)
    assert np.array_equal(r_list, [0, 1, 2])
    assert np.array_equal(c_list, [1, 0, 2])

    arr_int64 = np.array(nested_list, dtype=np.int64)
    res_i64 = solve(arr_int64)
    assert res_i64.cost == pytest.approx(5.0)

    arr_float32 = np.array(nested_list, dtype=np.float32)
    res_f32 = solve(arr_float32)
    assert res_f32.cost == pytest.approx(5.0)

    # Fortran-contiguous transposed view
    base_rect = np.array([[10.0, 1.0], [10.0, 9.0], [5.0, 10.0], [3.0, 8.0]], dtype=np.float64)
    transposed = base_rect.T
    assert not transposed.flags.c_contiguous
    assert transposed.flags.f_contiguous
    res_t = solve(transposed)
    assert res_t.cost == pytest.approx(float(transposed[res_t.row_ind, res_t.col_ind].sum()))
    assert res_t.cost == pytest.approx(4.0)

    # Non-contiguous strided slice with negative step
    big = np.arange(30, dtype=np.float64).reshape(6, 5)
    strided = big[::2, ::-1]
    assert not strided.flags.c_contiguous
    assert not strided.flags.f_contiguous
    res_s = solve(strided)
    assert res_s.row_ind.shape == (3,)
    assert res_s.col_ind.shape == (3,)
    assert res_s.cost == pytest.approx(float(strided[res_s.row_ind, res_s.col_ind].sum()))

    # Big-endian float64 array
    big_endian = np.asarray(nested_list, dtype=">f8")
    res_be = solve(big_endian)
    assert res_be.cost == pytest.approx(5.0)

    # Unaligned C-contiguous view
    buf = bytearray(8 * 4 + 1)
    unaligned = np.ndarray((2, 2), dtype=np.float64, buffer=buf, offset=1)
    unaligned[:] = [[1.0, 10.0], [10.0, 2.0]]
    assert not unaligned.flags.aligned
    res_unaligned = solve(unaligned)
    assert res_unaligned.cost == pytest.approx(3.0)


def test_validation_errors() -> None:
    # Wrong dimensionality -> ValueError
    with pytest.raises(ValueError, match="2-dimensional"):
        auction_assign.auction_assign(np.array(5.0))
    with pytest.raises(ValueError, match="2-dimensional"):
        auction_assign.auction_assign(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="2-dimensional"):
        solve(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="2-dimensional"):
        auction_assign.auction_assign([])

    # Non-numeric dtypes must not silently coerce strings/None/complex
    with pytest.raises(TypeError):
        solve(np.array([["1.0", "2.0"], ["3.0", "4.0"]]))
    with pytest.raises(TypeError):
        solve([[1.0, None], [2.0, 3.0]])
    with pytest.raises(TypeError):
        solve(np.array([[1.0 + 1.0j, 2.0], [3.0, 4.0]]))

    # NaN and unbounded-benefit infinities -> ValueError
    with pytest.raises(ValueError, match="NaN"):
        solve([[1.0, float("nan")], [2.0, 3.0]])
    with pytest.raises(ValueError, match="-inf"):
        solve([[1.0, float("-inf")], [2.0, 3.0]])
    with pytest.raises(ValueError, match="\\+inf"):
        solve([[1.0, float("inf")], [2.0, 3.0]], maximize=True)

    valid = [[1.0, 2.0], [3.0, 4.0]]

    # Invalid cost_limit
    with pytest.raises(ValueError, match="cost_limit"):
        solve(valid, cost_limit=float("nan"))
    with pytest.raises(TypeError):
        solve(valid, cost_limit=True)

    # Invalid eps_min
    for bad_eps in (0.0, -1e-6, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="eps_min"):
            solve(valid, eps_min=bad_eps)
    with pytest.raises(TypeError):
        solve(valid, eps_min=True)

    # Invalid eps_factor
    for bad_factor in (1.0, 0.5, -2.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="eps_factor"):
            solve(valid, eps_factor=bad_factor)
    with pytest.raises(TypeError):
        solve(valid, eps_factor=True)

    # Invalid max_iterations
    for bad_iter in (0, -1, -100):
        with pytest.raises(ValueError, match="max_iterations"):
            solve(valid, max_iterations=bad_iter)
    with pytest.raises(ValueError, match="max_iterations"):
        solve(valid, max_iterations=True)
    with pytest.raises(ValueError, match="max_iterations"):
        solve(valid, max_iterations=False)
    with pytest.raises(ValueError, match="max_iterations"):
        solve(valid, max_iterations=np.bool_(True))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_iterations"):
        solve(valid, max_iterations=np.bool_(False))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        solve(valid, max_iterations=1.5)  # type: ignore[arg-type]


def test_infeasible_error() -> None:
    inf = float("inf")
    infeasible_min = np.array(
        [
            [1.0, inf, inf],
            [2.0, inf, inf],
            [3.0, 4.0, 5.0],
        ]
    )
    with pytest.raises(InfeasibleError) as exc_info:
        auction_assign.auction_assign(infeasible_min)
    assert isinstance(exc_info.value, ValueError)

    ninf = float("-inf")
    infeasible_max = np.array(
        [
            [1.0, ninf, ninf],
            [2.0, ninf, ninf],
        ]
    )
    with pytest.raises(InfeasibleError) as max_exc_info:
        solve(infeasible_max, maximize=True)
    assert isinstance(max_exc_info.value, ValueError)


def test_convergence_error_max_iterations_and_ulp_stall() -> None:
    non_trivial = np.array(
        [
            [10.0, 1.0, 5.0, 3.0],
            [2.0, 9.0, 4.0, 8.0],
            [6.0, 3.0, 11.0, 2.0],
            [7.0, 8.0, 1.0, 9.0],
        ]
    )
    with pytest.raises(ConvergenceError) as exc_info:
        solve(non_trivial, max_iterations=1)
    err = exc_info.value
    assert isinstance(err, RuntimeError)
    assert err.iterations == 1
    assert err.final_eps > 0.0

    with pytest.raises(ConvergenceError) as assign_exc_info:
        auction_assign.auction_assign(non_trivial, max_iterations=1)
    assert assign_exc_info.value.iterations == 1
    assert assign_exc_info.value.final_eps > 0.0

    # Extreme dynamic range triggering IEEE-754 ULP stall
    extreme = np.array([[0.0, 1e12], [0.0, 1e12]])
    with pytest.raises(ConvergenceError) as ulp_exc:
        solve(extreme, eps_min=1e-20)
    assert isinstance(ulp_exc.value, RuntimeError)
    assert ulp_exc.value.iterations > 0
    assert ulp_exc.value.final_eps > 0.0


def test_maximize_cost_limit_and_edge_shapes() -> None:
    # maximize=True with negative costs
    neg_costs = np.array([[-10.0, -1.0, -5.0], [-2.0, -9.0, -4.0], [-3.0, -6.0, -2.0]])
    res_max = solve(neg_costs, maximize=True)
    assert np.array_equal(res_max.row_ind, [0, 1, 2])
    assert np.array_equal(res_max.col_ind, [1, 0, 2])
    assert res_max.cost == pytest.approx(-5.0)

    # cost_limit partial assignment (strict < limit for minimization)
    inf = float("inf")
    gated = np.array([[1.0, inf, inf], [inf, 2.0, inf], [inf, inf, 50.0]])
    res_part = solve(gated, cost_limit=10.0)
    assert np.array_equal(res_part.row_ind, [0, 1])
    assert np.array_equal(res_part.col_ind, [0, 1])
    assert res_part.cost == pytest.approx(3.0)

    # Exact boundary cost == limit is excluded
    boundary = np.array([[5.0, 10.0], [10.0, 5.0]])
    res_bound = solve(boundary, cost_limit=5.0)
    assert len(res_bound.row_ind) == 0
    assert len(res_bound.col_ind) == 0
    assert res_bound.cost == 0.0

    # cost_limit partial assignment with maximize=True (strict > limit)
    ninf = float("-inf")
    rewards = np.array([[10.0, ninf, ninf], [ninf, 2.0, ninf], [ninf, ninf, -5.0]])
    res_part_max = solve(rewards, maximize=True, cost_limit=0.0)
    assert np.array_equal(res_part_max.row_ind, [0, 1])
    assert np.array_equal(res_part_max.col_ind, [0, 1])
    assert res_part_max.cost == pytest.approx(12.0)

    # 1x1 shape
    res_1x1 = solve([[42.0]])
    assert np.array_equal(res_1x1.row_ind, [0])
    assert np.array_equal(res_1x1.col_ind, [0])
    assert res_1x1.cost == pytest.approx(42.0)
    assert res_1x1.info.row_prices.shape == (1,)
    assert res_1x1.info.col_prices.shape == (1,)

    # Empty shapes: 0x0, 0x3, 3x0
    for shape in ((0, 0), (0, 3), (3, 0)):
        empty = np.empty(shape, dtype=np.float64)
        res_empty = solve(empty)
        assert res_empty.row_ind.dtype == np.intp
        assert res_empty.col_ind.dtype == np.intp
        assert res_empty.row_ind.shape == (0,)
        assert res_empty.col_ind.shape == (0,)
        assert res_empty.cost == 0.0
        assert res_empty.info.iterations == 0
        assert res_empty.info.scaling_rounds == 0
        assert res_empty.info.final_eps == 0.0
        assert np.array_equal(res_empty.info.row_prices, np.zeros(shape[0]))
        assert np.array_equal(res_empty.info.col_prices, np.zeros(shape[1]))

        r_e, c_e = auction_assign.auction_assign(empty)
        assert r_e.dtype == np.intp
        assert c_e.dtype == np.intp
        assert r_e.shape == (0,)
        assert c_e.shape == (0,)


def test_gil_release_concurrent_threadpool() -> None:
    rng = np.random.default_rng(42)
    matrices: list[NDArray[np.float64]] = [rng.uniform(0.0, 100.0, size=(80, 80)) for _ in range(8)]
    expected = [solve(m) for m in matrices]

    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_results = list(pool.map(solve, matrices))

    for got, exp in zip(concurrent_results, expected, strict=True):
        assert np.array_equal(got.row_ind, exp.row_ind)
        assert np.array_equal(got.col_ind, exp.col_ind)
        assert got.cost == pytest.approx(exp.cost)

    # Also verify concurrent read-only access to the same ndarray across threads
    shared = matrices[0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        shared_results = list(pool.map(solve, [shared] * 8))

    for got in shared_results:
        assert np.array_equal(got.row_ind, expected[0].row_ind)
        assert np.array_equal(got.col_ind, expected[0].col_ind)
        assert got.cost == pytest.approx(expected[0].cost)


def test_core_direct_non_c_contiguous_and_unaligned() -> None:
    # Fortran-contiguous view passed directly to _core must still solve in logical row-major order
    base = np.array([[10.0, 1.0], [10.0, 9.0], [5.0, 10.0], [3.0, 8.0]], dtype=np.float64)
    f_view = base.T
    assert f_view.flags.f_contiguous
    assert not f_view.flags.c_contiguous
    row_ind, col_ind, cost, *_ = auction_assign._core.solve_dense(f_view)
    assert np.array_equal(row_ind, [0, 1])
    assert np.array_equal(col_ind, [3, 0])
    assert cost == pytest.approx(float(f_view[row_ind, col_ind].sum()))
    assert cost == pytest.approx(4.0)

    # Unaligned views passed directly to _core must raise ValueError cleanly rather than UB
    buf = bytearray(8 * 4 + 1)
    unaligned = np.ndarray((2, 2), dtype=np.float64, buffer=buf, offset=1)
    unaligned[:] = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="aligned"):
        auction_assign._core.solve_dense(unaligned)

    unaligned_non_c = unaligned.T
    assert not unaligned_non_c.flags.c_contiguous
    assert not unaligned_non_c.flags.aligned
    with pytest.raises(ValueError, match=r"^cost matrix must be aligned$"):
        auction_assign._core.solve_dense(unaligned_non_c)
