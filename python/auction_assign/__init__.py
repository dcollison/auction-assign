from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core as _core

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AuctionInfo",
    "AuctionResult",
    "ConvergenceError",
    "InfeasibleError",
    "auction_assign",
    "solve",
]


class InfeasibleError(ValueError):
    """Raised when a gated cost matrix has no valid complete assignment."""


class ConvergenceError(RuntimeError):
    """Raised when the solver hits ``max_iterations`` or floating-point price stagnation."""

    iterations: int
    final_eps: float

    def __init__(
        self,
        message: str,
        iterations: int = 0,
        final_eps: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.final_eps = final_eps

    def __reduce__(self) -> tuple[type[ConvergenceError], tuple[str, int, float]]:
        msg = str(self.args[0]) if self.args else ""
        return (type(self), (msg, self.iterations, self.final_eps))


@dataclass(frozen=True, slots=True)
class AuctionInfo:
    """Diagnostic telemetry and dual variables produced by the auction solver."""

    iterations: int
    scaling_rounds: int
    final_eps: float
    row_prices: NDArray[np.float64]
    col_prices: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AuctionResult:
    """Optimal assignment indices, exact total cost, and solver diagnostics."""

    row_ind: NDArray[np.intp]
    col_ind: NDArray[np.intp]
    cost: float
    info: AuctionInfo


def _coerce_cost_matrix(cost_matrix: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(cost_matrix)
    if raw.ndim != 2:
        raise ValueError(
            f"cost_matrix must be 2-dimensional, got {raw.ndim}D array with shape {raw.shape}"
        )
    if raw.dtype.kind not in ("f", "i", "u", "b"):
        raise TypeError(f"cost_matrix must have a real numeric dtype, got {raw.dtype}")
    arr = np.ascontiguousarray(raw, dtype=np.float64)
    if not arr.flags.aligned:
        arr = arr.copy()
    return arr


def _coerce_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _coerce_optional_float(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _coerce_float(name, value)


def _coerce_max_iterations(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"max_iterations must be an integer > 0, not bool ({value!r})")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"max_iterations must be an integer or None, got {type(value).__name__}")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"max_iterations must be > 0, got {parsed}")
    return parsed


def solve(
    cost_matrix: ArrayLike,
    *,
    maximize: bool = False,
    cost_limit: float | None = None,
    eps_min: float | None = None,
    eps_factor: float = 5.0,
    max_iterations: int | None = None,
) -> AuctionResult:
    """Solve a 2D linear sum assignment problem and return indices, cost, and diagnostics."""
    if not isinstance(maximize, (bool, np.bool_)):
        raise TypeError(f"maximize must be a bool, got {type(maximize).__name__}")
    arr = _coerce_cost_matrix(cost_matrix)
    limit_val = _coerce_optional_float("cost_limit", cost_limit)
    eps_min_val = _coerce_optional_float("eps_min", eps_min)
    eps_factor_val = _coerce_float("eps_factor", eps_factor)
    max_iter_val = _coerce_max_iterations(max_iterations)

    try:
        (
            row_ind,
            col_ind,
            cost,
            iterations,
            scaling_rounds,
            final_eps,
            row_prices,
            col_prices,
        ) = _core.solve_dense(
            arr,
            maximize=bool(maximize),
            cost_limit=limit_val,
            eps_min=eps_min_val,
            eps_factor=eps_factor_val,
            max_iterations=max_iter_val,
        )
    except _core._CoreInfeasibleError as exc:
        msg = str(exc.args[0]) if exc.args else str(exc)
        raise InfeasibleError(msg) from None
    except _core._CoreConvergenceError as exc:
        msg, iterations_err, final_eps_err = exc.args
        raise ConvergenceError(
            str(msg),
            iterations=int(iterations_err),
            final_eps=float(final_eps_err),
        ) from None

    return AuctionResult(
        row_ind=row_ind,
        col_ind=col_ind,
        cost=cost,
        info=AuctionInfo(
            iterations=iterations,
            scaling_rounds=scaling_rounds,
            final_eps=final_eps,
            row_prices=row_prices,
            col_prices=col_prices,
        ),
    )


def auction_assign(
    cost_matrix: ArrayLike,
    *,
    maximize: bool = False,
    cost_limit: float | None = None,
    eps_min: float | None = None,
    eps_factor: float = 5.0,
    max_iterations: int | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Solve a 2D linear sum assignment problem and return ``(row_ind, col_ind)``."""
    result = solve(
        cost_matrix,
        maximize=maximize,
        cost_limit=cost_limit,
        eps_min=eps_min,
        eps_factor=eps_factor,
        max_iterations=max_iterations,
    )
    return (result.row_ind, result.col_ind)
