import numpy as np
from numpy.typing import NDArray

class _CoreInfeasibleError(ValueError): ...
class _CoreConvergenceError(RuntimeError): ...

def assign_dense(
    costs: NDArray[np.float64],
    maximize: bool = False,
    cost_limit: float | None = None,
    eps_min: float | None = None,
    eps_factor: float = 5.0,
    max_iterations: int | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]: ...
def solve_dense(
    costs: NDArray[np.float64],
    maximize: bool = False,
    cost_limit: float | None = None,
    eps_min: float | None = None,
    eps_factor: float = 5.0,
    max_iterations: int | None = None,
) -> tuple[
    NDArray[np.intp],
    NDArray[np.intp],
    float,
    int,
    int,
    float,
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
