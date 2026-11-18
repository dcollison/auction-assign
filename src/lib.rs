use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod solver;

use crate::solver::{SolveConfig, SolverError};

create_exception!(
    _core,
    _CoreInfeasibleError,
    PyValueError,
    "Raised when a gated cost matrix has no valid complete assignment."
);
create_exception!(
    _core,
    _CoreConvergenceError,
    PyRuntimeError,
    "Raised when the solver hits max_iterations or floating-point price stagnation."
);

type SolveDenseOutput<'py> = (
    Bound<'py, PyArray1<isize>>,
    Bound<'py, PyArray1<isize>>,
    f64,
    usize,
    usize,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

fn usize_vec_to_isize(indices: Vec<usize>, name: &str) -> PyResult<Vec<isize>> {
    indices
        .into_iter()
        .map(|idx| {
            isize::try_from(idx).map_err(|_| {
                PyOverflowError::new_err(format!("{name} index {idx} overflows isize"))
            })
        })
        .collect()
}

fn map_solver_error(err: SolverError) -> PyErr {
    let full_msg = err.to_string();
    match err {
        SolverError::InvalidInput(_) => PyValueError::new_err(full_msg),
        SolverError::Infeasible(_) => _CoreInfeasibleError::new_err(full_msg),
        SolverError::Convergence {
            iterations,
            final_eps,
            ..
        } => _CoreConvergenceError::new_err((full_msg, iterations, final_eps)),
    }
}

#[pyfunction]
#[pyo3(signature = (costs, maximize = false, cost_limit = None, eps_min = None, eps_factor = 5.0, max_iterations = None))]
fn solve_dense<'py>(
    py: Python<'py>,
    costs: PyReadonlyArray2<'py, f64>,
    maximize: bool,
    cost_limit: Option<f64>,
    eps_min: Option<f64>,
    eps_factor: f64,
    max_iterations: Option<usize>,
) -> PyResult<SolveDenseOutput<'py>> {
    let shape = costs.shape();
    let n_rows = shape[0];
    let n_cols = shape[1];

    let owned_buf: Vec<f64>;
    let costs_slice: &[f64] = if n_rows == 0 || n_cols == 0 {
        &[]
    } else if costs.is_c_contiguous() {
        costs.as_slice().map_err(|err| {
            PyValueError::new_err(format!(
                "cost matrix must be aligned and C-contiguous: {err}"
            ))
        })?
    } else if costs.is_aligned() {
        owned_buf = costs.as_array().iter().copied().collect();
        &owned_buf
    } else {
        return Err(PyValueError::new_err(
            "cost matrix must be aligned and C-contiguous",
        ));
    };

    let config = SolveConfig {
        maximize,
        cost_limit,
        eps_min,
        eps_factor,
        max_iterations,
    };

    let output = py
        .detach(|| solver::solve_dense(costs_slice, n_rows, n_cols, &config))
        .map_err(map_solver_error)?;

    let row_ind = usize_vec_to_isize(output.row_ind, "row")?;
    let col_ind = usize_vec_to_isize(output.col_ind, "column")?;

    Ok((
        row_ind.into_pyarray(py),
        col_ind.into_pyarray(py),
        output.cost,
        output.iterations,
        output.scaling_rounds,
        output.final_eps,
        output.row_prices.into_pyarray(py),
        output.col_prices.into_pyarray(py),
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "_CoreInfeasibleError",
        m.py().get_type::<_CoreInfeasibleError>(),
    )?;
    m.add(
        "_CoreConvergenceError",
        m.py().get_type::<_CoreConvergenceError>(),
    )?;
    m.add_function(wrap_pyfunction!(solve_dense, m)?)?;
    Ok(())
}
