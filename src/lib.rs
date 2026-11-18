use pyo3::prelude::*;

mod solver;

#[pymodule]
fn _core(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
