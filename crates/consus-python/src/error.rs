use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// Bridge a `consus_core::Error` to a Python `RuntimeError`.
pub fn from_consus(e: consus_core::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:?}"))
}
