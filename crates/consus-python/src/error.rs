//! Error mapping from consus error types to PyO3 `PyErr`.

use consus_core::Error;
use consus_mat::MatError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// Convert a `consus_core::Error` into a Python `RuntimeError`.
pub fn from_consus(e: Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:?}"))
}

/// Convert a `consus_mat::MatError` into a Python `RuntimeError`.
pub fn from_consus_mat(e: MatError) -> PyErr {
    PyRuntimeError::new_err(format!("{e:?}"))
}
