// PyO3 0.22 macro-generated code triggers `unsafe_op_in_unsafe_fn` under
// Rust edition 2024.  The unsafe is inside proc-macro expansion we do not
// control; suppress the lint for this cdylib crate only.
#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;

mod dtype;
mod error;
mod file;

use file::{reader::from_bytes, PyDatasetInfo, PyFileBuilder, PyHdf5File};

/// Open an HDF5 file from bytes.
///
/// Parameters
/// ----------
/// data : bytes
///     Raw HDF5 file content.
///
/// Returns
/// -------
/// Hdf5File
#[pyfunction]
fn open(data: &[u8]) -> PyResult<PyHdf5File> {
    from_bytes(data)
}

#[pymodule]
fn consus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHdf5File>()?;
    m.add_class::<PyDatasetInfo>()?;
    m.add_class::<PyFileBuilder>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    Ok(())
}
