//! `consus` — Python bindings for the Consus scientific storage library.
//!
//! Exposes HDF5, Zarr v2, Parquet, netCDF-4, and MATLAB .mat read/write via PyO3.

// PyO3's `#[pymethods]`/`#[pyfunction]` macros expand to a `PyErr` error-path
// conversion that clippy ≥1.95 flags as `useless_conversion`. The lint fires on
// macro-generated code (the diagnostic span maps to the fn signature, where there
// is no editable `.into()`), so it is suppressed crate-wide rather than per-site.
#![allow(clippy::useless_conversion)]
// Some `#[pymethods]` mirror Python constructor signatures with many parameters.
#![allow(clippy::too_many_arguments)]

extern crate alloc;

mod error;
mod hdf5;
mod hdmf;
mod mat;
mod netcdf;
mod parquet;
mod zarr;

use hdf5::{PyDatasetInfo, PyFileBuilder, PyHdf5File};
use hdmf::{read_dynamic_table_bytes, PyDynamicTable, PyHdmfFileBuilder};
use mat::{loadmat_bytes, PyMatFile, PyMatVariable};
use netcdf::{PyNetcdfFile, PyNetcdfWriter};
use parquet::{PyParquetBuilder, PyParquetFile};
use zarr::PyZarrArray;

use pyo3::prelude::*;

/// Consus scientific storage library — Python interface.
///
/// Provides in-memory read/write for HDF5, Zarr v2, Apache Parquet,
/// netCDF-4, and MATLAB .mat files.
#[pymodule]
fn consus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // HDF5
    m.add_class::<PyHdf5File>()?;
    m.add_class::<PyFileBuilder>()?;
    m.add_class::<PyDatasetInfo>()?;

    // Zarr
    m.add_class::<PyZarrArray>()?;

    // Parquet
    m.add_class::<PyParquetFile>()?;
    m.add_class::<PyParquetBuilder>()?;

    // netCDF-4
    m.add_class::<PyNetcdfFile>()?;
    m.add_class::<PyNetcdfWriter>()?;

    // MATLAB .mat
    m.add_class::<PyMatFile>()?;
    m.add_class::<PyMatVariable>()?;
    m.add_function(wrap_pyfunction!(loadmat_bytes, m)?)?;

    // HDMF DynamicTable
    m.add_class::<PyDynamicTable>()?;
    m.add_class::<PyHdmfFileBuilder>()?;
    m.add_function(wrap_pyfunction!(read_dynamic_table_bytes, m)?)?;

    // Convenience aliases matching the existing public API.
    m.add("Hdf5File", m.getattr("PyHdf5File")?)?;
    m.add("FileBuilder", m.getattr("PyFileBuilder")?)?;
    m.add("DatasetInfo", m.getattr("PyDatasetInfo")?)?;
    m.add("ZarrArray", m.getattr("PyZarrArray")?)?;
    m.add("ParquetFile", m.getattr("PyParquetFile")?)?;
    m.add("ParquetBuilder", m.getattr("PyParquetBuilder")?)?;
    m.add("NetcdfFile", m.getattr("PyNetcdfFile")?)?;
    m.add("NetcdfWriter", m.getattr("PyNetcdfWriter")?)?;
    m.add("MatFile", m.getattr("PyMatFile")?)?;
    m.add("MatVariable", m.getattr("PyMatVariable")?)?;

    Ok(())
}
