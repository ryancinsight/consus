//! `consus` — Python bindings for the Consus scientific storage library.
//!
//! Exposes HDF5, Zarr v2, Parquet, netCDF-4, and MATLAB .mat read/write via PyO3.

// Some `#[pymethods]` mirror Python constructor signatures with many parameters.
#![expect(
    clippy::too_many_arguments,
    reason = "`#[pymethods]` mirror Python constructor signatures with many parameters"
)]

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
// PyO3 0.30 assumes free-threaded safety unless told otherwise. The binding
// surface has no free-threading audit or concurrency test yet, so the module
// keeps the GIL on free-threaded interpreters until that audit lands.
#[pymodule(gil_used = true)]
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
