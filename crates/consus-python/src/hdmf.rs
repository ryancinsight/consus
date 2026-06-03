//! PyO3 bindings for consus-hdmf DynamicTable read/write.
//!
//! Exposes:
//! - [`PyDynamicTable`] — read-only view of a decoded DynamicTable.
//! - [`PyHdmfFileBuilder`] — accumulate columns and emit an HDF5 byte image.
//! - [`read_dynamic_table_bytes`] — top-level function to open an HDF5
//!   file and extract the root DynamicTable in one call.

use consus_hdmf::{ColumnData, DynamicTable, HdmfFile, HdmfFileBuilder};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Error bridge
// ---------------------------------------------------------------------------

fn to_py(e: consus_core::Error) -> PyErr {
    PyValueError::new_err(format!("{}", e))
}

// ---------------------------------------------------------------------------
// PyDynamicTable
// ---------------------------------------------------------------------------

/// A decoded HDMF DynamicTable read from an HDF5 file.
#[pyclass(name = "DynamicTable")]
pub struct PyDynamicTable {
    inner: DynamicTable,
}

#[pymethods]
impl PyDynamicTable {
    /// Logical name of the table (or `"root"` for root-level tables).
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Human-readable description.
    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }

    /// Ordered list of column names.
    #[getter]
    fn colnames(&self) -> Vec<String> {
        self.inner.colnames.clone()
    }

    /// Row identifier array (`ElementIdentifiers` in HDMF terms).
    #[getter]
    fn id(&self) -> Vec<i64> {
        self.inner.id.clone()
    }

    /// Number of rows in the table.
    #[getter]
    fn num_rows(&self) -> usize {
        self.inner.id.len()
    }

    /// All column names in order.
    fn column_names(&self) -> Vec<String> {
        self.inner.columns.iter().map(|c| c.name.clone()).collect()
    }

    /// Return the named column as a list of `float`.
    ///
    /// ## Errors
    ///
    /// Raises `ValueError` when the column is absent or is not a float column.
    fn get_column_f64(&self, name: &str) -> PyResult<Vec<f64>> {
        let col = find_col(&self.inner, name)?;
        match &col.data {
            ColumnData::F64(v) => Ok(v.clone()),
            other => Err(PyValueError::new_err(format!(
                "column '{}' is not f64 (type: {})",
                name,
                column_type_name(other)
            ))),
        }
    }

    /// Return the named column as a list of `int`.
    ///
    /// Accepts both signed and unsigned 64-bit integer columns.
    fn get_column_i64(&self, name: &str) -> PyResult<Vec<i64>> {
        let col = find_col(&self.inner, name)?;
        match &col.data {
            ColumnData::I64(v) => Ok(v.clone()),
            ColumnData::U64(v) => Ok(v.iter().map(|&u| u as i64).collect()),
            other => Err(PyValueError::new_err(format!(
                "column '{}' is not an integer column (type: {})",
                name,
                column_type_name(other)
            ))),
        }
    }

    /// Return the named column as a list of `str`.
    fn get_column_str(&self, name: &str) -> PyResult<Vec<String>> {
        let col = find_col(&self.inner, name)?;
        match &col.data {
            ColumnData::Str(v) => Ok(v.clone()),
            other => Err(PyValueError::new_err(format!(
                "column '{}' is not a string column (type: {})",
                name,
                column_type_name(other)
            ))),
        }
    }

    /// Return the named column as a list of `bool`.
    fn get_column_bool(&self, name: &str) -> PyResult<Vec<bool>> {
        let col = find_col(&self.inner, name)?;
        match &col.data {
            ColumnData::Bool(v) => Ok(v.clone()),
            other => Err(PyValueError::new_err(format!(
                "column '{}' is not a bool column (type: {})",
                name,
                column_type_name(other)
            ))),
        }
    }

    /// Return the cumulative VectorIndex for a ragged column, or `None`.
    fn get_column_index(&self, name: &str) -> PyResult<Option<Vec<u64>>> {
        let col = find_col(&self.inner, name)?;
        Ok(col.index.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicTable(name='{}', rows={}, columns=[{}])",
            self.inner.name,
            self.inner.id.len(),
            self.inner.colnames.join(", ")
        )
    }
}

fn find_col<'t>(table: &'t DynamicTable, name: &str) -> PyResult<&'t consus_hdmf::table::Column> {
    table
        .columns
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| PyValueError::new_err(format!("column '{}' not found", name)))
}

fn column_type_name(data: &ColumnData) -> &'static str {
    match data {
        ColumnData::F64(_) => "f64",
        ColumnData::I64(_) => "i64",
        ColumnData::U64(_) => "u64",
        ColumnData::Bool(_) => "bool",
        ColumnData::Str(_) => "str",
    }
}

// ---------------------------------------------------------------------------
// PyHdmfFileBuilder
// ---------------------------------------------------------------------------

/// Builder that accumulates columns and writes an HDMF DynamicTable HDF5 file.
///
/// ## Example
///
/// ```python
/// import consus
/// b = consus.HdmfFileBuilder("trials", "trial table")
/// b.add_column_f64("rt", "reaction time", [0.3, 0.5, 0.4])
/// b.add_column_bool("correct", "outcome", [True, False, True])
/// data = b.finish()   # bytes — save or pass to HdmfFile.open
/// ```
#[pyclass(name = "HdmfFileBuilder")]
pub struct PyHdmfFileBuilder {
    name: String,
    description: String,
    columns: Vec<(String, String, ColumnData, Option<Vec<u64>>)>,
}

#[pymethods]
impl PyHdmfFileBuilder {
    /// Create a new builder.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Logical table name stored as the root group's `name` (informational).
    /// description : str
    ///     HDMF `description` attribute written to the root group.
    #[new]
    fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_owned(),
            description: description.to_owned(),
            columns: Vec::new(),
        }
    }

    /// Add a float64 column.
    fn add_column_f64(&mut self, name: &str, description: &str, data: Vec<f64>) {
        self.columns.push((
            name.to_owned(),
            description.to_owned(),
            ColumnData::F64(data),
            None,
        ));
    }

    /// Add a signed int64 column.
    fn add_column_i64(&mut self, name: &str, description: &str, data: Vec<i64>) {
        self.columns.push((
            name.to_owned(),
            description.to_owned(),
            ColumnData::I64(data),
            None,
        ));
    }

    /// Add an unsigned uint64 column.
    fn add_column_u64(&mut self, name: &str, description: &str, data: Vec<u64>) {
        self.columns.push((
            name.to_owned(),
            description.to_owned(),
            ColumnData::U64(data),
            None,
        ));
    }

    /// Add a boolean column.
    fn add_column_bool(&mut self, name: &str, description: &str, data: Vec<bool>) {
        self.columns.push((
            name.to_owned(),
            description.to_owned(),
            ColumnData::Bool(data),
            None,
        ));
    }

    /// Add a string column.
    fn add_column_str(&mut self, name: &str, description: &str, data: Vec<String>) {
        self.columns.push((
            name.to_owned(),
            description.to_owned(),
            ColumnData::Str(data),
            None,
        ));
    }

    /// Add a ragged (variable-length rows) column with a cumulative VectorIndex.
    ///
    /// `index` must have length equal to the number of rows; each element is
    /// the exclusive end position in `data` for that row.
    fn add_ragged_column_i64(
        &mut self,
        name: &str,
        description: &str,
        data: Vec<i64>,
        index: Vec<u64>,
    ) {
        self.columns.push((
            name.to_owned(),
            description.to_owned(),
            ColumnData::I64(data),
            Some(index),
        ));
    }

    /// Serialise the table and return the HDF5 image as `bytes`.
    ///
    /// ## Errors
    ///
    /// Raises `ValueError` if encoding or HDF5 serialisation fails.
    fn finish(&self) -> PyResult<Vec<u8>> {
        let mut b = HdmfFileBuilder::new(self.name.clone(), self.description.clone());
        for (name, desc, data, index) in &self.columns {
            if let Some(idx) = index {
                b = b.add_ragged_column(name.clone(), desc.clone(), data.clone(), idx.clone());
            } else {
                b = b.add_column(name.clone(), desc.clone(), data.clone());
            }
        }
        Python::with_gil(|py| py.allow_threads(|| b.finish().map_err(to_py)))
    }

    fn __repr__(&self) -> String {
        format!(
            "HdmfFileBuilder(name='{}', columns={})",
            self.name,
            self.columns.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Top-level function
// ---------------------------------------------------------------------------

/// Open an HDF5 byte string and extract the root DynamicTable.
///
/// Parameters
/// ----------
/// data : bytes
///     Raw HDF5 file contents.
///
/// Returns
/// -------
/// DynamicTable
///
/// ## Errors
///
/// Raises `ValueError` if the data is not a valid HDMF DynamicTable HDF5 file.
#[pyfunction]
pub fn read_dynamic_table_bytes(data: &[u8]) -> PyResult<PyDynamicTable> {
    let file = HdmfFile::open(data).map_err(to_py)?;
    let table = file.read_table().map_err(to_py)?;
    Ok(PyDynamicTable { inner: table })
}
