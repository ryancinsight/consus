//! Parquet Python bindings: `ParquetFile`, `ParquetBuilder`.
//!
//! `ParquetFile` reads an existing Parquet file from raw bytes.
//! `ParquetBuilder` constructs a new single-row-group Parquet file.

use consus_parquet::{
    CellValue, ColumnChunkDescriptor, ColumnValues, FieldDescriptor, FieldId,
    ParquetDatasetDescriptor, ParquetPhysicalType, ParquetReader, ParquetWriter,
    RowGroupDescriptor, RowSource, RowValue, SchemaDescriptor,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

use crate::error::from_consus;

// ---------------------------------------------------------------------------
// Cell value bridge
// ---------------------------------------------------------------------------

/// Owned version of `CellValue` for Python-side storage.
#[derive(Debug, Clone)]
enum OwnedCell {
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    Float(f32),
    Double(f64),
    ByteArray(Vec<u8>),
}

impl OwnedCell {
    fn as_cell(&self) -> CellValue<'_> {
        match self {
            OwnedCell::Boolean(b) => CellValue::Boolean(*b),
            OwnedCell::Int32(i) => CellValue::Int32(*i),
            OwnedCell::Int64(i) => CellValue::Int64(*i),
            OwnedCell::Float(f) => CellValue::Float(*f),
            OwnedCell::Double(d) => CellValue::Double(*d),
            OwnedCell::ByteArray(b) => CellValue::ByteArray(b),
        }
    }
}

// ---------------------------------------------------------------------------
// VecRowSource
// ---------------------------------------------------------------------------

/// Column-oriented in-memory row source for `ParquetWriter`.
struct VecRowSource {
    /// `columns[col_idx][row_idx]`
    columns: Vec<Vec<OwnedCell>>,
    row_count: usize,
}

impl RowSource for VecRowSource {
    fn row_count(&self) -> usize {
        self.row_count
    }

    fn row(&self, index: usize) -> consus_core::Result<RowValue<'_>> {
        let cells: Vec<CellValue<'_>> = self
            .columns
            .iter()
            .map(|col| col[index].as_cell())
            .collect();
        Ok(RowValue::new(cells))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_phys_type(s: &str) -> PyResult<ParquetPhysicalType> {
    match s {
        "BOOLEAN" => Ok(ParquetPhysicalType::Boolean),
        "INT32" => Ok(ParquetPhysicalType::Int32),
        "INT64" => Ok(ParquetPhysicalType::Int64),
        "INT96" => Ok(ParquetPhysicalType::Int96),
        "FLOAT" => Ok(ParquetPhysicalType::Float),
        "DOUBLE" => Ok(ParquetPhysicalType::Double),
        "BYTE_ARRAY" => Ok(ParquetPhysicalType::ByteArray),
        _ => Err(PyValueError::new_err(format!("unknown Parquet type: {s}"))),
    }
}

fn phys_type_str(pt: ParquetPhysicalType) -> &'static str {
    match pt {
        ParquetPhysicalType::Boolean => "BOOLEAN",
        ParquetPhysicalType::Int32 => "INT32",
        ParquetPhysicalType::Int64 => "INT64",
        ParquetPhysicalType::Int96 => "INT96",
        ParquetPhysicalType::Float => "FLOAT",
        ParquetPhysicalType::Double => "DOUBLE",
        ParquetPhysicalType::ByteArray => "BYTE_ARRAY",
        ParquetPhysicalType::FixedLenByteArray(_) => "FIXED_LEN_BYTE_ARRAY",
    }
}

fn column_values_to_python(cv: &ColumnValues, py: Python<'_>) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    match cv {
        ColumnValues::Boolean(v) => {
            for &b in v {
                list.append(b)?;
            }
        }
        ColumnValues::Int32(v) => {
            for &i in v {
                list.append(i)?;
            }
        }
        ColumnValues::Int64(v) => {
            for &i in v {
                list.append(i)?;
            }
        }
        ColumnValues::Float(v) => {
            // Widen to f64 for Python float compatibility.
            for &f in v {
                list.append(f64::from(f))?;
            }
        }
        ColumnValues::Double(v) => {
            for &d in v {
                list.append(d)?;
            }
        }
        ColumnValues::ByteArray(v) => {
            for b in v {
                list.append(PyBytes::new_bound(py, b))?;
            }
        }
        ColumnValues::FixedLenByteArray { values, .. } => {
            for b in values {
                list.append(PyBytes::new_bound(py, b))?;
            }
        }
        ColumnValues::Int96(v) => {
            for arr in v {
                list.append(PyBytes::new_bound(py, arr.as_ref()))?;
            }
        }
    }
    Ok(list.into_any().unbind())
}

// ---------------------------------------------------------------------------
// ParquetFile — reader
// ---------------------------------------------------------------------------

/// Read-only view of an in-memory Parquet file.
///
/// Instantiate with raw Parquet bytes via :meth:`ParquetFile.__init__`.
#[pyclass]
pub struct PyParquetFile {
    bytes: Vec<u8>,
}

#[pymethods]
impl PyParquetFile {
    /// Construct from raw Parquet bytes.
    #[new]
    fn new(data: &[u8]) -> Self {
        Self {
            bytes: data.to_vec(),
        }
    }

    /// Schema as a list of ``(name, physical_type_str)`` tuples.
    fn schema(&self) -> PyResult<Vec<(String, String)>> {
        let reader = ParquetReader::new(&self.bytes).map_err(from_consus)?;
        Ok(reader
            .dataset()
            .schema()
            .fields()
            .iter()
            .map(|f| (f.name().to_owned(), phys_type_str(f.physical_type()).into()))
            .collect())
    }

    /// Total number of logical rows across all row groups.
    fn row_count(&self) -> PyResult<usize> {
        let reader = ParquetReader::new(&self.bytes).map_err(from_consus)?;
        Ok(reader.metadata().num_rows as usize)
    }

    /// Number of row groups.
    fn num_row_groups(&self) -> PyResult<usize> {
        let reader = ParquetReader::new(&self.bytes).map_err(from_consus)?;
        Ok(reader.metadata().row_groups.len())
    }

    /// List of column names in schema order.
    fn column_names(&self) -> PyResult<Vec<String>> {
        let reader = ParquetReader::new(&self.bytes).map_err(from_consus)?;
        Ok(reader
            .dataset()
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().to_owned())
            .collect())
    }

    /// Read a column by ordinal index and row group index.
    ///
    /// Returns a ``list`` of Python-native values. Float columns are widened
    /// to Python ``float`` (``f64``). ``BYTE_ARRAY`` columns return ``bytes``.
    #[pyo3(signature = (col_ordinal, row_group_idx = 0))]
    fn read_column(
        &self,
        col_ordinal: usize,
        row_group_idx: usize,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let reader = ParquetReader::new(&self.bytes).map_err(from_consus)?;
        let cv = reader
            .read_column_chunk(row_group_idx, col_ordinal)
            .map_err(from_consus)?;
        column_values_to_python(&cv, py)
    }
}

// ---------------------------------------------------------------------------
// ParquetBuilder — writer
// ---------------------------------------------------------------------------

/// Single-row-group Parquet file builder.
///
/// Add columns with :meth:`add_column`, then call :meth:`write` with a
/// ``dict[str, list]`` of column data to produce a Parquet file as ``bytes``.
#[pyclass]
pub struct PyParquetBuilder {
    fields: Vec<(String, ParquetPhysicalType)>,
}

#[pymethods]
impl PyParquetBuilder {
    #[new]
    fn new() -> Self {
        Self { fields: Vec::new() }
    }

    /// Add a column to the schema.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Column name.
    /// dtype : str
    ///     Parquet physical type: ``"INT32"``, ``"INT64"``, ``"FLOAT"``,
    ///     ``"DOUBLE"``, ``"BYTE_ARRAY"``, ``"BOOLEAN"``, ``"INT96"``.
    fn add_column(&mut self, name: &str, dtype: &str) -> PyResult<()> {
        let pt = parse_phys_type(dtype)?;
        self.fields.push((name.to_owned(), pt));
        Ok(())
    }

    /// List of column names in schema order.
    fn column_names(&self) -> Vec<String> {
        self.fields.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Number of columns in the schema.
    fn num_columns(&self) -> usize {
        self.fields.len()
    }

    /// Write a single-row-group Parquet file from column data.
    ///
    /// Parameters
    /// ----------
    /// columns : dict[str, list]
    ///     Column data keyed by column name.  All lists must have the same
    ///     length.
    ///
    /// Returns
    /// -------
    /// bytes
    ///     Complete Parquet file bytes.
    fn write(
        &self,
        columns: &Bound<'_, pyo3::types::PyDict>,
        py: Python<'_>,
    ) -> PyResult<Py<PyBytes>> {
        if self.fields.is_empty() {
            return Err(PyValueError::new_err("no columns defined"));
        }

        // Build schema using FieldDescriptor::required (Required, no logical type, no children).
        let schema_fields: Vec<FieldDescriptor> = self
            .fields
            .iter()
            .enumerate()
            .map(|(i, (name, pt))| {
                FieldDescriptor::required(FieldId::new(i as u32), name.as_str(), *pt)
            })
            .collect();
        let schema = SchemaDescriptor::new(schema_fields);

        // Extract column data from Python dict in schema order.
        let mut col_data: Vec<Vec<OwnedCell>> = Vec::with_capacity(self.fields.len());
        let mut row_count: Option<usize> = None;

        for (name, pt) in &self.fields {
            let py_col = columns
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing column: {name}")))?;
            let py_list: &Bound<'_, PyList> = py_col
                .downcast()
                .map_err(|_| PyTypeError::new_err(format!("column '{name}' must be a list")))?;
            let n = py_list.len();
            if let Some(expected) = row_count {
                if n != expected {
                    return Err(PyValueError::new_err(format!(
                        "column '{name}' length {n} != expected {expected}"
                    )));
                }
            } else {
                row_count = Some(n);
            }

            let mut cells = Vec::with_capacity(n);
            for item in py_list.iter() {
                let cell = match pt {
                    ParquetPhysicalType::Boolean => OwnedCell::Boolean(item.extract::<bool>()?),
                    ParquetPhysicalType::Int32 => OwnedCell::Int32(item.extract::<i32>()?),
                    ParquetPhysicalType::Int64 => OwnedCell::Int64(item.extract::<i64>()?),
                    ParquetPhysicalType::Float => OwnedCell::Float(item.extract::<f32>()?),
                    ParquetPhysicalType::Double => OwnedCell::Double(item.extract::<f64>()?),
                    ParquetPhysicalType::ByteArray => {
                        let b: &[u8] = item.extract()?;
                        OwnedCell::ByteArray(b.to_vec())
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "unsupported physical type for write: {}",
                            phys_type_str(*pt)
                        )));
                    }
                };
                cells.push(cell);
            }
            col_data.push(cells);
        }

        let row_count = row_count.unwrap_or(0);
        if row_count == 0 {
            return Err(PyValueError::new_err(
                "cannot write Parquet file with 0 rows",
            ));
        }

        // Build dataset descriptor.
        let chunks: Vec<ColumnChunkDescriptor> = (0..self.fields.len())
            .map(|i| {
                ColumnChunkDescriptor::new(FieldId::new(i as u32), row_count, row_count)
                    .map_err(from_consus)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let row_group = RowGroupDescriptor::new(row_count, chunks).map_err(from_consus)?;
        let dataset =
            ParquetDatasetDescriptor::new(schema, vec![row_group]).map_err(from_consus)?;

        let row_source = VecRowSource {
            columns: col_data,
            row_count,
        };

        let bytes = ParquetWriter::new()
            .write_dataset_bytes(&dataset, &row_source)
            .map_err(from_consus)?;
        Ok(PyBytes::new_bound(py, &bytes).unbind())
    }
}
