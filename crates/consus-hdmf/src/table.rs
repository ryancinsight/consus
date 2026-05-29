//! DynamicTable domain model — columns, data variants, table struct.
//!
//! ## Format invariants
//!
//! - All columns in a `DynamicTable` have equal length equal to `id.len()`.
//! - Column names in `colnames` map 1-to-1 to `columns` in order.
//! - A ragged column carries a non-`None` `index` field; its cumulative
//!   index length equals the number of rows.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

/// Typed payload of a single DynamicTable column.
///
/// Each variant maps to the corresponding HDMF `VectorData` dtype:
///
/// | Variant | HDF5 dtype     |
/// |---------|----------------|
/// | `F64`   | float64        |
/// | `I64`   | int64 (signed) |
/// | `U64`   | uint64         |
/// | `Bool`  | uint8 (0/1)    |
/// | `Str`   | fixed-string   |
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    /// 64-bit IEEE 754 floating-point values.
    F64(Vec<f64>),
    /// Signed 64-bit integer values.
    I64(Vec<i64>),
    /// Unsigned 64-bit integer values.
    U64(Vec<u64>),
    /// Boolean values stored as 1-byte unsigned integers.
    Bool(Vec<bool>),
    /// Variable-length string values.
    Str(Vec<String>),
}

#[cfg(feature = "alloc")]
impl ColumnData {
    /// Number of elements in this column.
    pub fn len(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::Str(v) => v.len(),
        }
    }

    /// Returns `true` when this column contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A single named column of a [`DynamicTable`].
///
/// Corresponds to an HDMF `VectorData` dataset.  When the column is ragged
/// (variable-length rows), `index` carries the cumulative end indices of each
/// row (a `VectorIndex` in HDMF terms).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct Column {
    /// Column name, matching the dataset name in the HDF5 file.
    pub name: String,
    /// Human-readable description of the column contents.
    pub description: String,
    /// Column payload.
    pub data: ColumnData,
    /// Cumulative row-end indices for ragged columns (`VectorIndex`).
    ///
    /// `None` for dense (uniform-length) columns.
    pub index: Option<Vec<u64>>,
}

/// An HDMF DynamicTable read from or ready to be written to an HDF5 file.
///
/// ## Invariants
///
/// - `id.len()` equals the number of rows.
/// - `colnames.len()` equals `columns.len()`.
/// - Each `colnames[i]` equals `columns[i].name`.
/// - Each column's `ColumnData::len()` equals `id.len()`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DynamicTable {
    /// Table name (HDF5 group name or, for root tables, a logical name).
    pub name: String,
    /// Human-readable description of the table.
    pub description: String,
    /// Ordered list of column names.  Maps 1-to-1 to `columns`.
    pub colnames: Vec<String>,
    /// Row identifiers (`ElementIdentifiers` in HDMF terms).
    pub id: Vec<i64>,
    /// Column objects in the same order as `colnames`.
    pub columns: Vec<Column>,
}
