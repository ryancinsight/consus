//! Hierarchical Data Modeling Framework (HDMF) read/write support for Consus.
//!
//! Provides reading and writing of HDMF [`DynamicTable`] objects stored in
//! HDF5 files.  Compatible with HDMF Python 4.x and NWB 2.x files that embed
//! DynamicTable groups.
//!
//! ## Supported HDMF types
//!
//! | HDMF type            | Rust representation         |
//! |----------------------|-----------------------------|
//! | `DynamicTable`       | [`DynamicTable`]            |
//! | `VectorData` (f64)   | [`ColumnData::F64`]         |
//! | `VectorData` (i64)   | [`ColumnData::I64`]         |
//! | `VectorData` (u64)   | [`ColumnData::U64`]         |
//! | `VectorData` (bool)  | [`ColumnData::Bool`]        |
//! | `VectorData` (str)   | [`ColumnData::Str`]         |
//! | `VectorIndex`        | `Column::index`             |
//! | `ElementIdentifiers` | `DynamicTable::id`          |
//!
//! ## Example — write then read
//!
//! ```no_run
//! use consus_hdmf::{DynamicTable, Column, ColumnData, HdmfFileBuilder, HdmfFile};
//!
//! let bytes = HdmfFileBuilder::new("my_table", "example")
//!     .add_column("x", "x values", ColumnData::F64(vec![1.0, 2.0, 3.0]))
//!     .add_column("y", "y values", ColumnData::I64(vec![4, 5, 6]))
//!     .finish()
//!     .unwrap();
//!
//! let file = HdmfFile::open(&bytes).unwrap();
//! let table = file.read_table().unwrap();
//! assert_eq!(table.colnames, &["x", "y"]);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

/// DynamicTable domain model (columns, data variants).
#[cfg(feature = "alloc")]
pub mod table;

/// HDF5-backed file reader and writer for DynamicTable objects.
#[cfg(feature = "alloc")]
pub mod file;

/// Internal HDF5 read helpers (not part of the public API).
#[cfg(feature = "alloc")]
mod storage;

#[cfg(feature = "alloc")]
pub use table::{Column, ColumnData, DynamicTable};

#[cfg(feature = "alloc")]
pub use file::{HdmfFile, HdmfFileBuilder};
