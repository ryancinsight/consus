//! HDF5-backed storage helpers for NWB readers.
//!
//! Provides typed attribute and dataset access utilities used by
//! [`crate::file`] to extract NWB session metadata and neurodata values
//! from an HDF5 file without duplicating low-level decoding logic.
//!
//! ## Scope
//!
//! | Helper                    | Input                       | Output         |
//! |---------------------------|-----------------------------|----------------|
//! | [`read_string_attr`]      | attribute list + name       | `String`       |
//! | [`read_f64_attr`]         | attribute list + name       | `f64`          |
//! | [`read_f64_dataset`]      | `Hdf5File` + object address | `Vec<f64>`     |
//! | [`read_scalar_f64_dataset`]| `Hdf5File` + object address | `f64`         |
//! | [`read_string_dataset`]        | `Hdf5File` + object address | `Vec<String>` (FixedString + VariableString) |
//! | [`read_scalar_string_dataset`] | `Hdf5File` + object address | `String`       |
//! | [`read_u64_dataset`]           | `Hdf5File` + object address | `Vec<u64>`     |
//!
//! All helpers propagate [`consus_core::Error`] variants directly; no
//! intermediate error type is introduced.
//!
//! ## Invariants
//!
//! - [`read_string_attr`] returns `Error::NotFound` when the named attribute
//!   is absent; `Error::InvalidFormat` when the attribute exists but its
//!   decoded value is not a `String`.
//! - [`read_f64_attr`] returns `Error::NotFound` when absent; `Error::InvalidFormat`
//!   when the attribute exists but is non-numeric.
//! - [`read_f64_dataset`] supports `f32`, `f64`, and signed/unsigned integer
//!   (8, 16, 32, 64-bit) contiguous and chunked datasets. All values are
//!   promoted to `f64` (IEEE 754 double).
//! - [`read_scalar_f64_dataset`] is a thin wrapper over [`read_f64_dataset`]
//!   that extracts the single first element.
//! - All functions are pure with respect to the file image: they read but
//!   never mutate the underlying HDF5 source.

mod attribute;
mod dataset;

#[cfg(feature = "alloc")]
pub use attribute::{read_f64_attr, read_string_attr};
#[cfg(feature = "alloc")]
pub use dataset::{
    read_f64_dataset, read_scalar_f64_dataset, read_scalar_string_dataset, read_string_dataset,
    read_u64_dataset,
};
