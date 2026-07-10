//! Typed NPY and NPZ storage without an ndarray dependency.
//!
//! The implementation follows NumPy's published NPY format contract: magic
//! and version bytes, a length-prefixed dictionary header, and contiguous
//! typed payload bytes. NPZ is a ZIP archive containing named NPY members.
//!
//! Format reference: <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>

#![forbid(unsafe_code)]

mod array;
mod error;
mod format;
mod npz;

pub use array::{NpyArray, NpyElement};
pub use error::{Error, Result};
pub use format::{read_npy, write_npy};
pub use npz::{NpzReader, NpzWriter};
