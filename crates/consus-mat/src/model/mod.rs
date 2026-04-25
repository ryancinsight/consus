//! Canonical MATLAB array model.
//!
//! ## Design
//!
//! Every MATLAB value loaded from a .mat file is represented as a
//! [`MatArray`] variant. This provides a single heterogeneous container
//! without format-version-specific leakage into the public API.
//!
//! ## Variant Coverage
//!
//! | MATLAB class   | Rust variant                  |
//! |----------------|-------------------------------|
//! | double/single/intN/uintN | [`MatArray::Numeric`] |
//! | char           | [`MatArray::Char`]            |
//! | logical        | [`MatArray::Logical`]         |
//! | sparse         | [`MatArray::Sparse`]          |
//! | cell           | [`MatArray::Cell`]            |
//! | struct         | [`MatArray::Struct`]          |

pub mod cell;
pub mod character;
pub mod logical;
pub mod numeric;
pub mod sparse;
pub mod structure;

#[cfg(feature = "alloc")]
pub use cell::MatCellArray;
#[cfg(feature = "alloc")]
pub use character::MatCharArray;
#[cfg(feature = "alloc")]
pub use logical::MatLogicalArray;
pub use numeric::{MatNumericClass, MatNumericArray};
#[cfg(feature = "alloc")]
pub use sparse::MatSparseArray;
#[cfg(feature = "alloc")]
pub use structure::MatStructArray;

/// Top-level MATLAB array variant.
///
/// Returned by [`crate::loadmat_bytes`] and [`crate::loadmat`] as the
/// value type in the variable map.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub enum MatArray {
    /// Dense numeric array (real or complex).
    Numeric(MatNumericArray),
    /// Character array (string).
    Char(MatCharArray),
    /// Logical (boolean) array.
    Logical(MatLogicalArray),
    /// Sparse matrix in CSC format.
    Sparse(MatSparseArray),
    /// Cell array (heterogeneous elements).
    Cell(MatCellArray),
    /// Struct array (named fields).
    Struct(MatStructArray),
}
