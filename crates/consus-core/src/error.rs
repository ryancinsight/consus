//! Error types for the Consus storage library.
//!
//! ## Design
//!
//! Errors preserve full context: the operation attempted, the path within the
//! hierarchy, and the underlying cause. This enables precise diagnostics in
//! multi-format pipelines where a single logical operation may traverse
//! multiple storage backends.

#[cfg(feature = "alloc")]
use alloc::string::String;

use core::fmt;

/// Unified error type for all Consus operations.
///
/// Each variant captures a distinct failure mode. Variants are non-overlapping:
/// a given failure maps to exactly one variant.
#[derive(Debug)]
pub enum Error {
    /// The requested path does not exist in the hierarchy.
    NotFound {
        /// Absolute path within the container (e.g., `/group/dataset`).
        #[cfg(feature = "alloc")]
        path: String,
    },

    /// A structural or semantic constraint was violated.
    InvalidFormat {
        /// Human-readable description of the violation.
        #[cfg(feature = "alloc")]
        message: String,
    },

    /// The requested datatype conversion is not supported or would lose data.
    DatatypeMismatch {
        #[cfg(feature = "alloc")]
        expected: String,
        #[cfg(feature = "alloc")]
        found: String,
    },

    /// A dimension or shape constraint was violated.
    ShapeError {
        #[cfg(feature = "alloc")]
        message: String,
    },

    /// A selection (hyperslab, point list, etc.) is out of bounds.
    SelectionOutOfBounds,

    /// Compression or decompression failed.
    CompressionError {
        #[cfg(feature = "alloc")]
        message: String,
    },

    /// Underlying I/O error (only available with `std` feature).
    #[cfg(feature = "std")]
    Io(std::io::Error),

    /// An operation was attempted that requires a feature not enabled.
    UnsupportedFeature {
        #[cfg(feature = "alloc")]
        feature: String,
    },

    /// The file or container is corrupted.
    Corrupted {
        #[cfg(feature = "alloc")]
        message: String,
    },

    /// A buffer provided by the caller is too small.
    BufferTooSmall {
        required: usize,
        provided: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "alloc")]
            Error::NotFound { path } => write!(f, "path not found: {path}"),
            #[cfg(not(feature = "alloc"))]
            Error::NotFound {} => write!(f, "path not found"),

            #[cfg(feature = "alloc")]
            Error::InvalidFormat { message } => write!(f, "invalid format: {message}"),
            #[cfg(not(feature = "alloc"))]
            Error::InvalidFormat {} => write!(f, "invalid format"),

            #[cfg(feature = "alloc")]
            Error::DatatypeMismatch { expected, found } => {
                write!(f, "datatype mismatch: expected {expected}, found {found}")
            }
            #[cfg(not(feature = "alloc"))]
            Error::DatatypeMismatch {} => write!(f, "datatype mismatch"),

            #[cfg(feature = "alloc")]
            Error::ShapeError { message } => write!(f, "shape error: {message}"),
            #[cfg(not(feature = "alloc"))]
            Error::ShapeError {} => write!(f, "shape error"),

            Error::SelectionOutOfBounds => write!(f, "selection out of bounds"),

            #[cfg(feature = "alloc")]
            Error::CompressionError { message } => write!(f, "compression error: {message}"),
            #[cfg(not(feature = "alloc"))]
            Error::CompressionError {} => write!(f, "compression error"),

            #[cfg(feature = "std")]
            Error::Io(e) => write!(f, "I/O error: {e}"),

            #[cfg(feature = "alloc")]
            Error::UnsupportedFeature { feature } => {
                write!(f, "unsupported feature: {feature}")
            }
            #[cfg(not(feature = "alloc"))]
            Error::UnsupportedFeature {} => write!(f, "unsupported feature"),

            #[cfg(feature = "alloc")]
            Error::Corrupted { message } => write!(f, "corrupted: {message}"),
            #[cfg(not(feature = "alloc"))]
            Error::Corrupted {} => write!(f, "corrupted"),

            Error::BufferTooSmall { required, provided } => {
                write!(f, "buffer too small: need {required} bytes, got {provided}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Type alias for `core::result::Result<T, Error>`.
pub type Result<T> = core::result::Result<T, Error>;
