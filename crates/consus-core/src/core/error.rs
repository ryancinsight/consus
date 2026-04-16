//! Comprehensive error hierarchy for the Consus storage library.
//!
//! ## Design
//!
//! Errors preserve full context: the operation attempted, the path within
//! the hierarchy, and the underlying cause. Variants are non-overlapping:
//! a given failure mode maps to exactly one variant.
//!
//! ## `no_std` Compatibility
//!
//! - Without `alloc`: error variants carry no heap-allocated context.
//! - With `alloc`: variants include `String` fields for diagnostics.
//! - With `std`: `std::error::Error` and `From<std::io::Error>` are available.

#[cfg(feature = "alloc")]
use alloc::string::String;

/// Unified error type for all Consus operations.
///
/// Each variant captures a distinct failure mode. Variants are non-overlapping:
/// any given failure maps to exactly one variant.
#[derive(Debug)]
pub enum Error {
    /// The requested path does not exist in the hierarchy.
    NotFound {
        /// Absolute path within the container (e.g., `/group/dataset`).
        #[cfg(feature = "alloc")]
        path: String,
    },

    /// A structural or semantic format constraint was violated.
    InvalidFormat {
        /// Human-readable description of the violation.
        #[cfg(feature = "alloc")]
        message: String,
    },

    /// The requested datatype conversion is not supported or would lose data.
    DatatypeMismatch {
        /// Expected datatype description.
        #[cfg(feature = "alloc")]
        expected: String,
        /// Found datatype description.
        #[cfg(feature = "alloc")]
        found: String,
    },

    /// A dimension, rank, or shape constraint was violated.
    ShapeError {
        /// Description of the shape violation.
        #[cfg(feature = "alloc")]
        message: String,
    },

    /// A selection (hyperslab, point list) exceeds the dataspace bounds.
    SelectionOutOfBounds,

    /// Compression or decompression failed.
    CompressionError {
        /// Description of the compression failure.
        #[cfg(feature = "alloc")]
        message: String,
    },

    /// Underlying I/O error (requires `std` feature).
    #[cfg(feature = "std")]
    Io(std::io::Error),

    /// An operation requires a feature or capability not available.
    UnsupportedFeature {
        /// Description of the missing feature.
        #[cfg(feature = "alloc")]
        feature: String,
    },

    /// The file or container is corrupted or structurally invalid.
    Corrupted {
        /// Description of the corruption.
        #[cfg(feature = "alloc")]
        message: String,
    },

    /// A caller-provided buffer is too small for the requested operation.
    BufferTooSmall {
        /// Minimum required buffer size in bytes.
        required: usize,
        /// Actual buffer size provided.
        provided: usize,
    },

    /// An operation was attempted on a closed or invalid handle.
    InvalidHandle,

    /// A write operation exceeded the dataset's capacity or dimension limit.
    CapacityExceeded {
        /// Description of the capacity violation.
        #[cfg(feature = "alloc")]
        message: String,
    },

    /// An internal consistency check failed (indicates a library bug).
    InternalError {
        /// Description of the internal failure.
        #[cfg(feature = "alloc")]
        message: String,
    },

    /// A link target could not be resolved (dangling soft/external link).
    LinkResolutionFailed {
        /// The unresolvable link path.
        #[cfg(feature = "alloc")]
        path: String,
    },

    /// An integer overflow occurred during size or offset computation.
    Overflow,

    /// Permission or access mode violation (e.g., write on read-only file).
    ReadOnly,
}

impl ::core::fmt::Display for Error {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            #[cfg(feature = "alloc")]
            Self::NotFound { path } => write!(f, "path not found: {path}"),
            #[cfg(not(feature = "alloc"))]
            Self::NotFound {} => write!(f, "path not found"),

            #[cfg(feature = "alloc")]
            Self::InvalidFormat { message } => write!(f, "invalid format: {message}"),
            #[cfg(not(feature = "alloc"))]
            Self::InvalidFormat {} => write!(f, "invalid format"),

            #[cfg(feature = "alloc")]
            Self::DatatypeMismatch { expected, found } => {
                write!(f, "datatype mismatch: expected {expected}, found {found}")
            }
            #[cfg(not(feature = "alloc"))]
            Self::DatatypeMismatch {} => write!(f, "datatype mismatch"),

            #[cfg(feature = "alloc")]
            Self::ShapeError { message } => write!(f, "shape error: {message}"),
            #[cfg(not(feature = "alloc"))]
            Self::ShapeError {} => write!(f, "shape error"),

            Self::SelectionOutOfBounds => write!(f, "selection out of bounds"),

            #[cfg(feature = "alloc")]
            Self::CompressionError { message } => write!(f, "compression error: {message}"),
            #[cfg(not(feature = "alloc"))]
            Self::CompressionError {} => write!(f, "compression error"),

            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "I/O error: {e}"),

            #[cfg(feature = "alloc")]
            Self::UnsupportedFeature { feature } => {
                write!(f, "unsupported feature: {feature}")
            }
            #[cfg(not(feature = "alloc"))]
            Self::UnsupportedFeature {} => write!(f, "unsupported feature"),

            #[cfg(feature = "alloc")]
            Self::Corrupted { message } => write!(f, "corrupted: {message}"),
            #[cfg(not(feature = "alloc"))]
            Self::Corrupted {} => write!(f, "corrupted"),

            Self::BufferTooSmall { required, provided } => {
                write!(f, "buffer too small: need {required} bytes, got {provided}")
            }

            Self::InvalidHandle => write!(f, "invalid or closed handle"),

            #[cfg(feature = "alloc")]
            Self::CapacityExceeded { message } => {
                write!(f, "capacity exceeded: {message}")
            }
            #[cfg(not(feature = "alloc"))]
            Self::CapacityExceeded {} => write!(f, "capacity exceeded"),

            #[cfg(feature = "alloc")]
            Self::InternalError { message } => write!(f, "internal error: {message}"),
            #[cfg(not(feature = "alloc"))]
            Self::InternalError {} => write!(f, "internal error"),

            #[cfg(feature = "alloc")]
            Self::LinkResolutionFailed { path } => {
                write!(f, "link resolution failed: {path}")
            }
            #[cfg(not(feature = "alloc"))]
            Self::LinkResolutionFailed {} => write!(f, "link resolution failed"),

            Self::Overflow => write!(f, "integer overflow in size computation"),

            Self::ReadOnly => write!(f, "write operation on read-only handle"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Type alias for `core::result::Result<T, Error>`.
pub type Result<T> = ::core::result::Result<T, Error>;
