//! Parquet physical type identifiers and width semantics.
//!
//! ## Specification
//!
//! Physical types define the on-disk primitive encoding independent of
//! logical annotations. This module is the canonical source of physical
//! type identities used by the `consus-parquet` schema model.
//!
//! ## Invariants
//!
//! - Each physical type has one canonical width classification.
//! - Width is expressed in bytes when statically known.
//! - Variable-width encodings return `None` for fixed width.
//! - Logical annotations are handled in sibling modules.

/// Parquet primitive physical types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParquetPhysicalType {
    /// Boolean bit-packed logical values.
    Boolean,
    /// Signed or unsigned 32-bit integer storage.
    Int32,
    /// Signed or unsigned 64-bit integer storage.
    Int64,
    /// 96-bit timestamp storage.
    Int96,
    /// IEEE 754 single-precision float.
    Float,
    /// IEEE 754 double-precision float.
    Double,
    /// Length-prefixed byte array.
    ByteArray,
    /// Fixed-size byte array with compile-time byte width.
    FixedLenByteArray(usize),
}

impl ParquetPhysicalType {
    /// Return the static width in bytes when available.
    ///
    /// `ByteArray` is variable-width and returns `None`.
    #[must_use]
    pub const fn width(self) -> Option<usize> {
        match self {
            Self::Boolean => Some(1),
            Self::Int32 => Some(4),
            Self::Int64 => Some(8),
            Self::Int96 => Some(12),
            Self::Float => Some(4),
            Self::Double => Some(8),
            Self::ByteArray => None,
            Self::FixedLenByteArray(width) => Some(width),
        }
    }

    /// Whether the physical type uses variable-width storage.
    #[must_use]
    pub const fn is_variable_width(self) -> bool {
        matches!(self, Self::ByteArray)
    }

    /// Whether the physical type represents integer storage.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::Int32 | Self::Int64 | Self::Int96)
    }

    /// Whether the physical type represents floating-point storage.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float | Self::Double)
    }
}

/// Alias describing the width of a physical Parquet type in bytes.
///
/// `None` indicates a variable-width physical type such as `ByteArray`.
pub type ParquetPhysicalWidth = Option<usize>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_mapping_is_correct() {
        assert_eq!(ParquetPhysicalType::Boolean.width(), Some(1));
        assert_eq!(ParquetPhysicalType::Int32.width(), Some(4));
        assert_eq!(ParquetPhysicalType::Int64.width(), Some(8));
        assert_eq!(ParquetPhysicalType::Int96.width(), Some(12));
        assert_eq!(ParquetPhysicalType::Float.width(), Some(4));
        assert_eq!(ParquetPhysicalType::Double.width(), Some(8));
        assert_eq!(ParquetPhysicalType::ByteArray.width(), None);
        assert_eq!(ParquetPhysicalType::FixedLenByteArray(16).width(), Some(16));
    }

    #[test]
    fn classification_predicates_are_correct() {
        assert!(ParquetPhysicalType::Int32.is_integer());
        assert!(ParquetPhysicalType::Int64.is_integer());
        assert!(ParquetPhysicalType::Int96.is_integer());
        assert!(!ParquetPhysicalType::Double.is_integer());

        assert!(ParquetPhysicalType::Float.is_float());
        assert!(ParquetPhysicalType::Double.is_float());
        assert!(!ParquetPhysicalType::Int32.is_float());

        assert!(ParquetPhysicalType::ByteArray.is_variable_width());
        assert!(!ParquetPhysicalType::FixedLenByteArray(8).is_variable_width());
    }
}
