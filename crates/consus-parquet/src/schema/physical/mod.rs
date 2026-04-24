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

    /// Map a `parquet.thrift` Type enum i32 discriminant to the physical type.
    ///
    /// Discriminant mapping (parquet.thrift Type enum):
    ///
    /// | Value | Parquet type            |
    /// |-------|-------------------------|
    /// |     0 | BOOLEAN                 |
    /// |     1 | INT32                   |
    /// |     2 | INT64                   |
    /// |     3 | INT96                   |
    /// |     4 | FLOAT                   |
    /// |     5 | DOUBLE                  |
    /// |     6 | BYTE_ARRAY              |
    /// |     7 | FIXED_LEN_BYTE_ARRAY(0) |
    ///
    /// Returns `None` for unknown discriminants.
    /// For `FIXED_LEN_BYTE_ARRAY` (discriminant 7) the fixed length is set to 0;
    /// use [`Self::from_parquet_type_with_length`] when `type_length` is available.
    #[must_use]
    pub const fn from_parquet_type_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Boolean),
            1 => Some(Self::Int32),
            2 => Some(Self::Int64),
            3 => Some(Self::Int96),
            4 => Some(Self::Float),
            5 => Some(Self::Double),
            6 => Some(Self::ByteArray),
            7 => Some(Self::FixedLenByteArray(0)),
            _ => None,
        }
    }

    /// Map a `parquet.thrift` Type enum i32 discriminant to the physical type,
    /// using `type_length` to supply the fixed byte length for
    /// `FIXED_LEN_BYTE_ARRAY` (discriminant 7).
    ///
    /// When `v == 7` and `type_length` is `Some(n)` with `n >= 0`, returns
    /// `FixedLenByteArray(n as usize)`. When `type_length` is `None` or
    /// negative, falls back to `FixedLenByteArray(0)`.
    /// For all other discriminants delegates to [`Self::from_parquet_type_i32`].
    #[must_use]
    pub fn from_parquet_type_with_length(v: i32, type_length: Option<i32>) -> Option<Self> {
        if v == 7 {
            let len = type_length.unwrap_or(0).max(0) as usize;
            Some(Self::FixedLenByteArray(len))
        } else {
            Self::from_parquet_type_i32(v)
        }
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

    #[test]
    fn from_parquet_type_i32_all_known_discriminants() {
        assert_eq!(ParquetPhysicalType::from_parquet_type_i32(0), Some(ParquetPhysicalType::Boolean));
        assert_eq!(ParquetPhysicalType::from_parquet_type_i32(1), Some(ParquetPhysicalType::Int32));
        assert_eq!(ParquetPhysicalType::from_parquet_type_i32(2), Some(ParquetPhysicalType::Int64));
        assert_eq!(ParquetPhysicalType::from_parquet_type_i32(3), Some(ParquetPhysicalType::Int96));
        assert_eq!(ParquetPhysicalType::from_parquet_type_i32(4), Some(ParquetPhysicalType::Float));
        assert_eq!(ParquetPhysicalType::from_parquet_type_i32(5), Some(ParquetPhysicalType::Double));
        assert_eq!(ParquetPhysicalType::from_parquet_type_i32(6), Some(ParquetPhysicalType::ByteArray));
        assert_eq!(ParquetPhysicalType::from_parquet_type_i32(7), Some(ParquetPhysicalType::FixedLenByteArray(0)));
        assert_eq!(ParquetPhysicalType::from_parquet_type_i32(99), None);
    }

    #[test]
    fn from_parquet_type_with_length_fixed_len_byte_array() {
        assert_eq!(
            ParquetPhysicalType::from_parquet_type_with_length(7, Some(16)),
            Some(ParquetPhysicalType::FixedLenByteArray(16))
        );
        assert_eq!(
            ParquetPhysicalType::from_parquet_type_with_length(7, None),
            Some(ParquetPhysicalType::FixedLenByteArray(0))
        );
        assert_eq!(
            ParquetPhysicalType::from_parquet_type_with_length(1, Some(4)),
            Some(ParquetPhysicalType::Int32)
        );
    }
}
