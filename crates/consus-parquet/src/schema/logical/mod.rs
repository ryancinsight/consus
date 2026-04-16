//! Parquet logical type annotations and semantic constraints.
//!
//! ## Specification
//!
//! Logical types add semantic meaning to Parquet physical types without
//! changing the underlying storage layout. This module provides the
//! canonical logical annotation model used by `consus-parquet`.
//!
//! ## Invariants
//!
//! - A logical annotation refines a physical type.
//! - A logical annotation never changes the physical width.
//! - Schema evolution may relax or extend logical constraints, but it
//!   must preserve field identity and physical compatibility.
//!
//! ## Mapping Summary
//!
//! | Logical annotation | Typical physical type |
//! |---|---|
//! | `STRING` | `BYTE_ARRAY` or `FIXED_LEN_BYTE_ARRAY` |
//! | `ENUM` | `BYTE_ARRAY` |
//! | `DECIMAL` | `INT32`, `INT64`, `BYTE_ARRAY`, `FIXED_LEN_BYTE_ARRAY` |
//! | `DATE` | `INT32` |
//! | `TIME` | `INT32`, `INT64` |
//! | `TIMESTAMP` | `INT64` |
//! | `UUID` | `FIXED_LEN_BYTE_ARRAY(16)` |
//!
//! The module is intentionally small and canonical. Higher-level schema
//! code composes these annotations with physical types and field metadata.

#[cfg(feature = "alloc")]
use alloc::string::String;

/// Time unit used by Parquet temporal logical types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// Milliseconds since epoch or since midnight.
    Milliseconds,
    /// Microseconds since epoch or since midnight.
    Microseconds,
    /// Nanoseconds since epoch or since midnight.
    Nanoseconds,
}

/// Parquet logical type annotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicalType {
    /// UTF-8 string semantics.
    String,
    /// Enumeration semantics.
    Enum,
    /// Signed or unsigned decimal semantics.
    Decimal {
        /// Number of digits of precision.
        precision: u32,
        /// Number of digits to the right of the decimal point.
        scale: i32,
    },
    /// Date stored as days since the Unix epoch.
    Date,
    /// Time stored as a count of units since midnight.
    Time {
        /// Time unit.
        unit: TimeUnit,
        /// Whether the value is adjusted to UTC.
        is_adjusted_to_utc: bool,
    },
    /// Timestamp stored as a count of units since the Unix epoch.
    Timestamp {
        /// Time unit.
        unit: TimeUnit,
        /// Whether the value is adjusted to UTC.
        is_adjusted_to_utc: bool,
    },
    /// Signed integer with a constrained bit width.
    Integer {
        /// Bit width.
        bit_width: u8,
        /// Whether the integer is signed.
        signed: bool,
    },
    /// Unsigned integer with a constrained bit width.
    UnsignedInteger {
        /// Bit width.
        bit_width: u8,
    },
    /// JSON-encoded textual value.
    Json,
    /// BSON-encoded binary value.
    Bson,
    /// UUID stored in 16 bytes.
    Uuid,
    /// Arbitrary logical annotation for extension use.
    #[cfg(feature = "alloc")]
    Extension {
        /// Extension name.
        name: String,
    },
}

impl LogicalType {
    /// Returns `true` if the logical type requires a textual representation.
    #[must_use]
    pub fn is_string_like(&self) -> bool {
        matches!(self, Self::String | Self::Json)
    }

    /// Returns `true` if the logical type is temporal.
    #[must_use]
    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            Self::Date | Self::Time { .. } | Self::Timestamp { .. }
        )
    }

    /// Returns `true` if the logical type is numeric.
    #[must_use]
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Self::Decimal { .. } | Self::Integer { .. } | Self::UnsignedInteger { .. }
        )
    }
}

/// Semantic constraint describing whether a field may be absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nullability {
    /// Field must be present in every row.
    Required,
    /// Field may be absent.
    Optional,
}

/// Repetition level for a field in a Parquet schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Repetition {
    /// Required field.
    Required,
    /// Optional field.
    Optional,
    /// Repeated field.
    Repeated,
}

/// Canonical description of a schema-level semantic annotation.
///
/// This type combines logical annotation, nullability, and repetition
/// into one reusable descriptor used by higher-level field metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAnnotation {
    /// Logical semantic meaning, if any.
    pub logical_type: Option<LogicalType>,
    /// Nullability of the field.
    pub nullability: Nullability,
    /// Repetition semantics.
    pub repetition: Repetition,
}

impl TypeAnnotation {
    /// Create a required field annotation with no logical type.
    #[must_use]
    pub const fn required() -> Self {
        Self {
            logical_type: None,
            nullability: Nullability::Required,
            repetition: Repetition::Required,
        }
    }

    /// Create an optional field annotation with no logical type.
    #[must_use]
    pub const fn optional() -> Self {
        Self {
            logical_type: None,
            nullability: Nullability::Optional,
            repetition: Repetition::Optional,
        }
    }

    /// Attach a logical type annotation.
    #[must_use]
    pub fn with_logical_type(mut self, logical_type: LogicalType) -> Self {
        self.logical_type = Some(logical_type);
        self
    }

    /// Returns `true` if the field is optional.
    #[must_use]
    pub fn is_optional(&self) -> bool {
        matches!(self.nullability, Nullability::Optional)
    }

    /// Returns `true` if the field is repeated.
    #[must_use]
    pub fn is_repeated(&self) -> bool {
        matches!(self.repetition, Repetition::Repeated)
    }
}

/// Resolve whether a logical type is compatible with a physical width.
///
/// This function encodes the canonical validation rules used by the
/// schema layer. It rejects mismatched combinations instead of silently
/// weakening the schema.
#[must_use]
pub fn is_compatible_with_width(logical: &LogicalType, width_bytes: usize) -> bool {
    match logical {
        LogicalType::String | LogicalType::Enum | LogicalType::Json | LogicalType::Bson => {
            width_bytes > 0
        }
        LogicalType::Decimal { .. } => width_bytes > 0,
        LogicalType::Date => width_bytes == 4,
        LogicalType::Time { unit, .. } => match unit {
            TimeUnit::Milliseconds => width_bytes == 4,
            TimeUnit::Microseconds | TimeUnit::Nanoseconds => width_bytes == 8,
        },
        LogicalType::Timestamp { unit, .. } => match unit {
            TimeUnit::Milliseconds => width_bytes == 8,
            TimeUnit::Microseconds | TimeUnit::Nanoseconds => width_bytes == 8,
        },
        LogicalType::Integer { bit_width, .. } | LogicalType::UnsignedInteger { bit_width } => {
            usize::from(*bit_width) / 8 == width_bytes
        }
        LogicalType::Uuid => width_bytes == 16,
        #[cfg(feature = "alloc")]
        LogicalType::Extension { .. } => width_bytes > 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_types_are_detected() {
        assert!(LogicalType::Date.is_temporal());
        assert!(
            LogicalType::Time {
                unit: TimeUnit::Milliseconds,
                is_adjusted_to_utc: false,
            }
            .is_temporal()
        );
        assert!(
            LogicalType::Timestamp {
                unit: TimeUnit::Microseconds,
                is_adjusted_to_utc: true,
            }
            .is_temporal()
        );
        assert!(!LogicalType::String.is_temporal());
    }

    #[test]
    fn numeric_types_are_detected() {
        assert!(
            LogicalType::Decimal {
                precision: 10,
                scale: 2,
            }
            .is_numeric()
        );
        assert!(
            LogicalType::Integer {
                bit_width: 32,
                signed: true,
            }
            .is_numeric()
        );
        assert!(LogicalType::UnsignedInteger { bit_width: 16 }.is_numeric());
        assert!(!LogicalType::Uuid.is_numeric());
    }

    #[test]
    fn compatibility_rules_match_expected_widths() {
        assert!(is_compatible_with_width(&LogicalType::Date, 4));
        assert!(is_compatible_with_width(
            &LogicalType::Timestamp {
                unit: TimeUnit::Milliseconds,
                is_adjusted_to_utc: true,
            },
            8
        ));
        assert!(is_compatible_with_width(&LogicalType::Uuid, 16));
        assert!(!is_compatible_with_width(&LogicalType::Uuid, 8));
        assert!(!is_compatible_with_width(&LogicalType::Date, 8));
    }

    #[test]
    fn annotation_flags_are_consistent() {
        let annotation = TypeAnnotation::optional().with_logical_type(LogicalType::String);
        assert!(annotation.is_optional());
        assert!(!annotation.is_repeated());
        assert!(annotation.logical_type.is_some());
    }
}
