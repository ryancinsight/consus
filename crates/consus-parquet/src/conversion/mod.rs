//! Parquet schema conversion module.
//!
//! ## Specification
//!
//! This module defines the canonical conversion contracts between:
//! - Parquet schema model (`SchemaDescriptor`, `FieldDescriptor`, `ParquetPhysicalType`)
//! - Arrow schema model (via `consus-arrow` integration)
//! - Core Consus datatypes (via `consus-core`)
//!
//! ## Invariants
//!
//! - Parquet physical types map to Arrow types with explicit width preservation.
//! - Logical type annotations refine the mapping without changing physical storage.
//! - Schema evolution steps preserve field identity through stable `FieldId`.
//! - Nested schemas are converted recursively.
//! - Zero-copy eligibility is computed from physical type and repetition.
//! - Every canonical `consus-core::Datatype` variant maps deterministically.
//!
//! ## Architecture
//!
//! ```text
//! conversion/
//! ├── arrow_parquet # Arrow ↔ Parquet schema conversions
//! ├── core_parquet # Core ↔ Parquet datatype conversions
//! └── validation   # Schema compatibility validation
//! ```

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use consus_core::{ByteOrder, Datatype};
#[cfg(feature = "alloc")]
use consus_core::{CompoundField, EnumMember};

use crate::schema::{
    FieldDescriptor, LogicalType, ParquetPhysicalType, Repetition, SchemaDescriptor,
};

/// Minimal representation of an Arrow field for compatibility analysis.
///
/// This struct avoids direct dependency on `consus-arrow` in the conversion
/// module while still enabling compatibility checks.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ArrowFieldRepr {
    /// Field name.
    pub name: String,
    /// Whether the field is nullable.
    pub nullable: bool,
    /// Core datatype representation.
    pub datatype: Datatype,
}

/// Parquet-to-Arrow conversion mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParquetConversionMode {
    /// Preserve exact Parquet semantics.
    Exact,
    /// Allow type widening for better Arrow compatibility.
    AllowWidening,
    /// Use best-effort mapping when exact conversion is impossible.
    BestEffort,
}

impl Default for ParquetConversionMode {
    fn default() -> Self {
        Self::Exact
    }
}

/// Result of Parquet-to-Arrow schema compatibility analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParquetCompatibility {
    /// Schemas are directly compatible.
    Compatible,
    /// Conversion requires schema evolution.
    RequiresEvolution,
    /// Schemas are incompatible.
    Incompatible,
}

/// Compute a canonical storage width for decimal logical annotations.
///
/// The returned width is conservative and deterministic:
/// - precision contributes the minimum storage width needed for the digit count
/// - negative scales do not reduce the storage width
/// - the result is always at least 1 byte
#[must_use]
fn decimal_storage_width(precision: u32, scale: i32) -> usize {
    let digits = precision as usize;
    let scale_width = scale.unsigned_abs() as usize;
    let candidate = digits.saturating_add(scale_width).saturating_add(1);
    candidate.max(1)
}

/// Convert a Parquet physical type to a Core datatype.
///
/// ## Invariants
///
/// - Fixed-width types map to equivalent Core primitive types.
/// - Variable-width types map to `Opaque` or `VariableString`.
/// - Width is preserved exactly for fixed-width types.
/// - Every physical variant has one canonical Core mapping.
#[must_use]
pub fn parquet_physical_to_core(physical_type: ParquetPhysicalType) -> Datatype {
    match physical_type {
        ParquetPhysicalType::Boolean => Datatype::Boolean,
        ParquetPhysicalType::Int32 => Datatype::Integer {
            bits: core::num::NonZeroUsize::new(32).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        },
        ParquetPhysicalType::Int64 => Datatype::Integer {
            bits: core::num::NonZeroUsize::new(64).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        },
        ParquetPhysicalType::Int96 => Datatype::Integer {
            bits: core::num::NonZeroUsize::new(96).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        },
        ParquetPhysicalType::Float => Datatype::Float {
            bits: core::num::NonZeroUsize::new(32).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
        },
        ParquetPhysicalType::Double => Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
        },
        ParquetPhysicalType::ByteArray => Datatype::VariableString {
            encoding: consus_core::StringEncoding::Utf8,
        },
        ParquetPhysicalType::FixedLenByteArray(width) => Datatype::Opaque {
            size: width,
            #[cfg(feature = "alloc")]
            tag: None,
        },
    }
}

/// Convert a Core datatype to a Parquet physical type hint.
///
/// ## Invariants
///
/// - Selects the most appropriate Parquet physical type.
/// - May lose information about signedness for non-standard widths.
/// - Variable-width Core types map to `ByteArray`.
/// - All canonical Core datatypes map deterministically.
#[must_use]
pub fn core_to_parquet_physical_hint(datatype: &Datatype) -> ParquetPhysicalType {
    match datatype {
        Datatype::Boolean => ParquetPhysicalType::Boolean,
        Datatype::Integer { bits, .. } => match bits.get() {
            1..=32 => ParquetPhysicalType::Int32,
            33..=64 => ParquetPhysicalType::Int64,
            _ => ParquetPhysicalType::ByteArray,
        },
        Datatype::Float { bits, .. } => match bits.get() {
            16..=32 => ParquetPhysicalType::Float,
            33..=64 => ParquetPhysicalType::Double,
            _ => ParquetPhysicalType::ByteArray,
        },
        Datatype::Complex { component_bits, .. } => match component_bits.get() {
            32 => ParquetPhysicalType::FixedLenByteArray(8),
            64 => ParquetPhysicalType::FixedLenByteArray(16),
            _ => ParquetPhysicalType::ByteArray,
        },
        Datatype::FixedString { length, .. } => {
            if *length == 0 {
                ParquetPhysicalType::ByteArray
            } else {
                ParquetPhysicalType::FixedLenByteArray(*length)
            }
        }
        Datatype::VariableString { .. } => ParquetPhysicalType::ByteArray,
        Datatype::Opaque { size, .. } => {
            if *size == 0 {
                ParquetPhysicalType::ByteArray
            } else {
                ParquetPhysicalType::FixedLenByteArray(*size)
            }
        }
        Datatype::Reference(_) => ParquetPhysicalType::FixedLenByteArray(8),
        #[cfg(feature = "alloc")]
        Datatype::Compound { size, fields } => {
            if *size > 0 && !fields.is_empty() {
                ParquetPhysicalType::FixedLenByteArray(*size)
            } else {
                ParquetPhysicalType::ByteArray
            }
        }
        #[cfg(feature = "alloc")]
        Datatype::Array { base, dims } => match base.element_size() {
            Some(width) if !dims.is_empty() && dims.iter().all(|d| *d > 0) => {
                match width.checked_mul(dims.iter().product::<usize>()) {
                    Some(total) if total > 0 => ParquetPhysicalType::FixedLenByteArray(total),
                    _ => ParquetPhysicalType::ByteArray,
                }
            }
            _ => ParquetPhysicalType::ByteArray,
        },
        #[cfg(feature = "alloc")]
        Datatype::Enum { base, members } => match base.element_size() {
            Some(width) if !members.is_empty() && width > 0 => {
                ParquetPhysicalType::FixedLenByteArray(width)
            }
            _ => ParquetPhysicalType::ByteArray,
        },
        #[cfg(feature = "alloc")]
        Datatype::VarLen { .. } => ParquetPhysicalType::ByteArray,
        #[cfg(not(feature = "alloc"))]
        _ => ParquetPhysicalType::ByteArray,
    }
}

/// Convert a Parquet logical type to a Core datatype annotation.
///
/// ## Invariants
///
/// - Logical types refine the physical type mapping.
/// - Temporal logical types map to integer representations.
/// - String logical types map to `VariableString` or fixed-width textual storage.
/// - Decimal and UUID annotations preserve their canonical widths.
#[must_use]
pub fn parquet_logical_to_core_annotation(logical_type: &LogicalType) -> Option<Datatype> {
    match logical_type {
        LogicalType::String => Some(Datatype::VariableString {
            encoding: consus_core::StringEncoding::Utf8,
        }),
        LogicalType::Enum => Some(Datatype::VariableString {
            encoding: consus_core::StringEncoding::Utf8,
        }),
        LogicalType::Decimal { precision, scale } => Some(Datatype::FixedString {
            length: decimal_storage_width(*precision, *scale),
            encoding: consus_core::StringEncoding::Ascii,
        }),
        LogicalType::Date => Some(Datatype::Integer {
            bits: core::num::NonZeroUsize::new(32).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }),
        LogicalType::Time { unit, .. } => {
            let width = match unit {
                crate::schema::TimeUnit::Milliseconds => 32,
                crate::schema::TimeUnit::Microseconds | crate::schema::TimeUnit::Nanoseconds => 64,
            };
            Some(Datatype::Integer {
                bits: core::num::NonZeroUsize::new(width).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            })
        }
        LogicalType::Timestamp { .. } => Some(Datatype::Integer {
            bits: core::num::NonZeroUsize::new(64).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        }),
        LogicalType::Integer { bit_width, signed } => Some(Datatype::Integer {
            bits: core::num::NonZeroUsize::new(*bit_width as usize).expect("non-zero"),
            byte_order: ByteOrder::LittleEndian,
            signed: *signed,
        }),
        LogicalType::UnsignedInteger { bit_width } => Some(Datatype::Integer {
            bits: core::num::NonZeroUsize::new(*bit_width as usize).expect("non-zero"),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        }),
        LogicalType::Json => Some(Datatype::VariableString {
            encoding: consus_core::StringEncoding::Utf8,
        }),
        LogicalType::Bson => Some(Datatype::Opaque {
            size: 0,
            #[cfg(feature = "alloc")]
            tag: Some(String::from("bson")),
        }),
        LogicalType::Uuid => Some(Datatype::FixedString {
            length: 16,
            encoding: consus_core::StringEncoding::Ascii,
        }),
        #[cfg(feature = "alloc")]
        LogicalType::Extension { .. } => None,
        #[cfg(not(feature = "alloc"))]
        _ => None,
    }
}

/// Infer a logical type from a Core datatype.
///
/// ## Invariants
///
/// - Returns `None` for types without a clear Parquet logical mapping.
/// - Temporal Core types map to appropriate logical timestamps.
/// - String types map to `STRING` logical annotation.
/// - Fixed-width numerics preserve signedness and width.
#[must_use]
pub fn core_to_parquet_logical_hint(datatype: &Datatype) -> Option<LogicalType> {
    match datatype {
        Datatype::VariableString { encoding } => match encoding {
            consus_core::StringEncoding::Utf8 => Some(LogicalType::String),
            _ => None,
        },
        Datatype::FixedString { length, encoding } => {
            if *length == 16 && *encoding == consus_core::StringEncoding::Ascii {
                Some(LogicalType::Uuid)
            } else if *length > 0 {
                Some(LogicalType::Enum)
            } else {
                None
            }
        }
        Datatype::Opaque { size, .. } => {
            if *size > 0 {
                Some(LogicalType::Json)
            } else {
                None
            }
        }
        Datatype::Integer { bits, signed, .. } => {
            let bit_width = bits.get() as u8;
            if bit_width == 32 && *signed {
                Some(LogicalType::Date)
            } else if bit_width == 32 || bit_width == 64 {
                Some(if *signed {
                    LogicalType::Integer {
                        bit_width,
                        signed: true,
                    }
                } else {
                    LogicalType::UnsignedInteger { bit_width }
                })
            } else {
                None
            }
        }
        Datatype::Float { .. } => None,
        Datatype::Complex { component_bits, .. } => match component_bits.get() {
            32 => Some(LogicalType::Json),
            64 => Some(LogicalType::Bson),
            _ => None,
        },
        Datatype::Reference(_) => None,
        _ => None,
    }
}

/// Analyze compatibility between Parquet and Arrow field descriptors.
///
/// ## Note
///
/// This function requires the `consus-arrow` crate to be available.
/// When `consus-arrow` is not in scope, use the Core datatype comparison
/// functions instead.
#[cfg(feature = "alloc")]
#[must_use]
pub fn analyze_parquet_arrow_compatibility(
    parquet_field: &FieldDescriptor,
    arrow_field: &crate::ArrowFieldRepr,
) -> ParquetCompatibility {
    if parquet_field.name() != arrow_field.name {
        return ParquetCompatibility::Incompatible;
    }

    let parquet_nullable = parquet_field.is_optional();
    let arrow_nullable = arrow_field.nullable;
    if parquet_nullable && !arrow_nullable {
        return ParquetCompatibility::RequiresEvolution;
    }

    let parquet_core = parquet_field_to_core(parquet_field);
    let arrow_core = arrow_field.datatype.clone();

    if parquet_core == arrow_core {
        ParquetCompatibility::Compatible
    } else if types_are_compatible(&parquet_core, &arrow_core) {
        ParquetCompatibility::RequiresEvolution
    } else {
        ParquetCompatibility::Incompatible
    }
}

/// Check if two Core datatypes are compatible for conversion.
#[must_use]
fn types_are_compatible(left: &Datatype, right: &Datatype) -> bool {
    match (left, right) {
        (Datatype::Boolean, Datatype::Boolean) => true,
        (
            Datatype::Integer {
                bits: b1,
                signed: s1,
                ..
            },
            Datatype::Integer {
                bits: b2,
                signed: s2,
                ..
            },
        ) => s1 == s2 && b1.get() <= b2.get(),
        (Datatype::Float { bits: b1, .. }, Datatype::Float { bits: b2, .. }) => {
            b1.get() <= b2.get()
        }
        (
            Datatype::Complex {
                component_bits: b1, ..
            },
            Datatype::Complex {
                component_bits: b2, ..
            },
        ) => b1.get() <= b2.get(),
        (Datatype::FixedString { length: l1, .. }, Datatype::FixedString { length: l2, .. }) => {
            l1 <= l2
        }
        (Datatype::VariableString { .. }, Datatype::VariableString { .. }) => true,
        (Datatype::Opaque { size: s1, .. }, Datatype::Opaque { size: s2, .. }) => s1 <= s2,
        #[cfg(feature = "alloc")]
        (
            Datatype::Compound {
                fields: f1,
                size: s1,
            },
            Datatype::Compound {
                fields: f2,
                size: s2,
            },
        ) => s1 <= s2 && f1.len() <= f2.len(),
        #[cfg(feature = "alloc")]
        (Datatype::Array { base: b1, dims: d1 }, Datatype::Array { base: b2, dims: d2 }) => {
            d1.len() <= d2.len() && types_are_compatible(b1, b2)
        }
        #[cfg(feature = "alloc")]
        (
            Datatype::Enum {
                base: b1,
                members: m1,
            },
            Datatype::Enum {
                base: b2,
                members: m2,
            },
        ) => m1.len() <= m2.len() && types_are_compatible(b1, b2),
        #[cfg(feature = "alloc")]
        (Datatype::VarLen { base: b1 }, Datatype::VarLen { base: b2 }) => {
            types_are_compatible(b1, b2)
        }
        (Datatype::Reference(_), Datatype::Reference(_)) => true,
        _ => false,
    }
}

/// Convert a Parquet field descriptor to a Core datatype.
///
/// ## Invariants
///
/// - Logical type takes precedence over physical type when present.
/// - Returns the physical type mapping when no logical type is present.
#[must_use]
pub fn parquet_field_to_core(field: &FieldDescriptor) -> Datatype {
    if let Some(logical_type) = field.logical_type() {
        parquet_logical_to_core_annotation(logical_type)
            .unwrap_or_else(|| parquet_physical_to_core(field.physical_type()))
    } else {
        parquet_physical_to_core(field.physical_type())
    }
}

/// Convert a Parquet schema to a vector of Core datatype pairs.
///
/// Returns field names paired with their Core datatype equivalents.
#[cfg(feature = "alloc")]
#[must_use]
pub fn parquet_schema_to_core_pairs(schema: &SchemaDescriptor) -> Vec<(String, Datatype)> {
    schema
        .fields()
        .iter()
        .map(|field| (field.name().to_owned(), parquet_field_to_core(field)))
        .collect()
}

/// Convert repetition semantics between Parquet and Arrow.
#[must_use]
pub fn parquet_repetition_to_arrow_nullability(repetition: Repetition) -> bool {
    matches!(repetition, Repetition::Optional)
}

/// Convert Arrow nullability to Parquet repetition.
#[must_use]
pub fn arrow_nullability_to_parquet_repetition(nullable: bool) -> Repetition {
    if nullable {
        Repetition::Optional
    } else {
        Repetition::Required
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_type_mapping_is_correct() {
        assert!(matches!(
            parquet_physical_to_core(ParquetPhysicalType::Boolean),
            Datatype::Boolean
        ));
        assert!(matches!(
            parquet_physical_to_core(ParquetPhysicalType::Int32),
            Datatype::Integer { bits, signed: true, .. } if bits.get() == 32
        ));
        assert!(matches!(
            parquet_physical_to_core(ParquetPhysicalType::Double),
            Datatype::Float { bits, .. } if bits.get() == 64
        ));
        assert!(matches!(
            parquet_physical_to_core(ParquetPhysicalType::ByteArray),
            Datatype::VariableString { .. }
        ));
        assert!(matches!(
            parquet_physical_to_core(ParquetPhysicalType::FixedLenByteArray(12)),
            Datatype::Opaque { size: 12, .. }
        ));
    }

    #[test]
    fn logical_type_refines_mapping() {
        let logical = LogicalType::String;
        let result = parquet_logical_to_core_annotation(&logical);
        assert!(matches!(result, Some(Datatype::VariableString { .. })));

        let logical = LogicalType::Date;
        let result = parquet_logical_to_core_annotation(&logical);
        assert!(
            matches!(result, Some(Datatype::Integer { bits, signed: true, .. }) if bits.get() == 32)
        );

        let logical = LogicalType::Uuid;
        let result = parquet_logical_to_core_annotation(&logical);
        assert!(matches!(
            result,
            Some(Datatype::FixedString { length: 16, .. })
        ));

        let logical = LogicalType::Decimal {
            precision: 12,
            scale: 3,
        };
        let result = parquet_logical_to_core_annotation(&logical);
        assert!(matches!(result, Some(Datatype::FixedString { length, .. }) if length >= 15));
    }

    #[test]
    fn core_to_parquet_roundtrip() {
        let core_type = Datatype::Integer {
            bits: core::num::NonZeroUsize::new(32).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };

        let parquet_type = core_to_parquet_physical_hint(&core_type);
        assert_eq!(parquet_type, ParquetPhysicalType::Int32);

        let core_type = Datatype::Complex {
            component_bits: core::num::NonZeroUsize::new(64).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
        };
        assert_eq!(
            core_to_parquet_physical_hint(&core_type),
            ParquetPhysicalType::FixedLenByteArray(16)
        );

        let core_type = Datatype::Reference(consus_core::ReferenceType::Object);
        assert_eq!(
            core_to_parquet_physical_hint(&core_type),
            ParquetPhysicalType::FixedLenByteArray(8)
        );

        let core_type = Datatype::Reference(consus_core::ReferenceType::Region);
        assert_eq!(
            core_to_parquet_physical_hint(&core_type),
            ParquetPhysicalType::FixedLenByteArray(8)
        );
    }

    #[test]
    fn repetition_conversion() {
        assert!(parquet_repetition_to_arrow_nullability(
            Repetition::Optional
        ));
        assert!(!parquet_repetition_to_arrow_nullability(
            Repetition::Required
        ));

        assert_eq!(
            arrow_nullability_to_parquet_repetition(true),
            Repetition::Optional
        );
        assert_eq!(
            arrow_nullability_to_parquet_repetition(false),
            Repetition::Required
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn full_datatype_coverage_is_deterministic() {
        let compound = Datatype::Compound {
            fields: alloc::vec![CompoundField {
                name: String::from("x"),
                datatype: Datatype::Integer {
                    bits: core::num::NonZeroUsize::new(32).expect("constant"),
                    byte_order: ByteOrder::LittleEndian,
                    signed: true,
                },
                offset: 0,
            }],
            size: 4,
        };
        assert_eq!(
            core_to_parquet_physical_hint(&compound),
            ParquetPhysicalType::FixedLenByteArray(4)
        );

        let array = Datatype::Array {
            base: alloc::boxed::Box::new(Datatype::Integer {
                bits: core::num::NonZeroUsize::new(16).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
                signed: false,
            }),
            dims: alloc::vec![2, 2],
        };
        assert_eq!(
            core_to_parquet_physical_hint(&array),
            ParquetPhysicalType::FixedLenByteArray(8)
        );

        let enum_type = Datatype::Enum {
            base: alloc::boxed::Box::new(Datatype::Integer {
                bits: core::num::NonZeroUsize::new(8).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
                signed: false,
            }),
            members: alloc::vec![EnumMember {
                name: String::from("A"),
                value: 1,
            }],
        };
        assert_eq!(
            core_to_parquet_physical_hint(&enum_type),
            ParquetPhysicalType::FixedLenByteArray(1)
        );

        let varlen = Datatype::VarLen {
            base: alloc::boxed::Box::new(Datatype::Integer {
                bits: core::num::NonZeroUsize::new(32).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            }),
        };
        assert_eq!(
            core_to_parquet_physical_hint(&varlen),
            ParquetPhysicalType::ByteArray
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn field_to_core_conversion() {
        let field = FieldDescriptor::required(
            crate::schema::FieldId::new(1),
            "temperature",
            ParquetPhysicalType::Double,
        );
        let core_type = parquet_field_to_core(&field);
        assert!(matches!(core_type, Datatype::Float { bits, .. } if bits.get() == 64));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn schema_to_core_pairs() {
        let schema = SchemaDescriptor::new(alloc::vec![
            FieldDescriptor::required(
                crate::schema::FieldId::new(1),
                "x",
                ParquetPhysicalType::Int32
            ),
            FieldDescriptor::required(
                crate::schema::FieldId::new(2),
                "y",
                ParquetPhysicalType::Double
            ),
        ]);

        let pairs = parquet_schema_to_core_pairs(&schema);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "x");
        assert_eq!(pairs[1].0, "y");
    }
}
