//! Schema conversion module for Arrow ↔ Core ↔ Parquet transformations.
//!
//! ## Specification
//!
//! This module defines the canonical conversion contracts between:
//! - Arrow schema model (`ArrowSchema`, `ArrowField`, `ArrowDataType`)
//! - Core Consus datatypes (`Datatype`, `ByteOrder`)
//! - Parquet schema model (via `consus-parquet` integration)
//!
//! ## Invariants
//!
//! - Conversions preserve field identity and semantic meaning.
//! - Lossy conversions are explicit and require `AllowLossy` mode.
//! - Nested structures are converted recursively.
//! - Zero-copy eligibility is computed during conversion.
//! - All conversions are deterministic and reproducible.
//!
//! ## Architecture
//!
//! ```text
//! conversion/
//! ├── core_arrow     # Arrow ↔ Core Datatype conversions
//! ├── parquet_arrow  # Arrow ↔ Parquet schema conversions
//! └── traits         # Generic conversion traits
//! ```

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, string::String, vec::Vec};

use consus_core::{ByteOrder, Datatype};

use crate::datatype::{
    ArrowDataType, DecimalType, FixedSizeBinaryType, IntSign, TimeUnit, TimestampType,
};
use crate::field::{
    ArrowField, ArrowFieldId, ArrowFieldKind, ArrowFieldSemantics, ArrowNullability,
};
use crate::schema::ArrowSchema;

/// Conversion mode controlling how strictly types must match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversionMode {
    /// Reject any conversion that may lose information.
    Strict,
    /// Allow widening conversions (e.g., Int32 → Int64).
    AllowWidening,
    /// Allow lossy conversions with explicit acknowledgment.
    AllowLossy,
    /// Convert to the closest representable type.
    BestEffort,
}

impl Default for ConversionMode {
    fn default() -> Self {
        Self::Strict
    }
}

/// Result of a datatype conversion analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionCompatibility {
    /// Conversion is exact and reversible.
    Exact,
    /// Conversion is lossless but may change representation.
    Lossless,
    /// Conversion may lose precision or information.
    Lossy,
    /// Conversion is not possible.
    Incompatible,
}

impl ConversionCompatibility {
    /// Returns `true` if the conversion can proceed in the given mode.
    #[must_use]
    pub const fn is_permitted(self, mode: ConversionMode) -> bool {
        match (self, mode) {
            (Self::Exact | Self::Lossless, _) => true,
            (Self::Lossy, ConversionMode::AllowLossy | ConversionMode::BestEffort) => true,
            (Self::Incompatible, ConversionMode::BestEffort) => false,
            _ => false,
        }
    }
}

/// Convert an Arrow datatype to a Core datatype.
///
/// ## Invariants
///
/// - Fixed-width types map to equivalent Core primitive types.
/// - Variable-width types map to `VariableString` or `Opaque`.
/// - Nested types are converted recursively when `alloc` is enabled.
/// - Temporal types map to integer representations with appropriate widths.
#[must_use]
pub fn arrow_datatype_to_core(datatype: &ArrowDataType) -> Datatype {
    match datatype {
        ArrowDataType::Boolean => Datatype::Boolean,

        ArrowDataType::Int { bit_width, sign } => {
            let bits = core::num::NonZeroUsize::new((*bit_width).into())
                .expect("Arrow integer bit width must be non-zero");
            Datatype::Integer {
                bits,
                byte_order: ByteOrder::LittleEndian,
                signed: matches!(sign, IntSign::Signed),
            }
        }

        ArrowDataType::Float { bit_width } => {
            let bits = core::num::NonZeroUsize::new((*bit_width).into())
                .expect("Arrow float bit width must be non-zero");
            Datatype::Float {
                bits,
                byte_order: ByteOrder::LittleEndian,
            }
        }

        ArrowDataType::Utf8 => Datatype::VariableString {
            encoding: consus_core::StringEncoding::Utf8,
        },

        ArrowDataType::Binary => Datatype::Opaque {
            size: 0,
            #[cfg(feature = "alloc")]
            tag: None,
        },

        ArrowDataType::FixedSizeBinary(FixedSizeBinaryType { size }) => Datatype::Opaque {
            size: *size,
            #[cfg(feature = "alloc")]
            tag: None,
        },

        ArrowDataType::Decimal(DecimalType { precision, scale }) => {
            // Map decimal to fixed-string representation
            let length = precision
                .saturating_add(scale.unsigned_abs())
                .saturating_add(2); // Sign and decimal point
            Datatype::FixedString {
                length: length as usize,
                encoding: consus_core::StringEncoding::Ascii,
            }
        }

        ArrowDataType::Date32 => Datatype::Integer {
            bits: core::num::NonZeroUsize::new(32).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        },

        ArrowDataType::Date64 => Datatype::Integer {
            bits: core::num::NonZeroUsize::new(64).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        },

        ArrowDataType::Timestamp(TimestampType { unit, .. }) => {
            let _ = unit; // Timestamp always maps to Int64
            Datatype::Integer {
                bits: core::num::NonZeroUsize::new(64).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
                signed: true,
            }
        }

        ArrowDataType::Duration(_) => Datatype::Integer {
            bits: core::num::NonZeroUsize::new(64).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        },

        ArrowDataType::Interval => Datatype::Compound {
            #[cfg(feature = "alloc")]
            fields: Vec::new(),
            #[cfg(feature = "alloc")]
            size: 16,
        },

        #[cfg(feature = "alloc")]
        ArrowDataType::List(list_type) => Datatype::VarLen {
            base: Box::new(arrow_datatype_to_core(&list_type.element_type)),
        },

        #[cfg(feature = "alloc")]
        ArrowDataType::Map(map_type) => {
            let key_dt = arrow_datatype_to_core(&map_type.key_type);
            let val_dt = arrow_datatype_to_core(&map_type.value_type);
            let key_size = key_dt.element_size().unwrap_or(0);
            let val_size = val_dt.element_size().unwrap_or(0);
            Datatype::Compound {
                fields: vec![
                    consus_core::CompoundField {
                        name: String::from("key"),
                        datatype: key_dt,
                        offset: 0,
                    },
                    consus_core::CompoundField {
                        name: String::from("value"),
                        datatype: val_dt,
                        offset: key_size,
                    },
                ],
                size: key_size + val_size,
            }
        }

        #[cfg(feature = "alloc")]
        ArrowDataType::Struct(struct_type) => {
            let mut offset = 0usize;
            let fields: Vec<consus_core::CompoundField> = struct_type
                .fields
                .iter()
                .map(|f| {
                    let dt = arrow_datatype_to_core(&core_datatype_to_arrow_hint(&f.datatype));
                    let field_size = dt.element_size().unwrap_or(0);
                    let field_offset = offset;
                    offset += field_size;
                    consus_core::CompoundField {
                        name: f.name.clone(),
                        datatype: dt,
                        offset: field_offset,
                    }
                })
                .collect();
            let size = offset;
            Datatype::Compound { fields, size }
        }

        #[cfg(feature = "alloc")]
        ArrowDataType::Union(union_type) => {
            let fields: Vec<consus_core::CompoundField> = union_type
                .fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let dt = arrow_datatype_to_core(&core_datatype_to_arrow_hint(&f.datatype));
                    consus_core::CompoundField {
                        name: f.name.clone(),
                        datatype: dt,
                        offset: i,
                    }
                })
                .collect();
            Datatype::Compound { fields, size: 0 }
        }

        #[cfg(feature = "alloc")]
        ArrowDataType::Dictionary(dict_type) => arrow_datatype_to_core(&dict_type.value_type),

        ArrowDataType::Null => Datatype::Opaque {
            size: 0,
            #[cfg(feature = "alloc")]
            tag: None,
        },

        #[cfg(feature = "alloc")]
        ArrowDataType::Extension { .. } => Datatype::Opaque { size: 0, tag: None },

        #[cfg(not(feature = "alloc"))]
        _ => Datatype::Opaque { size: 0 },
    }
}

/// Convert a Core datatype to an Arrow datatype hint.
///
/// ## Invariants
///
/// - The conversion selects the most specific Arrow type that can represent the Core type.
/// - Information about byte order may be lost (Arrow assumes little-endian).
/// - Compound types map to Struct when `alloc` is enabled.
#[must_use]
pub fn core_datatype_to_arrow_hint(datatype: &Datatype) -> ArrowDataType {
    match datatype {
        Datatype::Boolean => ArrowDataType::Boolean,

        Datatype::Integer { bits, signed, .. } => {
            let bit_width = bits.get() as u8;
            let sign = if *signed {
                IntSign::Signed
            } else {
                IntSign::Unsigned
            };
            ArrowDataType::Int { bit_width, sign }
        }

        Datatype::Float { bits, .. } => {
            let bit_width = bits.get() as u8;
            ArrowDataType::Float { bit_width }
        }

        Datatype::FixedString { length, encoding } => {
            let _ = encoding;
            ArrowDataType::FixedSizeBinary(FixedSizeBinaryType { size: *length })
        }

        Datatype::VariableString { .. } => ArrowDataType::Utf8,

        Datatype::Opaque { size, .. } => {
            if *size == 0 {
                ArrowDataType::Binary
            } else {
                ArrowDataType::FixedSizeBinary(FixedSizeBinaryType { size: *size })
            }
        }

        Datatype::Reference(_) => ArrowDataType::Binary,

        #[cfg(feature = "alloc")]
        Datatype::Compound { fields, .. } => {
            use consus_core::CompoundField;
            let arrow_fields: Vec<ArrowField> = fields
                .iter()
                .enumerate()
                .map(
                    |(
                        i,
                        CompoundField {
                            name,
                            datatype,
                            offset,
                        },
                    )| {
                        let kind = crate::field::kind_from_datatype(datatype);
                        ArrowField {
                            id: ArrowFieldId::new(*offset as u32),
                            name: name.clone(),
                            kind,
                            semantics: ArrowFieldSemantics::required_scalar(),
                            datatype: datatype.clone(),
                            children: Vec::new(),
                        }
                    },
                )
                .collect();
            ArrowDataType::Struct(crate::datatype::StructType {
                fields: arrow_fields,
            })
        }

        #[cfg(feature = "alloc")]
        Datatype::Array { base, .. } => ArrowDataType::List(crate::datatype::ListType {
            element_type: Box::new(core_datatype_to_arrow_hint(base)),
            nullable: true,
        }),

        #[cfg(feature = "alloc")]
        Datatype::Enum { .. } => ArrowDataType::Dictionary(crate::datatype::DictionaryType::new(
            ArrowDataType::Int {
                bit_width: 32,
                sign: IntSign::Signed,
            },
            ArrowDataType::Utf8,
            false,
        )),

        #[cfg(feature = "alloc")]
        Datatype::VarLen { base } => ArrowDataType::List(crate::datatype::ListType {
            element_type: Box::new(core_datatype_to_arrow_hint(base)),
            nullable: true,
        }),

        Datatype::Complex { component_bits, .. } => {
            let bit_width = component_bits.get() as u8;
            let real_field = ArrowField {
                id: ArrowFieldId::new(0),
                name: String::from("real"),
                kind: ArrowFieldKind::Float,
                semantics: ArrowFieldSemantics::required_scalar(),
                datatype: Datatype::Float {
                    bits: *component_bits,
                    byte_order: ByteOrder::LittleEndian,
                },
                children: Vec::new(),
            };
            let imag_field = ArrowField {
                id: ArrowFieldId::new(1),
                name: String::from("imaginary"),
                kind: ArrowFieldKind::Float,
                semantics: ArrowFieldSemantics::required_scalar(),
                datatype: Datatype::Float {
                    bits: *component_bits,
                    byte_order: ByteOrder::LittleEndian,
                },
                children: Vec::new(),
            };
            ArrowDataType::Struct(crate::datatype::StructType {
                fields: vec![real_field, imag_field],
            })
        }

        #[cfg(not(feature = "alloc"))]
        _ => ArrowDataType::Binary,
    }
}

/// Analyze compatibility between an Arrow type and a Core datatype.
#[must_use]
pub fn analyze_conversion_compatibility(
    arrow_type: &ArrowDataType,
    core_type: &Datatype,
) -> ConversionCompatibility {
    match (arrow_type, core_type) {
        // Boolean → Boolean is exact
        (ArrowDataType::Boolean, Datatype::Boolean) => ConversionCompatibility::Exact,

        // Integer width/sign matching
        (ArrowDataType::Int { bit_width, sign }, Datatype::Integer { bits, signed, .. }) => {
            let core_bits = bits.get() as u8;
            // Sign mismatch is always incompatible
            if matches!(sign, IntSign::Signed) != *signed {
                return ConversionCompatibility::Incompatible;
            }
            match (*bit_width, core_bits, sign, *signed) {
                (aw, cw, _, _) if aw == cw => ConversionCompatibility::Exact,
                (aw, cw, IntSign::Signed, true) if aw < cw => ConversionCompatibility::Lossless,
                (aw, cw, IntSign::Unsigned, false) if aw < cw => ConversionCompatibility::Lossless,
                _ => ConversionCompatibility::Incompatible,
            }
        }

        // Float width matching
        (ArrowDataType::Float { bit_width }, Datatype::Float { bits, .. }) => {
            let core_bits = bits.get() as u8;
            match bit_width.cmp(&core_bits) {
                core::cmp::Ordering::Equal => ConversionCompatibility::Exact,
                core::cmp::Ordering::Greater => ConversionCompatibility::Lossless,
                core::cmp::Ordering::Less => ConversionCompatibility::Lossy,
            }
        }

        // String type mappings
        (ArrowDataType::Utf8, Datatype::VariableString { .. }) => ConversionCompatibility::Exact,
        (ArrowDataType::Utf8, Datatype::FixedString { .. }) => ConversionCompatibility::Lossy,

        // Binary mappings
        (ArrowDataType::Binary, Datatype::Opaque { .. }) => ConversionCompatibility::Lossless,
        (
            ArrowDataType::FixedSizeBinary(FixedSizeBinaryType { size }),
            Datatype::Opaque {
                size: core_size, ..
            },
        ) => {
            if *size == *core_size {
                ConversionCompatibility::Exact
            } else {
                ConversionCompatibility::Incompatible
            }
        }

        // Temporal mappings
        (
            ArrowDataType::Date32,
            Datatype::Integer {
                bits, signed: true, ..
            },
        ) if bits.get() == 32 => ConversionCompatibility::Exact,
        (
            ArrowDataType::Date64,
            Datatype::Integer {
                bits, signed: true, ..
            },
        ) if bits.get() == 64 => ConversionCompatibility::Exact,
        (
            ArrowDataType::Timestamp(_),
            Datatype::Integer {
                bits, signed: true, ..
            },
        ) if bits.get() == 64 => ConversionCompatibility::Lossless,

        // Default: assume incompatible
        _ => ConversionCompatibility::Incompatible,
    }
}

/// Builder for constructing Arrow fields from Core datatypes.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ArrowFieldFromCoreBuilder {
    name: String,
    datatype: Datatype,
    nullable: bool,
    id: ArrowFieldId,
}

#[cfg(feature = "alloc")]
impl ArrowFieldFromCoreBuilder {
    /// Create a new builder for a field with the given name and Core datatype.
    #[must_use]
    pub fn new(name: String, datatype: Datatype) -> Self {
        Self {
            name,
            datatype,
            nullable: false,
            id: ArrowFieldId::new(0),
        }
    }

    /// Set the field nullability.
    #[must_use]
    pub fn nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Set the field identifier.
    #[must_use]
    pub fn id(mut self, id: ArrowFieldId) -> Self {
        self.id = id;
        self
    }

    /// Build the Arrow field.
    #[must_use]
    pub fn build(self) -> ArrowField {
        let arrow_type = core_datatype_to_arrow_hint(&self.datatype);
        let kind = crate::field::kind_from_datatype(&self.datatype);

        ArrowField {
            id: self.id,
            name: self.name,
            kind,
            semantics: if self.nullable {
                ArrowFieldSemantics::optional_scalar()
            } else {
                ArrowFieldSemantics::required_scalar()
            },
            datatype: self.datatype,
            children: Vec::new(),
        }
    }
}

/// Convert an Arrow schema to a vector of Core datatype pairs.
///
/// Returns field names paired with their Core datatype equivalents.
#[cfg(feature = "alloc")]
#[must_use]
pub fn arrow_schema_to_core_pairs(schema: &ArrowSchema) -> Vec<(String, Datatype)> {
    schema
        .fields()
        .iter()
        .map(|field| {
            let core_type = arrow_datatype_to_core(&core_datatype_to_arrow_hint(&field.datatype));
            (field.name.clone(), core_type)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_conversion_is_exact() {
        assert_eq!(
            analyze_conversion_compatibility(&ArrowDataType::Boolean, &Datatype::Boolean),
            ConversionCompatibility::Exact
        );
    }

    #[test]
    fn integer_width_matching() {
        assert_eq!(
            analyze_conversion_compatibility(
                &ArrowDataType::Int {
                    bit_width: 32,
                    sign: IntSign::Signed
                },
                &Datatype::Integer {
                    bits: core::num::NonZeroUsize::new(32).unwrap(),
                    byte_order: ByteOrder::LittleEndian,
                    signed: true,
                }
            ),
            ConversionCompatibility::Exact
        );

        assert_eq!(
            analyze_conversion_compatibility(
                &ArrowDataType::Int {
                    bit_width: 32,
                    sign: IntSign::Signed
                },
                &Datatype::Integer {
                    bits: core::num::NonZeroUsize::new(64).unwrap(),
                    byte_order: ByteOrder::LittleEndian,
                    signed: true,
                }
            ),
            ConversionCompatibility::Lossless
        );
    }

    #[test]
    fn float_widening_is_lossless() {
        assert_eq!(
            analyze_conversion_compatibility(
                &ArrowDataType::Float { bit_width: 64 },
                &Datatype::Float {
                    bits: core::num::NonZeroUsize::new(32).unwrap(),
                    byte_order: ByteOrder::LittleEndian,
                }
            ),
            ConversionCompatibility::Lossless
        );
    }

    #[test]
    fn core_to_arrow_roundtrip_preserves_structure() {
        let core_type = Datatype::Integer {
            bits: core::num::NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        };

        let arrow_type = core_datatype_to_arrow_hint(&core_type);
        let core_again = arrow_datatype_to_core(&arrow_type);

        // Integer types should round-trip
        match (&arrow_type, &core_again) {
            (ArrowDataType::Int { bit_width, sign }, Datatype::Integer { bits, signed, .. }) => {
                assert_eq!(*bit_width, bits.get() as u8);
                assert_eq!(*sign == IntSign::Signed, *signed);
            }
            _ => panic!("roundtrip failed"),
        }
    }

    #[test]
    fn conversion_mode_permits_correctly() {
        use ConversionCompatibility::*;

        assert!(Exact.is_permitted(ConversionMode::Strict));
        assert!(Lossless.is_permitted(ConversionMode::Strict));
        assert!(!Lossy.is_permitted(ConversionMode::Strict));
        assert!(Lossy.is_permitted(ConversionMode::AllowLossy));
        assert!(!Incompatible.is_permitted(ConversionMode::AllowLossy));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn field_builder_constructs_valid_field() {
        let field = ArrowFieldFromCoreBuilder::new(
            String::from("temperature"),
            Datatype::Float {
                bits: core::num::NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::LittleEndian,
            },
        )
        .nullable(true)
        .id(ArrowFieldId::new(1))
        .build();
        assert_eq!(field.name, "temperature");
        assert!(field.is_nullable());
        assert_eq!(field.id.get(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compound_to_struct_preserves_fields() {
        use consus_core::CompoundField;
        let fields = vec![
            CompoundField {
                name: String::from("x"),
                datatype: Datatype::Integer {
                    bits: core::num::NonZeroUsize::new(32).unwrap(),
                    byte_order: ByteOrder::LittleEndian,
                    signed: true,
                },
                offset: 0,
            },
            CompoundField {
                name: String::from("y"),
                datatype: Datatype::Float {
                    bits: core::num::NonZeroUsize::new(64).unwrap(),
                    byte_order: ByteOrder::LittleEndian,
                },
                offset: 4,
            },
            CompoundField {
                name: String::from("label"),
                datatype: Datatype::VariableString {
                    encoding: consus_core::StringEncoding::Utf8,
                },
                offset: 12,
            },
        ];
        let core_type = Datatype::Compound { fields, size: 24 };
        let arrow_type = core_datatype_to_arrow_hint(&core_type);
        match arrow_type {
            ArrowDataType::Struct(struct_type) => {
                assert_eq!(struct_type.fields.len(), 3);
                assert_eq!(struct_type.fields[0].name, "x");
                assert_eq!(struct_type.fields[1].name, "y");
                assert_eq!(struct_type.fields[2].name, "label");
            }
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn array_to_list_preserves_element_type() {
        let core_type = Datatype::Array {
            base: Box::new(Datatype::Float {
                bits: core::num::NonZeroUsize::new(64).unwrap(),
                byte_order: ByteOrder::LittleEndian,
            }),
            dims: vec![3],
        };
        let arrow_type = core_datatype_to_arrow_hint(&core_type);
        match arrow_type {
            ArrowDataType::List(list_type) => {
                assert_eq!(
                    *list_type.element_type,
                    ArrowDataType::Float { bit_width: 64 }
                );
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn complex_to_struct_has_real_and_imaginary() {
        let core_type = Datatype::Complex {
            component_bits: core::num::NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let arrow_type = core_datatype_to_arrow_hint(&core_type);
        match arrow_type {
            ArrowDataType::Struct(struct_type) => {
                assert_eq!(struct_type.fields.len(), 2);
                assert_eq!(struct_type.fields[0].name, "real");
                assert_eq!(struct_type.fields[1].name, "imaginary");
                assert_eq!(
                    struct_type.fields[0].datatype,
                    Datatype::Float {
                        bits: core::num::NonZeroUsize::new(64).unwrap(),
                        byte_order: ByteOrder::LittleEndian,
                    }
                );
                assert_eq!(
                    struct_type.fields[1].datatype,
                    Datatype::Float {
                        bits: core::num::NonZeroUsize::new(64).unwrap(),
                        byte_order: ByteOrder::LittleEndian,
                    }
                );
            }
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn struct_to_compound_preserves_fields() {
        let struct_type = crate::datatype::StructType {
            fields: vec![
                ArrowField::new(
                    "id",
                    Datatype::Integer {
                        bits: core::num::NonZeroUsize::new(32).unwrap(),
                        byte_order: ByteOrder::LittleEndian,
                        signed: true,
                    },
                    false,
                ),
                ArrowField::new(
                    "value",
                    Datatype::Float {
                        bits: core::num::NonZeroUsize::new(64).unwrap(),
                        byte_order: ByteOrder::LittleEndian,
                    },
                    false,
                ),
            ],
        };
        let core_type = arrow_datatype_to_core(&ArrowDataType::Struct(struct_type));
        match core_type {
            Datatype::Compound { fields, size } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "id");
                assert_eq!(fields[1].name, "value");
                assert!(size > 0);
            }
            other => panic!("expected Compound, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn map_to_compound_preserves_key_value() {
        let map_type = crate::datatype::MapType::new(
            ArrowDataType::Utf8,
            ArrowDataType::Float { bit_width: 64 },
            true,
        );
        let core_type = arrow_datatype_to_core(&ArrowDataType::Map(map_type));
        match core_type {
            Datatype::Compound { fields, .. } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "key");
                assert_eq!(fields[1].name, "value");
                match &fields[1].datatype {
                    Datatype::Float { bits, .. } => {
                        assert_eq!(bits.get(), 64);
                    }
                    other => panic!("expected Float for value, got {other:?}"),
                }
            }
            other => panic!("expected Compound, got {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn roundtrip_compound_struct() {
        use consus_core::CompoundField;
        let original = Datatype::Compound {
            fields: vec![
                CompoundField {
                    name: String::from("a"),
                    datatype: Datatype::Integer {
                        bits: core::num::NonZeroUsize::new(32).unwrap(),
                        byte_order: ByteOrder::LittleEndian,
                        signed: true,
                    },
                    offset: 0,
                },
                CompoundField {
                    name: String::from("b"),
                    datatype: Datatype::Float {
                        bits: core::num::NonZeroUsize::new(64).unwrap(),
                        byte_order: ByteOrder::LittleEndian,
                    },
                    offset: 4,
                },
            ],
            size: 12,
        };
        let arrow_type = core_datatype_to_arrow_hint(&original);
        let roundtripped = arrow_datatype_to_core(&arrow_type);
        match roundtripped {
            Datatype::Compound { fields, size } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "a");
                assert_eq!(fields[1].name, "b");
                assert!(size > 0);
            }
            other => panic!("expected Compound, got {other:?}"),
        }
    }
}
