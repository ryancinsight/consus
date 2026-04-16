//! Integration tests for Arrow ↔ Parquet ↔ Core schema conversions.
//!
//! ## Scope
//!
//! These tests validate that schema conversions between the three representation
//! models (Arrow, Parquet, Core) preserve semantic meaning and field identity.
//!
//! ## Invariants
//!
//! - Round-trip conversions preserve field names and types when conversions are exact.
//! - Zero-copy eligibility is computed consistently across models.
//! - Nested schemas are converted recursively.

use consus_arrow::{
    ArrowDataType, ArrowFieldBuilder, ArrowFieldId, ArrowFieldKind, ArrowSchema,
    conversion::{
        ConversionCompatibility, ConversionMode, analyze_conversion_compatibility,
        arrow_datatype_to_core, core_datatype_to_arrow_hint,
    },
};
use consus_core::{ByteOrder, Datatype};
use consus_parquet::{
    ArrowFieldRepr, FieldDescriptor, FieldId, LogicalType, ParquetPhysicalType, Repetition,
    SchemaDescriptor,
    conversion::{
        ParquetCompatibility, analyze_parquet_arrow_compatibility,
        arrow_nullability_to_parquet_repetition, parquet_logical_to_core_annotation,
        parquet_physical_to_core, parquet_repetition_to_arrow_nullability,
    },
};

// ---------------------------------------------------------------------------
// Arrow ↔ Core Round-Trip Tests
// ---------------------------------------------------------------------------

#[test]
fn arrow_core_boolean_roundtrip() {
    let arrow_type = ArrowDataType::Boolean;
    let core_type = arrow_datatype_to_core(&arrow_type);
    let arrow_again = core_datatype_to_arrow_hint(&core_type);

    assert!(matches!(core_type, Datatype::Boolean));
    assert!(matches!(arrow_again, ArrowDataType::Boolean));
}

#[test]
fn arrow_core_integer_roundtrip() {
    let arrow_type = ArrowDataType::Int {
        bit_width: 32,
        sign: consus_arrow::IntSign::Signed,
    };
    let core_type = arrow_datatype_to_core(&arrow_type);
    let arrow_again = core_datatype_to_arrow_hint(&core_type);

    match (&arrow_type, &core_type, &arrow_again) {
        (
            ArrowDataType::Int {
                bit_width: aw1,
                sign: s1,
            },
            Datatype::Integer {
                bits, signed: true, ..
            },
            ArrowDataType::Int {
                bit_width: aw2,
                sign: s2,
            },
        ) => {
            assert_eq!(*aw1, 32);
            assert_eq!(bits.get(), 32);
            assert_eq!(*aw2, 32);
            assert_eq!(*s1, *s2);
        }
        _ => panic!("roundtrip failed"),
    }
}

#[test]
fn arrow_core_float_roundtrip() {
    let arrow_type = ArrowDataType::Float { bit_width: 64 };
    let core_type = arrow_datatype_to_core(&arrow_type);
    let arrow_again = core_datatype_to_arrow_hint(&core_type);

    match (&core_type, &arrow_again) {
        (Datatype::Float { bits, .. }, ArrowDataType::Float { bit_width }) => {
            assert_eq!(bits.get(), 64);
            assert_eq!(*bit_width, 64);
        }
        _ => panic!("roundtrip failed"),
    }
}

#[test]
fn arrow_core_string_mapping() {
    let arrow_utf8 = ArrowDataType::Utf8;
    let core_type = arrow_datatype_to_core(&arrow_utf8);

    match core_type {
        Datatype::VariableString { encoding } => {
            assert!(matches!(encoding, consus_core::StringEncoding::Utf8));
        }
        _ => panic!("expected VariableString"),
    }
}

// ---------------------------------------------------------------------------
// Parquet ↔ Core Round-Trip Tests
// ---------------------------------------------------------------------------

#[test]
fn parquet_core_boolean_mapping() {
    let parquet_type = ParquetPhysicalType::Boolean;
    let core_type = parquet_physical_to_core(parquet_type);

    assert!(matches!(core_type, Datatype::Boolean));
}

#[test]
fn parquet_core_integer_mapping() {
    let parquet_type = ParquetPhysicalType::Int32;
    let core_type = parquet_physical_to_core(parquet_type);

    match core_type {
        Datatype::Integer {
            bits, signed: true, ..
        } => {
            assert_eq!(bits.get(), 32);
        }
        _ => panic!("expected signed 32-bit integer"),
    }
}

#[test]
fn parquet_core_float_mapping() {
    let parquet_type = ParquetPhysicalType::Double;
    let core_type = parquet_physical_to_core(parquet_type);

    match core_type {
        Datatype::Float { bits, .. } => {
            assert_eq!(bits.get(), 64);
        }
        _ => panic!("expected 64-bit float"),
    }
}

#[test]
fn parquet_logical_string_mapping() {
    let logical = LogicalType::String;
    let core_annotation = parquet_logical_to_core_annotation(&logical);

    match core_annotation {
        Some(Datatype::VariableString { encoding }) => {
            assert!(matches!(encoding, consus_core::StringEncoding::Utf8));
        }
        _ => panic!("expected VariableString"),
    }
}

#[test]
fn parquet_logical_timestamp_mapping() {
    let logical = LogicalType::Timestamp {
        unit: consus_parquet::TimeUnit::Microseconds,
        is_adjusted_to_utc: true,
    };
    let core_annotation = parquet_logical_to_core_annotation(&logical);

    match core_annotation {
        Some(Datatype::Integer {
            bits, signed: true, ..
        }) => {
            assert_eq!(bits.get(), 64);
        }
        _ => panic!("expected 64-bit signed integer"),
    }
}

// ---------------------------------------------------------------------------
// Arrow ↔ Parquet Compatibility Tests
// ---------------------------------------------------------------------------

#[test]
fn arrow_parquet_compatible_fields() {
    let parquet_field =
        FieldDescriptor::required(FieldId::new(1), "temperature", ParquetPhysicalType::Double);

    let arrow_field = ArrowFieldRepr {
        name: String::from("temperature"),
        nullable: false,
        datatype: Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
        },
    };

    let compatibility = analyze_parquet_arrow_compatibility(&parquet_field, &arrow_field);
    assert!(matches!(compatibility, ParquetCompatibility::Compatible));
}

#[test]
fn arrow_parquet_nullable_evolution() {
    let parquet_field =
        FieldDescriptor::optional(FieldId::new(1), "value", ParquetPhysicalType::Int32, None);

    let arrow_field = ArrowFieldRepr {
        name: String::from("value"),
        nullable: false, // Arrow is required, Parquet is optional
        datatype: Datatype::Integer {
            bits: core::num::NonZeroUsize::new(32).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
            signed: true,
        },
    };

    let compatibility = analyze_parquet_arrow_compatibility(&parquet_field, &arrow_field);
    assert!(matches!(
        compatibility,
        ParquetCompatibility::RequiresEvolution
    ));
}

#[test]
fn arrow_parquet_incompatible_types() {
    let parquet_field =
        FieldDescriptor::required(FieldId::new(1), "data", ParquetPhysicalType::Double);

    let arrow_field = ArrowFieldRepr {
        name: String::from("data"),
        nullable: false,
        datatype: Datatype::Boolean,
    };

    let compatibility = analyze_parquet_arrow_compatibility(&parquet_field, &arrow_field);
    assert!(matches!(compatibility, ParquetCompatibility::Incompatible));
}

// ---------------------------------------------------------------------------
// Repetition ↔ Nullability Conversion Tests
// ---------------------------------------------------------------------------

#[test]
fn repetition_nullability_roundtrip() {
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

// ---------------------------------------------------------------------------
// Conversion Mode Tests
// ---------------------------------------------------------------------------

#[test]
fn conversion_mode_strict_rejects_lossy() {
    let compatible = ConversionCompatibility::Lossy;
    assert!(!compatible.is_permitted(ConversionMode::Strict));
    assert!(compatible.is_permitted(ConversionMode::AllowLossy));
}

#[test]
fn conversion_mode_allows_lossless() {
    let compatible = ConversionCompatibility::Lossless;
    assert!(compatible.is_permitted(ConversionMode::Strict));
    assert!(compatible.is_permitted(ConversionMode::AllowLossy));
}

#[test]
fn integer_widening_is_lossless() {
    let arrow_int32 = ArrowDataType::Int {
        bit_width: 32,
        sign: consus_arrow::IntSign::Signed,
    };
    let core_int64 = Datatype::Integer {
        bits: core::num::NonZeroUsize::new(64).expect("constant"),
        byte_order: ByteOrder::LittleEndian,
        signed: true,
    };

    let compatibility = analyze_conversion_compatibility(&arrow_int32, &core_int64);
    assert!(matches!(compatibility, ConversionCompatibility::Lossless));
}

// ---------------------------------------------------------------------------
// Schema-Level Tests
// ---------------------------------------------------------------------------

#[test]
fn parquet_schema_to_core_conversion() {
    let schema = SchemaDescriptor::new(vec![
        FieldDescriptor::required(FieldId::new(1), "id", ParquetPhysicalType::Int64),
        FieldDescriptor::optional(
            FieldId::new(2),
            "name",
            ParquetPhysicalType::ByteArray,
            Some(LogicalType::String),
        ),
        FieldDescriptor::required(FieldId::new(3), "score", ParquetPhysicalType::Double),
    ]);

    let core_pairs = consus_parquet::conversion::parquet_schema_to_core_pairs(&schema);

    assert_eq!(core_pairs.len(), 3);
    assert_eq!(core_pairs[0].0, "id");
    assert_eq!(core_pairs[1].0, "name");
    assert_eq!(core_pairs[2].0, "score");

    // Verify id is Int64
    match &core_pairs[0].1 {
        Datatype::Integer {
            bits, signed: true, ..
        } => {
            assert_eq!(bits.get(), 64);
        }
        _ => panic!("expected Int64"),
    }

    // Verify name is VariableString (from logical type)
    match &core_pairs[1].1 {
        Datatype::VariableString { .. } => {}
        _ => panic!("expected VariableString"),
    }
}

#[test]
fn arrow_schema_to_core_conversion() {
    let schema = ArrowSchema::new(vec![
        ArrowFieldBuilder::new(
            ArrowFieldId::new(1),
            String::from("x"),
            ArrowFieldKind::Float,
            Datatype::Float {
                bits: core::num::NonZeroUsize::new(64).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
            },
        )
        .nullable(false)
        .build()
        .expect("field must build"),
        ArrowFieldBuilder::new(
            ArrowFieldId::new(2),
            String::from("y"),
            ArrowFieldKind::Float,
            Datatype::Float {
                bits: core::num::NonZeroUsize::new(32).expect("constant"),
                byte_order: ByteOrder::LittleEndian,
            },
        )
        .nullable(true)
        .build()
        .expect("field must build"),
    ]);

    let core_pairs = consus_arrow::conversion::arrow_schema_to_core_pairs(&schema);

    assert_eq!(core_pairs.len(), 2);
    assert_eq!(core_pairs[0].0, "x");
    assert_eq!(core_pairs[1].0, "y");
}

// ---------------------------------------------------------------------------
// Zero-Copy Eligibility Tests
// ---------------------------------------------------------------------------

#[test]
fn zero_copy_eligibility_requires_fixed_width() {
    let parquet_field =
        FieldDescriptor::required(FieldId::new(1), "value", ParquetPhysicalType::Double);

    let arrow_field = ArrowFieldBuilder::new(
        ArrowFieldId::new(1),
        String::from("value"),
        ArrowFieldKind::Float,
        Datatype::Float {
            bits: core::num::NonZeroUsize::new(64).expect("constant"),
            byte_order: ByteOrder::LittleEndian,
        },
    )
    .nullable(false)
    .build()
    .expect("field must build");

    // Fixed-width, required fields should be zero-copy eligible
    assert!(!arrow_field.is_nullable());
    assert!(arrow_field.is_fixed_size());
}

#[test]
fn variable_width_disables_zero_copy() {
    let arrow_field = ArrowFieldBuilder::new(
        ArrowFieldId::new(1),
        String::from("name"),
        ArrowFieldKind::Utf8,
        Datatype::VariableString {
            encoding: consus_core::StringEncoding::Utf8,
        },
    )
    .nullable(false)
    .build()
    .expect("field must build");

    // Variable-width fields are not zero-copy eligible
    assert!(!arrow_field.is_fixed_size());
}

// ---------------------------------------------------------------------------
// Edge Cases and Error Conditions
// ---------------------------------------------------------------------------

#[test]
fn incompatible_integer_signs() {
    let arrow_signed = ArrowDataType::Int {
        bit_width: 32,
        sign: consus_arrow::IntSign::Signed,
    };
    let core_unsigned = Datatype::Integer {
        bits: core::num::NonZeroUsize::new(32).expect("constant"),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };

    let compatibility = analyze_conversion_compatibility(&arrow_signed, &core_unsigned);
    assert!(matches!(
        compatibility,
        ConversionCompatibility::Incompatible
    ));
}

#[test]
fn decimal_precision_mapping() {
    let logical = LogicalType::Decimal {
        precision: 10,
        scale: 2,
    };
    let core_type = parquet_logical_to_core_annotation(&logical);

    match core_type {
        Some(Datatype::FixedString { length, .. }) => {
            // 10 digits + 2 decimal places + sign + decimal point
            assert!(length >= 12);
        }
        _ => panic!("expected FixedString for decimal"),
    }
}

#[test]
fn uuid_logical_type_mapping() {
    let logical = LogicalType::Uuid;
    let core_type = parquet_logical_to_core_annotation(&logical);

    match core_type {
        Some(Datatype::FixedString { length, .. }) => {
            assert_eq!(length, 16);
        }
        _ => panic!("expected 16-byte FixedString for UUID"),
    }
}
