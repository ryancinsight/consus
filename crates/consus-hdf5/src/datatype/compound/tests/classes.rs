//! Reference, enum, variable-length, array, and rejection-path tests.

use super::*;

#[test]
fn parse_opaque_with_tag() {
    let hdr = dt_header(OPAQUE, 1, [0, 0, 0], 100);
    let mut msg = hdr.to_vec();
    // Tag "mytype" + null + padding to 8 bytes
    let tag = b"mytype\0\0"; // 8 bytes total
    msg.extend_from_slice(tag);
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Opaque { size, tag } => {
            assert_eq!(size, 100);
            assert_eq!(tag.as_deref(), Some("mytype"));
        }
        other => panic!("expected Opaque, got: {other:?}"),
    }
}

#[test]
fn parse_opaque_no_tag() {
    let hdr = dt_header(OPAQUE, 1, [0, 0, 0], 8);
    let msg = hdr.to_vec(); // empty properties
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Opaque { size, tag } => {
            assert_eq!(size, 8);
            assert!(tag.is_none());
        }
        other => panic!("expected Opaque, got: {other:?}"),
    }
}

// -- Class 6: Compound ---------------------------------------------------

#[test]
fn parse_object_reference() {
    let hdr = dt_header(REFERENCE, 1, [0x00, 0, 0], 8);
    let msg = hdr.to_vec();
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    assert_eq!(dt, Datatype::Reference(ReferenceType::Object));
}

#[test]
fn parse_region_reference() {
    let hdr = dt_header(REFERENCE, 1, [0x01, 0, 0], 12);
    let msg = hdr.to_vec();
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    assert_eq!(dt, Datatype::Reference(ReferenceType::Region));
}

// -- Class 8: Enum -------------------------------------------------------

#[test]
fn parse_enum_v3_two_members() {
    // Enum with u8 base, 2 members: RED=0, GREEN=1
    let num_members: u16 = 2;
    let hdr = dt_header(
        ENUM,
        3,
        [
            num_members.to_le_bytes()[0],
            num_members.to_le_bytes()[1],
            0,
        ],
        1, // enum element size = base size = 1
    );
    let mut msg = hdr.to_vec();

    // Base type: u8 unsigned LE
    msg.extend_from_slice(&int_msg(1, false, true));

    // Names (version 3: no padding)
    msg.extend_from_slice(b"RED\0");
    msg.extend_from_slice(b"GREEN\0");

    // Values: 0, 1 (each 1 byte)
    msg.push(0x00);
    msg.push(0x01);

    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Enum {
            ref base,
            ref members,
        } => {
            assert!(matches!(
                base.as_ref(),
                Datatype::Integer {
                    bits,
                    signed: false,
                    ..
                } if bits.get() == 8
            ));
            assert_eq!(members.len(), 2);
            assert_eq!(members[0].name, "RED");
            assert_eq!(members[0].value, 0);
            assert_eq!(members[1].name, "GREEN");
            assert_eq!(members[1].value, 1);
        }
        other => panic!("expected Enum, got: {other:?}"),
    }
}

#[test]
fn parse_enum_signed_base() {
    // Enum with i32 base, 1 member: NEGATIVE = -42
    let num_members: u16 = 1;
    let hdr = dt_header(
        ENUM,
        3,
        [
            num_members.to_le_bytes()[0],
            num_members.to_le_bytes()[1],
            0,
        ],
        4,
    );
    let mut msg = hdr.to_vec();
    msg.extend_from_slice(&int_msg(4, true, true)); // i32 LE
    msg.extend_from_slice(b"NEGATIVE\0");
    msg.extend_from_slice(&(-42i32).to_le_bytes());

    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Enum { ref members, .. } => {
            assert_eq!(members.len(), 1);
            assert_eq!(members[0].name, "NEGATIVE");
            assert_eq!(members[0].value, -42);
        }
        other => panic!("expected Enum, got: {other:?}"),
    }
}

// -- Class 9: Variable-length --------------------------------------------

#[test]
fn parse_vl_string_utf8() {
    // VL string: type=1, charset=1 (UTF-8) in flags byte 1
    let hdr = dt_header(VARIABLE_LENGTH, 1, [0x01, 0x01, 0], 16);
    let msg = hdr.to_vec();
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::VariableString { encoding } => {
            assert_eq!(encoding, StringEncoding::Utf8);
        }
        other => panic!("expected VariableString, got: {other:?}"),
    }
}

#[test]
fn parse_vl_sequence() {
    // VL sequence: type=0, base type = u32 LE
    let hdr = dt_header(VARIABLE_LENGTH, 1, [0x00, 0, 0], 16);
    let mut msg = hdr.to_vec();
    msg.extend_from_slice(&int_msg(4, false, true));
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::VarLen { ref base } => {
            assert!(matches!(
                base.as_ref(),
                Datatype::Integer {
                    bits,
                    signed: false,
                    ..
                } if bits.get() == 32
            ));
        }
        other => panic!("expected VarLen, got: {other:?}"),
    }
}

// -- Class 10: Array -----------------------------------------------------

#[test]
fn parse_array_v3_2d() {
    // 2-D array [3][5] of f64 LE
    let hdr = dt_header(ARRAY, 3, [0, 0, 0], 120); // 3*5*8=120
    let mut msg = hdr.to_vec();
    msg.push(2); // rank = 2
    msg.extend_from_slice(&3u32.to_le_bytes()); // dim 0
    msg.extend_from_slice(&5u32.to_le_bytes()); // dim 1
    // Base type: f64 LE (8 bytes + 12 props = 20 bytes)
    let base_hdr = dt_header(FLOATING_POINT, 1, [0x00, 0, 0], 8);
    msg.extend_from_slice(&base_hdr);
    msg.extend_from_slice(&[0u8; 12]);

    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Array { ref base, ref dims } => {
            assert_eq!(dims, &[3, 5]);
            assert!(matches!(
                base.as_ref(),
                Datatype::Float { bits, .. } if bits.get() == 64
            ));
        }
        other => panic!("expected Array, got: {other:?}"),
    }
}

#[test]
fn parse_array_v2_with_reserved() {
    // Version 2: rank + 3 reserved + dims + perm_indices + base
    let hdr = dt_header(ARRAY, 2, [0, 0, 0], 40); // [10] of u32 = 40
    let mut msg = hdr.to_vec();
    msg.push(1); // rank = 1
    msg.extend_from_slice(&[0u8; 3]); // reserved
    msg.extend_from_slice(&10u32.to_le_bytes()); // dim 0
    msg.extend_from_slice(&0u32.to_le_bytes()); // perm index (deprecated)
    msg.extend_from_slice(&int_msg(4, false, true)); // base: u32 LE

    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Array { ref base, ref dims } => {
            assert_eq!(dims, &[10]);
            assert!(matches!(
                base.as_ref(),
                Datatype::Integer {
                    bits,
                    signed: false,
                    ..
                } if bits.get() == 32
            ));
        }
        other => panic!("expected Array, got: {other:?}"),
    }
}

// -- Error paths ---------------------------------------------------------

#[test]
fn truncated_header_rejected() {
    let err = parse_datatype(&[0x00, 0x00, 0x00], &ParseBudget::DEFAULT).unwrap_err();
    assert!(matches!(err, Error::InvalidFormat { .. }));
}

#[test]
fn unknown_class_rejected() {
    let hdr = dt_header(15, 1, [0, 0, 0], 4);
    let err = parse_datatype(&hdr, &ParseBudget::DEFAULT).unwrap_err();
    assert!(matches!(err, Error::UnsupportedFeature { .. }));
}

#[test]
fn time_class_rejected() {
    let hdr = dt_header(TIME, 1, [0, 0, 0], 4);
    let err = parse_datatype(&hdr, &ParseBudget::DEFAULT).unwrap_err();
    assert!(matches!(err, Error::UnsupportedFeature { .. }));
}

#[test]
fn zero_numeric_size_rejected_without_panicking() {
    let err = parse_datatype(&int_msg(0, false, true), &ParseBudget::DEFAULT).unwrap_err();
    assert!(matches!(err, Error::InvalidFormat { .. }));
}

// -- Helpers unit tests --------------------------------------------------

#[test]
fn read_uint_le_widths() {
    let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    assert_eq!(read_uint_le(&data, 0), 0);
    assert_eq!(read_uint_le(&data, 1), 0x01);
    assert_eq!(read_uint_le(&data, 2), 0x0201);
    assert_eq!(read_uint_le(&data, 3), 0x0003_0201);
    assert_eq!(read_uint_le(&data, 4), 0x0403_0201);
    assert_eq!(read_uint_le(&data, 8), 0x0807_0605_0403_0201);
}

#[test]
fn read_uint_be_widths() {
    let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    assert_eq!(read_uint_be(&data, 0), 0);
    assert_eq!(read_uint_be(&data, 1), 0x01);
    assert_eq!(read_uint_be(&data, 2), 0x0102);
    assert_eq!(read_uint_be(&data, 4), 0x0102_0304);
    assert_eq!(read_uint_be(&data, 8), 0x0102_0304_0506_0708);
}

#[test]
fn sign_extend_values() {
    // Positive i8 (0x7F = 127)
    assert_eq!(sign_extend(0x7F, 1), 127);
    // Negative i8 (0xFF = -1)
    assert_eq!(sign_extend(0xFF, 1), -1);
    // Negative i16 (0xFFFE = -2)
    assert_eq!(sign_extend(0xFFFE, 2), -2);
    // Positive i32
    assert_eq!(sign_extend(42, 4), 42);
    // Negative i32 (0xFFFF_FFD6 = -42)
    assert_eq!(sign_extend(0xFFFF_FFD6, 4), -42);
}

#[test]
fn member_offset_byte_count_thresholds() {
    assert_eq!(member_offset_byte_count(0), 1);
    assert_eq!(member_offset_byte_count(255), 1);
    assert_eq!(member_offset_byte_count(256), 2);
    assert_eq!(member_offset_byte_count(65_535), 2);
    assert_eq!(member_offset_byte_count(65_536), 3);
    assert_eq!(member_offset_byte_count(16_777_215), 3);
    assert_eq!(member_offset_byte_count(16_777_216), 4);
}
