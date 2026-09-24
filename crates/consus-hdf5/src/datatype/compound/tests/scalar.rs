//! Scalar datatype parsing tests (fixed/float/string/bitfield).

use super::*;

#[test]
fn parse_u32_le() {
    let msg = int_msg(4, false, true);
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            assert_eq!(bits.get(), 32);
            assert_eq!(byte_order, ByteOrder::LittleEndian);
            assert!(!signed);
        }
        other => panic!("expected Integer, got: {other:?}"),
    }
}

#[test]
fn parse_i16_be() {
    let msg = int_msg(2, true, false);
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            assert_eq!(bits.get(), 16);
            assert_eq!(byte_order, ByteOrder::BigEndian);
            assert!(signed);
        }
        other => panic!("expected Integer, got: {other:?}"),
    }
}

// -- Class 1: Floating-point ---------------------------------------------

#[test]
fn parse_f64_le() {
    let hdr = dt_header(FLOATING_POINT, 1, [0x00, 0, 0], 8);
    let mut msg = hdr.to_vec();
    // 12 bytes of properties (content irrelevant for mapping)
    msg.extend_from_slice(&[0u8; 12]);
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Float { bits, byte_order } => {
            assert_eq!(bits.get(), 64);
            assert_eq!(byte_order, ByteOrder::LittleEndian);
        }
        other => panic!("expected Float, got: {other:?}"),
    }
}

// -- Class 3: String -----------------------------------------------------

#[test]
fn parse_fixed_string_ascii() {
    // Padding = 0 (null-terminate), charset = 0 (ASCII), size = 10
    let hdr = dt_header(STRING, 1, [0x00, 0x00, 0], 10);
    let msg = hdr.to_vec();
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::FixedString { length, encoding } => {
            assert_eq!(length, 10);
            assert_eq!(encoding, StringEncoding::Ascii);
        }
        other => panic!("expected FixedString, got: {other:?}"),
    }
}

#[test]
fn parse_fixed_string_utf8() {
    // charset = 1 (UTF-8)
    let hdr = dt_header(STRING, 1, [0x00, 0x01, 0], 32);
    let msg = hdr.to_vec();
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::FixedString { length, encoding } => {
            assert_eq!(length, 32);
            assert_eq!(encoding, StringEncoding::Utf8);
        }
        other => panic!("expected FixedString, got: {other:?}"),
    }
}

// -- Class 4: Bitfield ---------------------------------------------------

#[test]
fn parse_bitfield_2byte_le() {
    let hdr = dt_header(BITFIELD, 1, [0x00, 0, 0], 2);
    let mut msg = hdr.to_vec();
    msg.extend_from_slice(&[0u8; 4]); // properties
    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            assert_eq!(bits.get(), 16);
            assert_eq!(byte_order, ByteOrder::LittleEndian);
            assert!(!signed);
        }
        other => panic!("expected Integer from bitfield, got: {other:?}"),
    }
}

// -- Class 5: Opaque -----------------------------------------------------
