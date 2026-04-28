//! Proptest roundtrip suite for `encoding::plain`.
//!
//! ## Mathematical specification
//!
//! For each scalar type T with PLAIN encoding/decoding pair (enc, dec):
//!   dec(enc(v), 1) == [v]
//!
//! For BYTE_ARRAY: dec(enc(bytes), 1) == [bytes]
//! For BOOLEAN: see writer/tests_extra.rs (bit-packing depends on encode_bool_column_plain).
//!
//! Encoding is the canonical LE byte representation; decoding is the inverse.
//! Non-finite f32/f64 values (NaN, ±Inf) must also round-trip bit-exactly.

use super::plain::{
    decode_plain_byte_array, decode_plain_f32, decode_plain_f64, decode_plain_fixed_byte_array,
    decode_plain_i32, decode_plain_i64, decode_plain_i96,
};
use proptest::prelude::*;

// ── INT32 ────────────────────────────────────────────────────────────────────

proptest! {
    /// ∀ v ∈ i32: decode_plain_i32(v.to_le_bytes(), 1) == [v]
    #[test]
    fn prop_i32_plain_roundtrip(v in i32::MIN..=i32::MAX) {
        let encoded = v.to_le_bytes();
        let decoded = decode_plain_i32(&encoded, 1).unwrap();
        prop_assert_eq!(decoded, alloc::vec![v]);
    }
}

// ── INT64 ────────────────────────────────────────────────────────────────────

proptest! {
    /// ∀ v ∈ i64: decode_plain_i64(v.to_le_bytes(), 1) == [v]
    #[test]
    fn prop_i64_plain_roundtrip(v in i64::MIN..=i64::MAX) {
        let encoded = v.to_le_bytes();
        let decoded = decode_plain_i64(&encoded, 1).unwrap();
        prop_assert_eq!(decoded, alloc::vec![v]);
    }
}

// ── FLOAT (f32) ──────────────────────────────────────────────────────────────

proptest! {
    /// ∀ bits ∈ u32: decode_plain_f32(bits_as_le_bytes, 1) == [f32::from_bits(bits)]
    ///
    /// Uses raw bit representation to cover NaN payloads without triggering
    /// f32::NAN != f32::NAN comparisons.
    #[test]
    fn prop_f32_plain_roundtrip_bits(bits in u32::MIN..=u32::MAX) {
        let v = f32::from_bits(bits);
        let encoded = v.to_le_bytes();
        let decoded = decode_plain_f32(&encoded, 1).unwrap();
        prop_assert_eq!(decoded[0].to_bits(), v.to_bits(),
            "f32 bit pattern did not survive roundtrip: input bits={}", bits);
    }
}

// ── DOUBLE (f64) ─────────────────────────────────────────────────────────────

proptest! {
    /// ∀ bits ∈ u64: decode_plain_f64(bits_as_le_bytes, 1) == [f64::from_bits(bits)]
    #[test]
    fn prop_f64_plain_roundtrip_bits(bits in u64::MIN..=u64::MAX) {
        let v = f64::from_bits(bits);
        let encoded = v.to_le_bytes();
        let decoded = decode_plain_f64(&encoded, 1).unwrap();
        prop_assert_eq!(decoded[0].to_bits(), v.to_bits(),
            "f64 bit pattern did not survive roundtrip: input bits={}", bits);
    }
}

// ── INT96 ────────────────────────────────────────────────────────────────────

proptest! {
    /// ∀ raw ∈ [u8; 12]: decode_plain_i96(raw, 1) == [raw]
    #[test]
    fn prop_i96_plain_roundtrip(raw in proptest::array::uniform12(0u8..=255)) {
        let decoded = decode_plain_i96(&raw, 1).unwrap();
        prop_assert_eq!(decoded, alloc::vec![raw]);
    }
}

// ── BYTE_ARRAY ───────────────────────────────────────────────────────────────

proptest! {
    /// ∀ data ∈ Vec<u8>: decode_plain_byte_array(encode_ba(data), 1) == [data]
    ///
    /// PLAIN BYTE_ARRAY encoding: 4-byte LE u32 length prefix + raw bytes.
    #[test]
    fn prop_byte_array_plain_roundtrip(
        data in proptest::collection::vec(0u8..=255, 0..256)
    ) {
        let mut encoded = alloc::vec::Vec::with_capacity(4 + data.len());
        let len = data.len() as u32;
        encoded.extend_from_slice(&len.to_le_bytes());
        encoded.extend_from_slice(&data);
        let decoded = decode_plain_byte_array(&encoded, 1).unwrap();
        prop_assert_eq!(decoded, alloc::vec![data]);
    }
}

// ── FIXED_LEN_BYTE_ARRAY ─────────────────────────────────────────────────────

proptest! {
    /// ∀ data ∈ [u8; N], N ∈ 1..=16:
    ///   decode_plain_fixed_byte_array(data, 1, N) == [data]
    #[test]
    fn prop_fixed_len_byte_array_plain_roundtrip(
        data in proptest::collection::vec(0u8..=255, 1..=16)
    ) {
        let fl = data.len();
        let decoded = decode_plain_fixed_byte_array(&data, 1, fl).unwrap();
        prop_assert_eq!(decoded, alloc::vec![data]);
    }
}
