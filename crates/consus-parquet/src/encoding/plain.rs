//! PLAIN encoding (ID 0) decoders for all Parquet physical types.
//!
//! Reference: https://github.com/apache/parquet-format/blob/master/Encodings.md

use alloc::vec::Vec;
use consus_core::{Error, Result};

/// Decode count boolean values from PLAIN-encoded bytes.
///
/// Booleans are packed LSB-first: value[i] is bit (i % 8) of byte (i / 8).
/// Required bytes: ceil(count / 8).
pub fn decode_plain_boolean(bytes: &[u8], count: usize) -> Result<Vec<bool>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let required = count.checked_add(7).ok_or(Error::Overflow)? / 8;
    if bytes.len() < required {
        return Err(Error::BufferTooSmall {
            required,
            provided: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push((bytes[i / 8] >> (i % 8)) & 1 != 0);
    }
    Ok(out)
}

/// Decode count INT32 values from PLAIN-encoded bytes.
///
/// Each value is a 4-byte little-endian signed integer.
/// Required bytes: count * 4.
pub fn decode_plain_i32(bytes: &[u8], count: usize) -> Result<Vec<i32>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let required = count.checked_mul(4).ok_or(Error::Overflow)?;
    if bytes.len() < required {
        return Err(Error::BufferTooSmall {
            required,
            provided: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = i * 4;
        out.push(i32::from_le_bytes([
            bytes[b],
            bytes[b + 1],
            bytes[b + 2],
            bytes[b + 3],
        ]));
    }
    Ok(out)
}

/// Decode count INT64 values from PLAIN-encoded bytes.
///
/// Each value is an 8-byte little-endian signed integer.
/// Required bytes: count * 8.
pub fn decode_plain_i64(bytes: &[u8], count: usize) -> Result<Vec<i64>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let required = count.checked_mul(8).ok_or(Error::Overflow)?;
    if bytes.len() < required {
        return Err(Error::BufferTooSmall {
            required,
            provided: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = i * 8;
        out.push(i64::from_le_bytes([
            bytes[b],
            bytes[b + 1],
            bytes[b + 2],
            bytes[b + 3],
            bytes[b + 4],
            bytes[b + 5],
            bytes[b + 6],
            bytes[b + 7],
        ]));
    }
    Ok(out)
}

/// Decode count INT96 values from PLAIN-encoded bytes.
///
/// Each value is 12 raw bytes (deprecated Parquet timestamp type).
/// Required bytes: count * 12.
pub fn decode_plain_i96(bytes: &[u8], count: usize) -> Result<Vec<[u8; 12]>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let required = count.checked_mul(12).ok_or(Error::Overflow)?;
    if bytes.len() < required {
        return Err(Error::BufferTooSmall {
            required,
            provided: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = i * 12;
        let mut arr = [0u8; 12];
        arr.copy_from_slice(&bytes[b..b + 12]);
        out.push(arr);
    }
    Ok(out)
}

/// Decode count FLOAT values from PLAIN-encoded bytes.
///
/// Each value is a 4-byte little-endian IEEE 754 single-precision float.
/// Required bytes: count * 4.
pub fn decode_plain_f32(bytes: &[u8], count: usize) -> Result<Vec<f32>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let required = count.checked_mul(4).ok_or(Error::Overflow)?;
    if bytes.len() < required {
        return Err(Error::BufferTooSmall {
            required,
            provided: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = i * 4;
        out.push(f32::from_le_bytes([
            bytes[b],
            bytes[b + 1],
            bytes[b + 2],
            bytes[b + 3],
        ]));
    }
    Ok(out)
}

/// Decode count DOUBLE values from PLAIN-encoded bytes.
///
/// Each value is an 8-byte little-endian IEEE 754 double-precision float.
/// Required bytes: count * 8.
pub fn decode_plain_f64(bytes: &[u8], count: usize) -> Result<Vec<f64>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let required = count.checked_mul(8).ok_or(Error::Overflow)?;
    if bytes.len() < required {
        return Err(Error::BufferTooSmall {
            required,
            provided: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = i * 8;
        out.push(f64::from_le_bytes([
            bytes[b],
            bytes[b + 1],
            bytes[b + 2],
            bytes[b + 3],
            bytes[b + 4],
            bytes[b + 5],
            bytes[b + 6],
            bytes[b + 7],
        ]));
    }
    Ok(out)
}

/// Decode count BYTE_ARRAY values from PLAIN-encoded bytes.
///
/// Each value is prefixed by a 4-byte LE unsigned length, followed by that many bytes.
pub fn decode_plain_byte_array(bytes: &[u8], count: usize) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::with_capacity(count);
    let mut pos = 0usize;
    for _ in 0..count {
        let after_prefix = pos.checked_add(4).ok_or(Error::Overflow)?;
        if after_prefix > bytes.len() {
            return Err(Error::BufferTooSmall {
                required: after_prefix,
                provided: bytes.len(),
            });
        }
        let len = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        pos = after_prefix;
        let end = pos.checked_add(len).ok_or(Error::Overflow)?;
        if end > bytes.len() {
            return Err(Error::BufferTooSmall {
                required: end,
                provided: bytes.len(),
            });
        }
        out.push(bytes[pos..end].to_vec());
        pos = end;
    }
    Ok(out)
}

/// Decode count FIXED_LEN_BYTE_ARRAY values from PLAIN-encoded bytes.
///
/// Each value occupies exactly fixed_len bytes.
/// When fixed_len == 0, returns count empty Vecs without consuming any bytes.
/// Required bytes: count * fixed_len.
pub fn decode_plain_fixed_byte_array(
    bytes: &[u8],
    count: usize,
    fixed_len: usize,
) -> Result<Vec<Vec<u8>>> {
    if fixed_len == 0 {
        return Ok((0..count).map(|_| Vec::new()).collect());
    }
    let required = count.checked_mul(fixed_len).ok_or(Error::Overflow)?;
    if bytes.len() < required {
        return Err(Error::BufferTooSmall {
            required,
            provided: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = i * fixed_len;
        out.push(bytes[b..b + fixed_len].to_vec());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: [1, -1, i32::MAX] -> 01000000 FFFFFFFF FFFFFF7F
    #[test]
    fn decode_plain_i32_three_values() {
        let bytes = [
            0x01u8, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F,
        ];
        let result = decode_plain_i32(&bytes, 3).unwrap();
        assert_eq!(result, alloc::vec![1i32, -1i32, i32::MAX]);
    }

    // Test 2: [100i64, -200i64]
    // 100  LE: 64 00 00 00 00 00 00 00
    // -200 LE: 38 FF FF FF FF FF FF FF
    #[test]
    fn decode_plain_i64_two_values() {
        let bytes = [
            0x64u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF,
        ];
        let result = decode_plain_i64(&bytes, 2).unwrap();
        assert_eq!(result, alloc::vec![100i64, -200i64]);
    }

    // Test 3: [1.0f64, -0.5f64]
    // 1.0  = 3FF0000000000000 LE: 00 00 00 00 00 00 F0 3F
    // -0.5 = BFE0000000000000 LE: 00 00 00 00 00 00 E0 BF
    #[test]
    fn decode_plain_f64_two_values() {
        let bytes = [
            0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xE0, 0xBF,
        ];
        let result = decode_plain_f64(&bytes, 2).unwrap();
        assert_eq!(result, alloc::vec![1.0f64, -0.5f64]);
    }

    // Test 4: [0.0f32, 1.0f32]
    // 0.0 = 00000000 LE: 00 00 00 00
    // 1.0 = 3F800000 LE: 00 00 80 3F
    #[test]
    fn decode_plain_f32_two_values() {
        let bytes = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F];
        let result = decode_plain_f32(&bytes, 2).unwrap();
        assert_eq!(result, alloc::vec![0.0f32, 1.0f32]);
    }

    // Test 5: 10 booleans [T,F,T,T,F,F,T,F,T,F]
    // Byte 0: bit0=T,bit1=F,bit2=T,bit3=T,bit4=F,bit5=F,bit6=T,bit7=F = 0x4D
    // Byte 1: bit0=T,bit1=F = 0x01
    #[test]
    fn decode_plain_boolean_ten_values() {
        let result = decode_plain_boolean(&[0x4Du8, 0x01], 10).unwrap();
        assert_eq!(
            result,
            alloc::vec![true, false, true, true, false, false, true, false, true, false]
        );
    }

    // Test 6: byte arrays "ab" (len=2) and "xyz" (len=3)
    #[test]
    fn decode_plain_byte_array_two_values() {
        let bytes = [
            0x02u8, 0x00, 0x00, 0x00, 0x61, 0x62, 0x03, 0x00, 0x00, 0x00, 0x78, 0x79, 0x7A,
        ];
        let result = decode_plain_byte_array(&bytes, 2).unwrap();
        assert_eq!(result, alloc::vec![b"ab".to_vec(), b"xyz".to_vec()]);
    }

    // Test 7: fixed_len=3, values "abc" and "def"
    #[test]
    fn decode_plain_fixed_byte_array_two_values() {
        let bytes = [0x61u8, 0x62, 0x63, 0x64, 0x65, 0x66];
        let result = decode_plain_fixed_byte_array(&bytes, 2, 3).unwrap();
        assert_eq!(result, alloc::vec![b"abc".to_vec(), b"def".to_vec()]);
    }

    // Test 8: fixed_len=0 returns count empty Vecs
    #[test]
    fn decode_plain_fixed_byte_array_zero_len() {
        let result = decode_plain_fixed_byte_array(&[], 3, 0).unwrap();
        assert_eq!(result.len(), 3);
        for v in &result {
            assert!(v.is_empty());
        }
    }

    // Test 9: count=0 returns empty vec
    #[test]
    fn decode_plain_i32_zero_count() {
        let result = decode_plain_i32(&[], 0).unwrap();
        assert_eq!(result, alloc::vec![]);
    }

    // Test 10: 3-byte input for 1 i32 -> BufferTooSmall{required:4, provided:3}
    #[test]
    fn decode_plain_i32_truncated_errors() {
        let err = decode_plain_i32(&[0x01, 0x02, 0x03], 1).unwrap_err();
        assert!(matches!(
            err,
            consus_core::Error::BufferTooSmall {
                required: 4,
                provided: 3
            }
        ));
    }

    // Test 11: empty input for 1 i64 -> BufferTooSmall{required:8, provided:0}
    #[test]
    fn decode_plain_i64_empty_errors() {
        let err = decode_plain_i64(&[], 1).unwrap_err();
        assert!(matches!(
            err,
            consus_core::Error::BufferTooSmall {
                required: 8,
                provided: 0
            }
        ));
    }

    // Test 12: i96 with 12 sequential bytes 00..0B
    #[test]
    fn decode_plain_i96_one_value() {
        let bytes = [
            0x00u8, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ];
        let result = decode_plain_i96(&bytes, 1).unwrap();
        assert_eq!(
            result,
            alloc::vec![[0x00u8, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B]]
        );
    }

    // Test 13: 3-byte input for 1 byte_array -> truncated length prefix
    #[test]
    fn decode_plain_byte_array_truncated_length_errors() {
        let err = decode_plain_byte_array(&[0x05, 0x00, 0x00], 1).unwrap_err();
        assert!(matches!(err, consus_core::Error::BufferTooSmall { .. }));
    }

    // Test 14: boolean count=0 returns empty
    #[test]
    fn decode_plain_boolean_zero_count() {
        let result = decode_plain_boolean(&[], 0).unwrap();
        assert_eq!(result, alloc::vec![]);
    }
}
