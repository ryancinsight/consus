//! RLE/bit-packing hybrid encoding decoder for Parquet repetition and definition levels.
//!
//! Implements Parquet encoding ID 3 (RLE/bit-packing hybrid) and
//! encoding ID 4 (BIT_PACKED, deprecated).
//!
//! Reference: https://github.com/apache/parquet-format/blob/master/Encodings.md
//!
//! Wire format (encoding ID 3): each run has a 7-bit LSB-first varint header.
//! header bit 0 == 0: RLE run. header>>1 repetitions of one LE value.
//! header bit 0 == 1: bit-packed. (header>>1)+1 groups of 8 values.
//! Value i occupies bits [i*bit_width..(i+1)*bit_width-1] (LSB first).

use alloc::string::String;
use alloc::vec::Vec;
use consus_core::{Error, Result};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the bit-width required to encode levels in the range [0, max_level].
///
/// Formula: level_bit_width(0) = 0;
/// level_bit_width(n) = 32 - (n as u32).leading_zeros() for n > 0.
/// Equivalent to ceil(log2(max_level + 1)).
///
/// Invariants:
///   level_bit_width(0)  == 0,  level_bit_width(1)  == 1
///   level_bit_width(2)  == 2,  level_bit_width(3)  == 2
///   level_bit_width(4)  == 3,  level_bit_width(7)  == 3
///   level_bit_width(8)  == 4,  level_bit_width(15) == 4
///   level_bit_width(16) == 5
#[allow(clippy::cast_possible_truncation)]
pub fn level_bit_width(max_level: i32) -> u8 {
    if max_level == 0 {
        0
    } else {
        (32 - (max_level as u32).leading_zeros()) as u8
    }
}

/// Decode count level values from a RLE/bit-packing hybrid-encoded byte slice.
///
/// bit_width must be in [0, 32]. If bit_width == 0, returns count zeros without
/// consuming any bytes. If count == 0, returns an empty Vec.
/// The bytes slice must start at the first run header byte.
/// For DataPage v1 levels, strip the 4-byte LE length prefix before calling.
///
/// Errors:
///   Error::InvalidFormat  - bit_width > 32 or a varint overflows.
///   Error::BufferTooSmall - buffer is truncated before count values are decoded.
pub fn decode_levels(bytes: &[u8], bit_width: u8, count: usize) -> Result<Vec<i32>> {
    if bit_width > 32 {
        return Err(Error::InvalidFormat {
            message: String::from("bit_width exceeds 32"),
        });
    }
    if bit_width == 0 || count == 0 {
        return Ok(alloc::vec![0i32; count]);
    }

    let mut pos: usize = 0;
    let mut out: Vec<i32> = Vec::with_capacity(count);

    while out.len() < count {
        let header = read_rle_varint(bytes, &mut pos)?;
        if header & 1 == 0 {
            // RLE run: run_length copies of a single LE value.
            let run_len = (header >> 1) as usize;
            let value_bytes = (bit_width as usize + 7) / 8;
            if pos + value_bytes > bytes.len() {
                return Err(Error::BufferTooSmall {
                    required: pos + value_bytes,
                    provided: bytes.len(),
                });
            }
            let mut val: u64 = 0;
            for k in 0..value_bytes {
                val |= (bytes[pos + k] as u64) << (k * 8);
            }
            pos += value_bytes;
            let to_emit = run_len.min(count - out.len());
            for _ in 0..to_emit {
                #[allow(clippy::cast_possible_truncation)]
                out.push(val as i32);
            }
        } else {
            // Bit-packed run: (header >> 1) + 1 groups of 8 values.
            let num_groups = ((header >> 1) as usize) + 1;
            let group_bytes = bit_width as usize;
            let mut group_out = [0i32; 8];
            for _ in 0..num_groups {
                if pos + group_bytes > bytes.len() {
                    return Err(Error::BufferTooSmall {
                        required: pos + group_bytes,
                        provided: bytes.len(),
                    });
                }
                unpack_8_values(&bytes[pos..pos + group_bytes], bit_width, &mut group_out);
                pos += group_bytes;
                let to_emit = 8usize.min(count - out.len());
                out.extend_from_slice(&group_out[..to_emit]);
            }
        }
    }

    Ok(out)
}

/// Decode count values from raw bit-packed bytes without run headers.
///
/// Used for the deprecated BIT_PACKED encoding (Parquet encoding ID 4).
/// Total bytes required: ceil(count * bit_width / 8).
/// Values are packed LSB-first; same layout as individual hybrid groups.
///
/// Errors:
///   Error::InvalidFormat  - bit_width > 32.
///   Error::BufferTooSmall - bytes.len() < ceil(count * bit_width / 8).
#[allow(clippy::cast_possible_truncation)]
pub fn decode_bit_packed_raw(bytes: &[u8], bit_width: u8, count: usize) -> Result<Vec<i32>> {
    if bit_width > 32 {
        return Err(Error::InvalidFormat {
            message: String::from("bit_width exceeds 32"),
        });
    }
    if bit_width == 0 || count == 0 {
        return Ok(alloc::vec![0i32; count]);
    }

    let required = (count * bit_width as usize + 7) / 8;
    if bytes.len() < required {
        return Err(Error::BufferTooSmall {
            required,
            provided: bytes.len(),
        });
    }

    let mask: u64 = if bit_width == 32 {
        u32::MAX as u64
    } else {
        (1u64 << bit_width) - 1
    };

    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    let mut byte_idx: usize = 0;
    let mut out: Vec<i32> = Vec::with_capacity(count);

    for _ in 0..count {
        while acc_bits < bit_width as u32 {
            acc |= (bytes[byte_idx] as u64) << acc_bits;
            acc_bits += 8;
            byte_idx += 1;
        }
        out.push((acc & mask) as i32);
        acc >>= bit_width as u32;
        acc_bits -= bit_width as u32;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Read a 7-bit LSB-first unsigned varint from bytes[*pos..], advancing *pos.
///
/// Each byte contributes 7 bits; bit 7 is the continuation flag.
/// Returns Error::BufferTooSmall if exhausted before the terminal byte.
/// Returns Error::InvalidFormat if the varint would exceed 63 bits.
fn read_rle_varint(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *pos >= bytes.len() {
            return Err(Error::BufferTooSmall {
                required: *pos + 1,
                provided: bytes.len(),
            });
        }
        let byte = bytes[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            break;
        }
        if shift >= 63 {
            return Err(Error::InvalidFormat {
                message: String::from("varint overflow: exceeds 63 bits"),
            });
        }
    }
    Ok(result)
}

/// Unpack 8 values from a bit-packed group into out[0..8].
///
/// The group occupies exactly bit_width bytes.
/// Value i occupies bits [i*bit_width..(i+1)*bit_width-1] (LSB first).
/// Uses a 64-bit accumulator, loading bytes on demand.
#[allow(clippy::cast_possible_truncation)]
fn unpack_8_values(buf: &[u8], bit_width: u8, out: &mut [i32; 8]) {
    if bit_width == 0 {
        *out = [0i32; 8];
        return;
    }
    let mask: u64 = if bit_width == 32 {
        u32::MAX as u64
    } else {
        (1u64 << bit_width) - 1
    };
    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    let mut byte_idx: usize = 0;
    for value in out.iter_mut() {
        while acc_bits < bit_width as u32 && byte_idx < buf.len() {
            acc |= (buf[byte_idx] as u64) << acc_bits;
            acc_bits += 8;
            byte_idx += 1;
        }
        *value = (acc & mask) as i32;
        acc >>= bit_width as u32;
        acc_bits = acc_bits.saturating_sub(bit_width as u32);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_bit_width_matches_spec() {
        assert_eq!(level_bit_width(0), 0);
        assert_eq!(level_bit_width(1), 1);
        assert_eq!(level_bit_width(2), 2);
        assert_eq!(level_bit_width(3), 2);
        assert_eq!(level_bit_width(4), 3);
        assert_eq!(level_bit_width(7), 3);
        assert_eq!(level_bit_width(8), 4);
        assert_eq!(level_bit_width(15), 4);
        assert_eq!(level_bit_width(16), 5);
    }

    #[test]
    fn decode_levels_zero_bit_width_returns_zeros() {
        let levels = decode_levels(&[], 0, 5).unwrap();
        assert_eq!(levels, alloc::vec![0i32; 5]);
    }

    #[test]
    fn decode_levels_zero_count_returns_empty() {
        let levels = decode_levels(&[], 1, 0).unwrap();
        assert_eq!(levels, alloc::vec![0i32; 0]);
    }

    #[test]
    fn decode_levels_empty_input_errors() {
        let err = decode_levels(&[], 1, 3).unwrap_err();
        assert!(matches!(err, consus_core::Error::BufferTooSmall { .. }));
    }

    /// Header: (5 << 1) | 0 = 0x0A. Value byte: 0x01.
    #[test]
    fn decode_levels_rle_bit_width_1_five_ones() {
        let bytes = [0x0A_u8, 0x01];
        let levels = decode_levels(&bytes, 1, 5).unwrap();
        assert_eq!(levels, alloc::vec![1, 1, 1, 1, 1]);
    }

    /// Header: (4 << 1) | 0 = 0x08. Value byte: 0x03.
    #[test]
    fn decode_levels_rle_bit_width_2_four_threes() {
        let bytes = [0x08_u8, 0x03];
        let levels = decode_levels(&bytes, 2, 4).unwrap();
        assert_eq!(levels, alloc::vec![3, 3, 3, 3]);
    }

    /// Run1: (2<<1)|0=0x04 val=1->[1,1]. Run2: (3<<1)|0=0x06 val=0->[0,0,0].
    #[test]
    fn decode_levels_two_sequential_rle_runs() {
        let bytes = [0x04_u8, 0x01, 0x06, 0x00];
        let levels = decode_levels(&bytes, 1, 5).unwrap();
        assert_eq!(levels, alloc::vec![1, 1, 0, 0, 0]);
    }

    /// run_length=4 but count=2: emit only 2 values.
    #[test]
    fn decode_levels_rle_run_truncated_to_count() {
        let bytes = [0x08_u8, 0x03];
        let levels = decode_levels(&bytes, 2, 2).unwrap();
        assert_eq!(levels, alloc::vec![3, 3]);
    }

    /// Header: 0x01. Group byte: 0x4D=0b01001101 -> [1,0,1,1,0,0,1,0] LSB-first.
    #[test]
    fn decode_levels_bit_packed_bit_width_1_eight_values() {
        let bytes = [0x01_u8, 0x4D];
        let levels = decode_levels(&bytes, 1, 8).unwrap();
        assert_eq!(levels, alloc::vec![1, 0, 1, 1, 0, 0, 1, 0]);
    }

    /// Header: 0x01. Group: [0xE4, 0xE4]. 0xE4=0b11100100 -> [0,1,2,3] per 2-bit field.
    #[test]
    fn decode_levels_bit_packed_bit_width_2_eight_values() {
        let bytes = [0x01_u8, 0xE4, 0xE4];
        let levels = decode_levels(&bytes, 2, 8).unwrap();
        assert_eq!(levels, alloc::vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    /// Same bytes as above, count=4: first 4 values only.
    #[test]
    fn decode_levels_bit_packed_partial_count() {
        let bytes = [0x01_u8, 0xE4, 0xE4];
        let levels = decode_levels(&bytes, 2, 4).unwrap();
        assert_eq!(levels, alloc::vec![0, 1, 2, 3]);
    }

    /// 0xC9 = 0b11001001 -> [1, 2, 0, 3] at bit_width=2.
    #[test]
    fn decode_bit_packed_raw_bit_width_2_four_values() {
        let bytes = [0xC9_u8];
        let values = decode_bit_packed_raw(&bytes, 2, 4).unwrap();
        assert_eq!(values, alloc::vec![1, 2, 0, 3]);
    }

    /// [0x4D, 0x01] bit_width=1 count=10 -> [1,0,1,1,0,0,1,0,1,0].
    #[test]
    fn decode_bit_packed_raw_bit_width_1_ten_values() {
        let bytes = [0x4D_u8, 0x01];
        let values = decode_bit_packed_raw(&bytes, 1, 10).unwrap();
        assert_eq!(values, alloc::vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 0]);
    }

    /// Need 2 bytes for 8 values at bit_width=2; only 1 byte provided.
    #[test]
    fn decode_bit_packed_raw_truncated_errors() {
        let err = decode_bit_packed_raw(&[0xE4], 2, 8).unwrap_err();
        assert!(matches!(err, consus_core::Error::BufferTooSmall { .. }));
    }
}
