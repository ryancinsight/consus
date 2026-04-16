//! Variable-width integer encoding and low-level binary helpers.
//!
//! ## Scope
//!
//! - LEB128 unsigned/signed encoding and decoding.
//! - NUL-terminated byte-string reading.
//! - Alignment arithmetic.
//!
//! All decoding functions operate on `&[u8]` and require no allocator.
//! Encoding functions that produce variable-length output require `alloc`
//! (for `Vec<u8>` growth).
//!
//! ## Note on error construction
//!
//! The workspace dependency on `consus-core` does not set
//! `default-features = false`, so `consus-core`'s default features
//! (`std` → `alloc`) are always enabled. `Error::InvalidFormat` therefore
//! always carries `message: String`. All error-construction paths use
//! `alloc::string::String` unconditionally.

/// Maximum number of bytes in a ULEB128-encoded `u64`.
///
/// Proof: `ceil(64 / 7) = 10`. Each byte carries 7 payload bits.
const ULEB128_MAX_BYTES: usize = 10;

/// Maximum number of bytes in a SLEB128-encoded `i64`.
///
/// Proof: 64 data bits + 1 sign bit → `ceil(65 / 7) = 10`.
const SLEB128_MAX_BYTES: usize = 10;

/// Encode a `u64` as an unsigned LEB128 variable-length integer.
///
/// ## Encoding
///
/// Each output byte stores 7 data bits in bits \[6:0\]. Bit 7 is the
/// continuation flag: 1 = more bytes follow, 0 = final byte.
///
/// ## Properties
///
/// - Values 0..=127 encode in 1 byte.
/// - Values 128..=16383 encode in 2 bytes.
/// - Maximum encoding length: 10 bytes for `u64::MAX`.
///
/// Returns the number of bytes appended to `output`.
#[cfg(feature = "alloc")]
pub fn encode_uleb128(mut value: u64, output: &mut alloc::vec::Vec<u8>) -> usize {
    let start = output.len();
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        output.push(byte);
        if value == 0 {
            break;
        }
    }
    output.len() - start
}

/// Decode an unsigned LEB128 variable-length integer from a byte slice.
///
/// Returns `(decoded_value, bytes_consumed)`.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if:
/// - The encoding exceeds 10 bytes (would overflow `u64`).
/// - The slice is truncated (continuation bit set on the last available byte).
pub fn decode_uleb128(input: &[u8]) -> consus_core::Result<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    for (i, &byte) in input.iter().enumerate() {
        if i >= ULEB128_MAX_BYTES {
            return Err(consus_core::Error::InvalidFormat {
                message: alloc::string::String::from("ULEB128 encoding exceeds 10 bytes"),
            });
        }

        let low7 = (byte & 0x7F) as u64;
        result |= low7 << shift;
        shift += 7;

        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
    }

    // Reached end of slice with continuation bit still set.
    Err(consus_core::Error::InvalidFormat {
        message: alloc::string::String::from("truncated ULEB128 encoding"),
    })
}

/// Encode an `i64` as a signed LEB128 variable-length integer.
///
/// Uses two's complement representation with sign extension. The final byte's
/// bit 6 carries the sign: if the remaining value is negative the high bits
/// are sign-extended.
///
/// Returns the number of bytes appended to `output`.
#[cfg(feature = "alloc")]
pub fn encode_sleb128(mut value: i64, output: &mut alloc::vec::Vec<u8>) -> usize {
    let start = output.len();
    let mut more = true;

    while more {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        // If the sign bit of the current byte matches the remaining value,
        // this is the last byte.
        let sign_bit_set = byte & 0x40 != 0;
        if (value == 0 && !sign_bit_set) || (value == -1 && sign_bit_set) {
            more = false;
        } else {
            byte |= 0x80;
        }
        output.push(byte);
    }
    output.len() - start
}

/// Decode a signed LEB128 variable-length integer from a byte slice.
///
/// Returns `(decoded_value, bytes_consumed)`.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if the encoding exceeds 10 bytes
/// or the slice is truncated.
pub fn decode_sleb128(input: &[u8]) -> consus_core::Result<(i64, usize)> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    let mut last_byte: u8 = 0;

    for (i, &b) in input.iter().enumerate() {
        if i >= SLEB128_MAX_BYTES {
            return Err(consus_core::Error::InvalidFormat {
                message: alloc::string::String::from("SLEB128 encoding exceeds 10 bytes"),
            });
        }

        last_byte = b;
        let low7 = (last_byte & 0x7F) as i64;
        result |= low7 << shift;
        shift += 7;

        if last_byte & 0x80 == 0 {
            // Sign-extend if the sign bit (bit 6) of the final byte is set
            // and shift has not yet covered all 64 bits.
            if shift < 64 && (last_byte & 0x40) != 0 {
                result |= !0i64 << shift;
            }
            return Ok((result, i + 1));
        }
    }

    // Reached end of slice with continuation bit still set.
    let _ = last_byte;
    Err(consus_core::Error::InvalidFormat {
        message: alloc::string::String::from("truncated SLEB128 encoding"),
    })
}

/// Read a NUL-terminated byte string from a slice.
///
/// Returns `(payload, total_consumed)` where `payload` is the byte content
/// *excluding* the NUL terminator and `total_consumed` *includes* it.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` if no NUL byte is found in `input`.
pub fn read_null_terminated(input: &[u8]) -> consus_core::Result<(&[u8], usize)> {
    match input.iter().position(|&b| b == 0) {
        Some(pos) => Ok((&input[..pos], pos + 1)),
        None => Err(consus_core::Error::InvalidFormat {
            message: alloc::string::String::from("no NUL terminator found in input"),
        }),
    }
}

/// Align `offset` upward to the next multiple of `alignment`.
///
/// If `offset` is already aligned, it is returned unchanged.
///
/// ## Invariants
///
/// - `align_up(offset, a) >= offset`
/// - `align_up(offset, a) % a == 0`
/// - `align_up(offset, a) - offset < a`
///
/// ## Panics
///
/// Panics if `alignment` is zero or not a power of two.
pub const fn align_up(offset: u64, alignment: u64) -> u64 {
    assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two"
    );
    (offset + alignment - 1) & !(alignment - 1)
}

/// Compute the number of padding bytes required to align `offset` to
/// `alignment`.
///
/// Equivalent to `align_up(offset, alignment) - offset`.
///
/// ## Panics
///
/// Panics if `alignment` is zero or not a power of two.
pub const fn padding_for(offset: u64, alignment: u64) -> u64 {
    align_up(offset, alignment) - offset
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ULEB128 encode ---------------------------------------------------

    #[test]
    fn uleb128_encode_zero() {
        let mut buf = alloc::vec::Vec::new();
        let n = encode_uleb128(0, &mut buf);
        assert_eq!(n, 1);
        assert_eq!(buf.as_slice(), &[0x00]);
    }

    #[test]
    fn uleb128_encode_127() {
        let mut buf = alloc::vec::Vec::new();
        let n = encode_uleb128(127, &mut buf);
        assert_eq!(n, 1);
        assert_eq!(buf.as_slice(), &[0x7F]);
    }

    #[test]
    fn uleb128_encode_128() {
        let mut buf = alloc::vec::Vec::new();
        let n = encode_uleb128(128, &mut buf);
        assert_eq!(n, 2);
        assert_eq!(buf.as_slice(), &[0x80, 0x01]);
    }

    #[test]
    fn uleb128_encode_624485() {
        // 624485 = 0x98765
        // Binary: 1001 1000 0111 0110 0101
        // Groups of 7 (LSB first): 1100101  0001110  0100110
        //   byte0: 1100101 | 0x80 = 0xE5
        //   byte1: 0001110 | 0x80 = 0x8E
        //   byte2: 0100110         = 0x26
        let mut buf = alloc::vec::Vec::new();
        let n = encode_uleb128(624485, &mut buf);
        assert_eq!(n, 3);
        assert_eq!(buf.as_slice(), &[0xE5, 0x8E, 0x26]);
    }

    // -- ULEB128 decode ---------------------------------------------------

    #[test]
    fn uleb128_decode_zero() {
        let (val, consumed) = decode_uleb128(&[0x00]).unwrap();
        assert_eq!(val, 0);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn uleb128_decode_127() {
        let (val, consumed) = decode_uleb128(&[0x7F]).unwrap();
        assert_eq!(val, 127);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn uleb128_decode_128() {
        let (val, consumed) = decode_uleb128(&[0x80, 0x01]).unwrap();
        assert_eq!(val, 128);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn uleb128_decode_624485() {
        let (val, consumed) = decode_uleb128(&[0xE5, 0x8E, 0x26]).unwrap();
        assert_eq!(val, 624485);
        assert_eq!(consumed, 3);
    }

    // -- ULEB128 round-trip -----------------------------------------------

    #[test]
    fn uleb128_round_trip() {
        let cases: &[u64] = &[0, 1, 127, 128, 255, 65535, u32::MAX as u64, u64::MAX];
        for &v in cases {
            let mut buf = alloc::vec::Vec::new();
            encode_uleb128(v, &mut buf);
            let (decoded, consumed) = decode_uleb128(&buf).unwrap();
            assert_eq!(decoded, v, "round-trip failed for {v}");
            assert_eq!(consumed, buf.len());
        }
    }

    // -- ULEB128 error cases ----------------------------------------------

    #[test]
    fn uleb128_decode_truncated() {
        // Continuation bit set but no more data.
        assert!(decode_uleb128(&[0x80]).is_err());
    }

    #[test]
    fn uleb128_decode_empty() {
        assert!(decode_uleb128(&[]).is_err());
    }

    // -- SLEB128 encode ---------------------------------------------------

    #[test]
    fn sleb128_encode_zero() {
        let mut buf = alloc::vec::Vec::new();
        let n = encode_sleb128(0, &mut buf);
        assert_eq!(n, 1);
        assert_eq!(buf.as_slice(), &[0x00]);
    }

    #[test]
    fn sleb128_encode_minus_one() {
        // -1 in two's complement, all bits set.
        // 7-bit group: 1111111 → 0x7F, sign bit (bit 6) = 1,
        // remaining value after >>7 = -1; since sign bit set and value == -1, done.
        let mut buf = alloc::vec::Vec::new();
        let n = encode_sleb128(-1, &mut buf);
        assert_eq!(n, 1);
        assert_eq!(buf.as_slice(), &[0x7F]);
    }

    #[test]
    fn sleb128_encode_128() {
        // 128 = 0b10000000
        // Groups of 7 (LSB first): 0000000 (bit6=0, but value remaining = 1, not 0 → more)
        //   byte0: 0000000 | 0x80 = 0x80
        // Next: value = 1, byte = 0x01, bit6=0, remaining after >>7 = 0. Done.
        //   byte1: 0x01
        let mut buf = alloc::vec::Vec::new();
        let n = encode_sleb128(128, &mut buf);
        assert_eq!(n, 2);
        assert_eq!(buf.as_slice(), &[0x80, 0x01]);
    }

    #[test]
    fn sleb128_encode_minus_128() {
        // -128 = 0xFFFFFFFFFFFFFF80
        // byte & 0x7F = 0x00, remaining = -128 >> 7 = -1. bit6 of 0x00 = 0 → more needed.
        //   byte0: 0x00 | 0x80 = 0x80
        // Next: value = -1. byte = 0x7F. bit6 = 1, remaining = -1. Done.
        //   byte1: 0x7F
        let mut buf = alloc::vec::Vec::new();
        let n = encode_sleb128(-128, &mut buf);
        assert_eq!(n, 2);
        assert_eq!(buf.as_slice(), &[0x80, 0x7F]);
    }

    // -- SLEB128 round-trip -----------------------------------------------

    #[test]
    fn sleb128_round_trip() {
        let cases: &[i64] = &[
            0,
            1,
            -1,
            63,
            -64,
            64,
            -65,
            i32::MIN as i64,
            i32::MAX as i64,
            i64::MIN,
            i64::MAX,
        ];
        for &v in cases {
            let mut buf = alloc::vec::Vec::new();
            encode_sleb128(v, &mut buf);
            let (decoded, consumed) = decode_sleb128(&buf).unwrap();
            assert_eq!(decoded, v, "SLEB128 round-trip failed for {v}");
            assert_eq!(consumed, buf.len());
        }
    }

    // -- SLEB128 error cases ----------------------------------------------

    #[test]
    fn sleb128_decode_truncated() {
        assert!(decode_sleb128(&[0x80]).is_err());
    }

    #[test]
    fn sleb128_decode_empty() {
        assert!(decode_sleb128(&[]).is_err());
    }

    // -- read_null_terminated ---------------------------------------------

    #[test]
    fn null_terminated_hello() {
        let input = b"hello\0world";
        let (payload, consumed) = read_null_terminated(input).unwrap();
        assert_eq!(payload, b"hello");
        assert_eq!(consumed, 6);
    }

    #[test]
    fn null_terminated_empty() {
        let input = b"\0";
        let (payload, consumed) = read_null_terminated(input).unwrap();
        assert_eq!(payload, b"");
        assert_eq!(consumed, 1);
    }

    #[test]
    fn null_terminated_error_no_nul() {
        let input = b"no nul";
        assert!(read_null_terminated(input).is_err());
    }

    // -- align_up ---------------------------------------------------------

    #[test]
    fn align_up_values() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(5, 1), 5);
        assert_eq!(align_up(7, 4), 8);
        assert_eq!(align_up(4, 4), 4);
    }

    // -- padding_for ------------------------------------------------------

    #[test]
    fn padding_for_values() {
        assert_eq!(padding_for(0, 8), 0);
        assert_eq!(padding_for(1, 8), 7);
        assert_eq!(padding_for(8, 8), 0);
        assert_eq!(padding_for(9, 8), 7);
        assert_eq!(padding_for(0, 1), 0);
        assert_eq!(padding_for(3, 4), 1);
    }

    // -- align_up panic on non-power-of-two -------------------------------

    #[test]
    #[should_panic(expected = "alignment must be a power of two")]
    fn align_up_panics_on_zero() {
        let _ = align_up(0, 0);
    }

    #[test]
    #[should_panic(expected = "alignment must be a power of two")]
    fn align_up_panics_on_three() {
        let _ = align_up(0, 3);
    }
}
