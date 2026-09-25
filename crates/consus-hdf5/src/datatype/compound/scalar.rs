//! Scalar datatype parsers: fixed/floating point, strings, bitfields, opaque values.

use super::super::{byte_order_from_flags, map_fixed_point, map_floating_point};
use alloc::{format, string::String};
use consus_core::{Datatype, Error, Result, StringEncoding};
use core::num::NonZeroUsize;

// ---------------------------------------------------------------------------
// Class 0 — Fixed-Point (integer)
// ---------------------------------------------------------------------------

/// Parse a fixed-point (integer) datatype.
///
/// ### Class Bit Fields (byte 1 of header)
///
/// | Bit | Meaning                       |
/// |-----|-------------------------------|
/// | 0   | Byte order: 0 = LE, 1 = BE   |
/// | 3   | Signed: 0 = unsigned, 1 = yes |
///
/// ### Properties (4 bytes)
///
/// | Offset | Size | Field         |
/// |--------|------|---------------|
/// | 0      | 2    | Bit offset    |
/// | 2      | 2    | Bit precision |
pub(crate) fn parse_fixed_point(
    size: usize,
    flags: [u8; 3],
    props: &[u8],
) -> Result<(Datatype, usize)> {
    if props.len() < 4 {
        return Err(Error::InvalidFormat {
            message: String::from("fixed-point properties truncated"),
        });
    }
    let dt = map_fixed_point(size, flags[0]).ok_or_else(|| Error::InvalidFormat {
        message: String::from("fixed-point datatype size must be nonzero and fit the bit width"),
    })?;
    Ok((dt, 4))
}

// ---------------------------------------------------------------------------
// Class 1 — Floating-Point
// ---------------------------------------------------------------------------

/// Parse a floating-point datatype.
///
/// ### Class Bit Fields (byte 1 of header)
///
/// | Bit | Meaning                     |
/// |-----|-----------------------------|
/// | 0   | Byte order: 0 = LE, 1 = BE |
///
/// ### Properties (12 bytes)
///
/// | Offset | Size | Field              |
/// |--------|------|--------------------|
/// | 0      | 2    | Bit offset         |
/// | 2      | 2    | Bit precision      |
/// | 4      | 1    | Exponent location  |
/// | 5      | 1    | Exponent size      |
/// | 6      | 1    | Mantissa location  |
/// | 7      | 1    | Mantissa size      |
/// | 8      | 4    | Exponent bias      |
pub(crate) fn parse_floating_point(
    size: usize,
    flags: [u8; 3],
    props: &[u8],
) -> Result<(Datatype, usize)> {
    if props.len() < 12 {
        return Err(Error::InvalidFormat {
            message: String::from("floating-point properties truncated"),
        });
    }
    let dt = map_floating_point(size, flags[0]).ok_or_else(|| Error::InvalidFormat {
        message: String::from("floating-point datatype size must be nonzero and fit the bit width"),
    })?;
    Ok((dt, 12))
}

// ---------------------------------------------------------------------------
// Class 3 — String
// ---------------------------------------------------------------------------

/// Parse a fixed-length string datatype.
///
/// ### Class Bit Fields
///
/// | Byte | Bits | Meaning                                           |
/// |------|------|---------------------------------------------------|
/// | 0    | 0-3  | Padding: 0 = null-terminate, 1 = null-pad, 2 = space-pad |
/// | 1    | 0-3  | Character set: 0 = ASCII, 1 = UTF-8              |
///
/// ### Properties
///
/// None.  The element size from the header gives the fixed string length.
///
/// If `size == 0` this indicates a variable-length string that is only
/// meaningful inside a variable-length (class 9) wrapper; the returned
/// `FixedString` with `length = 0` should be interpreted accordingly by
/// the caller.
pub(crate) fn parse_string(size: usize, flags: [u8; 3]) -> Result<(Datatype, usize)> {
    let charset = flags[1] & 0x0F;
    let encoding = charset_to_encoding(charset)?;
    Ok((
        Datatype::FixedString {
            length: size,
            encoding,
        },
        0,
    ))
}

// ---------------------------------------------------------------------------
// Class 4 — Bitfield
// ---------------------------------------------------------------------------

/// Parse a bitfield datatype, mapped to an unsigned integer of the same size.
///
/// ### Class Bit Fields (byte 1 of header)
///
/// | Bit | Meaning                     |
/// |-----|-----------------------------|
/// | 0   | Byte order: 0 = LE, 1 = BE |
///
/// ### Properties (4 bytes)
///
/// | Offset | Size | Field         |
/// |--------|------|---------------|
/// | 0      | 2    | Bit offset    |
/// | 2      | 2    | Bit precision |
pub(crate) fn parse_bitfield(
    size: usize,
    flags: [u8; 3],
    props: &[u8],
) -> Result<(Datatype, usize)> {
    if props.len() < 4 {
        return Err(Error::InvalidFormat {
            message: String::from("bitfield properties truncated"),
        });
    }
    let byte_order = byte_order_from_flags(flags[0]);
    let bits = NonZeroUsize::new(size * 8).ok_or_else(|| Error::InvalidFormat {
        message: String::from("bitfield size must be > 0"),
    })?;
    Ok((
        Datatype::Integer {
            bits,
            byte_order,
            signed: false,
        },
        4,
    ))
}

// ---------------------------------------------------------------------------
// Class 5 — Opaque
// ---------------------------------------------------------------------------

/// Parse an opaque datatype.
///
/// ### Properties
///
/// A null-padded ASCII tag string.  The HDF5 specification states the tag
/// is NOT null-terminated, but in practice the C library writes it as a
/// null-padded string aligned to an 8-byte boundary.  This parser scans
/// for the first null byte to determine the tag extent.
///
/// When no tag is present (empty properties), the returned tag is `None`.
pub(crate) fn parse_opaque(size: usize, props: &[u8]) -> Result<(Datatype, usize)> {
    if props.is_empty() {
        return Ok((Datatype::Opaque { size, tag: None }, 0));
    }

    let null_pos = props.iter().position(|&b| b == 0);
    let (tag, consumed) = match null_pos {
        Some(0) => {
            // Empty tag; consume null + padding (up to 8-byte boundary).
            let padded = 8.min(props.len());
            (None, padded)
        }
        Some(n) => {
            let tag_str = core::str::from_utf8(&props[..n]).map_err(|_| Error::InvalidFormat {
                message: String::from("opaque tag is not valid UTF-8"),
            })?;
            // Consume tag + null, rounded up to 8-byte boundary.
            let padded = ((n + 1) + 7) & !7;
            (Some(String::from(tag_str)), padded.min(props.len()))
        }
        None => {
            // No null found; consume all available bytes as the tag.
            let tag_str = core::str::from_utf8(props).map_err(|_| Error::InvalidFormat {
                message: String::from("opaque tag is not valid UTF-8"),
            })?;
            (Some(String::from(tag_str)), props.len())
        }
    };

    Ok((Datatype::Opaque { size, tag }, consumed))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map an HDF5 character-set code to the canonical [`StringEncoding`].
///
/// | Code | Encoding |
/// |------|----------|
/// | 0    | ASCII    |
/// | 1    | UTF-8    |
pub(crate) fn charset_to_encoding(charset: u8) -> Result<StringEncoding> {
    match charset {
        0 => Ok(StringEncoding::Ascii),
        1 => Ok(StringEncoding::Utf8),
        _ => Err(Error::UnsupportedFeature {
            feature: format!("character set code {charset}"),
        }),
    }
}

/// Read an unsigned little-endian integer of 0–8 bytes.
pub(crate) fn read_uint_le(data: &[u8], size: usize) -> u64 {
    match size {
        0 => 0,
        1 => data[0] as u64,
        2 => u16::from_le_bytes([data[0], data[1]]) as u64,
        3 => u32::from_le_bytes([data[0], data[1], data[2], 0]) as u64,
        4 => u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64,
        5 => u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], 0, 0, 0]),
        6 => u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], data[5], 0, 0]),
        7 => u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], 0,
        ]),
        _ => u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
    }
}

/// Read an unsigned big-endian integer of 0–8 bytes.
pub(crate) fn read_uint_be(data: &[u8], size: usize) -> u64 {
    match size {
        0 => 0,
        1 => data[0] as u64,
        2 => u16::from_be_bytes([data[0], data[1]]) as u64,
        3 => u32::from_be_bytes([0, data[0], data[1], data[2]]) as u64,
        4 => u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as u64,
        5 => u64::from_be_bytes([0, 0, 0, data[0], data[1], data[2], data[3], data[4]]),
        6 => u64::from_be_bytes([0, 0, data[0], data[1], data[2], data[3], data[4], data[5]]),
        7 => u64::from_be_bytes([
            0, data[0], data[1], data[2], data[3], data[4], data[5], data[6],
        ]),
        _ => u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
    }
}

/// Sign-extend an unsigned value of `size` bytes to `i64`.
///
/// If the most-significant bit of the `size`-byte value is set, the upper
/// bits of the returned `i64` are filled with ones (arithmetic shift).
pub(crate) fn sign_extend(val: u64, size: usize) -> i64 {
    let bits = size * 8;
    if bits == 0 || bits >= 64 {
        return val as i64;
    }
    let shift = 64 - bits;
    ((val as i64) << shift) >> shift
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
