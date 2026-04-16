//! Multi-byte integer read/write and byte-swap utilities.
//!
//! Provides endian-aware reading of variable-width integers, which is
//! required by HDF5's variable `offset_size`/`length_size` fields and by
//! other format parsers.
//!
//! ## Supported Widths
//!
//! All read/write functions accept widths in {1, 2, 4, 8} bytes.
//! Any other width panics at runtime.
//!
//! ## Invariants
//!
//! - `read_uint_le(buf, w)` and `write_uint_le(buf, w, v)` are inverses:
//!   writing `v` then reading back yields `v` for all `v < 2^(8*w)`.
//! - `read_uint_be(buf, w)` and `write_uint_be(buf, w, v)` are inverses
//!   under the same constraint.
//! - `swap_bytes(buf, w)` is an involution: applying it twice restores the
//!   original byte order.

use byteorder::{BigEndian, ByteOrder, LittleEndian};

/// Read a little-endian unsigned integer of `width` bytes from `buf`.
///
/// ## Supported widths
///
/// 1, 2, 4, 8 bytes.
///
/// ## Panics
///
/// Panics if `buf.len() < width` or `width` is not in {1, 2, 4, 8}.
#[inline]
pub fn read_uint_le(buf: &[u8], width: usize) -> u64 {
    match width {
        1 => buf[0] as u64,
        2 => LittleEndian::read_u16(buf) as u64,
        4 => LittleEndian::read_u32(buf) as u64,
        8 => LittleEndian::read_u64(buf),
        _ => panic!("unsupported integer width: {width}"),
    }
}

/// Read a big-endian unsigned integer of `width` bytes from `buf`.
///
/// ## Supported widths
///
/// 1, 2, 4, 8 bytes.
///
/// ## Panics
///
/// Panics if `buf.len() < width` or `width` is not in {1, 2, 4, 8}.
#[inline]
pub fn read_uint_be(buf: &[u8], width: usize) -> u64 {
    match width {
        1 => buf[0] as u64,
        2 => BigEndian::read_u16(buf) as u64,
        4 => BigEndian::read_u32(buf) as u64,
        8 => BigEndian::read_u64(buf),
        _ => panic!("unsupported integer width: {width}"),
    }
}

/// Write a little-endian unsigned integer of `width` bytes to `buf`.
///
/// ## Supported widths
///
/// 1, 2, 4, 8 bytes.
///
/// ## Panics
///
/// Panics if `buf.len() < width`, `width` is not in {1, 2, 4, 8},
/// or `value` exceeds the range representable in `width` bytes
/// (for width < 8).
#[inline]
pub fn write_uint_le(buf: &mut [u8], width: usize, value: u64) {
    match width {
        1 => buf[0] = value as u8,
        2 => LittleEndian::write_u16(buf, value as u16),
        4 => LittleEndian::write_u32(buf, value as u32),
        8 => LittleEndian::write_u64(buf, value),
        _ => panic!("unsupported integer width: {width}"),
    }
}

/// Write a big-endian unsigned integer of `width` bytes to `buf`.
///
/// ## Supported widths
///
/// 1, 2, 4, 8 bytes.
///
/// ## Panics
///
/// Panics if `buf.len() < width`, `width` is not in {1, 2, 4, 8},
/// or `value` exceeds the range representable in `width` bytes
/// (for width < 8).
#[inline]
pub fn write_uint_be(buf: &mut [u8], width: usize, value: u64) {
    match width {
        1 => buf[0] = value as u8,
        2 => BigEndian::write_u16(buf, value as u16),
        4 => BigEndian::write_u32(buf, value as u32),
        8 => BigEndian::write_u64(buf, value),
        _ => panic!("unsupported integer width: {width}"),
    }
}

/// Read a file offset of `size` bytes (little-endian) from a buffer.
///
/// This is the canonical replacement for `consus-hdf5::primitives::read_offset`.
/// Supports 2, 4, and 8 byte offsets as per the HDF5 specification.
///
/// ## Panics
///
/// Panics if `size` is not in {2, 4, 8}, or if `buf.len() < size`.
#[inline]
pub fn read_offset(buf: &[u8], size: usize) -> u64 {
    read_uint_le(buf, size)
}

/// Read a file length of `size` bytes (little-endian) from a buffer.
///
/// Semantically identical to [`read_offset`] but distinct for documentation
/// clarity. HDF5 files store lengths and offsets with the same encoding but
/// distinct semantic roles.
///
/// ## Panics
///
/// Panics if `size` is not in {2, 4, 8}, or if `buf.len() < size`.
#[inline]
pub fn read_length(buf: &[u8], size: usize) -> u64 {
    read_uint_le(buf, size)
}

/// Swap byte order of a value in-place within a mutable buffer.
///
/// Reverses the first `width` bytes of `buf`.
///
/// ## Proof of involution
///
/// Reversing a sequence twice yields the original sequence:
/// `reverse(reverse(s)) = s` for all finite sequences `s`.
///
/// ## Panics
///
/// Panics if `buf.len() < width`.
#[inline]
pub fn swap_bytes(buf: &mut [u8], width: usize) {
    let slice = &mut buf[..width];
    slice.reverse();
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // read_uint_le
    // -----------------------------------------------------------------------

    #[test]
    fn read_uint_le_width1() {
        assert_eq!(read_uint_le(&[0xAB], 1), 0xAB);
    }

    #[test]
    fn read_uint_le_width2_one() {
        // 0x0001 in LE = [0x01, 0x00]
        assert_eq!(read_uint_le(&[0x01, 0x00], 2), 1);
    }

    #[test]
    fn read_uint_le_width4_255() {
        // 255 in LE u32 = [0xFF, 0x00, 0x00, 0x00]
        assert_eq!(read_uint_le(&[0xFF, 0x00, 0x00, 0x00], 4), 255);
    }

    #[test]
    fn read_uint_le_width8_max() {
        let buf = [0xFF; 8];
        assert_eq!(read_uint_le(&buf, 8), u64::MAX);
    }

    // -----------------------------------------------------------------------
    // read_uint_be
    // -----------------------------------------------------------------------

    #[test]
    fn read_uint_be_width1() {
        assert_eq!(read_uint_be(&[0xAB], 1), 0xAB);
    }

    #[test]
    fn read_uint_be_width2_one() {
        // 0x0001 in BE = [0x00, 0x01]
        assert_eq!(read_uint_be(&[0x00, 0x01], 2), 1);
    }

    #[test]
    fn read_uint_be_width4_255() {
        // 255 in BE u32 = [0x00, 0x00, 0x00, 0xFF]
        assert_eq!(read_uint_be(&[0x00, 0x00, 0x00, 0xFF], 4), 255);
    }

    #[test]
    fn read_uint_be_width8_max() {
        let buf = [0xFF; 8];
        assert_eq!(read_uint_be(&buf, 8), u64::MAX);
    }

    // -----------------------------------------------------------------------
    // write/read round-trip (LE)
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_le_width1() {
        let mut buf = [0u8; 1];
        write_uint_le(&mut buf, 1, 0x42);
        assert_eq!(read_uint_le(&buf, 1), 0x42);
    }

    #[test]
    fn round_trip_le_width2() {
        let mut buf = [0u8; 2];
        write_uint_le(&mut buf, 2, 0xBEEF);
        assert_eq!(read_uint_le(&buf, 2), 0xBEEF);
    }

    #[test]
    fn round_trip_le_width4() {
        let mut buf = [0u8; 4];
        write_uint_le(&mut buf, 4, 0xDEAD_BEEF);
        assert_eq!(read_uint_le(&buf, 4), 0xDEAD_BEEF);
    }

    #[test]
    fn round_trip_le_width8() {
        let mut buf = [0u8; 8];
        write_uint_le(&mut buf, 8, 0x0123_4567_89AB_CDEF);
        assert_eq!(read_uint_le(&buf, 8), 0x0123_4567_89AB_CDEF);
    }

    // -----------------------------------------------------------------------
    // write/read round-trip (BE)
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_be_width1() {
        let mut buf = [0u8; 1];
        write_uint_be(&mut buf, 1, 0x42);
        assert_eq!(read_uint_be(&buf, 1), 0x42);
    }

    #[test]
    fn round_trip_be_width2() {
        let mut buf = [0u8; 2];
        write_uint_be(&mut buf, 2, 0xBEEF);
        assert_eq!(read_uint_be(&buf, 2), 0xBEEF);
    }

    #[test]
    fn round_trip_be_width4() {
        let mut buf = [0u8; 4];
        write_uint_be(&mut buf, 4, 0xDEAD_BEEF);
        assert_eq!(read_uint_be(&buf, 4), 0xDEAD_BEEF);
    }

    #[test]
    fn round_trip_be_width8() {
        let mut buf = [0u8; 8];
        write_uint_be(&mut buf, 8, 0x0123_4567_89AB_CDEF);
        assert_eq!(read_uint_be(&buf, 8), 0x0123_4567_89AB_CDEF);
    }

    // -----------------------------------------------------------------------
    // LE vs BE encoding byte layout
    // -----------------------------------------------------------------------

    #[test]
    fn le_be_byte_layout_u16() {
        let mut le = [0u8; 2];
        let mut be = [0u8; 2];
        write_uint_le(&mut le, 2, 0x0102);
        write_uint_be(&mut be, 2, 0x0102);
        // LE: least significant byte first
        assert_eq!(le, [0x02, 0x01]);
        // BE: most significant byte first
        assert_eq!(be, [0x01, 0x02]);
    }

    #[test]
    fn le_be_byte_layout_u32() {
        let mut le = [0u8; 4];
        let mut be = [0u8; 4];
        write_uint_le(&mut le, 4, 0x01020304);
        write_uint_be(&mut be, 4, 0x01020304);
        assert_eq!(le, [0x04, 0x03, 0x02, 0x01]);
        assert_eq!(be, [0x01, 0x02, 0x03, 0x04]);
    }

    // -----------------------------------------------------------------------
    // read_offset / read_length
    // -----------------------------------------------------------------------

    #[test]
    fn read_offset_matches_read_uint_le() {
        let buf = [0x78, 0x56, 0x34, 0x12];
        assert_eq!(read_offset(&buf, 4), read_uint_le(&buf, 4));
        assert_eq!(read_offset(&buf, 4), 0x12345678);
    }

    #[test]
    fn read_length_matches_read_uint_le() {
        let buf = [0xEF, 0xBE, 0xAD, 0xDE, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(read_length(&buf, 8), read_uint_le(&buf, 8));
        assert_eq!(read_length(&buf, 8), 0x00000000_DEADBEEF);
    }

    #[test]
    fn read_offset_width2() {
        let buf = [0x00, 0x80];
        assert_eq!(read_offset(&buf, 2), 0x8000);
    }

    // -----------------------------------------------------------------------
    // swap_bytes
    // -----------------------------------------------------------------------

    #[test]
    fn swap_bytes_width2() {
        let mut buf = [0x01, 0x02];
        swap_bytes(&mut buf, 2);
        assert_eq!(buf, [0x02, 0x01]);
    }

    #[test]
    fn swap_bytes_width4() {
        let mut buf = [0x01, 0x02, 0x03, 0x04];
        swap_bytes(&mut buf, 4);
        assert_eq!(buf, [0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn swap_bytes_width8() {
        let mut buf = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        swap_bytes(&mut buf, 8);
        assert_eq!(buf, [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn swap_bytes_involution() {
        let original = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut buf = original;
        swap_bytes(&mut buf, 4);
        // After one swap, bytes are reversed.
        assert_eq!(buf, [0xEF, 0xBE, 0xAD, 0xDE]);
        // After two swaps, original is restored (involution property).
        swap_bytes(&mut buf, 4);
        assert_eq!(buf, original);
    }

    #[test]
    fn swap_bytes_width1_noop() {
        let mut buf = [0xFF];
        swap_bytes(&mut buf, 1);
        assert_eq!(buf, [0xFF]);
    }

    // -----------------------------------------------------------------------
    // Panic tests
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "unsupported integer width: 3")]
    fn read_uint_le_unsupported_width() {
        read_uint_le(&[0; 8], 3);
    }

    #[test]
    #[should_panic(expected = "unsupported integer width: 5")]
    fn read_uint_be_unsupported_width() {
        read_uint_be(&[0; 8], 5);
    }

    #[test]
    #[should_panic(expected = "unsupported integer width: 0")]
    fn write_uint_le_unsupported_width() {
        write_uint_le(&mut [0; 8], 0, 0);
    }

    #[test]
    #[should_panic(expected = "unsupported integer width: 7")]
    fn write_uint_be_unsupported_width() {
        write_uint_be(&mut [0; 8], 7, 0);
    }

    // -----------------------------------------------------------------------
    // Endian conversion: swap_bytes converts between LE and BE
    // -----------------------------------------------------------------------

    #[test]
    fn swap_converts_le_to_be() {
        let mut buf = [0u8; 4];
        write_uint_le(&mut buf, 4, 0x01020304);
        // buf is now LE layout: [0x04, 0x03, 0x02, 0x01]
        swap_bytes(&mut buf, 4);
        // buf is now BE layout: [0x01, 0x02, 0x03, 0x04]
        assert_eq!(read_uint_be(&buf, 4), 0x01020304);
    }
}
