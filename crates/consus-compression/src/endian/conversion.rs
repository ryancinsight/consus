//! Multi-byte integer read/write and byte-swap utilities (re-export shim).
//!
//! The implementations live in [`consus_core::decode`], the single home for
//! fixed-width byte-order decoding. This module re-exports them so the
//! historical `consus_compression::endian::conversion` path keeps working
//! without a second copy.
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

pub use consus_core::decode::{
    read_length, read_offset, read_uint_be, read_uint_le, swap_bytes, write_uint_be, write_uint_le,
};

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
