//! RLE_DICTIONARY encoding decoder for Parquet column data.
//!
//! ## Specification
//!
//! Parquet encoding ID 8 (RLE_DICTIONARY, also called PLAIN_DICTIONARY for backward
//! compatibility) stores dictionary indices as a one-byte bit_width header followed
//! by an RLE/bit-packing hybrid byte stream of exactly `count` index values.
//!
//! Reference: <https://github.com/apache/parquet-format/blob/master/Encodings.md#dictionary-encoding-plain_dictionary--2-and-rle_dictionary--8>
//!
//! ## Wire format
//!
//! ```text
//! byte[0]      : bit_width  (u8; width of each index in bits)
//! byte[1..]    : RLE/bit-packing hybrid stream (see encoding::levels::decode_levels)
//! ```
//!
//! ## Invariants
//!
//! - `bytes.len() >= 1` (bit_width byte required even for count=0)
//! - Returned indices are non-negative integers in `[0, 2^bit_width - 1]`
//! - `count == 0` consumes exactly the bit_width byte and returns an empty Vec

use alloc::vec::Vec;
use consus_core::{Error, Result};

use super::levels::decode_levels;

/// Decode `count` dictionary indices from RLE_DICTIONARY-encoded `bytes`.
///
/// The first byte of `bytes` is the bit_width for the RLE/bit-packing hybrid.
/// The remaining bytes are the hybrid-encoded index stream.
///
/// ## Errors
///
/// - `Error::BufferTooSmall` if `bytes` is empty.
/// - Any error propagated from `decode_levels`.
pub fn decode_rle_dict_indices(bytes: &[u8], count: usize) -> Result<Vec<i32>> {
    if bytes.is_empty() {
        return Err(Error::BufferTooSmall {
            required: 1,
            provided: 0,
        });
    }
    let bit_width = bytes[0];
    if count == 0 {
        return Ok(Vec::new());
    }
    decode_levels(&bytes[1..], bit_width, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytically derived: bit_width=2, 4 values [1, 2, 0, 3]
    ///
    /// Encode [1,2,0,3] as one bit-packed group (num_groups=1):
    ///   header = ((1-1) << 1) | 1 = 0x01
    ///   group byte 0 (bit_width=2):
    ///     v[0]=1 bits[0,1]=01, v[1]=2 bits[2,3]=10,
    ///     v[2]=0 bits[4,5]=00, v[3]=3 bits[6,7]=11 => 0b11_00_10_01 = 0xC9
    ///   group byte 1: remaining 4 values are 0 => 0x00
    /// Full bytes: [bit_width=2, header=0x01, 0xC9, 0x00]
    #[test]
    fn decode_rle_dict_indices_bit_packed_four_values() {
        let bytes = [0x02u8, 0x01, 0xC9, 0x00];
        let indices = decode_rle_dict_indices(&bytes, 4).unwrap();
        assert_eq!(indices, vec![1, 2, 0, 3]);
    }

    /// RLE run: bit_width=3, 5 copies of index 7
    ///
    /// header = (5 << 1) | 0 = 0x0A
    /// value bytes: ceil(3/8) = 1 byte, value = 7 = 0x07
    /// Full bytes: [bit_width=3, 0x0A, 0x07]
    #[test]
    fn decode_rle_dict_indices_rle_run() {
        let bytes = [0x03u8, 0x0A, 0x07];
        let indices = decode_rle_dict_indices(&bytes, 5).unwrap();
        assert_eq!(indices, vec![7, 7, 7, 7, 7]);
    }

    #[test]
    fn decode_rle_dict_indices_zero_count_returns_empty() {
        // Only the bit_width byte is required; no data bytes needed.
        let bytes = [0x02u8];
        let indices = decode_rle_dict_indices(&bytes, 0).unwrap();
        assert_eq!(indices, vec![]);
    }

    #[test]
    fn decode_rle_dict_indices_empty_input_errors() {
        let err = decode_rle_dict_indices(&[], 1).unwrap_err();
        assert!(matches!(
            err,
            consus_core::Error::BufferTooSmall {
                required: 1,
                provided: 0
            }
        ));
    }

    /// Two sequential RLE runs: [0,0, 1,1,1] with bit_width=1
    ///
    /// Run1: (2<<1)|0 = 0x04, value=0x00
    /// Run2: (3<<1)|0 = 0x06, value=0x01
    /// Full bytes: [bit_width=1, 0x04, 0x00, 0x06, 0x01]
    #[test]
    fn decode_rle_dict_indices_two_rle_runs() {
        let bytes = [0x01u8, 0x04, 0x00, 0x06, 0x01];
        let indices = decode_rle_dict_indices(&bytes, 5).unwrap();
        assert_eq!(indices, vec![0, 0, 1, 1, 1]);
    }
}
