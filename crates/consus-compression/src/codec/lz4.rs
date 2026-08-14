//! LZ4 codec implementation.
//!
//! Uses the `lz4_flex` pure-Rust crate.
//!
//! ## HDF5 Mapping
//!
//! HDF5 filter ID 32004.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::traits::{Codec, CompressionLevel};
use consus_core::{Error, ParseBudget, Result};

/// LZ4 block codec.
#[derive(Debug, Default)]
pub struct Lz4Codec;

impl Codec for Lz4Codec {
    fn name(&self) -> &str {
        "lz4"
    }

    fn hdf5_filter_id(&self) -> Option<u16> {
        Some(32004)
    }

    #[cfg(feature = "alloc")]
    fn compress(&self, input: &[u8], _level: CompressionLevel) -> Result<Vec<u8>> {
        // LZ4 block format does not have compression levels.
        Ok(lz4_flex::compress_prepend_size(input))
    }

    #[cfg(feature = "alloc")]
    fn decompress(&self, input: &[u8], _expected_size: usize) -> Result<Vec<u8>> {
        let (decoded_size, payload) =
            lz4_flex::block::uncompressed_size(input).map_err(|e| Error::CompressionError {
                message: alloc::format!("lz4 decompress failed: {e}"),
            })?;
        let mut output = ParseBudget::default().zeroed(
            u64::try_from(decoded_size).map_err(|_| Error::Overflow)?,
            "decompressed output",
        )?;
        let written = lz4_flex::block::decompress_into(payload, &mut output).map_err(|e| {
            Error::CompressionError {
                message: alloc::format!("lz4 decompress failed: {e}"),
            }
        })?;
        output.truncate(written);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_lz4() {
        let codec = Lz4Codec;
        let input: Vec<u8> = (0u8..=255).cycle().take(2048).collect();
        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");
        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed, input);
    }

    #[test]
    fn decompression_output_is_bounded() {
        let codec = Lz4Codec;
        let mut compressed = u32::MAX.to_le_bytes().to_vec();
        compressed.push(0);
        let error = codec
            .decompress(&compressed, 0)
            .expect_err("oversized prepended output must be rejected");
        assert!(matches!(error, Error::ResourceLimit { .. }));
    }
}
