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
use consus_core::{Error, Result};

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
        lz4_flex::decompress_size_prepended(input).map_err(|e| Error::CompressionError {
            message: alloc::format!("lz4 decompress failed: {e}"),
        })
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
}
