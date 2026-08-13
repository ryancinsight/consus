//! Zstandard codec implementation.
//!
//! ## HDF5 Mapping
//!
//! HDF5 filter ID 32015. Compression levels -8..=4 (default: 1).

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::traits::{Codec, CompressionLevel};
use consus_core::{Error, Result};

/// Zstandard codec.
#[derive(Debug, Default)]
pub struct ZstdCodec;

impl Codec for ZstdCodec {
    fn name(&self) -> &str {
        "zstd"
    }

    fn hdf5_filter_id(&self) -> Option<u16> {
        Some(32015)
    }

    #[cfg(feature = "alloc")]
    fn compress(&self, input: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        let effective_level = if level.0 == 0 { 1 } else { level.0 };
        let clamped = effective_level.clamp(-8, 4);
        zrip::compress(input, clamped).map_err(|e| Error::CompressionError {
            message: alloc::format!("zstd compress failed: {e}"),
        })
    }

    #[cfg(feature = "alloc")]
    fn decompress(&self, input: &[u8], _expected_size: usize) -> Result<Vec<u8>> {
        // zstd frames encode the decompressed size; expected_size is a hint.
        zrip::decompress_with_limit(input, 64 * 1024 * 1024).map_err(|e| Error::CompressionError {
            message: alloc::format!("zstd decompress failed: {e}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_zstd() {
        let codec = ZstdCodec;
        let input: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let compressed = codec
            .compress(&input, CompressionLevel(3))
            .expect("compress must succeed");
        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed, input);
    }

    #[test]
    fn round_trip_zstd_level_zero_maps_to_supported_encoder_level() {
        let codec = ZstdCodec;
        let input: Vec<u8> = (0u8..=127).cycle().take(2048).collect();
        let compressed = codec
            .compress(&input, CompressionLevel(0))
            .expect("compress at level 0 must succeed");
        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed, input);
    }
}
