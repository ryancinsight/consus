//! Deflate/zlib codec implementation.
//!
//! Uses the `flate2` crate with the pure-Rust `miniz_oxide` backend.
//!
//! ## HDF5 Mapping
//!
//! HDF5 filter ID 1 (deflate). Compression levels 0-9.

use alloc::vec::Vec;

use flate2::read::{DeflateDecoder, DeflateEncoder};
use std::io::Read;

use super::traits::{Codec, CompressionLevel};
use consus_core::{Error, Result};

/// Deflate codec.
#[derive(Debug, Default)]
pub struct DeflateCodec;

impl Codec for DeflateCodec {
    fn name(&self) -> &str {
        "deflate"
    }

    fn hdf5_filter_id(&self) -> Option<u16> {
        Some(1) // HDF5 deflate filter ID
    }

    fn compress(&self, input: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        let clamped = level.0.clamp(0, 9) as u32;
        let compression = flate2::Compression::new(clamped);
        let mut encoder = DeflateEncoder::new(input, compression);
        let mut output = Vec::new();
        encoder
            .read_to_end(&mut output)
            .map_err(|e| Error::CompressionError {
                message: alloc::format!("deflate compress failed: {e}"),
            })?;
        Ok(output)
    }

    fn decompress(&self, input: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        let mut decoder = DeflateDecoder::new(input);
        let mut output = Vec::with_capacity(expected_size);
        decoder
            .read_to_end(&mut output)
            .map_err(|e| Error::CompressionError {
                message: alloc::format!("deflate decompress failed: {e}"),
            })?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip invariant: decompress(compress(data)) == data.
    ///
    /// Tests with a non-trivial payload to verify real compression occurs.
    #[test]
    fn round_trip_deflate() {
        let codec = DeflateCodec;
        // Non-trivial data: 1024 bytes of repeating pattern
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress must succeed");
        // Compressed output should be smaller for repetitive data
        assert!(
            compressed.len() < input.len(),
            "compressed size {} must be < input size {}",
            compressed.len(),
            input.len()
        );
        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed, input, "round-trip must be lossless");
    }
}
