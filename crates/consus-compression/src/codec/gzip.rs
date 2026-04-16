//! Gzip codec implementation.
//!
//! Uses the `flate2` crate with gzip framing (RFC 1952).
//!
//! ## Zarr Mapping
//!
//! Gzip is a Zarr-specific codec and does not correspond to any HDF5 filter ID.
//! Compression levels 0–9 map directly to `flate2::Compression` levels.
//!
//! ## Invariant
//!
//! For all byte sequences `data` and compression levels `l` in `[0, 9]`:
//!
//! ```text
//! decompress(compress(data, l)?) == data
//! ```
//!
//! ## Compression Level Semantics
//!
//! | Level | Meaning            |
//! |-------|--------------------|
//! | 0     | No compression     |
//! | 1     | Fastest            |
//! | 6     | Default (balanced) |
//! | 9     | Maximum            |
//!
//! Out-of-range values are clamped to `[0, 9]`.

use alloc::vec::Vec;

use flate2::read::{GzDecoder, GzEncoder};
use std::io::Read;

use super::traits::{Codec, CompressionLevel};
use consus_core::{Error, Result};

/// Gzip codec using RFC 1952 framing.
///
/// Wraps deflate-compressed data in a gzip container with CRC-32 integrity
/// checking. This codec is used by Zarr and other array storage formats
/// that specify gzip as a named codec.
///
/// ## Contract
///
/// - `compress` produces a valid gzip stream that `decompress` inverts exactly.
/// - `decompress` validates the gzip CRC-32 checksum and returns
///   `Error::CompressionError` on any stream or integrity failure.
/// - Compression level is clamped to `[0, 9]`.
#[derive(Debug, Default)]
pub struct GzipCodec;

impl Codec for GzipCodec {
    fn name(&self) -> &str {
        "gzip"
    }

    /// Returns `None`; gzip is a Zarr-specific codec with no HDF5 filter ID.
    fn hdf5_filter_id(&self) -> Option<u16> {
        None
    }

    fn compress(&self, input: &[u8], level: CompressionLevel) -> Result<Vec<u8>> {
        let clamped = level.0.clamp(0, 9) as u32;
        let compression = flate2::Compression::new(clamped);
        let mut encoder = GzEncoder::new(input, compression);
        let mut output = Vec::new();
        encoder
            .read_to_end(&mut output)
            .map_err(|e| Error::CompressionError {
                message: alloc::format!("gzip compress failed: {e}"),
            })?;
        Ok(output)
    }

    fn decompress(&self, input: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        let mut decoder = GzDecoder::new(input);
        let mut output = Vec::with_capacity(expected_size);
        decoder
            .read_to_end(&mut output)
            .map_err(|e| Error::CompressionError {
                message: alloc::format!("gzip decompress failed: {e}"),
            })?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip invariant: `decompress(compress(data, 6)?) == data`.
    ///
    /// Payload: 1024 bytes of repeating 0..=255 cycle. This pattern is
    /// compressible, so the test also asserts `compressed.len() < input.len()`.
    #[test]
    fn round_trip_gzip_default_level() {
        let codec = GzipCodec;
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        assert_eq!(input.len(), 1024);

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress must succeed");

        assert!(
            compressed.len() < input.len(),
            "compressed size {} must be < input size {} for patterned data",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(
            decompressed.len(),
            input.len(),
            "decompressed length must equal original"
        );
        assert_eq!(decompressed, input, "round-trip must be lossless");
    }

    /// Round-trip at compression level 0 (store mode, no compression).
    ///
    /// The gzip container overhead means `compressed.len() > input.len()`
    /// at level 0 for small payloads, but lossless round-trip still holds.
    #[test]
    fn round_trip_gzip_level_0() {
        let codec = GzipCodec;
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(0))
            .expect("compress at level 0 must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(
            decompressed.len(),
            input.len(),
            "decompressed length must equal original"
        );
        assert_eq!(decompressed, input, "round-trip at level 0 must be lossless");
    }

    /// Round-trip at maximum compression level 9.
    ///
    /// Verifies both lossless round-trip and size reduction on compressible data.
    #[test]
    fn round_trip_gzip_level_9() {
        let codec = GzipCodec;
        let input: Vec<u8> = (0u8..=255).cycle().take(2048).collect();
        assert_eq!(input.len(), 2048);

        let compressed = codec
            .compress(&input, CompressionLevel(9))
            .expect("compress at level 9 must succeed");

        assert!(
            compressed.len() < input.len(),
            "max compression must reduce size for patterned data: {} vs {}",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(
            decompressed.len(),
            input.len(),
            "decompressed length must equal original"
        );
        assert_eq!(decompressed, input, "round-trip at level 9 must be lossless");
    }

    /// Out-of-range compression levels are clamped without error.
    ///
    /// Negative values clamp to 0; values above 9 clamp to 9.
    #[test]
    fn out_of_range_levels_clamped() {
        let codec = GzipCodec;
        let input: Vec<u8> = (0u8..=127).cycle().take(1024).collect();

        for &level in &[-5, -1, 10, 100] {
            let compressed = codec
                .compress(&input, CompressionLevel(level))
                .expect("compress with clamped level must succeed");
            let decompressed = codec
                .decompress(&compressed, input.len())
                .expect("decompress must succeed");
            assert_eq!(
                decompressed, input,
                "round-trip must be lossless for clamped level {level}"
            );
        }
    }

    /// Codec metadata accessors return expected values.
    #[test]
    fn codec_metadata() {
        let codec = GzipCodec;
        assert_eq!(codec.name(), "gzip");
        assert_eq!(codec.hdf5_filter_id(), None);
    }

    /// Round-trip with a larger payload (4096 bytes) of non-trivial structure.
    ///
    /// Data: alternating runs of repeated bytes to produce a compressible
    /// but non-trivial stream. Verifies byte-exact fidelity.
    #[test]
    fn round_trip_large_patterned_payload() {
        let codec = GzipCodec;
        let mut input = Vec::with_capacity(4096);
        for i in 0u8..=255 {
            // 16 repetitions of each byte value = 4096 bytes total
            for _ in 0..16 {
                input.push(i);
            }
        }
        assert_eq!(input.len(), 4096);

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress must succeed");
        assert!(
            compressed.len() < input.len(),
            "run-length-friendly data must compress: {} vs {}",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed.len(), 4096);
        assert_eq!(decompressed, input, "round-trip must be lossless");
    }
}
