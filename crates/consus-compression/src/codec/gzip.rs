//! Gzip codec implementation.
//!
//! Uses the `flate2` crate with gzip framing (RFC 1952).
//!
//! ## Format Mapping
//!
//! Gzip is the Zarr gzip codec and Parquet's GZIP codec (`CompressionCodec`
//! number 2); it does not correspond to any HDF5 filter ID.
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

use flate2::read::GzEncoder;
use std::io::Read;

use super::flate::{GzipFraming, flate2_decompress};
use super::traits::{Codec, CompressionLevel};
use consus_core::{Error, ParseBudget, Result};

/// Gzip codec using RFC 1952 framing.
///
/// Wraps deflate-compressed data in a gzip container with CRC-32 integrity
/// checking. This codec is used by Zarr and other array storage formats
/// that specify gzip as a named codec.
///
/// ## Contract
///
/// - `compress` produces a valid gzip stream that `decompress` inverts exactly.
/// - `decompress` returns the concatenated data of every gzip member
///   (RFC 1952 section 2.2), validates each member's CRC-32 checksum and
///   length, and returns `Error::CompressionError` on any stream or integrity
///   failure, including bytes after the last member that are not a member.
/// - Compression level is clamped to `[0, 9]`.
#[derive(Debug, Default)]
pub struct GzipCodec;

impl Codec for GzipCodec {
    fn name(&self) -> &str {
        "gzip"
    }

    /// Returns `None`; gzip has no HDF5 filter ID.
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
        flate2_decompress::<GzipFraming>(input, expected_size, &ParseBudget::default())
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
        assert_eq!(
            decompressed, input,
            "round-trip at level 0 must be lossless"
        );
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
        assert_eq!(
            decompressed, input,
            "round-trip at level 9 must be lossless"
        );
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

    /// A declared output above the shared bound must be rejected, not
    /// allocated.
    ///
    /// Mirrors `deflate`'s `decompression_output_is_bounded`. The previous
    /// `Vec::with_capacity(expected_size)` + `read_to_end` scaffold had no
    /// ceiling, so a hostile gzip stream grew the buffer until the allocator
    /// aborted an uncatchable process.
    #[test]
    fn decompression_output_is_bounded() {
        let codec = GzipCodec;
        let compressed = codec
            .compress(&vec![7u8; 8192], CompressionLevel::default())
            .expect("compress must succeed");
        let error = codec
            .decompress(&compressed, usize::MAX)
            .expect_err("declared output above the shared bound must be rejected");
        assert!(matches!(error, Error::ResourceLimit { .. }));
    }

    /// RFC 1952 section 2.2: a gzip file is a series of members that appear
    /// one after another, so the decoded data is the concatenation of every
    /// member's data.
    ///
    /// Derivation: `gz(a) ++ gz(b)` decodes to `a ++ b`; a reader that stops
    /// after the first member returns `a`, which has fewer bytes.
    #[test]
    fn decompress_concatenated_members_returns_every_member() {
        let codec = GzipCodec;
        let first: Vec<u8> = (0u8..=255).cycle().take(300).collect();
        let second: Vec<u8> = (0u8..=255).rev().cycle().take(500).collect();
        let third = Vec::new();
        let fourth = vec![9u8; 64];

        let mut stream = Vec::new();
        let mut expected = Vec::new();
        for member in [&first, &second, &third, &fourth] {
            stream.extend(
                codec
                    .compress(member, CompressionLevel::default())
                    .expect("compress must succeed"),
            );
            expected.extend_from_slice(member);
        }

        let decoded = codec
            .decompress(&stream, expected.len())
            .expect("a multi-member stream must decode");
        assert_eq!(decoded.len(), 300 + 500 + 64);
        assert_eq!(decoded, expected);
    }

    /// RFC 1952 section 2.3.1: each member ends with CRC-32 and ISIZE of its
    /// own data, so damage in a later member is detected, not skipped.
    #[test]
    fn decompress_rejects_corrupt_later_member() {
        let codec = GzipCodec;
        let first = vec![1u8; 100];
        let second = vec![2u8; 100];
        let mut stream = codec
            .compress(&first, CompressionLevel::default())
            .expect("compress must succeed");
        let second_start = stream.len();
        stream.extend(
            codec
                .compress(&second, CompressionLevel::default())
                .expect("compress must succeed"),
        );
        // The CRC-32 field is the 8 bytes before the end, ISIZE the last 4.
        let crc_byte = stream.len() - 8;
        assert!(crc_byte > second_start);
        stream[crc_byte] ^= 0xFF;

        let error = codec
            .decompress(&stream, 200)
            .expect_err("a later member with a wrong CRC-32 must be rejected");
        assert!(matches!(error, Error::CompressionError { .. }), "{error:?}");
    }

    /// RFC 1952 section 2.2 allows no additional information after the last
    /// member, so bytes that do not start a member are an invalid stream,
    /// not silently ignored padding.
    #[test]
    fn decompress_rejects_trailing_bytes_after_last_member() {
        let codec = GzipCodec;
        let mut stream = codec
            .compress(&[5u8; 50], CompressionLevel::default())
            .expect("compress must succeed");
        stream.extend_from_slice(&[0u8; 4]);

        let error = codec
            .decompress(&stream, 50)
            .expect_err("trailing non-member bytes must be rejected");
        assert!(matches!(error, Error::CompressionError { .. }), "{error:?}");
    }

    /// For a capacity hint within the shared allocation budget, the output
    /// buffer doubles as it grows but its capacity, not only its length,
    /// stays within the budget.
    ///
    /// Derivation: 40 MiB of output under a 40,000,000-byte capacity hint
    /// fills the buffer at the hint, and uncapped doubling would then
    /// reserve 80,000,000 bytes, above the 67,108,864-byte budget.
    #[test]
    fn decompress_capacity_stays_within_the_budget() {
        let codec = GzipCodec;
        let member = codec
            .compress(&vec![0u8; 1 << 20], CompressionLevel::default())
            .expect("compress must succeed");
        let stream = member.repeat(40);

        let decoded = codec
            .decompress(&stream, 40_000_000)
            .expect("40 MiB is within the budget");
        assert_eq!(decoded.len(), 40 << 20);
        assert!(decoded.iter().all(|&byte| byte == 0));
        let budget = ParseBudget::default().max_alloc_bytes;
        assert!(
            decoded.capacity() <= budget,
            "capacity {} above budget {budget}",
            decoded.capacity()
        );
    }
}
