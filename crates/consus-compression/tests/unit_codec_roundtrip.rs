//! Round-trip tests for compression codecs.
//!
//! ## Contract
//!
//! For all codecs `C` and all byte sequences `data`:
//!
//! ```text
//! C.decompress(C.compress(data, level)?) == data
//! ```
//!
//! Tests verify:
//! - Empty input handling
//! - Single byte handling
//! - Large data blocks (64 KiB)
//! - Multiple compression levels (where applicable)
//! - Exact byte equality on round-trip

#![cfg(feature = "std")]

use consus_compression::{Codec, CodecId, CompressionLevel};

// =============================================================================
// DEFLATE (HDF5 filter ID 1)
// =============================================================================

#[cfg(feature = "deflate")]
mod deflate_tests {
    use super::*;

    fn codec() -> consus_compression::codec::deflate::DeflateCodec {
        consus_compression::codec::deflate::DeflateCodec
    }

    /// Round-trip invariant for empty input.
    ///
    /// Empty input is a valid edge case: compress → decompress must yield empty.
    #[test]
    fn empty_input() {
        let codec = codec();
        let input: Vec<u8> = Vec::new();

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress empty must succeed");

        let decompressed = codec
            .decompress(&compressed, 0)
            .expect("decompress empty must succeed");

        assert_eq!(decompressed.len(), 0, "empty round-trip must yield empty");
        assert_eq!(decompressed, input, "empty round-trip must be lossless");
    }

    /// Round-trip invariant for single byte.
    #[test]
    fn single_byte() {
        let codec = codec();
        let input: Vec<u8> = vec![0x42];

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress single byte must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress single byte must succeed");

        assert_eq!(
            decompressed, input,
            "single byte round-trip must be lossless"
        );
    }

    /// Round-trip for large data block (64 KiB).
    ///
    /// Uses gradient pattern (0, 1, 2, ..., 255, 0, 1, ...) to ensure
    /// non-trivial, compressible content.
    #[test]
    fn large_block_64k() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(65536).collect();
        assert_eq!(input.len(), 65536);

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress 64k must succeed");

        assert!(
            compressed.len() < input.len(),
            "compressed {} must be smaller than input {} for gradient pattern",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress 64k must succeed");

        assert_eq!(decompressed.len(), input.len(), "length must match");
        assert_eq!(decompressed, input, "64k round-trip must be lossless");
    }

    /// Round-trip at compression level 0 (no compression/store mode).
    #[test]
    fn level_0_store_mode() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(0))
            .expect("compress at level 0 must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress at level 0 must succeed");

        assert_eq!(decompressed, input, "level 0 round-trip must be lossless");
    }

    /// Round-trip at compression level 1 (fastest).
    #[test]
    fn level_1_fastest() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(4096).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(1))
            .expect("compress at level 1 must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress at level 1 must succeed");

        assert_eq!(decompressed, input, "level 1 round-trip must be lossless");
    }

    /// Round-trip at compression level 9 (maximum).
    #[test]
    fn level_9_maximum() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(8192).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(9))
            .expect("compress at level 9 must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress at level 9 must succeed");

        assert_eq!(decompressed, input, "level 9 round-trip must be lossless");
    }

    /// Round-trip at default compression level (6).
    #[test]
    fn default_level() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(2048).collect();

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress at default level must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress at default level must succeed");

        assert_eq!(
            decompressed, input,
            "default level round-trip must be lossless"
        );
    }

    /// All-zero input compresses well and round-trips exactly.
    #[test]
    fn all_zeroes() {
        let codec = codec();
        let input: Vec<u8> = vec![0u8; 4096];

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress zeroes must succeed");

        assert!(
            compressed.len() < input.len(),
            "zeroes must compress well: {} vs {}",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress zeroes must succeed");

        assert_eq!(decompressed, input, "zeroes round-trip must be lossless");
    }

    /// All-0xFF input compresses well and round-trips exactly.
    #[test]
    fn all_ones() {
        let codec = codec();
        let input: Vec<u8> = vec![0xFFu8; 4096];

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress ones must succeed");

        assert!(
            compressed.len() < input.len(),
            "ones must compress well: {} vs {}",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress ones must succeed");

        assert_eq!(decompressed, input, "ones round-trip must be lossless");
    }

    /// Codec metadata accessors return expected values.
    #[test]
    fn codec_metadata() {
        let codec = codec();
        assert_eq!(codec.name(), "deflate");
        assert_eq!(codec.hdf5_filter_id(), Some(1));
    }
}

// =============================================================================
// GZIP (Zarr-specific, no HDF5 filter ID)
// =============================================================================

#[cfg(feature = "gzip")]
mod gzip_tests {
    use super::*;

    fn codec() -> consus_compression::codec::gzip::GzipCodec {
        consus_compression::codec::gzip::GzipCodec
    }

    /// Round-trip invariant for empty input.
    #[test]
    fn empty_input() {
        let codec = codec();
        let input: Vec<u8> = Vec::new();

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress empty must succeed");

        let decompressed = codec
            .decompress(&compressed, 0)
            .expect("decompress empty must succeed");

        assert_eq!(decompressed.len(), 0, "empty round-trip must yield empty");
        assert_eq!(decompressed, input, "empty round-trip must be lossless");
    }

    /// Round-trip invariant for single byte.
    #[test]
    fn single_byte() {
        let codec = codec();
        let input: Vec<u8> = vec![0x7F];

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress single byte must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress single byte must succeed");

        assert_eq!(
            decompressed, input,
            "single byte round-trip must be lossless"
        );
    }

    /// Round-trip for large data block (64 KiB).
    #[test]
    fn large_block_64k() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(65536).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(6))
            .expect("compress 64k must succeed");

        assert!(
            compressed.len() < input.len(),
            "compressed {} must be smaller than input {}",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress 64k must succeed");

        assert_eq!(decompressed.len(), input.len());
        assert_eq!(decompressed, input, "64k round-trip must be lossless");
    }

    /// Round-trip at level 0 (store mode).
    #[test]
    fn level_0_store_mode() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(0))
            .expect("compress at level 0 must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress at level 0 must succeed");

        assert_eq!(decompressed, input, "level 0 round-trip must be lossless");
    }

    /// Round-trip at level 9 (maximum compression).
    #[test]
    fn level_9_maximum() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(8192).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(9))
            .expect("compress at level 9 must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress at level 9 must succeed");

        assert_eq!(decompressed, input, "level 9 round-trip must be lossless");
    }

    /// Codec metadata accessors return expected values.
    #[test]
    fn codec_metadata() {
        let codec = codec();
        assert_eq!(codec.name(), "gzip");
        assert_eq!(codec.hdf5_filter_id(), None);
    }
}

// =============================================================================
// ZSTD (HDF5 filter ID 32015)
// =============================================================================

#[cfg(feature = "zstd")]
mod zstd_tests {
    use super::*;

    fn codec() -> consus_compression::codec::zstd::ZstdCodec {
        consus_compression::codec::zstd::ZstdCodec
    }

    /// Round-trip invariant for empty input.
    #[test]
    fn empty_input() {
        let codec = codec();
        let input: Vec<u8> = Vec::new();

        let compressed = codec
            .compress(&input, CompressionLevel(3))
            .expect("compress empty must succeed");

        let decompressed = codec
            .decompress(&compressed, 0)
            .expect("decompress empty must succeed");

        assert_eq!(decompressed.len(), 0, "empty round-trip must yield empty");
        assert_eq!(decompressed, input, "empty round-trip must be lossless");
    }

    /// Round-trip invariant for single byte.
    #[test]
    fn single_byte() {
        let codec = codec();
        let input: Vec<u8> = vec![0xAB];

        let compressed = codec
            .compress(&input, CompressionLevel(3))
            .expect("compress single byte must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress single byte must succeed");

        assert_eq!(
            decompressed, input,
            "single byte round-trip must be lossless"
        );
    }

    /// Round-trip for large data block (64 KiB).
    #[test]
    fn large_block_64k() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(65536).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(3))
            .expect("compress 64k must succeed");

        assert!(
            compressed.len() < input.len(),
            "compressed {} must be smaller than input {}",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress 64k must succeed");

        assert_eq!(decompressed.len(), input.len());
        assert_eq!(decompressed, input, "64k round-trip must be lossless");
    }

    /// Round-trip at level 1 (fastest).
    #[test]
    fn level_1_fastest() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(4096).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(1))
            .expect("compress at level 1 must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress at level 1 must succeed");

        assert_eq!(decompressed, input, "level 1 round-trip must be lossless");
    }

    /// Round-trip at level 22 (maximum zstd compression).
    #[test]
    fn level_22_maximum() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(8192).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(22))
            .expect("compress at level 22 must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress at level 22 must succeed");

        assert_eq!(decompressed, input, "level 22 round-trip must be lossless");
    }

    /// Out-of-range levels are clamped without error.
    #[test]
    fn out_of_range_levels_clamped() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=127).cycle().take(1024).collect();

        for &level in &[-5, 0, 23, 100] {
            let compressed = codec
                .compress(&input, CompressionLevel(level))
                .expect("compress with clamped level must succeed");

            let decompressed = codec
                .decompress(&compressed, input.len())
                .expect("decompress must succeed");

            assert_eq!(
                decompressed, input,
                "round-trip must be lossless for clamped level {}",
                level
            );
        }
    }

    /// Codec metadata accessors return expected values.
    #[test]
    fn codec_metadata() {
        let codec = codec();
        assert_eq!(codec.name(), "zstd");
        assert_eq!(codec.hdf5_filter_id(), Some(32015));
    }
}

// =============================================================================
// LZ4 (HDF5 filter ID 32004)
// =============================================================================

#[cfg(feature = "lz4")]
mod lz4_tests {
    use super::*;

    fn codec() -> consus_compression::codec::lz4::Lz4Codec {
        consus_compression::codec::lz4::Lz4Codec
    }

    /// Round-trip invariant for empty input.
    #[test]
    fn empty_input() {
        let codec = codec();
        let input: Vec<u8> = Vec::new();

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress empty must succeed");

        let decompressed = codec
            .decompress(&compressed, 0)
            .expect("decompress empty must succeed");

        assert_eq!(decompressed.len(), 0, "empty round-trip must yield empty");
        assert_eq!(decompressed, input, "empty round-trip must be lossless");
    }

    /// Round-trip invariant for single byte.
    #[test]
    fn single_byte() {
        let codec = codec();
        let input: Vec<u8> = vec![0xCD];

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress single byte must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress single byte must succeed");

        assert_eq!(
            decompressed, input,
            "single byte round-trip must be lossless"
        );
    }

    /// Round-trip for large data block (64 KiB).
    #[test]
    fn large_block_64k() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(65536).collect();

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress 64k must succeed");

        assert!(
            compressed.len() < input.len(),
            "compressed {} must be smaller than input {} for gradient pattern",
            compressed.len(),
            input.len()
        );

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress 64k must succeed");

        assert_eq!(decompressed.len(), input.len());
        assert_eq!(decompressed, input, "64k round-trip must be lossless");
    }

    /// LZ4 compresses random-ish data (less compressible than gradient).
    #[test]
    fn pseudo_random_data() {
        let codec = codec();
        // Pattern with lower repetition: 0, 2, 4, ..., 254, 1, 3, 5, ..., 255
        let input: Vec<u8> = (0u8..128)
            .flat_map(|i| [i * 2, i * 2 + 1])
            .cycle()
            .take(4096)
            .collect();

        let compressed = codec
            .compress(&input, CompressionLevel::default())
            .expect("compress must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress must succeed");

        assert_eq!(
            decompressed, input,
            "pseudo-random round-trip must be lossless"
        );
    }

    /// Codec metadata accessors return expected values.
    #[test]
    fn codec_metadata() {
        let codec = codec();
        assert_eq!(codec.name(), "lz4");
        assert_eq!(codec.hdf5_filter_id(), Some(32004));
    }
}

// =============================================================================
// CROSS-CODEK INVARIANT TESTS
// =============================================================================

/// Verify that the round-trip invariant holds across all enabled codecs
/// with the same input data. This catches edge cases that might only
/// appear when comparing codec behaviors.
#[test]
fn all_enabled_codecs_roundtrip_same_input() {
    let input: Vec<u8> = (0u8..=255).cycle().take(8192).collect();

    #[cfg(feature = "deflate")]
    {
        use consus_compression::codec::deflate::DeflateCodec;
        let codec = DeflateCodec;
        let compressed = codec.compress(&input, CompressionLevel(6)).unwrap();
        let decompressed = codec.decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input, "deflate round-trip failed");
    }

    #[cfg(feature = "gzip")]
    {
        use consus_compression::codec::gzip::GzipCodec;
        let codec = GzipCodec;
        let compressed = codec.compress(&input, CompressionLevel(6)).unwrap();
        let decompressed = codec.decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input, "gzip round-trip failed");
    }

    #[cfg(feature = "zstd")]
    {
        use consus_compression::codec::zstd::ZstdCodec;
        let codec = ZstdCodec;
        let compressed = codec.compress(&input, CompressionLevel(3)).unwrap();
        let decompressed = codec.decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input, "zstd round-trip failed");
    }

    #[cfg(feature = "lz4")]
    {
        use consus_compression::codec::lz4::Lz4Codec;
        let codec = Lz4Codec;
        let compressed = codec.compress(&input, CompressionLevel::default()).unwrap();
        let decompressed = codec.decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input, "lz4 round-trip failed");
    }
}
