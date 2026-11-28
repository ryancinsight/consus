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

use consus_compression::{Codec, CompressionLevel};

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
// GZIP (Zarr and Parquet, no HDF5 filter ID)
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

    /// Round-trip at level 4 (maximum compression level supported by zrip).
    #[test]
    fn level_4_maximum() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=255).cycle().take(8192).collect();

        let compressed = codec
            .compress(&input, CompressionLevel(4))
            .expect("compress at level 4 must succeed");

        let decompressed = codec
            .decompress(&compressed, input.len())
            .expect("decompress at level 4 must succeed");

        assert_eq!(decompressed, input, "level 4 round-trip must be lossless");
    }

    /// Out-of-range levels are clamped without error.
    #[test]
    fn out_of_range_levels_clamped() {
        let codec = codec();
        let input: Vec<u8> = (0u8..=127).cycle().take(1024).collect();

        for &level in &[-20, 0, 23, 100] {
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

// =============================================================================
// Declared size above the default budget
// =============================================================================

/// Every zlib-, gzip- and raw-DEFLATE-framed codec decodes a stream whose
/// declared size exceeds the default 64 MiB budget.
///
/// Derivation: the ceiling is the size the compressed bytes can produce, not
/// the budget, so a declaration of `budget + 1` admits `budget + 1` bytes. The
/// input is all zeros so level-1 output stays small and every codec must
/// return the same `budget + 1` zero bytes.
#[cfg(all(feature = "deflate", feature = "gzip"))]
mod above_default_budget {
    use super::*;
    use consus_compression::codec::deflate::DeflateCodec;
    use consus_compression::codec::flate::MAX_DEFLATE_EXPANSION;
    use consus_compression::codec::gzip::GzipCodec;
    use consus_compression::codec::raw_deflate::RawDeflateCodec;
    use consus_compression::codec::zlib::ZlibCodec;
    use consus_core::ParseBudget;

    fn assert_decodes_declared_size<C: Codec>(codec: &C, input: &[u8]) {
        let compressed = codec
            .compress(input, CompressionLevel(1))
            .expect("level-1 compress must succeed");
        let decoded = codec
            .decompress(&compressed, input.len())
            .unwrap_or_else(|error| {
                panic!(
                    "{}: a declared size above the default budget must decode: {error:?}",
                    codec.name()
                )
            });
        assert_eq!(decoded.len(), input.len(), "{}: length", codec.name());
        assert!(decoded == input, "{}: content", codec.name());
    }

    #[test]
    fn every_flate_codec_decodes_above_default_budget() {
        let input = vec![0u8; ParseBudget::default().max_alloc_bytes + 1];

        assert_decodes_declared_size(&DeflateCodec, &input);
        assert_decodes_declared_size(&ZlibCodec, &input);
        assert_decodes_declared_size(&GzipCodec, &input);
        assert_decodes_declared_size(&RawDeflateCodec, &input);
    }

    /// Bits of a DEFLATE stream, least significant bit of each byte first.
    struct BitWriter {
        bytes: Vec<u8>,
        used: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                used: 0,
            }
        }

        fn bit(&mut self, bit: u32) {
            if self.used == 0 {
                self.bytes.push(0);
            }
            let value = u8::try_from(bit & 1).expect("invariant: one bit fits a byte");
            *self.bytes.last_mut().expect("invariant: a byte was pushed") |= value << self.used;
            self.used = (self.used + 1) % 8;
        }

        /// A header field or extra bits: least significant bit first.
        fn field(&mut self, value: u32, width: u32) {
            for index in 0..width {
                self.bit(value >> index);
            }
        }

        /// A Huffman code: most significant bit first.
        fn code(&mut self, value: u32, width: u32) {
            for index in (0..width).rev() {
                self.bit(value >> index);
            }
        }
    }

    /// One dynamic block whose output is `1 + 258 * matches` zero bytes.
    ///
    /// RFC 1951 section 3.2.7. Code-length code: symbol 18 has length 1
    /// (code `0`), symbols 1 and 2 length 2 (codes `10`, `11`). Literal/length
    /// code: symbol 285 (length 258, no extra bits) has length 1 (code `0`),
    /// symbols 0 and 256 length 2 (codes `10`, `11`). One distance code of
    /// length 1 (code `0`, distance 1). Data: literal 0, then `matches`
    /// pairs of one length bit and one distance bit, then end-of-block: the
    /// densest stream DEFLATE can carry, two bits per 258 bytes.
    fn densest_deflate_stream(matches: usize) -> Vec<u8> {
        let mut bits = BitWriter::new();
        bits.field(1, 1); // BFINAL
        bits.field(2, 2); // BTYPE: dynamic Huffman
        bits.field(29, 5); // HLIT: 286 literal/length codes
        bits.field(0, 5); // HDIST: 1 distance code
        bits.field(14, 4); // HCLEN: 18 code length code lengths
        // Order 16 17 18 0 8 7 9 6 10 5 11 4 12 3 13 2 14 1.
        for length in [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2] {
            bits.field(length, 3);
        }
        let zeros = |bits: &mut BitWriter, run: u32| {
            bits.code(0, 1); // symbol 18
            bits.field(run - 11, 7);
        };
        bits.code(0b11, 2); // literal 0: length 2 (symbol 2)
        zeros(&mut bits, 138); // literals 1..=138
        zeros(&mut bits, 117); // literals 139..=255
        bits.code(0b11, 2); // end-of-block 256: length 2
        zeros(&mut bits, 28); // 257..=284
        bits.code(0b10, 2); // symbol 285: length 1 (symbol 1)
        bits.code(0b10, 2); // distance code 0: length 1
        bits.code(0b10, 2); // literal 0
        for _ in 0..matches {
            bits.code(0, 1); // length 258
            bits.code(0, 1); // distance 1
        }
        bits.code(0b11, 2); // end-of-block
        bits.bytes
    }

    fn zlib_framed(raw: &[u8], zeros: usize) -> Vec<u8> {
        // Adler-32 of `zeros` zero bytes: a stays 1, b counts the bytes.
        let b = u32::try_from(zeros % 65_521).expect("invariant: below 65521");
        let mut stream = vec![0x78, 0x9c];
        stream.extend_from_slice(raw);
        stream.extend_from_slice(&((b << 16) | 1).to_be_bytes());
        stream
    }

    fn gzip_framed(raw: &[u8], zeros: usize) -> Vec<u8> {
        let mut crc = flate2::Crc::new();
        let block = vec![0u8; 1 << 16];
        let mut left = zeros;
        while left > 0 {
            let take = left.min(block.len());
            crc.update(&block[..take]);
            left -= take;
        }
        let size = u32::try_from(zeros % (1 << 32)).expect("invariant: reduced modulo 2^32");
        let mut stream = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
        stream.extend_from_slice(raw);
        stream.extend_from_slice(&crc.sum().to_le_bytes());
        stream.extend_from_slice(&size.to_le_bytes());
        stream
    }

    /// The densest honest stream decodes in full under a declaration of
    /// zero, past the 64 MiB budget, whichever framing carries it.
    ///
    /// Derivation: `matches = 260_200` gives `1 + 258 * 260_200 = 67_131_601`
    /// bytes, above `64 MiB = 67_108_864`, and the stream stays within
    /// `MAX_DEFLATE_EXPANSION` times its length (asserted).
    #[test]
    fn densest_stream_decodes_in_full_under_a_zero_declaration() {
        let matches = 260_200;
        let zeros = 1 + 258 * matches;
        assert!(zeros > ParseBudget::default().max_alloc_bytes);
        let raw = densest_deflate_stream(matches);
        assert!(zeros <= raw.len() * MAX_DEFLATE_EXPANSION);
        let streams: [(&str, Vec<u8>, Vec<&dyn Codec>); 3] = [
            ("raw", raw.clone(), vec![&RawDeflateCodec]),
            (
                "zlib",
                zlib_framed(&raw, zeros),
                vec![&DeflateCodec, &ZlibCodec],
            ),
            (
                "gzip",
                gzip_framed(&raw, zeros),
                vec![&GzipCodec, &ZlibCodec],
            ),
        ];
        for (framing, stream, codecs) in &streams {
            for codec in codecs {
                for declared in [0, zeros] {
                    let decoded = codec.decompress(stream, declared).unwrap_or_else(|error| {
                        panic!(
                            "{framing} via {}: declared {declared}: {error:?}",
                            codec.name()
                        )
                    });
                    assert_eq!(decoded.len(), zeros, "{framing} via {}", codec.name());
                    assert!(decoded.iter().all(|byte| *byte == 0), "{framing}: content");
                }
            }
        }
    }
}
