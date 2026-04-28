//! Proptest roundtrip suite for `encoding::compression`.
//!
//! ## Mathematical specification
//!
//! For each enabled codec c and any byte sequence d:
//!   decompress(compress(d, c), c, |d|) == d
//!
//! This verifies the compress↔decompress identity for all feature-enabled codecs
//! over the full byte value domain [0x00, 0xFF] and lengths [0, 1023].

#[cfg(any(
    feature = "gzip",
    feature = "snappy",
    feature = "zstd",
    feature = "lz4"
))]
use super::compression::{CompressionCodec, compress_page_values, decompress_page_values};
#[cfg(any(
    feature = "gzip",
    feature = "snappy",
    feature = "zstd",
    feature = "lz4"
))]
use proptest::prelude::*;

/// Shared strategy: arbitrary byte vectors up to 1 KiB.
#[cfg(any(
    feature = "gzip",
    feature = "snappy",
    feature = "zstd",
    feature = "lz4"
))]
fn byte_data() -> impl Strategy<Value = alloc::vec::Vec<u8>> {
    proptest::collection::vec(0u8..=255, 0..1024)
}

#[cfg(feature = "gzip")]
proptest! {
    /// ∀ data ∈ Vec<u8>: decompress_gzip(compress_gzip(data), |data|) == data
    #[test]
    fn prop_gzip_compress_decompress_identity(data in byte_data()) {
        let compressed = compress_page_values(&data, CompressionCodec::Gzip).unwrap();
        let decompressed =
            decompress_page_values(&compressed, CompressionCodec::Gzip, data.len()).unwrap();
        prop_assert_eq!(decompressed, data);
    }

    /// Same invariant for ZLIB (shares deflate backend with GZIP).
    #[test]
    fn prop_zlib_compress_decompress_identity(data in byte_data()) {
        let compressed = compress_page_values(&data, CompressionCodec::Zlib).unwrap();
        let decompressed =
            decompress_page_values(&compressed, CompressionCodec::Zlib, data.len()).unwrap();
        prop_assert_eq!(decompressed, data);
    }
}

#[cfg(feature = "snappy")]
proptest! {
    /// ∀ data ∈ Vec<u8>: decompress_snappy(compress_snappy(data), |data|) == data
    #[test]
    fn prop_snappy_compress_decompress_identity(data in byte_data()) {
        let compressed = compress_page_values(&data, CompressionCodec::Snappy).unwrap();
        let decompressed =
            decompress_page_values(&compressed, CompressionCodec::Snappy, data.len()).unwrap();
        prop_assert_eq!(decompressed, data);
    }
}

#[cfg(feature = "zstd")]
proptest! {
    /// ∀ data ∈ Vec<u8>: decompress_zstd(compress_zstd(data), |data|) == data
    #[test]
    fn prop_zstd_compress_decompress_identity(data in byte_data()) {
        let compressed = compress_page_values(&data, CompressionCodec::Zstd).unwrap();
        let decompressed =
            decompress_page_values(&compressed, CompressionCodec::Zstd, data.len()).unwrap();
        prop_assert_eq!(decompressed, data);
    }
}

#[cfg(feature = "lz4")]
proptest! {
    /// ∀ data ∈ Vec<u8>: decompress_lz4_raw(compress_lz4_raw(data), |data|) == data
    #[test]
    fn prop_lz4_raw_compress_decompress_identity(data in byte_data()) {
        let compressed = compress_page_values(&data, CompressionCodec::Lz4Raw).unwrap();
        let decompressed =
            decompress_page_values(&compressed, CompressionCodec::Lz4Raw, data.len()).unwrap();
        prop_assert_eq!(decompressed, data);
    }

    /// ∀ data ∈ Vec<u8>: decompress_lz4(compress_lz4(data), |data|) == data
    #[test]
    fn prop_lz4_compress_decompress_identity(data in byte_data()) {
        let compressed = compress_page_values(&data, CompressionCodec::Lz4).unwrap();
        let decompressed =
            decompress_page_values(&compressed, CompressionCodec::Lz4, data.len()).unwrap();
        prop_assert_eq!(decompressed, data);
    }
}
