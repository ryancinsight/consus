//! Parquet compression codec dispatch for page value decompression.
//!
//! Maps Parquet thrift `CompressionCodec` discriminants to concrete
//! decompression routines. Feature-gated: codecs whose feature is not
//! enabled return `Error::UnsupportedFeature` with an actionable message.
//!
//! ## Codec mapping
//!
//! | Discriminant | Parquet name | Rust feature | Backend |
//! |--------------|-------------|-------------|---------|
//! | 0 | UNCOMPRESSED | (always) | pass-through |
//! | 1 | SNAPPY | `snappy` | `snap` crate |
//! | 2 | GZIP | `gzip` | `flate2::read::DeflateDecoder` (raw deflate) |
//! | 3 | LZ4_RAW | `lz4` | `lz4_flex::decompress` (raw block) |
//! | 4 | ZSTD | `zstd` | `zrip::decompress_with_limit` |
//! | 5 | LZ4 | `lz4` | `lz4_flex::decompress_size_prepended` |
//! | 6 | BROTLI | (none) | always `UnsupportedFeature` |
//! | 7 | ZLIB | `gzip` | `flate2::read::DeflateDecoder` (raw deflate) |

use alloc::{format, vec::Vec};

use consus_core::{Error, ParseBudget, Result};

/// Parquet compression codec, discriminant matches thrift `CompressionCodec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCodec {
    Uncompressed = 0,
    Snappy = 1,
    Gzip = 2,
    Lz4Raw = 3,
    Zstd = 4,
    Lz4 = 5,
    Brotli = 6,
    Zlib = 7,
}

impl CompressionCodec {
    /// Convert an i32 discriminant to a `CompressionCodec`.
    ///
    /// Returns `None` for values outside the range 0..=7.
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Uncompressed),
            1 => Some(Self::Snappy),
            2 => Some(Self::Gzip),
            3 => Some(Self::Lz4Raw),
            4 => Some(Self::Zstd),
            5 => Some(Self::Lz4),
            6 => Some(Self::Brotli),
            7 => Some(Self::Zlib),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Uncompressed => "UNCOMPRESSED",
            Self::Snappy => "SNAPPY",
            Self::Gzip => "GZIP",
            Self::Lz4Raw => "LZ4_RAW",
            Self::Zstd => "ZSTD",
            Self::Lz4 => "LZ4",
            Self::Brotli => "BROTLI",
            Self::Zlib => "ZLIB",
        }
    }

    fn unsupported(self, feature_flag: Option<&str>) -> Error {
        let feature = match feature_flag {
            Some(feature_flag) => format!(
                "parquet compression codec {} ({}) — enable feature '{feature_flag}'",
                self as i32,
                self.label()
            ),
            None => format!(
                "parquet compression codec {} ({})",
                self as i32,
                self.label()
            ),
        };
        Error::UnsupportedFeature { feature }
    }
}

fn compression_error<E: core::fmt::Display>(codec: &str, action: &str, error: E) -> Error {
    Error::CompressionError {
        message: format!("{codec} {action} failed: {error}"),
    }
}

fn allocate_output_buffer(size: usize) -> Result<Vec<u8>> {
    let size = checked_output_size(size)?;
    ParseBudget::default().zeroed(
        u64::try_from(size).map_err(|_| Error::Overflow)?,
        "decompressed output",
    )
}

fn decode_into_buffer<F>(output_size: usize, decode: F) -> Result<Vec<u8>>
where
    F: FnOnce(&mut [u8]) -> Result<usize>,
{
    let mut output = allocate_output_buffer(output_size)?;
    let written = decode(&mut output)?;
    output.truncate(written);
    Ok(output)
}

/// Compress page value bytes using the given Parquet compression codec.
///
/// UNCOMPRESSED: returns `data.to_vec()` (pass-through, no error).
/// GZIP / ZLIB: raw deflate via `flate2::read::DeflateEncoder`.
///              Level: `flate2::Compression::default()`.
/// SNAPPY: `snap::raw::Encoder::new().compress_vec(data)`.
/// ZSTD: `zrip::compress(data, 3)`.
/// LZ4_RAW: `lz4_flex::compress(data)`.
/// LZ4: `lz4_flex::compress_prepend_size(data)`.
/// BROTLI: always returns `Error::UnsupportedFeature { feature: "parquet compression codec BROTLI (6)" }`.
///
/// Feature gating matches `decompress_page_values`: same `#[cfg(feature = "gzip")]` etc.
/// If the feature is not enabled, returns `Error::UnsupportedFeature` with an actionable message.
pub(crate) fn compress_page_values(data: &[u8], codec: CompressionCodec) -> Result<Vec<u8>> {
    match codec {
        CompressionCodec::Uncompressed => Ok(data.to_vec()),

        CompressionCodec::Gzip | CompressionCodec::Zlib => {
            #[cfg(feature = "gzip")]
            {
                use flate2::read::DeflateEncoder;
                use std::io::Read;
                let mut encoder = DeflateEncoder::new(data, flate2::Compression::default());
                let mut output = Vec::with_capacity(data.len());
                encoder
                    .read_to_end(&mut output)
                    .map_err(|e| compression_error("deflate", "compress", e))?;
                Ok(output)
            }
            #[cfg(not(feature = "gzip"))]
            {
                let _ = data;
                Err(codec.unsupported(Some("gzip")))
            }
        }

        CompressionCodec::Snappy => {
            #[cfg(feature = "snappy")]
            {
                snap::raw::Encoder::new()
                    .compress_vec(data)
                    .map_err(|e| compression_error("snappy", "compress", e))
            }
            #[cfg(not(feature = "snappy"))]
            {
                let _ = data;
                Err(codec.unsupported(Some("snappy")))
            }
        }

        CompressionCodec::Zstd => {
            #[cfg(feature = "zstd")]
            {
                zrip::compress(data, 3).map_err(|e| compression_error("zstd", "compress", e))
            }
            #[cfg(not(feature = "zstd"))]
            {
                let _ = data;
                Err(codec.unsupported(Some("zstd")))
            }
        }

        CompressionCodec::Lz4Raw => {
            #[cfg(feature = "lz4")]
            {
                Ok(lz4_flex::compress(data))
            }
            #[cfg(not(feature = "lz4"))]
            {
                let _ = data;
                Err(codec.unsupported(Some("lz4")))
            }
        }

        CompressionCodec::Lz4 => {
            #[cfg(feature = "lz4")]
            {
                Ok(lz4_flex::compress_prepend_size(data))
            }
            #[cfg(not(feature = "lz4"))]
            {
                let _ = data;
                Err(codec.unsupported(Some("lz4")))
            }
        }

        CompressionCodec::Brotli => {
            let _ = data;
            Err(codec.unsupported(None))
        }
    }
}

/// Decompress page value bytes according to the Parquet compression codec.
///
/// `uncompressed_size` is the expected decompressed size, used as a capacity
/// hint for allocators and as a required parameter for some codecs (zstd, lz4).
///
/// # Errors
///
/// - `Error::UnsupportedFeature` — codec is not implemented (Brotli) or the
///   corresponding feature flag is not enabled.
/// - `Error::CompressionError` — the compressed data is malformed or
///   decompression fails.
pub fn decompress_page_values(
    data: &[u8],
    codec: CompressionCodec,
    uncompressed_size: usize,
) -> Result<Vec<u8>> {
    match codec {
        CompressionCodec::Uncompressed => copy_bounded(data),

        CompressionCodec::Gzip | CompressionCodec::Zlib => {
            #[cfg(feature = "gzip")]
            {
                decompress_deflate(data, uncompressed_size)
            }
            #[cfg(not(feature = "gzip"))]
            {
                let _ = (data, uncompressed_size);
                Err(codec.unsupported(Some("gzip")))
            }
        }

        CompressionCodec::Snappy => {
            #[cfg(feature = "snappy")]
            {
                let decoded_size = snap::raw::decompress_len(data)
                    .map_err(|e| compression_error("snappy header", "decode", e))?;
                decode_into_buffer(decoded_size, |output| {
                    snap::raw::Decoder::new()
                        .decompress(data, output)
                        .map_err(|e| compression_error("snappy", "decompress", e))
                })
            }
            #[cfg(not(feature = "snappy"))]
            {
                let _ = (data, uncompressed_size);
                Err(codec.unsupported(Some("snappy")))
            }
        }

        CompressionCodec::Zstd => {
            #[cfg(feature = "zstd")]
            {
                let limit = checked_output_size(uncompressed_size)?;
                zrip::decompress_with_limit(data, limit)
                    .map_err(|e| compression_error("zstd", "decompress", e))
            }
            #[cfg(not(feature = "zstd"))]
            {
                let _ = (data, uncompressed_size);
                Err(codec.unsupported(Some("zstd")))
            }
        }

        CompressionCodec::Lz4Raw => {
            #[cfg(feature = "lz4")]
            {
                let output_size = checked_output_size(uncompressed_size)?;
                decode_into_buffer(output_size, |output| {
                    lz4_flex::block::decompress_into(data, output)
                        .map_err(|e| compression_error("lz4_raw", "decompress", e))
                })
            }
            #[cfg(not(feature = "lz4"))]
            {
                let _ = (data, uncompressed_size);
                Err(codec.unsupported(Some("lz4")))
            }
        }

        CompressionCodec::Lz4 => {
            #[cfg(feature = "lz4")]
            {
                let (decoded_size, payload) = lz4_flex::block::uncompressed_size(data)
                    .map_err(|e| compression_error("lz4 header", "decode", e))?;
                decode_into_buffer(decoded_size, |output| {
                    lz4_flex::block::decompress_into(payload, output)
                        .map_err(|e| compression_error("lz4", "decompress", e))
                })
            }
            #[cfg(not(feature = "lz4"))]
            {
                let _ = (data, uncompressed_size);
                Err(codec.unsupported(Some("lz4")))
            }
        }

        CompressionCodec::Brotli => {
            let _ = (data, uncompressed_size);
            Err(codec.unsupported(None))
        }
    }
}

/// Raw deflate decompression using `flate2::read::DeflateDecoder`.
///
/// Parquet codec 2 (GZIP) and codec 7 (ZLIB) both use raw deflate,
/// NOT gzip framing, as defined in the Parquet specification.
#[cfg(feature = "gzip")]
fn decompress_deflate(data: &[u8], uncompressed_size: usize) -> Result<Vec<u8>> {
    use flate2::read::DeflateDecoder;

    let mut decoder = DeflateDecoder::new(data);
    ParseBudget::default()
        .read_bounded(&mut decoder, uncompressed_size, "decompressed output")
        .map_err(|error| match error {
            Error::Io(error) => Error::CompressionError {
                message: format!("deflate decompress failed: {error}"),
            },
            error => error,
        })
}

fn checked_output_size(size: usize) -> Result<usize> {
    ParseBudget::default().checked_bytes(
        u64::try_from(size).map_err(|_| Error::Overflow)?,
        "decompressed output",
    )
}

fn copy_bounded(data: &[u8]) -> Result<Vec<u8>> {
    let mut output = allocate_output_buffer(data.len())?;
    output.copy_from_slice(data);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_codec_from_i32_all_known_discriminants() {
        assert_eq!(
            CompressionCodec::from_i32(0),
            Some(CompressionCodec::Uncompressed)
        );
        assert_eq!(
            CompressionCodec::from_i32(1),
            Some(CompressionCodec::Snappy)
        );
        assert_eq!(CompressionCodec::from_i32(2), Some(CompressionCodec::Gzip));
        assert_eq!(
            CompressionCodec::from_i32(3),
            Some(CompressionCodec::Lz4Raw)
        );
        assert_eq!(CompressionCodec::from_i32(4), Some(CompressionCodec::Zstd));
        assert_eq!(CompressionCodec::from_i32(5), Some(CompressionCodec::Lz4));
        assert_eq!(
            CompressionCodec::from_i32(6),
            Some(CompressionCodec::Brotli)
        );
        assert_eq!(CompressionCodec::from_i32(7), Some(CompressionCodec::Zlib));
        assert_eq!(CompressionCodec::from_i32(99), None);
    }

    #[test]
    fn decompress_uncompressed_returns_input() {
        let data: Vec<u8> = alloc::vec![0xDE, 0xAD, 0xBE, 0xEF];
        let out = decompress_page_values(&data, CompressionCodec::Uncompressed, data.len())
            .expect("uncompressed must succeed");
        assert_eq!(out, data);
    }

    #[test]
    fn shared_output_budget_rejects_untrusted_size_hints() {
        let error = checked_output_size(usize::MAX)
            .expect_err("a hostile decompressed-size hint must be rejected");
        assert!(matches!(error, Error::ResourceLimit { .. }));
    }

    #[test]
    fn decompress_brotli_returns_unsupported() {
        let err = decompress_page_values(&[], CompressionCodec::Brotli, 0).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedFeature { ref feature } if feature.contains("BROTLI")),
            "expected UnsupportedFeature with BROTLI, got: {err:?}"
        );
    }

    #[cfg(not(feature = "gzip"))]
    #[test]
    fn decompress_disabled_codec_returns_unsupported() {
        let err = decompress_page_values(&[], CompressionCodec::Gzip, 0).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedFeature { ref feature } if feature.contains("gzip")),
            "expected UnsupportedFeature mentioning 'gzip', got: {err:?}"
        );
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn decompress_gzip_round_trip() {
        use flate2::read::DeflateEncoder;
        use std::io::Read;

        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let mut encoder = DeflateEncoder::new(&input[..], flate2::Compression::new(6));
        let mut compressed = Vec::new();
        encoder
            .read_to_end(&mut compressed)
            .expect("deflate encode must succeed");

        let decompressed = decompress_page_values(&compressed, CompressionCodec::Gzip, input.len())
            .expect("gzip decompress must succeed");
        assert_eq!(decompressed, input, "gzip round-trip must be lossless");
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn decompress_zlib_round_trip() {
        use flate2::read::DeflateEncoder;
        use std::io::Read;

        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let mut encoder = DeflateEncoder::new(&input[..], flate2::Compression::new(6));
        let mut compressed = Vec::new();
        encoder
            .read_to_end(&mut compressed)
            .expect("deflate encode must succeed");

        let decompressed = decompress_page_values(&compressed, CompressionCodec::Zlib, input.len())
            .expect("zlib decompress must succeed");
        assert_eq!(decompressed, input, "zlib round-trip must be lossless");
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn decompress_malformed_gzip_returns_compression_error() {
        let garbage: Vec<u8> = alloc::vec![0xFF, 0xFE, 0xFD, 0xFC, 0x00, 0x01, 0x02, 0x03];
        let err = decompress_page_values(&garbage, CompressionCodec::Gzip, 1024).unwrap_err();
        assert!(
            matches!(err, Error::CompressionError { .. }),
            "expected CompressionError for malformed deflate, got: {err:?}"
        );
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn decompress_zstd_round_trip() {
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let compressed = zrip::compress(&input, 3).expect("zstd compress must succeed");

        let decompressed = decompress_page_values(&compressed, CompressionCodec::Zstd, input.len())
            .expect("zstd decompress must succeed");
        assert_eq!(decompressed, input, "zstd round-trip must be lossless");
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn decompress_malformed_zstd_returns_compression_error() {
        let garbage: Vec<u8> = alloc::vec![0xFF, 0xFE, 0xFD, 0xFC, 0x00, 0x01, 0x02, 0x03];
        let err = decompress_page_values(&garbage, CompressionCodec::Zstd, 1024).unwrap_err();
        assert!(
            matches!(err, Error::CompressionError { .. }),
            "expected CompressionError for malformed zstd, got: {err:?}"
        );
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn decompress_lz4_raw_round_trip() {
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let compressed = lz4_flex::compress(&input);

        let decompressed =
            decompress_page_values(&compressed, CompressionCodec::Lz4Raw, input.len())
                .expect("lz4_raw decompress must succeed");
        assert_eq!(decompressed, input, "lz4_raw round-trip must be lossless");
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn decompress_lz4_round_trip() {
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let compressed = lz4_flex::compress_prepend_size(&input);

        let decompressed = decompress_page_values(&compressed, CompressionCodec::Lz4, input.len())
            .expect("lz4 decompress must succeed");
        assert_eq!(decompressed, input, "lz4 round-trip must be lossless");
    }

    #[cfg(feature = "snappy")]
    #[test]
    fn decompress_snappy_round_trip() {
        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let mut encoder = snap::raw::Encoder::new();
        let compressed = encoder
            .compress_vec(&input)
            .expect("snappy compress must succeed");

        let decompressed =
            decompress_page_values(&compressed, CompressionCodec::Snappy, input.len())
                .expect("snappy decompress must succeed");
        assert_eq!(decompressed, input, "snappy round-trip must be lossless");
    }
}
