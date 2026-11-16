//! Parquet compression codec dispatch for page value decompression.
//!
//! Maps Parquet thrift `CompressionCodec` discriminants to concrete
//! decompression routines. Feature-gated: codecs whose feature is not
//! enabled return `Error::UnsupportedFeature` with an actionable message.
//!
//! ## Codec mapping
//!
//! Numbers: `CompressionCodec` in apache/parquet-format `parquet.thrift`, lines 684-693.
//!
//! | Number | Parquet name | Rust feature | Backend |
//! |--------|--------------|--------------|---------|
//! | 0 | UNCOMPRESSED | (always) | pass-through |
//! | 1 | SNAPPY | `snappy` | `snap` crate |
//! | 2 | GZIP | `gzip` | `GzipCodec` of `consus-compression` (RFC 1952) |
//! | 3 | LZO | (none) | always `UnsupportedFeature` |
//! | 4 | BROTLI | (none) | always `UnsupportedFeature` |
//! | 5 | LZ4 | (none) | always `UnsupportedFeature` (deprecated framing) |
//! | 6 | ZSTD | `zstd` | `zrip::decompress_with_limit` |
//! | 7 | LZ4_RAW | `lz4` | `lz4_flex::decompress` (raw block) |

#[cfg(not(all(
    feature = "gzip",
    feature = "snappy",
    feature = "zstd",
    feature = "lz4"
)))]
use alloc::string::String;
use alloc::{format, vec::Vec};

#[cfg(feature = "gzip")]
use consus_compression::{Codec, CompressionLevel, codec::gzip::GzipCodec};
use consus_core::{Error, ParseBudget, Result};

/// Parquet compression codec; [`Self::to_i32`] gives the thrift `CompressionCodec` number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCodec {
    Uncompressed,
    Snappy,
    Gzip,
    Lzo,
    Brotli,
    Lz4,
    Zstd,
    Lz4Raw,
}

impl CompressionCodec {
    /// Convert an i32 discriminant to a `CompressionCodec`.
    ///
    /// Returns `None` for values outside the specified range 0..=7.
    pub const fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Uncompressed),
            1 => Some(Self::Snappy),
            2 => Some(Self::Gzip),
            3 => Some(Self::Lzo),
            4 => Some(Self::Brotli),
            5 => Some(Self::Lz4),
            6 => Some(Self::Zstd),
            7 => Some(Self::Lz4Raw),
            _ => None,
        }
    }

    /// The thrift `CompressionCodec` number written to column chunk metadata.
    pub const fn to_i32(self) -> i32 {
        match self {
            Self::Uncompressed => 0,
            Self::Snappy => 1,
            Self::Gzip => 2,
            Self::Lzo => 3,
            Self::Brotli => 4,
            Self::Lz4 => 5,
            Self::Zstd => 6,
            Self::Lz4Raw => 7,
        }
    }
}

/// Parquet name of `codec` as written in `CompressionCodec` of `parquet.thrift`.
const fn codec_name(codec: CompressionCodec) -> &'static str {
    match codec {
        CompressionCodec::Uncompressed => "UNCOMPRESSED",
        CompressionCodec::Snappy => "SNAPPY",
        CompressionCodec::Gzip => "GZIP",
        CompressionCodec::Lzo => "LZO",
        CompressionCodec::Brotli => "BROTLI",
        CompressionCodec::Lz4 => "LZ4",
        CompressionCodec::Zstd => "ZSTD",
        CompressionCodec::Lz4Raw => "LZ4_RAW",
    }
}

/// Error for a codec this crate has no implementation of (LZO, BROTLI, LZ4).
fn unsupported_codec(codec: CompressionCodec) -> Error {
    Error::UnsupportedFeature {
        feature: format!(
            "parquet compression codec {} ({})",
            codec.to_i32(),
            codec_name(codec)
        ),
    }
}

/// Compress page value bytes using the given Parquet compression codec.
///
/// UNCOMPRESSED: returns `data.to_vec()` (pass-through, no error).
/// GZIP: one RFC 1952 member via `GzipCodec` at the default level (6).
/// SNAPPY: `snap::raw::Encoder::new().compress_vec(data)`.
/// ZSTD: `zrip::compress(data, 3)`.
/// LZ4_RAW: `lz4_flex::compress(data)`.
/// LZO, BROTLI, LZ4 (deprecated framing): always `Error::UnsupportedFeature`.
///
/// Feature gating matches `decompress_page_values`: same `#[cfg(feature = "gzip")]` etc.
/// If the feature is not enabled, returns `Error::UnsupportedFeature` with an actionable message.
pub(crate) fn compress_page_values(data: &[u8], codec: CompressionCodec) -> Result<Vec<u8>> {
    match codec {
        CompressionCodec::Uncompressed => Ok(data.to_vec()),

        CompressionCodec::Gzip => {
            #[cfg(feature = "gzip")]
            {
                GzipCodec.compress(data, CompressionLevel::default())
            }
            #[cfg(not(feature = "gzip"))]
            {
                let _ = data;
                Err(Error::UnsupportedFeature {
                    feature: String::from(
                        "parquet compression codec 2 (GZIP) — enable feature 'gzip'",
                    ),
                })
            }
        }

        CompressionCodec::Snappy => {
            #[cfg(feature = "snappy")]
            {
                snap::raw::Encoder::new()
                    .compress_vec(data)
                    .map_err(|e| Error::CompressionError {
                        message: format!("snappy compress failed: {e}"),
                    })
            }
            #[cfg(not(feature = "snappy"))]
            {
                let _ = data;
                Err(Error::UnsupportedFeature {
                    feature: String::from(
                        "parquet compression codec 1 (SNAPPY) — enable feature 'snappy'",
                    ),
                })
            }
        }

        CompressionCodec::Zstd => {
            #[cfg(feature = "zstd")]
            {
                zrip::compress(data, 3).map_err(|e| Error::CompressionError {
                    message: format!("zstd compress failed: {e}"),
                })
            }
            #[cfg(not(feature = "zstd"))]
            {
                let _ = data;
                Err(Error::UnsupportedFeature {
                    feature: String::from(
                        "parquet compression codec 6 (ZSTD) — enable feature 'zstd'",
                    ),
                })
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
                Err(Error::UnsupportedFeature {
                    feature: String::from(
                        "parquet compression codec 7 (LZ4_RAW) — enable feature 'lz4'",
                    ),
                })
            }
        }

        CompressionCodec::Lzo | CompressionCodec::Brotli | CompressionCodec::Lz4 => {
            Err(unsupported_codec(codec))
        }
    }
}

/// Decompress page value bytes according to the Parquet compression codec.
///
/// `uncompressed_size` is the size the page header declares. It bounds the
/// output allocation through the shared [`ParseBudget`] and is a required
/// parameter for some codecs (zstd, lz4). The decoded length must equal it for
/// every codec; the header is untrusted, so a page that decodes to any other
/// length is refused, never returned truncated or padded.
///
/// # Errors
///
/// - `Error::UnsupportedFeature` — codec is not implemented (LZO, BROTLI, LZ4)
///   or the corresponding feature flag is not enabled.
/// - `Error::ResourceLimit` — `uncompressed_size` or the decoded output
///   exceeds the shared budget.
/// - `Error::CompressionError` — the compressed data is malformed,
///   decompression fails, or the page decodes to a length other than
///   `uncompressed_size`.
pub fn decompress_page_values(
    data: &[u8],
    codec: CompressionCodec,
    uncompressed_size: usize,
) -> Result<Vec<u8>> {
    let output = decode_page_bytes(data, codec, uncompressed_size)?;
    if output.len() != uncompressed_size {
        return Err(Error::CompressionError {
            message: format!(
                "{} page decoded to {} bytes, declared {uncompressed_size}",
                codec_name(codec),
                output.len()
            ),
        });
    }
    Ok(output)
}

/// Decode `data` under `codec`, bounded by the declared `uncompressed_size`.
fn decode_page_bytes(
    data: &[u8],
    codec: CompressionCodec,
    uncompressed_size: usize,
) -> Result<Vec<u8>> {
    match codec {
        CompressionCodec::Uncompressed => copy_bounded(data),

        CompressionCodec::Gzip => {
            #[cfg(feature = "gzip")]
            {
                let limit = checked_output_size(uncompressed_size)?;
                GzipCodec.decompress(data, limit)
            }
            #[cfg(not(feature = "gzip"))]
            {
                let _ = (data, uncompressed_size);
                Err(Error::UnsupportedFeature {
                    feature: String::from(
                        "parquet compression codec 2 (GZIP) — enable feature 'gzip'",
                    ),
                })
            }
        }

        CompressionCodec::Snappy => {
            #[cfg(feature = "snappy")]
            {
                let decoded_size =
                    snap::raw::decompress_len(data).map_err(|e| Error::CompressionError {
                        message: format!("snappy header decode failed: {e}"),
                    })?;
                let mut output = ParseBudget::default().zeroed(
                    u64::try_from(decoded_size).map_err(|_| Error::Overflow)?,
                    "decompressed output",
                )?;
                let written = snap::raw::Decoder::new()
                    .decompress(data, &mut output)
                    .map_err(|e| Error::CompressionError {
                        message: format!("snappy decompress failed: {e}"),
                    })?;
                output.truncate(written);
                Ok(output)
            }
            #[cfg(not(feature = "snappy"))]
            {
                let _ = (data, uncompressed_size);
                Err(Error::UnsupportedFeature {
                    feature: String::from(
                        "parquet compression codec 1 (SNAPPY) — enable feature 'snappy'",
                    ),
                })
            }
        }

        CompressionCodec::Zstd => {
            #[cfg(feature = "zstd")]
            {
                let limit = checked_output_size(uncompressed_size)?;
                zrip::decompress_with_limit(data, limit).map_err(|e| Error::CompressionError {
                    message: format!("zstd decompress failed: {e}"),
                })
            }
            #[cfg(not(feature = "zstd"))]
            {
                let _ = (data, uncompressed_size);
                Err(Error::UnsupportedFeature {
                    feature: String::from(
                        "parquet compression codec 6 (ZSTD) — enable feature 'zstd'",
                    ),
                })
            }
        }

        CompressionCodec::Lz4Raw => {
            #[cfg(feature = "lz4")]
            {
                let output_size = checked_output_size(uncompressed_size)?;
                let mut output = ParseBudget::default().zeroed(
                    u64::try_from(output_size).map_err(|_| Error::Overflow)?,
                    "decompressed output",
                )?;
                let written = lz4_flex::block::decompress_into(data, &mut output).map_err(|e| {
                    Error::CompressionError {
                        message: format!("lz4_raw decompress failed: {e}"),
                    }
                })?;
                output.truncate(written);
                Ok(output)
            }
            #[cfg(not(feature = "lz4"))]
            {
                let _ = (data, uncompressed_size);
                Err(Error::UnsupportedFeature {
                    feature: String::from(
                        "parquet compression codec 7 (LZ4_RAW) — enable feature 'lz4'",
                    ),
                })
            }
        }

        CompressionCodec::Lzo | CompressionCodec::Brotli | CompressionCodec::Lz4 => {
            let _ = (data, uncompressed_size);
            Err(unsupported_codec(codec))
        }
    }
}

fn checked_output_size(size: usize) -> Result<usize> {
    ParseBudget::default().checked_bytes(
        u64::try_from(size).map_err(|_| Error::Overflow)?,
        "decompressed output",
    )
}

fn copy_bounded(data: &[u8]) -> Result<Vec<u8>> {
    let len = checked_output_size(data.len())?;
    let mut output = ParseBudget::default().zeroed(
        u64::try_from(len).map_err(|_| Error::Overflow)?,
        "decompressed output",
    )?;
    output.copy_from_slice(data);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows of `CompressionCodec` in parquet.thrift lines 684-693 of
    /// apache/parquet-format.
    #[test]
    fn codec_numbers_match_parquet_thrift() {
        for (number, codec) in [
            (0, CompressionCodec::Uncompressed),
            (1, CompressionCodec::Snappy),
            (2, CompressionCodec::Gzip),
            (3, CompressionCodec::Lzo),
            (4, CompressionCodec::Brotli),
            (5, CompressionCodec::Lz4),
            (6, CompressionCodec::Zstd),
            (7, CompressionCodec::Lz4Raw),
        ] {
            assert_eq!(CompressionCodec::from_i32(number), Some(codec));
            assert_eq!(codec.to_i32(), number);
        }
        for number in [-1, 8, 99, i32::MIN, i32::MAX] {
            assert_eq!(CompressionCodec::from_i32(number), None, "{number}");
        }
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
    fn codecs_without_an_implementation_name_themselves_in_the_error() {
        for (codec, number_and_name) in [
            (CompressionCodec::Lzo, "3 (LZO)"),
            (CompressionCodec::Brotli, "4 (BROTLI)"),
            (CompressionCodec::Lz4, "5 (LZ4)"),
        ] {
            let expected = format!("parquet compression codec {number_and_name}");
            let decode = decompress_page_values(&[], codec, 0).unwrap_err();
            let encode = compress_page_values(&[], codec).unwrap_err();
            for err in [decode, encode] {
                assert!(
                    matches!(err, Error::UnsupportedFeature { ref feature } if *feature == expected),
                    "expected UnsupportedFeature({expected}), got: {err:?}"
                );
            }
        }
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

    /// RFC 1952 section 2.3.1: a member starts with ID1 = 0x1f, ID2 = 0x8b, CM = 8.
    #[cfg(feature = "gzip")]
    #[test]
    fn compress_gzip_emits_rfc1952_member() {
        let input: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let compressed =
            compress_page_values(&input, CompressionCodec::Gzip).expect("compress must succeed");
        assert_eq!(&compressed[..3], &[0x1F, 0x8B, 0x08]);
        let decompressed = decompress_page_values(&compressed, CompressionCodec::Gzip, input.len())
            .expect("decompress must succeed");
        assert_eq!(decompressed, input, "gzip round-trip must be lossless");
    }

    /// An independent `flate2` gzip encoder produces a stream the reader
    /// accepts; a bare RFC 1951 stream is not GZIP and is rejected.
    #[cfg(feature = "gzip")]
    #[test]
    fn decompress_gzip_accepts_independent_gzip_and_rejects_bare_deflate() {
        use flate2::read::{DeflateEncoder, GzEncoder};
        use std::io::Read;

        let input: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let mut gzip = Vec::new();
        GzEncoder::new(&input[..], flate2::Compression::new(6))
            .read_to_end(&mut gzip)
            .expect("gzip encode must succeed");
        let decompressed = decompress_page_values(&gzip, CompressionCodec::Gzip, input.len())
            .expect("gzip decompress must succeed");
        assert_eq!(decompressed, input);

        let mut bare = Vec::new();
        DeflateEncoder::new(&input[..], flate2::Compression::new(6))
            .read_to_end(&mut bare)
            .expect("deflate encode must succeed");
        let err = decompress_page_values(&bare, CompressionCodec::Gzip, input.len()).unwrap_err();
        assert!(
            matches!(err, Error::CompressionError { .. }),
            "bare deflate is not RFC 1952: {err:?}"
        );
    }

    /// The page header's size hint is untrusted: it must not raise the output ceiling.
    #[cfg(feature = "gzip")]
    #[test]
    fn decompress_gzip_rejects_hostile_size_hint() {
        let compressed = compress_page_values(&[0u8; 16], CompressionCodec::Gzip).unwrap();
        let budget = ParseBudget::default().max_alloc_bytes;
        for hint in [budget + 1, usize::MAX] {
            let err =
                decompress_page_values(&compressed, CompressionCodec::Gzip, hint).unwrap_err();
            assert!(
                matches!(err, Error::ResourceLimit { .. }),
                "{hint}: {err:?}"
            );
        }
    }

    /// Compression.md (GZIP section) says readers should support pages
    /// containing multiple GZIP members; every member is decoded.
    #[cfg(feature = "gzip")]
    #[test]
    fn decompress_gzip_with_two_members_returns_both() {
        let plain: Vec<u8> = (0u8..=255).cycle().take(400).collect();
        let (head, tail) = plain.split_at(200);
        let mut members = compress_page_values(head, CompressionCodec::Gzip).unwrap();
        members.extend(compress_page_values(tail, CompressionCodec::Gzip).unwrap());
        let decoded = decompress_page_values(&members, CompressionCodec::Gzip, plain.len())
            .expect("a two-member page must decode");
        assert_eq!(decoded, plain);
    }

    /// A header size that is one byte off the decoded length is a typed
    /// error in both directions, never a truncated or padded page.
    #[cfg(feature = "gzip")]
    #[test]
    fn decompress_gzip_with_size_off_by_one_is_a_typed_error() {
        let plain: Vec<u8> = (0u8..=255).cycle().take(400).collect();
        let (head, tail) = plain.split_at(200);
        let mut members = compress_page_values(head, CompressionCodec::Gzip).unwrap();
        members.extend(compress_page_values(tail, CompressionCodec::Gzip).unwrap());
        for declared in [plain.len() - 1, plain.len() + 1] {
            let err =
                decompress_page_values(&members, CompressionCodec::Gzip, declared).unwrap_err();
            assert!(
                matches!(err, Error::CompressionError { ref message }
                    if *message == format!("GZIP page decoded to 400 bytes, declared {declared}")),
                "{declared}: {err:?}"
            );
        }
    }

    /// The declared size must equal the decoded length for every codec, in
    /// both directions. Zstd and LZ4_RAW decode into a buffer sized from the
    /// declaration, so a too-small declaration fails inside the codec; every
    /// other refusal is the shared length error.
    #[test]
    fn decompress_size_off_by_one_is_a_typed_error_for_every_codec() {
        let plain: Vec<u8> = (0u8..=255).cycle().take(400).collect();
        let mut codecs = alloc::vec![CompressionCodec::Uncompressed];
        #[cfg(feature = "snappy")]
        codecs.push(CompressionCodec::Snappy);
        #[cfg(feature = "gzip")]
        codecs.push(CompressionCodec::Gzip);
        #[cfg(feature = "zstd")]
        codecs.push(CompressionCodec::Zstd);
        #[cfg(feature = "lz4")]
        codecs.push(CompressionCodec::Lz4Raw);

        for codec in codecs {
            let name = codec_name(codec);
            let page = compress_page_values(&plain, codec).unwrap();
            assert_eq!(
                decompress_page_values(&page, codec, plain.len()).unwrap(),
                plain,
                "{name}"
            );

            let too_large = decompress_page_values(&page, codec, plain.len() + 1).unwrap_err();
            assert!(
                matches!(too_large, Error::CompressionError { ref message }
                    if *message == format!("{name} page decoded to 400 bytes, declared 401")),
                "{name}: {too_large:?}"
            );

            let too_small = decompress_page_values(&page, codec, plain.len() - 1).unwrap_err();
            let native_prefix = match codec {
                CompressionCodec::Zstd => Some("zstd decompress failed"),
                CompressionCodec::Lz4Raw => Some("lz4_raw decompress failed"),
                _ => None,
            };
            assert!(
                matches!(too_small, Error::CompressionError { ref message }
                if native_prefix.map_or(
                    *message == format!("{name} page decoded to 400 bytes, declared 399"),
                    |prefix| message.starts_with(prefix),
                )),
                "{name}: {too_small:?}"
            );
        }
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn decompress_malformed_gzip_returns_compression_error() {
        let garbage: Vec<u8> = alloc::vec![0xFF, 0xFE, 0xFD, 0xFC, 0x00, 0x01, 0x02, 0x03];
        let err = decompress_page_values(&garbage, CompressionCodec::Gzip, 1024).unwrap_err();
        assert!(
            matches!(err, Error::CompressionError { .. }),
            "expected CompressionError for malformed gzip, got: {err:?}"
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
