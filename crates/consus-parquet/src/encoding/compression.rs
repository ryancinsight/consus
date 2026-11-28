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
//! | 6 | ZSTD | `zstd` | `zrip::DecompressContext::decompress_into_with_limit` |
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
use consus_compression::{
    Codec, CompressionLevel,
    codec::{
        flate::{GzipFraming, flate2_decompress},
        gzip::GzipCodec,
    },
};
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

/// Bytes `zrip-decode` 0.8.13 reserves past its output limit (`WILDCOPY_OVERLENGTH`,
/// `fast_vec.rs:7`), so the buffer for a ZSTD page is the declared size plus these.
#[cfg(feature = "zstd")]
pub(crate) const ZSTD_OVERLENGTH: usize = 64;

/// Decompress page value bytes according to the Parquet compression codec.
///
/// `uncompressed_size` is the size the page header declares, and the header is
/// untrusted; a DataPage v2 caller passes the declared size less the level
/// sections and refuses, with `InvalidFormat`, a header smaller than them.
/// The outcome by case, for every codec that is enabled:
///
/// - A declaration above the shared [`ParseBudget`] allocation bound is
///   `ResourceLimit`, before any data is decoded; for ZSTD the bound applies
///   to the declaration plus 64 bytes (see Allocation).
/// - A declaration within the bound, with decoded output above the bound:
///   `ResourceLimit` for UNCOMPRESSED (payload length), SNAPPY (the length
///   its stream states) and GZIP (decoded output); `CompressionError` for
///   ZSTD and LZ4_RAW, which decode into a limit equal to the declaration.
/// - A GZIP declaration within the bound but above what the stored bytes can
///   produce: the decoded length is shorter than declared, so the page reports
///   `CompressionError`.
/// - Any other decoded length that differs from the declaration:
///   `CompressionError`. A page is never returned truncated or padded.
///
/// Allocation: UNCOMPRESSED reserves the payload length, SNAPPY the length
/// its stream states, LZ4_RAW the declared size, each at most the bound.
/// GZIP reserves the smaller of the declaration and the output the stored
/// bytes can produce (1032 times their length, `MAX_DEFLATE_EXPANSION`), so a
/// declaration above that is a length mismatch, never a reservation, and it
/// grows from there within the bound. ZSTD reserves the declared size plus
/// `ZSTD_OVERLENGTH` (64) bytes, which must also be within the bound, and decodes
/// with the declaration as its output limit. `zrip-decode` 0.8.13 then never
/// grows that buffer: its only reservations are `lib.rs:279`,
/// `fast_vec.rs:23` and `fast_vec.rs:262`, each at most the output limit
/// plus 64 bytes (`WILDCOPY_OVERLENGTH`, `fast_vec.rs:7`), and every write is
/// checked against the limit first (`lib.rs:339-353`, `lib.rs:477-491`,
/// `fast_vec.rs:41-58`). A frame's limit is the limit less what earlier
/// frames wrote (`context.rs:180`), so the bound holds for a stream of many
/// frames. A `Vec` doubles only when a request exceeds its capacity.
///
/// # Errors
///
/// - `Error::UnsupportedFeature` — codec is not implemented (LZO, BROTLI, LZ4)
///   or the corresponding feature flag is not enabled.
/// - `Error::ResourceLimit` — the declaration (for ZSTD, the declaration plus
///   64 bytes), or for UNCOMPRESSED, SNAPPY and GZIP the decoded output,
///   exceeds the shared budget.
/// - `Error::CompressionError` — the compressed data is malformed,
///   decompression fails (ZSTD and LZ4_RAW include output above the
///   declaration), or the page decodes to a length other than
///   `uncompressed_size`.
pub fn decompress_page_values(
    data: &[u8],
    codec: CompressionCodec,
    uncompressed_size: usize,
) -> Result<Vec<u8>> {
    checked_output_size(uncompressed_size)?;
    let output = decode_page_bytes(data, codec, uncompressed_size)?;
    check_decoded_length(codec, output.len(), uncompressed_size)?;
    Ok(output)
}

/// Refuses a page whose decoded length differs from the length its header declares.
pub(crate) fn check_decoded_length(
    codec: CompressionCodec,
    decoded: usize,
    declared: usize,
) -> Result<()> {
    if decoded == declared {
        return Ok(());
    }
    Err(Error::CompressionError {
        message: format!(
            "{} page decoded to {decoded} bytes, declared {declared}",
            codec_name(codec)
        ),
    })
}

/// Decode `data` under `codec`; `uncompressed_size` is already within the budget.
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
                // The declaration is a reservation hint clamped to what `data`
                // can produce; the page length is checked below.
                flate2_decompress::<GzipFraming>(data, uncompressed_size, &ParseBudget::default())
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
                let capacity = uncompressed_size
                    .checked_add(ZSTD_OVERLENGTH)
                    .ok_or(Error::Overflow)?;
                // `zeroed` reserves exactly `capacity`, which `clear` keeps.
                let mut output = ParseBudget::default().zeroed(
                    u64::try_from(capacity).map_err(|_| Error::Overflow)?,
                    "decompressed output",
                )?;
                output.clear();
                zrip::DecompressContext::new()
                    .decompress_into_with_limit(data, &mut output, uncompressed_size)
                    .map_err(|e| Error::CompressionError {
                        message: format!("zstd decompress failed: {e}"),
                    })?;
                Ok(output)
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
                let mut output = ParseBudget::default().zeroed(
                    u64::try_from(uncompressed_size).map_err(|_| Error::Overflow)?,
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

/// Codecs whose feature is enabled in this build, `UNCOMPRESSED` first.
#[cfg(test)]
pub(crate) fn enabled_codecs() -> impl Iterator<Item = CompressionCodec> {
    [
        (true, CompressionCodec::Uncompressed),
        (cfg!(feature = "snappy"), CompressionCodec::Snappy),
        (cfg!(feature = "gzip"), CompressionCodec::Gzip),
        (cfg!(feature = "zstd"), CompressionCodec::Zstd),
        (cfg!(feature = "lz4"), CompressionCodec::Lz4Raw),
    ]
    .into_iter()
    .filter_map(|(enabled, codec)| enabled.then_some(codec))
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

    /// The page header's size hint is untrusted: above the budget it is a
    /// resource-limit error for every codec, whatever the payload.
    #[test]
    fn decompress_rejects_hostile_size_hint_for_every_codec() {
        let budget = ParseBudget::default().max_alloc_bytes;
        for codec in enabled_codecs() {
            let page = compress_page_values(&[0u8; 16], codec).unwrap();
            for hint in [budget + 1, usize::MAX] {
                let err = decompress_page_values(&page, codec, hint).unwrap_err();
                assert!(
                    matches!(err, Error::ResourceLimit { .. }),
                    "{}: {hint}: {err:?}",
                    codec_name(codec)
                );
            }
        }
    }

    /// Decoded output above the budget under an in-budget declaration:
    /// `ResourceLimit` for UNCOMPRESSED (payload length), SNAPPY (stream-stated
    /// length) and GZIP (decoded output); `CompressionError` from the codec
    /// for ZSTD and LZ4_RAW, whose limit is the declaration.
    #[test]
    fn decompress_output_above_budget_under_small_declaration_for_every_codec() {
        let budget = ParseBudget::default().max_alloc_bytes;
        let budget_u64 = u64::try_from(budget).unwrap();
        let plain = alloc::vec![0u8; budget + 1];
        for codec in enabled_codecs() {
            let name = codec_name(codec);
            let page = compress_page_values(&plain, codec).unwrap();
            let err = decompress_page_values(&page, codec, 12).unwrap_err();
            match codec {
                CompressionCodec::Uncompressed | CompressionCodec::Snappy => assert!(
                    matches!(err, Error::ResourceLimit { requested, limit, .. }
                        if requested == budget_u64 + 1 && limit == budget_u64),
                    "{name}: {err:?}"
                ),
                CompressionCodec::Gzip => assert!(
                    matches!(err, Error::ResourceLimit { requested, limit, .. }
                        if requested > budget_u64 && limit == budget_u64),
                    "{name}: {err:?}"
                ),
                CompressionCodec::Zstd => assert!(
                    matches!(err, Error::CompressionError { ref message }
                        if message.starts_with("zstd decompress failed")),
                    "{name}: {err:?}"
                ),
                CompressionCodec::Lz4Raw => assert!(
                    matches!(err, Error::CompressionError { ref message }
                        if message.starts_with("lz4_raw decompress failed")),
                    "{name}: {err:?}"
                ),
                CompressionCodec::Lzo | CompressionCodec::Brotli | CompressionCodec::Lz4 => {
                    unreachable!("invariant: enabled_codecs yields only implemented codecs")
                }
            }
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

    /// Members that together decode past the budget are `ResourceLimit` under
    /// an in-budget declaration. `read_bounded` fails on the read that first
    /// takes the output past the budget, and one read adds at most its
    /// 8192-byte chunk.
    #[cfg(feature = "gzip")]
    #[test]
    fn decompress_gzip_members_past_the_budget_are_a_resource_limit() {
        const MEMBER_LEN: usize = 1 << 20;
        const READ_CHUNK: u64 = 8192;
        let budget = ParseBudget::default().max_alloc_bytes;
        let limit = u64::try_from(budget).unwrap();
        let member =
            compress_page_values(&alloc::vec![0u8; MEMBER_LEN], CompressionCodec::Gzip).unwrap();
        let members = member.repeat(budget / MEMBER_LEN + 1);
        let err = decompress_page_values(&members, CompressionCodec::Gzip, 12).unwrap_err();
        assert!(
            matches!(&err, Error::ResourceLimit { what: "decompressed output", requested, limit: reported }
                if *reported == limit && (limit + 1..=limit + READ_CHUNK).contains(requested)),
            "{err:?}"
        );
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
        for codec in enabled_codecs() {
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

    /// `zrip-decode` reserves per frame and per block; under one declaration
    /// of many frames the buffer must not double past the budget. 9 frames of
    /// 4 MiB double an unreserved buffer from 4 MiB to above 64 MiB.
    #[cfg(feature = "zstd")]
    #[test]
    fn decompress_zstd_of_many_frames_stays_within_the_budget() {
        const FRAME_BYTES: usize = 4 * 1024 * 1024;
        const FRAMES: usize = 9;
        let frame = zrip::compress(&alloc::vec![0u8; FRAME_BYTES], 3).expect("zstd compress");
        let page = frame.repeat(FRAMES);
        let declared = FRAME_BYTES * FRAMES;
        let decoded = decompress_page_values(&page, CompressionCodec::Zstd, declared)
            .expect("nine frames within the declaration must decode");
        assert_eq!(decoded.len(), declared);
        assert!(decoded.iter().all(|&byte| byte == 0));
        assert!(
            decoded.capacity() <= ParseBudget::default().max_alloc_bytes,
            "capacity {} above the budget",
            decoded.capacity()
        );
    }

    /// The buffer is the declaration plus [`ZSTD_OVERLENGTH`], so the largest
    /// declaration the budget admits is the budget less that slack.
    #[cfg(feature = "zstd")]
    #[test]
    fn decompress_zstd_budget_edge_is_the_declaration_plus_overlength() {
        let budget = ParseBudget::default().max_alloc_bytes;
        let page = zrip::compress(b"hello", 3).expect("zstd compress");
        let largest = budget - ZSTD_OVERLENGTH;
        let err = decompress_page_values(&page, CompressionCodec::Zstd, largest).unwrap_err();
        assert!(
            matches!(err, Error::CompressionError { ref message }
                if *message == format!("ZSTD page decoded to 5 bytes, declared {largest}")),
            "{err:?}"
        );
        let err = decompress_page_values(&page, CompressionCodec::Zstd, largest + 1).unwrap_err();
        assert!(
            matches!(err, Error::ResourceLimit { requested, limit, .. }
                if requested == u64::try_from(budget + 1).unwrap()
                    && limit == u64::try_from(budget).unwrap()),
            "{err:?}"
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
