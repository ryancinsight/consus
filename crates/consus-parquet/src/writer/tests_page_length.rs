//! A DataPage v2 header's declared size against the stored values and the
//! shared allocation budget, through both `ColumnPageDecoder` entry points.

use alloc::{format, vec::Vec};

use consus_core::{Error, ParseBudget};

use super::thrift::encode_page_header;
use crate::encoding::column::ColumnValues;
#[cfg(feature = "zstd")]
use crate::encoding::compression::ZSTD_OVERLENGTH;
use crate::encoding::compression::{CompressionCodec, compress_page_values, enabled_codecs};
use crate::reader::ColumnPageDecoder;
use crate::schema::physical::ParquetPhysicalType;
use crate::wire::page::{DataPageHeaderV2, PageHeader, PageType};

/// PLAIN INT32 values [10, 20, 30].
const VALUES: [u8; 12] = [10, 0, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0];
/// RLE/bit-packing hybrid levels at bit width 1: three repetition levels 0,
/// then three definition levels 1 (run header `3 << 1`, then the value).
const LEVELS: [u8; 4] = [0x06, 0x00, 0x06, 0x01];

/// Decode one DataPage v2 chunk of `stored` values behind `levels` with both
/// entry points. `declared` is the header's `uncompressed_page_size`.
fn decode_v2(
    codec: CompressionCodec,
    levels: &[u8],
    stored: &[u8],
    is_compressed: bool,
    declared: usize,
) -> [consus_core::Result<ColumnValues>; 2] {
    let half = i32::try_from(levels.len() / 2).unwrap();
    let mut payload = levels.to_vec();
    payload.extend_from_slice(stored);
    let header = PageHeader {
        type_: PageType::DataPageV2,
        uncompressed_page_size: i32::try_from(declared).unwrap(),
        compressed_page_size: i32::try_from(payload.len()).unwrap(),
        crc: None,
        data_page_header: None,
        dictionary_page_header: None,
        data_page_header_v2: Some(DataPageHeaderV2 {
            num_values: 3,
            num_nulls: 0,
            num_rows: 3,
            encoding: 0,
            definition_levels_byte_length: half,
            repetition_levels_byte_length: half,
            is_compressed: Some(is_compressed),
        }),
    };
    let mut chunk = Vec::new();
    encode_page_header(&header, &mut chunk);
    chunk.extend_from_slice(&payload);

    let max_level = i32::from(!levels.is_empty());
    let new = || ColumnPageDecoder::new(ParquetPhysicalType::Int32, codec, max_level, max_level);
    [
        new().decode_pages_from_chunk_bytes(&chunk),
        new().decode_pages_with_levels(&chunk).map(|r| r.values),
    ]
}

fn assert_values(result: &consus_core::Result<ColumnValues>, context: &str) {
    assert!(
        matches!(result, Ok(ColumnValues::Int32(v)) if *v == [10, 20, 30]),
        "{context}: {result:?}"
    );
}

/// Pages whose values are stored as is: UNCOMPRESSED, or `is_compressed =
/// false` under every other codec. The declared values size is 12 bytes.
#[test]
fn stored_values_must_match_the_declared_size() {
    let stored_as_is = enabled_codecs()
        .map(|codec| (codec, false))
        .chain([(CompressionCodec::Uncompressed, true)]);
    for (codec, is_compressed) in stored_as_is {
        for levels in [&[][..], &LEVELS[..]] {
            let total = levels.len() + VALUES.len();
            for result in decode_v2(codec, levels, &VALUES, is_compressed, total) {
                assert_values(&result, &format!("{codec:?} {is_compressed} exact"));
            }
            for declared in [total - 1, total + 1] {
                let expected = format!(
                    "UNCOMPRESSED page decoded to 12 bytes, declared {}",
                    declared - levels.len()
                );
                for result in decode_v2(codec, levels, &VALUES, is_compressed, declared) {
                    assert!(
                        matches!(&result, Err(Error::CompressionError { message })
                            if *message == expected),
                        "{codec:?} {is_compressed} {declared}: {result:?}"
                    );
                }
            }
        }
    }
}

/// Compressed values behind nonzero level sections decode when the declared
/// size covers levels plus values, and fail by one byte either way.
#[test]
fn compressed_values_behind_levels_use_the_levels_subtracted_size() {
    for codec in enabled_codecs() {
        let stored = compress_page_values(&VALUES, codec).unwrap();
        let total = LEVELS.len() + VALUES.len();
        for result in decode_v2(codec, &LEVELS, &stored, true, total) {
            assert_values(&result, &format!("{codec:?} exact"));
        }
        for declared in [total - 1, total + 1] {
            for result in decode_v2(codec, &LEVELS, &stored, true, declared) {
                assert!(
                    matches!(result, Err(Error::CompressionError { .. })),
                    "{codec:?} {declared}: {result:?}"
                );
            }
        }
    }
}

/// A declared size smaller than the level sections cannot be subtracted from.
#[test]
fn declared_size_below_the_levels_is_a_typed_error() {
    for result in decode_v2(
        CompressionCodec::Uncompressed,
        &LEVELS,
        &[],
        false,
        LEVELS.len() - 1,
    ) {
        assert!(
            matches!(result, Err(Error::InvalidFormat { .. })),
            "{result:?}"
        );
    }
}

/// Bytes a codec reserves beyond the declared size, which count toward the budget.
#[cfg(feature = "zstd")]
fn buffer_slack(codec: CompressionCodec) -> usize {
    if codec == CompressionCodec::Zstd {
        ZSTD_OVERLENGTH
    } else {
        0
    }
}

#[cfg(not(feature = "zstd"))]
fn buffer_slack(_: CompressionCodec) -> usize {
    0
}

/// Compressed values behind four level bytes: the budget applies to the
/// declared size, the header less the levels, not to the header field. Headers
/// up to the budget plus the level bytes, less the codec's buffer slack, are
/// within-budget declarations and fail the length comparison; the next header
/// is the first above the budget.
#[test]
fn compressed_values_budget_boundary_is_the_size_less_the_levels() {
    let budget = ParseBudget::default().max_alloc_bytes;
    let budget_u64 = u64::try_from(budget).unwrap();
    for codec in enabled_codecs().filter(|&codec| codec != CompressionCodec::Uncompressed) {
        let stored = compress_page_values(&VALUES, codec).unwrap();
        let last_within = budget + LEVELS.len() - buffer_slack(codec);
        for declared in [budget + 1 - buffer_slack(codec), last_within] {
            let expected = format!("declared {}", declared - LEVELS.len());
            for result in decode_v2(codec, &LEVELS, &stored, true, declared) {
                assert!(
                    matches!(&result, Err(Error::CompressionError { message })
                        if message.ends_with(&expected)),
                    "{codec:?} {declared}: {result:?}"
                );
            }
        }
        for result in decode_v2(codec, &LEVELS, &stored, true, last_within + 1) {
            assert!(
                matches!(&result, Err(Error::ResourceLimit { requested, limit, .. })
                    if *requested == budget_u64 + 1 && *limit == budget_u64),
                "{codec:?}: {result:?}"
            );
        }
    }
}

/// Values stored as is are compared with the declared size and never
/// decompressed, so a header above the budget is a length mismatch, not a
/// resource limit.
#[test]
fn stored_values_above_the_budget_are_a_length_mismatch() {
    let budget = ParseBudget::default().max_alloc_bytes;
    let declared = LEVELS.len() + budget + 1;
    let expected = format!(
        "UNCOMPRESSED page decoded to 12 bytes, declared {}",
        budget + 1
    );
    for result in decode_v2(
        CompressionCodec::Uncompressed,
        &LEVELS,
        &VALUES,
        true,
        declared,
    ) {
        assert!(
            matches!(&result, Err(Error::CompressionError { message })
                if *message == expected),
            "{result:?}"
        );
    }
}
