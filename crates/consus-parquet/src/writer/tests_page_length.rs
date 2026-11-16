//! DataPage v2 values behind level sections, stored or compressed, through
//! both `ColumnPageDecoder` entry points.

use alloc::{format, vec::Vec};

use consus_core::Error;

use super::thrift::encode_page_header;
use crate::encoding::column::ColumnValues;
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
/// false` under every other codec.
#[test]
fn stored_values_decode_behind_levels() {
    let stored_as_is = enabled_codecs()
        .map(|codec| (codec, false))
        .chain([(CompressionCodec::Uncompressed, true)]);
    for (codec, is_compressed) in stored_as_is {
        for levels in [&[][..], &LEVELS[..]] {
            let total = levels.len() + VALUES.len();
            for result in decode_v2(codec, levels, &VALUES, is_compressed, total) {
                assert_values(&result, &format!("{codec:?} {is_compressed}"));
            }
        }
    }
}

/// Compressed values behind nonzero level sections decode when the declared
/// size covers levels plus values, and fail by one byte either way. Values
/// under UNCOMPRESSED are stored, not compressed, and are not compared.
#[test]
fn compressed_values_behind_levels_use_the_levels_subtracted_size() {
    for codec in enabled_codecs().filter(|&codec| codec != CompressionCodec::Uncompressed) {
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
