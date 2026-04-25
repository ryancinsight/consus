//! Parquet data page payload splitter.
//!
//! ## Specification
//!
//! Splits the raw byte payload of a Parquet data page into three components:
//! repetition levels, definition levels, and value bytes.
//!
//! Reference: <https://github.com/apache/parquet-format/blob/master/README.md>
//!
//! ## DataPage v1 layout
//!
//! ```text
//! [repetition levels][definition levels][values]
//! ```
//!
//! Level encoding is controlled by the DataPageHeader encoding fields:
//! - Encoding::RLE (3): 4-byte LE uint32 length prefix + RLE/bit-packing hybrid data.
//! - Encoding::BIT_PACKED (4): raw bit-packed data without a length prefix.
//!   Byte count = ceil(num_values * bit_width / 8).
//! If `max_rep_level == 0` no repetition level bytes are present.
//! If `max_def_level == 0` no definition level bytes are present.
//!
//! ## DataPage v2 layout
//!
//! ```text
//! [repetition levels: rep_byte_len bytes]
//! [definition levels: def_byte_len bytes]
//! [values: remaining bytes]
//! ```
//!
//! Byte lengths come from `DataPageHeaderV2::repetition_levels_byte_length` and
//! `definition_levels_byte_length`. Levels are always RLE/bit-packing hybrid,
//! never compressed. Values may be compressed.
//!
//! ## Invariants
//!
//! - `PagePayload::rep_levels.len() == 0` when `max_rep_level == 0`.
//! - `PagePayload::def_levels.len() == 0` when `max_def_level == 0`.
//! - `PagePayload::values_bytes` starts immediately after the last level byte.

use alloc::string::String;
use alloc::vec::Vec;
use consus_core::{Error, Result};

use super::page::{DataPageHeader, DataPageHeaderV2};
use crate::encoding::levels::{decode_bit_packed_raw, decode_levels, level_bit_width};

/// Parquet encoding discriminant constants (parquet.thrift Encoding enum).
const ENCODING_RLE: i32 = 3;
const ENCODING_BIT_PACKED: i32 = 4;

/// Decoded components of a Parquet data page payload.
///
/// `values_bytes` is a sub-slice of the original page buffer; it is not copied.
#[derive(Debug)]
pub struct PagePayload<'a> {
    /// Decoded repetition levels (empty when max_rep_level == 0).
    pub rep_levels: Vec<i32>,
    /// Decoded definition levels (empty when max_def_level == 0).
    pub def_levels: Vec<i32>,
    /// Raw bytes of the encoded values section (encoding determined by page header).
    pub values_bytes: &'a [u8],
    /// Number of values as declared in the page header.
    pub num_values: i32,
}

/// Decode levels for DataPage v1 with RLE or BIT_PACKED encoding.
///
/// Advances `pos` past the consumed bytes.
fn decode_levels_v1(
    bytes: &[u8],
    pos: &mut usize,
    bit_width: u8,
    count: usize,
    encoding: i32,
) -> Result<Vec<i32>> {
    match encoding {
        ENCODING_RLE => {
            // 4-byte LE uint32 length prefix.
            if *pos + 4 > bytes.len() {
                return Err(Error::BufferTooSmall {
                    required: *pos + 4,
                    provided: bytes.len(),
                });
            }
            let len = u32::from_le_bytes([
                bytes[*pos],
                bytes[*pos + 1],
                bytes[*pos + 2],
                bytes[*pos + 3],
            ]) as usize;
            *pos += 4;
            if *pos + len > bytes.len() {
                return Err(Error::BufferTooSmall {
                    required: *pos + len,
                    provided: bytes.len(),
                });
            }
            let level_bytes = &bytes[*pos..*pos + len];
            *pos += len;
            decode_levels(level_bytes, bit_width, count)
        }
        ENCODING_BIT_PACKED => {
            // Raw bit-packed, no length prefix.
            // Byte count = ceil(count * bit_width / 8).
            let required_bytes = (count * bit_width as usize + 7) / 8;
            if *pos + required_bytes > bytes.len() {
                return Err(Error::BufferTooSmall {
                    required: *pos + required_bytes,
                    provided: bytes.len(),
                });
            }
            let level_bytes = &bytes[*pos..*pos + required_bytes];
            *pos += required_bytes;
            decode_bit_packed_raw(level_bytes, bit_width, count)
        }
        _ => Err(Error::InvalidFormat {
            message: String::from("parquet: unsupported level encoding for DataPage v1"),
        }),
    }
}

/// Split a DataPage v1 payload into repetition levels, definition levels,
/// and value bytes.
///
/// `max_rep_level` and `max_def_level` are the maximum level values for the
/// column as derived from the schema nesting depth. Pass 0 for required,
/// non-nested columns.
///
/// The `bytes` argument is the raw page payload (not including the page header).
pub fn split_data_page_v1<'a>(
    bytes: &'a [u8],
    header: &DataPageHeader,
    max_rep_level: i32,
    max_def_level: i32,
) -> Result<PagePayload<'a>> {
    let count = header.num_values as usize;
    let mut pos: usize = 0;

    let rep_levels = if max_rep_level == 0 {
        Vec::new()
    } else {
        let bit_width = level_bit_width(max_rep_level);
        decode_levels_v1(bytes, &mut pos, bit_width, count, header.repetition_level_encoding)?
    };

    let def_levels = if max_def_level == 0 {
        Vec::new()
    } else {
        let bit_width = level_bit_width(max_def_level);
        decode_levels_v1(bytes, &mut pos, bit_width, count, header.definition_level_encoding)?
    };

    Ok(PagePayload {
        rep_levels,
        def_levels,
        values_bytes: &bytes[pos..],
        num_values: header.num_values,
    })
}

/// Split a DataPage v2 payload into repetition levels, definition levels,
/// and value bytes.
///
/// For DataPage v2, the level byte lengths are provided in the header and
/// levels are always encoded as RLE/bit-packing hybrid (never compressed).
/// Value bytes may be compressed; callers must decompress before value decoding.
pub fn split_data_page_v2<'a>(
    bytes: &'a [u8],
    header: &DataPageHeaderV2,
    max_rep_level: i32,
    max_def_level: i32,
) -> Result<PagePayload<'a>> {
    let count = header.num_values as usize;
    let rep_len = header.repetition_levels_byte_length as usize;
    let def_len = header.definition_levels_byte_length as usize;
    let mut pos: usize = 0;

    let rep_levels = {
        if pos + rep_len > bytes.len() {
            return Err(Error::BufferTooSmall {
                required: pos + rep_len,
                provided: bytes.len(),
            });
        }
        let level_bytes = &bytes[pos..pos + rep_len];
        pos += rep_len;
        if max_rep_level == 0 || rep_len == 0 {
            Vec::new()
        } else {
            let bit_width = level_bit_width(max_rep_level);
            decode_levels(level_bytes, bit_width, count)?
        }
    };

    let def_levels = {
        if pos + def_len > bytes.len() {
            return Err(Error::BufferTooSmall {
                required: pos + def_len,
                provided: bytes.len(),
            });
        }
        let level_bytes = &bytes[pos..pos + def_len];
        pos += def_len;
        if max_def_level == 0 || def_len == 0 {
            Vec::new()
        } else {
            let bit_width = level_bit_width(max_def_level);
            decode_levels(level_bytes, bit_width, count)?
        }
    };

    Ok(PagePayload {
        rep_levels,
        def_levels,
        values_bytes: &bytes[pos..],
        num_values: header.num_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::page::{DataPageHeader, DataPageHeaderV2};

    fn make_data_page_header(num_values: i32, encoding: i32, def_enc: i32, rep_enc: i32) -> DataPageHeader {
        DataPageHeader {
            num_values,
            encoding,
            definition_level_encoding: def_enc,
            repetition_level_encoding: rep_enc,
        }
    }

    fn make_data_page_header_v2(
        num_values: i32,
        rep_len: i32,
        def_len: i32,
    ) -> DataPageHeaderV2 {
        DataPageHeaderV2 {
            num_values,
            num_nulls: 0,
            num_rows: num_values,
            encoding: 0,
            definition_levels_byte_length: def_len,
            repetition_levels_byte_length: rep_len,
            is_compressed: Some(false),
        }
    }

    /// Required, non-nullable column: max_rep=0, max_def=0.
    /// The entire payload is value bytes. No level bytes consumed.
    ///
    /// 5 INT32 values: [1, 2, 3, 4, 5]
    #[test]
    fn split_v1_required_column_no_levels() {
        let values: &[u8] = &[
            0x01, 0x00, 0x00, 0x00,  // 1
            0x02, 0x00, 0x00, 0x00,  // 2
            0x03, 0x00, 0x00, 0x00,  // 3
            0x04, 0x00, 0x00, 0x00,  // 4
            0x05, 0x00, 0x00, 0x00,  // 5
        ];
        let header = make_data_page_header(5, 0, 0, 0);
        let payload = split_data_page_v1(values, &header, 0, 0).unwrap();
        assert_eq!(payload.rep_levels, vec![]);
        assert_eq!(payload.def_levels, vec![]);
        assert_eq!(payload.values_bytes, values);
        assert_eq!(payload.num_values, 5);
    }

    /// Optional column: max_rep=0, max_def=1.
    /// Definition levels encoded as RLE (3) with bit_width=1.
    ///
    /// 5 values, def_levels=[1,0,1,1,0]:
    ///   RLE run 1: (1<<1)|0=0x02, val=0x01  → emit 1x 1
    ///   RLE run 2: (1<<1)|0=0x02, val=0x00  → emit 1x 0
    ///   RLE run 3: (2<<1)|0=0x04, val=0x01  → emit 2x 1
    ///   RLE run 4: (1<<1)|0=0x02, val=0x00  → emit 1x 0
    /// RLE data = [0x02,0x01, 0x02,0x00, 0x04,0x01, 0x02,0x00] = 8 bytes
    /// Length prefix (LE u32): [0x08,0x00,0x00,0x00]
    /// def level section = [0x08,0x00,0x00,0x00, 0x02,0x01,0x02,0x00,0x04,0x01,0x02,0x00]
    /// Values: 3 non-null INT32 = [1, 2, 3]
    #[test]
    fn split_v1_optional_column_rle_def_levels() {
        let def_section: &[u8] = &[
            0x08, 0x00, 0x00, 0x00,  // length = 8
            0x02, 0x01,              // RLE: 1x value=1
            0x02, 0x00,              // RLE: 1x value=0
            0x04, 0x01,              // RLE: 2x value=1
            0x02, 0x00,              // RLE: 1x value=0
        ];
        let value_bytes: &[u8] = &[
            0x01, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00,
        ];
        let mut payload_bytes = Vec::from(def_section);
        payload_bytes.extend_from_slice(value_bytes);

        let header = make_data_page_header(5, 0, 3 /*RLE*/, 0);
        let payload = split_data_page_v1(&payload_bytes, &header, 0, 1).unwrap();

        assert_eq!(payload.rep_levels, vec![]);
        assert_eq!(payload.def_levels, vec![1, 0, 1, 1, 0]);
        assert_eq!(payload.values_bytes, value_bytes);
        assert_eq!(payload.num_values, 5);
    }

    /// DataPage v2: required column with zero-length level sections.
    /// rep_len=0, def_len=0 — the entire payload is values.
    #[test]
    fn split_v2_required_column_zero_level_lengths() {
        let values: &[u8] = &[
            0x01, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00,
        ];
        let header = make_data_page_header_v2(3, 0, 0);
        let payload = split_data_page_v2(values, &header, 0, 0).unwrap();
        assert_eq!(payload.rep_levels, vec![]);
        assert_eq!(payload.def_levels, vec![]);
        assert_eq!(payload.values_bytes, values);
        assert_eq!(payload.num_values, 3);
    }

    /// DataPage v2: optional column with def levels.
    /// 4 values, max_def_level=1, def_levels=[1,1,0,1].
    ///
    /// bit_width = level_bit_width(1) = 1
    /// RLE run: 2x value=1 -> header=(2<<1)|0=0x04, val=0x01 -> [0x04, 0x01]
    /// RLE run: 1x value=0 -> header=(1<<1)|0=0x02, val=0x00 -> [0x02, 0x00]
    /// RLE run: 1x value=1 -> header=(1<<1)|0=0x02, val=0x01 -> [0x02, 0x01]
    /// def_bytes = [0x04,0x01, 0x02,0x00, 0x02,0x01] = 6 bytes
    #[test]
    fn split_v2_optional_column_def_levels() {
        let def_rle: &[u8] = &[0x04, 0x01, 0x02, 0x00, 0x02, 0x01];
        let value_bytes: &[u8] = &[0xAA, 0xBB, 0xCC];
        let mut payload = Vec::from(def_rle);
        payload.extend_from_slice(value_bytes);

        let header = make_data_page_header_v2(4, 0, 6 /*def_len*/);
        let result = split_data_page_v2(&payload, &header, 0, 1).unwrap();
        assert_eq!(result.rep_levels, vec![]);
        assert_eq!(result.def_levels, vec![1, 1, 0, 1]);
        assert_eq!(result.values_bytes, value_bytes);
    }

    /// split_v1 with unsupported encoding returns InvalidFormat.
    #[test]
    fn split_v1_unsupported_level_encoding_errors() {
        let payload = &[0x00u8; 10];
        let header = make_data_page_header(5, 0, 99 /*unknown*/, 0);
        let err = split_data_page_v1(payload, &header, 0, 1).unwrap_err();
        assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
    }

    /// split_v2 errors when payload is too short for rep_len bytes.
    #[test]
    fn split_v2_truncated_rep_level_section_errors() {
        let payload = &[0x00u8; 3]; // rep_len=5 but only 3 bytes available
        let header = make_data_page_header_v2(4, 5 /*rep_len*/, 0);
        let err = split_data_page_v2(payload, &header, 1, 0).unwrap_err();
        assert!(matches!(err, consus_core::Error::BufferTooSmall { .. }));
    }
}
