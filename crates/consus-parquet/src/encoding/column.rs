//! Typed column value extraction.
//! PLAIN(0), PLAIN_DICTIONARY(2), RLE_DICTIONARY(8).

use super::compression::{CompressionCodec, decompress_page_values};
use super::plain::{
    decode_plain_boolean, decode_plain_byte_array, decode_plain_f32, decode_plain_f64,
    decode_plain_fixed_byte_array, decode_plain_i32, decode_plain_i64, decode_plain_i96,
};
use super::rle_dict::decode_rle_dict_indices;
use crate::schema::physical::ParquetPhysicalType;
use crate::wire::page::DictionaryPageHeader;
use alloc::{format, string::String, vec::Vec};
use consus_core::{Error, Result};

const ENCODING_PLAIN: i32 = 0;
const ENCODING_PLAIN_DICTIONARY: i32 = 2;
const ENCODING_RLE_DICTIONARY: i32 = 8;

/// Typed column values for all Parquet physical types.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnValues {
    Boolean(Vec<bool>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Int96(Vec<[u8; 12]>),
    Float(Vec<f32>),
    Double(Vec<f64>),
    ByteArray(Vec<Vec<u8>>),
    FixedLenByteArray {
        fixed_len: usize,
        values: Vec<Vec<u8>>,
    },
}

impl ColumnValues {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Boolean(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::Int96(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::Double(v) => v.len(),
            Self::ByteArray(v) => v.len(),
            Self::FixedLenByteArray { values, .. } => values.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn physical_type(&self) -> ParquetPhysicalType {
        match self {
            Self::Boolean(_) => ParquetPhysicalType::Boolean,
            Self::Int32(_) => ParquetPhysicalType::Int32,
            Self::Int64(_) => ParquetPhysicalType::Int64,
            Self::Int96(_) => ParquetPhysicalType::Int96,
            Self::Float(_) => ParquetPhysicalType::Float,
            Self::Double(_) => ParquetPhysicalType::Double,
            Self::ByteArray(_) => ParquetPhysicalType::ByteArray,
            Self::FixedLenByteArray { fixed_len, .. } => {
                ParquetPhysicalType::FixedLenByteArray(*fixed_len)
            }
        }
    }
}

/// Column values with associated Dremel repetition and definition levels.
///
/// `values` contains only the non-null (defined) leaf values.
/// `rep_levels` and `def_levels` each have one entry per logical value position
/// (including null / empty-list positions), matching the Dremel paper's column
/// representation.
///
/// For required flat columns (`max_rep=0`, `max_def=0`), both level vectors are
/// empty and `values.len() == total_value_count`.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnValuesWithLevels {
    /// Non-null leaf values (one entry per position where `def_level == max_def_level`).
    pub values: ColumnValues,
    /// Repetition levels (empty when `max_rep_level == 0`).
    pub rep_levels: Vec<i32>,
    /// Definition levels (empty when `max_def_level == 0`).
    pub def_levels: Vec<i32>,
    /// Maximum repetition level for the column.
    pub max_rep_level: i32,
    /// Maximum definition level for the column.
    pub max_def_level: i32,
}

impl ColumnValuesWithLevels {
    /// Total logical value count, including null / empty-list positions.
    ///
    /// Equal to `def_levels.len()` when `max_def_level > 0`, or `values.len()`
    /// for required columns.
    #[must_use]
    pub fn total_count(&self) -> usize {
        if self.max_def_level > 0 {
            self.def_levels.len()
        } else {
            self.values.len()
        }
    }

    /// Count of non-null (defined) values.
    ///
    /// Equal to `values.len()` by construction.
    #[must_use]
    pub fn non_null_count(&self) -> usize {
        self.values.len()
    }
}

fn decode_plain_column(bytes: &[u8], n: usize, pt: ParquetPhysicalType) -> Result<ColumnValues> {
    match pt {
        ParquetPhysicalType::Boolean => Ok(ColumnValues::Boolean(decode_plain_boolean(bytes, n)?)),
        ParquetPhysicalType::Int32 => Ok(ColumnValues::Int32(decode_plain_i32(bytes, n)?)),
        ParquetPhysicalType::Int64 => Ok(ColumnValues::Int64(decode_plain_i64(bytes, n)?)),
        ParquetPhysicalType::Int96 => Ok(ColumnValues::Int96(decode_plain_i96(bytes, n)?)),
        ParquetPhysicalType::Float => Ok(ColumnValues::Float(decode_plain_f32(bytes, n)?)),
        ParquetPhysicalType::Double => Ok(ColumnValues::Double(decode_plain_f64(bytes, n)?)),
        ParquetPhysicalType::ByteArray => {
            Ok(ColumnValues::ByteArray(decode_plain_byte_array(bytes, n)?))
        }
        ParquetPhysicalType::FixedLenByteArray(fl) => Ok(ColumnValues::FixedLenByteArray {
            fixed_len: fl,
            values: decode_plain_fixed_byte_array(bytes, n, fl)?,
        }),
    }
}

fn lookup_indices<T: Clone>(dict: &[T], indices: &[i32]) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx < 0 {
            return Err(Error::InvalidFormat {
                message: format!("parquet: negative dictionary index {idx}"),
            });
        }
        let i = idx as usize;
        if i >= dict.len() {
            return Err(Error::InvalidFormat {
                message: format!(
                    "parquet: dictionary index {i} out of bounds (size {})",
                    dict.len()
                ),
            });
        }
        out.push(dict[i].clone());
    }
    Ok(out)
}

fn apply_dict_lookup(dict: &ColumnValues, indices: &[i32]) -> Result<ColumnValues> {
    match dict {
        ColumnValues::Boolean(v) => Ok(ColumnValues::Boolean(lookup_indices(v, indices)?)),
        ColumnValues::Int32(v) => Ok(ColumnValues::Int32(lookup_indices(v, indices)?)),
        ColumnValues::Int64(v) => Ok(ColumnValues::Int64(lookup_indices(v, indices)?)),
        ColumnValues::Int96(v) => Ok(ColumnValues::Int96(lookup_indices(v, indices)?)),
        ColumnValues::Float(v) => Ok(ColumnValues::Float(lookup_indices(v, indices)?)),
        ColumnValues::Double(v) => Ok(ColumnValues::Double(lookup_indices(v, indices)?)),
        ColumnValues::ByteArray(v) => Ok(ColumnValues::ByteArray(lookup_indices(v, indices)?)),
        ColumnValues::FixedLenByteArray { fixed_len, values } => {
            Ok(ColumnValues::FixedLenByteArray {
                fixed_len: *fixed_len,
                values: lookup_indices(values, indices)?,
            })
        }
    }
}

/// Decode a dictionary page payload into typed ColumnValues.
/// Dictionary pages always use PLAIN encoding; num_values from header.
pub fn decode_dictionary_page(
    data: &[u8],
    header: &DictionaryPageHeader,
    physical_type: ParquetPhysicalType,
) -> Result<ColumnValues> {
    decode_plain_column(data, header.num_values as usize, physical_type)
}

/// Decode a data page values section into typed ColumnValues.
/// encoding: 0=PLAIN, 2=PLAIN_DICTIONARY, 8=RLE_DICTIONARY.
///
/// `values_bytes` must already be decompressed. For compressed pages, use
/// [`decode_compressed_column_values`] instead.
pub fn decode_column_values(
    values_bytes: &[u8],
    num_values: usize,
    encoding: i32,
    physical_type: ParquetPhysicalType,
    dictionary: Option<&ColumnValues>,
) -> Result<ColumnValues> {
    match encoding {
        ENCODING_PLAIN => decode_plain_column(values_bytes, num_values, physical_type),
        ENCODING_PLAIN_DICTIONARY | ENCODING_RLE_DICTIONARY => {
            let dict = dictionary.ok_or_else(|| Error::InvalidFormat {
                message: String::from("parquet: dictionary encoding requires a dictionary"),
            })?;
            if dict.physical_type() != physical_type {
                return Err(Error::DatatypeMismatch {
                    expected: format!("{physical_type:?}"),
                    found: format!("{:?}", dict.physical_type()),
                });
            }
            let indices = decode_rle_dict_indices(values_bytes, num_values)?;
            apply_dict_lookup(dict, &indices)
        }
        _ => Err(Error::UnsupportedFeature {
            feature: format!("parquet encoding ID {encoding}"),
        }),
    }
}

/// Decompress (if needed) and decode a data page values section into typed
/// [`ColumnValues`].
///
/// Integrated entry point combining decompression and value decoding. Use when
/// page values may be compressed (codec != UNCOMPRESSED). For already-decompressed
/// bytes, [`decode_column_values`] avoids the extra allocation.
///
/// ## Errors
///
/// Propagates `Error::CompressionError` for malformed compressed data,
/// `Error::UnsupportedFeature` for unknown codecs or encodings, and any
/// value-decoding errors from [`decode_column_values`].
pub fn decode_compressed_column_values(
    values_bytes: &[u8],
    num_values: usize,
    encoding: i32,
    physical_type: ParquetPhysicalType,
    dictionary: Option<&ColumnValues>,
    codec: CompressionCodec,
    uncompressed_size: usize,
) -> Result<ColumnValues> {
    let decompressed = decompress_page_values(values_bytes, codec, uncompressed_size)?;
    decode_column_values(
        &decompressed,
        num_values,
        encoding,
        physical_type,
        dictionary,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict_header(n: i32) -> DictionaryPageHeader {
        DictionaryPageHeader {
            num_values: n,
            encoding: 0,
            is_sorted: None,
        }
    }

    #[test]
    fn decode_column_plain_i32_three_values() {
        let b = [
            0x01u8, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F,
        ];
        let v = decode_column_values(&b, 3, 0, ParquetPhysicalType::Int32, None).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v.physical_type(), ParquetPhysicalType::Int32);
        assert!(matches!(v, ColumnValues::Int32(ref x) if *x == alloc::vec![1,-1,i32::MAX]));
    }

    #[test]
    fn decode_column_plain_i64_two_values() {
        let b = [
            0x64u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF,
        ];
        let v = decode_column_values(&b, 2, 0, ParquetPhysicalType::Int64, None).unwrap();
        assert_eq!(v.len(), 2);
        assert!(matches!(v, ColumnValues::Int64(ref x) if *x == alloc::vec![100i64,-200i64]));
    }

    #[test]
    fn decode_column_plain_f64_two_values() {
        let b = [
            0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xE0, 0xBF,
        ];
        let v = decode_column_values(&b, 2, 0, ParquetPhysicalType::Double, None).unwrap();
        assert!(matches!(v, ColumnValues::Double(ref x) if *x == alloc::vec![1.0f64,-0.5f64]));
    }

    #[test]
    fn decode_column_plain_byte_array_two_strings() {
        let b = [
            0x02u8, 0x00, 0x00, 0x00, 0x61, 0x62, 0x03, 0x00, 0x00, 0x00, 0x78, 0x79, 0x7A,
        ];
        let v = decode_column_values(&b, 2, 0, ParquetPhysicalType::ByteArray, None).unwrap();
        assert_eq!(v.len(), 2);
        assert!(matches!(v, ColumnValues::ByteArray(ref x)
            if *x == alloc::vec![alloc::vec![0x61u8,0x62], alloc::vec![0x78,0x79,0x7A]]));
    }

    #[test]
    fn decode_column_plain_fixed_len_byte_array() {
        let b = [0x61u8, 0x62, 0x63, 0x64, 0x65, 0x66];
        let v = decode_column_values(&b, 2, 0, ParquetPhysicalType::FixedLenByteArray(3), None)
            .unwrap();
        assert_eq!(v.len(), 2);
        assert!(
            matches!(&v, ColumnValues::FixedLenByteArray { fixed_len:3, values }
            if *values == alloc::vec![alloc::vec![0x61u8,0x62,0x63], alloc::vec![0x64,0x65,0x66]])
        );
    }

    #[test]
    fn decode_column_plain_boolean_eight_values() {
        let v = decode_column_values(&[0x4Du8], 8, 0, ParquetPhysicalType::Boolean, None).unwrap();
        assert_eq!(v.len(), 8);
        assert!(matches!(&v, ColumnValues::Boolean(x)
            if *x == alloc::vec![true,false,true,true,false,false,true,false]));
    }

    #[test]
    fn decode_dictionary_page_i32_three_values() {
        let b = [
            0x0Au8, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00,
        ];
        let d = decode_dictionary_page(&b, &dict_header(3), ParquetPhysicalType::Int32).unwrap();
        assert_eq!(d.len(), 3);
        assert!(matches!(&d, ColumnValues::Int32(x) if *x == alloc::vec![10,20,30]));
    }

    #[test]
    fn decode_dictionary_page_byte_array_two_strings() {
        let b = [
            0x03u8, 0x00, 0x00, 0x00, 0x66, 0x6F, 0x6F, 0x03, 0x00, 0x00, 0x00, 0x62, 0x61, 0x72,
        ];
        let d =
            decode_dictionary_page(&b, &dict_header(2), ParquetPhysicalType::ByteArray).unwrap();
        assert_eq!(d.len(), 2);
        assert!(matches!(&d, ColumnValues::ByteArray(x)
            if *x == alloc::vec![alloc::vec![0x66u8,0x6F,0x6F], alloc::vec![0x62,0x61,0x72]]));
    }

    // indices [1,2,0,1] -> [20,30,10,20]; bit_width=2, 1 group header=0x01
    // byte0: v0=1(pos0=1,pos1=0) v1=2(pos2=0,pos3=1) v2=0(0) v3=1(pos6=1) = 1+8+64=73=0x49
    #[test]
    fn decode_column_rle_dict_i32_bit_packed() {
        let dict = ColumnValues::Int32(alloc::vec![10, 20, 30]);
        let b = [0x02u8, 0x01, 0x49, 0x00];
        let v = decode_column_values(&b, 4, 8, ParquetPhysicalType::Int32, Some(&dict)).unwrap();
        assert_eq!(v.len(), 4);
        assert!(matches!(&v, ColumnValues::Int32(x) if *x == alloc::vec![20,30,10,20]));
    }

    #[test]
    fn decode_column_plain_dict_i32_same_as_rle() {
        let dict = ColumnValues::Int32(alloc::vec![10, 20, 30]);
        let b = [0x02u8, 0x01, 0x49, 0x00];
        let v = decode_column_values(&b, 4, 2, ParquetPhysicalType::Int32, Some(&dict)).unwrap();
        assert!(matches!(&v, ColumnValues::Int32(x) if *x == alloc::vec![20,30,10,20]));
    }

    // 5 copies of index 2; bit_width=2 RLE: header=(5<<1)|0=0x0A value=0x02
    #[test]
    fn decode_column_rle_dict_rle_run_five_copies() {
        let dict = ColumnValues::Int32(alloc::vec![10, 20, 30]);
        let b = [0x02u8, 0x0A, 0x02];
        let v = decode_column_values(&b, 5, 8, ParquetPhysicalType::Int32, Some(&dict)).unwrap();
        assert!(matches!(&v, ColumnValues::Int32(x) if *x == alloc::vec![30,30,30,30,30]));
    }

    // indices [0,2,1] -> ["foo","baz","bar"]; bit_width=2
    // v0=0(00) v1=2(pos2=0,pos3=1) v2=1(pos4=1,pos5=0) -> byte0=0+8+16=24=0x18
    #[test]
    fn decode_column_rle_dict_byte_array_three_values() {
        let dict = ColumnValues::ByteArray(alloc::vec![
            alloc::vec![0x66u8, 0x6F, 0x6F],
            alloc::vec![0x62, 0x61, 0x72],
            alloc::vec![0x62, 0x61, 0x7A],
        ]);
        let b = [0x02u8, 0x01, 0x18, 0x00];
        let v =
            decode_column_values(&b, 3, 8, ParquetPhysicalType::ByteArray, Some(&dict)).unwrap();
        assert_eq!(v.len(), 3);
        assert!(matches!(&v, ColumnValues::ByteArray(x)
            if *x == alloc::vec![alloc::vec![0x66u8,0x6F,0x6F], alloc::vec![0x62,0x61,0x7A], alloc::vec![0x62,0x61,0x72]]));
    }

    #[test]
    fn decode_column_unsupported_encoding_returns_error() {
        let e = decode_column_values(&[], 0, 5, ParquetPhysicalType::Int32, None).unwrap_err();
        assert!(matches!(e, consus_core::Error::UnsupportedFeature { .. }));
    }

    #[test]
    fn decode_column_rle_dict_missing_dictionary_returns_error() {
        let e = decode_column_values(
            &[0x01u8, 0x02, 0x00],
            1,
            8,
            ParquetPhysicalType::Int32,
            None,
        )
        .unwrap_err();
        assert!(matches!(e, consus_core::Error::InvalidFormat { .. }));
    }

    #[test]
    fn decode_column_rle_dict_index_out_of_bounds_returns_error() {
        let dict = ColumnValues::Int32(alloc::vec![10, 20]);
        let e = decode_column_values(
            &[0x03u8, 0x02, 0x05],
            1,
            8,
            ParquetPhysicalType::Int32,
            Some(&dict),
        )
        .unwrap_err();
        assert!(matches!(e, consus_core::Error::InvalidFormat { .. }));
    }

    #[test]
    fn decode_column_rle_dict_type_mismatch_returns_error() {
        let dict = ColumnValues::Int64(alloc::vec![10i64, 20i64]);
        let e = decode_column_values(
            &[0x01u8, 0x02, 0x00],
            1,
            8,
            ParquetPhysicalType::Int32,
            Some(&dict),
        )
        .unwrap_err();
        assert!(matches!(e, consus_core::Error::DatatypeMismatch { .. }));
    }

    #[test]
    fn column_values_len_is_empty_consistent() {
        let empty = ColumnValues::Int32(alloc::vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        let one = ColumnValues::Int32(alloc::vec![42]);
        assert_eq!(one.len(), 1);
        assert!(!one.is_empty());
    }

    #[test]
    fn column_values_physical_type_all_variants() {
        assert_eq!(
            ColumnValues::Boolean(alloc::vec![]).physical_type(),
            ParquetPhysicalType::Boolean
        );
        assert_eq!(
            ColumnValues::Int32(alloc::vec![]).physical_type(),
            ParquetPhysicalType::Int32
        );
        assert_eq!(
            ColumnValues::Int64(alloc::vec![]).physical_type(),
            ParquetPhysicalType::Int64
        );
        assert_eq!(
            ColumnValues::Int96(alloc::vec![]).physical_type(),
            ParquetPhysicalType::Int96
        );
        assert_eq!(
            ColumnValues::Float(alloc::vec![]).physical_type(),
            ParquetPhysicalType::Float
        );
        assert_eq!(
            ColumnValues::Double(alloc::vec![]).physical_type(),
            ParquetPhysicalType::Double
        );
        assert_eq!(
            ColumnValues::ByteArray(alloc::vec![]).physical_type(),
            ParquetPhysicalType::ByteArray
        );
        assert_eq!(
            ColumnValues::FixedLenByteArray {
                fixed_len: 16,
                values: alloc::vec![]
            }
            .physical_type(),
            ParquetPhysicalType::FixedLenByteArray(16)
        );
    }
}

#[cfg(test)]
mod column_integration;
