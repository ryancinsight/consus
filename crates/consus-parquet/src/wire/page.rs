//! Parquet page header types and Thrift compact binary decoder.
//!
//! ## Specification
//!
//! Decodes PageHeader and all nested header structs from the Thrift compact
//! binary encoding defined in parquet.thrift.
//!
//! ## Encoding reference
//!
//! - I32 type code: 0x05; field header byte: `(delta << 4) | 0x05`
//! - STRUCT type code: 0x0C
//! - BOOL_TRUE = 0x01, BOOL_FALSE = 0x02 (value in type nibble, no extra read)
//! - Field stop: 0x00
//!
//! ## Invariants
//!
//! - `decode_page_header` returns `consumed == reader.position()` after decoding.
//! - `bytes[consumed..]` is the start of the page data payload.

use alloc::string::String;

use consus_core::{Error, Result};

use super::thrift::ThriftReader;

/// Parquet page type discriminant.
///
/// Matches the `PageType` enum in parquet.thrift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageType {
    DataPage = 0,
    IndexPage = 1,
    DictionaryPage = 2,
    DataPageV2 = 3,
}

impl PageType {
    /// Decode a raw `i32` Thrift enum value.
    ///
    /// Returns `None` for any value not defined in parquet.thrift.
    #[inline]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::DataPage),
            1 => Some(Self::IndexPage),
            2 => Some(Self::DictionaryPage),
            3 => Some(Self::DataPageV2),
            _ => None,
        }
    }
}

/// Decoded DataPageHeader (parquet.thrift field 5 of PageHeader).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPageHeader {
    pub num_values: i32,
    pub encoding: i32,
    pub definition_level_encoding: i32,
    pub repetition_level_encoding: i32,
}

/// Decoded DictionaryPageHeader (parquet.thrift field 7 of PageHeader).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictionaryPageHeader {
    pub num_values: i32,
    pub encoding: i32,
    /// Value encoded in the Thrift field type nibble: 0x01=true, 0x02=false.
    pub is_sorted: Option<bool>,
}

/// Decoded DataPageHeaderV2 (parquet.thrift field 8 of PageHeader).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPageHeaderV2 {
    pub num_values: i32,
    pub num_nulls: i32,
    pub num_rows: i32,
    pub encoding: i32,
    pub definition_levels_byte_length: i32,
    pub repetition_levels_byte_length: i32,
    /// Value encoded in the Thrift field type nibble: 0x01=true, 0x02=false.
    pub is_compressed: Option<bool>,
}

/// Decoded top-level Parquet page header.
#[derive(Debug, Clone, PartialEq)]
pub struct PageHeader {
    pub type_: PageType,
    pub uncompressed_page_size: i32,
    pub compressed_page_size: i32,
    pub crc: Option<i32>,
    pub data_page_header: Option<DataPageHeader>,
    pub dictionary_page_header: Option<DictionaryPageHeader>,
    pub data_page_header_v2: Option<DataPageHeaderV2>,
}

fn decode_data_page_header(r: &mut ThriftReader<'_>) -> Result<DataPageHeader> {
    let (mut nv, mut enc, mut def, mut rep) = (None, None, None, None);
    let mut last = 0i16;
    loop {
        match r.read_field_header(&mut last)? {
            None            => break,
            Some((1, 0x05)) => nv  = Some(r.read_i32()?),
            Some((2, 0x05)) => enc = Some(r.read_i32()?),
            Some((3, 0x05)) => def = Some(r.read_i32()?),
            Some((4, 0x05)) => rep = Some(r.read_i32()?),
            Some((_, tc))   => r.skip(tc)?,
        }
    }
    Ok(DataPageHeader {
        num_values:                nv.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DataPageHeader: field 1 (num_values) missing"),
        })?,
        encoding:                 enc.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DataPageHeader: field 2 (encoding) missing"),
        })?,
        definition_level_encoding: def.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DataPageHeader: field 3 (definition_level_encoding) missing"),
        })?,
        repetition_level_encoding: rep.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DataPageHeader: field 4 (repetition_level_encoding) missing"),
        })?,
    })
}

fn decode_dictionary_page_header(r: &mut ThriftReader<'_>) -> Result<DictionaryPageHeader> {
    let (mut nv, mut enc, mut is_sorted) = (None, None, None);
    let mut last = 0i16;
    loop {
        match r.read_field_header(&mut last)? {
            None            => break,
            Some((1, 0x05)) => nv  = Some(r.read_i32()?),
            Some((2, 0x05)) => enc = Some(r.read_i32()?),
            Some((3, 0x01)) => is_sorted = Some(true),
            Some((3, 0x02)) => is_sorted = Some(false),
            Some((_, tc))   => r.skip(tc)?,
        }
    }
    Ok(DictionaryPageHeader {
        num_values: nv.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DictionaryPageHeader: field 1 (num_values) missing"),
        })?,
        encoding:   enc.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DictionaryPageHeader: field 2 (encoding) missing"),
        })?,
        is_sorted,
    })
}

fn decode_data_page_header_v2(r: &mut ThriftReader<'_>) -> Result<DataPageHeaderV2> {
    let (mut nv, mut nnull, mut nr, mut enc, mut dbl, mut rbl, mut isc) =
        (None, None, None, None, None, None, None);
    let mut last = 0i16;
    loop {
        match r.read_field_header(&mut last)? {
            None            => break,
            Some((1, 0x05)) => nv    = Some(r.read_i32()?),
            Some((2, 0x05)) => nnull = Some(r.read_i32()?),
            Some((3, 0x05)) => nr    = Some(r.read_i32()?),
            Some((4, 0x05)) => enc   = Some(r.read_i32()?),
            Some((5, 0x05)) => dbl   = Some(r.read_i32()?),
            Some((6, 0x05)) => rbl   = Some(r.read_i32()?),
            Some((7, 0x01)) => isc   = Some(true),
            Some((7, 0x02)) => isc   = Some(false),
            Some((_, tc))   => r.skip(tc)?,
        }
    }
    Ok(DataPageHeaderV2 {
        num_values:                    nv.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DataPageHeaderV2: field 1 (num_values) missing"),
        })?,
        num_nulls:                     nnull.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DataPageHeaderV2: field 2 (num_nulls) missing"),
        })?,
        num_rows:                      nr.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DataPageHeaderV2: field 3 (num_rows) missing"),
        })?,
        encoding:                      enc.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DataPageHeaderV2: field 4 (encoding) missing"),
        })?,
        definition_levels_byte_length: dbl.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DataPageHeaderV2: field 5 (definition_levels_byte_length) missing"),
        })?,
        repetition_levels_byte_length: rbl.ok_or_else(|| Error::InvalidFormat {
            message: String::from("DataPageHeaderV2: field 6 (repetition_levels_byte_length) missing"),
        })?,
        is_compressed: isc,
    })
}

fn decode_page_header_inner(r: &mut ThriftReader<'_>) -> Result<PageHeader> {
    let (mut ty, mut usz, mut csz, mut crc) = (None, None, None, None);
    let (mut dph, mut dicph, mut v2ph) = (None, None, None);
    let mut last = 0i16;
    loop {
        match r.read_field_header(&mut last)? {
            None => break,
            Some((1, 0x05)) => {
                let v = r.read_i32()?;
                ty = Some(PageType::from_i32(v).ok_or_else(|| Error::InvalidFormat {
                    message: String::from("parquet: unknown PageType discriminant"),
                })?);
            }
            Some((2, 0x05)) => usz  = Some(r.read_i32()?),
            Some((3, 0x05)) => csz  = Some(r.read_i32()?),
            Some((4, 0x05)) => crc  = Some(r.read_i32()?),
            Some((5, 0x0C)) => dph  = Some(decode_data_page_header(r)?),
            Some((7, 0x0C)) => dicph = Some(decode_dictionary_page_header(r)?),
            Some((8, 0x0C)) => v2ph = Some(decode_data_page_header_v2(r)?),
            Some((_, tc))   => r.skip(tc)?,
        }
    }
    Ok(PageHeader {
        type_: ty.ok_or_else(|| Error::InvalidFormat {
            message: String::from("parquet: PageHeader missing type_"),
        })?,
        uncompressed_page_size: usz.ok_or_else(|| Error::InvalidFormat {
            message: String::from("parquet: PageHeader missing uncompressed_page_size"),
        })?,
        compressed_page_size: csz.ok_or_else(|| Error::InvalidFormat {
            message: String::from("parquet: PageHeader missing compressed_page_size"),
        })?,
        crc,
        data_page_header:       dph,
        dictionary_page_header: dicph,
        data_page_header_v2:    v2ph,
    })
}

/// Decode a Parquet page header from `bytes`, returning (header, bytes_consumed).
pub fn decode_page_header(bytes: &[u8]) -> Result<(PageHeader, usize)> {
    let mut r = ThriftReader::new(bytes);
    let header = decode_page_header_inner(&mut r)?;
    Ok((header, r.position()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_full_page_header_data_page() {
        // PageHeader: type_=DATA_PAGE=0, uncompressed=40, compressed=40, data_page_header
        // Field 1 (I32, delta=1): 0x15, 0x00  [type_=0, zigzag(0)=0]
        // Field 2 (I32, delta=1): 0x15, 0x50  [uncompressed=40, zigzag(80)=40]
        // Field 3 (I32, delta=1): 0x15, 0x50  [compressed=40]
        // Field 5 (STRUCT, delta=2): 0x2C
        //   Field 1..4 of DataPageHeader, then stop 0x00
        // PageHeader stop: 0x00
        let bytes = [
            0x15u8, 0x00,  // field 1: type_=0
            0x15, 0x50,    // field 2: uncompressed=40
            0x15, 0x50,    // field 3: compressed=40
            0x2C,          // field 5: STRUCT (delta=2 from 3)
            0x15, 0x0A,    // DPH field 1: num_values=5
            0x15, 0x00,    // DPH field 2: encoding=0
            0x15, 0x00,    // DPH field 3: def_level=0
            0x15, 0x00,    // DPH field 4: rep_level=0
            0x00,          // DPH stop
            0x00,          // PageHeader stop
        ];
        let (header, consumed) = decode_page_header(&bytes).unwrap();
        assert_eq!(header.type_, PageType::DataPage);
        assert_eq!(header.uncompressed_page_size, 40);
        assert_eq!(header.compressed_page_size, 40);
        assert_eq!(header.crc, None);
        assert!(header.data_page_header.is_some());
        assert_eq!(header.data_page_header.as_ref().unwrap().num_values, 5);
        assert_eq!(consumed, 17);
    }

    #[test]
    fn decode_dictionary_page_header_with_sorted() {
        // DictionaryPageHeader: num_values=256, encoding=2, is_sorted=true
        // zigzag(256)=512; varint(512): 0x80, 0x04
        // Field 1 (I32, delta=1): 0x15, 0x80, 0x04  [num_values=256]
        // Field 2 (I32, delta=1): 0x15, 0x04         [encoding=2, zigzag(4)=2]
        // Field 3 (BOOL_TRUE, delta=1): 0x11          [is_sorted=true]
        // Stop: 0x00
        let bytes = [0x15u8, 0x80, 0x04, 0x15, 0x04, 0x11, 0x00];
        let mut r = super::super::thrift::ThriftReader::new(&bytes);
        let dph = decode_dictionary_page_header(&mut r).unwrap();
        assert_eq!(dph.num_values, 256);
        assert_eq!(dph.encoding, 2);
        assert_eq!(dph.is_sorted, Some(true));
    }

    #[test]
    fn decode_page_header_rejects_empty() {
        let err = decode_page_header(&[]).unwrap_err();
        assert!(matches!(err, consus_core::Error::BufferTooSmall { .. }));
    }

    #[test]
    fn decode_data_page_header_v2_minimal() {
        // num_values=10, num_nulls=0, num_rows=10, encoding=0, def_byte_len=20, rep_byte_len=0
        let bytes = [
            0x15u8, 0x14, // field 1: num_values=10, zigzag(20)=10
            0x15, 0x00,   // field 2: num_nulls=0
            0x15, 0x14,   // field 3: num_rows=10
            0x15, 0x00,   // field 4: encoding=0
            0x15, 0x28,   // field 5: def_byte_len=20, zigzag(40)=20
            0x15, 0x00,   // field 6: rep_byte_len=0
            0x00,         // stop
        ];
        let mut r = super::super::thrift::ThriftReader::new(&bytes);
        let v2 = decode_data_page_header_v2(&mut r).unwrap();
        assert_eq!(v2.num_values, 10);
        assert_eq!(v2.num_rows, 10);
        assert_eq!(v2.definition_levels_byte_length, 20);
        assert_eq!(v2.is_compressed, None);
    }

    #[test]
    fn page_type_from_i32_covers_all_variants() {
        assert_eq!(PageType::from_i32(0), Some(PageType::DataPage));
        assert_eq!(PageType::from_i32(1), Some(PageType::IndexPage));
        assert_eq!(PageType::from_i32(2), Some(PageType::DictionaryPage));
        assert_eq!(PageType::from_i32(3), Some(PageType::DataPageV2));
        assert_eq!(PageType::from_i32(99), None);
    }
}
