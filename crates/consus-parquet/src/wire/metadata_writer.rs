//! Canonical Parquet FileMetaData types and Thrift compact binary decoders.
//!
//! ## Specification
//!
//! Decodes the Parquet FileMetaData and nested structs from the Thrift compact
//! binary encoding defined in parquet.thrift.
//!
//! ## Type codes (Thrift Compact Protocol)
//!
//! | Code | Meaning       |
//! |------|---------------|
//! | 0x04 | I16           |
//! | 0x05 | I32           |
//! | 0x06 | I64           |
//! | 0x08 | Binary/String |
//! | 0x09 | List          |
//! | 0x0C | Struct        |

use alloc::{string::String, vec::Vec};
use consus_core::{Error, Result};
use super::thrift::ThriftReader;

/// Parquet key-value metadata pair (parquet.thrift KeyValue).
#[derive(Debug, Clone, PartialEq)]
pub struct KeyValue {
    pub key: String,
    pub value: Option<String>,
}

/// Parquet schema element; leaf when type_ is Some, group when None.
///
/// Numeric field values match the parquet.thrift Type and ConvertedType enums.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaElement {
    pub type_: Option<i32>,
    pub type_length: Option<i32>,
    pub repetition_type: Option<i32>,
    pub name: String,
    pub num_children: Option<i32>,
    pub converted_type: Option<i32>,
    pub scale: Option<i32>,
    pub precision: Option<i32>,
    pub field_id: Option<i32>,
}

/// Physical storage metadata for one column chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnMetadata {
    pub type_: i32,
    pub encodings: Vec<i32>,
    pub path_in_schema: Vec<String>,
    pub codec: i32,
    pub num_values: i64,
    pub total_uncompressed_size: i64,
    pub total_compressed_size: i64,
    pub data_page_offset: i64,
    pub index_page_offset: Option<i64>,
    pub dictionary_page_offset: Option<i64>,
}

/// Column chunk location and optional inline metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnChunkMetadata {
    pub file_path: Option<String>,
    pub file_offset: i64,
    pub meta_data: Option<ColumnMetadata>,
}

/// Row group metadata including all contained column chunks.
#[derive(Debug, Clone, PartialEq)]
pub struct RowGroupMetadata {
    pub columns: Vec<ColumnChunkMetadata>,
    pub total_byte_size: i64,
    pub num_rows: i64,
    pub file_offset: Option<i64>,
    pub total_compressed_size: Option<i64>,
    pub ordinal: Option<i16>,
}

/// Top-level Parquet file metadata decoded from the footer payload.
#[derive(Debug, Clone, PartialEq)]
pub struct FileMetadata {
    pub version: i32,
    pub schema: Vec<SchemaElement>,
    pub num_rows: i64,
    pub row_groups: Vec<RowGroupMetadata>,
    pub key_value_metadata: Vec<KeyValue>,
    pub created_by: Option<String>,
}

#[inline]
fn missing(ctx: &'static str) -> Error {
    Error::InvalidFormat { message: String::from(ctx) }
}

fn decode_key_value(r: &mut ThriftReader<'_>) -> Result<KeyValue> {
    let (mut key, mut value) = (None, None);
    let mut last = 0i16;
    loop {
        match r.read_field_header(&mut last)? {
            None            => break,
            Some((1, 0x08)) => key   = Some(r.read_string()?),
            Some((2, 0x08)) => value = Some(r.read_string()?),
            Some((_, tc))   => r.skip(tc)?,
        }
    }
    Ok(KeyValue {
        key: key.ok_or_else(|| missing("parquet: KeyValue missing key"))?,
        value,
    })
}

fn decode_schema_element(r: &mut ThriftReader<'_>) -> Result<SchemaElement> {
    let (mut ty, mut tl, mut rep, mut name) = (None, None, None, None);
    let (mut nc, mut ct, mut sc, mut pr, mut fi) = (None, None, None, None, None);
    let mut last = 0i16;
    loop {
        match r.read_field_header(&mut last)? {
            None            => break,
            Some((1, 0x05)) => ty   = Some(r.read_i32()?),
            Some((2, 0x05)) => tl   = Some(r.read_i32()?),
            Some((3, 0x05)) => rep  = Some(r.read_i32()?),
            Some((4, 0x08)) => name = Some(r.read_string()?),
            Some((5, 0x05)) => nc   = Some(r.read_i32()?),
            Some((6, 0x05)) => ct   = Some(r.read_i32()?),
            Some((7, 0x05)) => sc   = Some(r.read_i32()?),
            Some((8, 0x05)) => pr   = Some(r.read_i32()?),
            Some((9, 0x05)) => fi   = Some(r.read_i32()?),
            Some((_, tc))   => r.skip(tc)?,
        }
    }
    Ok(SchemaElement {
        type_: ty, type_length: tl, repetition_type: rep,
        name: name.ok_or_else(|| missing("parquet: SchemaElement missing name"))?,
        num_children: nc, converted_type: ct, scale: sc, precision: pr, field_id: fi,
    })
}

fn decode_column_metadata(r: &mut ThriftReader<'_>) -> Result<ColumnMetadata> {
    let (mut ty, mut codec, mut nv, mut tus, mut tcs, mut dpo) =
        (None, None, None, None, None, None);
    let (mut ipo, mut dicpo) = (None, None);
    let (mut enc, mut path) = (Vec::new(), Vec::new());
    let mut last = 0i16;
    loop {
        match r.read_field_header(&mut last)? {
            None             => break,
            Some((1,  0x05)) => ty    = Some(r.read_i32()?),
            Some((2,  0x09)) => {
                let (et, n) = r.read_list_header()?;
                for _ in 0..n {
                    if et == 0x05 { enc.push(r.read_i32()?) } else { r.skip(et)? }
                }
            }
            Some((3,  0x09)) => {
                let (et, n) = r.read_list_header()?;
                for _ in 0..n {
                    if et == 0x08 { path.push(r.read_string()?) } else { r.skip(et)? }
                }
            }
            Some((4,  0x05)) => codec = Some(r.read_i32()?),
            Some((5,  0x06)) => nv    = Some(r.read_i64()?),
            Some((6,  0x06)) => tus   = Some(r.read_i64()?),
            Some((7,  0x06)) => tcs   = Some(r.read_i64()?),
            Some((9,  0x06)) => dpo   = Some(r.read_i64()?),
            Some((10, 0x06)) => ipo   = Some(r.read_i64()?),
            Some((11, 0x06)) => dicpo = Some(r.read_i64()?),
            Some((_, tc))    => r.skip(tc)?,
        }
    }
    Ok(ColumnMetadata {
        type_:                   ty.ok_or_else(|| missing("parquet: ColumnMetadata missing type_"))?,
        encodings:               enc,
        path_in_schema:          path,
        codec:                   codec.ok_or_else(|| missing("parquet: ColumnMetadata missing codec"))?,
        num_values:              nv.ok_or_else(|| missing("parquet: ColumnMetadata missing num_values"))?,
        total_uncompressed_size: tus.ok_or_else(|| missing("parquet: ColumnMetadata missing total_uncompressed_size"))?,
        total_compressed_size:   tcs.ok_or_else(|| missing("parquet: ColumnMetadata missing total_compressed_size"))?,
        data_page_offset:        dpo.ok_or_else(|| missing("parquet: ColumnMetadata missing data_page_offset"))?,
        index_page_offset:       ipo,
        dictionary_page_offset:  dicpo,
    })
}

fn decode_column_chunk(r: &mut ThriftReader<'_>) -> Result<ColumnChunkMetadata> {
    let (mut fp, mut fo, mut md) = (None, None, None);
    let mut last = 0i16;
    loop {
        match r.read_field_header(&mut last)? {
            None            => break,
            Some((1, 0x08)) => fp = Some(r.read_string()?),
            Some((2, 0x06)) => fo = Some(r.read_i64()?),
            Some((3, 0x0C)) => md = Some(decode_column_metadata(r)?),
            Some((_, tc))   => r.skip(tc)?,
        }
    }
    Ok(ColumnChunkMetadata {
        file_path:   fp,
        file_offset: fo.ok_or_else(|| missing("parquet: ColumnChunkMetadata missing file_offset"))?,
        meta_data:   md,
    })
}

fn decode_row_group(r: &mut ThriftReader<'_>) -> Result<RowGroupMetadata> {
    let (mut cols, mut tbs, mut nr, mut fo, mut tcs, mut ord) =
        (None, None, None, None, None, None);
    let mut last = 0i16;
    loop {
        match r.read_field_header(&mut last)? {
            None            => break,
            Some((1, 0x09)) => {
                let (et, n) = r.read_list_header()?;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    if et == 0x0C { v.push(decode_column_chunk(r)?) } else { r.skip(et)? }
                }
                cols = Some(v);
            }
            Some((2, 0x06)) => tbs = Some(r.read_i64()?),
            Some((3, 0x06)) => nr  = Some(r.read_i64()?),
            Some((5, 0x06)) => fo  = Some(r.read_i64()?),
            Some((6, 0x06)) => tcs = Some(r.read_i64()?),
            Some((7, 0x04)) => ord = Some(r.read_i16()?),
            Some((_, tc))   => r.skip(tc)?,
        }
    }
    Ok(RowGroupMetadata {
        columns:               cols.ok_or_else(|| missing("parquet: RowGroupMetadata missing columns"))?,
        total_byte_size:       tbs.ok_or_else(|| missing("parquet: RowGroupMetadata missing total_byte_size"))?,
        num_rows:              nr.ok_or_else(|| missing("parquet: RowGroupMetadata missing num_rows"))?,
        file_offset:           fo,
        total_compressed_size: tcs,
        ordinal:               ord,
    })
}

fn decode_file_metadata_inner(r: &mut ThriftReader<'_>) -> Result<FileMetadata> {
    let (mut ver, mut schema, mut nr, mut rgs) = (None, None, None, None);
    let (mut kvm, mut cb) = (Vec::new(), None);
    let mut last = 0i16;
    loop {
        match r.read_field_header(&mut last)? {
            None            => break,
            Some((1, 0x05)) => ver = Some(r.read_i32()?),
            Some((2, 0x09)) => {
                let (et, n) = r.read_list_header()?;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    if et == 0x0C { v.push(decode_schema_element(r)?) } else { r.skip(et)? }
                }
                schema = Some(v);
            }
            Some((3, 0x06)) => nr = Some(r.read_i64()?),
            Some((4, 0x09)) => {
                let (et, n) = r.read_list_header()?;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    if et == 0x0C { v.push(decode_row_group(r)?) } else { r.skip(et)? }
                }
                rgs = Some(v);
            }
            Some((5, 0x09)) => {
                let (et, n) = r.read_list_header()?;
                for _ in 0..n {
                    if et == 0x0C { kvm.push(decode_key_value(r)?) } else { r.skip(et)? }
                }
            }
            Some((6, 0x08)) => cb = Some(r.read_string()?),
            Some((_, tc))   => r.skip(tc)?,
        }
    }
    Ok(FileMetadata {
        version:            ver.unwrap_or(1),
        schema:             schema.ok_or_else(|| missing("parquet: FileMetadata missing schema"))?,
        num_rows:           nr.ok_or_else(|| missing("parquet: FileMetadata missing num_rows"))?,
        row_groups:         rgs.ok_or_else(|| missing("parquet: FileMetadata missing row_groups"))?,
        key_value_metadata: kvm,
        created_by:         cb,
    })
}

/// Extract and Thrift-decode the Parquet FileMetaData from raw file bytes and a validated prelude.
pub fn decode_file_metadata(bytes: &[u8], prelude: &super::FooterPrelude) -> Result<FileMetadata> {
    let start = prelude.footer_offset();
    let end   = prelude.footer_end_offset();
    if end > bytes.len() {
        return Err(missing("parquet footer bounds exceed file length"));
    }
    let mut r = ThriftReader::new(&bytes[start..end]);
    decode_file_metadata_inner(&mut r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::validate_footer_prelude;
    use crate::wire::thrift::ThriftReader;

    #[test]
    fn decode_schema_element_leaf() {
        // type_=INT32(1), repetition=REQUIRED(0), name="x", field_id=1
        // Field 1 (I32, delta=1): 0x15, 0x02  [zigzag(2)=1]
        // Field 3 (I32, delta=2): 0x25, 0x00  [zigzag(0)=0]
        // Field 4 (BINARY, delta=1): 0x18, 0x01, b'x'
        // Field 9 (I32, delta=5): 0x55, 0x02  [zigzag(2)=1]
        // Stop: 0x00
        let bytes = [0x15u8, 0x02, 0x25, 0x00, 0x18, 0x01, b'x', 0x55, 0x02, 0x00];
        let mut r = ThriftReader::new(&bytes);
        let elem = decode_schema_element(&mut r).unwrap();
        assert_eq!(elem.name, "x");
        assert_eq!(elem.type_, Some(1));
        assert_eq!(elem.repetition_type, Some(0));
        assert_eq!(elem.field_id, Some(1));
        assert_eq!(elem.num_children, None);
    }

    #[test]
    fn decode_schema_element_group() {
        // Group "point" with num_children=2
        // Field 4 (BINARY, delta=4): 0x48, 0x05, "point"
        // Field 5 (I32, delta=1): 0x15, 0x04  [zigzag(4)=2]
        // Stop: 0x00
        let bytes = [0x48u8, 0x05, b'p', b'o', b'i', b'n', b't', 0x15, 0x04, 0x00];
        let mut r = ThriftReader::new(&bytes);
        let elem = decode_schema_element(&mut r).unwrap();
        assert_eq!(elem.name, "point");
        assert_eq!(elem.type_, None);
        assert_eq!(elem.num_children, Some(2));
    }

    #[test]
    fn decode_file_metadata_minimal() {
        let file_bytes: &[u8] = &[
            b'P', b'A', b'R', b'1',
            0x15, 0x04,       // FM field 1: version=2
            0x19, 0x2C,       // FM field 2: schema list 2 structs
            // SchemaElement root: name="schema", num_children=1
            0x48, 0x06, b's', b'c', b'h', b'e', b'm', b'a', 0x15, 0x02, 0x00,
            // SchemaElement "x": type_=1, repetition=0, name="x"
            0x15, 0x02, 0x25, 0x00, 0x18, 0x01, b'x', 0x00,
            0x16, 0x0A,       // FM field 3: num_rows=5
            0x19, 0x1C,       // FM field 4: row_groups list 1 struct
            0x19, 0x1C,       // RG field 1: columns list 1 struct
            0x26, 0x08,       // CC field 2: file_offset=4
            0x1C,             // CC field 3: meta_data struct
            0x15, 0x02,       // CM field 1: type_=INT32=1
            0x19, 0x15, 0x00, // CM field 2: encodings=[PLAIN=0]
            0x19, 0x18, 0x01, b'x', // CM field 3: path=["x"]
            0x15, 0x00,       // CM field 4: codec=0
            0x16, 0x0A,       // CM field 5: num_values=5
            0x16, 0x28,       // CM field 6: total_uncompressed=20
            0x16, 0x28,       // CM field 7: total_compressed=20
            0x26, 0x08,       // CM field 9: data_page_offset=4
            0x00,             // CM stop
            0x00,             // CC stop
            0x16, 0x28,       // RG field 2: total_byte_size=20
            0x16, 0x0A,       // RG field 3: num_rows=5
            0x00,             // RG stop
            0x00,             // FM stop
            0x3B, 0x00, 0x00, 0x00, // footer_len=59
            b'P', b'A', b'R', b'1',
        ];
        let prelude = validate_footer_prelude(file_bytes).unwrap();
        let meta = decode_file_metadata(file_bytes, &prelude).unwrap();
        assert_eq!(meta.version, 2);
        assert_eq!(meta.num_rows, 5);
        assert_eq!(meta.schema.len(), 2);
        assert_eq!(meta.schema[1].name, "x");
        assert_eq!(meta.schema[1].type_, Some(1));
        assert_eq!(meta.row_groups.len(), 1);
        assert_eq!(meta.row_groups[0].num_rows, 5);
        assert_eq!(meta.row_groups[0].columns.len(), 1);
        assert_eq!(meta.row_groups[0].columns[0].file_offset, 4);
        let cm = meta.row_groups[0].columns[0].meta_data.as_ref().unwrap();
        assert_eq!(cm.num_values, 5);
        assert_eq!(cm.path_in_schema, vec!["x".to_string()]);
    }

    #[test]
    fn decode_file_metadata_rejects_missing_required_field() {
        // FileMetaData with only version=2 then stop; missing schema/num_rows/row_groups.
        let footer_bytes = [0x15u8, 0x04, 0x00];
        let mut file = Vec::new();
        file.extend_from_slice(b"PAR1");
        file.extend_from_slice(&footer_bytes);
        let len = footer_bytes.len() as u32;
        file.extend_from_slice(&len.to_le_bytes());
        file.extend_from_slice(b"PAR1");
        let prelude = validate_footer_prelude(&file).unwrap();
        let err = decode_file_metadata(&file, &prelude).unwrap_err();
        assert!(matches!(err, consus_core::Error::InvalidFormat { .. }));
    }
}
