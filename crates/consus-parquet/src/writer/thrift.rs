use alloc::{string::String, vec::Vec};

use crate::schema::field::{FieldDescriptor, SchemaDescriptor};
use crate::schema::logical::Repetition;
use crate::schema::physical::ParquetPhysicalType;
use crate::wire::metadata::{
    ColumnChunkMetadata, ColumnMetadata, FileMetadata, KeyValue, RowGroupMetadata, SchemaElement,
};
use crate::wire::page::{
    DataPageHeader, DataPageHeaderV2, DictionaryPageHeader, PageHeader, PageType,
};

pub(super) fn encode_unsigned_varint(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn zigzag_i16(value: i16) -> u64 {
    ((value << 1) ^ (value >> 15)) as u16 as u64
}

fn zigzag_i32(value: i32) -> u64 {
    ((value << 1) ^ (value >> 31)) as u32 as u64
}

fn zigzag_i64(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn encode_stop(out: &mut Vec<u8>) {
    out.push(0x00);
}

fn encode_list_header(elem_type: u8, len: usize, out: &mut Vec<u8>) {
    if len <= 14 {
        out.push(((len as u8) << 4) | (elem_type & 0x0F));
    } else {
        out.push(0xF0 | (elem_type & 0x0F));
        encode_unsigned_varint(len as u64, out);
    }
}

// ── Thrift compact binary field emitters with correct relative-delta tracking ──
//
// Thrift compact binary encodes each field header as:
//   high nibble = delta from previous field ID in this struct
//   low  nibble = type code
// `last` must be initialized to 0 at the start of each struct and is updated
// by every field emitter so that consecutive optional fields produce the right
// relative delta even when earlier fields were omitted.

#[inline]
fn field_header(field_id: i16, type_code: u8, last: &mut i16, out: &mut Vec<u8>) {
    let delta = (field_id - *last) as u8;
    *last = field_id;
    out.push((delta << 4) | (type_code & 0x0F));
}

#[inline]
fn field_i32(field_id: i16, value: i32, last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x05, last, out);
    encode_unsigned_varint(zigzag_i32(value), out);
}

#[inline]
fn field_i64(field_id: i16, value: i64, last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x06, last, out);
    encode_unsigned_varint(zigzag_i64(value), out);
}

#[inline]
fn field_i16(field_id: i16, value: i16, last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x04, last, out);
    encode_unsigned_varint(zigzag_i16(value), out);
}

#[inline]
fn field_binary(field_id: i16, bytes: &[u8], last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x08, last, out);
    encode_unsigned_varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

#[inline]
fn field_bool(field_id: i16, value: bool, last: &mut i16, out: &mut Vec<u8>) {
    let tc = if value { 0x01 } else { 0x02 };
    let delta = (field_id - *last) as u8;
    *last = field_id;
    out.push((delta << 4) | tc);
}

#[inline]
fn field_list(field_id: i16, elem_type: u8, count: usize, last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x09, last, out);
    encode_list_header(elem_type, count, out);
}

/// Emit the struct field header (type=0x0C). The caller writes struct content
/// then `encode_stop`.
#[inline]
fn field_struct_header(field_id: i16, last: &mut i16, out: &mut Vec<u8>) {
    field_header(field_id, 0x0C, last, out);
}

fn encode_schema_element(element: &SchemaElement, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    if let Some(t) = element.type_ {
        field_i32(1, t, &mut last, out);
    }
    if let Some(tl) = element.type_length {
        field_i32(2, tl, &mut last, out);
    }
    if let Some(r) = element.repetition_type {
        field_i32(3, r, &mut last, out);
    }
    field_binary(4, element.name.as_bytes(), &mut last, out);
    if let Some(nc) = element.num_children {
        field_i32(5, nc, &mut last, out);
    }
    if let Some(ct) = element.converted_type {
        field_i32(6, ct, &mut last, out);
    }
    if let Some(scale) = element.scale {
        field_i32(7, scale, &mut last, out);
    }
    if let Some(precision) = element.precision {
        field_i32(8, precision, &mut last, out);
    }
    if let Some(fid) = element.field_id {
        field_i32(9, fid, &mut last, out);
    }
    encode_stop(out);
}

fn encode_key_value(kv: &KeyValue, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_binary(1, kv.key.as_bytes(), &mut last, out);
    if let Some(value) = &kv.value {
        field_binary(2, value.as_bytes(), &mut last, out);
    }
    encode_stop(out);
}

fn encode_column_metadata(meta: &ColumnMetadata, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, meta.type_, &mut last, out);
    if !meta.encodings.is_empty() {
        field_list(2, 0x05, meta.encodings.len(), &mut last, out);
        for encoding in &meta.encodings {
            encode_unsigned_varint(zigzag_i32(*encoding), out);
        }
    }
    if !meta.path_in_schema.is_empty() {
        field_list(3, 0x08, meta.path_in_schema.len(), &mut last, out);
        for path in &meta.path_in_schema {
            encode_unsigned_varint(path.len() as u64, out);
            out.extend_from_slice(path.as_bytes());
        }
    }
    field_i32(4, meta.codec, &mut last, out);
    field_i64(5, meta.num_values, &mut last, out);
    field_i64(6, meta.total_uncompressed_size, &mut last, out);
    field_i64(7, meta.total_compressed_size, &mut last, out);
    // Field 8 does not exist in parquet.thrift ColumnMetaData; field 9 is
    // data_page_offset, so delta from field 7 is 2.
    field_i64(9, meta.data_page_offset, &mut last, out);
    if let Some(value) = meta.index_page_offset {
        field_i64(10, value, &mut last, out);
    }
    if let Some(value) = meta.dictionary_page_offset {
        field_i64(11, value, &mut last, out);
    }
    encode_stop(out);
}

fn encode_column_chunk_metadata(meta: &ColumnChunkMetadata, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    if let Some(file_path) = &meta.file_path {
        field_binary(1, file_path.as_bytes(), &mut last, out);
    }
    field_i64(2, meta.file_offset, &mut last, out);
    if let Some(inner) = &meta.meta_data {
        field_struct_header(3, &mut last, out);
        encode_column_metadata(inner, out);
    }
    encode_stop(out);
}

fn encode_row_group_metadata(meta: &RowGroupMetadata, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    if !meta.columns.is_empty() {
        field_list(1, 0x0C, meta.columns.len(), &mut last, out);
        for column in &meta.columns {
            encode_column_chunk_metadata(column, out);
        }
    }
    field_i64(2, meta.total_byte_size, &mut last, out);
    field_i64(3, meta.num_rows, &mut last, out);
    // Field 4 does not exist in parquet.thrift RowGroup; field 5 is
    // file_offset, so delta from field 3 is 2.
    if let Some(value) = meta.file_offset {
        field_i64(5, value, &mut last, out);
    }
    if let Some(value) = meta.total_compressed_size {
        field_i64(6, value, &mut last, out);
    }
    if let Some(value) = meta.ordinal {
        field_i16(7, value, &mut last, out);
    }
    encode_stop(out);
}

pub(super) fn encode_file_metadata(meta: &FileMetadata, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, meta.version, &mut last, out);
    field_list(2, 0x0C, meta.schema.len(), &mut last, out);
    for element in &meta.schema {
        encode_schema_element(element, out);
    }
    field_i64(3, meta.num_rows, &mut last, out);
    field_list(4, 0x0C, meta.row_groups.len(), &mut last, out);
    for row_group in &meta.row_groups {
        encode_row_group_metadata(row_group, out);
    }
    if !meta.key_value_metadata.is_empty() {
        field_list(5, 0x0C, meta.key_value_metadata.len(), &mut last, out);
        for kv in &meta.key_value_metadata {
            encode_key_value(kv, out);
        }
    }
    if let Some(created_by) = &meta.created_by {
        field_binary(6, created_by.as_bytes(), &mut last, out);
    }
    encode_stop(out);
}

pub(super) fn encode_page_header(header: &PageHeader, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, header.type_ as i32, &mut last, out);
    field_i32(2, header.uncompressed_page_size, &mut last, out);
    field_i32(3, header.compressed_page_size, &mut last, out);
    if let Some(crc) = header.crc {
        field_i32(4, crc, &mut last, out);
    }
    match header.type_ {
        PageType::DataPage => {
            if let Some(dph) = &header.data_page_header {
                field_struct_header(5, &mut last, out);
                encode_data_page_header(dph, out);
            }
        }
        PageType::DictionaryPage => {
            if let Some(dph) = &header.dictionary_page_header {
                field_struct_header(7, &mut last, out);
                encode_dictionary_page_header(dph, out);
            }
        }
        PageType::DataPageV2 => {
            if let Some(dph) = &header.data_page_header_v2 {
                field_struct_header(8, &mut last, out);
                encode_data_page_header_v2(dph, out);
            }
        }
        PageType::IndexPage => {}
    }
    encode_stop(out);
}

fn encode_data_page_header(header: &DataPageHeader, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, header.num_values, &mut last, out);
    field_i32(2, header.encoding, &mut last, out);
    field_i32(3, header.definition_level_encoding, &mut last, out);
    field_i32(4, header.repetition_level_encoding, &mut last, out);
    encode_stop(out);
}

fn encode_dictionary_page_header(header: &DictionaryPageHeader, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, header.num_values, &mut last, out);
    field_i32(2, header.encoding, &mut last, out);
    if let Some(sorted) = header.is_sorted {
        field_bool(3, sorted, &mut last, out);
    }
    encode_stop(out);
}

fn encode_data_page_header_v2(header: &DataPageHeaderV2, out: &mut Vec<u8>) {
    let mut last: i16 = 0;
    field_i32(1, header.num_values, &mut last, out);
    field_i32(2, header.num_nulls, &mut last, out);
    field_i32(3, header.num_rows, &mut last, out);
    field_i32(4, header.encoding, &mut last, out);
    field_i32(5, header.definition_levels_byte_length, &mut last, out);
    field_i32(6, header.repetition_levels_byte_length, &mut last, out);
    if let Some(compressed) = header.is_compressed {
        field_bool(7, compressed, &mut last, out);
    }
    encode_stop(out);
}

pub(super) fn build_schema_elements(schema: &SchemaDescriptor) -> Vec<SchemaElement> {
    let mut elements = Vec::new();
    elements.push(SchemaElement {
        type_: None,
        type_length: None,
        repetition_type: Some(0),
        name: String::from("schema"),
        num_children: Some(schema.field_count() as i32),
        converted_type: None,
        scale: None,
        precision: None,
        field_id: None,
    });

    let mut i = 0usize;
    while i < schema.fields().len() {
        push_schema_element(&schema.fields()[i], &mut elements);
        i += 1;
    }

    elements
}

fn push_schema_element(field: &FieldDescriptor, out: &mut Vec<SchemaElement>) {
    let repetition_type = match field.repetition() {
        Repetition::Required => Some(0),
        Repetition::Optional => Some(1),
        Repetition::Repeated => Some(2),
    };

    if field.is_group() {
        out.push(SchemaElement {
            type_: None,
            type_length: None,
            repetition_type,
            name: field.name().to_owned(),
            num_children: Some(field.children().len() as i32),
            converted_type: None,
            scale: None,
            precision: None,
            field_id: Some(field.id().get() as i32),
        });
        let mut i = 0usize;
        while i < field.children().len() {
            push_schema_element(&field.children()[i], out);
            i += 1;
        }
    } else {
        let (type_, type_length) = match field.physical_type() {
            ParquetPhysicalType::Boolean => (Some(0), None),
            ParquetPhysicalType::Int32 => (Some(1), None),
            ParquetPhysicalType::Int64 => (Some(2), None),
            ParquetPhysicalType::Int96 => (Some(3), None),
            ParquetPhysicalType::Float => (Some(4), None),
            ParquetPhysicalType::Double => (Some(5), None),
            ParquetPhysicalType::ByteArray => (Some(6), None),
            ParquetPhysicalType::FixedLenByteArray(width) => (Some(7), Some(width as i32)),
        };

        out.push(SchemaElement {
            type_,
            type_length,
            repetition_type,
            name: field.name().to_owned(),
            num_children: None,
            converted_type: None,
            scale: None,
            precision: None,
            field_id: Some(field.id().get() as i32),
        });
    }
}
