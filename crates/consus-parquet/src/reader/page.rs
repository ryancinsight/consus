//! Column chunk page iterator and value merger.
//!
//! ## DataPage v1 vs v2 decompression contract
//!
//! DataPage v1: repetition levels + definition levels + values are compressed
//! as one unit. Decompress the entire payload, then split into levels and values.
//!
//! DataPage v2: levels are never compressed. Split first using the byte lengths
//! from the header; only `values_bytes` is optionally compressed.

use alloc::{format, string::String, vec::Vec};
use consus_core::{Error, Result};

use crate::encoding::column::{
    ColumnValues, ColumnValuesWithLevels, decode_column_values, decode_compressed_column_values,
    decode_dictionary_page,
};
use crate::encoding::compression::{CompressionCodec, decompress_page_values};
use crate::schema::physical::ParquetPhysicalType;
use crate::wire::page::{PageType, decode_page_header};
use crate::wire::payload::{split_data_page_v1, split_data_page_v2};

/// Concatenate a non-empty sequence of per-page `ColumnValues` into one.
///
/// All elements must share the same physical type; this invariant holds by
/// construction since all pages within one column chunk encode the same column.
///
/// # Errors
///
/// - `InvalidFormat` — `parts` is empty.
/// - `DatatypeMismatch` — pages have different physical types (malformed file).
pub(crate) fn merge_column_values(parts: Vec<ColumnValues>) -> Result<ColumnValues> {
    if parts.is_empty() {
        return Err(Error::InvalidFormat {
            message: String::from("parquet: column chunk contains no data pages"),
        });
    }
    if parts.len() == 1 {
        let mut p = parts;
        return Ok(p.remove(0));
    }
    let pt = parts[0].physical_type();
    for part in &parts[1..] {
        if part.physical_type() != pt {
            return Err(Error::DatatypeMismatch {
                expected: format!("{pt:?}"),
                found: format!("{:?}", part.physical_type()),
            });
        }
    }
    Ok(match pt {
        ParquetPhysicalType::Boolean => {
            let mut out = Vec::new();
            for p in parts {
                if let ColumnValues::Boolean(v) = p {
                    out.extend(v);
                }
            }
            ColumnValues::Boolean(out)
        }
        ParquetPhysicalType::Int32 => {
            let mut out = Vec::new();
            for p in parts {
                if let ColumnValues::Int32(v) = p {
                    out.extend(v);
                }
            }
            ColumnValues::Int32(out)
        }
        ParquetPhysicalType::Int64 => {
            let mut out = Vec::new();
            for p in parts {
                if let ColumnValues::Int64(v) = p {
                    out.extend(v);
                }
            }
            ColumnValues::Int64(out)
        }
        ParquetPhysicalType::Int96 => {
            let mut out = Vec::new();
            for p in parts {
                if let ColumnValues::Int96(v) = p {
                    out.extend(v);
                }
            }
            ColumnValues::Int96(out)
        }
        ParquetPhysicalType::Float => {
            let mut out = Vec::new();
            for p in parts {
                if let ColumnValues::Float(v) = p {
                    out.extend(v);
                }
            }
            ColumnValues::Float(out)
        }
        ParquetPhysicalType::Double => {
            let mut out = Vec::new();
            for p in parts {
                if let ColumnValues::Double(v) = p {
                    out.extend(v);
                }
            }
            ColumnValues::Double(out)
        }
        ParquetPhysicalType::ByteArray => {
            let mut out = Vec::new();
            for p in parts {
                if let ColumnValues::ByteArray(v) = p {
                    out.extend(v);
                }
            }
            ColumnValues::ByteArray(out)
        }
        ParquetPhysicalType::FixedLenByteArray(fl) => {
            let mut out = Vec::new();
            for p in parts {
                if let ColumnValues::FixedLenByteArray { values, .. } = p {
                    out.extend(values);
                }
            }
            ColumnValues::FixedLenByteArray {
                fixed_len: fl,
                values: out,
            }
        }
    })
}

/// Stateful page decoder for a single Parquet column chunk.
///
/// Retains the dictionary decoded from the column's dictionary page, reusing
/// it across all subsequent data pages in the same chunk. Construct once per
/// column chunk and call [`Self::decode_pages_from_chunk_bytes`].
pub struct ColumnPageDecoder {
    physical_type: ParquetPhysicalType,
    codec: CompressionCodec,
    max_rep_level: i32,
    max_def_level: i32,
    dictionary: Option<ColumnValues>,
}

impl ColumnPageDecoder {
    /// Construct a decoder for a column with the given physical type and codec.
    ///
    /// `max_rep_level` and `max_def_level` are the Dremel maximum nesting levels
    /// for the column.
    pub fn new(
        physical_type: ParquetPhysicalType,
        codec: CompressionCodec,
        max_rep_level: i32,
        max_def_level: i32,
    ) -> Self {
        Self {
            physical_type,
            codec,
            max_rep_level,
            max_def_level,
            dictionary: None,
        }
    }

    /// Decode all pages from a column chunk byte slice and return concatenated values.
    ///
    /// `bytes` must span from the first page (dictionary or data) through the last
    /// data page in the chunk, as identified by `ColumnMetadata` offsets and sizes.
    ///
    /// # Errors
    ///
    /// - `InvalidFormat` — missing required page sub-headers or no data pages found.
    /// - `BufferTooSmall` — a page's `compressed_page_size` exceeds the slice.
    /// - Propagates decompression and value-decoding errors.
    pub fn decode_pages_from_chunk_bytes(&mut self, bytes: &[u8]) -> Result<ColumnValues> {
        let mut pos = 0usize;
        let mut parts: Vec<ColumnValues> = Vec::new();

        while pos < bytes.len() {
            let (header, consumed) = decode_page_header(&bytes[pos..])?;
            pos += consumed;

            let compressed_sz = header.compressed_page_size as usize;
            let uncompressed_sz = header.uncompressed_page_size as usize;

            if pos + compressed_sz > bytes.len() {
                return Err(Error::BufferTooSmall {
                    required: pos + compressed_sz,
                    provided: bytes.len(),
                });
            }

            let page_bytes = &bytes[pos..pos + compressed_sz];
            pos += compressed_sz;

            match header.type_ {
                PageType::DictionaryPage => {
                    let dict_hdr =
                        header
                            .dictionary_page_header
                            .ok_or_else(|| Error::InvalidFormat {
                                message: String::from(
                                    "parquet: DictionaryPage missing dictionary_page_header",
                                ),
                            })?;
                    // Dictionary pages may be compressed; decompress before PLAIN decode.
                    let plain = decompress_page_values(page_bytes, self.codec, uncompressed_sz)?;
                    self.dictionary = Some(decode_dictionary_page(
                        &plain,
                        &dict_hdr,
                        self.physical_type,
                    )?);
                }

                PageType::DataPage => {
                    let dph = header
                        .data_page_header
                        .ok_or_else(|| Error::InvalidFormat {
                            message: String::from("parquet: DataPage missing data_page_header"),
                        })?;
                    // DataPage v1: levels and values are compressed as one unit.
                    let plain = decompress_page_values(page_bytes, self.codec, uncompressed_sz)?;
                    let payload =
                        split_data_page_v1(&plain, &dph, self.max_rep_level, self.max_def_level)?;
                    // For optional/repeated columns, values section contains only non-null values.
                    // Count defined positions from def_levels when max_def_level > 0.
                    let non_null = if self.max_def_level > 0 && !payload.def_levels.is_empty() {
                        payload
                            .def_levels
                            .iter()
                            .filter(|&&d| d == self.max_def_level)
                            .count()
                    } else {
                        payload.num_values as usize
                    };
                    parts.push(decode_column_values(
                        payload.values_bytes,
                        non_null,
                        dph.encoding,
                        self.physical_type,
                        self.dictionary.as_ref(),
                    )?);
                }

                PageType::DataPageV2 => {
                    let v2 = header
                        .data_page_header_v2
                        .ok_or_else(|| Error::InvalidFormat {
                            message: String::from(
                                "parquet: DataPageV2 missing data_page_header_v2",
                            ),
                        })?;
                    // DataPage v2: levels are never compressed; values section is optional.
                    let payload = split_data_page_v2(
                        page_bytes,
                        &v2,
                        self.max_rep_level,
                        self.max_def_level,
                    )?;
                    // is_compressed defaults true per spec when absent; UNCOMPRESSED codec
                    // short-circuits to a copy so the compressed path is always safe.
                    let non_null_v2 = if self.max_def_level > 0 && !payload.def_levels.is_empty() {
                        payload
                            .def_levels
                            .iter()
                            .filter(|&&d| d == self.max_def_level)
                            .count()
                    } else {
                        payload.num_values as usize
                    };
                    let is_compressed = v2.is_compressed.unwrap_or(true)
                        && self.codec != CompressionCodec::Uncompressed;
                    let values = if is_compressed {
                        let levels_len = (v2.repetition_levels_byte_length
                            + v2.definition_levels_byte_length)
                            as usize;
                        let values_uncomp = uncompressed_sz.saturating_sub(levels_len);
                        decode_compressed_column_values(
                            payload.values_bytes,
                            non_null_v2,
                            v2.encoding,
                            self.physical_type,
                            self.dictionary.as_ref(),
                            self.codec,
                            values_uncomp,
                        )?
                    } else {
                        decode_column_values(
                            payload.values_bytes,
                            non_null_v2,
                            v2.encoding,
                            self.physical_type,
                            self.dictionary.as_ref(),
                        )?
                    };
                    parts.push(values);
                }

                PageType::IndexPage => {
                    // Index pages carry no value data; advance past payload and continue.
                }
            }
        }

        merge_column_values(parts)
    }

    /// Decode all pages and return values with Dremel levels.
    ///
    /// Like [`Self::decode_pages_from_chunk_bytes`] but also returns the
    /// accumulated repetition and definition levels from all data pages.
    ///
    /// `bytes` must span from the first page through the last data page.
    pub fn decode_pages_with_levels(&mut self, bytes: &[u8]) -> Result<ColumnValuesWithLevels> {
        let mut pos = 0usize;
        let mut parts: Vec<ColumnValues> = Vec::new();
        let mut all_rep: Vec<i32> = Vec::new();
        let mut all_def: Vec<i32> = Vec::new();

        while pos < bytes.len() {
            let (header, consumed) = decode_page_header(&bytes[pos..])?;
            pos += consumed;

            let compressed_sz = header.compressed_page_size as usize;
            let uncompressed_sz = header.uncompressed_page_size as usize;

            if pos + compressed_sz > bytes.len() {
                return Err(Error::BufferTooSmall {
                    required: pos + compressed_sz,
                    provided: bytes.len(),
                });
            }

            let page_bytes = &bytes[pos..pos + compressed_sz];
            pos += compressed_sz;

            match header.type_ {
                PageType::DictionaryPage => {
                    let dict_hdr =
                        header
                            .dictionary_page_header
                            .ok_or_else(|| Error::InvalidFormat {
                                message: String::from(
                                    "parquet: DictionaryPage missing dictionary_page_header",
                                ),
                            })?;
                    let plain = decompress_page_values(page_bytes, self.codec, uncompressed_sz)?;
                    self.dictionary = Some(decode_dictionary_page(
                        &plain,
                        &dict_hdr,
                        self.physical_type,
                    )?);
                }

                PageType::DataPage => {
                    let dph = header
                        .data_page_header
                        .ok_or_else(|| Error::InvalidFormat {
                            message: String::from("parquet: DataPage missing data_page_header"),
                        })?;
                    let plain = decompress_page_values(page_bytes, self.codec, uncompressed_sz)?;
                    let payload =
                        split_data_page_v1(&plain, &dph, self.max_rep_level, self.max_def_level)?;
                    all_rep.extend_from_slice(&payload.rep_levels);
                    all_def.extend_from_slice(&payload.def_levels);
                    let non_null = if self.max_def_level > 0 && !payload.def_levels.is_empty() {
                        payload
                            .def_levels
                            .iter()
                            .filter(|&&d| d == self.max_def_level)
                            .count()
                    } else {
                        payload.num_values as usize
                    };
                    parts.push(decode_column_values(
                        payload.values_bytes,
                        non_null,
                        dph.encoding,
                        self.physical_type,
                        self.dictionary.as_ref(),
                    )?);
                }

                PageType::DataPageV2 => {
                    let v2 = header
                        .data_page_header_v2
                        .ok_or_else(|| Error::InvalidFormat {
                            message: String::from(
                                "parquet: DataPageV2 missing data_page_header_v2",
                            ),
                        })?;
                    let payload = split_data_page_v2(
                        page_bytes,
                        &v2,
                        self.max_rep_level,
                        self.max_def_level,
                    )?;
                    all_rep.extend_from_slice(&payload.rep_levels);
                    all_def.extend_from_slice(&payload.def_levels);
                    let non_null_v2 = if self.max_def_level > 0 && !payload.def_levels.is_empty() {
                        payload
                            .def_levels
                            .iter()
                            .filter(|&&d| d == self.max_def_level)
                            .count()
                    } else {
                        payload.num_values as usize
                    };
                    let is_compressed = v2.is_compressed.unwrap_or(true)
                        && self.codec != CompressionCodec::Uncompressed;
                    let values = if is_compressed {
                        let levels_len = (v2.repetition_levels_byte_length
                            + v2.definition_levels_byte_length)
                            as usize;
                        let values_uncomp = uncompressed_sz.saturating_sub(levels_len);
                        decode_compressed_column_values(
                            payload.values_bytes,
                            non_null_v2,
                            v2.encoding,
                            self.physical_type,
                            self.dictionary.as_ref(),
                            self.codec,
                            values_uncomp,
                        )?
                    } else {
                        decode_column_values(
                            payload.values_bytes,
                            non_null_v2,
                            v2.encoding,
                            self.physical_type,
                            self.dictionary.as_ref(),
                        )?
                    };
                    parts.push(values);
                }

                PageType::IndexPage => {}
            }
        }

        let values = merge_column_values(parts)?;
        Ok(ColumnValuesWithLevels {
            values,
            rep_levels: all_rep,
            def_levels: all_def,
            max_rep_level: self.max_rep_level,
            max_def_level: self.max_def_level,
        })
    }
}
