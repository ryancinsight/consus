//! File-backed Parquet reader.
//!
//! ## Architecture
//!
//! ```text
//! ParquetReader<'a>
//!   ├── new(bytes) → validate footer → decode FileMetadata → materialize dataset
//!   └── read_column_chunk(rg, col) → ColumnPageDecoder → ColumnValues
//! ```
//!
//! ## Level derivation
//!
//! For flat (non-nested) top-level columns, Dremel nesting levels are:
//!
//! | Repetition | max_rep | max_def |
//! |------------|---------|---------|
//! | Required   | 0       | 0       |
//! | Optional   | 0       | 1       |
//! | Repeated   | 1       | 1       |
//!
//! Nested (group) columns return `Error::UnsupportedFeature`.

use alloc::{format, string::String};
use consus_core::{Error, Result};

use crate::dataset::{ParquetDatasetDescriptor, dataset_from_file_metadata};
use crate::encoding::column::ColumnValues;
use crate::encoding::compression::CompressionCodec;
use crate::schema::field::FieldDescriptor;
use crate::schema::logical::Repetition;
use crate::wire::metadata::{FileMetadata, decode_file_metadata};
use crate::wire::validate_footer_prelude;

mod page;

pub use page::ColumnPageDecoder;

/// Derive Dremel definition and repetition max levels for a flat top-level field.
///
/// Returns `(max_rep_level, max_def_level)`. For nested schemas a full
/// ancestor-traversal is required; this covers only the flat case used by
/// [`ParquetReader::read_column_chunk`].
fn max_levels_for_field(field: &FieldDescriptor) -> (i32, i32) {
    match field.repetition() {
        Repetition::Required => (0, 0),
        Repetition::Optional => (0, 1),
        Repetition::Repeated => (1, 1),
    }
}

/// File-backed Parquet reader for in-memory byte slices.
///
/// Validates the Parquet footer trailer, decodes `FileMetadata` via the Thrift
/// compact binary decoder, and materializes a `ParquetDatasetDescriptor`.
/// Column chunks are decoded on demand via [`Self::read_column_chunk`].
///
/// ## Limitations
///
/// - The byte slice must contain the complete file.
/// - Nested (group) columns return `UnsupportedFeature`; only leaf columns
///   corresponding to flat top-level schema fields are supported.
#[derive(Debug)]
pub struct ParquetReader<'a> {
    bytes: &'a [u8],
    metadata: FileMetadata,
    dataset: ParquetDatasetDescriptor,
}

impl<'a> ParquetReader<'a> {
    /// Construct a reader from file bytes, validating and decoding the footer.
    ///
    /// # Errors
    ///
    /// - `BufferTooSmall` — fewer than 8 bytes.
    /// - `InvalidFormat` — invalid PAR1 magic, malformed footer length, or
    ///   Thrift decoding failure.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let prelude = validate_footer_prelude(bytes)?;
        let metadata = decode_file_metadata(bytes, &prelude)?;
        let dataset = dataset_from_file_metadata(&metadata)?;
        Ok(Self {
            bytes,
            metadata,
            dataset,
        })
    }

    /// Decoded wire `FileMetadata`.
    #[must_use]
    pub fn metadata(&self) -> &FileMetadata {
        &self.metadata
    }

    /// Materialized dataset descriptor.
    #[must_use]
    pub fn dataset(&self) -> &ParquetDatasetDescriptor {
        &self.dataset
    }

    /// Read and decode all values for one column chunk in one row group.
    ///
    /// `row_group_idx` is zero-based. `column_ordinal` is the column's position
    /// in the row group's column list (matches schema field order).
    ///
    /// # Errors
    ///
    /// - `InvalidFormat` — index out of bounds, missing inline `ColumnMetadata`,
    ///   or unknown codec discriminant.
    /// - `BufferTooSmall` — column chunk byte range exceeds file slice bounds.
    /// - `UnsupportedFeature` — nested (group) column or disabled codec.
    /// - Propagates decompression and value-decoding errors.
    pub fn read_column_chunk(
        &self,
        row_group_idx: usize,
        column_ordinal: usize,
    ) -> Result<ColumnValues> {
        let rg =
            self.metadata
                .row_groups
                .get(row_group_idx)
                .ok_or_else(|| Error::InvalidFormat {
                    message: format!(
                        "parquet: row group index {row_group_idx} out of bounds ({})",
                        self.metadata.row_groups.len()
                    ),
                })?;

        let chunk_col = rg
            .columns
            .get(column_ordinal)
            .ok_or_else(|| Error::InvalidFormat {
                message: format!(
                    "parquet: column ordinal {column_ordinal} out of bounds ({} columns)",
                    rg.columns.len()
                ),
            })?;

        let chunk_meta = chunk_col
            .meta_data
            .as_ref()
            .ok_or_else(|| Error::InvalidFormat {
                message: String::from("parquet: column chunk missing inline ColumnMetadata"),
            })?;

        // Physical type from the column descriptor preserves FixedLenByteArray length
        // derived during schema parsing; ColumnMetadata.type_ alone loses that length.
        let col_desc =
            self.dataset
                .columns()
                .get(column_ordinal)
                .ok_or_else(|| Error::InvalidFormat {
                    message: format!(
                        "parquet: column ordinal {column_ordinal} not found in dataset descriptor"
                    ),
                })?;

        if col_desc.is_nested() {
            return Err(Error::UnsupportedFeature {
                feature: String::from(
                    "parquet: nested (group) column chunk decoding is not yet supported",
                ),
            });
        }

        let physical_type = col_desc.field().physical_type();
        let codec =
            CompressionCodec::from_i32(chunk_meta.codec).ok_or_else(|| Error::InvalidFormat {
                message: format!("parquet: unknown codec discriminant {}", chunk_meta.codec),
            })?;

        let (max_rep, max_def) = max_levels_for_field(col_desc.field());

        // Dictionary page offset is always before data pages when present; use it
        // as the chunk start so the dictionary page is included in the decoded range.
        let start_offset = chunk_meta
            .dictionary_page_offset
            .unwrap_or(chunk_meta.data_page_offset) as usize;

        let end_offset = start_offset
            .checked_add(chunk_meta.total_compressed_size as usize)
            .ok_or(Error::Overflow)?;

        if end_offset > self.bytes.len() {
            return Err(Error::BufferTooSmall {
                required: end_offset,
                provided: self.bytes.len(),
            });
        }

        let chunk_bytes = &self.bytes[start_offset..end_offset];
        let mut decoder = ColumnPageDecoder::new(physical_type, codec, max_rep, max_def);
        decoder.decode_pages_from_chunk_bytes(chunk_bytes)
    }
}

#[cfg(test)]
mod tests;
