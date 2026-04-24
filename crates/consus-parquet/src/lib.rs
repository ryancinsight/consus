#![cfg_attr(not(feature = "std"), no_std)]
//! # consus-parquet
//!
//! Apache Parquet interoperability layer for the Consus storage library.
//!
//! ## Scope
//!
//! This crate provides the canonical schema model, Thrift footer decoder,
//! page header decoder, Arrow bridge descriptors, hybrid container metadata,
//! and dataset materialization bridge for Parquet interop.
//!
//! ## Architecture
//!
//! ```text
//! consus-parquet
//! ├── schema/ # Field, logical type, and schema descriptors
//! ├── arrow_bridge/ # Arrow bridge descriptors and plans
//! ├── conversion/ # Arrow/Parquet/Core conversion utilities
//! ├── hybrid/ # Hybrid Parquet-in-Consus storage metadata
//! ├── wire/ # Trailer validation, Thrift decoder, FileMetaData, PageHeader
//! ├── encoding/ # Page value decoders: PLAIN, RLE/dict, levels, compression
//! └── dataset/ # In-memory dataset descriptor and projection model
//! ```
//!
//! ## Design Constraints
//!
//! - Parquet concepts map onto `consus-core` datatypes and shapes.
//! - Schema evolution preserves field identity and compatibility rules.
//! - Arrow integration is descriptive and does not depend on the Arrow crate.
//! - Hybrid mode keeps tabular data inside hierarchical containers without
//!   duplicating the canonical schema model.
//! - Footer decoding uses a zero-dependency inline Thrift compact binary decoder.

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod arrow_bridge;
pub mod conversion;
pub mod hybrid;
pub mod schema;
pub mod wire;

#[cfg(feature = "alloc")]
mod dataset;

#[cfg(feature = "alloc")]
pub mod encoding;

pub use arrow_bridge::{
    ArrowBridgeMode, ArrowDataTypeHint, ArrowFieldDescriptor, ArrowIntegrationPlan,
    ArrowSchemaMapping, ArrowZeroCopyConstraint,
};

pub use conversion::{
    ParquetCompatibility, ParquetConversionMode, arrow_nullability_to_parquet_repetition,
    core_to_parquet_logical_hint, core_to_parquet_physical_hint, parquet_field_to_core,
    parquet_logical_to_core_annotation, parquet_physical_to_core,
    parquet_repetition_to_arrow_nullability,
};

#[cfg(feature = "alloc")]
pub use conversion::{ArrowFieldRepr, analyze_parquet_arrow_compatibility};

#[cfg(feature = "alloc")]
pub use dataset::{
    ColumnChunkDescriptor, ColumnProjection, ColumnStorage, ParquetColumnDescriptor,
    ParquetDatasetDescriptor, ParquetProjection, RowGroupDescriptor, dataset_from_file_metadata,
    schema_elements_to_schema,
};

#[cfg(feature = "alloc")]
pub use encoding::{
    ColumnValues, CompressionCodec, decode_bit_packed_raw, decode_column_values,
    decode_compressed_column_values, decode_dictionary_page, decode_levels, decode_plain_boolean,
    decode_plain_byte_array, decode_plain_f32, decode_plain_f64, decode_plain_fixed_byte_array,
    decode_plain_i32, decode_plain_i64, decode_plain_i96, decode_rle_dict_indices,
    decompress_page_values, level_bit_width,
};

#[cfg(feature = "alloc")]
pub use wire::payload::{PagePayload, split_data_page_v1, split_data_page_v2};

pub use hybrid::{
    HybridDatasetLayout, HybridMode, HybridPartitioning, HybridStorageDescriptor,
    HybridStorageEncoding, HybridTableLayout, HybridTableRelation,
};

pub use wire::{
    ColumnChunkLocation, FooterPrelude, ParquetFooterDescriptor, ParquetFooterError,
    RowGroupLocation, validate_footer_prelude,
};

#[cfg(feature = "alloc")]
pub use wire::metadata::{
    ColumnChunkMetadata, ColumnMetadata, FileMetadata, KeyValue, RowGroupMetadata,
    SchemaElement as WireSchemaElement, decode_file_metadata,
};

pub use wire::page::{
    DataPageHeader, DataPageHeaderV2, DictionaryPageHeader, PageHeader, PageType,
    decode_page_header,
};

pub use schema::{
    FieldDescriptor, FieldId, LogicalType, Nullability, ParquetPhysicalType, ParquetPhysicalWidth,
    Repetition, SchemaDescriptor, SchemaEvolution, SchemaEvolutionStep, SchemaMergeError,
    SchemaMergeMode, SchemaProjection, SchemaProjectionError, TimeUnit, TypeAnnotation,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exports_schema_types() {
        let schema = SchemaDescriptor::empty();
        assert_eq!(schema.field_count(), 0);
        assert!(schema.is_empty());
    }

    #[test]
    fn exports_hybrid_types() {
        let descriptor = HybridStorageDescriptor::default();
        assert!(!descriptor.is_columnar());
        assert!(descriptor.table_layout().is_none());
    }

    #[test]
    fn conversion_exports_are_available() {
        let physical = ParquetPhysicalType::Boolean;
        let core_type = parquet_physical_to_core(physical);
        assert!(matches!(core_type, consus_core::Datatype::Boolean));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn dataset_exports_are_available() {
        let schema = SchemaDescriptor::new(alloc::vec![FieldDescriptor::required(
            FieldId::new(1),
            "value",
            ParquetPhysicalType::Int32,
        )]);
        let row_groups = alloc::vec![
            RowGroupDescriptor::new(
                2,
                alloc::vec![ColumnChunkDescriptor::new(FieldId::new(1), 2, 8).unwrap()],
            )
            .unwrap()
        ];
        let dataset = ParquetDatasetDescriptor::new(schema, row_groups).unwrap();
        assert_eq!(dataset.total_rows(), 2);
        assert_eq!(dataset.column_count(), 1);
    }

    #[test]
    fn wire_exports_are_available() {
        let prelude = validate_footer_prelude(b"abcdefghijkl\x04\x00\x00\x00PAR1").unwrap();
        let footer = ParquetFooterDescriptor::new(
            prelude,
            alloc::vec![
                RowGroupLocation::new(
                    2,
                    alloc::vec![
                        ColumnChunkLocation::new(0, 0, 2).unwrap(),
                        ColumnChunkLocation::new(1, 2, 2).unwrap(),
                    ],
                )
                .unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(footer.row_group_count(), 1);
        assert_eq!(footer.total_rows(), 2);
    }

    #[test]
    fn page_type_exports_are_available() {
        assert_eq!(PageType::from_i32(0), Some(PageType::DataPage));
        assert_eq!(PageType::from_i32(3), Some(PageType::DataPageV2));
        assert_eq!(PageType::from_i32(99), None);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn compression_exports_are_available() {
        use crate::CompressionCodec;
        assert_eq!(
            CompressionCodec::from_i32(0),
            Some(CompressionCodec::Uncompressed)
        );
        assert_eq!(CompressionCodec::from_i32(2), Some(CompressionCodec::Gzip));
        assert_eq!(CompressionCodec::from_i32(99), None);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn wire_metadata_exports_are_available() {
        // Verify FileMetadata and decode_file_metadata are in scope.
        let file_bytes: &[u8] = &[
            b'P', b'A', b'R', b'1', 0x15, 0x04, 0x19, 0x2C, 0x48, 0x06, b's', b'c', b'h', b'e',
            b'm', b'a', 0x15, 0x02, 0x00, 0x15, 0x02, 0x25, 0x00, 0x18, 0x01, b'x', 0x00, 0x16,
            0x0A, 0x19, 0x1C, 0x19, 0x1C, 0x26, 0x08, 0x1C, 0x15, 0x02, 0x19, 0x15, 0x00, 0x19,
            0x18, 0x01, b'x', 0x15, 0x00, 0x16, 0x0A, 0x16, 0x28, 0x16, 0x28, 0x26, 0x08, 0x00,
            0x00, 0x16, 0x28, 0x16, 0x0A, 0x00, 0x00, 0x3B, 0x00, 0x00, 0x00, b'P', b'A', b'R',
            b'1',
        ];
        let prelude = validate_footer_prelude(file_bytes).unwrap();
        let meta = decode_file_metadata(file_bytes, &prelude).unwrap();
        assert_eq!(meta.version, 2);
        assert_eq!(meta.num_rows, 5);
        let dataset = dataset_from_file_metadata(&meta).unwrap();
        assert_eq!(dataset.total_rows(), 5);
        assert_eq!(dataset.column_count(), 1);
    }
}
