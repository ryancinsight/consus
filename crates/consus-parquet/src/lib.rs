#![cfg_attr(not(feature = "std"), no_std)]
//! # consus-parquet
//!
//! Apache Parquet interoperability layer for the Consus storage library.
//!
//! ## Scope
//!
//! This crate defines the canonical schema model, Arrow bridge descriptors,
//! and hybrid container metadata used by Consus for Parquet interop.
//!
//! ## Architecture
//!
//! ```text
//! consus-parquet
//! ├── schema/              # Field, logical type, and schema descriptors
//! ├── arrow_bridge/        # Arrow bridge descriptors and plans
//! ├── conversion/          # Arrow/Parquet/Core conversion utilities
//! └── hybrid/              # Hybrid Parquet-in-Consus storage metadata
//! ```
//!
//! ## Design Constraints
//!
//! - Parquet concepts map onto `consus-core` datatypes and shapes.
//! - Schema evolution preserves field identity and compatibility rules.
//! - Arrow integration is descriptive and does not depend on the Arrow crate.
//! - Hybrid mode keeps tabular data inside hierarchical containers without
//!   duplicating the canonical schema model.
//!
//! ## Status
//!
//! This crate defines the authoritative Parquet-facing model for Consus.

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod arrow_bridge;
pub mod conversion;
pub mod hybrid;
pub mod schema;

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

pub use hybrid::{
    HybridDatasetLayout, HybridMode, HybridPartitioning, HybridStorageDescriptor,
    HybridStorageEncoding, HybridTableLayout, HybridTableRelation,
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
}
