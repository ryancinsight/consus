//! Parquet field identifier and schema model.
//!
//! ## Specification
//!
//! This module defines the canonical schema boundary for `consus-parquet`.
//! It separates:
//! - physical storage types
//! - logical annotations
//! - field identity and repetition
//! - schema evolution and projection
//! - statistics and column ordering
//!
//! The module is authoritative for schema semantics inside the crate.
//! It does not implement wire encoding or decoding.
//!
//! ## Invariants
//!
//! - Field names are non-empty.
//! - Field ids are stable across compatible schema evolution.
//! - Repetition is explicit and never inferred from the physical type.
//! - Schema descriptors preserve field order.
//! - Nested fields are modeled recursively.
//!
//! ## Architecture
//!
//! ```text
//! schema/
//! ├── physical/        # Parquet physical type identifiers
//! ├── logical/         # Logical annotations and repetition semantics
//! ├── field/           # Field descriptors and schema descriptor
//! ├── evolution/       # Schema merge and projection rules
//! └── arrow/           # Arrow bridge descriptors
//! ```
//!
//! ## Reexports
//!
//! This module only reexports items that are actually defined in the
//! implemented submodules.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
pub mod arrow;
#[cfg(feature = "alloc")]
pub mod evolution;
#[cfg(feature = "alloc")]
pub mod field;
pub mod logical;
pub mod physical;

#[cfg(feature = "alloc")]
pub use arrow::{ArrowBridge, ArrowBridgeMode, ArrowFieldDescriptor, ArrowIntegrationPlan};
#[cfg(feature = "alloc")]
pub use evolution::{
    SchemaEvolution, SchemaEvolutionStep, SchemaMergeError, SchemaMergeMode, SchemaProjection,
    SchemaProjectionError,
};
#[cfg(feature = "alloc")]
pub use field::{FieldDescriptor, FieldId, SchemaDescriptor};
pub use logical::{LogicalType, Nullability, Repetition, TimeUnit, TypeAnnotation};
pub use physical::ParquetPhysicalType;

#[cfg(feature = "alloc")]
pub use physical::ParquetPhysicalWidth;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn schema_module_exports_are_available() {
        let physical = ParquetPhysicalType::Boolean;
        assert_eq!(physical.width(), Some(1));

        let field =
            FieldDescriptor::required(FieldId::new(1), "temperature", ParquetPhysicalType::Double);
        assert_eq!(field.name(), "temperature");
        assert!(field.is_required());
    }

    #[cfg(not(feature = "alloc"))]
    #[test]
    fn no_alloc_exports_physical_and_logical_types() {
        assert_eq!(ParquetPhysicalType::Boolean.width(), Some(1));
        assert!(
            LogicalType::Timestamp {
                unit: TimeUnit::Nanoseconds,
                is_adjusted_to_utc: true,
            }
            .is_temporal()
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn schema_descriptor_construction_works() {
        let schema = SchemaDescriptor::new(alloc::vec![]);
        assert_eq!(schema.field_count(), 0);
        assert!(schema.is_empty());
    }
}
