#![cfg_attr(not(feature = "std"), no_std)]

//! # consus-arrow
//!
//! Arrow-facing array, field, schema, and bridge abstractions for Consus.
//!
//! ## Scope
//!
//! This crate provides a Rust-native Arrow interoperability layer:
//! - in-memory array models
//! - schema and field descriptors
//! - datatype conversion boundaries
//! - bridge descriptors for Consus and Parquet integration
//! - lightweight IPC and compute descriptors
//!
//! It does not embed the external Arrow crate. The crate defines the
//! internal semantic model and conversion contracts that a future adapter
//! layer can map onto the Arrow ecosystem.
//!
//! ## Module Hierarchy
//!
//! ```text
//! consus-arrow
//! ├── array/      # Columnar array descriptors and buffers
//! ├── bridge/     # Consus/Parquet <-> Arrow integration bridge
//! ├── compute/    # Compute and kernel planning descriptors
//! ├── datatype/   # Arrow datatype model
//! ├── field/      # Arrow field descriptors
//! ├── ipc/        # IPC and record batch descriptors
//! ├── memory/     # Buffer and buffer-view abstractions
//! └── schema/     # Arrow schema model
//! ```
//!
//! ## Design Constraints
//!
//! - No duplicated type-specific APIs when one generic model suffices.
//! - Schema and field identity remain stable across bridge conversion.
//! - Zero-copy eligibility is explicit and value-semantic.
//! - The crate stays dependency-light and no_std-compatible by default.
//!
//! ## Status
//!
//! This is the authoritative Arrow model crate for Consus.

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod array;
pub mod bridge;
pub mod compute;
pub mod conversion;
pub mod datatype;
pub mod field;
pub mod ipc;
pub mod memory;
pub mod schema;

pub use array::{ArrayData, ArrowArray, ValidityBitmap};
pub use bridge::{
    ArrowBridge, ArrowBridgeMode, ArrowBridgePlan, ArrowDataTypeHint, ArrowFieldDescriptor,
    ArrowSchemaMapping, ArrowZeroCopyConstraint,
};
pub use datatype::{
    ArrowDataType, DecimalType, DictionaryType, DurationType, FixedSizeBinaryType, IntSign,
    ListType, MapType, StructType, TimeUnit, TimestampType, UnionType,
};
pub use field::{
    ArrowField, ArrowFieldBuilder, ArrowFieldId, ArrowFieldKind, ArrowFieldMultiplicity,
    ArrowFieldSemantics, ArrowNullability, field_from_datatype, kind_from_datatype,
};
pub use ipc::{
    BatchKind, BufferOwnership, DictionaryBatchDescriptor, IpcFraming, IpcMaterializationPolicy,
    IpcPlan, IpcPlanError, RecordBatchDescriptor,
};
pub use memory::{ArrowArrayMemory, ArrowBitmap, ArrowBuffer, ArrowOffsets};

#[cfg(feature = "alloc")]
pub use schema::{
    ArrowSchema, ArrowSchemaError, ArrowSchemaMergePlan, ArrowSchemaMergeStep, SchemaProjectionPlan,
};

pub use conversion::{
    ConversionCompatibility, ConversionMode, analyze_conversion_compatibility,
    arrow_datatype_to_core, core_datatype_to_arrow_hint,
};

#[cfg(feature = "alloc")]
pub use conversion::{ArrowFieldFromCoreBuilder, arrow_schema_to_core_pairs};

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn crate_exports_schema_types() {
        let schema = schema::ArrowSchema::empty();
        assert_eq!(schema.field_count(), 0);
        assert!(schema.is_empty());
    }
}
