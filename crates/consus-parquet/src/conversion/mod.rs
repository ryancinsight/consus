//! Parquet schema conversion module.
//!
//! ## Specification
//!
//! This module defines the canonical conversion contracts between:
//! - Parquet schema model (`SchemaDescriptor`, `FieldDescriptor`, `ParquetPhysicalType`)
//! - Arrow schema model (via `consus-arrow` integration)
//! - Core Consus datatypes (via `consus-core`)
//!
//! ## Invariants
//!
//! - Parquet physical types map to Arrow types with explicit width preservation.
//! - Logical type annotations refine the mapping without changing physical storage.
//! - Schema evolution steps preserve field identity through stable `FieldId`.
//! - Nested schemas are converted recursively.
//! - Zero-copy eligibility is computed from physical type and repetition.
//! - Every canonical `consus-core::Datatype` variant maps deterministically.
//!
//! ## Architecture
//!
//! ```text
//! conversion/
//! ├── types    # Shared conversion enums and Arrow field repr
//! └── convert  # Parquet ↔ Core conversion helpers
//! ```

mod convert;
mod types;

pub use convert::{
    arrow_nullability_to_parquet_repetition, core_to_parquet_logical_hint,
    core_to_parquet_physical_hint, parquet_field_to_core, parquet_logical_to_core_annotation,
    parquet_physical_to_core, parquet_repetition_to_arrow_nullability,
};
pub use types::{ParquetCompatibility, ParquetConversionMode};

#[cfg(feature = "alloc")]
pub use convert::{analyze_parquet_arrow_compatibility, parquet_schema_to_core_pairs};
#[cfg(feature = "alloc")]
pub use types::ArrowFieldRepr;
