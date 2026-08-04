//! Schema conversion module for Arrow ↔ Core ↔ Parquet transformations.
//!
//! ## Specification
//!
//! This module defines the canonical conversion contracts between:
//! - Arrow schema model (`ArrowSchema`, `ArrowField`, `ArrowDataType`)
//! - Core Consus datatypes (`Datatype`, `ByteOrder`)
//! - Parquet schema model (via `consus-parquet` integration)
//!
//! ## Invariants
//!
//! - Conversions preserve field identity and semantic meaning.
//! - Lossy conversions are explicit and require `AllowLossy` mode.
//! - Nested structures are converted recursively.
//! - Zero-copy eligibility is computed during conversion.
//! - All conversions are deterministic and reproducible.
//!
//! ## Architecture
//!
//! ```text
//! conversion/
//! ├── types        # Shared enums and builders
//! ├── convert      # Arrow ↔ Core conversion functions
//! ├── core_arrow   # Arrow ↔ Core Datatype conversions
//! ├── parquet_arrow  # Arrow ↔ Parquet schema conversions
//! └── traits       # Generic conversion traits
//! ```

mod convert;
mod types;

pub use convert::{
    analyze_conversion_compatibility, arrow_datatype_to_core, core_datatype_to_arrow_hint,
};
pub use types::{ConversionCompatibility, ConversionMode};

#[cfg(feature = "alloc")]
pub use convert::arrow_schema_to_core_pairs;
#[cfg(feature = "alloc")]
pub use types::ArrowFieldFromCoreBuilder;
