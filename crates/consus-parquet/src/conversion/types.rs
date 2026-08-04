#[cfg(feature = "alloc")]
use alloc::string::String;

use consus_core::Datatype;

/// Minimal representation of an Arrow field for compatibility analysis.
///
/// This struct avoids direct dependency on `consus-arrow` in the conversion
/// module while still enabling compatibility checks.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ArrowFieldRepr {
    /// Field name.
    pub name: String,
    /// Whether the field is nullable.
    pub nullable: bool,
    /// Core datatype representation.
    pub datatype: Datatype,
}

/// Parquet-to-Arrow conversion mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ParquetConversionMode {
    /// Preserve exact Parquet semantics.
    #[default]
    Exact,
    /// Allow type widening for better Arrow compatibility.
    AllowWidening,
    /// Use best-effort mapping when exact conversion is impossible.
    BestEffort,
}

/// Result of Parquet-to-Arrow schema compatibility analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParquetCompatibility {
    /// Schemas are directly compatible.
    Compatible,
    /// Conversion requires schema evolution.
    RequiresEvolution,
    /// Schemas are incompatible.
    Incompatible,
}
