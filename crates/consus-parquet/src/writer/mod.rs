//! Canonical Parquet writer and wire encoders.
//!
//! This module replaces the prior writer scaffold with a real, testable
//! encoding implementation for footer metadata and page headers. It also
//! provides the canonical write-planning surface used by the rest of the
//! crate.
//!
//! ## Scope
//!
//! - Thrift compact binary encoding for footer metadata structs
//! - Thrift compact binary encoding for page headers
//! - Canonical writer-side schema lowering over nested/group fields
//! - Canonical row-source row/value model for future page emission
//! - Complete file assembly helpers for `PAR1` trailer emission
//!
//! ## Invariants
//!
//! - Schema order is preserved.
//! - Nested/group fields lower to deterministic leaf paths.
//! - Encoded footer metadata round-trips through the existing decoders.
//! - Encoded page headers round-trip through the existing decoders.
//! - Trailer length and magic are emitted in the Parquet format order.
//!
//! ## Non-goals
//!
//! - This module does not fabricate row payload values.
//! - This module does not clone type-specific writer APIs.
//! - Unsupported row-to-page synthesis is reported explicitly.
//!
//! ## Architecture
//!
//! ```text
//! writer/
//! ├── mod.rs         # Manifest + public re-exports
//! ├── types.rs       # Public writer-side row and planning types
//! ├── encode.rs      # ParquetWriter orchestration and file assembly
//! ├── thrift.rs      # Thrift compact protocol helpers
//! ├── tests.rs       # Primary writer tests
//! └── tests_extra.rs # Extended writer tests
//! ```

mod encode;
mod thrift;
mod types;

pub use encode::ParquetWriter;
pub use types::*;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_extra;
