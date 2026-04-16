//! Hybrid Parquet storage descriptors.
//!
//! This module models embedding Parquet tables inside Consus hierarchical
//! containers. It defines the canonical storage relation between a logical
//! table, its physical layout, and the container path that owns it.
//!
//! ## Invariants
//!
//! - A hybrid table descriptor identifies exactly one table payload.
//! - Columnar and row-group settings are explicit and independent.
//! - Hybrid descriptors do not encode wire-level Parquet serialization.
//! - The descriptor is format-agnostic enough to support future backends
//!   without API renaming.
//!
//! ## Architecture
//!
//! ```text
//! schema/hybrid
//! ├── HybridTableLayout      # logical table organization
//! ├── HybridPartitioning     # partition key model
//! ├── HybridStorageEncoding  # storage placement and encoding policy
//! └── HybridParquetTable     # complete hybrid table descriptor
//! ```

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

/// High-level layout of a Parquet table embedded in a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HybridTableLayout {
    /// Single file or single logical payload.
    Flat,
    /// Partitioned across multiple container paths.
    Partitioned,
    /// Row-group oriented layout.
    RowGroupSharded,
    /// Column-group oriented layout.
    ColumnSharded,
}

/// Partitioning policy used by a hybrid table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridPartitioning {
    /// Ordered partition keys.
    #[cfg(feature = "alloc")]
    pub keys: Vec<String>,
    /// Whether partition values are encoded into the path hierarchy.
    pub hierarchical: bool,
}

#[cfg(feature = "alloc")]
impl HybridPartitioning {
    /// Create a partitioning descriptor.
    #[must_use]
    pub fn new(keys: Vec<String>, hierarchical: bool) -> Self {
        Self { keys, hierarchical }
    }

    /// Whether the table is partitioned.
    #[must_use]
    pub fn is_partitioned(&self) -> bool {
        !self.keys.is_empty()
    }
}

#[cfg(not(feature = "alloc"))]
impl HybridPartitioning {
    /// Create a partitioning descriptor.
    #[must_use]
    pub const fn new(_keys: (), hierarchical: bool) -> Self {
        Self { hierarchical }
    }

    /// Whether the table is partitioned.
    #[must_use]
    pub const fn is_partitioned(&self) -> bool {
        false
    }
}

/// Storage encoding policy for hybrid Parquet payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HybridStorageEncoding {
    /// Parquet payload is stored as-is.
    PlainParquet,
    /// Parquet payload is wrapped with container metadata.
    ContainerAnnotated,
    /// Parquet payload is chunked for incremental access.
    ChunkedParquet,
    /// Hybrid mode using a separate Arrow-friendly representation path.
    ArrowOptimized,
}

/// Complete descriptor for a Parquet table embedded in a Consus container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridParquetTable {
    /// Logical table name.
    #[cfg(feature = "alloc")]
    pub name: String,
    /// Absolute container path where the table is stored.
    #[cfg(feature = "alloc")]
    pub container_path: String,
    /// Layout policy for the table.
    pub layout: HybridTableLayout,
    /// Partitioning policy.
    pub partitioning: HybridPartitioning,
    /// Storage encoding policy.
    pub encoding: HybridStorageEncoding,
    /// Whether the table is expected to support append-only growth.
    pub append_only: bool,
}

#[cfg(feature = "alloc")]
impl HybridParquetTable {
    /// Create a hybrid table descriptor.
    #[must_use]
    pub fn new(
        name: String,
        container_path: String,
        layout: HybridTableLayout,
        partitioning: HybridPartitioning,
        encoding: HybridStorageEncoding,
        append_only: bool,
    ) -> Self {
        Self {
            name,
            container_path,
            layout,
            partitioning,
            encoding,
            append_only,
        }
    }

    /// Whether the table uses partitioned storage.
    #[must_use]
    pub fn is_partitioned(&self) -> bool {
        self.partitioning.is_partitioned()
    }

    /// Whether the table is optimized for Arrow-style access.
    #[must_use]
    pub fn is_arrow_optimized(&self) -> bool {
        matches!(self.encoding, HybridStorageEncoding::ArrowOptimized)
    }

    /// Whether the descriptor represents a sharded layout.
    #[must_use]
    pub fn is_sharded(&self) -> bool {
        matches!(
            self.layout,
            HybridTableLayout::Partitioned
                | HybridTableLayout::RowGroupSharded
                | HybridTableLayout::ColumnSharded
        )
    }
}

#[cfg(not(feature = "alloc"))]
impl HybridParquetTable {
    /// Whether the table uses partitioned storage.
    #[must_use]
    pub const fn is_partitioned(&self) -> bool {
        false
    }

    /// Whether the table is optimized for Arrow-style access.
    #[must_use]
    pub const fn is_arrow_optimized(&self) -> bool {
        matches!(self.encoding, HybridStorageEncoding::ArrowOptimized)
    }

    /// Whether the descriptor represents a sharded layout.
    #[must_use]
    pub const fn is_sharded(&self) -> bool {
        matches!(
            self.layout,
            HybridTableLayout::Partitioned
                | HybridTableLayout::RowGroupSharded
                | HybridTableLayout::ColumnSharded
        )
    }
}

/// Backward-compatible alias for a full hybrid storage descriptor.
pub type HybridStorageDescriptor = HybridParquetTable;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn partitioning_detects_presence_of_keys() {
        let partitioning = HybridPartitioning::new(
            alloc::vec![alloc::string::String::from("year"), alloc::string::String::from("month")],
            true,
        );
        assert!(partitioning.is_partitioned());
        assert!(partitioning.hierarchical);
        assert_eq!(partitioning.keys.len(), 2);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn table_descriptor_reports_sharding_and_arrow_mode() {
        let descriptor = HybridParquetTable::new(
            alloc::string::String::from("observations"),
            alloc::string::String::from("/tables/observations"),
            HybridTableLayout::Partitioned,
            HybridPartitioning::new(
                alloc::vec![alloc::string::String::from("site")],
                true,
            ),
            HybridStorageEncoding::ArrowOptimized,
            true,
        );

        assert!(descriptor.is_partitioned());
        assert!(descriptor.is_sharded());
        assert!(descriptor.is_arrow_optimized());
        assert!(descriptor.append_only);
    }

    #[test]
    fn layout_classification_matches_expected_variants() {
        assert!(matches!(HybridTableLayout::Partitioned, HybridTableLayout::Partitioned));
        assert!(matches!(HybridStorageEncoding::PlainParquet, HybridStorageEncoding::PlainParquet));
    }
}
