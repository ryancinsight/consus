//! Hybrid Parquet-in-Consus container metadata.
//!
//! ## Specification
//!
//! Hybrid mode embeds Parquet table payloads inside a hierarchical
//! Consus container. This module defines the canonical metadata model
//! for that embedding without implementing a wire-level Parquet encoder
//! or decoder.
//!
//! ## Invariants
//!
//! - A hybrid descriptor identifies one logical table payload.
//! - Table layout metadata is independent from physical storage encoding.
//! - Partitioning and row-group metadata remain stable across schema
//!   evolution when field identity is preserved.
//! - The descriptor is a pure model; execution and I/O belong to higher
//!   layers.
//!
//! ## Module Hierarchy
//!
//! ```text
//! hybrid/
//! ├── descriptor/      # Container-level hybrid table descriptor
//! ├── layout/          # Logical table layout within Consus
//! ├── partitioning/    # Partitioning and row-group placement
//! └── encoding/        # Hybrid storage encoding descriptors
//! ```

#[cfg(feature = "alloc")]
use alloc::string::String;

use crate::schema::ParquetPhysicalType;

/// Hybrid storage mode for a Parquet-backed dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HybridMode {
    /// No hybrid storage; the dataset is not Parquet-backed.
    #[default]
    Disabled,
    /// Parquet payload embedded in the Consus container.
    Embedded,
    /// Parquet payload referenced by an external path or object store key.
    External,
}

/// Logical relation between the container and the embedded table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HybridTableRelation {
    /// The embedded table is the primary representation.
    #[default]
    Primary,
    /// The embedded table is a secondary materialization.
    MaterializedView,
    /// The embedded table is a cache of another source.
    Cache,
    /// The embedded table is a partition of a larger logical table.
    Partition,
}

/// Hybrid storage encoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HybridStorageEncoding {
    /// Columnar Parquet payload.
    #[default]
    ColumnarParquet,
    /// Row-group segmented Parquet payload.
    RowGroupSegmentedParquet,
    /// Arrow-oriented intermediate representation.
    ArrowIntermediate,
}

/// Table layout within the Consus hierarchy.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridTableLayout {
    /// Logical table name.
    pub table_name: String,
    /// Path of the payload within the container.
    pub payload_path: String,
    /// Relation to the container's canonical dataset.
    pub relation: HybridTableRelation,
    /// Encoding strategy for the payload.
    pub encoding: HybridStorageEncoding,
}

#[cfg(feature = "alloc")]
impl HybridTableLayout {
    /// Create a new hybrid table layout.
    #[must_use]
    pub fn new(
        table_name: String,
        payload_path: String,
        relation: HybridTableRelation,
        encoding: HybridStorageEncoding,
    ) -> Self {
        Self {
            table_name,
            payload_path,
            relation,
            encoding,
        }
    }

    /// Return `true` when the payload is embedded.
    #[must_use]
    pub fn is_embedded(&self) -> bool {
        self.relation != HybridTableRelation::Cache
    }
}

/// Partitioning metadata for hybrid Parquet storage.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridPartitioning {
    /// Partition key names in declaration order.
    pub keys: alloc::vec::Vec<String>,
    /// Physical partition paths relative to the container.
    pub paths: alloc::vec::Vec<String>,
}

#[cfg(feature = "alloc")]
impl HybridPartitioning {
    /// Create an empty partitioning descriptor.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            keys: alloc::vec::Vec::new(),
            paths: alloc::vec::Vec::new(),
        }
    }

    /// Returns `true` if the table is partitioned.
    #[must_use]
    pub fn is_partitioned(&self) -> bool {
        !self.keys.is_empty()
    }
}

/// Physical dataset layout for a hybrid table.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridDatasetLayout {
    /// Number of columns in the table.
    pub column_count: usize,
    /// Number of row groups.
    pub row_group_count: usize,
    /// Physical type of the dominant column representation.
    pub physical_type: ParquetPhysicalType,
}

#[cfg(feature = "alloc")]
impl HybridDatasetLayout {
    /// Create a layout descriptor.
    #[must_use]
    pub fn new(column_count: usize, row_group_count: usize, physical_type: ParquetPhysicalType) -> Self {
        Self {
            column_count,
            row_group_count,
            physical_type,
        }
    }

    /// Returns `true` if the dataset is column-major by storage policy.
    #[must_use]
    pub fn is_columnar(&self) -> bool {
        self.column_count > 0
    }
}

/// Canonical descriptor for a hybrid Parquet-in-Consus table.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridStorageDescriptor {
    /// Hybrid mode.
    pub mode: HybridMode,
    /// Table layout.
    pub table_layout: Option<HybridTableLayout>,
    /// Partitioning metadata.
    pub partitioning: Option<HybridPartitioning>,
    /// Dataset layout metadata.
    pub dataset_layout: Option<HybridDatasetLayout>,
}

#[cfg(feature = "alloc")]
impl HybridStorageDescriptor {
    /// Create the default descriptor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mode: HybridMode::Disabled,
            table_layout: None,
            partitioning: None,
            dataset_layout: None,
        }
    }

    /// Attach a table layout and switch to embedded mode.
    #[must_use]
    pub fn with_table_layout(mut self, table_layout: HybridTableLayout) -> Self {
        self.mode = HybridMode::Embedded;
        self.table_layout = Some(table_layout);
        self
    }

    /// Attach partitioning metadata.
    #[must_use]
    pub fn with_partitioning(mut self, partitioning: HybridPartitioning) -> Self {
        self.partitioning = Some(partitioning);
        self
    }

    /// Attach dataset layout metadata.
    #[must_use]
    pub fn with_dataset_layout(mut self, dataset_layout: HybridDatasetLayout) -> Self {
        self.dataset_layout = Some(dataset_layout);
        self
    }

    /// Returns `true` when the descriptor represents a columnar table.
    #[must_use]
    pub fn is_columnar(&self) -> bool {
        self.dataset_layout.as_ref().map_or(false, HybridDatasetLayout::is_columnar)
    }

    /// Borrow the table layout.
    #[must_use]
    pub fn table_layout(&self) -> Option<&HybridTableLayout> {
        self.table_layout.as_ref()
    }
}

#[cfg(feature = "alloc")]
impl Default for HybridStorageDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    #[test]
    fn default_descriptor_is_disabled() {
        let descriptor = HybridStorageDescriptor::default();
        assert_eq!(descriptor.mode, HybridMode::Disabled);
        assert!(descriptor.table_layout().is_none());
        assert!(!descriptor.is_columnar());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn embedded_descriptor_tracks_layout() {
        let layout = HybridTableLayout::new(
            String::from("table"),
            String::from("/payload/table.parquet"),
            HybridTableRelation::Primary,
            HybridStorageEncoding::ColumnarParquet,
        );
        let descriptor = HybridStorageDescriptor::default().with_table_layout(layout.clone());
        assert_eq!(descriptor.mode, HybridMode::Embedded);
        assert_eq!(descriptor.table_layout().unwrap(), &layout);
    }
}
