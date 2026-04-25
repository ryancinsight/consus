//! Canonical Parquet dataset and projection model.
//!
//! ## Specification
//!
//! This module defines the authoritative in-memory dataset descriptors used
//! by  before wire-level decoding is introduced.
//!
//! A dataset descriptor is composed of:
//! - one validated 
//! - one or more row groups
//! - one column descriptor per top-level schema field
//! - exact row counts and byte counts for each row group and column chunk
//!
//! ## Invariants
//!
//! - 
//! - 
//! - 
//! - each row group contains exactly one chunk per top-level schema field
//! - each projected column exists in the source schema
//! - projection preserves source field order
//!
//! ## Non-goals
//!
//! - No Parquet wire decoding is implemented here.
//! - No fabricated payload values are stored here.
//! - No public API claims file-read support.
//!
//! ## Architecture
//!
//! ```text
//! dataset/
//! +-- ParquetColumnDescriptor   # Canonical top-level column metadata
//! +-- ColumnChunkDescriptor     # Per-row-group physical chunk metadata
//! +-- RowGroupDescriptor        # Row-group row count and chunk set
//! +-- ParquetDatasetDescriptor  # Whole-dataset validated descriptor
//! +-- ColumnProjection          # One projected column
//! +-- ParquetProjection         # Ordered projected dataset view
//! ```