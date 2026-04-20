//! HDF5 property list types for object creation configuration.
//!
//! Property lists control how objects are created in HDF5 files.
//! They provide the write-side configuration that maps to various
//! header messages in the created objects.
//!
//! ## Property List Types
//!
//! | Type | HDF5 Concept | Controls |
//! |------|-------------|----------|
//! | [`DatasetCreationProps`] | DCPL | Layout, chunking, compression, fill value, filters |
//! | [`GroupCreationProps`] | GCPL | Link storage thresholds, creation order tracking |
//! | [`FileCreationProps`] | FCPL | Superblock version, offset/length sizes, B-tree K values |
//!
//! These types are not parsed from HDF5 binary data. They are Rust-side
//! configuration objects used when writing HDF5 files.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use consus_core::Compression;

/// Storage layout choice for datasets.
///
/// Determines how raw data is organized in the HDF5 file.
///
/// | Variant | Header Message Value | Description |
/// |---------|---------------------|-------------|
/// | Compact | 0 | Data stored in the object header |
/// | Contiguous | 1 | Single contiguous allocation |
/// | Chunked | 2 | B-tree indexed fixed-size chunks |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DatasetLayout {
    /// Data stored directly in the object header (small datasets only).
    ///
    /// Maximum size is limited by the object header chunk size,
    /// typically 64 KiB. No external storage allocation occurs.
    Compact,
    /// Single contiguous block in the file.
    ///
    /// All data occupies one allocation. No chunking overhead,
    /// but compression and extensibility are not supported.
    #[default]
    Contiguous,
    /// B-tree indexed fixed-size chunks.
    ///
    /// Requires [`DatasetCreationProps::chunk_dims`] to be set.
    /// Supports compression, extensible dimensions, and partial I/O.
    Chunked,
}

/// Fill value write policy.
///
/// Controls when fill values are written to uninitialized storage.
///
/// | Variant | HDF5 Value | Behavior |
/// |---------|-----------|----------|
/// | IfSet | 1 | Write only if a fill value is defined |
/// | Always | 0 | Write fill value unconditionally |
/// | Never | 2 | Never write fill values |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillTime {
    /// Write fill value at allocation time only if a fill value is defined.
    #[default]
    IfSet,
    /// Write fill values unconditionally at allocation time.
    Always,
    /// Never write fill values. Uninitialized storage contains
    /// indeterminate data.
    Never,
}

/// Space allocation policy for dataset storage.
///
/// Controls when the file driver allocates space for raw data.
///
/// | Variant | HDF5 Value | Behavior |
/// |---------|-----------|----------|
/// | Early | 0 | Allocate at dataset creation |
/// | Late | 1 | Allocate at first write |
/// | Incremental | 2 | Allocate per-chunk on demand |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocationTime {
    /// Allocate all space at dataset creation time.
    ///
    /// Guarantees contiguous allocation but may waste space
    /// for sparse datasets.
    Early,
    /// Allocate space at first write.
    ///
    /// Default for contiguous layout. Defers allocation until
    /// data is available.
    #[default]
    Late,
    /// Allocate space incrementally per chunk.
    ///
    /// Only valid for chunked layout. Minimizes wasted space
    /// for sparse datasets.
    Incremental,
}

/// Dataset creation property list (DCPL).
///
/// Controls storage layout, chunking, compression, fill value,
/// and filter pipeline for new datasets.
///
/// ## Defaults
///
/// | Field | Default |
/// |-------|---------|
/// | `layout` | [`DatasetLayout::Contiguous`] |
/// | `chunk_dims` | `None` |
/// | `compression` | [`Compression::None`] |
/// | `fill_value` | `None` |
/// | `fill_time` | [`FillTime::IfSet`] |
/// | `alloc_time` | [`AllocationTime::Late`] |
/// | `filters` | empty |
/// | `track_attribute_order` | `false` |
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DatasetCreationProps {
    /// Storage layout strategy.
    pub layout: DatasetLayout,
    /// Chunk dimensions (required when `layout` is [`DatasetLayout::Chunked`]).
    ///
    /// Each element specifies the chunk extent along the corresponding
    /// dataset dimension. The rank must match the dataset's rank.
    pub chunk_dims: Option<Vec<usize>>,
    /// Compression configuration applied to chunked data.
    ///
    /// Ignored for non-chunked layouts. The compression codec is
    /// inserted into the filter pipeline during dataset creation.
    pub compression: Compression,
    /// Fill value as raw bytes in the dataset's native byte order.
    ///
    /// `None` indicates no user-defined fill value (HDF5 library
    /// default of zero-fill applies when `fill_time` is not `Never`).
    pub fill_value: Option<Vec<u8>>,
    /// Policy controlling when fill values are written.
    pub fill_time: FillTime,
    /// Policy controlling when storage space is allocated.
    pub alloc_time: AllocationTime,
    /// Additional HDF5 filter IDs to apply in the pipeline.
    ///
    /// Standard filter IDs:
    /// - 1: Deflate
    /// - 2: Shuffle
    /// - 3: Fletcher32 checksum
    /// - 4: Szip
    /// - 5: Nbit
    /// - 6: ScaleOffset
    pub filters: Vec<u16>,
    /// Track attribute creation order in the object header.
    ///
    /// When `true`, attributes are indexed by creation order in
    /// addition to name order.
    pub track_attribute_order: bool,
    /// Layout message version to emit (3 or 4).
    ///
    /// Version 3 uses a v1 B-tree chunk index.
    /// Version 4 uses a v2 B-tree chunk index with record type 10 (non-filtered)
    /// or 11 (filtered).
    ///
    /// Defaults to `None`, which selects version 3 for compatibility.
    pub layout_version: Option<u8>,
}

#[cfg(feature = "alloc")]
impl Default for DatasetCreationProps {
    fn default() -> Self {
        Self {
            layout: DatasetLayout::default(),
            chunk_dims: None,
            compression: Compression::None,
            fill_value: None,
            fill_time: FillTime::default(),
            alloc_time: AllocationTime::default(),
            filters: Vec::new(),
            track_attribute_order: false,
            layout_version: None,
        }
    }
}

/// Group creation property list (GCPL).
///
/// Controls link storage strategy and creation order tracking.
///
/// ## Link Storage Transitions
///
/// Groups store links in compact (object header) or dense (fractal heap +
/// B-tree v2) format. The transition thresholds control when the group
/// switches between formats:
///
/// - Compact → Dense: when link count exceeds `max_compact_links`
/// - Dense → Compact: when link count drops below `min_dense_links`
///
/// ## Defaults
///
/// | Field | Default |
/// |-------|---------|
/// | `max_compact_links` | 8 |
/// | `min_dense_links` | 6 |
/// | `track_creation_order` | `false` |
/// | `est_num_entries` | 4 |
/// | `est_link_name_length` | 8 |
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GroupCreationProps {
    /// Maximum number of links stored in compact (object header) format.
    ///
    /// When the link count exceeds this threshold, the group converts
    /// to dense storage (fractal heap + B-tree v2).
    pub max_compact_links: u16,
    /// Minimum number of links before reverting from dense to compact storage.
    ///
    /// Must be less than `max_compact_links` to prevent oscillation.
    pub min_dense_links: u16,
    /// Track link creation order.
    ///
    /// When `true`, links are indexed by insertion order in addition
    /// to name order, stored in a creation order B-tree.
    pub track_creation_order: bool,
    /// Estimated number of links (used for initial allocation sizing).
    pub est_num_entries: u16,
    /// Estimated average link name length in bytes (used for initial
    /// local heap allocation sizing).
    pub est_link_name_length: u16,
}

#[cfg(feature = "alloc")]
impl Default for GroupCreationProps {
    fn default() -> Self {
        Self {
            max_compact_links: 8,
            min_dense_links: 6,
            track_creation_order: false,
            est_num_entries: 4,
            est_link_name_length: 8,
        }
    }
}

/// File creation property list (FCPL).
///
/// Controls superblock format, address widths, and B-tree parameters
/// for new HDF5 files.
///
/// ## Defaults
///
/// | Field | Default | Notes |
/// |-------|---------|-------|
/// | `superblock_version` | 2 | Compact format with checksum |
/// | `offset_size` | 8 | 8-byte file offsets (up to 2^64 bytes) |
/// | `length_size` | 8 | 8-byte lengths |
/// | `group_leaf_k` | 0 | Not used in v2/v3 superblocks |
/// | `group_internal_k` | 0 | Not used in v2/v3 superblocks |
#[derive(Debug, Clone)]
pub struct FileCreationProps {
    /// Superblock version (0, 1, 2, or 3).
    ///
    /// - 0: Original format
    /// - 1: Added shared message table
    /// - 2: Compact format with checksum (recommended)
    /// - 3: Adds file space management
    pub superblock_version: u8,
    /// Size of file offsets in bytes (2, 4, or 8).
    ///
    /// Determines the maximum addressable file size:
    /// - 2: up to 64 KiB
    /// - 4: up to 4 GiB
    /// - 8: up to 16 EiB
    pub offset_size: u8,
    /// Size of file lengths in bytes (2, 4, or 8).
    pub length_size: u8,
    /// Group leaf node K value for v1 B-trees.
    ///
    /// Only used with superblock versions 0 and 1. Set to 0 for
    /// superblock version 2 or 3 (which use B-tree v2).
    pub group_leaf_k: u16,
    /// Group internal node K value for v1 B-trees.
    ///
    /// Only used with superblock versions 0 and 1. Set to 0 for
    /// superblock version 2 or 3.
    pub group_internal_k: u16,
}

impl Default for FileCreationProps {
    fn default() -> Self {
        Self {
            superblock_version: 2,
            offset_size: 8,
            length_size: 8,
            group_leaf_k: 0,
            group_internal_k: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `DatasetLayout` default is `Contiguous`.
    #[test]
    fn dataset_layout_default() {
        assert_eq!(DatasetLayout::default(), DatasetLayout::Contiguous);
    }

    /// Verify `FillTime` default is `IfSet`.
    #[test]
    fn fill_time_default() {
        assert_eq!(FillTime::default(), FillTime::IfSet);
    }

    /// Verify `AllocationTime` default is `Late`.
    #[test]
    fn allocation_time_default() {
        assert_eq!(AllocationTime::default(), AllocationTime::Late);
    }

    /// Verify `FileCreationProps` default field values.
    #[test]
    fn file_creation_props_defaults() {
        let fcpl = FileCreationProps::default();
        assert_eq!(fcpl.superblock_version, 2);
        assert_eq!(fcpl.offset_size, 8);
        assert_eq!(fcpl.length_size, 8);
        assert_eq!(fcpl.group_leaf_k, 0);
        assert_eq!(fcpl.group_internal_k, 0);
    }

    /// Verify `DatasetCreationProps` default field values.
    #[cfg(feature = "alloc")]
    #[test]
    fn dataset_creation_props_defaults() {
        let dcpl = DatasetCreationProps::default();
        assert_eq!(dcpl.layout, DatasetLayout::Contiguous);
        assert!(dcpl.chunk_dims.is_none());
        assert!(matches!(dcpl.compression, Compression::None));
        assert!(dcpl.fill_value.is_none());
        assert_eq!(dcpl.fill_time, FillTime::IfSet);
        assert_eq!(dcpl.alloc_time, AllocationTime::Late);
        assert!(dcpl.filters.is_empty());
        assert!(!dcpl.track_attribute_order);
    }

    /// Verify `GroupCreationProps` default field values.
    #[cfg(feature = "alloc")]
    #[test]
    fn group_creation_props_defaults() {
        let gcpl = GroupCreationProps::default();
        assert_eq!(gcpl.max_compact_links, 8);
        assert_eq!(gcpl.min_dense_links, 6);
        assert!(!gcpl.track_creation_order);
        assert_eq!(gcpl.est_num_entries, 4);
        assert_eq!(gcpl.est_link_name_length, 8);
    }

    /// Verify `min_dense_links < max_compact_links` in defaults
    /// to prevent oscillation between storage modes.
    #[cfg(feature = "alloc")]
    #[test]
    fn group_creation_props_hysteresis_invariant() {
        let gcpl = GroupCreationProps::default();
        assert!(
            gcpl.min_dense_links < gcpl.max_compact_links,
            "min_dense_links ({}) must be strictly less than max_compact_links ({}) \
             to prevent storage mode oscillation",
            gcpl.min_dense_links,
            gcpl.max_compact_links,
        );
    }
}
