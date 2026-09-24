//! B-tree version 2 types, structures, and parsing.
//!
//! ## Specification (HDF5 File Format Specification, Section III.A.2)
//!
//! Used in newer files (superblock v2+). B-tree v2 provides a more
//! compact and efficient indexing structure than v1, with explicit
//! record types, configurable node sizes, and depth tracking.
//!
//! ### B-tree v2 Header Layout (signature "BTHD")
//!
//! | Offset | Size | Field                              |
//! |--------|------|------------------------------------|
//! | 0      | 4    | Signature ("BTHD")                 |
//! | 4      | 1    | Version (0)                        |
//! | 5      | 1    | Type (record type identifier)      |
//! | 6      | 4    | Node size in bytes                 |
//! | 10     | 2    | Record size in bytes               |
//! | 12     | 2    | Depth                              |
//! | 14     | 1    | Split percent                      |
//! | 15     | 1    | Merge percent                      |
//! | 16     | S    | Root node address (offset_size)    |
//! | 16+S   | 2    | Number of records in root node     |
//! | 18+S   | S    | Total number of records in tree    |
//! | 18+2S  | 4    | Checksum                           |
//!
//! ### B-tree v2 Record Types
//!
//! | Type | Description                                     |
//! |------|-------------------------------------------------|
//! | 1    | Shared object header message (v1)               |
//! | 2    | Shared object header message (v2)               |
//! | 3    | Unsorted, non-filtered, non-paged              |
//! | 4    | Unsorted, filtered, non-paged                  |
//! | 5    | Link name (for indexed group)                   |
//! | 6    | Creation order (for indexed group)              |
//! | 7    | Shared header message, sorted by ref count      |
//! | 8    | Attribute name (for indexed attributes)         |
//! | 9    | Attribute creation order                        |
//! | 10   | Chunked data, non-filtered, non-paged (v4 layout) |
//! | 11   | Chunked data, filtered, non-paged (v4 layout)  |
//!
//! ### B-tree v2 Internal Node Layout (signature "BTIN")
//!
//! | Offset | Size | Field                              |
//! |--------|------|------------------------------------|
//! | 0      | 4    | Signature ("BTIN")                 |
//! | 4      | 1    | Version (0)                        |
//! | 5      | 1    | Type                               |
//! | 6      | var  | Records (record_count × record_size) |
//! | var    | var  | Child pointers and record counts   |
//! | var    | 4    | Checksum                           |
//!
//! ### B-tree v2 Leaf Node Layout (signature "BTLF")
//!
//! | Offset | Size | Field                              |
//! |--------|------|------------------------------------|
//! | 0      | 4    | Signature ("BTLF")                 |
//! | 4      | 1    | Version (0)                        |
//! | 5      | 1    | Type                               |
//! | 6      | var  | Records (record_count × record_size) |
//! | var    | 4    | Checksum                           |

/// B-tree v2 header signature.
pub const BTREE_V2_SIGNATURE: [u8; 4] = *b"BTHD";

/// B-tree v2 internal node signature.
pub const BTREE_V2_INTERNAL_SIGNATURE: [u8; 4] = *b"BTIN";

/// B-tree v2 leaf node signature.
pub const BTREE_V2_LEAF_SIGNATURE: [u8; 4] = *b"BTLF";

/// Smallest node a B-tree v2 leaf or internal node can occupy:
/// signature(4) + version(1) + type(1) + checksum(4).
///
/// A `node_size` below this cannot hold the fields the parser then indexes,
/// so it is rejected up front rather than panicking on `buf[0..4]`.
pub(crate) const MIN_NODE_BYTES: usize = 10;

mod collect;
mod header;
mod huge;
mod internal;
mod leaf;
pub mod record_type;
#[cfg(test)]
mod tests;
mod util;

#[cfg(all(feature = "async", feature = "alloc"))]
pub use collect::async_collect_all_records;
pub use collect::collect_all_records;
pub use header::BTreeV2Header;
pub use huge::{HugeObjectLocation, find_huge_object_record};
#[cfg(feature = "alloc")]
pub use internal::BTreeV2InternalNode;
#[cfg(feature = "alloc")]
pub use leaf::{BTreeV2LeafNode, BTreeV2Record};
pub(crate) use util::{compute_num_records_width, read_variable_width_uint};
