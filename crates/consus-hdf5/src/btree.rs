//! B-tree implementations for HDF5 (v1 and v2).
//!
//! ## Specification
//!
//! HDF5 uses B-trees to index:
//! - Group members (symbol table B-tree, type 0)
//! - Chunked dataset storage (chunk index B-tree, type 1)
//!
//! ### B-tree v1 Node Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 4 | Signature ("TREE") |
//! | 4 | 1 | Node type (0=group, 1=chunk) |
//! | 5 | 1 | Node level (0=leaf) |
//! | 6 | 2 | Entries used |
//! | 8 | S | Left sibling address |
//! | 8+S | S | Right sibling address |
//! | 8+2S | var | Keys and child pointers |
//!
//! ### B-tree v2
//!
//! Used in newer files (superblock v2+). Has a header with depth,
//! split ratios, and record counts, plus internal and leaf nodes.

/// B-tree v1 signature.
pub const BTREE_V1_SIGNATURE: [u8; 4] = *b"TREE";

/// B-tree v2 header signature.
pub const BTREE_V2_SIGNATURE: [u8; 4] = *b"BTHD";

/// B-tree v1 node type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTreeV1Type {
    /// Type 0: Group nodes (symbol table).
    Group,
    /// Type 1: Raw data chunk nodes.
    RawDataChunk,
}

/// Parsed B-tree v1 node header.
#[derive(Debug, Clone)]
pub struct BTreeV1Header {
    /// Node type.
    pub node_type: BTreeV1Type,
    /// Node level (0 = leaf).
    pub level: u8,
    /// Number of entries currently used.
    pub entries_used: u16,
    /// Address of left sibling (u64::MAX if none).
    pub left_sibling: u64,
    /// Address of right sibling (u64::MAX if none).
    pub right_sibling: u64,
}

/// Parsed B-tree v2 header.
#[derive(Debug, Clone)]
pub struct BTreeV2Header {
    /// Record type.
    pub record_type: u8,
    /// Node size in bytes.
    pub node_size: u32,
    /// Record size in bytes.
    pub record_size: u16,
    /// Tree depth.
    pub depth: u16,
    /// Total number of records.
    pub total_records: u64,
    /// Address of root node.
    pub root_address: u64,
}
