//! B-tree implementations for HDF5 (v1 and v2).
//!
//! ## Specification
//!
//! HDF5 uses B-trees to index:
//! - Group members (symbol table B-tree, type 0)
//! - Chunked dataset storage (chunk index B-tree, type 1)

pub mod v1;
pub mod v2;

pub use v1::{BTREE_V1_SIGNATURE, BTreeV1Header, BTreeV1Type};
pub use v2::{
    BTREE_V2_SIGNATURE, BTreeV2Header, BTreeV2InternalNode, BTreeV2LeafNode,
    record_type as btree_v2_record_type,
};
