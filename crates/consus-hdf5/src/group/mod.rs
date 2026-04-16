//! HDF5 group operations.
//!
//! ## Specification
//!
//! Groups are containers that hold other groups and datasets.
//! They are represented by:
//! - **v1**: Symbol table message pointing to a B-tree + local heap
//! - **v2**: Link info + link messages, or fractal heap + B-tree v2
//!
//! This module provides the logical group interface; actual parsing
//! delegates to `btree`, `heap`, and `object_header`.

pub mod symbol_table;

/// HDF5 group handle.
///
/// Phase 1 implementation target: read group member names and
/// navigate the hierarchy.
pub struct Hdf5Group {
    /// Absolute path of this group within the file.
    #[cfg(feature = "alloc")]
    pub path: alloc::string::String,
    /// Address of this group's object header.
    pub object_header_address: u64,
}
