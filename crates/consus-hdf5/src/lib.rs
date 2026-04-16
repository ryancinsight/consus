//! # consus-hdf5
//!
//! Pure-Rust implementation of the HDF5 file format.
//!
//! ## HDF5 File Format Specification
//!
//! Reference: *HDF5 File Format Specification Version 3.0*
//! (The HDF Group, <https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html>)
//!
//! ### File Structure
//!
//! An HDF5 file consists of:
//! 1. **Superblock** (offset 0, 512, 1024, or 2048): file metadata and root group address
//! 2. **B-tree nodes**: index chunked data and group members
//! 3. **Heap blocks**: store variable-length data (names, VL strings)
//! 4. **Object headers**: metadata for groups and datasets
//! 5. **Data blocks**: raw array data (contiguous or chunked)
//!
//! ## Module Hierarchy
//!
//! ```text
//! consus-hdf5
//! ├── constants        # Magic bytes, search offsets, format constants
//! ├── primitives       # Low-level binary reading helpers
//! ├── superblock/      # Superblock parsing/writing
//! │   ├── v0_v1        # Version 0/1 superblock format
//! │   └── v2_v3        # Version 2/3 superblock format
//! ├── object_header/   # Object header parsing
//! │   └── message_types # Header message type identifiers
//! ├── btree/           # B-tree v1 and v2 implementations
//! │   ├── v1           # B-tree version 1 (group/chunk index)
//! │   └── v2           # B-tree version 2 (newer files)
//! ├── heap/            # Local and global heaps
//! │   ├── local        # Local heap (group member names)
//! │   └── global       # Global heap (variable-length objects)
//! ├── dataspace/       # Dataspace (shape) encoding
//! ├── datatype/        # HDF5 <-> canonical datatype mapping
//! │   └── classes      # HDF5 datatype class constants
//! ├── dataset/         # Dataset read/write operations
//! ├── group/           # Group operations
//! └── file/            # File-level API
//! ```
//!
//! ### Supported Superblock Versions
//!
//! - Version 0: original format
//! - Version 1: added shared message table
//! - Version 2: compact format with checksum (preferred for new files)
//! - Version 3: adds file space management

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod address;
pub mod attribute;
pub mod btree;
pub mod constants;
pub mod dataset;
pub mod dataspace;
pub mod datatype;
pub mod file;
pub mod filter;
pub mod group;
pub mod heap;
pub mod link;
pub mod object_header;
pub mod primitives;
pub mod property_list;
pub mod superblock;
