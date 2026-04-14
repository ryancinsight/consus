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
//! ### Supported Superblock Versions
//!
//! - Version 0: original format
//! - Version 1: added shared message table
//! - Version 2: compact format with checksum (preferred for new files)
//! - Version 3: adds file space management
//!
//! ## Architecture
//!
//! ```text
//! consus-hdf5
//! ├── superblock  # Superblock parsing/writing
//! ├── btree       # B-tree v1 and v2 implementations
//! ├── heap        # Local and global heaps
//! ├── header      # Object header parsing
//! ├── dataspace   # Dataspace (shape) encoding
//! ├── datatype    # HDF5 ↔ canonical datatype mapping
//! ├── group       # Group operations
//! ├── dataset     # Dataset read/write
//! └── file        # File-level API
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod superblock;
pub mod btree;
pub mod heap;
pub mod object_header;
pub mod dataspace;
pub mod datatype_map;
pub mod group;
pub mod dataset;
pub mod file;

/// HDF5 format magic bytes: `\x89HDF\r\n\x1a\n`
pub const HDF5_MAGIC: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// Maximum superblock search offset (spec: 0, 512, 1024, 2048).
pub const SUPERBLOCK_SEARCH_OFFSETS: [u64; 4] = [0, 512, 1024, 2048];
