//! HDF5 format constants: magic bytes, search offsets, and sentinel values.

/// HDF5 format magic bytes: `\x89HDF\r\n\x1a\n`
pub const HDF5_MAGIC: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// Maximum superblock search offset (spec: 0, 512, 1024, 2048).
pub const SUPERBLOCK_SEARCH_OFFSETS: [u64; 4] = [0, 512, 1024, 2048];

/// Undefined address sentinel (all bits set).
pub const UNDEFINED_ADDRESS: u64 = u64::MAX;
