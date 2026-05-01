//! HDF5 Superblock parsing and writing.
//!
//! ## Specification
//!
//! The superblock is the entry point of an HDF5 file. It appears at offset
//! 0, 512, 1024, or 2048 and begins with the 8-byte magic `\x89HDF\r\n\x1a\n`.

mod v0_v1;
mod v2_v3;

use consus_core::{Error, Result};
use consus_io::ReadAt;

use crate::constants::{HDF5_MAGIC, SUPERBLOCK_SEARCH_OFFSETS};

/// Parsed HDF5 superblock.
///
/// Contains the structural parameters needed to navigate the file.
#[derive(Debug, Clone)]
pub struct Superblock {
    /// Byte offset of the superblock within the file.
    pub offset: u64,
    /// Superblock format version (0, 1, 2, or 3).
    pub version: u8,
    /// Size of file offsets in bytes (typically 8).
    pub offset_size: u8,
    /// Size of file lengths in bytes (typically 8).
    pub length_size: u8,
    /// File consistency flags.
    pub consistency_flags: u32,
    /// Base address for all other addresses in the file.
    pub base_address: u64,
    /// Address of the root group object header (v2/v3) or symbol table entry (v0/v1).
    pub root_group_address: u64,
    /// End-of-file address.
    pub eof_address: u64,
    /// Group leaf node K (v0/v1 only).
    pub group_leaf_k: u16,
    /// Group internal node K (v0/v1 only).
    pub group_internal_k: u16,
}

impl Superblock {
    /// Locate and parse the superblock from a source.
    ///
    /// Searches at offsets 0, 512, 1024, 2048 per the HDF5 spec.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidFormat` if no valid superblock is found.
    pub fn read_from<R: ReadAt>(source: &R) -> Result<Self> {
        let mut magic_buf = [0u8; 8];

        for &offset in &SUPERBLOCK_SEARCH_OFFSETS {
            if source.read_at(offset, &mut magic_buf).is_err() {
                continue;
            }
            if magic_buf != HDF5_MAGIC {
                continue;
            }
            return Self::parse_at(source, offset);
        }

        #[cfg(feature = "alloc")]
        return Err(Error::InvalidFormat {
            message: alloc::string::String::from("no HDF5 superblock found at expected offsets"),
        });
        #[cfg(not(feature = "alloc"))]
        return Err(Error::InvalidFormat { message: alloc::string::String::from("invalid superblock format") });
    }

    /// Parse superblock at a known offset.
    fn parse_at<R: ReadAt>(source: &R, offset: u64) -> Result<Self> {
        // Read enough for version detection (byte 8 after magic)
        let mut header = [0u8; 12];
        source.read_at(offset, &mut header)?;

        let version = header[8];

        match version {
            0 | 1 => v0_v1::parse(source, offset),
            2 | 3 => v2_v3::parse(source, offset),
            _ => {
                #[cfg(feature = "alloc")]
                return Err(Error::InvalidFormat {
                    message: alloc::format!("unsupported superblock version: {version}"),
                });
                #[cfg(not(feature = "alloc"))]
                return Err(Error::InvalidFormat { message: alloc::string::String::from("invalid superblock version") });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{ByteOrder, LittleEndian};
    use consus_io::MemCursor;

    /// Construct a minimal valid v2 superblock and verify parsing.
    #[test]
    fn parse_v2_superblock() {
        let mut data = vec![0u8; 64];

        // Magic
        data[0..8].copy_from_slice(&HDF5_MAGIC);
        // Version 2
        data[8] = 2;
        // Offset size = 8
        data[9] = 8;
        // Length size = 8
        data[10] = 8;
        // Consistency flags = 0
        data[11] = 0;

        // Base address = 0
        LittleEndian::write_u64(&mut data[12..20], 0);
        // Extension address = UNDEF (0xFFFFFFFFFFFFFFFF)
        LittleEndian::write_u64(&mut data[20..28], u64::MAX);
        // EOF address = 4096
        LittleEndian::write_u64(&mut data[28..36], 4096);
        // Root group OH address = 96
        LittleEndian::write_u64(&mut data[36..44], 96);
        // Checksum placeholder
        LittleEndian::write_u32(&mut data[44..48], 0);

        let cursor = MemCursor::from_bytes(data);
        let sb = Superblock::read_from(&cursor).expect("must parse v2 superblock");

        assert_eq!(sb.version, 2);
        assert_eq!(sb.offset_size, 8);
        assert_eq!(sb.length_size, 8);
        assert_eq!(sb.base_address, 0);
        assert_eq!(sb.eof_address, 4096);
        assert_eq!(sb.root_group_address, 96);
    }

    /// Verify that absence of magic at all valid offsets yields InvalidFormat.
    #[test]
    fn no_magic_returns_error() {
        let cursor = MemCursor::from_bytes(vec![0u8; 4096]);
        let result = Superblock::read_from(&cursor);
        assert!(result.is_err());
    }
}
