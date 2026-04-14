//! HDF5 Superblock parsing and writing.
//!
//! ## Specification
//!
//! The superblock is the entry point of an HDF5 file. It appears at offset
//! 0, 512, 1024, or 2048 and begins with the 8-byte magic `\x89HDF\r\n\x1a\n`.
//!
//! ### Version 0/1 Layout (variable size, ≥ 56 bytes)
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 8 | Magic |
//! | 8 | 1 | Superblock version |
//! | 9 | 1 | Free-space storage version |
//! | 10 | 1 | Root group symbol table version |
//! | 11 | 1 | Reserved |
//! | 12 | 1 | Shared header message format version |
//! | 13 | 1 | Size of offsets (bytes) |
//! | 14 | 1 | Size of lengths (bytes) |
//! | 15 | 1 | Reserved |
//! | 16 | 2 | Group leaf node K |
//! | 18 | 2 | Group internal node K |
//! | 20 | 4 | File consistency flags |
//! | 24+ | var | Addresses (base, free-space, EOF, root group) |
//!
//! ### Version 2/3 Layout (fixed size)
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 8 | Magic |
//! | 8 | 1 | Superblock version |
//! | 9 | 1 | Size of offsets |
//! | 10 | 1 | Size of lengths |
//! | 11 | 1 | File consistency flags |
//! | 12 | var | Base address |
//! | 12+S | var | Superblock extension address |
//! | 12+2S | var | End-of-file address |
//! | 12+3S | var | Root group object header address |
//! | 12+4S | 4 | Superblock checksum |

use byteorder::{ByteOrder, LittleEndian};
use consus_core::error::{Error, Result};
use consus_io::source::ReadAt;

use crate::{HDF5_MAGIC, SUPERBLOCK_SEARCH_OFFSETS};

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
        return Err(Error::InvalidFormat {});
    }

    /// Parse superblock at a known offset.
    fn parse_at<R: ReadAt>(source: &R, offset: u64) -> Result<Self> {
        // Read enough for version detection (byte 8 after magic)
        let mut header = [0u8; 12];
        source.read_at(offset, &mut header)?;

        let version = header[8];

        match version {
            0 | 1 => Self::parse_v0_v1(source, offset),
            2 | 3 => Self::parse_v2_v3(source, offset),
            _ => {
                #[cfg(feature = "alloc")]
                return Err(Error::InvalidFormat {
                    message: alloc::format!("unsupported superblock version: {version}"),
                });
                #[cfg(not(feature = "alloc"))]
                return Err(Error::InvalidFormat {});
            }
        }
    }

    /// Parse version 0 or 1 superblock.
    fn parse_v0_v1<R: ReadAt>(source: &R, offset: u64) -> Result<Self> {
        // Read the full v0/v1 header (we need at least 64 bytes for 8-byte offsets)
        let mut buf = [0u8; 96];
        source.read_at(offset, &mut buf)?;

        let offset_size = buf[13];
        let length_size = buf[14];
        let group_leaf_k = LittleEndian::read_u16(&buf[16..18]);
        let group_internal_k = LittleEndian::read_u16(&buf[18..20]);
        let consistency_flags = LittleEndian::read_u32(&buf[20..24]);

        let s = offset_size as usize;
        let mut pos = 24;

        // v1 has additional indexed storage internal node K (2 bytes) + reserved (2 bytes)
        if buf[8] == 1 {
            pos += 4; // skip indexed storage K + reserved
        }

        let base_address = read_offset(&buf[pos..], s);
        pos += s;
        let _free_space_address = read_offset(&buf[pos..], s);
        pos += s;
        let eof_address = read_offset(&buf[pos..], s);
        pos += s;
        let _driver_info_address = read_offset(&buf[pos..], s);
        pos += s;

        // Root group symbol table entry follows
        // The symbol table entry contains: link name offset (S), object header address (S),
        // cache type (4), reserved (4), scratch (16)
        let _link_name_offset = read_offset(&buf[pos..], s);
        pos += s;
        let root_group_address = read_offset(&buf[pos..], s);

        Ok(Superblock {
            offset,
            version: buf[8],
            offset_size,
            length_size,
            consistency_flags,
            base_address,
            root_group_address,
            eof_address,
            group_leaf_k,
            group_internal_k,
        })
    }

    /// Parse version 2 or 3 superblock.
    fn parse_v2_v3<R: ReadAt>(source: &R, offset: u64) -> Result<Self> {
        let mut buf = [0u8; 64];
        source.read_at(offset, &mut buf)?;

        let version = buf[8];
        let offset_size = buf[9];
        let length_size = buf[10];
        let consistency_flags = buf[11] as u32;

        let s = offset_size as usize;
        let mut pos = 12;

        let base_address = read_offset(&buf[pos..], s);
        pos += s;
        let _extension_address = read_offset(&buf[pos..], s);
        pos += s;
        let eof_address = read_offset(&buf[pos..], s);
        pos += s;
        let root_group_address = read_offset(&buf[pos..], s);
        // pos += s;
        // Next 4 bytes: checksum (skipped for now)

        Ok(Superblock {
            offset,
            version,
            offset_size,
            length_size,
            consistency_flags,
            base_address,
            root_group_address,
            eof_address,
            group_leaf_k: 0,
            group_internal_k: 0,
        })
    }
}

/// Read a file offset of `size` bytes (little-endian) from a buffer.
///
/// Supports 2, 4, and 8 byte offsets as per HDF5 spec.
fn read_offset(buf: &[u8], size: usize) -> u64 {
    match size {
        2 => LittleEndian::read_u16(buf) as u64,
        4 => LittleEndian::read_u32(buf) as u64,
        8 => LittleEndian::read_u64(buf),
        _ => 0, // invalid; caught during validation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use consus_io::cursor::MemCursor;

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
