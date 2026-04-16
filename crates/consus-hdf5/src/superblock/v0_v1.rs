//! Version 0/1 superblock parsing.
//!
//! ## Layout (variable size, >= 56 bytes)
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

use byteorder::{ByteOrder, LittleEndian};
use consus_core::Result;
use consus_io::ReadAt;

use super::Superblock;
use crate::primitives::read_offset;

/// Parse version 0 or 1 superblock at the given offset.
pub(super) fn parse<R: ReadAt>(source: &R, offset: u64) -> Result<Superblock> {
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
