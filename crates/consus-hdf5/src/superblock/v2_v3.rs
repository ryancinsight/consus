//! Version 2/3 superblock parsing.
//!
//! ## Layout (fixed size)
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

use consus_core::Result;
use consus_io::ReadAt;

use super::Superblock;
use crate::primitives::read_offset;

/// Parse version 2 or 3 superblock at the given offset.
pub(super) fn parse<R: ReadAt>(source: &R, offset: u64) -> Result<Superblock> {
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
