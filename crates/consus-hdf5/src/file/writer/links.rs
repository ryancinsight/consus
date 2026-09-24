//! Hard and soft link message encoders for group object headers.

#[cfg(feature = "alloc")]
use super::write_offset;
#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};
#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};
#[cfg(feature = "alloc")]
use consus_core::Result;

// ---------------------------------------------------------------------------
// Link message encoding
// ---------------------------------------------------------------------------

/// Encode a hard link message for v2 groups.
///
/// ## Layout (link message version 1)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 1 | Version (1) |
/// | 1 | 1 | Flags |
/// | var | 1..8 | Link name length |
/// | var | N | Link name |
/// | var | S | Object header address (hard link) |
#[cfg(feature = "alloc")]
pub fn encode_hard_link(name: &str, target_address: u64, ctx: &ParseContext) -> Result<Vec<u8>> {
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len();

    // Determine name length field width (flags bits 0-1)
    let (len_width, len_flags) = if name_len < 256 {
        (1usize, 0u8)
    } else if name_len < 65536 {
        (2, 1)
    } else {
        (4, 2)
    };

    // Flags: bits 0-1 = name length size, bit 3 = link type present (0 for hard = default)
    let flags = len_flags;

    let s = ctx.offset_bytes();
    let total = 1 + 1 + len_width + name_len + s; // version + flags + name_len + name + address
    let mut buf = vec![0u8; total];
    let mut pos = 0;

    buf[pos] = 1; // version
    pos += 1;
    buf[pos] = flags;
    pos += 1;

    // Name length
    match len_width {
        1 => {
            buf[pos] = name_len as u8;
            pos += 1;
        }
        2 => {
            LittleEndian::write_u16(&mut buf[pos..], name_len as u16);
            pos += 2;
        }
        4 => {
            LittleEndian::write_u32(&mut buf[pos..], name_len as u32);
            pos += 4;
        }
        _ => {}
    }

    // Name
    buf[pos..pos + name_len].copy_from_slice(name_bytes);
    pos += name_len;

    // Hard link value: object header address
    write_offset(&mut buf[pos..], s, target_address);

    Ok(buf)
}

/// Encode a soft link message.
///
/// Soft link value: 2-byte length + target path string.
#[cfg(feature = "alloc")]
pub fn encode_soft_link(name: &str, target_path: &str, _ctx: &ParseContext) -> Result<Vec<u8>> {
    let name_bytes = name.as_bytes();
    let target_bytes = target_path.as_bytes();
    let name_len = name_bytes.len();

    let (len_width, len_flags) = if name_len < 256 {
        (1usize, 0u8)
    } else if name_len < 65536 {
        (2, 1)
    } else {
        (4, 2)
    };

    // Flags: name length encoding + link type present (bit 3)
    let flags = len_flags | 0x08; // bit 3 set: link type field present
    let link_type: u8 = 1; // soft link

    let total = 1 + 1 + 1 + len_width + name_len + 2 + target_bytes.len();
    let mut buf = vec![0u8; total];
    let mut pos = 0;

    buf[pos] = 1; // version
    pos += 1;
    buf[pos] = flags;
    pos += 1;

    // Link type
    buf[pos] = link_type;
    pos += 1;

    // Name length
    match len_width {
        1 => {
            buf[pos] = name_len as u8;
            pos += 1;
        }
        2 => {
            LittleEndian::write_u16(&mut buf[pos..], name_len as u16);
            pos += 2;
        }
        4 => {
            LittleEndian::write_u32(&mut buf[pos..], name_len as u32);
            pos += 4;
        }
        _ => {}
    }

    // Name
    buf[pos..pos + name_len].copy_from_slice(name_bytes);
    pos += name_len;

    // Soft link value: 2-byte length + path
    LittleEndian::write_u16(&mut buf[pos..], target_bytes.len() as u16);
    pos += 2;
    buf[pos..pos + target_bytes.len()].copy_from_slice(target_bytes);

    Ok(buf)
}
