//! Version 2 object header parser.
//!
//! ## Specification (HDF5 File Format Specification, Section IV.A.2)
//!
//! Version 2 object headers use a chunk-based layout with explicit signatures
//! and CRC-32 checksums. The first chunk starts with the `OHDR` signature;
//! continuation chunks start with the `OCHK` signature.
//!
//! ### Version 2 Object Header (first chunk)
//!
//! | Offset | Size      | Field                                                                  |
//! |--------|-----------|------------------------------------------------------------------------|
//! | 0      | 4         | Signature `"OHDR"`                                                     |
//! | 4      | 1         | Version (2)                                                            |
//! | 5      | 1         | Flags                                                                  |
//! | 6      | (opt) 16  | Timestamps: access, modification, change, birth (4 bytes each, bit 5)  |
//! | var    | (opt) 4   | Max compact attrs (2) + min dense attrs (2) (bit 4)                    |
//! | var    | 1/2/4/8   | Chunk 0 data size (width from flags bits 0–1)                          |
//! | var    | var       | Header messages                                                        |
//! | var    | 4         | CRC-32 checksum over entire chunk from signature                       |
//!
//! ### Version 2 Header Message
//!
//! | Offset | Size | Field                                                      |
//! |--------|------|------------------------------------------------------------|
//! | 0      | 2    | Message type (LE u16)                                      |
//! | 2      | 2    | Message data size (LE u16)                                 |
//! | 4      | 1    | Flags                                                      |
//! | 5      | 2    | Creation order (LE u16, present only if header flags bit 2) |
//! | var    | size | Message data                                               |
//!
//! ### Continuation Chunk (`OCHK`)
//!
//! | Offset | Size | Field                                          |
//! |--------|------|------------------------------------------------|
//! | 0      | 4    | Signature `"OCHK"`                               |
//! | 4      | var  | Header messages                                  |
//! | var    | 4    | CRC-32 checksum over entire chunk from signature |
//!
//! ### Flags Encoding
//!
//! | Bit(s) | Meaning                                              |
//! |--------|------------------------------------------------------|
//! | 0–1    | Chunk data-size width: 00→1, 01→2, 10→4, 11→8 bytes |
//! | 2      | Attribute creation order tracked (messages carry 2B)  |
//! | 4      | Non-default attribute storage phase-change values     |
//! | 5      | Timestamps stored (4 × u32 = 16 bytes)               |
//!
//! ## Checksums
//!
//! Each chunk (OHDR or OCHK) is protected by a CRC-32 (IEEE 802.3) over
//! all bytes from the signature through the last byte before the stored
//! checksum field. The stored checksum is the 4-byte LE u32 immediately
//! following the message area.

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};
use consus_compression::checksum::Crc32;
use consus_core::{Error, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;
use crate::constants::UNDEFINED_ADDRESS;
use crate::object_header::message_types::CONTINUATION;
use crate::object_header::{HeaderMessage, OCHK_SIGNATURE, OHDR_SIGNATURE, ObjectHeader};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Flags bits 0–1 mask: chunk data-size width encoding.
const SIZE_WIDTH_MASK: u8 = 0b0000_0011;

/// Flags bit 2: attribute creation order tracked — each message header
/// carries an additional 2-byte creation-order field.
const FLAG_CREATION_ORDER_TRACKED: u8 = 1 << 2;

/// Flags bit 4: non-default attribute storage phase-change values present
/// (2 × u16: max compact count, min dense count).
const FLAG_ATTR_PHASE_CHANGE: u8 = 1 << 4;

/// Flags bit 5: access/modification/change/birth timestamps stored
/// (4 × u32 = 16 bytes).
const FLAG_TIMESTAMPS: u8 = 1 << 5;

/// NIL message type — padding, skipped without recording.
const MSG_TYPE_NIL: u16 = 0x0000;

/// Fixed preamble length: signature(4) + version(1) + flags(1).
const OHDR_PREAMBLE_LEN: usize = 6;

/// OCHK preamble length: signature(4).
const OCHK_PREAMBLE_LEN: usize = 4;

/// CRC-32 stored value size in bytes.
const CHECKSUM_LEN: usize = 4;

/// Maximum single-chunk allocation to prevent unbounded memory usage (64 MiB).
const MAX_CHUNK_BYTES: usize = 64 * 1024 * 1024;

/// Maximum continuation hops to prevent infinite loops from circular chains.
const MAX_CONTINUATION_DEPTH: usize = 256;

/// Base v2 message header size: type(2) + data_size(2) + flags(1).
const V2_MSG_HEADER_BASE: usize = 5;

/// Optional creation-order field size.
const V2_MSG_CREATION_ORDER_LEN: usize = 2;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse a version 2 object header starting at `address`.
///
/// Reads the OHDR first chunk, extracts the flags byte (which determines
/// per-message layout), then follows any continuation messages to OCHK
/// chunks. All non-NIL, non-continuation messages are collected into the
/// returned [`ObjectHeader`].
///
/// ## Arguments
///
/// - `source`: Random-access byte source implementing [`ReadAt`].
/// - `address`: Byte offset of the OHDR signature within `source`.
/// - `ctx`: Parsing context carrying superblock-derived offset/length sizes.
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] — signature mismatch, unexpected version,
///   chunk data-size exceeds safety limit, or message overflows chunk.
/// - [`Error::Corrupted`] — CRC-32 verification failure or continuation
///   chain exceeds [`MAX_CONTINUATION_DEPTH`].
#[cfg(feature = "alloc")]
pub(crate) fn parse<R: ReadAt>(
    source: &R,
    address: u64,
    ctx: &ParseContext,
) -> Result<ObjectHeader> {
    let mut messages: Vec<HeaderMessage> = Vec::new();
    // Continuation queue: (address, length, track_creation_order).
    let mut continuations: Vec<(u64, u64, bool)> = Vec::new();

    // ── First chunk (OHDR) ────────────────────────────────────────────
    parse_ohdr_chunk(source, address, ctx, &mut messages, &mut continuations)?;

    // ── Follow continuation chains ────────────────────────────────────
    let mut depth: usize = 0;
    while let Some((cont_addr, cont_len, track_co)) = continuations.pop() {
        if cont_addr == UNDEFINED_ADDRESS || cont_len == 0 {
            continue;
        }
        depth += 1;
        if depth > MAX_CONTINUATION_DEPTH {
            return Err(Error::Corrupted {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "v2 object header continuation chain exceeded {} hops",
                    MAX_CONTINUATION_DEPTH
                ),
            });
        }
        parse_ochk_chunk(
            source,
            cont_addr,
            cont_len,
            track_co,
            ctx,
            &mut messages,
            &mut continuations,
        )?;
    }

    Ok(ObjectHeader {
        version: 2,
        #[cfg(feature = "alloc")]
        messages,
    })
}

// ---------------------------------------------------------------------------
// First chunk (OHDR)
// ---------------------------------------------------------------------------

/// Parse the initial OHDR chunk and extract its messages.
///
/// Returns `Ok(())` after appending messages and any continuation targets.
/// The OHDR flags byte is decoded internally and the creation-order-tracked
/// bit is propagated into each continuation tuple.
#[cfg(feature = "alloc")]
fn parse_ohdr_chunk<R: ReadAt>(
    source: &R,
    address: u64,
    ctx: &ParseContext,
    messages: &mut Vec<HeaderMessage>,
    continuations: &mut Vec<(u64, u64, bool)>,
) -> Result<()> {
    // Read the fixed preamble to obtain flags.
    let mut preamble = [0u8; OHDR_PREAMBLE_LEN];
    source.read_at(address, &mut preamble)?;

    if preamble[0..4] != OHDR_SIGNATURE {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "expected OHDR signature at offset {:#x}, found {:02x?}",
                address,
                &preamble[0..4]
            ),
        });
    }

    let version = preamble[4];
    if version != 2 {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!("expected object header version 2, found {}", version),
        });
    }

    let flags = preamble[5];
    let track_creation_order = flags & FLAG_CREATION_ORDER_TRACKED != 0;

    // Compute the variable preamble length after the fixed 6 bytes.
    let mut variable_len: usize = 0;
    if flags & FLAG_TIMESTAMPS != 0 {
        variable_len += 16; // 4 × u32
    }
    if flags & FLAG_ATTR_PHASE_CHANGE != 0 {
        variable_len += 4; // 2 × u16
    }
    let size_width = chunk_data_size_width(flags)?;
    variable_len += size_width;

    // Read variable portion.
    let var_start = address + OHDR_PREAMBLE_LEN as u64;
    let mut var_buf = vec![0u8; variable_len];
    source.read_at(var_start, &mut var_buf)?;

    // The chunk data-size field is the last `size_width` bytes of the
    // variable region.
    let size_field_offset = variable_len - size_width;
    let chunk_data_size = read_chunk_data_size(&var_buf[size_field_offset..], size_width)?;

    if chunk_data_size as usize > MAX_CHUNK_BYTES {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "v2 OHDR chunk data size {} exceeds safety limit {}",
                chunk_data_size,
                MAX_CHUNK_BYTES
            ),
        });
    }

    // Total chunk bytes: preamble + variable + messages + checksum.
    let total_chunk_bytes =
        OHDR_PREAMBLE_LEN + variable_len + chunk_data_size as usize + CHECKSUM_LEN;

    // Read entire chunk into one contiguous buffer for CRC verification.
    let mut chunk_buf = vec![0u8; total_chunk_bytes];
    source.read_at(address, &mut chunk_buf)?;

    verify_checksum(&chunk_buf)?;

    // Message region within the buffer.
    let msg_start = OHDR_PREAMBLE_LEN + variable_len;
    let msg_end = msg_start + chunk_data_size as usize;

    extract_messages(
        &chunk_buf[msg_start..msg_end],
        track_creation_order,
        ctx,
        messages,
        continuations,
    )
}

// ---------------------------------------------------------------------------
// Continuation chunk (OCHK)
// ---------------------------------------------------------------------------

/// Parse an OCHK continuation chunk.
///
/// `chunk_len` is the total byte length of the chunk (signature through
/// checksum inclusive), as recorded in the continuation message data.
///
/// `track_creation_order` is inherited from the parent OHDR flags byte.
#[cfg(feature = "alloc")]
fn parse_ochk_chunk<R: ReadAt>(
    source: &R,
    address: u64,
    chunk_len: u64,
    track_creation_order: bool,
    ctx: &ParseContext,
    messages: &mut Vec<HeaderMessage>,
    continuations: &mut Vec<(u64, u64, bool)>,
) -> Result<()> {
    let total = chunk_len as usize;

    if total < OCHK_PREAMBLE_LEN + CHECKSUM_LEN {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "OCHK chunk length {} too small for preamble + checksum (minimum {})",
                total,
                OCHK_PREAMBLE_LEN + CHECKSUM_LEN
            ),
        });
    }
    if total > MAX_CHUNK_BYTES {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "OCHK chunk length {} exceeds safety limit {}",
                total,
                MAX_CHUNK_BYTES
            ),
        });
    }

    let mut chunk_buf = vec![0u8; total];
    source.read_at(address, &mut chunk_buf)?;

    if chunk_buf[0..4] != OCHK_SIGNATURE {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "expected OCHK signature at offset {:#x}, found {:02x?}",
                address,
                &chunk_buf[0..4]
            ),
        });
    }

    verify_checksum(&chunk_buf)?;

    let msg_start = OCHK_PREAMBLE_LEN;
    let msg_end = total - CHECKSUM_LEN;

    extract_messages(
        &chunk_buf[msg_start..msg_end],
        track_creation_order,
        ctx,
        messages,
        continuations,
    )
}

// ---------------------------------------------------------------------------
// Message extraction
// ---------------------------------------------------------------------------

/// Extract header messages from a raw message-data region.
///
/// Iterates the version 2 message entries within `data`. Each entry has a
/// base header of [`V2_MSG_HEADER_BASE`] bytes, optionally followed by a
/// 2-byte creation-order field (when `track_creation_order` is true).
///
/// - **NIL messages** (type 0x0000) are padding and are skipped.
/// - **Continuation messages** (type 0x0010) are decoded and their
///   `(address, length, track_creation_order)` tuples appended to
///   `continuations` for the caller to follow.
/// - All other messages are appended to `messages`.
#[cfg(feature = "alloc")]
fn extract_messages(
    data: &[u8],
    track_creation_order: bool,
    ctx: &ParseContext,
    messages: &mut Vec<HeaderMessage>,
    continuations: &mut Vec<(u64, u64, bool)>,
) -> Result<()> {
    let msg_header_len = if track_creation_order {
        V2_MSG_HEADER_BASE + V2_MSG_CREATION_ORDER_LEN
    } else {
        V2_MSG_HEADER_BASE
    };

    let mut pos: usize = 0;

    while pos + msg_header_len <= data.len() {
        let msg_type = LittleEndian::read_u16(&data[pos..]);
        let data_size = LittleEndian::read_u16(&data[pos + 2..]);
        let msg_flags = data[pos + 4];
        // Creation-order field (if present) is at data[pos + 5..pos + 7];
        // we skip it because HeaderMessage does not store it.

        pos += msg_header_len;

        let payload_end = pos + data_size as usize;
        if payload_end > data.len() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "v2 message (type {:#06x}) data size {} overflows chunk \
                     ({} bytes remaining at offset {})",
                    msg_type,
                    data_size,
                    data.len() - pos,
                    pos
                ),
            });
        }

        // NIL messages are padding — skip without recording.
        if msg_type == MSG_TYPE_NIL {
            pos = payload_end;
            continue;
        }

        let msg_data = data[pos..payload_end].to_vec();
        pos = payload_end;

        // Continuation messages: payload is offset(offset_size) + length(length_size).
        if msg_type == CONTINUATION {
            let min_size = ctx.offset_bytes() + ctx.length_bytes();
            if msg_data.len() < min_size {
                return Err(Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: alloc::format!(
                        "v2 continuation message payload {} bytes, need at least {} \
                         (offset_size={}, length_size={})",
                        msg_data.len(),
                        min_size,
                        ctx.offset_bytes(),
                        ctx.length_bytes()
                    ),
                });
            }
            let cont_addr = ctx.read_offset(&msg_data);
            let cont_len = ctx.read_length(&msg_data[ctx.offset_bytes()..]);
            continuations.push((cont_addr, cont_len, track_creation_order));
            continue;
        }

        messages.push(HeaderMessage {
            message_type: msg_type,
            data_size,
            flags: msg_flags,
            #[cfg(feature = "alloc")]
            data: msg_data,
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Determine chunk data-size field width from flags bits 0–1.
///
/// | Bits 0–1 | Width   |
/// |----------|---------|
/// | `0b00`   | 1 byte  |
/// | `0b01`   | 2 bytes |
/// | `0b10`   | 4 bytes |
/// | `0b11`   | 8 bytes |
fn chunk_data_size_width(flags: u8) -> Result<usize> {
    match flags & SIZE_WIDTH_MASK {
        0b00 => Ok(1),
        0b01 => Ok(2),
        0b10 => Ok(4),
        0b11 => Ok(8),
        // Exhaustive for a 2-bit field; unreachable in practice.
        _ => Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "invalid chunk data-size width encoding: {:#04b}",
                flags & SIZE_WIDTH_MASK
            ),
        }),
    }
}

/// Read a chunk data-size value of `width` bytes (LE) from `buf`.
fn read_chunk_data_size(buf: &[u8], width: usize) -> Result<u64> {
    if buf.len() < width {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "chunk data-size field requires {} bytes, {} available",
                width,
                buf.len()
            ),
        });
    }
    match width {
        1 => Ok(buf[0] as u64),
        2 => Ok(LittleEndian::read_u16(buf) as u64),
        4 => Ok(LittleEndian::read_u32(buf) as u64),
        8 => Ok(LittleEndian::read_u64(buf)),
        _ => Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!("unsupported chunk data-size width: {}", width),
        }),
    }
}

/// Verify the CRC-32 checksum of a complete chunk buffer.
///
/// The last [`CHECKSUM_LEN`] bytes of `chunk` are the stored checksum
/// (LE u32). The CRC-32 is computed over all preceding bytes.
///
/// ## Errors
///
/// Returns [`Error::Corrupted`] if the computed CRC does not match the
/// stored value, or if the buffer is too short to contain a checksum.
fn verify_checksum(chunk: &[u8]) -> Result<()> {
    if chunk.len() < CHECKSUM_LEN {
        return Err(Error::Corrupted {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "chunk too short for CRC-32 verification: {} bytes",
                chunk.len()
            ),
        });
    }

    let data_end = chunk.len() - CHECKSUM_LEN;
    let stored = LittleEndian::read_u32(&chunk[data_end..]);
    let computed = Crc32::compute_slice(&chunk[..data_end]);

    if computed != stored {
        return Err(Error::Corrupted {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "CRC-32 mismatch: computed {:#010x}, stored {:#010x}",
                computed,
                stored
            ),
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_data_size_width_encoding() {
        assert_eq!(chunk_data_size_width(0b0000_0000).unwrap(), 1);
        assert_eq!(chunk_data_size_width(0b0000_0001).unwrap(), 2);
        assert_eq!(chunk_data_size_width(0b0000_0010).unwrap(), 4);
        assert_eq!(chunk_data_size_width(0b0000_0011).unwrap(), 8);
        // Higher bits must not affect the result.
        assert_eq!(chunk_data_size_width(0b1111_1100).unwrap(), 1);
        assert_eq!(chunk_data_size_width(0b1111_1101).unwrap(), 2);
        assert_eq!(chunk_data_size_width(0b1111_1110).unwrap(), 4);
        assert_eq!(chunk_data_size_width(0b1111_1111).unwrap(), 8);
    }

    #[test]
    fn read_chunk_data_size_1byte() {
        assert_eq!(read_chunk_data_size(&[0x42], 1).unwrap(), 0x42);
    }

    #[test]
    fn read_chunk_data_size_2byte() {
        // 0x0100 LE = 256
        assert_eq!(read_chunk_data_size(&[0x00, 0x01], 2).unwrap(), 256);
    }

    #[test]
    fn read_chunk_data_size_4byte() {
        let mut buf = [0u8; 4];
        LittleEndian::write_u32(&mut buf, 0xDEAD_BEEF);
        assert_eq!(read_chunk_data_size(&buf, 4).unwrap(), 0xDEAD_BEEF);
    }

    #[test]
    fn read_chunk_data_size_8byte() {
        let mut buf = [0u8; 8];
        LittleEndian::write_u64(&mut buf, 0x0102_0304_0506_0708);
        assert_eq!(
            read_chunk_data_size(&buf, 8).unwrap(),
            0x0102_0304_0506_0708
        );
    }

    #[test]
    fn read_chunk_data_size_buffer_too_short() {
        assert!(read_chunk_data_size(&[0x00], 2).is_err());
    }

    #[test]
    fn verify_checksum_valid() {
        // Construct: 4 data bytes + 4 checksum bytes.
        let data = [0x01, 0x02, 0x03, 0x04];
        let crc = Crc32::compute_slice(&data);
        let mut chunk = [0u8; 8];
        chunk[..4].copy_from_slice(&data);
        LittleEndian::write_u32(&mut chunk[4..], crc);
        assert!(verify_checksum(&chunk).is_ok());
    }

    #[test]
    fn verify_checksum_invalid() {
        let mut chunk = [0u8; 8];
        chunk[0..4].copy_from_slice(&[0x01, 0x02, 0x03, 0x04]);
        LittleEndian::write_u32(&mut chunk[4..], 0xBAD0_BAD0);
        assert!(verify_checksum(&chunk).is_err());
    }

    #[test]
    fn verify_checksum_too_short() {
        assert!(verify_checksum(&[0x00, 0x01, 0x02]).is_err());
    }

    #[test]
    fn flag_constants_correct() {
        assert_eq!(FLAG_CREATION_ORDER_TRACKED, 0x04);
        assert_eq!(FLAG_ATTR_PHASE_CHANGE, 0x10);
        assert_eq!(FLAG_TIMESTAMPS, 0x20);
    }

    #[test]
    fn msg_header_len_without_creation_order() {
        // type(2) + size(2) + flags(1) = 5
        assert_eq!(V2_MSG_HEADER_BASE, 5);
    }

    #[test]
    fn msg_header_len_with_creation_order() {
        assert_eq!(V2_MSG_HEADER_BASE + V2_MSG_CREATION_ORDER_LEN, 7);
    }
}
