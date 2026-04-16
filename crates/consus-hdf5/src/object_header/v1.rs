//! Version 1 object header parser.
//!
//! ## Specification (HDF5 File Format Specification, Section IV.A.1)
//!
//! ### Version 1 Object Header Layout
//!
//! | Offset | Size | Field                                      |
//! |--------|------|--------------------------------------------|
//! | 0      | 1    | Version (always 1)                         |
//! | 1      | 1    | Reserved                                   |
//! | 2      | 2    | Number of header messages (little-endian)  |
//! | 4      | 4    | Object reference count (little-endian)     |
//! | 8      | 4    | Object header data size (little-endian)    |
//! | 12     | var  | Header messages (concatenated)             |
//!
//! ### Version 1 Header Message Layout
//!
//! | Offset | Size      | Field                                 |
//! |--------|-----------|---------------------------------------|
//! | 0      | 2         | Message type (little-endian)          |
//! | 2      | 2         | Message data size (little-endian)     |
//! | 4      | 1         | Flags                                 |
//! | 5      | 3         | Reserved                              |
//! | 8      | data_size | Message data (padded to 8-byte align) |
//!
//! ### Continuation Messages (type 0x0010)
//!
//! When a continuation message is encountered, its data contains:
//!
//! | Offset       | Size          | Field                          |
//! |--------------|---------------|--------------------------------|
//! | 0            | offset_size   | Continuation block address     |
//! | offset_size  | length_size   | Continuation block length      |
//!
//! The continuation block contains additional header messages in the
//! same format (no prefix header, raw message entries only).
//!
//! ## Invariants
//!
//! - All messages are padded to 8-byte alignment boundaries.
//! - The total consumed size of messages within each block equals the
//!   declared header data size (for the initial block) or the continuation
//!   length (for continuation blocks).
//! - NIL messages (type 0x0000) are skipped and not included in output.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};
use consus_core::{Error, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;
use crate::constants::UNDEFINED_ADDRESS;
use crate::object_header::message_types::CONTINUATION;
use crate::object_header::{HeaderMessage, ObjectHeader};

/// Fixed size of the version 1 object header prefix (before messages).
///
/// Layout: version(1) + reserved(1) + num_messages(2) + ref_count(4) + header_size(4) = 12.
const V1_HEADER_PREFIX_SIZE: usize = 12;

/// Fixed size of a version 1 message entry header (before message data).
///
/// Layout: type(2) + data_size(2) + flags(1) + reserved(3) = 8.
const V1_MESSAGE_HEADER_SIZE: usize = 8;

/// Alignment boundary for version 1 header messages.
const V1_MESSAGE_ALIGNMENT: u64 = 8;

/// Maximum number of continuation hops to follow before declaring corruption.
///
/// Prevents infinite loops from circular continuation chains in malformed files.
/// Value chosen to exceed any realistic HDF5 object header chain depth while
/// remaining bounded. The HDF5 specification does not define an explicit limit,
/// but practical files rarely exceed single-digit continuation depths.
const MAX_CONTINUATION_DEPTH: usize = 256;

/// Parse a version 1 object header at `address` from `source`.
///
/// Reads the 12-byte header prefix, then iterates over message entries
/// within the declared header data region. Continuation messages (type
/// 0x0010) are followed recursively up to [`MAX_CONTINUATION_DEPTH`]
/// hops to collect all messages.
///
/// ## Arguments
///
/// - `source`: Random-access byte source implementing [`ReadAt`].
/// - `address`: Byte offset of the object header within the source.
/// - `ctx`: Parsing context carrying superblock-derived offset/length sizes.
///
/// ## Returns
///
/// An [`ObjectHeader`] with `version = 1` and all collected [`HeaderMessage`]
/// entries (excluding NIL messages and the continuation messages themselves,
/// whose payloads are consumed for navigation).
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] if the version byte is not 1.
/// - [`Error::Corrupted`] if message sizes exceed the declared block size,
///   the continuation chain exceeds [`MAX_CONTINUATION_DEPTH`], or a
///   continuation address is [`UNDEFINED_ADDRESS`].
#[cfg(feature = "alloc")]
pub(crate) fn parse<R: ReadAt>(
    source: &R,
    address: u64,
    ctx: &ParseContext,
) -> Result<ObjectHeader> {
    let mut prefix_buf = [0u8; V1_HEADER_PREFIX_SIZE];
    source.read_at(address, &mut prefix_buf)?;

    let version = prefix_buf[0];
    if version != 1 {
        return Err(Error::InvalidFormat {
            message: alloc::format!("expected object header version 1, found {version}"),
        });
    }
    // prefix_buf[1] is reserved.

    let _num_messages = LittleEndian::read_u16(&prefix_buf[2..4]);
    let _ref_count = LittleEndian::read_u32(&prefix_buf[4..8]);
    let header_data_size = LittleEndian::read_u32(&prefix_buf[8..12]) as u64;

    let messages_start = address + V1_HEADER_PREFIX_SIZE as u64;

    let mut messages: Vec<HeaderMessage> = Vec::new();

    // Collect continuation targets as (address, length) pairs.
    // Start with the initial message block.
    let mut pending_blocks: Vec<(u64, u64)> = Vec::new();
    pending_blocks.push((messages_start, header_data_size));

    let mut blocks_processed: usize = 0;

    while let Some((block_addr, block_len)) = pending_blocks.pop() {
        if blocks_processed >= MAX_CONTINUATION_DEPTH {
            return Err(Error::Corrupted {
                message: alloc::format!(
                    "object header continuation chain exceeded {MAX_CONTINUATION_DEPTH} blocks"
                ),
            });
        }
        blocks_processed += 1;

        parse_message_block(
            source,
            block_addr,
            block_len,
            ctx,
            &mut messages,
            &mut pending_blocks,
        )?;
    }

    Ok(ObjectHeader {
        version: 1,
        messages,
    })
}

/// Parse a contiguous block of version 1 header messages.
///
/// Iterates message entries starting at `block_addr` for up to `block_len`
/// bytes. Each message entry consists of an 8-byte header followed by
/// `data_size` bytes of payload, padded to 8-byte alignment.
///
/// Continuation messages are not added to `messages`; instead their
/// target (address, length) pairs are appended to `pending_blocks`
/// for subsequent processing.
///
/// NIL messages (type 0x0000) are skipped.
#[cfg(feature = "alloc")]
fn parse_message_block<R: ReadAt>(
    source: &R,
    block_addr: u64,
    block_len: u64,
    ctx: &ParseContext,
    messages: &mut Vec<HeaderMessage>,
    pending_blocks: &mut Vec<(u64, u64)>,
) -> Result<()> {
    let mut cursor: u64 = 0;

    while cursor + V1_MESSAGE_HEADER_SIZE as u64 <= block_len {
        let abs_pos = block_addr + cursor;

        // Read the 8-byte message header.
        let mut msg_hdr = [0u8; V1_MESSAGE_HEADER_SIZE];
        source.read_at(abs_pos, &mut msg_hdr)?;

        let message_type = LittleEndian::read_u16(&msg_hdr[0..2]);
        let data_size = LittleEndian::read_u16(&msg_hdr[2..4]);
        let flags = msg_hdr[4];
        // msg_hdr[5..8] reserved.

        cursor += V1_MESSAGE_HEADER_SIZE as u64;

        let data_size_u64 = data_size as u64;

        // Validate that the message data fits within the block.
        if cursor + data_size_u64 > block_len {
            return Err(Error::Corrupted {
                message: alloc::format!(
                    "v1 header message data (type 0x{message_type:04X}, size {data_size}) \
                     overflows block at offset {cursor} (block length {block_len})"
                ),
            });
        }

        // Read message payload.
        let mut data = alloc::vec![0u8; data_size as usize];
        if !data.is_empty() {
            source.read_at(block_addr + cursor, &mut data)?;
        }

        // Advance cursor past data, then align to 8 bytes.
        let unaligned = cursor + data_size_u64;
        let aligned = align_to_v1_boundary(unaligned);
        cursor = aligned;

        if message_type == CONTINUATION {
            // Continuation message: extract target address and length.
            let cont_data = &data;
            let min_cont_size = ctx.offset_bytes() + ctx.length_bytes();
            if cont_data.len() < min_cont_size {
                return Err(Error::Corrupted {
                    message: alloc::format!(
                        "v1 continuation message data too small: {} bytes, \
                         need at least {} (offset_size={}, length_size={})",
                        cont_data.len(),
                        min_cont_size,
                        ctx.offset_bytes(),
                        ctx.length_bytes(),
                    ),
                });
            }

            let cont_addr = ctx.read_offset(cont_data);
            let cont_len = ctx.read_length(&cont_data[ctx.offset_bytes()..]);

            if cont_addr == UNDEFINED_ADDRESS {
                return Err(Error::Corrupted {
                    message: alloc::string::String::from(
                        "v1 continuation message references undefined address",
                    ),
                });
            }

            pending_blocks.push((cont_addr, cont_len));
        } else if message_type != 0x0000 {
            // Skip NIL messages (type 0x0000); collect all others.
            messages.push(HeaderMessage {
                message_type,
                data_size,
                flags,
                data,
            });
        }
    }

    Ok(())
}

/// Align a byte offset upward to the version 1 message alignment boundary (8 bytes).
///
/// If `offset` is already aligned, it is returned unchanged.
///
/// ## Invariant
///
/// `align_to_v1_boundary(n) % 8 == 0` for all `n`.
const fn align_to_v1_boundary(offset: u64) -> u64 {
    let mask = V1_MESSAGE_ALIGNMENT - 1;
    (offset + mask) & !mask
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the alignment helper for the 8-byte boundary.
    #[test]
    fn alignment_boundary() {
        assert_eq!(align_to_v1_boundary(0), 0);
        assert_eq!(align_to_v1_boundary(1), 8);
        assert_eq!(align_to_v1_boundary(7), 8);
        assert_eq!(align_to_v1_boundary(8), 8);
        assert_eq!(align_to_v1_boundary(9), 16);
        assert_eq!(align_to_v1_boundary(16), 16);
        assert_eq!(align_to_v1_boundary(12), 16);
    }

    /// Constants are consistent with the HDF5 spec.
    #[test]
    fn layout_constants() {
        assert_eq!(V1_HEADER_PREFIX_SIZE, 12);
        assert_eq!(V1_MESSAGE_HEADER_SIZE, 8);
    }
}
