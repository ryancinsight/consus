//! Local heap: stores group member names.
//!
//! ## Specification (HDF5 File Format Specification, Section IV.B)
//!
//! The local heap stores variable-length strings (typically group member
//! names) referenced by byte offset from symbol table entries.
//!
//! ### Layout
//!
//! | Offset | Size | Field                                     |
//! |--------|------|-------------------------------------------|
//! | 0      | 4    | Signature `"HEAP"`                        |
//! | 4      | 1    | Version (0)                               |
//! | 5      | 3    | Reserved                                  |
//! | 8      | L    | Data segment size (length_size bytes)      |
//! | 8+L    | L    | Free list head offset within data segment |
//! | 8+2L   | S    | Address of data segment (offset_size bytes)|
//!
//! where `S` = offset_size and `L` = length_size from the superblock.
//!
//! ### Invariants
//!
//! - Signature must be exactly `b"HEAP"`.
//! - Version must be 0.
//! - `data_segment_size >= 0`.
//! - `data_address` is a valid file offset (not `UNDEFINED_ADDRESS`
//!   unless the heap is empty).
//! - Free list offset `u64::MAX` indicates an empty free list.
//! - Strings within the data segment are null-terminated.

use consus_core::{Error, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;

/// Local heap signature: ASCII `"HEAP"`.
pub const LOCAL_HEAP_SIGNATURE: [u8; 4] = *b"HEAP";

/// Parsed local heap header.
///
/// The local heap contains a contiguous data segment of variable-length
/// strings. Symbol table entries reference strings by byte offset into
/// this data segment.
#[derive(Debug, Clone)]
pub struct LocalHeap {
    /// Version (always 0 in current spec).
    pub version: u8,
    /// Total size of the data segment in bytes.
    pub data_segment_size: u64,
    /// Byte offset of the free list head within the data segment.
    ///
    /// `u64::MAX` indicates an empty free list (no free blocks).
    pub free_list_offset: u64,
    /// File address of the data segment.
    pub data_address: u64,
}

impl LocalHeap {
    /// Parse a local heap header from the given file address.
    ///
    /// Reads the signature, version, data segment size, free list offset,
    /// and data segment address from the I/O source at `address`.
    ///
    /// ## Arguments
    ///
    /// - `source`: positioned I/O source implementing `ReadAt`.
    /// - `address`: file byte offset of the local heap header.
    /// - `ctx`: parsing context carrying offset/length sizes.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature is not `"HEAP"`.
    /// - [`Error::InvalidFormat`] if the version is not 0.
    /// - [`Error::InvalidFormat`] if the header is truncated.
    /// - I/O errors propagated from the source.
    #[cfg(feature = "alloc")]
    pub fn parse<R: ReadAt>(source: &R, address: u64, ctx: &ParseContext) -> Result<Self> {
        let s = ctx.offset_bytes();
        let l = ctx.length_bytes();

        // Header size: signature(4) + version(1) + reserved(3) + data_seg_size(L)
        //              + free_list_offset(L) + data_address(S)
        let header_size = 4 + 1 + 3 + l + l + s;

        let mut buf = alloc::vec![0u8; header_size];
        source.read_at(address, &mut buf)?;

        // Validate signature.
        if buf[0..4] != LOCAL_HEAP_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "invalid local heap signature at offset 0x{:x}: expected {:?}, got {:?}",
                    address,
                    LOCAL_HEAP_SIGNATURE,
                    &buf[0..4],
                ),
            });
        }

        // Validate version.
        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: alloc::format!("unsupported local heap version: {version}, expected 0"),
            });
        }

        // Bytes 5..8 are reserved (ignored).

        let mut pos = 8;

        // Data segment size (L bytes, little-endian).
        let data_segment_size = ctx.read_length(&buf[pos..]);
        pos += l;

        // Free list head offset (L bytes, little-endian).
        let free_list_offset = ctx.read_length(&buf[pos..]);
        pos += l;

        // Data segment address (S bytes, little-endian).
        let data_address = ctx.read_offset(&buf[pos..]);
        let _ = pos + s; // consumed

        Ok(Self {
            version,
            data_segment_size,
            free_list_offset,
            data_address,
        })
    }

    /// Read a null-terminated string from the data segment at the given
    /// byte offset.
    ///
    /// ## Arguments
    ///
    /// - `source`: positioned I/O source.
    /// - `offset`: byte offset within the data segment (from a symbol
    ///   table entry's `name_offset` field).
    ///
    /// ## Algorithm
    ///
    /// 1. Compute the file address: `data_address + offset`.
    /// 2. Read bytes until a null terminator is found or a safety
    ///    limit (4096 bytes) is reached.
    /// 3. Interpret the bytes as UTF-8 (with lossy fallback).
    ///
    /// ## Errors
    ///
    /// - I/O errors from the source.
    /// - [`Error::InvalidFormat`] if no null terminator is found within
    ///   the safety limit.
    #[cfg(feature = "alloc")]
    pub fn read_name<R: ReadAt>(&self, source: &R, offset: u64) -> Result<alloc::string::String> {
        use alloc::string::String;

        // Safety limit to prevent unbounded reads on corrupt data.
        const MAX_NAME_LEN: usize = 4096;

        let file_offset = self.data_address + offset;

        // Read a chunk that's likely large enough for most names.
        let read_size = MAX_NAME_LEN.min(self.data_segment_size as usize);
        if read_size == 0 {
            return Ok(String::new());
        }

        let mut buf = alloc::vec![0u8; read_size];
        // Clamp the read to the data segment boundary.
        let remaining = self.data_segment_size.saturating_sub(offset) as usize;
        let actual_read = read_size.min(remaining);
        if actual_read == 0 {
            return Ok(String::new());
        }

        source.read_at(file_offset, &mut buf[..actual_read])?;

        // Find null terminator.
        let end = buf[..actual_read]
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| Error::InvalidFormat {
                message: alloc::format!(
                    "local heap name at offset {} not null-terminated within {} bytes",
                    offset,
                    actual_read,
                ),
            })?;

        core::str::from_utf8(&buf[..end])
            .map(|s| String::from(s))
            .map_err(|_| Error::InvalidFormat {
                message: alloc::format!("local heap name at offset {} is not valid UTF-8", offset,),
            })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the local heap signature constant.
    #[test]
    fn signature_value() {
        assert_eq!(&LOCAL_HEAP_SIGNATURE, b"HEAP");
    }

    /// Parse a valid local heap with 8-byte offset/length sizes.
    #[cfg(feature = "alloc")]
    #[test]
    fn parse_valid_8byte() {
        use byteorder::{ByteOrder, LittleEndian};
        use consus_io::MemCursor;

        let ctx = ParseContext::new(8, 8);

        // Build a valid local heap header.
        // signature(4) + version(1) + reserved(3) + data_seg_size(8)
        // + free_list(8) + data_addr(8) = 32 bytes
        let mut buf = [0u8; 32];
        buf[0..4].copy_from_slice(b"HEAP");
        buf[4] = 0; // version
        // reserved = 0
        LittleEndian::write_u64(&mut buf[8..16], 256); // data segment size
        LittleEndian::write_u64(&mut buf[16..24], u64::MAX); // free list = empty
        LittleEndian::write_u64(&mut buf[24..32], 0x100); // data address

        let cursor = MemCursor::from_bytes(buf.to_vec());
        let heap = LocalHeap::parse(&cursor, 0, &ctx).unwrap();

        assert_eq!(heap.version, 0);
        assert_eq!(heap.data_segment_size, 256);
        assert_eq!(heap.free_list_offset, u64::MAX);
        assert_eq!(heap.data_address, 0x100);
    }

    /// Parse a valid local heap with 4-byte offset/length sizes.
    #[cfg(feature = "alloc")]
    #[test]
    fn parse_valid_4byte() {
        use byteorder::{ByteOrder, LittleEndian};
        use consus_io::MemCursor;

        let ctx = ParseContext::new(4, 4);

        // signature(4) + version(1) + reserved(3) + data_seg_size(4)
        // + free_list(4) + data_addr(4) = 20 bytes
        let mut buf = [0u8; 20];
        buf[0..4].copy_from_slice(b"HEAP");
        buf[4] = 0;
        LittleEndian::write_u32(&mut buf[8..12], 128); // data segment size
        LittleEndian::write_u32(&mut buf[12..16], 0xFFFFFFFF); // free list
        LittleEndian::write_u32(&mut buf[16..20], 0x80); // data address

        let cursor = MemCursor::from_bytes(buf.to_vec());
        let heap = LocalHeap::parse(&cursor, 0, &ctx).unwrap();

        assert_eq!(heap.version, 0);
        assert_eq!(heap.data_segment_size, 128);
        assert_eq!(heap.free_list_offset, 0xFFFFFFFF);
        assert_eq!(heap.data_address, 0x80);
    }

    /// Reject invalid signature.
    #[cfg(feature = "alloc")]
    #[test]
    fn reject_bad_signature() {
        use consus_io::MemCursor;

        let ctx = ParseContext::new(8, 8);
        let mut buf = [0u8; 32];
        buf[0..4].copy_from_slice(b"XXXX"); // wrong signature
        buf[4] = 0;

        let cursor = MemCursor::from_bytes(buf.to_vec());
        let err = LocalHeap::parse(&cursor, 0, &ctx).unwrap_err();
        match err {
            Error::InvalidFormat { message } => {
                assert!(message.contains("invalid local heap signature"));
            }
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    /// Reject unsupported version.
    #[cfg(feature = "alloc")]
    #[test]
    fn reject_bad_version() {
        use consus_io::MemCursor;

        let ctx = ParseContext::new(8, 8);
        let mut buf = [0u8; 32];
        buf[0..4].copy_from_slice(b"HEAP");
        buf[4] = 1; // unsupported version

        let cursor = MemCursor::from_bytes(buf.to_vec());
        let err = LocalHeap::parse(&cursor, 0, &ctx).unwrap_err();
        match err {
            Error::InvalidFormat { message } => {
                assert!(message.contains("unsupported local heap version"));
            }
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    /// Read a null-terminated name from the data segment.
    #[cfg(feature = "alloc")]
    #[test]
    fn read_name_from_data_segment() {
        use consus_io::MemCursor;

        // Place the data segment at byte 64 in the file.
        let mut file_data = alloc::vec![0u8; 128];
        // Write "temperature\0" at data segment offset 0.
        let name = b"temperature\0";
        file_data[64..64 + name.len()].copy_from_slice(name);
        // Write "pressure\0" at data segment offset 16.
        let name2 = b"pressure\0";
        file_data[80..80 + name2.len()].copy_from_slice(name2);

        let cursor = MemCursor::from_bytes(file_data);

        let heap = LocalHeap {
            version: 0,
            data_segment_size: 64,
            free_list_offset: u64::MAX,
            data_address: 64,
        };

        let n1 = heap.read_name(&cursor, 0).unwrap();
        assert_eq!(n1, "temperature");

        let n2 = heap.read_name(&cursor, 16).unwrap();
        assert_eq!(n2, "pressure");
    }

    /// Read an empty name (null byte at the start).
    #[cfg(feature = "alloc")]
    #[test]
    fn read_empty_name() {
        use consus_io::MemCursor;

        let mut file_data = alloc::vec![0u8; 32];
        file_data[16] = 0; // null terminator at offset 0 of data segment

        let cursor = MemCursor::from_bytes(file_data);

        let heap = LocalHeap {
            version: 0,
            data_segment_size: 16,
            free_list_offset: u64::MAX,
            data_address: 16,
        };

        let name = heap.read_name(&cursor, 0).unwrap();
        assert_eq!(name, "");
    }

    /// Parse a heap at a non-zero file offset.
    #[cfg(feature = "alloc")]
    #[test]
    fn parse_at_nonzero_offset() {
        use byteorder::{ByteOrder, LittleEndian};
        use consus_io::MemCursor;

        let ctx = ParseContext::new(8, 8);
        let file_offset = 512u64;

        let mut file_data = alloc::vec![0u8; 1024];
        let heap_buf = &mut file_data[file_offset as usize..file_offset as usize + 32];
        heap_buf[0..4].copy_from_slice(b"HEAP");
        heap_buf[4] = 0;
        LittleEndian::write_u64(&mut heap_buf[8..16], 64);
        LittleEndian::write_u64(&mut heap_buf[16..24], 8);
        LittleEndian::write_u64(&mut heap_buf[24..32], 0x300);

        let cursor = MemCursor::from_bytes(file_data);
        let heap = LocalHeap::parse(&cursor, file_offset, &ctx).unwrap();

        assert_eq!(heap.data_segment_size, 64);
        assert_eq!(heap.free_list_offset, 8);
        assert_eq!(heap.data_address, 0x300);
    }
}
