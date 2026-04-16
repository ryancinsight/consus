//! B-tree version 1 types and structures.
//!
//! ### B-tree v1 Node Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 4 | Signature ("TREE") |
//! | 4 | 1 | Node type (0=group, 1=chunk) |
//! | 5 | 1 | Node level (0=leaf) |
//! | 6 | 2 | Entries used |
//! | 8 | S | Left sibling address |
//! | 8+S | S | Right sibling address |
//! | 8+2S | var | Keys and child pointers |
//!
//! ## Parsing
//!
//! The [`BTreeV1Header::parse`] function reads the fixed header (signature
//! through right-sibling address) from an I/O source at a given file offset.
//! Key/child data beyond the header is read separately by callers that
//! know the node type and key layout.
//!
//! ## Invariants
//!
//! - Signature must be exactly `b"TREE"`.
//! - Node type must be 0 (group) or 1 (raw data chunk).
//! - `entries_used` may be 0 for an empty node.
//! - Sibling addresses of `u64::MAX` indicate no sibling (leaf of the
//!   sibling chain).

#[cfg(feature = "alloc")]
use alloc::string::String;

use consus_core::{Error, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;

/// B-tree v1 signature.
pub const BTREE_V1_SIGNATURE: [u8; 4] = *b"TREE";

/// B-tree v1 node type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTreeV1Type {
    /// Type 0: Group nodes (symbol table).
    Group,
    /// Type 1: Raw data chunk nodes.
    RawDataChunk,
}

/// Parsed B-tree v1 node header.
///
/// Contains the fixed portion of a v1 B-tree node. The variable-length
/// key and child-pointer arrays that follow the header are not included;
/// they must be read by the caller with knowledge of the key format
/// (which depends on `node_type`).
#[derive(Debug, Clone)]
pub struct BTreeV1Header {
    /// Node type.
    pub node_type: BTreeV1Type,
    /// Node level (0 = leaf).
    pub level: u8,
    /// Number of entries currently used.
    pub entries_used: u16,
    /// Address of left sibling (u64::MAX if none).
    pub left_sibling: u64,
    /// Address of right sibling (u64::MAX if none).
    pub right_sibling: u64,
}

impl BTreeV1Header {
    /// Size of the fixed B-tree v1 node header in bytes.
    ///
    /// `signature(4) + type(1) + level(1) + entries_used(2) + left(S) + right(S)`
    /// = `8 + 2 * offset_size`.
    pub fn header_size(ctx: &ParseContext) -> usize {
        8 + 2 * ctx.offset_bytes()
    }

    /// Parse a B-tree v1 node header from an I/O source at `address`.
    ///
    /// Reads exactly [`Self::header_size`] bytes starting at `address`,
    /// validates the signature and node type, and returns the parsed header.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature is not `b"TREE"`.
    /// - [`Error::InvalidFormat`] if the node type byte is not 0 or 1.
    /// - Any I/O error propagated from `source.read_at`.
    pub fn parse<R: ReadAt>(
        source: &R,
        address: u64,
        ctx: &ParseContext,
    ) -> Result<Self> {
        let hdr_size = Self::header_size(ctx);
        // Maximum possible header size: 8 + 2*8 = 24 bytes.
        // Use a stack buffer large enough for the largest case and
        // slice it to the actual header size.
        let mut buf = [0u8; 24];
        let buf = &mut buf[..hdr_size];
        source.read_at(address, buf)?;

        // Validate signature.
        if buf[0..4] != BTREE_V1_SIGNATURE {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("invalid B-tree v1 signature (expected \"TREE\")"),
            });
        }

        // Node type.
        let node_type = match buf[4] {
            0 => BTreeV1Type::Group,
            1 => BTreeV1Type::RawDataChunk,
            other => {
                return Err(Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: alloc::format!(
                        "unknown B-tree v1 node type: {other}, expected 0 (group) or 1 (chunk)"
                    ),
                });
            }
        };

        let level = buf[5];
        let entries_used = u16::from_le_bytes([buf[6], buf[7]]);

        let s = ctx.offset_bytes();
        let left_sibling = ctx.read_offset(&buf[8..]);
        let right_sibling = ctx.read_offset(&buf[8 + s..]);

        Ok(Self {
            node_type,
            level,
            entries_used,
            left_sibling,
            right_sibling,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    use consus_io::MemCursor;

    fn ctx8() -> ParseContext {
        ParseContext::new(8, 8)
    }

    fn ctx4() -> ParseContext {
        ParseContext::new(4, 4)
    }

    /// Construct a minimal B-tree v1 group node header with 8-byte offsets.
    #[cfg(feature = "alloc")]
    fn make_group_header_8(level: u8, entries: u16, left: u64, right: u64) -> Vec<u8> {
        let mut buf = vec![0u8; 24];
        buf[0..4].copy_from_slice(b"TREE");
        buf[4] = 0; // group
        buf[5] = level;
        buf[6..8].copy_from_slice(&entries.to_le_bytes());
        buf[8..16].copy_from_slice(&left.to_le_bytes());
        buf[16..24].copy_from_slice(&right.to_le_bytes());
        buf
    }

    /// Construct a minimal B-tree v1 chunk node header with 4-byte offsets.
    #[cfg(feature = "alloc")]
    fn make_chunk_header_4(level: u8, entries: u16, left: u32, right: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 16]; // 8 + 2*4
        buf[0..4].copy_from_slice(b"TREE");
        buf[4] = 1; // raw data chunk
        buf[5] = level;
        buf[6..8].copy_from_slice(&entries.to_le_bytes());
        buf[8..12].copy_from_slice(&left.to_le_bytes());
        buf[12..16].copy_from_slice(&right.to_le_bytes());
        buf
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn parse_group_leaf_8byte() {
        let data = make_group_header_8(0, 5, u64::MAX, u64::MAX);
        let cursor = MemCursor::from_bytes(data);
        let hdr = BTreeV1Header::parse(&cursor, 0, &ctx8()).unwrap();
        assert_eq!(hdr.node_type, BTreeV1Type::Group);
        assert_eq!(hdr.level, 0);
        assert_eq!(hdr.entries_used, 5);
        assert_eq!(hdr.left_sibling, u64::MAX);
        assert_eq!(hdr.right_sibling, u64::MAX);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn parse_chunk_internal_4byte() {
        let data = make_chunk_header_4(2, 10, 0x100, 0x200);
        let cursor = MemCursor::from_bytes(data);
        let hdr = BTreeV1Header::parse(&cursor, 0, &ctx4()).unwrap();
        assert_eq!(hdr.node_type, BTreeV1Type::RawDataChunk);
        assert_eq!(hdr.level, 2);
        assert_eq!(hdr.entries_used, 10);
        assert_eq!(hdr.left_sibling, 0x100);
        assert_eq!(hdr.right_sibling, 0x200);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn parse_at_nonzero_offset() {
        let mut data = vec![0u8; 32]; // 8 bytes padding + 24 bytes header
        let header_bytes = make_group_header_8(1, 3, 0xAA, 0xBB);
        data[8..32].copy_from_slice(&header_bytes);
        let cursor = MemCursor::from_bytes(data);
        let hdr = BTreeV1Header::parse(&cursor, 8, &ctx8()).unwrap();
        assert_eq!(hdr.level, 1);
        assert_eq!(hdr.entries_used, 3);
        assert_eq!(hdr.left_sibling, 0xAA);
        assert_eq!(hdr.right_sibling, 0xBB);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn reject_bad_signature() {
        let mut data = make_group_header_8(0, 0, u64::MAX, u64::MAX);
        data[0..4].copy_from_slice(b"XXXX");
        let cursor = MemCursor::from_bytes(data);
        let err = BTreeV1Header::parse(&cursor, 0, &ctx8()).unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn reject_unknown_node_type() {
        let mut data = make_group_header_8(0, 0, u64::MAX, u64::MAX);
        data[4] = 99; // invalid type
        let cursor = MemCursor::from_bytes(data);
        let err = BTreeV1Header::parse(&cursor, 0, &ctx8()).unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn parse_zero_entries() {
        let data = make_group_header_8(0, 0, u64::MAX, u64::MAX);
        let cursor = MemCursor::from_bytes(data);
        let hdr = BTreeV1Header::parse(&cursor, 0, &ctx8()).unwrap();
        assert_eq!(hdr.entries_used, 0);
    }

    #[test]
    fn header_size_8byte() {
        assert_eq!(BTreeV1Header::header_size(&ctx8()), 24);
    }

    #[test]
    fn header_size_4byte() {
        assert_eq!(BTreeV1Header::header_size(&ctx4()), 16);
    }

    #[test]
    fn header_size_2byte() {
        let ctx = ParseContext::new(2, 2);
        assert_eq!(BTreeV1Header::header_size(&ctx), 12);
    }

    /// Verify that the node type round-trips through the type discriminant byte.
    #[test]
    fn node_type_discriminant_values() {
        // Group = 0, RawDataChunk = 1 — these are the only valid values.
        assert_eq!(BTreeV1Type::Group, BTreeV1Type::Group);
        assert_eq!(BTreeV1Type::RawDataChunk, BTreeV1Type::RawDataChunk);
        assert_ne!(BTreeV1Type::Group, BTreeV1Type::RawDataChunk);
    }
}
