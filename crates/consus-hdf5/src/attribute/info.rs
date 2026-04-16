//! HDF5 Attribute Info message parser (header message type 0x0015).
//!
//! ## Specification (HDF5 File Format Specification, Section IV.A.2.x)
//!
//! The Attribute Info message records how attributes are stored for an
//! object: whether creation order is tracked, and the addresses of the
//! fractal heap and v2 B-tree structures used for dense attribute storage.
//!
//! ### Layout
//!
//! | Offset | Size | Field                                                    |
//! |--------|------|----------------------------------------------------------|
//! | 0      | 1    | Version (must be 0)                                      |
//! | 1      | 1    | Flags                                                    |
//! |        |      |   bit 0: maximum creation order index is present          |
//! |        |      |   bit 1: creation order index B-tree address is present   |
//! | 2      | 2    | Maximum creation order index (if flags bit 0 set)        |
//! | var    | S    | Fractal heap address (S = offset_size from superblock)    |
//! | var    | S    | Name index v2 B-tree address                             |
//! | var    | S    | Creation order index v2 B-tree address (if flags bit 1)   |
//!
//! ## Invariants
//!
//! - Version must be 0; any other value is a format violation.
//! - The minimum message size is 2 + S + S bytes (version + flags +
//!   fractal heap address + name B-tree address).
//! - When flags bit 0 is set, a 2-byte maximum creation order field is
//!   present between the flags byte and the fractal heap address.
//! - When flags bit 1 is set, an additional S-byte creation order B-tree
//!   address follows the name B-tree address.

use crate::address::ParseContext;
use consus_core::{Error, Result};

/// Flag bit indicating that the maximum creation order index is present.
const FLAG_MAX_CREATION_ORDER: u8 = 0x01;

/// Flag bit indicating that a creation-order-indexed v2 B-tree address
/// is present after the name-indexed B-tree address.
const FLAG_CREATION_ORDER_BTREE: u8 = 0x02;

/// Parsed attribute info message.
///
/// Contains the addresses needed to locate dense attribute storage
/// structures (fractal heap for attribute data, v2 B-trees for name
/// and optional creation-order indexing).
#[derive(Debug, Clone)]
pub struct AttributeInfo {
    /// Flags byte controlling which optional fields are present.
    pub flags: u8,
    /// Maximum creation order index assigned so far (if tracked).
    ///
    /// Present only when `flags & 0x01 != 0`. Used to assign monotonically
    /// increasing creation order indices to new attributes.
    pub max_creation_order: Option<u16>,
    /// Fractal heap address for dense attribute name/value storage.
    pub fractal_heap_address: u64,
    /// v2 B-tree address for the attribute name index.
    pub name_btree_address: u64,
    /// v2 B-tree address for the creation order index.
    ///
    /// Present only when `flags & 0x02 != 0`.
    pub creation_order_btree_address: Option<u64>,
}

impl AttributeInfo {
    /// Parse an attribute info message from raw header message bytes.
    ///
    /// ## Arguments
    ///
    /// - `data`: Raw bytes of the attribute info header message payload
    ///   (excluding the common header message envelope).
    /// - `ctx`: Parsing context carrying superblock-derived offset size.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the version is not 0.
    /// - [`Error::InvalidFormat`] if `data` is shorter than the minimum
    ///   required length implied by the flags and offset size.
    pub fn parse(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        // Minimum: version(1) + flags(1) = 2 bytes before variable fields.
        if data.len() < 2 {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::string::String::from(
                    "attribute info message too short for version and flags",
                ),
            });
        }

        let version = data[0];
        if version != 0 {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "unsupported attribute info message version: {version}, expected 0"
                ),
            });
        }

        let flags = data[1];
        let has_max_creation_order = (flags & FLAG_MAX_CREATION_ORDER) != 0;
        let has_creation_order_btree = (flags & FLAG_CREATION_ORDER_BTREE) != 0;

        let mut cursor: usize = 2;

        // Optional: 2-byte maximum creation order index.
        let max_creation_order = if has_max_creation_order {
            if cursor + 2 > data.len() {
                return Err(Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: alloc::string::String::from(
                        "attribute info message truncated at max creation order field",
                    ),
                });
            }
            let val = u16::from_le_bytes([data[cursor], data[cursor + 1]]);
            cursor += 2;
            Some(val)
        } else {
            None
        };

        let s = ctx.offset_bytes();

        // Fractal heap address (S bytes).
        if cursor + s > data.len() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::string::String::from(
                    "attribute info message truncated at fractal heap address",
                ),
            });
        }
        let fractal_heap_address = ctx.read_offset(&data[cursor..]);
        cursor += s;

        // Name index v2 B-tree address (S bytes).
        if cursor + s > data.len() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::string::String::from(
                    "attribute info message truncated at name B-tree address",
                ),
            });
        }
        let name_btree_address = ctx.read_offset(&data[cursor..]);
        cursor += s;

        // Optional: creation order index v2 B-tree address (S bytes).
        let creation_order_btree_address = if has_creation_order_btree {
            if cursor + s > data.len() {
                return Err(Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: alloc::string::String::from(
                        "attribute info message truncated at creation order B-tree address",
                    ),
                });
            }
            let addr = ctx.read_offset(&data[cursor..]);
            Some(addr)
        } else {
            None
        };

        Ok(Self {
            flags,
            max_creation_order,
            fractal_heap_address,
            name_btree_address,
            creation_order_btree_address,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal attribute info message with no optional fields.
    ///
    /// Layout: version(0) + flags(0x00) + heap_addr(8) + name_btree(8) = 18 bytes.
    #[test]
    fn parse_minimal_no_optional_fields() {
        let ctx = ParseContext::new(8, 8);
        let mut buf = [0u8; 18];
        buf[0] = 0; // version
        buf[1] = 0x00; // flags: no optional fields
        // fractal heap address = 0x1000 (little-endian, 8 bytes)
        buf[2] = 0x00;
        buf[3] = 0x10;
        // name btree address = 0x2000 (little-endian, 8 bytes)
        buf[10] = 0x00;
        buf[11] = 0x20;

        let info = AttributeInfo::parse(&buf, &ctx).unwrap();
        assert_eq!(info.flags, 0x00);
        assert!(info.max_creation_order.is_none());
        assert_eq!(info.fractal_heap_address, 0x1000);
        assert_eq!(info.name_btree_address, 0x2000);
        assert!(info.creation_order_btree_address.is_none());
    }

    /// Parse with both optional fields present (flags = 0x03).
    ///
    /// Layout: version(1) + flags(1) + max_creation_order(2) +
    ///         heap_addr(8) + name_btree(8) + order_btree(8) = 28 bytes.
    #[test]
    fn parse_all_optional_fields() {
        let ctx = ParseContext::new(8, 8);
        let mut buf = [0u8; 28];
        buf[0] = 0; // version
        buf[1] = 0x03; // flags: both bits set
        // max creation order = 42 (little-endian)
        buf[2] = 42;
        buf[3] = 0;
        // fractal heap address = 0x3000 at offset 4
        buf[4] = 0x00;
        buf[5] = 0x30;
        // name btree address = 0x4000 at offset 12
        buf[12] = 0x00;
        buf[13] = 0x40;
        // creation order btree address = 0x5000 at offset 20
        buf[20] = 0x00;
        buf[21] = 0x50;

        let info = AttributeInfo::parse(&buf, &ctx).unwrap();
        assert_eq!(info.flags, 0x03);
        assert_eq!(info.max_creation_order, Some(42));
        assert_eq!(info.fractal_heap_address, 0x3000);
        assert_eq!(info.name_btree_address, 0x4000);
        assert_eq!(info.creation_order_btree_address, Some(0x5000));
    }

    /// Parse with only creation order tracking (flags bit 0 set, bit 1 clear).
    ///
    /// Layout: version(1) + flags(1) + max_creation_order(2) +
    ///         heap_addr(8) + name_btree(8) = 20 bytes.
    #[test]
    fn parse_creation_order_tracked_only() {
        let ctx = ParseContext::new(8, 8);
        let mut buf = [0u8; 20];
        buf[0] = 0; // version
        buf[1] = 0x01; // flags: max creation order present only
        // max creation order = 7
        buf[2] = 7;
        buf[3] = 0;
        // fractal heap address = 0x100 at offset 4
        buf[4] = 0x00;
        buf[5] = 0x01;
        // name btree address = 0x200 at offset 12
        buf[12] = 0x00;
        buf[13] = 0x02;

        let info = AttributeInfo::parse(&buf, &ctx).unwrap();
        assert_eq!(info.flags, 0x01);
        assert_eq!(info.max_creation_order, Some(7));
        assert_eq!(info.fractal_heap_address, 0x100);
        assert_eq!(info.name_btree_address, 0x200);
        assert!(info.creation_order_btree_address.is_none());
    }

    /// Parse with 4-byte offset size.
    #[test]
    fn parse_4byte_offsets() {
        let ctx = ParseContext::new(4, 4);
        // version(1) + flags(1) + heap_addr(4) + name_btree(4) = 10 bytes
        let mut buf = [0u8; 10];
        buf[0] = 0; // version
        buf[1] = 0x00; // flags: no optional fields
        // fractal heap address = 0xABCD at offset 2
        buf[2] = 0xCD;
        buf[3] = 0xAB;
        buf[4] = 0x00;
        buf[5] = 0x00;
        // name btree address = 0x1234 at offset 6
        buf[6] = 0x34;
        buf[7] = 0x12;
        buf[8] = 0x00;
        buf[9] = 0x00;

        let info = AttributeInfo::parse(&buf, &ctx).unwrap();
        assert_eq!(info.fractal_heap_address, 0xABCD);
        assert_eq!(info.name_btree_address, 0x1234);
    }

    /// Reject unsupported version.
    #[test]
    fn reject_invalid_version() {
        let ctx = ParseContext::new(8, 8);
        let buf = [1u8, 0x00]; // version 1 is invalid
        let err = AttributeInfo::parse(&buf, &ctx).unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    /// Reject truncated message (too short for version + flags).
    #[test]
    fn reject_truncated_header() {
        let ctx = ParseContext::new(8, 8);
        let buf = [0u8; 1];
        let err = AttributeInfo::parse(&buf, &ctx).unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }

    /// Reject truncated message when fractal heap address is incomplete.
    #[test]
    fn reject_truncated_at_heap_address() {
        let ctx = ParseContext::new(8, 8);
        // version(1) + flags(1) = 2 bytes, but heap address needs 8 more
        let buf = [0u8; 5];
        let err = AttributeInfo::parse(&buf, &ctx).unwrap_err();
        match err {
            Error::InvalidFormat { .. } => {}
            other => panic!("expected InvalidFormat, got: {other:?}"),
        }
    }
}
