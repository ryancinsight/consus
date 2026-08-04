//! Link Info message parsing (header message type 0x0002).

use byteorder::{ByteOrder, LittleEndian};
use consus_core::{Error, Result};

use crate::address::ParseContext;

/// Parsed Link Info message (header message type 0x0002).
///
/// Records how links are stored in a v2 group: compact (inline link messages)
/// or dense (fractal heap + B-tree v2).
///
/// ### Layout (HDF5 Spec SS.IV.A.2.b)
///
/// | Offset  | Size | Field                                          |
/// |---------|------|------------------------------------------------|
/// | 0       | 1    | Version (must be 0)                            |
/// | 1       | 1    | Flags                                          |
/// | if bit0 | 8    | Maximum creation index (u64 LE)                |
/// | var     | S    | Fractal heap address                           |
/// | var     | S    | Name index v2 B-tree address                   |
/// | if bit1 | S    | Creation order index v2 B-tree address         |
///
/// When fractal_heap_address == UNDEFINED_ADDRESS, links are compact.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct LinkInfo {
    pub flags: u8,
    pub max_creation_index: Option<u64>,
    pub fractal_heap_address: u64,
    pub name_btree_address: u64,
    pub creation_order_btree_address: Option<u64>,
}

#[cfg(feature = "alloc")]
impl LinkInfo {
    pub fn parse(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::string::String::from(
                    "link info message too short for version and flags",
                ),
            });
        }

        let version = data[0];
        if version != 0 {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "unsupported link info message version: {version}, expected 0"
                ),
            });
        }

        let flags = data[1];
        let has_max_creation_index = (flags & 0x01) != 0;
        let has_creation_order_btree = (flags & 0x02) != 0;
        let mut cursor: usize = 2;

        let max_creation_index = if has_max_creation_index {
            if cursor + 8 > data.len() {
                return Err(Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: alloc::string::String::from(
                        "link info message truncated at max creation index",
                    ),
                });
            }
            let value = LittleEndian::read_u64(&data[cursor..cursor + 8]);
            cursor += 8;
            Some(value)
        } else {
            None
        };

        let offset_size = ctx.offset_bytes();
        if cursor + offset_size > data.len() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::string::String::from(
                    "link info message truncated at fractal heap address",
                ),
            });
        }
        let fractal_heap_address = ctx.read_offset(&data[cursor..]);
        cursor += offset_size;

        if cursor + offset_size > data.len() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::string::String::from(
                    "link info message truncated at name B-tree address",
                ),
            });
        }
        let name_btree_address = ctx.read_offset(&data[cursor..]);
        cursor += offset_size;

        let creation_order_btree_address = if has_creation_order_btree {
            if cursor + offset_size > data.len() {
                return Err(Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: alloc::string::String::from(
                        "link info message truncated at creation order B-tree address",
                    ),
                });
            }
            Some(ctx.read_offset(&data[cursor..]))
        } else {
            None
        };

        Ok(Self {
            flags,
            max_creation_index,
            fractal_heap_address,
            name_btree_address,
            creation_order_btree_address,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::ParseContext;

    #[cfg(feature = "alloc")]
    #[test]
    fn link_info_parse_minimal() {
        let ctx = ParseContext::new(8, 8);
        let mut buf = [0u8; 18];
        buf[0] = 0;
        buf[1] = 0x00;
        buf[2] = 0x00;
        buf[3] = 0x10;
        buf[10] = 0x00;
        buf[11] = 0x20;
        let info = LinkInfo::parse(&buf, &ctx).unwrap();
        assert_eq!(info.flags, 0x00);
        assert!(info.max_creation_index.is_none());
        assert_eq!(info.fractal_heap_address, 0x1000);
        assert_eq!(info.name_btree_address, 0x2000);
        assert!(info.creation_order_btree_address.is_none());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn link_info_parse_with_max_creation_index() {
        let ctx = ParseContext::new(8, 8);
        let mut buf = [0u8; 26];
        buf[0] = 0;
        buf[1] = 0x01;
        buf[2] = 42;
        buf[10] = 0x00;
        buf[11] = 0x30;
        buf[18] = 0x00;
        buf[19] = 0x40;
        let info = LinkInfo::parse(&buf, &ctx).unwrap();
        assert_eq!(info.flags, 0x01);
        assert_eq!(info.max_creation_index, Some(42));
        assert_eq!(info.fractal_heap_address, 0x3000);
        assert_eq!(info.name_btree_address, 0x4000);
        assert!(info.creation_order_btree_address.is_none());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn link_info_parse_all_optional_fields() {
        let ctx = ParseContext::new(8, 8);
        let mut buf = [0u8; 34];
        buf[0] = 0;
        buf[1] = 0x03;
        buf[2] = 99;
        buf[10] = 0x00;
        buf[11] = 0x50;
        buf[18] = 0x00;
        buf[19] = 0x60;
        buf[26] = 0x00;
        buf[27] = 0x70;
        let info = LinkInfo::parse(&buf, &ctx).unwrap();
        assert_eq!(info.flags, 0x03);
        assert_eq!(info.max_creation_index, Some(99));
        assert_eq!(info.fractal_heap_address, 0x5000);
        assert_eq!(info.name_btree_address, 0x6000);
        assert_eq!(info.creation_order_btree_address, Some(0x7000));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn link_info_parse_4byte_offsets() {
        let ctx = ParseContext::new(4, 4);
        let mut buf = [0u8; 10];
        buf[0] = 0;
        buf[1] = 0x00;
        buf[2] = 0xCD;
        buf[3] = 0xAB;
        buf[6] = 0x34;
        buf[7] = 0x12;
        let info = LinkInfo::parse(&buf, &ctx).unwrap();
        assert_eq!(info.fractal_heap_address, 0xABCD);
        assert_eq!(info.name_btree_address, 0x1234);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn link_info_reject_invalid_version() {
        let ctx = ParseContext::new(8, 8);
        let buf = [1u8, 0x00];
        match LinkInfo::parse(&buf, &ctx).unwrap_err() {
            Error::InvalidFormat { .. } => {}
            err => panic!("expected InvalidFormat, got {err:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn link_info_reject_truncated_header() {
        let ctx = ParseContext::new(8, 8);
        match LinkInfo::parse(&[0u8; 1], &ctx).unwrap_err() {
            Error::InvalidFormat { .. } => {}
            err => panic!("{err:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn link_info_reject_truncated_at_heap_address() {
        let ctx = ParseContext::new(8, 8);
        match LinkInfo::parse(&[0u8; 5], &ctx).unwrap_err() {
            Error::InvalidFormat { .. } => {}
            err => panic!("{err:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn link_info_reject_truncated_at_max_creation_index() {
        let ctx = ParseContext::new(8, 8);
        let mut buf = [0u8; 5];
        buf[1] = 0x01;
        match LinkInfo::parse(&buf, &ctx).unwrap_err() {
            Error::InvalidFormat { .. } => {}
            err => panic!("{err:?}"),
        }
    }
}
