//! B-tree v2 header (`BTHD`) model, parsing, and root-node fixups.

use super::BTREE_V2_SIGNATURE;
#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use alloc::{format, vec};
use consus_core::{Error, Result};
#[cfg(feature = "alloc")]
use consus_io::ReadAt;
#[cfg(all(feature = "async", feature = "alloc"))]
use moirai_async::io::AsyncReadAt;

/// Parsed B-tree v2 header.
///
/// Contains the structural parameters of the B-tree required to
/// navigate internal and leaf nodes.
#[derive(Debug, Clone)]
pub struct BTreeV2Header {
    /// Record type identifier (see [`record_type`][super::record_type] constants).
    pub record_type: u8,
    /// Node size in bytes (both internal and leaf nodes).
    pub node_size: u32,
    /// Size of one record in bytes.
    pub record_size: u16,
    /// Tree depth (0 = root is a leaf node).
    pub depth: u16,
    /// Split percent (threshold for splitting a node).
    pub split_percent: u8,
    /// Merge percent (threshold for merging nodes).
    pub merge_percent: u8,
    /// Address of the root node.
    ///
    /// `u64::MAX` (undefined address) indicates an empty tree.
    pub root_address: u64,
    /// Number of records in the root node.
    pub root_num_records: u16,
    /// Total number of records in the entire tree.
    pub total_records: u64,
}

#[cfg(feature = "alloc")]
impl BTreeV2Header {
    /// Parse a B-tree v2 header from the given file address.
    ///
    /// Reads and validates the "BTHD" signature, version, and all
    /// structural fields. The checksum is read but not yet verified.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature is not "BTHD".
    /// - [`Error::InvalidFormat`] if the version is not 0.
    /// - [`Error::InvalidFormat`] if the data is truncated.
    pub fn parse<R: ReadAt>(source: &R, address: u64, ctx: &ParseContext) -> Result<Self> {
        let s = ctx.offset_bytes();
        // Minimum size: signature(4) + version(1) + type(1) + node_size(4) +
        //   record_size(2) + depth(2) + split%(1) + merge%(1) +
        //   root_addr(S) + root_nrec(2) + total_records(S) + checksum(4)
        let min_size = 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + s + 2 + s + 4;

        let mut buf = vec![0u8; min_size];
        source.read_at(address, &mut buf)?;

        // Validate signature
        if buf[0..4] != BTREE_V2_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: format!(
                    "expected B-tree v2 header signature 'BTHD' at offset {:#x}, \
                     found [{:#04x}, {:#04x}, {:#04x}, {:#04x}]",
                    address, buf[0], buf[1], buf[2], buf[3],
                ),
            });
        }

        // Validate version
        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: format!("unsupported B-tree v2 header version: {version}, expected 0"),
            });
        }

        let record_type = buf[5];
        let node_size = u32::from_le_bytes([buf[6], buf[7], buf[8], buf[9]]);
        let record_size = u16::from_le_bytes([buf[10], buf[11]]);
        let depth = u16::from_le_bytes([buf[12], buf[13]]);
        let split_percent = buf[14];
        let merge_percent = buf[15];

        let mut pos = 16;
        let root_address = ctx.read_offset(&buf[pos..]);
        pos += s;
        let root_num_records = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let total_records = ctx.read_length(&buf[pos..]);
        // pos += s; // next would be checksum, which we skip for now

        Ok(Self {
            record_type,
            node_size,
            record_size,
            depth,
            split_percent,
            merge_percent,
            root_address,
            root_num_records,
            total_records,
        })
    }

    /// Parse a B-tree v2 header from an async I/O source.
    #[cfg(all(feature = "async", feature = "alloc"))]
    pub async fn async_parse<R: AsyncReadAt>(
        source: &R,
        address: u64,
        ctx: &ParseContext,
    ) -> Result<Self> {
        let s = ctx.offset_bytes();
        let min_size = 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + s + 2 + s + 4;

        let mut buf = vec![0u8; min_size];
        source
            .read_at(address, &mut buf)
            .await
            .map_err(Error::from)?;

        if buf[0..4] != BTREE_V2_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: format!(
                    "expected B-tree v2 header signature 'BTHD' at offset {:#x}, \
                     found [{:#04x}, {:#04x}, {:#04x}, {:#04x}]",
                    address, buf[0], buf[1], buf[2], buf[3],
                ),
            });
        }

        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: format!("unsupported B-tree v2 header version: {version}, expected 0"),
            });
        }

        let record_type = buf[5];
        let node_size = u32::from_le_bytes([buf[6], buf[7], buf[8], buf[9]]);
        let record_size = u16::from_le_bytes([buf[10], buf[11]]);
        let depth = u16::from_le_bytes([buf[12], buf[13]]);
        let split_percent = buf[14];
        let merge_percent = buf[15];

        let mut pos = 16;
        let root_address = ctx.read_offset(&buf[pos..]);
        pos += s;
        let root_num_records = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let total_records = ctx.read_length(&buf[pos..]);

        Ok(Self {
            record_type,
            node_size,
            record_size,
            depth,
            split_percent,
            merge_percent,
            root_address,
            root_num_records,
            total_records,
        })
    }
}
