//! B-tree v2 leaf-node (`BTLF`) record and node parsing.

use super::{BTREE_V2_LEAF_SIGNATURE, BTreeV2Header, MIN_NODE_BYTES};
#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use alloc::{format, vec::Vec};
use consus_core::{Error, Result};
#[cfg(feature = "alloc")]
use consus_io::ReadAt;
#[cfg(all(feature = "async", feature = "alloc"))]
use moirai_async::io::AsyncReadAt;

/// A raw record extracted from a B-tree v2 leaf or internal node.
///
/// The interpretation of the record bytes depends on the B-tree's
/// `record_type`. Callers must decode the bytes according to the
/// applicable record schema.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct BTreeV2Record {
    /// Raw record bytes (length = `BTreeV2Header::record_size`).
    pub data: Vec<u8>,
}

/// Parsed B-tree v2 leaf node.
///
/// Contains the raw records stored in the leaf. The record format
/// depends on the B-tree's record type.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct BTreeV2LeafNode {
    /// Record type (must match the tree header's record type).
    pub record_type: u8,
    /// Records in this leaf.
    pub records: Vec<BTreeV2Record>,
}

#[cfg(feature = "alloc")]
impl BTreeV2LeafNode {
    /// Parse a B-tree v2 leaf node at the given address.
    ///
    /// Reads the "BTLF" signature, version, type, and `num_records`
    /// records of `record_size` bytes each.
    ///
    /// ## Arguments
    ///
    /// - `source`: I/O source.
    /// - `address`: file address of the leaf node.
    /// - `header`: the B-tree v2 header (provides record size and node size).
    /// - `num_records`: number of records in this leaf node.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature is not "BTLF".
    /// - [`Error::InvalidFormat`] if the version is not 0.
    pub fn parse<R: ReadAt>(
        source: &R,
        address: u64,
        header: &BTreeV2Header,
        num_records: u16,
        ctx: &ParseContext,
    ) -> Result<Self> {
        // Read the full node. `node_size` is a u32 straight out of the B-tree
        // header, so it is bounded against the budget and grown only as the
        // reads confirm the bytes exist.
        let buf = consus_io::read_at_bounded(
            source,
            address,
            u64::from(header.node_size),
            &ctx.budget,
            "b-tree v2 leaf node size",
        )?;
        let node_size = buf.len();
        if node_size < MIN_NODE_BYTES {
            return Err(Error::InvalidFormat {
                message: format!(
                    "b-tree v2 leaf node at offset {address:#x} declares {node_size} bytes, \
                     below the {MIN_NODE_BYTES}-byte minimum"
                ),
            });
        }

        // Validate signature
        if buf[0..4] != BTREE_V2_LEAF_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: format!(
                    "expected B-tree v2 leaf signature 'BTLF' at offset {:#x}, \
                     found [{:#04x}, {:#04x}, {:#04x}, {:#04x}]",
                    address, buf[0], buf[1], buf[2], buf[3],
                ),
            });
        }

        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: format!("unsupported B-tree v2 leaf version: {version}, expected 0"),
            });
        }

        let record_type = buf[5];
        let rec_size = header.record_size as usize;

        // `num_records` is attacker-chosen, but every record must be backed by
        // `rec_size` real bytes inside this node, so the node's own length is
        // the tighter bound on the reservation.
        let mut records = Vec::with_capacity(
            ctx.budget
                .capacity_hint(u64::from(num_records), size_of::<BTreeV2Record>())
                .min(node_size / rec_size.max(1)),
        );
        let mut pos = 6; // after signature(4) + version(1) + type(1)

        for _ in 0..num_records {
            if pos + rec_size > buf.len().saturating_sub(4) {
                // Don't read into checksum territory
                break;
            }
            let data = Vec::from(&buf[pos..pos + rec_size]);
            records.push(BTreeV2Record { data });
            pos += rec_size;
        }

        Ok(Self {
            record_type,
            records,
        })
    }

    /// Parse a B-tree v2 leaf node from an async I/O source.
    #[cfg(all(feature = "async", feature = "alloc"))]
    pub async fn async_parse<R: AsyncReadAt>(
        source: &R,
        address: u64,
        header: &BTreeV2Header,
        num_records: u16,
        ctx: &ParseContext,
    ) -> Result<Self> {
        let buf = crate::file::async_reader::read_at_bounded(
            source,
            address,
            u64::from(header.node_size),
            &ctx.budget,
            "b-tree v2 leaf node size",
        )
        .await?;
        let node_size = buf.len();
        if node_size < MIN_NODE_BYTES {
            return Err(Error::InvalidFormat {
                message: format!(
                    "b-tree v2 leaf node at offset {address:#x} declares {node_size} bytes, \
                     below the {MIN_NODE_BYTES}-byte minimum"
                ),
            });
        }

        if buf[0..4] != BTREE_V2_LEAF_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: format!(
                    "expected B-tree v2 leaf signature 'BTLF' at offset {:#x}, \
                     found [{:#04x}, {:#04x}, {:#04x}, {:#04x}]",
                    address, buf[0], buf[1], buf[2], buf[3],
                ),
            });
        }

        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: format!("unsupported B-tree v2 leaf version: {version}, expected 0"),
            });
        }

        let record_type = buf[5];
        let rec_size = header.record_size as usize;

        let mut records = Vec::with_capacity(
            ctx.budget
                .capacity_hint(u64::from(num_records), size_of::<BTreeV2Record>())
                .min(node_size / rec_size.max(1)),
        );
        let mut pos = 6;

        for _ in 0..num_records {
            if pos + rec_size > buf.len().saturating_sub(4) {
                break;
            }
            let data = Vec::from(&buf[pos..pos + rec_size]);
            records.push(BTreeV2Record { data });
            pos += rec_size;
        }

        Ok(Self {
            record_type,
            records,
        })
    }
}
