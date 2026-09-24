//! B-tree v2 internal-node (`BTIN`) child pointers and routing.

use super::{
    BTREE_V2_INTERNAL_SIGNATURE, BTreeV2Header, BTreeV2Record, MIN_NODE_BYTES,
    compute_num_records_width, read_variable_width_uint,
};
#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use alloc::{format, vec::Vec};
use consus_core::{Error, Result};
#[cfg(feature = "alloc")]
use consus_io::ReadAt;
#[cfg(all(feature = "async", feature = "alloc"))]
use moirai_async::io::AsyncReadAt;

/// Parsed B-tree v2 internal node.
///
/// Contains records and child node pointers for tree traversal.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct BTreeV2InternalNode {
    /// Record type (must match the tree header's record type).
    pub record_type: u8,
    /// Records in this internal node.
    pub records: Vec<BTreeV2Record>,
    /// Addresses of child nodes (length = num_records + 1).
    pub child_addresses: Vec<u64>,
    /// Number of records in each child node.
    pub child_num_records: Vec<u16>,
}

#[cfg(feature = "alloc")]
impl BTreeV2InternalNode {
    /// Parse a B-tree v2 internal node at the given address.
    ///
    /// Reads the "BTIN" signature, records, and child pointer table.
    ///
    /// ## Arguments
    ///
    /// - `source`: I/O source.
    /// - `address`: file address of the internal node.
    /// - `header`: B-tree v2 header (provides record size, node size).
    /// - `num_records`: number of records in this node.
    /// - `ctx`: parsing context for variable-width addresses.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature is not "BTIN".
    pub fn parse<R: ReadAt>(
        source: &R,
        address: u64,
        header: &BTreeV2Header,
        num_records: u16,
        ctx: &ParseContext,
    ) -> Result<Self> {
        let buf = consus_io::read_at_bounded(
            source,
            address,
            u64::from(header.node_size),
            &ctx.budget,
            "b-tree v2 internal node size",
        )?;
        let node_size = buf.len();
        if node_size < MIN_NODE_BYTES {
            return Err(Error::InvalidFormat {
                message: format!(
                    "b-tree v2 internal node at offset {address:#x} declares {node_size} bytes, \
                     below the {MIN_NODE_BYTES}-byte minimum"
                ),
            });
        }

        // Validate signature
        if buf[0..4] != BTREE_V2_INTERNAL_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: format!(
                    "expected B-tree v2 internal signature 'BTIN' at offset {:#x}, \
                     found [{:#04x}, {:#04x}, {:#04x}, {:#04x}]",
                    address, buf[0], buf[1], buf[2], buf[3],
                ),
            });
        }

        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: format!(
                    "unsupported B-tree v2 internal node version: {version}, expected 0"
                ),
            });
        }

        let record_type = buf[5];
        let rec_size = header.record_size as usize;
        let s = ctx.offset_bytes();
        let n_rec = num_records as usize;
        let n_children = n_rec + 1;

        // Parse records
        // Bound the reservation by what the node can physically hold: each
        // record needs `rec_size` bytes inside `buf`.
        let mut records = Vec::with_capacity(n_rec.min(node_size / rec_size.max(1)));
        let mut pos = 6;
        for _ in 0..n_rec {
            if pos + rec_size > buf.len() {
                break;
            }
            let data = Vec::from(&buf[pos..pos + rec_size]);
            records.push(BTreeV2Record { data });
            pos += rec_size;
        }

        // Parse child pointers: each child has address(S) + num_records(variable).
        // The num_records field width depends on the maximum possible records
        // that can fit in a node. For simplicity, we use 2 bytes (u16) which
        // is sufficient for the vast majority of HDF5 files.
        //
        // Per the spec, the width is ceil(log2(max_records_per_node + 1)) / 8,
        // but 2 bytes handles nodes up to 65535 records.
        let num_records_width = compute_num_records_width(header);

        // Each child entry needs `s + num_records_width` bytes inside `buf`.
        let child_entry_bytes = s + num_records_width;
        let max_children = node_size / child_entry_bytes.max(1);
        let mut child_addresses = Vec::with_capacity(n_children.min(max_children));
        let mut child_num_records = Vec::with_capacity(n_children.min(max_children));

        for _ in 0..n_children {
            if pos + s > buf.len() {
                break;
            }
            let addr = ctx.read_offset(&buf[pos..]);
            pos += s;

            let nrec = read_variable_width_uint(&buf[pos..], num_records_width);
            pos += num_records_width;

            child_addresses.push(addr);
            child_num_records.push(nrec as u16);
        }

        // If depth > 1, there may also be total_records fields for each child,
        // but we skip those for the current implementation level.

        Ok(Self {
            record_type,
            records,
            child_addresses,
            child_num_records,
        })
    }

    /// Parse a B-tree v2 internal node from an async I/O source.
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
            "b-tree v2 internal node size",
        )
        .await?;
        let node_size = buf.len();
        if node_size < MIN_NODE_BYTES {
            return Err(Error::InvalidFormat {
                message: format!(
                    "b-tree v2 internal node at offset {address:#x} declares {node_size} bytes, \
                     below the {MIN_NODE_BYTES}-byte minimum"
                ),
            });
        }

        if buf[0..4] != BTREE_V2_INTERNAL_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: format!(
                    "expected B-tree v2 internal signature 'BTIN' at offset {:#x}, \
                     found [{:#04x}, {:#04x}, {:#04x}, {:#04x}]",
                    address, buf[0], buf[1], buf[2], buf[3],
                ),
            });
        }

        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: format!(
                    "unsupported B-tree v2 internal node version: {version}, expected 0"
                ),
            });
        }

        let record_type = buf[5];
        let rec_size = header.record_size as usize;
        let s = ctx.offset_bytes();
        let n_rec = num_records as usize;
        let n_children = n_rec + 1;

        // Bound the reservation by what the node can physically hold: each
        // record needs `rec_size` bytes inside `buf`.
        let mut records = Vec::with_capacity(n_rec.min(node_size / rec_size.max(1)));
        let mut pos = 6;
        for _ in 0..n_rec {
            if pos + rec_size > buf.len() {
                break;
            }
            let data = Vec::from(&buf[pos..pos + rec_size]);
            records.push(BTreeV2Record { data });
            pos += rec_size;
        }

        let num_records_width = compute_num_records_width(header);

        // Each child entry needs `s + num_records_width` bytes inside `buf`.
        let child_entry_bytes = s + num_records_width;
        let max_children = node_size / child_entry_bytes.max(1);
        let mut child_addresses = Vec::with_capacity(n_children.min(max_children));
        let mut child_num_records = Vec::with_capacity(n_children.min(max_children));

        for _ in 0..n_children {
            if pos + s > buf.len() {
                break;
            }
            let addr = ctx.read_offset(&buf[pos..]);
            pos += s;

            let nrec = read_variable_width_uint(&buf[pos..], num_records_width);
            pos += num_records_width;

            child_addresses.push(addr);
            child_num_records.push(nrec as u16);
        }

        Ok(Self {
            record_type,
            records,
            child_addresses,
            child_num_records,
        })
    }
}
