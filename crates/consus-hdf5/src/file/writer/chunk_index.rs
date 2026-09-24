//! Chunk index model and the v1/v2/fixed-array on-disk chunk index writers.

#[cfg(feature = "alloc")]
use super::{WriteState, write_offset};
#[cfg(feature = "alloc")]
use crate::constants::UNDEFINED_ADDRESS;
#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};
#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};
#[cfg(feature = "alloc")]
use consus_compression::Checksum;
#[cfg(feature = "alloc")]
use consus_core::{Error, Result};
#[cfg(feature = "alloc")]
use consus_io::WriteAt;

#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub(crate) struct ChunkIndexEntry {
    pub(crate) chunk_offsets: Vec<u64>,
    pub(crate) filter_mask: u32,
    pub(crate) chunk_size: u32,
    pub(crate) chunk_address: u64,
}
#[cfg(feature = "alloc")]
pub(crate) fn write_chunk_btree_v1<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    chunk_dims: &[usize],
    entries: &[ChunkIndexEntry],
    dataset_dims: &[usize],
    element_size: usize,
) -> Result<u64> {
    let s = state.ctx.offset_bytes();
    let ndims = chunk_dims.len() + 1;
    let key_size = 4 + 4 + 8 * ndims;
    let header_size = 8 + 2 * s;

    // HDF5 B-tree v1 nodes have a fixed capacity of 2*K entries where
    // K = btree_ik (32 by default, matching HDF5_BTREE_IK_DEF).
    // libhdf5 reads exactly (header + (2K+1)*key_size + 2K*s) bytes from the
    // node address, regardless of entries_used.  Writing a smaller buffer
    // causes an addr-overflow EOF error when the library reads the node.
    const BTREE_IK: usize = 32;
    let max_entries = 2 * BTREE_IK; // 64
    let node_size = header_size + (max_entries + 1) * key_size + max_entries * s;

    let addr = state.allocate_aligned(node_size as u64);
    let mut buf = vec![0u8; node_size];

    buf[0..4].copy_from_slice(b"TREE");
    buf[4] = 1;
    buf[5] = 0;
    LittleEndian::write_u16(&mut buf[6..8], entries.len() as u16);
    write_offset(&mut buf[8..8 + s], s, UNDEFINED_ADDRESS);
    write_offset(&mut buf[8 + s..8 + 2 * s], s, UNDEFINED_ADDRESS);

    // Per HDF5 spec §IV.A.2.b: node layout is
    //   key[0] | addr[0] | key[1] | addr[1] | ... | key[N-1] | addr[N-1] | key[N]
    // where key[i] describes the chunk at addr[i], and key[N] is the sentinel.
    let mut pos = header_size;

    if entries.is_empty() {
        return Err(Error::InvalidFormat {
            message: String::from("chunked dataset requires at least one chunk index entry"),
        });
    }

    for entry in entries {
        // Write key[i]: chunk_size, filter_mask, element offsets, extra-zero dim.
        LittleEndian::write_u32(&mut buf[pos..pos + 4], entry.chunk_size);
        pos += 4;
        LittleEndian::write_u32(&mut buf[pos..pos + 4], entry.filter_mask);
        pos += 4;
        for &offset in &entry.chunk_offsets {
            LittleEndian::write_u64(&mut buf[pos..pos + 8], offset);
            pos += 8;
        }
        LittleEndian::write_u64(&mut buf[pos..pos + 8], 0);
        pos += 8;

        // Write addr[i].
        write_offset(&mut buf[pos..pos + s], s, entry.chunk_address);
        pos += s;
    }

    // Write key[N]: sentinel key.
    // chunk_size = 0, filter_mask = 0 (buf already zero-initialised).
    // offset[i] = dataset_dims[i] (the upper bound of the dataset in each dim).
    // extra_dim = element_size (the size in bytes of one element).
    pos += 4; // chunk_size = 0
    pos += 4; // filter_mask = 0
    for &dim in dataset_dims {
        LittleEndian::write_u64(&mut buf[pos..pos + 8], dim as u64);
        pos += 8;
    }
    LittleEndian::write_u64(&mut buf[pos..pos + 8], element_size as u64);
    // Remaining bytes in the full node stay zero (unused capacity slots).

    sink.write_at(addr, &buf)?;
    Ok(addr)
}

/// Write a B-tree v2 chunk index (BTHD header + BTLF leaf node).
///
/// ## B-tree v2 Structure (HDF5 spec III.A.2)
///
/// The B-tree v2 consists of a header (signature `BTHD`) and one or more
/// nodes. This writer emits a single leaf node (depth 0) containing all
/// chunk index records, which is sufficient for datasets where the total
/// number of chunks fits within a single leaf.
///
/// ### Record Types
///
/// | Type | Description | Record Layout |
/// |------|------------|---------------|
/// | 10 | Non-filtered chunks | address + scaled_offsets |
/// | 11 | Filtered chunks | address + chunk_size + filter_mask + scaled_offsets |
///
/// ### Scaled Offsets
///
/// V4 records store chunk grid coordinates (chunk index per dimension),
/// not raw byte offsets. Each scaled offset = `chunk_offset[i] / chunk_dim[i]`.
///
/// ## Returns
///
/// The file address of the BTHD header. This address is stored in the
/// v4 layout message's index address field.
#[cfg(feature = "alloc")]
pub(crate) fn write_chunk_btree_v2<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    chunk_dims: &[usize],
    entries: &[ChunkIndexEntry],
    has_filters: bool,
) -> Result<u64> {
    let s = state.ctx.offset_bytes();
    let l = state.ctx.length_bytes();
    let rank = chunk_dims.len();
    let record_type: u8 = if has_filters { 11 } else { 10 };
    let num_records = entries.len();

    // Record size per HDF5 spec:
    //   type 10: offset_size + rank * 8
    //   type 11: offset_size + length_size + 4 + rank * 8
    let record_size: usize = if has_filters {
        s + l + 4 + rank * 8
    } else {
        s + rank * 8
    };

    // -- Leaf node (BTLF) --
    // Layout: signature(4) + version(1) + type(1) + records(N * rec_size) + checksum(4)
    let leaf_size = 10 + num_records * record_size;
    let leaf_addr = state.allocate_aligned(leaf_size as u64);
    let mut leaf_buf = vec![0u8; leaf_size];

    leaf_buf[0..4].copy_from_slice(b"BTLF");
    leaf_buf[4] = 0; // version
    leaf_buf[5] = record_type;

    let mut pos = 6;
    for entry in entries {
        // Address of chunk data
        write_offset(&mut leaf_buf[pos..], s, entry.chunk_address);
        pos += s;

        if has_filters {
            // On-disk chunk size (length_size bytes)
            write_offset(&mut leaf_buf[pos..], l, u64::from(entry.chunk_size));
            pos += l;
            // Filter mask
            LittleEndian::write_u32(&mut leaf_buf[pos..], entry.filter_mask);
            pos += 4;
        }

        // Scaled offsets: chunk grid coordinates per dimension.
        // chunk_offsets are element offsets: coord * chunk_dim.
        // scaled = coord = element_offset / chunk_dim.
        for (i, &offset) in entry.chunk_offsets.iter().enumerate() {
            let dim = if i < chunk_dims.len() {
                chunk_dims[i]
            } else {
                1
            };
            let scaled = if dim > 0 { offset / dim as u64 } else { 0 };
            LittleEndian::write_u64(&mut leaf_buf[pos..], scaled);
            pos += 8;
        }
    }

    // Jenkins lookup3 checksum over all bytes preceding the checksum field
    let leaf_cksum = consus_compression::Lookup3::compute(&leaf_buf[..pos]);
    leaf_buf[pos..pos + 4].copy_from_slice(&leaf_cksum.to_le_bytes());

    sink.write_at(leaf_addr, &leaf_buf)?;

    // -- Header (BTHD) --
    // Layout: signature(4) + version(1) + type(1) + node_size(4) + record_size(2)
    //       + depth(2) + split%(1) + merge%(1) + root_addr(s) + root_nrec(2)
    //       + total_records(l) + checksum(4)
    let header_size = 22 + s + l;
    let header_addr = state.allocate_aligned(header_size as u64);
    let mut hdr_buf = vec![0u8; header_size];

    hdr_buf[0..4].copy_from_slice(b"BTHD");
    hdr_buf[4] = 0; // version
    hdr_buf[5] = record_type;
    LittleEndian::write_u32(&mut hdr_buf[6..10], leaf_size as u32); // node size
    LittleEndian::write_u16(&mut hdr_buf[10..12], record_size as u16); // record size
    LittleEndian::write_u16(&mut hdr_buf[12..14], 0); // depth = 0 (single leaf)
    hdr_buf[14] = 75; // split percent
    hdr_buf[15] = 25; // merge percent

    let mut hpos = 16;
    // Root node address = leaf node address
    write_offset(&mut hdr_buf[hpos..], s, leaf_addr);
    hpos += s;
    // Number of records in root node
    LittleEndian::write_u16(&mut hdr_buf[hpos..], num_records as u16);
    hpos += 2;
    // Total records in entire B-tree (length_size bytes)
    write_offset(&mut hdr_buf[hpos..], l, num_records as u64);
    hpos += l;

    // Jenkins lookup3 checksum per HDF5 spec §IV.A.2 (B-tree v2 header)
    let hdr_cksum = consus_compression::Lookup3::compute(&hdr_buf[..hpos]);
    hdr_buf[hpos..hpos + 4].copy_from_slice(&hdr_cksum.to_le_bytes());

    sink.write_at(header_addr, &hdr_buf)?;
    Ok(header_addr)
}

// ---------------------------------------------------------------------------
/// Write a Fixed Array (FARRAY) chunk index for a v4 layout.
///
/// Used for fixed-dimension datasets without filters.  Writes a Fixed Array
/// Header (FAHD) immediately followed by a Fixed Array Data Block (FADB)
/// containing each chunk's file address in chunk-coordinate order.
///
/// Returns the FAHD address (stored in the v4 layout message).
#[cfg(feature = "alloc")]
pub(crate) fn write_chunk_farray<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    entries: &[ChunkIndexEntry],
) -> Result<u64> {
    let s = state.ctx.offset_bytes();
    let nelmts = entries.len() as u64;
    let entry_size: usize = s; // one file-address per chunk (no filters)

    // FAHD: sig(4)+ver(1)+cid(1)+entry_size(1)+max_nelmts_bits(1)+
    //       nelmts(8)+fadb_addr(8)+checksum(4) = 28 bytes.
    // FADB: sig(4)+ver(1)+cid(1)+hdr_addr(8)+nelmts*entry_size+checksum(4).
    let fahd_size: usize = 28;
    let fadb_size: usize = 18 + entries.len() * entry_size;
    let combined = fahd_size + fadb_size;

    let fahd_addr = state.allocate_aligned(combined as u64);
    let fadb_addr = fahd_addr + fahd_size as u64;

    let mut buf = vec![0u8; combined];

    // -- FAHD --
    buf[0..4].copy_from_slice(b"FAHD");
    // buf[4] = 0; // version = 0
    // buf[5] = 0; // client ID = 0 (as observed in HDF5-generated files)
    buf[6] = entry_size as u8; // element/record size (= offset_size for no-filter)
    buf[7] = 10; // max_nelmts_bits = H5D_FARRAY_MAX_NELMTS_BITS
    LittleEndian::write_u64(&mut buf[8..16], nelmts);
    write_offset(&mut buf[16..], s, fadb_addr);
    let fahd_cksum = consus_compression::Lookup3::compute(&buf[..24]);
    LittleEndian::write_u32(&mut buf[24..28], fahd_cksum);

    // -- FADB --
    let fd = fahd_size;
    buf[fd..fd + 4].copy_from_slice(b"FADB");
    // buf[fd+4] = 0; // version = 0
    // buf[fd+5] = 0; // client ID = 0
    write_offset(&mut buf[fd + 6..], s, fahd_addr);
    let mut pos = fd + 14;
    for entry in entries {
        write_offset(&mut buf[pos..], s, entry.chunk_address);
        pos += s;
    }
    let fadb_cksum = consus_compression::Lookup3::compute(&buf[fd..pos]);
    LittleEndian::write_u32(&mut buf[pos..pos + 4], fadb_cksum);

    sink.write_at(fahd_addr, &buf)?;
    Ok(fahd_addr)
}
