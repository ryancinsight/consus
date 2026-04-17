//! Async read coordination layer for HDF5.
//!
//! ## Design
//!
//! Each async function collects raw bytes by issuing positioned [`AsyncReadAt`]
//! reads, assembles them into a [`MultiRegionBuffer`], then delegates format
//! parsing to the existing sync parsers that operate on [`ReadAt`].
//!
//! Format logic is not duplicated. Only the I/O coordination layer is async.
//!
//! ## Continuation Depth
//!
//! Each async header read issues a bounded number of `read_at` calls,
//! bounded by the continuation chain depth limit of 256 hops.

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};

use consus_core::{Error, Result};

#[cfg(feature = "async-io")]
use consus_io::{AsyncLength, AsyncReadAt};

#[cfg(feature = "alloc")]
use consus_io::ReadAt;

#[cfg(feature = "alloc")]
use crate::address::ParseContext;

#[cfg(feature = "alloc")]
use crate::constants::UNDEFINED_ADDRESS;

#[cfg(feature = "alloc")]
use crate::object_header::{OCHK_SIGNATURE, OHDR_SIGNATURE};

#[cfg(feature = "alloc")]
use crate::object_header::message_types;

#[cfg(all(feature = "async-io", feature = "alloc"))]
use crate::superblock::Superblock;

// ---------------------------------------------------------------------------
// MultiRegionBuffer
// ---------------------------------------------------------------------------

/// A [`ReadAt`] adapter over a sorted collection of `(file_offset, bytes)` pairs.
///
/// Assembles multiple positioned reads into a unified view suitable for
/// consumption by sync format parsers.
///
/// ## Invariant
///
/// Regions must be sorted by offset ascending before any `read_at` call
/// (call [`sort`] after all [`add_region`] calls). Each `read_at` request
/// must fall entirely within one region; reads spanning region boundaries
/// return an error.
///
/// [`sort`]: MultiRegionBuffer::sort
/// [`add_region`]: MultiRegionBuffer::add_region
#[cfg(feature = "alloc")]
pub(crate) struct MultiRegionBuffer {
    regions: Vec<(u64, Vec<u8>)>,
}

#[cfg(feature = "alloc")]
impl MultiRegionBuffer {
    /// Create an empty buffer with no regions.
    pub(crate) fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Append a region starting at `offset` with the given `data`.
    pub(crate) fn add_region(&mut self, offset: u64, data: Vec<u8>) {
        self.regions.push((offset, data));
    }

    /// Sort all regions by file offset ascending.
    ///
    /// Must be called before any `read_at` invocation when regions were
    /// added out of order.
    pub(crate) fn sort(&mut self) {
        self.regions.sort_unstable_by_key(|(off, _)| *off);
    }

    /// Construct from a single pre-fetched region.
    pub(crate) fn from_single(offset: u64, data: Vec<u8>) -> Self {
        Self {
            regions: vec![(offset, data)],
        }
    }
}

#[cfg(feature = "alloc")]
impl ReadAt for MultiRegionBuffer {
    /// Read `buf.len()` bytes at `pos` from the assembled buffer.
    ///
    /// Uses `partition_point` to locate the containing region in O(log n).
    /// Returns `Error::BufferTooSmall` if `pos` precedes all regions or if
    /// the read extends past the region end.
    fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        // Find the first region whose start offset is strictly greater than pos.
        // The containing region (if any) is at index `idx - 1`.
        let idx = self.regions.partition_point(|(off, _)| *off <= pos);
        if idx == 0 {
            return Err(Error::BufferTooSmall {
                required: buf.len(),
                provided: 0,
            });
        }
        let (region_offset, region_data) = &self.regions[idx - 1];
        let local = (pos - region_offset) as usize;
        let end = local + buf.len();
        if end > region_data.len() {
            return Err(Error::BufferTooSmall {
                required: end,
                provided: region_data.len().saturating_sub(local),
            });
        }
        buf.copy_from_slice(&region_data[local..end]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// read_region helper
// ---------------------------------------------------------------------------

/// Read exactly `len` bytes from `source` at `offset`.
///
/// Returns `Ok(vec![])` immediately for `len == 0` without issuing I/O.
#[cfg(all(feature = "async-io", feature = "alloc"))]
pub(crate) async fn read_region<R: AsyncReadAt>(
    source: &R,
    offset: u64,
    len: usize,
) -> Result<Vec<u8>> {
    if len == 0 {
        return Ok(vec![]);
    }
    let mut buf = vec![0u8; len];
    source.read_at(offset, &mut buf).await?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Continuation scanners
// ---------------------------------------------------------------------------

/// Scan v2-format message bytes for CONTINUATION messages (type `0x0010`).
///
/// Returns `(address, length)` pairs for each valid continuation target.
/// Targets with `UNDEFINED_ADDRESS` or zero length are excluded.
///
/// ## Arguments
///
/// - `msg_data`: raw message bytes (chunk payload, excluding OHDR/OCHK
///   preamble and trailing checksum).
/// - `flags`: OHDR flags byte - bit 2 (`FLAG_CREATION_ORDER_TRACKED`) controls
///   whether each message header carries a 2-byte creation-order field.
/// - `ctx`: parsing context supplying offset/length field widths.
#[cfg(feature = "alloc")]
pub(crate) fn scan_v2_continuations(
    msg_data: &[u8],
    flags: u8,
    ctx: &ParseContext,
) -> Result<Vec<(u64, u64)>> {
    use crate::object_header::v2::FLAG_CREATION_ORDER_TRACKED;

    let track_co = flags & FLAG_CREATION_ORDER_TRACKED != 0;
    // type(2) + data_size(2) + flags(1) [+ creation_order(2) if tracked]
    let msg_hdr_len: usize = if track_co { 7 } else { 5 };

    let mut continuations: Vec<(u64, u64)> = Vec::new();
    let mut pos: usize = 0;

    while pos + msg_hdr_len <= msg_data.len() {
        let msg_type = LittleEndian::read_u16(&msg_data[pos..]);
        let data_size = LittleEndian::read_u16(&msg_data[pos + 2..]) as usize;
        pos += msg_hdr_len;

        if pos + data_size > msg_data.len() {
            // Truncated message - stop scanning.
            break;
        }

        if msg_type == message_types::CONTINUATION
            && data_size >= ctx.offset_bytes() + ctx.length_bytes()
        {
            let cont_addr = ctx.read_offset(&msg_data[pos..]);
            let cont_len = ctx.read_length(&msg_data[pos + ctx.offset_bytes()..]);
            if cont_addr != UNDEFINED_ADDRESS && cont_len > 0 {
                continuations.push((cont_addr, cont_len));
            }
        }

        pos += data_size;
    }

    Ok(continuations)
}

/// Scan v1-format message bytes for CONTINUATION messages (type `0x0010`).
///
/// Returns `(address, length)` pairs for each valid continuation target.
/// Messages are padded to 8-byte alignment boundaries per the HDF5 v1 spec.
///
/// ## Arguments
///
/// - `msg_data`: raw message bytes from the object header block (excluding
///   the 12-byte prefix header).
/// - `ctx`: parsing context supplying offset/length field widths.
#[cfg(feature = "alloc")]
pub(crate) fn scan_v1_continuations(
    msg_data: &[u8],
    ctx: &ParseContext,
) -> Result<Vec<(u64, u64)>> {
    // type(2) + data_size(2) + flags(1) + reserved(3) = 8
    const MSG_HDR: usize = 8;

    let mut continuations: Vec<(u64, u64)> = Vec::new();
    let mut pos: usize = 0;

    while pos + MSG_HDR <= msg_data.len() {
        let msg_type = LittleEndian::read_u16(&msg_data[pos..]);
        let data_size = LittleEndian::read_u16(&msg_data[pos + 2..]) as usize;
        pos += MSG_HDR;

        if pos + data_size > msg_data.len() {
            // Truncated message - stop scanning.
            break;
        }

        if msg_type == message_types::CONTINUATION
            && data_size >= ctx.offset_bytes() + ctx.length_bytes()
        {
            let cont_addr = ctx.read_offset(&msg_data[pos..]);
            let cont_len = ctx.read_length(&msg_data[pos + ctx.offset_bytes()..]);
            if cont_addr != UNDEFINED_ADDRESS && cont_len > 0 {
                continuations.push((cont_addr, cont_len));
            }
        }

        // Advance past data, then align to 8-byte boundary.
        pos = pos + data_size;
        pos = (pos + 7) & !7;
    }

    Ok(continuations)
}

// ---------------------------------------------------------------------------
// Superblock
// ---------------------------------------------------------------------------

/// Read and parse the HDF5 superblock from `source` asynchronously.
///
/// Reads a window of up to 2112 bytes covering all valid superblock
/// positions (0, 512, 1024, 2048) plus the maximum superblock size,
/// then delegates to [`Superblock::read_from`].
///
/// ## Errors
///
/// - `Error::InvalidFormat` if `source` is empty or contains no valid
///   HDF5 superblock at any expected offset.
#[cfg(all(feature = "async-io", feature = "alloc"))]
pub async fn async_read_superblock<R: AsyncReadAt + AsyncLength>(
    source: &R,
) -> Result<Superblock> {
    let file_len = AsyncLength::len(source).await? as usize;
    // 2048 (last valid search offset) + 64 (max superblock size) = 2112
    let window = file_len.min(2112);
    if window == 0 {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::string::String::from("empty source"),
        });
    }
    let data = read_region(source, 0, window).await?;
    let buf = MultiRegionBuffer::from_single(0, data);
    Superblock::read_from(&buf)
}

// ---------------------------------------------------------------------------
// Object header - version 2
// ---------------------------------------------------------------------------

/// Read and parse a version 2 object header (OHDR) asynchronously.
///
/// Issues positioned reads to collect the initial OHDR chunk and any OCHK
/// continuation chunks, assembles all regions into a [`MultiRegionBuffer`],
/// then delegates to [`crate::object_header::v2::parse`].
///
/// ## Errors
///
/// - `Error::InvalidFormat` - signature mismatch, version mismatch, or
///   chunk data-size exceeds the 64 MiB safety limit.
/// - `Error::Corrupted` - continuation chain exceeds 256 hops.
#[cfg(all(feature = "async-io", feature = "alloc"))]
async fn async_read_ohdr_v2<R: AsyncReadAt>(
    source: &R,
    address: u64,
    ctx: &ParseContext,
) -> Result<crate::object_header::ObjectHeader> {
    use crate::object_header::v2::{
        CHECKSUM_LEN, FLAG_ATTR_PHASE_CHANGE, FLAG_TIMESTAMPS, MAX_CHUNK_BYTES, OCHK_PREAMBLE_LEN,
        OHDR_PREAMBLE_LEN,
    };

    // -- Preamble: signature(4) + version(1) + flags(1) -------------------
    let preamble = read_region(source, address, OHDR_PREAMBLE_LEN).await?;
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
    if preamble[4] != 2 {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "expected object header version 2, found {}",
                preamble[4]
            ),
        });
    }
    let flags = preamble[5];

    // -- Variable-length portion ------------------------------------------
    // Layout (present fields only): [timestamps(16)] [phase_change(4)] [chunk_size(W)]
    let size_width = crate::object_header::v2::chunk_data_size_width(flags)?;
    let mut vlen: usize = size_width;
    if flags & FLAG_TIMESTAMPS != 0 {
        vlen += 16;
    }
    if flags & FLAG_ATTR_PHASE_CHANGE != 0 {
        vlen += 4;
    }
    let variable_len = vlen;

    let var_data =
        read_region(source, address + OHDR_PREAMBLE_LEN as u64, variable_len).await?;
    // Chunk data-size field occupies the last `size_width` bytes of the variable region.
    let size_field_offset = variable_len - size_width;
    let chunk_data_size = crate::object_header::v2::read_chunk_data_size(
        &var_data[size_field_offset..],
        size_width,
    )?;

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

    // -- Full first chunk -------------------------------------------------
    let total =
        OHDR_PREAMBLE_LEN + variable_len + chunk_data_size as usize + CHECKSUM_LEN;
    let ohdr_data = read_region(source, address, total).await?;

    // -- Scan initial continuations ---------------------------------------
    let msg_start = OHDR_PREAMBLE_LEN + variable_len;
    let msg_end = msg_start + chunk_data_size as usize;
    let initial_conts =
        scan_v2_continuations(&ohdr_data[msg_start..msg_end], flags, ctx)?;

    // -- Build multi-region buffer ----------------------------------------
    let mut multi_buf = MultiRegionBuffer::new();
    multi_buf.add_region(address, ohdr_data);

    // -- Follow continuation chain ----------------------------------------
    let mut pending = initial_conts;
    let mut depth: usize = 0;
    const MAX_DEPTH: usize = 256;

    while let Some((cont_addr, cont_len)) = pending.pop() {
        if cont_addr == UNDEFINED_ADDRESS || cont_len == 0 {
            continue;
        }
        depth += 1;
        if depth > MAX_DEPTH {
            return Err(Error::Corrupted {
                #[cfg(feature = "alloc")]
                message: alloc::string::String::from(
                    "continuation chain exceeded 256 hops",
                ),
            });
        }
        let ochk_data = read_region(source, cont_addr, cont_len as usize).await?;
        if ochk_data.len() >= 4 && ochk_data[0..4] == OCHK_SIGNATURE {
            let ochk_msg_start = OCHK_PREAMBLE_LEN;
            let ochk_msg_end = (cont_len as usize).saturating_sub(CHECKSUM_LEN);
            if ochk_msg_end > ochk_msg_start {
                let more = scan_v2_continuations(
                    &ochk_data[ochk_msg_start..ochk_msg_end],
                    flags,
                    ctx,
                )?;
                pending.extend(more);
            }
        }
        multi_buf.add_region(cont_addr, ochk_data);
    }

    multi_buf.sort();
    crate::object_header::v2::parse(&multi_buf, address, ctx)
}

// ---------------------------------------------------------------------------
// Object header - version 1
// ---------------------------------------------------------------------------

/// Read and parse a version 1 object header asynchronously.
///
/// Issues positioned reads to collect the initial header block and any
/// continuation blocks, assembles all regions into a [`MultiRegionBuffer`],
/// then delegates to [`crate::object_header::v1::parse`].
///
/// ## Errors
///
/// - `Error::InvalidFormat` - version byte is not 1.
/// - `Error::Corrupted` - continuation chain exceeds 256 hops.
#[cfg(all(feature = "async-io", feature = "alloc"))]
async fn async_read_ohdr_v1<R: AsyncReadAt>(
    source: &R,
    address: u64,
    ctx: &ParseContext,
) -> Result<crate::object_header::ObjectHeader> {
    // version(1) + reserved(1) + num_messages(2) + ref_count(4) + header_size(4) = 12
    const V1_HEADER_PREFIX_SIZE: usize = 12;

    let prefix = read_region(source, address, V1_HEADER_PREFIX_SIZE).await?;
    if prefix[0] != 1 {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "expected object header version 1, found {}",
                prefix[0]
            ),
        });
    }

    let header_data_size = LittleEndian::read_u32(&prefix[8..12]) as usize;
    let initial_len = V1_HEADER_PREFIX_SIZE + header_data_size;
    let initial_data = read_region(source, address, initial_len).await?;

    let continuations =
        scan_v1_continuations(&initial_data[V1_HEADER_PREFIX_SIZE..], ctx)?;

    let mut multi_buf = MultiRegionBuffer::new();
    multi_buf.add_region(address, initial_data);

    let mut pending = continuations;
    let mut depth: usize = 0;

    while let Some((cont_addr, cont_len)) = pending.pop() {
        depth += 1;
        if depth > 256 {
            return Err(Error::Corrupted {
                #[cfg(feature = "alloc")]
                message: alloc::string::String::from(
                    "v1 object header continuation chain exceeded 256 hops",
                ),
            });
        }
        let cont_data = read_region(source, cont_addr, cont_len as usize).await?;
        let more = scan_v1_continuations(&cont_data, ctx)?;
        multi_buf.add_region(cont_addr, cont_data);
        pending.extend(more);
    }

    multi_buf.sort();
    crate::object_header::v1::parse(&multi_buf, address, ctx)
}

// ---------------------------------------------------------------------------
// Public dispatch
// ---------------------------------------------------------------------------

/// Read and parse the object header at `address` asynchronously.
///
/// Dispatches to the v1 or v2 parser based on a 4-byte signature peek:
/// - `OHDR` signature -> version 2.
/// - Anything else -> version 1 (first byte is the version number `1`).
///
/// ## Errors
///
/// Propagates errors from the selected v1 or v2 parser.
#[cfg(all(feature = "async-io", feature = "alloc"))]
pub async fn async_read_object_header<R: AsyncReadAt>(
    source: &R,
    address: u64,
    ctx: &ParseContext,
) -> Result<crate::object_header::ObjectHeader> {
    let peek = read_region(source, address, 4).await?;
    if peek[0..4] == OHDR_SIGNATURE {
        async_read_ohdr_v2(source, address, ctx).await
    } else {
        async_read_ohdr_v1(source, address, ctx).await
    }
}
