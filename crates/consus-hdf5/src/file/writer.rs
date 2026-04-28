//! HDF5 file writer: create new files with the v2 superblock format.
//!
//! ## Specification
//!
//! This module provides low-level primitives for constructing HDF5 files.
//! It writes the v2 superblock, object headers (v2 format), and raw data
//! blocks. Higher-level file creation logic composes these primitives.
//!
//! ### Write Model
//!
//! The writer uses an append-only allocation model: new structures are
//! appended at the current end-of-file (EOF) position. The superblock's
//! EOF address is updated as the final step before closing.
//!
//! ### Object Header Construction
//!
//! Object headers are written in v2 format with CRC-32 checksums.
//! Messages are packed sequentially within a single header chunk.
//! Continuation chunks are not emitted by the writer; all messages
//! for an object must fit in the initial allocation.
//!
//! ## Dependencies (DIP)
//!
//! - `consus_core`: Error types, canonical data model types.
//! - `consus_io`: `WriteAt` trait for positioned byte output.
//! - `consus_compression`: `Crc32` checksum for v2 header integrity.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};

#[cfg(feature = "alloc")]
use consus_core::{Compression, Datatype, Error, Extent, Result, Shape};

#[cfg(feature = "alloc")]
use consus_io::WriteAt;

#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use crate::constants::{HDF5_MAGIC, UNDEFINED_ADDRESS};
#[cfg(feature = "alloc")]
use crate::dataset::chunk::{ChunkLocation, edge_chunk_dims, write_chunk_raw};
#[cfg(feature = "alloc")]
use crate::object_header::message_types;
#[cfg(feature = "alloc")]
use crate::property_list::{DatasetLayout, GroupCreationProps};

// Re-export property list types so consumers can import them via this module.
#[cfg(feature = "alloc")]
pub use crate::property_list::{DatasetCreationProps, FileCreationProps};

// ---------------------------------------------------------------------------
// Write state
// ---------------------------------------------------------------------------

/// Tracks the current write position and structural parameters.
///
/// The writer allocates space by advancing the EOF pointer. All
/// allocations are sequential; no free-space management is performed.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct WriteState {
    /// Current end-of-file byte offset.
    pub eof: u64,
    /// Parsing context derived from file creation properties.
    pub ctx: ParseContext,
    /// File creation properties governing superblock format.
    pub file_props: FileCreationProps,
}

#[cfg(feature = "alloc")]
impl WriteState {
    /// Create a new write state from file creation properties.
    pub fn new(props: FileCreationProps) -> Self {
        Self {
            eof: 0,
            ctx: ParseContext::new(props.offset_size, props.length_size),
            file_props: props,
        }
    }

    /// Allocate `size` bytes at the current EOF and advance the pointer.
    ///
    /// Returns the byte offset where the allocation begins.
    pub fn allocate(&mut self, size: u64) -> u64 {
        let addr = self.eof;
        self.eof += size;
        addr
    }

    /// Align EOF to an 8-byte boundary, then allocate `size` bytes.
    ///
    /// Returns the (aligned) byte offset where the allocation begins.
    pub fn allocate_aligned(&mut self, size: u64) -> u64 {
        self.eof = (self.eof + 7) & !7;
        self.allocate(size)
    }
}

// ---------------------------------------------------------------------------
// Offset / length encoding
// ---------------------------------------------------------------------------

/// Write a file offset of `size` bytes (little-endian) into `buf`.
///
/// Supports 2, 4, and 8-byte widths per the HDF5 specification.
fn write_offset(buf: &mut [u8], size: usize, value: u64) {
    match size {
        2 => LittleEndian::write_u16(buf, value as u16),
        4 => LittleEndian::write_u32(buf, value as u32),
        8 => LittleEndian::write_u64(buf, value),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Superblock
// ---------------------------------------------------------------------------

/// Write an HDF5 v2 superblock at the current EOF.
///
/// ## Layout (v2)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 8 | Magic `\x89HDF\r\n\x1a\n` |
/// | 8 | 1 | Version (2) |
/// | 9 | 1 | Offset size |
/// | 10 | 1 | Length size |
/// | 11 | 1 | Consistency flags |
/// | 12 | S | Base address |
/// | 12+S | S | Extension address |
/// | 12+2S | S | End-of-file address (placeholder) |
/// | 12+3S | S | Root group object header address |
/// | 12+4S | 4 | CRC-32 checksum |
///
/// The EOF address is written as a placeholder (0) and must be updated
/// via [`update_superblock_eof`] before closing the file.
#[cfg(feature = "alloc")]
pub fn write_superblock<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    root_group_address: u64,
) -> Result<()> {
    let s = state.ctx.offset_bytes();
    let data_len = 12 + 4 * s; // bytes before checksum
    let total_len = data_len + 4; // + CRC-32

    let addr = 0;
    if state.eof < total_len as u64 {
        state.eof = total_len as u64;
    }
    let mut buf = vec![0u8; total_len];

    // Magic bytes
    buf[0..8].copy_from_slice(&HDF5_MAGIC);
    // Superblock version
    buf[8] = state.file_props.superblock_version;
    // Offset size
    buf[9] = state.file_props.offset_size;
    // Length size
    buf[10] = state.file_props.length_size;
    // Consistency flags (0 = file not in inconsistent state)
    buf[11] = 0;

    let mut pos = 12;
    // Base address = 0
    write_offset(&mut buf[pos..], s, 0);
    pos += s;
    // Superblock extension address = undefined
    write_offset(&mut buf[pos..], s, UNDEFINED_ADDRESS);
    pos += s;
    // EOF address = placeholder (updated at close)
    write_offset(&mut buf[pos..], s, 0);
    pos += s;
    // Root group object header address
    write_offset(&mut buf[pos..], s, root_group_address);
    pos += s;

    // CRC-32 checksum over bytes [0..data_len)
    let checksum = consus_compression::Crc32::compute_slice(&buf[..pos]);
    buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());

    sink.write_at(addr, &buf)
}

/// Overwrite the EOF address field in an existing v2 superblock at offset 0.
///
/// The EOF field is located at byte offset `12 + 2 * offset_size`.
/// After updating, the CRC-32 checksum is recomputed and written.
///
/// ## Precondition
///
/// The sink must also implement `consus_io::ReadAt` so the superblock
/// can be re-read for checksum computation. This function accepts
/// `W: WriteAt + consus_io::ReadAt` to enforce this.
#[cfg(feature = "alloc")]
pub fn update_superblock_eof<W: WriteAt + consus_io::ReadAt>(
    sink: &mut W,
    state: &WriteState,
) -> Result<()> {
    let s = state.ctx.offset_bytes();
    let data_len = 12 + 4 * s;
    let total_len = data_len + 4;

    // Read the current superblock
    let mut buf = vec![0u8; total_len];
    sink.read_at(0, &mut buf)?;

    // Patch EOF address at offset 12 + 2*S
    let eof_field_offset = 12 + 2 * s;
    write_offset(&mut buf[eof_field_offset..], s, state.eof);

    // Recompute checksum
    let checksum = consus_compression::Crc32::compute_slice(&buf[..data_len]);
    buf[data_len..total_len].copy_from_slice(&checksum.to_le_bytes());

    sink.write_at(0, &buf)
}

// ---------------------------------------------------------------------------
// V2 Object header helpers
// ---------------------------------------------------------------------------

/// Determine the chunk-size field width and corresponding flags bits
/// for a v2 object header given the chunk data size.
///
/// Returns `(field_width_bytes, flags_bits_0_1)`.
///
/// | Data size range | Width | Flags bits 0-1 |
/// |-----------------|-------|----------------|
/// | 0..256 | 1 | 0b00 |
/// | 256..65536 | 2 | 0b01 |
/// | 65536..2^32 | 4 | 0b10 |
/// | ≥ 2^32 | 8 | 0b11 |
fn chunk_size_encoding(data_size: usize) -> (usize, u8) {
    if data_size < 256 {
        (1, 0x00)
    } else if data_size < 65536 {
        (2, 0x01)
    } else if data_size < (1 << 32) {
        (4, 0x02)
    } else {
        (8, 0x03)
    }
}

/// Write the chunk data size into `buf` using the given field width.
#[cfg(feature = "alloc")]
fn write_chunk_size(buf: &mut [u8], width: usize, value: usize) {
    match width {
        1 => buf[0] = value as u8,
        2 => LittleEndian::write_u16(buf, value as u16),
        4 => LittleEndian::write_u32(buf, value as u32),
        8 => LittleEndian::write_u64(buf, value as u64),
        _ => {}
    }
}

/// Serialize a single v2 header message into `buf` at `pos`.
///
/// V2 message header layout:
/// | 0 | 2 | Message type |
/// | 2 | 2 | Data size |
/// | 4 | 1 | Flags |
///
/// Returns the number of bytes written (5 + data.len()).
#[cfg(feature = "alloc")]
fn write_v2_message(buf: &mut [u8], pos: usize, msg_type: u16, flags: u8, data: &[u8]) -> usize {
    LittleEndian::write_u16(&mut buf[pos..], msg_type);
    LittleEndian::write_u16(&mut buf[pos + 2..], data.len() as u16);
    buf[pos + 4] = flags;
    buf[pos + 5..pos + 5 + data.len()].copy_from_slice(data);
    5 + data.len()
}

// ---------------------------------------------------------------------------
// Group header
// ---------------------------------------------------------------------------

/// Write a minimal v2 object header for a group.
///
/// The header is allocated with padding space (NIL messages) to allow
/// future in-place addition of link messages.
///
/// Returns the file address of the written object header.
#[cfg(feature = "alloc")]
pub fn write_group_header<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    _props: &GroupCreationProps,
) -> Result<u64> {
    // Reserve 256 bytes for the group header (generous for future links)
    let reserved = 256usize;

    // Compute chunk data size: one NIL message fills the space.
    // OHDR signature(4) + version(1) + flags(1) + chunk_size_field(?) + data + checksum(4)
    // We solve for data_size such that total = reserved.
    let (csf_width, csf_flags) = chunk_size_encoding(reserved);
    let overhead = 4 + 1 + 1 + csf_width + 4; // signature + version + flags + csf + checksum
    let chunk_data_size = reserved - overhead;

    let addr = state.allocate_aligned(reserved as u64);
    let mut buf = vec![0u8; reserved];

    // OHDR signature
    buf[0..4].copy_from_slice(b"OHDR");
    buf[4] = 2; // version 2
    buf[5] = csf_flags; // flags: chunk size encoding only
    let mut pos = 6;
    write_chunk_size(&mut buf[pos..], csf_width, chunk_data_size);
    pos += csf_width;

    // Fill chunk with a single NIL message spanning all available space.
    let nil_data_len = chunk_data_size.saturating_sub(5); // 5-byte message header
    let written = write_v2_message(&mut buf, pos, 0x0000, 0, &vec![0u8; nil_data_len]);
    pos += written;

    // CRC-32 checksum of entire header (signature through end of messages)
    let checksum = consus_compression::Crc32::compute_slice(&buf[..pos]);
    buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());

    sink.write_at(addr, &buf)?;
    Ok(addr)
}

// ---------------------------------------------------------------------------
// Dataset header
// ---------------------------------------------------------------------------

/// Write a v2 object header for a dataset.
///
/// Emits three header messages:
/// 1. Datatype (0x0003)
/// 2. Dataspace (0x0001)
/// 3. Data Layout (0x0008)
///
/// Returns the file address of the written object header.
#[cfg(feature = "alloc")]
pub fn write_dataset_header<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    datatype: &Datatype,
    shape: &Shape,
    data_address: u64,
    props: &DatasetCreationProps,
) -> Result<u64> {
    let dt_bytes = encode_datatype(datatype)?;
    let ds_bytes = encode_dataspace(shape)?;
    let filter_ids = dataset_filter_ids(props);
    let layout_bytes = match props.layout {
        DatasetLayout::Chunked => {
            let element_size =
                datatype
                    .element_size()
                    .ok_or_else(|| Error::UnsupportedFeature {
                        feature: String::from("chunked write requires fixed-size element datatype"),
                    })?;
            encode_layout_with_chunk_index(
                data_address,
                props,
                &state.ctx,
                Some(data_address),
                Some(element_size as u32),
            )?
        }
        _ => encode_layout(data_address, props, &state.ctx)?,
    };

    let filter_bytes = if filter_ids.is_empty() {
        None
    } else {
        Some(encode_filter_pipeline(&filter_ids)?)
    };

    let mut messages: Vec<(u16, Vec<u8>)> = vec![
        (message_types::DATATYPE, dt_bytes),
        (message_types::DATASPACE, ds_bytes),
        (message_types::DATA_LAYOUT, layout_bytes),
    ];

    if let Some(filter_bytes) = filter_bytes {
        messages.push((message_types::FILTER_PIPELINE, filter_bytes));
    }

    let msg_refs: Vec<(u16, &[u8])> = messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
    write_object_header_v2(sink, state, &msg_refs)
}

// ---------------------------------------------------------------------------
// Datatype encoding
// ---------------------------------------------------------------------------

/// Encode a canonical `Datatype` into HDF5 binary datatype message bytes.
///
/// ## Supported types
///
/// | Canonical type | HDF5 class | Encoded |
/// |---------------|-----------|---------|
/// | `Boolean` | 0 (fixed-point) | 1-byte unsigned integer |
/// | `Integer` | 0 (fixed-point) | Matching width/sign/order |
/// | `Float` | 1 (floating-point) | IEEE 754 f32 or f64 |
/// | `FixedString` | 3 (string) | Null-padded fixed-length |
///
/// Returns `Error::UnsupportedFeature` for types not yet supported
/// in the write path.
#[cfg(feature = "alloc")]
pub fn encode_datatype(dt: &Datatype) -> Result<Vec<u8>> {
    match dt {
        Datatype::Boolean => {
            // 1-byte unsigned integer
            let mut buf = vec![0u8; 12]; // header(8) + properties(4)
            buf[0] = 0x10; // class=0 (fixed-point), version=1
            buf[1] = 0x00; // LE, unsigned
            LittleEndian::write_u32(&mut buf[4..8], 1); // size = 1 byte
            LittleEndian::write_u16(&mut buf[8..10], 0); // bit offset
            LittleEndian::write_u16(&mut buf[10..12], 8); // bit precision
            Ok(buf)
        }
        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            let size = bits.get() / 8;
            let mut buf = vec![0u8; 12]; // header(8) + properties(4)
            buf[0] = 0x10; // class=0, version=1
            let mut flags = 0u8;
            if *byte_order == consus_core::ByteOrder::BigEndian {
                flags |= 0x01;
            }
            if *signed {
                flags |= 0x08;
            }
            buf[1] = flags;
            LittleEndian::write_u32(&mut buf[4..8], size as u32);
            LittleEndian::write_u16(&mut buf[8..10], 0); // bit offset
            LittleEndian::write_u16(&mut buf[10..12], (size * 8) as u16); // bit precision
            Ok(buf)
        }
        Datatype::Float { bits, byte_order } => {
            let size = bits.get() / 8;
            let mut buf = vec![0u8; 20]; // header(8) + properties(12)
            buf[0] = 0x11; // class=1 (float), version=1
            let flags = if *byte_order == consus_core::ByteOrder::BigEndian {
                0x01u8
            } else {
                0x00
            };
            buf[1] = flags;
            LittleEndian::write_u32(&mut buf[4..8], size as u32);

            // IEEE 754 properties: bit_offset(2) + bit_precision(2) +
            //   exp_location(1) + exp_size(1) + mant_location(1) + mant_size(1) + exp_bias(4)
            match size {
                4 => {
                    // f32: exponent at bit 23, 8 bits; mantissa at bit 0, 23 bits; bias 127
                    LittleEndian::write_u16(&mut buf[8..10], 0); // bit offset
                    LittleEndian::write_u16(&mut buf[10..12], 32); // bit precision
                    buf[12] = 23; // exponent location
                    buf[13] = 8; // exponent size
                    buf[14] = 0; // mantissa location
                    buf[15] = 23; // mantissa size
                    LittleEndian::write_u32(&mut buf[16..20], 127); // exponent bias
                }
                8 => {
                    // f64: exponent at bit 52, 11 bits; mantissa at bit 0, 52 bits; bias 1023
                    LittleEndian::write_u16(&mut buf[8..10], 0);
                    LittleEndian::write_u16(&mut buf[10..12], 64);
                    buf[12] = 52;
                    buf[13] = 11;
                    buf[14] = 0;
                    buf[15] = 52;
                    LittleEndian::write_u32(&mut buf[16..20], 1023);
                }
                2 => {
                    // f16: exponent at bit 10, 5 bits; mantissa at bit 0, 10 bits; bias 15
                    LittleEndian::write_u16(&mut buf[8..10], 0);
                    LittleEndian::write_u16(&mut buf[10..12], 16);
                    buf[12] = 10;
                    buf[13] = 5;
                    buf[14] = 0;
                    buf[15] = 10;
                    LittleEndian::write_u32(&mut buf[16..20], 15);
                }
                _ => {
                    return Err(Error::UnsupportedFeature {
                        feature: alloc::format!(
                            "write-path float encoding for {}-byte floats not supported",
                            size
                        ),
                    });
                }
            }
            Ok(buf)
        }
        Datatype::FixedString { length, encoding } => {
            let mut buf = vec![0u8; 8]; // header only, no additional properties
            buf[0] = 0x13; // class=3 (string), version=1
            // flags byte 0: padding type = 1 (null-pad)
            // flags byte 1: charset (0=ASCII, 1=UTF-8)
            buf[1] = 0x01; // null-pad
            buf[2] = match encoding {
                consus_core::StringEncoding::Ascii => 0x00,
                consus_core::StringEncoding::Utf8 => 0x01,
            };
            LittleEndian::write_u32(&mut buf[4..8], *length as u32);
            Ok(buf)
        }
        _ => Err(Error::UnsupportedFeature {
            feature: alloc::format!("write-path encoding for datatype {dt:?} not yet supported"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Dataspace encoding
// ---------------------------------------------------------------------------

/// Encode a `Shape` into an HDF5 version 2 dataspace message.
///
/// ## Layout (version 2)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 1 | Version (2) |
/// | 1 | 1 | Dimensionality (rank) |
/// | 2 | 1 | Flags (bit 0: max dims present) |
/// | 3 | 1 | Type (0=scalar, 1=simple, 2=null) |
/// | 4 | 8×rank | Current dimension sizes |
/// | var | 8×rank | Maximum dimension sizes (if flags bit 0) |
#[cfg(feature = "alloc")]
pub fn encode_dataspace(shape: &Shape) -> Result<Vec<u8>> {
    let rank = shape.rank();
    let has_unlimited = shape.has_unlimited();
    let flags: u8 = if has_unlimited { 0x01 } else { 0x00 };
    let ds_type: u8 = if rank == 0 { 0 } else { 1 }; // scalar vs simple

    let max_dims_bytes = if has_unlimited { 8 * rank } else { 0 };
    let total = 4 + 8 * rank + max_dims_bytes;
    let mut buf = vec![0u8; total];

    buf[0] = 2; // version
    buf[1] = rank as u8;
    buf[2] = flags;
    buf[3] = ds_type;

    let mut pos = 4;
    for ext in shape.extents() {
        LittleEndian::write_u64(&mut buf[pos..], ext.current_size() as u64);
        pos += 8;
    }

    if has_unlimited {
        for ext in shape.extents() {
            let max = match ext {
                Extent::Fixed(n) => *n as u64,
                Extent::Unlimited { .. } => u64::MAX,
            };
            LittleEndian::write_u64(&mut buf[pos..], max);
            pos += 8;
        }
    }

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Layout encoding
// ---------------------------------------------------------------------------

/// Encode a data layout message (version 3).
///
/// ## Contiguous (class 1)
///
/// | 0 | 1 | Version (3) |
/// | 1 | 1 | Class (1) |
/// | 2 | S | Data address |
/// | 2+S | L | Data size |
///
/// ## Compact (class 0)
///
/// | 0 | 1 | Version (3) |
/// | 1 | 1 | Class (0) |
/// | 2 | 2 | Data size |
/// | 4 | N | Compact data |
///
/// ## Chunked (class 2)
///
/// | 0 | 1 | Version (3) |
/// | 1 | 1 | Class (2) |
/// | 2 | 1 | Dimensionality (rank + 1) |
/// | 3 | S | B-tree v1 address |
/// | 3+S | 4×(rank+1) | Chunk dimension sizes |
#[cfg(feature = "alloc")]
pub fn encode_layout(
    data_address: u64,
    props: &DatasetCreationProps,
    ctx: &ParseContext,
) -> Result<Vec<u8>> {
    encode_layout_with_chunk_index(data_address, props, ctx, None, None)
}

#[cfg(feature = "alloc")]
pub fn encode_layout_with_chunk_index(
    data_address: u64,
    props: &DatasetCreationProps,
    ctx: &ParseContext,
    chunk_index_address: Option<u64>,
    chunk_element_size: Option<u32>,
) -> Result<Vec<u8>> {
    let s = ctx.offset_bytes();

    match props.layout {
        DatasetLayout::Contiguous => {
            let l = ctx.length_bytes();
            let total = 2 + s + l;
            let mut buf = vec![0u8; total];
            buf[0] = 3; // version 3
            buf[1] = 1; // contiguous
            write_offset(&mut buf[2..], s, data_address);
            // Data size = 0 placeholder (computed from shape × element_size at write time)
            Ok(buf)
        }
        DatasetLayout::Compact => {
            let mut buf = vec![0u8; 4];
            buf[0] = 3; // version 3
            buf[1] = 0; // compact
            LittleEndian::write_u16(&mut buf[2..4], 0); // data size placeholder
            Ok(buf)
        }
        DatasetLayout::Chunked => {
            let chunk_dims = props
                .chunk_dims
                .as_ref()
                .ok_or_else(|| Error::InvalidFormat {
                    message: String::from(
                        "chunked layout requires chunk_dims in DatasetCreationProps",
                    ),
                })?;
            // V4 layout with B-tree v2 index
            if props.layout_version == Some(4) {
                let btree_address = chunk_index_address.ok_or_else(|| Error::InvalidFormat {
                    message: String::from(
                        "v4 chunked layout requires a materialized chunk index address",
                    ),
                })?;
                return encode_layout_v4_chunked(chunk_dims, btree_address, ctx);
            }
            let element_size = chunk_element_size.ok_or_else(|| Error::InvalidFormat {
                message: String::from(
                    "chunked layout requires a resolved element size in the terminal dimension",
                ),
            })?;
            let btree_address = chunk_index_address.ok_or_else(|| Error::InvalidFormat {
                message: String::from("chunked layout requires a materialized chunk index address"),
            })?;
            // Dimensionality = rank + 1 (extra element for type size)
            let ndims = chunk_dims.len() + 1;
            let total = 3 + s + 4 * ndims;
            let mut buf = vec![0u8; total];
            buf[0] = 3; // version 3
            buf[1] = 2; // chunked
            buf[2] = ndims as u8; // dimensionality
            write_offset(&mut buf[3..], s, btree_address);
            let mut pos = 3 + s;
            for &d in chunk_dims {
                LittleEndian::write_u32(&mut buf[pos..], d as u32);
                pos += 4;
            }
            LittleEndian::write_u32(&mut buf[pos..], element_size);
            Ok(buf)
        }
        DatasetLayout::Virtual => {
            // Emit a minimal version 3 class 3 (virtual) layout message.
            // The HDF5 read path surfaces this as StorageLayout::Virtual.
            Ok(vec![3u8, 3u8])
        }
    }
}

/// Encode a v4 chunked layout message with B-tree v2 index reference.
///
/// ## V4 Layout Message Format (HDF5 spec IV.A.2.l)
///
/// | Field | Size | Value |
/// |-------|------|-------|
/// | Version | 1 | 4 |
/// | Layout class | 1 | 2 (chunked) |
/// | Flags | 1 | 0 (B-tree v2 indexed) |
/// | Dimensionality | 1 | rank |
/// | Encoded size width | 1 | length_size |
/// | Chunk dimensions | rank x 4 | u32 LE each |
/// | Chunk index type | 1 | 5 (B-tree v2) |
/// | Index address | offset_size | B-tree v2 header address |
///
/// Total size = 6 + rank * 4 + offset_size
#[cfg(feature = "alloc")]
fn encode_layout_v4_chunked(
    chunk_dims: &[usize],
    btree_address: u64,
    ctx: &ParseContext,
) -> Result<Vec<u8>> {
    let s = ctx.offset_bytes();
    let l = ctx.length_bytes();
    let rank = chunk_dims.len();
    let total = 6 + rank * 4 + s;
    let mut buf = vec![0u8; total];

    buf[0] = 4; // version 4
    buf[1] = 2; // layout class = chunked
    buf[2] = 0; // flags = 0 (no special single-chunk encoding)
    buf[3] = rank as u8; // dimensionality
    buf[4] = l as u8; // encoded size width = length_size

    let mut pos = 5;
    for &d in chunk_dims {
        LittleEndian::write_u32(&mut buf[pos..], d as u32);
        pos += 4;
    }

    buf[pos] = 5; // chunk index type = B-tree v2
    pos += 1;

    write_offset(&mut buf[pos..], s, btree_address);

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Contiguous data write
// ---------------------------------------------------------------------------

/// Write raw contiguous data to the file.
///
/// Allocates space at the current EOF and writes `data` verbatim.
/// Returns the file address where the data was written.
#[cfg(feature = "alloc")]
pub fn write_contiguous_data<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    data: &[u8],
) -> Result<u64> {
    let addr = state.allocate_aligned(data.len() as u64);
    sink.write_at(addr, data)?;
    Ok(addr)
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
struct ChunkIndexEntry {
    chunk_offsets: Vec<u64>,
    filter_mask: u32,
    chunk_size: u32,
    chunk_address: u64,
}

#[cfg(feature = "alloc")]
fn dataset_filter_ids(props: &DatasetCreationProps) -> Vec<u16> {
    let mut integrity_filters = Vec::new();
    let mut transform_filters = Vec::new();
    let mut compression_filters = Vec::new();

    match props.compression {
        Compression::None => {}
        Compression::Deflate { .. } => compression_filters.push(1),
        Compression::Zstd { .. } => compression_filters.push(32015),
        Compression::Lz4 => compression_filters.push(32004),
        Compression::Gzip { .. } => compression_filters.push(1),
    }

    for &filter_id in &props.filters {
        if filter_id == 3 {
            if !integrity_filters.contains(&filter_id) {
                integrity_filters.push(filter_id);
            }
        } else if filter_id == 1 || filter_id == 32015 || filter_id == 32004 {
            if !compression_filters.contains(&filter_id) {
                compression_filters.push(filter_id);
            }
        } else if !transform_filters.contains(&filter_id) {
            transform_filters.push(filter_id);
        }
    }

    integrity_filters
        .into_iter()
        .chain(transform_filters)
        .chain(compression_filters)
        .collect()
}

#[cfg(feature = "alloc")]
fn encode_filter_pipeline(filter_ids: &[u16]) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; 2];
    buf[0] = 2; // version 2
    buf[1] = filter_ids.len() as u8;

    for &filter_id in filter_ids {
        let client_data: &[u32] = &[];
        let name_length: u16 = 0;
        let flags: u16 = 0;
        let num_client_data = client_data.len() as u16;

        let start = buf.len();
        buf.resize(start + 8 + client_data.len() * 4, 0);

        LittleEndian::write_u16(&mut buf[start..start + 2], filter_id);
        LittleEndian::write_u16(&mut buf[start + 2..start + 4], name_length);
        LittleEndian::write_u16(&mut buf[start + 4..start + 6], flags);
        LittleEndian::write_u16(&mut buf[start + 6..start + 8], num_client_data);

        let mut pos = start + 8;
        for &value in client_data {
            LittleEndian::write_u32(&mut buf[pos..pos + 4], value);
            pos += 4;
        }
    }

    Ok(buf)
}

#[cfg(feature = "alloc")]
fn linear_index(coords: &[usize], dims: &[usize]) -> usize {
    let mut index = 0usize;
    for (&coord, &dim) in coords.iter().zip(dims.iter()) {
        index = index * dim + coord;
    }
    index
}

#[cfg(feature = "alloc")]
fn increment_chunk_coord(coord: &mut [usize], grid_dims: &[usize]) -> bool {
    if coord.is_empty() {
        return false;
    }

    for dim in (0..coord.len()).rev() {
        coord[dim] += 1;
        if coord[dim] < grid_dims[dim] {
            return true;
        }
        coord[dim] = 0;
    }

    false
}

#[cfg(feature = "alloc")]
fn extract_chunk_bytes(
    raw_data: &[u8],
    dataset_dims: &[usize],
    chunk_coord: &[usize],
    chunk_dims: &[usize],
    element_size: usize,
) -> Result<Vec<u8>> {
    let actual_chunk_dims = edge_chunk_dims(chunk_coord, chunk_dims, dataset_dims);
    let chunk_elements = actual_chunk_dims.iter().product::<usize>();
    let mut chunk = vec![0u8; chunk_elements * element_size];

    if dataset_dims.is_empty() {
        if raw_data.len() != element_size {
            return Err(Error::InvalidFormat {
                message: String::from("scalar dataset raw byte length does not match element size"),
            });
        }
        chunk.copy_from_slice(raw_data);
        return Ok(chunk);
    }

    let rank = dataset_dims.len();
    let chunk_origin: Vec<usize> = chunk_coord
        .iter()
        .zip(chunk_dims.iter())
        .map(|(&coord, &dim)| coord.checked_mul(dim).ok_or(Error::Overflow))
        .collect::<Result<Vec<usize>>>()?;

    let mut local_coord = vec![0usize; rank];
    let mut done = false;

    while !done {
        let mut dataset_coord = Vec::with_capacity(rank);
        for d in 0..rank {
            dataset_coord.push(
                chunk_origin[d]
                    .checked_add(local_coord[d])
                    .ok_or(Error::Overflow)?,
            );
        }

        let dataset_linear = linear_index(&dataset_coord, dataset_dims);
        let chunk_linear = linear_index(&local_coord, &actual_chunk_dims);

        let src_start = dataset_linear
            .checked_mul(element_size)
            .ok_or(Error::Overflow)?;
        let src_end = src_start.checked_add(element_size).ok_or(Error::Overflow)?;
        let dst_start = chunk_linear
            .checked_mul(element_size)
            .ok_or(Error::Overflow)?;
        let dst_end = dst_start.checked_add(element_size).ok_or(Error::Overflow)?;

        chunk[dst_start..dst_end].copy_from_slice(&raw_data[src_start..src_end]);

        for dim in (0..rank).rev() {
            local_coord[dim] += 1;
            if local_coord[dim] < actual_chunk_dims[dim] {
                break;
            }
            local_coord[dim] = 0;
            if dim == 0 {
                done = true;
            }
        }
    }

    Ok(chunk)
}

#[cfg(feature = "alloc")]
fn write_chunk_btree_v1<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    chunk_dims: &[usize],
    entries: &[ChunkIndexEntry],
) -> Result<u64> {
    let s = state.ctx.offset_bytes();
    let ndims = chunk_dims.len() + 1;
    let key_size = 4 + 4 + 8 * ndims;
    let header_size = 8 + 2 * s;
    let node_size = header_size + key_size + entries.len() * (s + key_size);

    let addr = state.allocate_aligned(node_size as u64);
    let mut buf = vec![0u8; node_size];

    buf[0..4].copy_from_slice(b"TREE");
    buf[4] = 1;
    buf[5] = 0;
    LittleEndian::write_u16(&mut buf[6..8], entries.len() as u16);
    write_offset(&mut buf[8..8 + s], s, UNDEFINED_ADDRESS);
    write_offset(&mut buf[8 + s..8 + 2 * s], s, UNDEFINED_ADDRESS);

    let mut pos = header_size;

    let first = entries.first().ok_or_else(|| Error::InvalidFormat {
        message: String::from("chunked dataset requires at least one chunk index entry"),
    })?;
    LittleEndian::write_u32(&mut buf[pos..pos + 4], first.chunk_size);
    pos += 4;
    LittleEndian::write_u32(&mut buf[pos..pos + 4], first.filter_mask);
    pos += 4;
    for &offset in &first.chunk_offsets {
        LittleEndian::write_u64(&mut buf[pos..pos + 8], offset);
        pos += 8;
    }
    LittleEndian::write_u64(&mut buf[pos..pos + 8], 0);
    pos += 8;

    for entry in entries {
        write_offset(&mut buf[pos..pos + s], s, entry.chunk_address);
        pos += s;

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
    }

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
fn write_chunk_btree_v2<W: WriteAt>(
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

        // Scaled offsets: chunk_offset[i] / chunk_dim[i] per dimension
        for (i, &offset) in entry.chunk_offsets.iter().enumerate() {
            let dim = if i < chunk_dims.len() {
                chunk_dims[i]
            } else {
                1
            };
            let scaled = if dim > 0 { offset / (dim as u64) } else { 0 };
            LittleEndian::write_u64(&mut leaf_buf[pos..], scaled);
            pos += 8;
        }
    }

    // CRC-32 checksum over all bytes preceding the checksum field
    let leaf_cksum = consus_compression::Crc32::compute_slice(&leaf_buf[..pos]);
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

    // CRC-32 checksum
    let hdr_cksum = consus_compression::Crc32::compute_slice(&hdr_buf[..hpos]);
    hdr_buf[hpos..hpos + 4].copy_from_slice(&hdr_cksum.to_le_bytes());

    sink.write_at(header_addr, &hdr_buf)?;
    Ok(header_addr)
}

#[cfg(feature = "alloc")]
fn write_chunked_data<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    datatype: &Datatype,
    shape: &Shape,
    raw_data: &[u8],
    props: &DatasetCreationProps,
) -> Result<u64> {
    let chunk_dims = props
        .chunk_dims
        .as_ref()
        .ok_or_else(|| Error::InvalidFormat {
            message: String::from("chunked dataset write requires chunk dimensions"),
        })?;

    if chunk_dims.len() != shape.rank() {
        return Err(Error::ShapeError {
            message: alloc::format!(
                "chunk rank mismatch: dataset rank {}, chunk rank {}",
                shape.rank(),
                chunk_dims.len()
            ),
        });
    }

    let element_size = datatype
        .element_size()
        .ok_or_else(|| Error::UnsupportedFeature {
            feature: String::from("chunked write requires fixed-size element datatype"),
        })?;
    let dataset_dims = shape.current_dims();
    let expected_len = shape
        .num_elements()
        .checked_mul(element_size)
        .ok_or(Error::Overflow)?;
    if raw_data.len() != expected_len {
        return Err(Error::ShapeError {
            message: alloc::format!(
                "dataset payload byte length mismatch: expected {expected_len}, found {}",
                raw_data.len()
            ),
        });
    }

    let grid_dims: Vec<usize> = if dataset_dims.is_empty() {
        Vec::new()
    } else {
        dataset_dims
            .iter()
            .zip(chunk_dims.iter())
            .map(|(&dataset_dim, &chunk_dim)| dataset_dim.div_ceil(chunk_dim))
            .collect()
    };

    let filter_ids = dataset_filter_ids(props);
    let registry = consus_compression::DefaultCodecRegistry::new();
    let mut entries = Vec::new();

    if dataset_dims.is_empty() {
        let location = write_chunk_raw(
            sink,
            state.eof,
            raw_data,
            &filter_ids,
            element_size,
            &registry,
        )?;
        state.allocate_aligned(location.size);
        entries.push(ChunkIndexEntry {
            chunk_offsets: Vec::new(),
            filter_mask: location.filter_mask,
            chunk_size: location.size as u32,
            chunk_address: location.address,
        });
    } else {
        let mut chunk_coord = vec![0usize; grid_dims.len()];
        loop {
            let chunk_bytes = extract_chunk_bytes(
                raw_data,
                &dataset_dims,
                &chunk_coord,
                chunk_dims,
                element_size,
            )?;
            let location: ChunkLocation = write_chunk_raw(
                sink,
                state.eof,
                &chunk_bytes,
                &filter_ids,
                element_size,
                &registry,
            )?;
            state.allocate_aligned(location.size);

            let chunk_offsets: Vec<u64> = chunk_coord
                .iter()
                .zip(chunk_dims.iter())
                .map(|(&coord, &dim)| {
                    u64::try_from(coord.checked_mul(dim).ok_or(Error::Overflow)?)
                        .map_err(|_| Error::Overflow)
                })
                .collect::<Result<Vec<u64>>>()?;

            entries.push(ChunkIndexEntry {
                chunk_offsets,
                filter_mask: location.filter_mask,
                chunk_size: location.size as u32,
                chunk_address: location.address,
            });

            if !increment_chunk_coord(&mut chunk_coord, &grid_dims) {
                break;
            }
        }
    }

    let has_filters = !filter_ids.is_empty();
    if props.layout_version == Some(4) {
        write_chunk_btree_v2(sink, state, chunk_dims, &entries, has_filters)
    } else {
        write_chunk_btree_v1(sink, state, chunk_dims, &entries)
    }
}

// ---------------------------------------------------------------------------
// Link message encoding
// ---------------------------------------------------------------------------

/// Encode a hard link message for v2 groups.
///
/// ## Layout (link message version 1)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 1 | Version (1) |
/// | 1 | 1 | Flags |
/// | var | 1..8 | Link name length |
/// | var | N | Link name |
/// | var | S | Object header address (hard link) |
#[cfg(feature = "alloc")]
pub fn encode_hard_link(name: &str, target_address: u64, ctx: &ParseContext) -> Result<Vec<u8>> {
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len();

    // Determine name length field width (flags bits 0-1)
    let (len_width, len_flags) = if name_len < 256 {
        (1usize, 0u8)
    } else if name_len < 65536 {
        (2, 1)
    } else {
        (4, 2)
    };

    // Flags: bits 0-1 = name length size, bit 3 = link type present (0 for hard = default)
    let flags = len_flags;

    let s = ctx.offset_bytes();
    let total = 1 + 1 + len_width + name_len + s; // version + flags + name_len + name + address
    let mut buf = vec![0u8; total];
    let mut pos = 0;

    buf[pos] = 1; // version
    pos += 1;
    buf[pos] = flags;
    pos += 1;

    // Name length
    match len_width {
        1 => {
            buf[pos] = name_len as u8;
            pos += 1;
        }
        2 => {
            LittleEndian::write_u16(&mut buf[pos..], name_len as u16);
            pos += 2;
        }
        4 => {
            LittleEndian::write_u32(&mut buf[pos..], name_len as u32);
            pos += 4;
        }
        _ => {}
    }

    // Name
    buf[pos..pos + name_len].copy_from_slice(name_bytes);
    pos += name_len;

    // Hard link value: object header address
    write_offset(&mut buf[pos..], s, target_address);

    Ok(buf)
}

/// Encode a soft link message.
///
/// Soft link value: 2-byte length + target path string.
#[cfg(feature = "alloc")]
pub fn encode_soft_link(name: &str, target_path: &str, _ctx: &ParseContext) -> Result<Vec<u8>> {
    let name_bytes = name.as_bytes();
    let target_bytes = target_path.as_bytes();
    let name_len = name_bytes.len();

    let (len_width, len_flags) = if name_len < 256 {
        (1usize, 0u8)
    } else if name_len < 65536 {
        (2, 1)
    } else {
        (4, 2)
    };

    // Flags: name length encoding + link type present (bit 3)
    let flags = len_flags | 0x08; // bit 3 set: link type field present
    let link_type: u8 = 1; // soft link

    let total = 1 + 1 + 1 + len_width + name_len + 2 + target_bytes.len();
    let mut buf = vec![0u8; total];
    let mut pos = 0;

    buf[pos] = 1; // version
    pos += 1;
    buf[pos] = flags;
    pos += 1;

    // Link type
    buf[pos] = link_type;
    pos += 1;

    // Name length
    match len_width {
        1 => {
            buf[pos] = name_len as u8;
            pos += 1;
        }
        2 => {
            LittleEndian::write_u16(&mut buf[pos..], name_len as u16);
            pos += 2;
        }
        4 => {
            LittleEndian::write_u32(&mut buf[pos..], name_len as u32);
            pos += 4;
        }
        _ => {}
    }

    // Name
    buf[pos..pos + name_len].copy_from_slice(name_bytes);
    pos += name_len;

    // Soft link value: 2-byte length + path
    LittleEndian::write_u16(&mut buf[pos..], target_bytes.len() as u16);
    pos += 2;
    buf[pos..pos + target_bytes.len()].copy_from_slice(target_bytes);

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Generalised V2 object header writer
// ---------------------------------------------------------------------------

/// Write a v2 object header containing the given messages.
///
/// Each entry in `messages` is `(message_type, data)`. The header is
/// allocated at an 8-byte-aligned address and appended at the current EOF.
///
/// Returns the file address of the written header.
///
/// ## Buffer sizing
///
/// Uses at least 5 bytes for chunk data to accommodate a minimal NIL
/// message when `messages` is empty, preventing a CRC-field buffer overflow.
#[cfg(feature = "alloc")]
pub fn write_object_header_v2<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    messages: &[(u16, &[u8])],
) -> Result<u64> {
    let chunk_data_size: usize = messages.iter().map(|(_, d)| 5 + d.len()).sum();
    // Minimum 5 bytes: a NIL message header (type:2 + size:2 + flags:1).
    let effective_size = chunk_data_size.max(5);
    let (csf_width, csf_flags) = chunk_size_encoding(effective_size);
    let total = 4 + 1 + 1 + csf_width + effective_size + 4;

    let addr = state.allocate_aligned(total as u64);
    let mut buf = vec![0u8; total];

    buf[0..4].copy_from_slice(b"OHDR");
    buf[4] = 2; // version
    buf[5] = csf_flags;
    let mut pos = 6;
    write_chunk_size(&mut buf[pos..], csf_width, effective_size);
    pos += csf_width;

    if messages.is_empty() {
        // Write one NIL message (type=0, size=0, flags=0) to fill space.
        pos += write_v2_message(&mut buf, pos, 0x0000, 0, &[]);
    } else {
        for (msg_type, data) in messages {
            pos += write_v2_message(&mut buf, pos, *msg_type, 0, data);
        }
    }

    let checksum = consus_compression::Crc32::compute_slice(&buf[..pos]);
    buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());

    sink.write_at(addr, &buf)?;
    Ok(addr)
}

// ---------------------------------------------------------------------------
// Attribute message encoding
// ---------------------------------------------------------------------------

/// Encode an attribute as a version-3 attribute message (type 0x000C).
///
/// ### Version 3 Layout
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 1 | Version (3) |
/// | 1 | 1 | Flags (0) |
/// | 2 | 2 | Name size (byte count, no null terminator) |
/// | 4 | 2 | Datatype size |
/// | 6 | 2 | Dataspace size |
/// | 8 | 1 | Encoding (0=ASCII, 1=UTF-8) |
/// | 9 | N | Name bytes |
/// | var | var | Datatype |
/// | var | var | Dataspace |
/// | var | var | Raw data |
#[cfg(feature = "alloc")]
pub fn encode_attribute(
    name: &str,
    dt: &Datatype,
    shape: &Shape,
    raw_data: &[u8],
) -> Result<Vec<u8>> {
    let name_bytes = name.as_bytes();
    let dt_bytes = encode_datatype(dt)?;
    let ds_bytes = encode_dataspace(shape)?;

    // Detect UTF-8 name (non-ASCII bytes present).
    let encoding: u8 = if name_bytes.iter().any(|&b| b > 0x7F) {
        1
    } else {
        0
    };

    let total = 9 + name_bytes.len() + dt_bytes.len() + ds_bytes.len() + raw_data.len();

    let mut buf = vec![0u8; total];
    buf[0] = 3; // version
    buf[1] = 0; // flags
    LittleEndian::write_u16(&mut buf[2..4], name_bytes.len() as u16);
    LittleEndian::write_u16(&mut buf[4..6], dt_bytes.len() as u16);
    LittleEndian::write_u16(&mut buf[6..8], ds_bytes.len() as u16);
    buf[8] = encoding;

    let mut pos = 9;
    buf[pos..pos + name_bytes.len()].copy_from_slice(name_bytes);
    pos += name_bytes.len();
    buf[pos..pos + dt_bytes.len()].copy_from_slice(&dt_bytes);
    pos += dt_bytes.len();
    buf[pos..pos + ds_bytes.len()].copy_from_slice(&ds_bytes);
    pos += ds_bytes.len();
    buf[pos..pos + raw_data.len()].copy_from_slice(raw_data);

    Ok(buf)
}

// ---------------------------------------------------------------------------
// Recursive group node writer
// ---------------------------------------------------------------------------

/// Write a group node recursively: datasets first, then sub-groups, then the
/// group object header.
///
/// Returns the byte offset of the written group object header.
/// The caller is responsible for recording the returned address in its own
/// link table (root group or parent group).
///
/// ### Write order (depth-first, leaf-first)
///
/// 1. Each dataset in `datasets`: contiguous data block → object header.
/// 2. Each sub-group in `sub_groups`: recurse into `write_group_node`.
/// 3. Group object header containing LINK messages for all children and
///    ATTRIBUTE messages for `group_attributes`.
#[cfg(feature = "alloc")]
fn write_group_node(
    sink: &mut consus_io::MemCursor,
    state: &mut WriteState,
    group_attributes: &[(&str, &Datatype, &Shape, &[u8])],
    datasets: &[ChildDatasetSpec<'_>],
    sub_groups: &[ChildGroupSpec<'_>],
) -> Result<u64> {
    let ctx = state.ctx;
    let mut child_links: Vec<(String, u64)> = Vec::with_capacity(datasets.len() + sub_groups.len());

    // Step 1: write each child dataset (data block + object header).
    for child in datasets {
        let data_addr = write_contiguous_data(sink, state, child.raw_data)?;

        let dt_bytes = encode_datatype(child.datatype)?;
        let ds_bytes = encode_dataspace(child.shape)?;
        let layout_bytes = encode_layout(data_addr, &child.dcpl, &ctx)?;

        let mut child_msgs: Vec<(u16, Vec<u8>)> = vec![
            (message_types::DATATYPE, dt_bytes),
            (message_types::DATASPACE, ds_bytes),
            (message_types::DATA_LAYOUT, layout_bytes),
        ];

        for (attr_name, attr_dt, attr_shape, attr_data) in child.attributes {
            child_msgs.push((
                message_types::ATTRIBUTE,
                encode_attribute(attr_name, attr_dt, attr_shape, attr_data)?,
            ));
        }

        let msg_refs: Vec<(u16, &[u8])> =
            child_msgs.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let child_addr = write_object_header_v2(sink, state, &msg_refs)?;
        child_links.push((String::from(child.name), child_addr));
    }

    // Step 2: recursively write each sub-group and record its address.
    for sub in sub_groups {
        let addr = write_group_node(sink, state, sub.attributes, sub.datasets, sub.sub_groups)?;
        child_links.push((String::from(sub.name), addr));
    }

    // Step 3: build group object header messages (links + group attributes).
    let mut group_msgs: Vec<(u16, Vec<u8>)> = Vec::new();

    for (child_name, child_addr) in &child_links {
        group_msgs.push((
            message_types::LINK,
            encode_hard_link(child_name, *child_addr, &ctx)?,
        ));
    }

    for (attr_name, attr_dt, attr_shape, attr_data) in group_attributes {
        group_msgs.push((
            message_types::ATTRIBUTE,
            encode_attribute(attr_name, attr_dt, attr_shape, attr_data)?,
        ));
    }

    // Step 4: write the group object header and return its address.
    let msg_refs: Vec<(u16, &[u8])> = group_msgs.iter().map(|(t, d)| (*t, d.as_slice())).collect();
    write_object_header_v2(sink, state, &msg_refs)
}

// ---------------------------------------------------------------------------
// High-level file builder
// ---------------------------------------------------------------------------

/// High-level HDF5 file builder.
///
/// Accumulates datasets and root-group attributes in memory, then writes a
/// well-formed HDF5 v2 file on [`finish`][Self::finish].
///
/// ### Write Order (bottom-up; no back-patching required)
///
/// 1. Superblock space reserved at offset 0.
/// 2. Dataset raw data blocks (addresses known before headers).
/// 3. Dataset object headers (data address known).
/// 4. Root group object header (all child addresses known).
/// 5. Actual superblock written at offset 0; EOF address patched.
///
/// This ordering guarantees every address reference in each structure
/// points to an already-allocated region.
#[cfg(feature = "alloc")]
pub struct Hdf5FileBuilder {
    sink: consus_io::MemCursor,
    state: WriteState,
    /// Encoded hard-link (name, address) pairs for the root group.
    root_links: Vec<(String, u64)>,
    /// Encoded attribute message payloads for the root group.
    root_attr_bytes: Vec<Vec<u8>>,
}

#[cfg(feature = "alloc")]
impl Hdf5FileBuilder {
    /// Create a new builder with the given file creation properties.
    pub fn new(props: FileCreationProps) -> Self {
        let mut state = WriteState::new(props);
        // Reserve superblock space at offset 0 so subsequent allocations
        // do not overwrite it.
        let sb_size = 12 + 4 * state.ctx.offset_bytes() + 4;
        state.eof = sb_size as u64;
        Self {
            sink: consus_io::MemCursor::new(),
            state,
            root_links: Vec::new(),
            root_attr_bytes: Vec::new(),
        }
    }

    /// Add a contiguous dataset to the root group.
    ///
    /// The dataset is written immediately (data block + object header).
    /// Returns the object header address.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] or [`Error::UnsupportedFeature`] if the
    ///   datatype is not encodable.
    pub fn add_dataset(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        raw_data: &[u8],
        dcpl: &DatasetCreationProps,
    ) -> Result<u64> {
        let data_addr = match dcpl.layout {
            DatasetLayout::Chunked => {
                write_chunked_data(&mut self.sink, &mut self.state, dt, shape, raw_data, dcpl)?
            }
            _ => write_contiguous_data(&mut self.sink, &mut self.state, raw_data)?,
        };
        let header_addr =
            write_dataset_header(&mut self.sink, &mut self.state, dt, shape, data_addr, dcpl)?;
        self.root_links.push((String::from(name), header_addr));
        Ok(header_addr)
    }

    /// Add a dataset with attached attributes to the root group.
    ///
    /// `attributes` is a slice of `(name, datatype, shape, raw_data)`.
    /// Attributes are encoded into the dataset's object header.
    pub fn add_dataset_with_attributes(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        raw_data: &[u8],
        dcpl: &DatasetCreationProps,
        attributes: &[(&str, &Datatype, &Shape, &[u8])],
    ) -> Result<u64> {
        let data_addr = match dcpl.layout {
            DatasetLayout::Chunked => {
                write_chunked_data(&mut self.sink, &mut self.state, dt, shape, raw_data, dcpl)?
            }
            _ => write_contiguous_data(&mut self.sink, &mut self.state, raw_data)?,
        };

        let dt_bytes = encode_datatype(dt)?;
        let ds_bytes = encode_dataspace(shape)?;
        let ctx = self.state.ctx;
        let filter_ids = dataset_filter_ids(dcpl);
        let layout_bytes = match dcpl.layout {
            DatasetLayout::Chunked => {
                let element_size = dt.element_size().ok_or_else(|| Error::UnsupportedFeature {
                    feature: String::from("chunked write requires fixed-size element datatype"),
                })?;
                encode_layout_with_chunk_index(
                    data_addr,
                    dcpl,
                    &ctx,
                    Some(data_addr),
                    Some(element_size as u32),
                )?
            }
            _ => encode_layout(data_addr, dcpl, &ctx)?,
        };

        let mut msgs: Vec<(u16, Vec<u8>)> = vec![
            (message_types::DATATYPE, dt_bytes),
            (message_types::DATASPACE, ds_bytes),
            (message_types::DATA_LAYOUT, layout_bytes),
        ];

        if !filter_ids.is_empty() {
            msgs.push((
                message_types::FILTER_PIPELINE,
                encode_filter_pipeline(&filter_ids)?,
            ));
        }

        for (attr_name, attr_dt, attr_shape, attr_data) in attributes {
            let attr_bytes = encode_attribute(attr_name, attr_dt, attr_shape, attr_data)?;
            msgs.push((message_types::ATTRIBUTE, attr_bytes));
        }

        let msg_refs: Vec<(u16, &[u8])> = msgs.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let header_addr = write_object_header_v2(&mut self.sink, &mut self.state, &msg_refs)?;

        self.root_links.push((String::from(name), header_addr));
        Ok(header_addr)
    }

    /// Add an attribute to the root group.
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] or [`Error::UnsupportedFeature`] if the
    ///   datatype is not encodable.
    pub fn add_root_attribute(
        &mut self,
        name: &str,
        dt: &Datatype,
        shape: &Shape,
        raw_data: &[u8],
    ) -> Result<()> {
        let bytes = encode_attribute(name, dt, shape, raw_data)?;
        self.root_attr_bytes.push(bytes);
        Ok(())
    }

    /// Finalise the file and return the complete HDF5 image as bytes.
    ///
    /// Writes the root group object header (with all accumulated links and
    /// attributes), then writes and patches the superblock.
    pub fn finish(mut self) -> Result<Vec<u8>> {
        let ctx = self.state.ctx;

        // Collect all root group messages.
        let mut msgs: Vec<(u16, Vec<u8>)> = Vec::new();

        for (name, addr) in &self.root_links {
            let link_bytes = encode_hard_link(name, *addr, &ctx)?;
            msgs.push((0x0006, link_bytes));
        }

        for attr_bytes in &self.root_attr_bytes {
            msgs.push((0x000C, attr_bytes.clone()));
        }

        let msg_refs: Vec<(u16, &[u8])> = msgs.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let root_addr = write_object_header_v2(&mut self.sink, &mut self.state, &msg_refs)?;

        write_superblock(&mut self.sink, &mut self.state, root_addr)?;
        update_superblock_eof(&mut self.sink, &self.state)?;

        Ok(self.sink.into_bytes())
    }

    /// Add a named group to the root group with attached attributes and child datasets.
    ///
    /// ## Write model
    ///
    /// For each child in `children`:
    /// 1. Data bytes are written as a contiguous block.
    /// 2. A dataset object header is written with Datatype + Dataspace + Layout
    ///    messages and optional attribute messages.
    ///
    /// Then the group object header is written with:
    /// - One LINK message per child mapping the child name to its header address.
    /// - One ATTRIBUTE message per entry in `group_attributes`.
    ///
    /// The group is linked from the root group.
    ///
    /// ## Errors
    ///
    /// Returns an error if any datatype or layout cannot be encoded.
    /// `ChildDatasetSpec::dcpl` must specify `DatasetLayout::Contiguous`.
    pub fn add_group_with_attributes(
        &mut self,
        group_name: &str,
        group_attributes: &[(&str, &Datatype, &Shape, &[u8])],
        children: &[ChildDatasetSpec<'_>],
    ) -> Result<u64> {
        let group_addr = write_group_node(
            &mut self.sink,
            &mut self.state,
            group_attributes,
            children,
            &[],
        )?;
        self.root_links.push((String::from(group_name), group_addr));
        Ok(group_addr)
    }

    /// Add a named group to the root group with both child datasets and child sub-groups.
    ///
    /// Sub-groups support arbitrary nesting depth: each [`ChildGroupSpec`] can
    /// contain further `ChildGroupSpec` values in its `sub_groups` field.
    /// The hierarchy is written depth-first (leaf nodes first).
    ///
    /// ## Errors
    ///
    /// Returns an error if any datatype or layout cannot be encoded.
    pub fn add_group_with_children(
        &mut self,
        group_name: &str,
        group_attributes: &[(&str, &Datatype, &Shape, &[u8])],
        datasets: &[ChildDatasetSpec<'_>],
        sub_groups: &[ChildGroupSpec<'_>],
    ) -> Result<u64> {
        let group_addr = write_group_node(
            &mut self.sink,
            &mut self.state,
            group_attributes,
            datasets,
            sub_groups,
        )?;
        self.root_links.push((String::from(group_name), group_addr));
        Ok(group_addr)
    }
}

// ---------------------------------------------------------------------------
// Nested group authoring
// ---------------------------------------------------------------------------

/// Specification for a child dataset to be authored inside a nested group.
///
/// Only [`DatasetLayout::Contiguous`] is supported for `dcpl`. Compact and
/// Chunked layouts require additional writer parameters not available in this
/// context.
#[cfg(feature = "alloc")]
pub struct ChildDatasetSpec<'a> {
    /// Dataset name within the parent group.
    pub name: &'a str,
    /// Element datatype.
    pub datatype: &'a Datatype,
    /// Dataset shape.
    pub shape: &'a Shape,
    /// Raw data bytes in dataset storage order.
    pub raw_data: &'a [u8],
    /// Dataset creation properties. Only `DatasetLayout::Contiguous` is supported.
    pub dcpl: DatasetCreationProps,
    /// Attribute messages attached to this dataset.
    ///
    /// Each entry is `(attribute_name, datatype, shape, raw_data)`.
    pub attributes: &'a [(&'a str, &'a Datatype, &'a Shape, &'a [u8])],
}

/// Specification for a child sub-group to be authored inside a parent group.
///
/// Supports arbitrary nesting depth: [`sub_groups`][ChildGroupSpec::sub_groups]
/// can contain further `ChildGroupSpec` values, written recursively.
///
/// Only [`DatasetLayout::Contiguous`] is supported for dataset children.
#[cfg(feature = "alloc")]
pub struct ChildGroupSpec<'a> {
    /// Name of this group within its parent group.
    pub name: &'a str,
    /// Attribute messages attached to this group's object header.
    ///
    /// Each entry is `(attribute_name, datatype, shape, raw_data)`.
    pub attributes: &'a [(&'a str, &'a Datatype, &'a Shape, &'a [u8])],
    /// Dataset children of this group.
    pub datasets: &'a [ChildDatasetSpec<'a>],
    /// Sub-group children of this group (written recursively before this group).
    pub sub_groups: &'a [ChildGroupSpec<'a>],
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::ByteOrder as _;
    use consus_core::ByteOrder;
    use core::num::NonZeroUsize;

    #[cfg(feature = "alloc")]
    use crate::file::Hdf5File;

    #[test]
    fn write_state_allocate() {
        let props = FileCreationProps::default();
        let mut state = WriteState::new(props);
        assert_eq!(state.eof, 0);
        let a1 = state.allocate(16);
        assert_eq!(a1, 0);
        assert_eq!(state.eof, 16);
        let a2 = state.allocate(32);
        assert_eq!(a2, 16);
        assert_eq!(state.eof, 48);
    }

    #[test]
    fn write_state_allocate_aligned() {
        let props = FileCreationProps::default();
        let mut state = WriteState::new(props);
        state.eof = 5;
        let a = state.allocate_aligned(10);
        assert_eq!(a, 8); // 5 aligned up to 8
        assert_eq!(state.eof, 18);
    }

    #[test]
    fn chunk_size_encoding_boundaries() {
        assert_eq!(chunk_size_encoding(0), (1, 0x00));
        assert_eq!(chunk_size_encoding(255), (1, 0x00));
        assert_eq!(chunk_size_encoding(256), (2, 0x01));
        assert_eq!(chunk_size_encoding(65535), (2, 0x01));
        assert_eq!(chunk_size_encoding(65536), (4, 0x02));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn encode_integer_datatype_le_u32() {
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: ByteOrder::LittleEndian,
            signed: false,
        };
        let bytes = encode_datatype(&dt).unwrap();
        assert_eq!(bytes.len(), 12);
        assert_eq!(bytes[0] & 0x0F, 0); // class 0
        assert_eq!(bytes[1] & 0x01, 0); // LE
        assert_eq!(bytes[1] & 0x08, 0); // unsigned
        assert_eq!(LittleEndian::read_u32(&bytes[4..8]), 4); // 4 bytes
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn encode_integer_datatype_be_i16() {
        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: ByteOrder::BigEndian,
            signed: true,
        };
        let bytes = encode_datatype(&dt).unwrap();
        assert_eq!(bytes[1] & 0x01, 1); // BE
        assert_eq!(bytes[1] & 0x08, 0x08); // signed
        assert_eq!(LittleEndian::read_u32(&bytes[4..8]), 2); // 2 bytes
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn encode_float_f64() {
        let dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: ByteOrder::LittleEndian,
        };
        let bytes = encode_datatype(&dt).unwrap();
        assert_eq!(bytes.len(), 20);
        assert_eq!(bytes[0] & 0x0F, 1); // class 1 (float)
        assert_eq!(LittleEndian::read_u32(&bytes[4..8]), 8); // 8 bytes
        assert_eq!(bytes[12], 52); // exponent location
        assert_eq!(bytes[13], 11); // exponent size
        assert_eq!(LittleEndian::read_u32(&bytes[16..20]), 1023); // bias
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn encode_dataspace_scalar() {
        let shape = Shape::scalar();
        let bytes = encode_dataspace(&shape).unwrap();
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes[0], 2); // version
        assert_eq!(bytes[1], 0); // rank 0
        assert_eq!(bytes[3], 0); // type: scalar
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn encode_dataspace_2d() {
        let shape = Shape::fixed(&[10, 20]);
        let bytes = encode_dataspace(&shape).unwrap();
        assert_eq!(bytes.len(), 4 + 16); // header + 2×8 dims
        assert_eq!(bytes[1], 2); // rank 2
        assert_eq!(LittleEndian::read_u64(&bytes[4..12]), 10);
        assert_eq!(LittleEndian::read_u64(&bytes[12..20]), 20);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn encode_contiguous_layout() {
        let ctx = ParseContext::new(8, 8);
        let props = DatasetCreationProps::default(); // contiguous
        let bytes = encode_layout(0x1000, &props, &ctx).unwrap();
        assert_eq!(bytes[0], 3); // version 3
        assert_eq!(bytes[1], 1); // contiguous
        assert_eq!(LittleEndian::read_u64(&bytes[2..10]), 0x1000);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn encode_chunked_layout_with_materialized_index() {
        let ctx = ParseContext::new(8, 8);
        let props = DatasetCreationProps {
            layout: DatasetLayout::Chunked,
            chunk_dims: Some(vec![5, 7]),
            ..DatasetCreationProps::default()
        };
        let bytes = encode_layout_with_chunk_index(0, &props, &ctx, Some(0x4000), Some(4)).unwrap();
        assert_eq!(bytes[0], 3);
        assert_eq!(bytes[1], 2);
        assert_eq!(bytes[2], 3);
        assert_eq!(LittleEndian::read_u64(&bytes[3..11]), 0x4000);
        assert_eq!(LittleEndian::read_u32(&bytes[11..15]), 5);
        assert_eq!(LittleEndian::read_u32(&bytes[15..19]), 7);
        assert_eq!(LittleEndian::read_u32(&bytes[19..23]), 4);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn write_superblock_roundtrip() {
        let mut cursor = consus_io::MemCursor::new();
        let props = FileCreationProps::default();
        let mut state = WriteState::new(props);
        write_superblock(&mut cursor, &mut state, 0x100).unwrap();

        let data = cursor.as_bytes();
        assert_eq!(&data[0..8], &HDF5_MAGIC);
        assert_eq!(data[8], 2); // version
        assert_eq!(data[9], 8); // offset size
        assert_eq!(data[10], 8); // length size
        // Root group address at offset 12 + 3*8 = 36
        assert_eq!(LittleEndian::read_u64(&data[36..44]), 0x100);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn write_group_header_has_ohdr_signature() {
        let mut cursor = consus_io::MemCursor::new();
        let props = FileCreationProps::default();
        let mut state = WriteState::new(props);
        state.eof = 48; // after superblock
        let gcpl = GroupCreationProps::default();
        let addr = write_group_header(&mut cursor, &mut state, &gcpl).unwrap();
        let data = cursor.as_bytes();
        assert_eq!(&data[addr as usize..addr as usize + 4], b"OHDR");
        assert_eq!(data[addr as usize + 4], 2); // version 2
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn encode_hard_link_roundtrip() {
        let ctx = ParseContext::new(8, 8);
        let bytes = encode_hard_link("temperature", 0x200, &ctx).unwrap();
        assert_eq!(bytes[0], 1); // version
        // Name length = 11 (fits in 1 byte)
        assert_eq!(bytes[2], 11);
        // Name bytes
        assert_eq!(&bytes[3..14], b"temperature");
        // Address at offset 14
        assert_eq!(LittleEndian::read_u64(&bytes[14..22]), 0x200);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn encode_attribute_roundtrip() {
        use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};

        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: CoreByteOrder::LittleEndian,
            signed: false,
        };
        let shape = Shape::scalar();
        let raw = 99u32.to_le_bytes();
        let bytes = encode_attribute("count", &dt, &shape, &raw).unwrap();
        assert_eq!(bytes[0], 3); // version 3
        assert_eq!(LittleEndian::read_u16(&bytes[2..4]), 5); // "count".len()
        assert_eq!(&bytes[9..14], b"count");
        assert_eq!(LittleEndian::read_u32(&bytes[bytes.len() - 4..]), 99);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn hdf5_file_builder_produces_valid_superblock() {
        use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};

        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: CoreByteOrder::LittleEndian,
            signed: false,
        };
        let shape = Shape::fixed(&[3]);
        let raw: Vec<u8> = [10u32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        builder
            .add_dataset("temps", &dt, &shape, &raw, &DatasetCreationProps::default())
            .unwrap();
        let bytes = builder.finish().unwrap();

        // Verify HDF5 magic bytes at offset 0.
        assert_eq!(&bytes[0..8], &crate::constants::HDF5_MAGIC);
        assert_eq!(bytes[8], 2); // superblock version 2
        // Root group address at offset 36 (12 + 3*8) must be non-zero.
        assert_ne!(LittleEndian::read_u64(&bytes[36..44]), 0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn add_group_with_children_creates_navigable_nested_group() {
        use consus_core::{Datatype, Shape, StringEncoding};

        let species_bytes = b"Mus musculus";
        let species_dt = Datatype::FixedString {
            length: species_bytes.len(),
            encoding: StringEncoding::Ascii,
        };
        let species_shape = Shape::scalar();

        let subject_attrs: &[(&str, &Datatype, &Shape, &[u8])] =
            &[("species", &species_dt, &species_shape, species_bytes)];

        let subject_spec = ChildGroupSpec {
            name: "subject",
            attributes: subject_attrs,
            datasets: &[],
            sub_groups: &[],
        };

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        builder
            .add_group_with_children("general", &[], &[], &[subject_spec])
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let addr = file
            .open_path("general/subject")
            .expect("navigate to general/subject");
        assert_ne!(addr, 0, "subject group address must be non-zero");

        let attrs = file
            .attributes_at(addr)
            .expect("read attributes at subject");
        let species_attr = attrs
            .iter()
            .find(|a| a.name == "species")
            .expect("species attribute must be present");
        assert_eq!(
            species_attr.raw_data.as_slice(),
            species_bytes,
            "species attribute raw bytes must match 'Mus musculus'"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn add_group_with_children_nested_group_datasets_are_readable() {
        use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};

        let f64_dt = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: CoreByteOrder::LittleEndian,
        };
        let values_shape = Shape::fixed(&[3]);
        let raw_data: Vec<u8> = [1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let raw_data_spec = ChildDatasetSpec {
            name: "raw_data",
            datatype: &f64_dt,
            shape: &values_shape,
            raw_data: &raw_data,
            dcpl: DatasetCreationProps::default(),
            attributes: &[],
        };
        let values_spec = ChildGroupSpec {
            name: "values",
            attributes: &[],
            datasets: &[raw_data_spec],
            sub_groups: &[],
        };

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        builder
            .add_group_with_children("data_container", &[], &[], &[values_spec])
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let dataset_addr = file
            .open_path("data_container/values/raw_data")
            .expect("navigate to raw_data dataset");
        assert_ne!(dataset_addr, 0, "raw_data dataset address must be non-zero");

        let dataset = file
            .dataset_at(dataset_addr)
            .expect("read dataset metadata");
        let data_addr = dataset
            .data_address
            .expect("contiguous dataset must have a data_address");

        let mut buf = [0u8; 24]; // 3 × 8 bytes
        file.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)
            .expect("read contiguous dataset bytes");

        let v0 = LittleEndian::read_f64(&buf[0..8]);
        let v1 = LittleEndian::read_f64(&buf[8..16]);
        let v2 = LittleEndian::read_f64(&buf[16..24]);
        assert_eq!(
            [v0, v1, v2],
            [1.0f64, 2.0, 3.0],
            "decoded f64 values must equal [1.0, 2.0, 3.0]"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn add_group_with_attributes_still_works_after_refactor() {
        use consus_core::{ByteOrder as CoreByteOrder, Datatype, Shape};

        let dt = Datatype::Integer {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: CoreByteOrder::LittleEndian,
            signed: false,
        };
        let shape = Shape::fixed(&[2]);
        let raw: Vec<u8> = [42u32, 99].iter().flat_map(|v| v.to_le_bytes()).collect();

        let child = ChildDatasetSpec {
            name: "my_dataset",
            datatype: &dt,
            shape: &shape,
            raw_data: &raw,
            dcpl: DatasetCreationProps::default(),
            attributes: &[],
        };

        let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
        builder
            .add_group_with_attributes("my_group", &[], &[child])
            .unwrap();
        let bytes = builder.finish().unwrap();

        let file = Hdf5File::open(consus_io::MemCursor::from_bytes(bytes)).expect("open hdf5 file");
        let dataset_addr = file
            .open_path("my_group/my_dataset")
            .expect("navigate to my_group/my_dataset");
        assert_ne!(dataset_addr, 0, "dataset address must be non-zero");

        let dataset = file
            .dataset_at(dataset_addr)
            .expect("read dataset metadata");
        let data_addr = dataset
            .data_address
            .expect("contiguous dataset must have a data_address");

        let mut buf = [0u8; 8]; // 2 × 4 bytes
        file.read_contiguous_dataset_bytes(data_addr, 0, &mut buf)
            .expect("read contiguous dataset bytes");

        assert_eq!(
            LittleEndian::read_u32(&buf[0..4]),
            42,
            "first element must be 42"
        );
        assert_eq!(
            LittleEndian::read_u32(&buf[4..8]),
            99,
            "second element must be 99"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn write_chunk_btree_v1_leaf_header() {
        let mut cursor = consus_io::MemCursor::new();
        let props = FileCreationProps::default();
        let mut state = WriteState::new(props);
        state.eof = 64;

        let entries = vec![ChunkIndexEntry {
            chunk_offsets: vec![0, 0],
            filter_mask: 0,
            chunk_size: 16,
            chunk_address: 0x2000,
        }];

        let addr = write_chunk_btree_v1(&mut cursor, &mut state, &[2, 2], &entries).unwrap();
        let bytes = cursor.as_bytes();
        assert_eq!(&bytes[addr as usize..addr as usize + 4], b"TREE");
        assert_eq!(bytes[addr as usize + 4], 1);
        assert_eq!(bytes[addr as usize + 5], 0);
        assert_eq!(
            LittleEndian::read_u16(&bytes[addr as usize + 6..addr as usize + 8]),
            1
        );
    }
}
