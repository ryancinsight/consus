//! Write-state bookkeeping, superblock emission, and low-level v2 message encoding primitives.

#[cfg(feature = "alloc")]
use super::FileCreationProps;
#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use crate::constants::{HDF5_MAGIC, UNDEFINED_ADDRESS};
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};
#[cfg(feature = "alloc")]
use consus_compression::Checksum;
#[cfg(feature = "alloc")]
use consus_core::Result;
#[cfg(feature = "alloc")]
use consus_io::WriteAt;

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
pub(crate) fn write_offset(buf: &mut [u8], size: usize, value: u64) {
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
/// | 12+4S | 4 | Jenkins lookup3 checksum |
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
    let total_len = data_len + 4; // + Jenkins lookup3 checksum (4 bytes)

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

    // Jenkins lookup3 checksum over bytes [0..data_len) per HDF5 spec §III.A
    let checksum = consus_compression::Lookup3::compute(&buf[..pos]);
    buf[pos..pos + 4].copy_from_slice(&checksum.to_le_bytes());

    sink.write_at(addr, &buf)
}

/// Overwrite the EOF address field in an existing v2 superblock at offset 0.
///
/// The EOF field is located at byte offset `12 + 2 * offset_size`.
/// After updating, the Jenkins lookup3 checksum is recomputed and written.
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

    // Recompute Jenkins lookup3 checksum per HDF5 spec §III.A
    let checksum = consus_compression::Lookup3::compute(&buf[..data_len]);
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
pub(crate) fn chunk_size_encoding(data_size: usize) -> (usize, u8) {
    if data_size < 256 {
        (1, 0x00)
    } else if data_size < 65536 {
        (2, 0x01)
    } else if usize::try_from(1_u64 << 32).map_or(true, |limit| data_size < limit) {
        (4, 0x02)
    } else {
        (8, 0x03)
    }
}

/// Write the chunk data size into `buf` using the given field width.
#[cfg(feature = "alloc")]
pub(crate) fn write_chunk_size(buf: &mut [u8], width: usize, value: usize) {
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
/// | 0 | 1 | Message type |
/// | 1 | 2 | Data size |
/// | 3 | 1 | Flags |
///
/// Returns the number of bytes written (4 + data.len()).
#[cfg(feature = "alloc")]
pub(crate) fn write_v2_message(
    buf: &mut [u8],
    pos: usize,
    msg_type: u16,
    flags: u8,
    data: &[u8],
) -> usize {
    buf[pos] = msg_type as u8;
    LittleEndian::write_u16(&mut buf[pos + 1..], data.len() as u16);
    buf[pos + 3] = flags;
    buf[pos + 4..pos + 4 + data.len()].copy_from_slice(data);
    4 + data.len()
}
