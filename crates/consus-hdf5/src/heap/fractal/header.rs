//! Fractal heap header (`FRHP`) model and parsing.

#[cfg(feature = "alloc")]
use crate::address::ParseContext;
use consus_core::{Error, Result};
#[cfg(feature = "alloc")]
use consus_io::ReadAt;

/// Fractal heap header signature.
pub const FRACTAL_HEAP_SIGNATURE: [u8; 4] = *b"FRHP";

/// Direct block signature.
pub const DIRECT_BLOCK_SIGNATURE: [u8; 4] = *b"FHDB";

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// Parsed fractal heap header.
///
/// All variable-width fields (offset-size and length-size) have been widened
/// to `u64` during parsing. The original encoding widths are determined by
/// [`ParseContext`].
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    /// Number of bytes in a heap ID (used by link messages to reference
    /// objects stored in this heap).
    pub heap_id_length: u16,

    /// Size of the I/O filter encoding; 0 if no filters are applied to
    /// direct blocks.
    pub io_filter_size: u16,

    /// Flags (bit 0: huge-ID-direct, bit 1: direct-block checksums,
    /// bit 2: huge-objects-filtered).
    pub flags: u8,

    /// Maximum size of a managed object (objects larger than this are
    /// stored as huge objects via the v2 B-tree).
    pub max_managed_object_size: u32,

    /// v2 B-tree address used for huge object storage.
    pub huge_object_btree_address: u64,

    /// Amount of free space in managed blocks.
    pub free_managed_space: u64,

    /// Address of the free-space manager for managed blocks.
    pub free_space_manager_address: u64,

    /// Total amount of managed space in the heap (bytes).
    pub managed_space: u64,

    /// Amount of managed space that has been allocated (bytes).
    pub allocated_managed_space: u64,

    /// Number of managed objects currently stored.
    pub managed_object_count: u64,

    /// Total size of all huge objects (bytes).
    pub huge_object_size: u64,

    /// Number of huge objects.
    pub huge_object_count: u64,

    /// Total size of all tiny objects (bytes).
    pub tiny_object_size: u64,

    /// Number of tiny objects.
    pub tiny_object_count: u64,

    /// Table width (number of direct block columns per indirect-block row).
    pub table_width: u16,

    /// Size of the first direct block allocated (bytes).
    pub starting_block_size: u64,

    /// Maximum direct block size (bytes); blocks larger than this are
    /// stored through indirect blocks.
    pub max_direct_block_size: u64,

    /// Log₂ of the maximum heap address space (bits).  Used to size the
    /// offset field in managed heap IDs and the block-offset field in
    /// direct blocks.
    pub max_heap_size_bits: u16,

    /// File address of the root block (direct or indirect).
    pub root_block_address: u64,

    /// Current number of rows in the root indirect block.  When 0 the
    /// root block is a direct block.
    pub root_indirect_rows: u16,

    /// Initial ("starting") number of rows in the root indirect block.
    /// Rows 0..starting_rows all have block size equal to `starting_block_size`;
    /// row `starting_rows + k` has size `starting_block_size << (k + 1)`.
    /// The value 0 in the file is treated as 1 per the HDF5 spec.
    pub starting_rows: u16,
}

// ---------------------------------------------------------------------------
// Alloc-dependent: parsing, heap-ID decoding, managed-object reading
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl FractalHeapHeader {
    /// Maximum buffer size required for the header (with 8-byte offsets and
    /// lengths, no I/O filters): 146 bytes.  256 provides headroom for
    /// filter-info fields.
    const HEADER_BUF_SIZE: usize = 256;

    /// Parse a fractal heap header from `source` at the given `address`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidFormat`] if the signature or version is wrong.
    /// - Propagates I/O errors from `source.read_at`.
    ///
    /// # Layout
    ///
    /// See module-level documentation for the byte layout.
    pub fn parse<R: ReadAt>(source: &R, address: u64, ctx: &ParseContext) -> Result<Self> {
        let mut buf = [0u8; Self::HEADER_BUF_SIZE];
        source.read_at(address, &mut buf)?;

        // -- Signature -------------------------------------------------------
        if buf[0..4] != FRACTAL_HEAP_SIGNATURE {
            return Err(Error::InvalidFormat {
                message: String::from("invalid fractal heap signature"),
            });
        }

        // -- Version ---------------------------------------------------------
        let version = buf[4];
        if version != 0 {
            return Err(Error::InvalidFormat {
                message: alloc::format!("unsupported fractal heap version: {version}"),
            });
        }

        // -- Fixed-width fields (bytes 5-13) ---------------------------------
        let heap_id_length = u16::from_le_bytes([buf[5], buf[6]]);
        let io_filter_size = u16::from_le_bytes([buf[7], buf[8]]);
        let flags = buf[9];
        let max_managed_object_size = u32::from_le_bytes([buf[10], buf[11], buf[12], buf[13]]);

        // -- Variable-width fields -------------------------------------------
        let s = ctx.length_bytes();
        let o = ctx.offset_bytes();
        let mut pos: usize = 14;

        let _next_huge_id = ctx.read_length(&buf[pos..]);
        pos += s;

        let huge_object_btree_address = ctx.read_offset(&buf[pos..]);
        pos += o;

        let free_managed_space = ctx.read_length(&buf[pos..]);
        pos += s;

        let free_space_manager_address = ctx.read_offset(&buf[pos..]);
        pos += o;

        let managed_space = ctx.read_length(&buf[pos..]);
        pos += s;

        let allocated_managed_space = ctx.read_length(&buf[pos..]);
        pos += s;

        let _iter_offset = ctx.read_length(&buf[pos..]);
        pos += s;

        let managed_object_count = ctx.read_length(&buf[pos..]);
        pos += s;

        let huge_object_size = ctx.read_length(&buf[pos..]);
        pos += s;

        let huge_object_count = ctx.read_length(&buf[pos..]);
        pos += s;

        let tiny_object_size = ctx.read_length(&buf[pos..]);
        pos += s;

        let tiny_object_count = ctx.read_length(&buf[pos..]);
        pos += s;

        let table_width = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;

        let starting_block_size = ctx.read_length(&buf[pos..]);
        pos += s;

        let max_direct_block_size = ctx.read_length(&buf[pos..]);
        pos += s;

        let max_heap_size_bits = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;

        let raw_starting_rows = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        let starting_rows = if raw_starting_rows == 0 {
            1
        } else {
            raw_starting_rows
        };
        pos += 2;

        let root_block_address = ctx.read_offset(&buf[pos..]);
        pos += o;

        let root_indirect_rows = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
        // pos += 2;  (not needed; remaining bytes are optional filter info + checksum)

        Ok(Self {
            heap_id_length,
            io_filter_size,
            flags,
            max_managed_object_size,
            huge_object_btree_address,
            free_managed_space,
            free_space_manager_address,
            managed_space,
            allocated_managed_space,
            managed_object_count,
            huge_object_size,
            huge_object_count,
            tiny_object_size,
            tiny_object_count,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size_bits,
            root_block_address,
            root_indirect_rows,
            starting_rows,
        })
    }
}
