//! Fractal heap structures and parser (HDF5 v2 group link storage).
//!
//! ## Specification (HDF5 File Format Specification, Section IV.A.6.1)
//!
//! The fractal heap stores variable-size objects using a hierarchy of blocks:
//!
//! 1. **Header** (signature `"FRHP"`) — global metadata for the heap.
//! 2. **Direct blocks** (signature `"FHDB"`) — contain actual object data.
//! 3. **Indirect blocks** — point to direct or other indirect blocks,
//!    forming a tree that scales to arbitrary heap sizes.
//!
//! ### Fractal Heap Header Layout
//!
//! | Offset | Size            | Field                                          |
//! |--------|-----------------|------------------------------------------------|
//! | 0      | 4               | Signature `"FRHP"`                             |
//! | 4      | 1               | Version (0)                                    |
//! | 5      | 2               | Heap ID length                                 |
//! | 7      | 2               | I/O filter encoding size (0 = no filters)      |
//! | 9      | 1               | Flags                                          |
//! | 10     | 4               | Maximum managed object size                    |
//! | 14     | L               | Next huge object ID                            |
//! | 14+L   | O               | v2 B-tree address for huge objects              |
//! | …      | O               | Free-space manager address                     |
//! | …      | L               | Managed space amount                           |
//! | …      | L               | Allocated managed space                        |
//! | …      | L               | Iterator offset for managed object allocation  |
//! | …      | L               | Managed objects count                          |
//! | …      | L               | Huge objects size                              |
//! | …      | L               | Huge objects count                             |
//! | …      | L               | Tiny objects size                              |
//! | …      | L               | Tiny objects count                             |
//! | …      | 2               | Table width                                    |
//! | …      | L               | Starting block size                            |
//! | …      | L               | Maximum direct block size                      |
//! | …      | 2               | Max heap size (log₂ bits)                      |
//! | …      | 2               | Starting # of rows in root indirect block      |
//! | …      | O               | Root block address                             |
//! | …      | 2               | Current # of rows in root indirect block       |
//! | …      | (optional)      | Filter info (if I/O filter encoding > 0)       |
//! | …      | 4               | Checksum                                       |
//!
//! Where `L` = superblock length-size, `O` = superblock offset-size.
//!
//! ### Direct Block Layout (signature `"FHDB"`)
//!
//! | Offset | Size                           | Field                     |
//! |--------|--------------------------------|---------------------------|
//! | 0      | 4                              | Signature `"FHDB"`        |
//! | 4      | 1                              | Version (0)               |
//! | 5      | O                              | Heap header address       |
//! | 5+O    | ⌈max_heap_size_bits / 8⌉       | Block offset within heap  |
//! | …      | variable                       | Object data               |
//! | …      | 4 (if flags bit 1 set)         | Checksum                  |
//!
//! ### Heap ID Encoding
//!
//! Bits 6-7 of byte 0 encode the ID type:
//!
//! | Value | Type    | Payload                                    |
//! |-------|---------|--------------------------------------------|
//! | 0     | Managed | Offset within managed space + length       |
//! | 1     | Tiny    | Inline data in the remaining ID bytes      |
//! | 2     | Huge    | v2 B-tree key or direct address            |

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
/// [`ParseContext`](crate::address::ParseContext).
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
}

// ---------------------------------------------------------------------------
// Alloc-dependent: parsing, heap-ID decoding, managed-object reading
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

#[cfg(feature = "alloc")]
use consus_core::{Error, Result};

#[cfg(feature = "alloc")]
use consus_io::ReadAt;

#[cfg(feature = "alloc")]
use crate::address::ParseContext;

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
        println!("FRACTAL HEAP HEADER BYTES: {:02x?}", &buf[0..64]);
        println!("FRACTAL HEAP HEADER BYTES 64..128: {:02x?}", &buf[64..128]);

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

        let _starting_rows = u16::from_le_bytes([buf[pos], buf[pos + 1]]);
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
        })
    }
}

// ---------------------------------------------------------------------------
// Heap ID
// ---------------------------------------------------------------------------

/// Decoded fractal heap object ID.
///
/// The encoding scheme is selected by bits 6-7 of the first byte of the raw
/// heap ID (see module-level documentation).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub enum FractalHeapId {
    /// Managed object: located at `offset` within the managed address space
    /// of the fractal heap, with the given `length` in bytes.
    Managed {
        /// Byte offset within the heap's managed space.
        offset: u64,
        /// Object length in bytes.
        length: u64,
    },

    /// Tiny object: the object data is stored inline in the heap ID bytes
    /// themselves (max ~14 bytes depending on heap-ID length).
    Tiny {
        /// Inline object data.
        data: Vec<u8>,
    },

    /// Huge object: stored outside the managed space and indexed by a v2
    /// B-tree.  `btree_key` is the lookup key (or a direct address when the
    /// heap header's flags bit 0 is set).
    Huge {
        /// B-tree key or direct file address.
        btree_key: u64,
    },
}

/// Decode a raw heap ID according to the fractal heap header parameters.
///
/// # Encoding
///
/// ```text
/// byte 0:  bits 6-7 = type  (0 = managed, 1 = tiny, 2 = huge)
///          bits 4-5 = version (must be 0)
///          bits 0-3 = type-specific
/// ```
///
/// **Managed (type 0)**
///
/// | Offset              | Size                              | Field  |
/// |---------------------|-----------------------------------|--------|
/// | 1                   | ⌈max_heap_size_bits / 8⌉          | Offset |
/// | 1 + offset_bytes    | heap_id_length − 1 − offset_bytes | Length |
///
/// **Tiny (type 1)**
///
/// Remaining bytes (1 .. heap_id_length) carry inline object data.
///
/// **Huge (type 2)**
///
/// Remaining bytes encode the v2 B-tree key as a little-endian integer.
///
/// # Errors
///
/// - [`Error::InvalidFormat`] on empty input, unknown type, or truncated ID.
#[cfg(feature = "alloc")]
pub fn decode_heap_id(id_bytes: &[u8], header: &FractalHeapHeader) -> Result<FractalHeapId> {
    if id_bytes.is_empty() {
        return Err(Error::InvalidFormat {
            message: String::from("empty fractal heap ID"),
        });
    }

    let first = id_bytes[0];
    let id_type = (first >> 6) & 0x03;

    match id_type {
        // -- Managed ---------------------------------------------------------
        0 => {
            let offset_bytes = ((header.max_heap_size_bits as usize) + 7) / 8;
            let min_len = 1 + offset_bytes;
            if id_bytes.len() < min_len {
                return Err(Error::InvalidFormat {
                    message: String::from("managed heap ID too short"),
                });
            }
            let offset = read_uint_le(&id_bytes[1..], offset_bytes);
            let length_bytes = id_bytes.len() - 1 - offset_bytes;
            let length = if length_bytes > 0 {
                read_uint_le(&id_bytes[1 + offset_bytes..], length_bytes)
            } else {
                0
            };
            Ok(FractalHeapId::Managed { offset, length })
        }

        // -- Tiny ------------------------------------------------------------
        1 => {
            // Bits 0-3 of byte 0 encode (actual_length − 1) for normal tiny
            // objects.  The inline data occupies bytes 1..heap_id_length.
            let data = id_bytes[1..].to_vec();
            Ok(FractalHeapId::Tiny { data })
        }

        // -- Huge ------------------------------------------------------------
        2 => {
            let key_bytes = id_bytes.len() - 1;
            if key_bytes == 0 {
                return Err(Error::InvalidFormat {
                    message: String::from("huge heap ID has no key bytes"),
                });
            }
            let btree_key = read_uint_le(&id_bytes[1..], key_bytes.min(8));
            Ok(FractalHeapId::Huge { btree_key })
        }

        _ => Err(Error::InvalidFormat {
            message: alloc::format!("unknown fractal heap ID type: {id_type}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Managed object reading
// ---------------------------------------------------------------------------

/// Read a managed object from a fractal heap.
///
/// Given a managed-object `offset` and `length` (as decoded from a
/// [`FractalHeapId::Managed`] variant), compute the absolute file address
/// within the root direct block and read the data.
///
/// # Current Limitations
///
/// This implementation handles only the case where the root block is a
/// **direct block** (`header.root_indirect_rows == 0`).  Indirect-block
/// traversal (needed for larger heaps) returns
/// [`Error::UnsupportedFeature`].
///
/// # Direct Block Overhead
///
/// ```text
/// overhead = 5 (sig + ver)
///          + offset_size          (heap header address)
///          + ⌈max_heap_size_bits / 8⌉  (block offset field)
/// ```
///
/// The object data begins at `root_block_address + overhead + managed_offset`.
///
/// # Errors
///
/// - [`Error::UnsupportedFeature`] if the root is an indirect block.
/// - Propagates I/O errors from `source.read_at`.
#[cfg(feature = "alloc")]
pub fn read_managed_object<R: ReadAt>(
    source: &R,
    header: &FractalHeapHeader,
    offset: u64,
    length: u64,
    ctx: &ParseContext,
) -> Result<Vec<u8>> {
    // Determine the direct block address containing the object
    let mut block_address = header.root_block_address;
    let mut indirect_rows = header.root_indirect_rows;

    let heap_offset_field_bytes = ((header.max_heap_size_bits as usize) + 7) / 8;
    
    // Direct block sizes and counts
    let table_width = header.table_width as usize;
    let starting_block_size = header.starting_block_size;
    let max_direct_rows = if starting_block_size > 0 {
        (header.max_direct_block_size.ilog2() as usize) - (starting_block_size.ilog2() as usize) + 2
    } else {
        0
    };

    let mut current_offset = offset;

    // Traverse indirect blocks if necessary
    while indirect_rows > 0 {
        // Read indirect block header (sig: 4, ver: 1, heap_addr: offset_size, block_offset: var)
        let ib_overhead = 5 + ctx.offset_bytes() + heap_offset_field_bytes;
        
        let mut row = 0;
        let mut col;
        let mut block_size = starting_block_size;
        let mut row_offset = 0;
        
        // Find which row/col the current_offset falls into
        loop {
            let row_size = block_size * table_width as u64;
            if current_offset < row_offset + row_size {
                col = ((current_offset - row_offset) / block_size) as usize;
                break;
            }
            row_offset += row_size;
            row += 1;
            if row == 0 || row == 1 {
                // row 0 and 1 have same block size
                block_size = starting_block_size;
            } else {
                block_size *= 2;
            }
        }
        
        // Read the address from the indirect block
        let entry_idx = row * table_width + col;
        let entry_offset = block_address + ib_overhead as u64 + (entry_idx * ctx.offset_bytes()) as u64;
        
        let mut addr_buf = vec![0u8; ctx.offset_bytes()];
        source.read_at(entry_offset, &mut addr_buf)?;
        let child_address = ctx.read_offset(&addr_buf);
        if child_address == crate::constants::UNDEFINED_ADDRESS {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!("Fractal heap indirect block points to undefined address"),
            });
        }
        
        if row < max_direct_rows {
            // It points to a direct block
            block_address = child_address;
            indirect_rows = 0;
            current_offset = current_offset - (row_offset + col as u64 * block_size);
        } else {
            // It points to another indirect block
            // The number of rows in the child indirect block depends on its row
            // wait, calculating child rows can be complex, but for small files maybe it's just 1 level?
            return Err(Error::UnsupportedFeature {
                feature: alloc::format!("Fractal heap nested indirect block traversal (row > max_direct_rows)"),
            });
        }
    }

    // Direct block header: signature(4) + version(1) + heap_addr(O) +
    // block_offset(variable).  Checksum, if present (flags bit 1), is at the
    // END of the block and does not affect the data start position.
    let direct_block_overhead: u64 = (5 + ctx.offset_bytes() + heap_offset_field_bytes) as u64;

    let mut db_header_buf = vec![0u8; direct_block_overhead as usize];
    source.read_at(block_address, &mut db_header_buf)?;
    println!("DIRECT BLOCK HEADER at {}: {:02x?}", block_address, db_header_buf);

    let data_address = block_address
        .checked_add(current_offset)
        .ok_or(Error::Overflow)?;

    let len = length as usize;
    let mut buf = vec![0u8; len];
    source.read_at(data_address, &mut buf)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// HUGE object reading
// ---------------------------------------------------------------------------

/// Read a HUGE object from a fractal heap.
///
/// HUGE objects are stored outside the managed address space and indexed
/// by a v2 B-tree. The `btree_key` from [`FractalHeapId::Huge`] is either:
/// - A **direct file address** (when header flag bit 0 is SET), or
/// - A **B-tree key** to search (when header flag bit 0 is CLEAR).
///
/// ## Arguments
///
/// - `source`: I/O source.
/// - `header`: Parsed fractal heap header.
/// - `btree_key`: The key from `FractalHeapId::Huge`.
/// - `ctx`: Parse context.
///
/// ## Returns
///
/// The raw object bytes, or an error on I/O or parse failure.
/// Filtered huge objects (flag bit 2 set) currently return
/// [`Error::UnsupportedFeature`] — I/O filter pipeline integration
/// is deferred.
///
/// ## Errors
///
/// - [`Error::UnsupportedFeature`] if flag bit 2 (huge-objects-filtered)
///   is set (filtered data requires I/O filter pipeline).
/// - Propagates errors from B-tree search or direct address read.
#[cfg(feature = "alloc")]
pub fn read_huge_object<R: ReadAt>(
    source: &R,
    header: &FractalHeapHeader,
    btree_key: u64,
    ctx: &ParseContext,
) -> Result<Vec<u8>> {
    use crate::btree::v2::find_huge_object_record;

    // Case 1: Direct address mode (flag bit 0 set)
    if header.flags & 0x01 != 0 {
        // btree_key IS the direct file address of the object.
        // We still need to know the length. For direct-mode HUGE objects,
        // the length is stored separately in the B-tree record — we must
        // search the B-tree to get the length even in direct mode.
        if header.huge_object_btree_address == crate::constants::UNDEFINED_ADDRESS {
            return Err(Error::InvalidFormat {
                message: String::from(
                    "huge object direct-address mode but B-tree address is undefined",
                ),
            });
        }
        let btree_header =
            crate::btree::v2::BTreeV2Header::parse(source, header.huge_object_btree_address, ctx)?;
        let location = find_huge_object_record(
            source,
            header.huge_object_btree_address,
            &btree_header,
            btree_key,
            ctx,
        )?
        .ok_or_else(|| Error::InvalidFormat {
            message: String::from("huge object record not found in B-tree"),
        })?;

        let mut buf = vec![0u8; location.length as usize];
        source.read_at(location.address, &mut buf)?;
        return Ok(buf);
    }

    // Case 2: B-tree lookup mode (flag bit 0 clear)
    // btree_key is the search key in the HUGE object B-tree.
    if header.huge_object_btree_address == crate::constants::UNDEFINED_ADDRESS {
        return Err(Error::InvalidFormat {
            message: String::from("huge object B-tree address is undefined"),
        });
    }

    // Check if huge objects are filtered (flag bit 2)
    if header.flags & 0x04 != 0 {
        // HUGE objects are filtered — would need I/O filter pipeline.
        // This is not yet implemented.
        return Err(Error::UnsupportedFeature {
            feature: String::from("filtered huge fractal heap objects require I/O filter pipeline"),
        });
    }

    let btree_header =
        crate::btree::v2::BTreeV2Header::parse(source, header.huge_object_btree_address, ctx)?;
    let location = find_huge_object_record(
        source,
        header.huge_object_btree_address,
        &btree_header,
        btree_key,
        ctx,
    )?
    .ok_or_else(|| Error::InvalidFormat {
        message: String::from("huge object record not found in B-tree"),
    })?;

    let mut buf = vec![0u8; location.length as usize];
    source.read_at(location.address, &mut buf)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Helpers (crate-private)
// ---------------------------------------------------------------------------

/// Read an unsigned little-endian integer of 1–8 bytes.
#[cfg(feature = "alloc")]
fn read_uint_le(data: &[u8], size: usize) -> u64 {
    match size {
        0 => 0,
        1 => data[0] as u64,
        2 => u16::from_le_bytes([data[0], data[1]]) as u64,
        3 => u32::from_le_bytes([data[0], data[1], data[2], 0]) as u64,
        4 => u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64,
        5 => u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], 0, 0, 0]),
        6 => u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], data[5], 0, 0]),
        7 => u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], 0,
        ]),
        _ => u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::address::ParseContext;

    /// Round-trip: build a minimal fractal heap header image, parse it, and
    /// verify every field.
    #[test]
    fn parse_header_round_trip() {
        let ctx = ParseContext::new(8, 8);
        let s = ctx.length_bytes();
        let o = ctx.offset_bytes();

        // Pre-compute expected total size (before optional filter info and
        // checksum): 14 fixed + 10*s + 3*o + 2 + 2*s + 2 + 2 + o + 2
        // With s=8, o=8: 14 + 80 + 24 + 2 + 16 + 2 + 2 + 8 + 2 = 150
        let header_size = 14 + 10 * s + 3 * o + 2 + 2 * s + 2 + 2 + o + 2;
        let total = header_size + 4; // + checksum
        let mut data = vec![0u8; total.max(256)];

        // Signature + version
        data[0..4].copy_from_slice(b"FRHP");
        data[4] = 0; // version

        // Fixed fields
        data[5..7].copy_from_slice(&42u16.to_le_bytes()); // heap_id_length
        data[7..9].copy_from_slice(&0u16.to_le_bytes()); // io_filter_size
        data[9] = 0x03; // flags
        data[10..14].copy_from_slice(&1024u32.to_le_bytes()); // max_managed_object_size

        let mut pos = 14usize;

        // next_huge_id (skip)
        data[pos..pos + s].copy_from_slice(&99u64.to_le_bytes()[..s]);
        pos += s;

        // huge_object_btree_address
        data[pos..pos + o].copy_from_slice(&0x1000u64.to_le_bytes()[..o]);
        pos += o;

        // free_managed_space
        data[pos..pos + s].copy_from_slice(&50u64.to_le_bytes()[..s]);
        pos += s;

        // free_space_manager_address
        data[pos..pos + o].copy_from_slice(&0x2000u64.to_le_bytes()[..o]);
        pos += o;

        // managed_space
        data[pos..pos + s].copy_from_slice(&4096u64.to_le_bytes()[..s]);
        pos += s;

        // allocated_managed_space
        data[pos..pos + s].copy_from_slice(&2048u64.to_le_bytes()[..s]);
        pos += s;

        // iter_offset (skip)
        pos += s;

        // managed_object_count
        data[pos..pos + s].copy_from_slice(&10u64.to_le_bytes()[..s]);
        pos += s;

        // huge_object_size
        data[pos..pos + s].copy_from_slice(&500u64.to_le_bytes()[..s]);
        pos += s;

        // huge_object_count
        data[pos..pos + s].copy_from_slice(&2u64.to_le_bytes()[..s]);
        pos += s;

        // tiny_object_size
        data[pos..pos + s].copy_from_slice(&30u64.to_le_bytes()[..s]);
        pos += s;

        // tiny_object_count
        data[pos..pos + s].copy_from_slice(&5u64.to_le_bytes()[..s]);
        pos += s;

        // table_width
        data[pos..pos + 2].copy_from_slice(&4u16.to_le_bytes());
        pos += 2;

        // starting_block_size
        data[pos..pos + s].copy_from_slice(&512u64.to_le_bytes()[..s]);
        pos += s;

        // max_direct_block_size
        data[pos..pos + s].copy_from_slice(&65536u64.to_le_bytes()[..s]);
        pos += s;

        // max_heap_size_bits
        data[pos..pos + 2].copy_from_slice(&16u16.to_le_bytes());
        pos += 2;

        // starting_rows
        data[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
        pos += 2;

        // root_block_address
        data[pos..pos + o].copy_from_slice(&0x3000u64.to_le_bytes()[..o]);
        pos += o;

        // root_indirect_rows
        data[pos..pos + 2].copy_from_slice(&0u16.to_le_bytes());
        // pos += 2;

        let reader = consus_io::SliceReader::new(&data);
        let hdr = FractalHeapHeader::parse(&reader, 0, &ctx).unwrap();

        assert_eq!(hdr.heap_id_length, 42);
        assert_eq!(hdr.io_filter_size, 0);
        assert_eq!(hdr.flags, 0x03);
        assert_eq!(hdr.max_managed_object_size, 1024);
        assert_eq!(hdr.huge_object_btree_address, 0x1000);
        assert_eq!(hdr.free_managed_space, 50);
        assert_eq!(hdr.free_space_manager_address, 0x2000);
        assert_eq!(hdr.managed_space, 4096);
        assert_eq!(hdr.allocated_managed_space, 2048);
        assert_eq!(hdr.managed_object_count, 10);
        assert_eq!(hdr.huge_object_size, 500);
        assert_eq!(hdr.huge_object_count, 2);
        assert_eq!(hdr.tiny_object_size, 30);
        assert_eq!(hdr.tiny_object_count, 5);
        assert_eq!(hdr.table_width, 4);
        assert_eq!(hdr.starting_block_size, 512);
        assert_eq!(hdr.max_direct_block_size, 65536);
        assert_eq!(hdr.max_heap_size_bits, 16);
        assert_eq!(hdr.root_block_address, 0x3000);
        assert_eq!(hdr.root_indirect_rows, 0);
    }

    #[test]
    fn parse_header_bad_signature() {
        let ctx = ParseContext::new(8, 8);
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(b"XXXX");
        let reader = consus_io::SliceReader::new(&data);
        let err = FractalHeapHeader::parse(&reader, 0, &ctx).unwrap_err();
        assert!(
            matches!(err, Error::InvalidFormat { .. }),
            "expected InvalidFormat, got: {err:?}"
        );
    }

    #[test]
    fn decode_managed_heap_id() {
        let header = FractalHeapHeader {
            heap_id_length: 7,
            io_filter_size: 0,
            flags: 0,
            max_managed_object_size: 1024,
            huge_object_btree_address: 0,
            free_space_manager_address: 0,
            managed_space: 4096,
            allocated_managed_space: 2048,
            managed_object_count: 1,
            huge_object_size: 0,
            huge_object_count: 0,
            tiny_object_size: 0,
            tiny_object_count: 0,
            table_width: 4,
            starting_block_size: 512,
            max_direct_block_size: 65536,
            max_heap_size_bits: 16,
            root_block_address: 0,
            root_indirect_rows: 0,
        };

        // Type = 0 (managed), version = 0 → byte 0 = 0x00
        // max_heap_size_bits = 16 → offset_bytes = 2
        // id_bytes = [0x00, offset_lo, offset_hi, len_lo, len_mid, len_hi, len_top]
        //            type+ver   offset=0x0100         length=0x00000080
        let id_bytes: [u8; 7] = [0x00, 0x00, 0x01, 0x80, 0x00, 0x00, 0x00];
        let id = decode_heap_id(&id_bytes, &header).unwrap();
        match id {
            FractalHeapId::Managed { offset, length } => {
                assert_eq!(offset, 0x0100);
                assert_eq!(length, 0x80);
            }
            other => panic!("expected Managed, got: {other:?}"),
        }
    }

    #[test]
    fn decode_tiny_heap_id() {
        let header = FractalHeapHeader {
            heap_id_length: 5,
            io_filter_size: 0,
            flags: 0,
            max_managed_object_size: 64,
            huge_object_btree_address: 0,
            free_space_manager_address: 0,
            managed_space: 0,
            allocated_managed_space: 0,
            managed_object_count: 0,
            huge_object_size: 0,
            huge_object_count: 0,
            tiny_object_size: 0,
            tiny_object_count: 0,
            table_width: 4,
            starting_block_size: 256,
            max_direct_block_size: 4096,
            max_heap_size_bits: 8,
            root_block_address: 0,
            root_indirect_rows: 0,
        };

        // Type = 1 (tiny) → bits 6-7 = 01 → byte 0 = 0b0100_0000 = 0x40
        let id_bytes: [u8; 5] = [0x40, 0xAA, 0xBB, 0xCC, 0xDD];
        let id = decode_heap_id(&id_bytes, &header).unwrap();
        match id {
            FractalHeapId::Tiny { data } => {
                assert_eq!(data, vec![0xAA, 0xBB, 0xCC, 0xDD]);
            }
            other => panic!("expected Tiny, got: {other:?}"),
        }
    }

    #[test]
    fn decode_huge_heap_id() {
        let header = FractalHeapHeader {
            heap_id_length: 9,
            io_filter_size: 0,
            flags: 0,
            max_managed_object_size: 128,
            huge_object_btree_address: 0x5000,
            free_space_manager_address: 0,
            managed_space: 0,
            allocated_managed_space: 0,
            managed_object_count: 0,
            huge_object_size: 0,
            huge_object_count: 0,
            tiny_object_size: 0,
            tiny_object_count: 0,
            table_width: 4,
            starting_block_size: 256,
            max_direct_block_size: 4096,
            max_heap_size_bits: 16,
            root_block_address: 0,
            root_indirect_rows: 0,
        };

        // Type = 2 (huge) → bits 6-7 = 10 → byte 0 = 0b1000_0000 = 0x80
        let mut id_bytes = [0u8; 9];
        id_bytes[0] = 0x80;
        id_bytes[1..9].copy_from_slice(&0x0000_DEAD_BEEF_0000u64.to_le_bytes());
        let id = decode_heap_id(&id_bytes, &header).unwrap();
        match id {
            FractalHeapId::Huge { btree_key } => {
                assert_eq!(btree_key, 0x0000_DEAD_BEEF_0000);
            }
            other => panic!("expected Huge, got: {other:?}"),
        }
    }

    #[test]
    fn read_managed_object_direct_root() {
        let ctx = ParseContext::new(8, 8);
        let header = FractalHeapHeader {
            heap_id_length: 7,
            io_filter_size: 0,
            flags: 0,
            max_managed_object_size: 1024,
            huge_object_btree_address: 0,
            free_space_manager_address: 0,
            managed_space: 256,
            allocated_managed_space: 256,
            managed_object_count: 1,
            huge_object_size: 0,
            huge_object_count: 0,
            tiny_object_size: 0,
            tiny_object_count: 0,
            table_width: 4,
            starting_block_size: 256,
            max_direct_block_size: 65536,
            max_heap_size_bits: 16,
            root_block_address: 0, // root block at file offset 0
            root_indirect_rows: 0, // direct block
        };

        // Build a direct block image at offset 0.
        // Overhead: 5 (sig+ver) + 8 (heap addr) + 2 (block offset, ceil(16/8))
        // = 15 bytes of header before data.
        let overhead = 5 + ctx.offset_bytes() + 2; // 15
        let block_size = overhead + 256; // data area = 256 bytes
        let mut image = vec![0u8; block_size];

        image[0..4].copy_from_slice(b"FHDB");
        image[4] = 0; // version
        // heap header address (8 bytes, irrelevant for this test)
        // block offset (2 bytes, 0)

        // Write a known payload at managed offset 10, length 5.
        let payload = b"HELLO";
        let data_start = overhead + 10;
        image[data_start..data_start + 5].copy_from_slice(payload);

        let reader = consus_io::SliceReader::new(&image);
        let result = read_managed_object(&reader, &header, 10, 5, &ctx).unwrap();
        assert_eq!(result, b"HELLO");
    }

    #[test]
    fn read_managed_object_rejects_indirect_root() {
        let ctx = ParseContext::new(8, 8);
        let header = FractalHeapHeader {
            heap_id_length: 7,
            io_filter_size: 0,
            flags: 0,
            max_managed_object_size: 1024,
            huge_object_btree_address: 0,
            free_space_manager_address: 0,
            managed_space: 0,
            allocated_managed_space: 0,
            managed_object_count: 0,
            huge_object_size: 0,
            huge_object_count: 0,
            tiny_object_size: 0,
            tiny_object_count: 0,
            table_width: 4,
            starting_block_size: 256,
            max_direct_block_size: 65536,
            max_heap_size_bits: 16,
            root_block_address: 0x1000,
            root_indirect_rows: 2, // indirect
        };

        let data = vec![0u8; 4096];
        let reader = consus_io::SliceReader::new(&data);
        let err = read_managed_object(&reader, &header, 0, 10, &ctx).unwrap_err();
        assert!(
            matches!(err, Error::UnsupportedFeature { .. }),
            "expected UnsupportedFeature, got: {err:?}"
        );
    }

    #[test]
    fn read_uint_le_various_widths() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(read_uint_le(&data, 0), 0);
        assert_eq!(read_uint_le(&data, 1), 0x01);
        assert_eq!(read_uint_le(&data, 2), 0x0201);
        assert_eq!(read_uint_le(&data, 3), 0x0003_0201);
        assert_eq!(read_uint_le(&data, 4), 0x0403_0201);
        assert_eq!(read_uint_le(&data, 8), 0x0807_0605_0403_0201);
    }
}
