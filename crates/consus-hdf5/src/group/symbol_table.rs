//! HDF5 v1 group symbol table structures.
//!
//! ## Specification
//!
//! Version 1 groups store their membership via a **symbol table message**
//! (header message type 0x0011) that references a B-tree v1 and a local
//! heap. The B-tree leaf nodes point to **symbol table nodes** ("SNOD"),
//! each containing an array of **symbol table entries**.
//!
//! Reference: *HDF5 File Format Specification Version 3.0*, Sections
//! III.D (Symbol Table Entry), III.C (Symbol Table Node), IV.A.2.v
//! (Symbol Table Message).
//! (<https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html>)
//!
//! ## Symbol Table Message Layout (header message type 0x0011)
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0      | S    | B-tree v1 address (group name index) |
//! | S      | S    | Local heap address (link name storage) |
//!
//! Where `S` = `offset_size` from the superblock (2, 4, or 8 bytes).
//!
//! ## Symbol Table Node Layout (signature "SNOD")
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0      | 4    | Signature (`"SNOD"`, `[0x53, 0x4E, 0x4F, 0x44]`) |
//! | 4      | 1    | Version (must be 1) |
//! | 5      | 1    | Reserved |
//! | 6      | 2    | Number of symbols (little-endian u16) |
//! | 8      | var  | Array of symbol table entries |
//!
//! ## Symbol Table Entry Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0      | S    | Link name offset into local heap data segment |
//! | S      | S    | Object header address |
//! | 2S     | 4    | Cache type (little-endian u32) |
//! | 2S+4   | 4    | Reserved (must be zero) |
//! | 2S+8   | 16   | Scratch-pad space (interpretation depends on cache type) |
//!
//! Total entry size: `2*S + 24` bytes.
//!
//! ### Cache Types
//!
//! | Value | Meaning | Scratch-pad contents |
//! |-------|---------|----------------------|
//! | 0     | None    | Unused (all zeros) |
//! | 1     | Group   | `[0..S]` B-tree address of child group, `[S..2S]` heap address of child group |
//! | 2     | Symbolic link | `[0..4]` offset into local heap for link target string (little-endian u32) |
//!
//! For cache type 1, the scratch-pad stores two `offset_size`-byte addresses
//! packed into the first `2*S` bytes of the 16-byte scratch region. The
//! remaining `16 - 2*S` bytes are unused.
//!
//! For cache type 2, only the first 4 bytes carry data (the heap offset);
//! the remaining 12 bytes are unused.

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

use consus_core::{Error, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;
use crate::constants::UNDEFINED_ADDRESS;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Symbol table node signature: ASCII `"SNOD"`.
pub const SNOD_SIGNATURE: [u8; 4] = *b"SNOD";

/// Expected symbol table node version.
const SNOD_VERSION: u8 = 1;

/// Fixed size of the symbol table node header (signature + version +
/// reserved + num_symbols): 4 + 1 + 1 + 2 = 8 bytes.
const SNOD_HEADER_SIZE: usize = 8;

/// Fixed size of the scratch-pad region in each symbol table entry.
const SCRATCH_PAD_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// Symbol Table Message
// ---------------------------------------------------------------------------

/// Parsed symbol table message (header message type 0x0011).
///
/// Contains the two addresses that together define a v1 group's membership:
/// a B-tree v1 for the name index and a local heap for link name storage.
///
/// ## Layout
///
/// Two consecutive `offset_size`-byte addresses:
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | S    | B-tree v1 address |
/// | S      | S    | Local heap address |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolTableMessage {
    /// File address of the B-tree v1 root node that indexes group members
    /// by name.
    pub btree_address: u64,
    /// File address of the local heap that stores the null-terminated link
    /// name strings referenced by symbol table entries.
    pub local_heap_address: u64,
}

impl SymbolTableMessage {
    /// Parse a symbol table message from raw header message payload bytes.
    ///
    /// # Arguments
    ///
    /// * `data` — Raw bytes of the symbol table header message payload
    ///   (after the standard header-message envelope has been stripped).
    /// * `ctx`  — Parsing context carrying `offset_size`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidFormat` if `data` is shorter than
    /// `2 * offset_size` bytes.
    pub fn parse(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        let s = ctx.offset_bytes();
        let required = 2 * s;
        if data.len() < required {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "symbol table message too short: need {required} bytes, have {}",
                    data.len()
                ),
            });
        }
        let btree_address = ctx.read_offset(data);
        let local_heap_address = ctx.read_offset(&data[s..]);
        Ok(Self {
            btree_address,
            local_heap_address,
        })
    }
}

// ---------------------------------------------------------------------------
// Cache Type
// ---------------------------------------------------------------------------

/// Cache type discriminant stored in a symbol table entry's scratch-pad.
///
/// Determines how the 16-byte scratch-pad region is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolCacheType {
    /// Cache type 0: no cached metadata. The scratch-pad is unused.
    None,

    /// Cache type 1: the target object is a group. The scratch-pad stores
    /// the child group's B-tree and heap addresses (each `offset_size`
    /// bytes, packed into the first `2*S` bytes).
    Group {
        /// File address of the child group's B-tree v1 root node.
        btree_address: u64,
        /// File address of the child group's local heap.
        heap_address: u64,
    },

    /// Cache type 2: the entry is a symbolic (soft) link. The scratch-pad
    /// stores a 4-byte little-endian offset into the local heap where the
    /// null-terminated link target string resides.
    SymbolicLink {
        /// Byte offset into the local heap data segment for the link
        /// target path string.
        link_target_offset: u32,
    },
}

// ---------------------------------------------------------------------------
// Symbol Table Entry
// ---------------------------------------------------------------------------

/// Parsed symbol table entry.
///
/// Each entry associates a link name (stored as an offset into the local
/// heap data segment) with a target object header address and optional
/// cached metadata.
///
/// ## Layout
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | S    | Link name offset into local heap |
/// | S      | S    | Object header address |
/// | 2S     | 4    | Cache type |
/// | 2S+4   | 4    | Reserved |
/// | 2S+8   | 16   | Scratch-pad space |
///
/// Total: `2*S + 24` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolTableEntry {
    /// Byte offset of the null-terminated link name within the local heap
    /// data segment.
    pub name_offset: u64,
    /// File address of the linked object's object header.
    pub object_header_address: u64,
    /// Cached metadata from the scratch-pad region.
    pub cache: SymbolCacheType,
}

impl SymbolTableEntry {
    /// Size in bytes of one symbol table entry for the given context.
    ///
    /// Formula: `2 * offset_size + 4 (cache type) + 4 (reserved) + 16 (scratch-pad) = 2S + 24`.
    pub const fn entry_size(ctx: &ParseContext) -> usize {
        2 * ctx.offset_bytes() + 4 + 4 + SCRATCH_PAD_SIZE
    }

    /// Parse a single symbol table entry from a buffer.
    ///
    /// # Arguments
    ///
    /// * `data` — Buffer starting at the first byte of the entry. Must
    ///   contain at least [`entry_size`](Self::entry_size) bytes.
    /// * `ctx`  — Parsing context carrying `offset_size`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidFormat` if:
    /// - `data` is shorter than the required entry size.
    ///
    /// # Cache Type Handling
    ///
    /// Unknown cache type values (≥ 3) are treated as [`SymbolCacheType::None`]
    /// per forward-compatibility guidance in the HDF5 specification. The
    /// scratch-pad bytes are ignored for unknown types.
    pub fn parse(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        let s = ctx.offset_bytes();
        let min_size = Self::entry_size(ctx);
        if data.len() < min_size {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "symbol table entry too short: need {min_size} bytes, have {}",
                    data.len()
                ),
            });
        }

        let name_offset = ctx.read_offset(data);
        let object_header_address = ctx.read_offset(&data[s..]);

        let cache_type_raw = u32::from_le_bytes([
            data[2 * s],
            data[2 * s + 1],
            data[2 * s + 2],
            data[2 * s + 3],
        ]);

        // Scratch-pad starts at offset 2S + 8 and spans 16 bytes.
        let scratch_start = 2 * s + 8;
        let scratch = &data[scratch_start..scratch_start + SCRATCH_PAD_SIZE];

        let cache = match cache_type_raw {
            0 => SymbolCacheType::None,
            1 => {
                // Cache type 1 (group): two offset_size-byte addresses
                // packed at the start of the 16-byte scratch region.
                let btree_address = ctx.read_offset(scratch);
                let heap_address = ctx.read_offset(&scratch[s..]);
                SymbolCacheType::Group {
                    btree_address,
                    heap_address,
                }
            }
            2 => {
                // Cache type 2 (symbolic link): 4-byte LE offset into
                // the local heap for the link target string.
                let link_target_offset =
                    u32::from_le_bytes([scratch[0], scratch[1], scratch[2], scratch[3]]);
                SymbolCacheType::SymbolicLink { link_target_offset }
            }
            _ => {
                // Forward-compatible: treat unrecognised cache types as
                // having no cached metadata.
                SymbolCacheType::None
            }
        };

        Ok(Self {
            name_offset,
            object_header_address,
            cache,
        })
    }
}

// ---------------------------------------------------------------------------
// Symbol Table Node
// ---------------------------------------------------------------------------

/// Parsed symbol table node ("SNOD").
///
/// A symbol table node is a leaf-level structure referenced by a B-tree v1
/// (group type). Each node contains a fixed-capacity array of
/// [`SymbolTableEntry`] values. Deleted entries are identified by an
/// [`UNDEFINED_ADDRESS`](crate::constants::UNDEFINED_ADDRESS) in the
/// `object_header_address` field and are excluded from the parsed result.
///
/// ## Layout
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 4    | Signature `"SNOD"` |
/// | 4      | 1    | Version (1) |
/// | 5      | 1    | Reserved |
/// | 6      | 2    | Number of symbols |
/// | 8      | var  | Symbol table entries |
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct SymbolTableNode {
    /// Parsed symbol table entries with valid (non-undefined) object
    /// header addresses. Deleted entries are excluded.
    pub entries: Vec<SymbolTableEntry>,
}

#[cfg(feature = "alloc")]
impl SymbolTableNode {
    /// Parse a symbol table node by reading from `source` at `address`.
    ///
    /// # Arguments
    ///
    /// * `source`  — Random-access byte source implementing [`ReadAt`].
    /// * `address` — File offset of the first byte of the SNOD signature.
    /// * `ctx`     — Parsing context carrying `offset_size`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidFormat` if:
    /// - The 4-byte signature does not match `"SNOD"`.
    /// - The version byte is not 1.
    /// - The underlying read fails (propagated from `source.read_at`).
    /// - Any entry's buffer is too short (propagated from
    ///   [`SymbolTableEntry::parse`]).
    ///
    /// # Deleted Entry Filtering
    ///
    /// Entries whose `object_header_address` equals
    /// [`UNDEFINED_ADDRESS`](crate::constants::UNDEFINED_ADDRESS) are
    /// interpreted as deleted (freed) slots and are excluded from the
    /// returned `entries` vector.
    pub fn parse<R: ReadAt>(source: &R, address: u64, ctx: &ParseContext) -> Result<Self> {
        // -- Read fixed header (8 bytes) ---------------------------------------
        let mut header = [0u8; SNOD_HEADER_SIZE];
        source.read_at(address, &mut header)?;

        // -- Validate signature ------------------------------------------------
        if header[0..4] != SNOD_SIGNATURE {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "invalid symbol table node signature at address {:#x}: \
                     expected {:?}, found {:?}",
                    address,
                    SNOD_SIGNATURE,
                    &header[0..4]
                ),
            });
        }

        // -- Validate version --------------------------------------------------
        let version = header[4];
        if version != SNOD_VERSION {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "unsupported symbol table node version {version} at address {:#x}, \
                     expected {SNOD_VERSION}",
                    address
                ),
            });
        }

        // header[5] is reserved; ignored.

        // -- Number of symbols -------------------------------------------------
        let num_symbols = u16::from_le_bytes([header[6], header[7]]) as usize;

        // -- Read entry data ---------------------------------------------------
        let entry_size = SymbolTableEntry::entry_size(ctx);
        let data_size = num_symbols * entry_size;
        let mut data = vec![0u8; data_size];
        source.read_at(address + SNOD_HEADER_SIZE as u64, &mut data)?;

        // -- Parse entries, skipping deleted slots ------------------------------
        let mut entries = Vec::with_capacity(num_symbols);
        for i in 0..num_symbols {
            let offset = i * entry_size;
            let entry = SymbolTableEntry::parse(&data[offset..], ctx)?;
            if entry.object_header_address != UNDEFINED_ADDRESS {
                entries.push(entry);
            }
        }

        Ok(Self { entries })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a `ParseContext` with 8-byte offsets (common case).
    fn ctx8() -> ParseContext {
        ParseContext::new(8, 8)
    }

    /// Construct a `ParseContext` with 4-byte offsets.
    fn ctx4() -> ParseContext {
        ParseContext::new(4, 4)
    }

    // -- SymbolTableMessage ---------------------------------------------------

    #[test]
    fn parse_symbol_table_message_8byte_offsets() {
        let ctx = ctx8();
        let mut data = [0u8; 16];
        // B-tree address = 0x0000_0000_0000_1000
        data[0..8].copy_from_slice(&0x1000u64.to_le_bytes());
        // Local heap address = 0x0000_0000_0000_2000
        data[8..16].copy_from_slice(&0x2000u64.to_le_bytes());

        let msg = SymbolTableMessage::parse(&data, &ctx).expect("parse must succeed");
        assert_eq!(msg.btree_address, 0x1000);
        assert_eq!(msg.local_heap_address, 0x2000);
    }

    #[test]
    fn parse_symbol_table_message_4byte_offsets() {
        let ctx = ctx4();
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&0x0800u32.to_le_bytes());
        data[4..8].copy_from_slice(&0x0C00u32.to_le_bytes());

        let msg = SymbolTableMessage::parse(&data, &ctx).expect("parse must succeed");
        assert_eq!(msg.btree_address, 0x0800);
        assert_eq!(msg.local_heap_address, 0x0C00);
    }

    #[test]
    fn parse_symbol_table_message_too_short() {
        let ctx = ctx8();
        let data = [0u8; 15]; // need 16
        assert!(SymbolTableMessage::parse(&data, &ctx).is_err());
    }

    // -- SymbolTableEntry -----------------------------------------------------

    #[test]
    fn entry_size_matches_spec() {
        assert_eq!(SymbolTableEntry::entry_size(&ctx8()), 2 * 8 + 24);
        assert_eq!(SymbolTableEntry::entry_size(&ctx4()), 2 * 4 + 24);
    }

    /// Build a raw entry buffer for 8-byte offsets.
    fn build_entry_8(
        name_offset: u64,
        obj_addr: u64,
        cache_type: u32,
        scratch: &[u8; 16],
    ) -> [u8; 40] {
        let mut buf = [0u8; 40]; // 2*8 + 24
        buf[0..8].copy_from_slice(&name_offset.to_le_bytes());
        buf[8..16].copy_from_slice(&obj_addr.to_le_bytes());
        buf[16..20].copy_from_slice(&cache_type.to_le_bytes());
        // buf[20..24] reserved = 0
        buf[24..40].copy_from_slice(scratch);
        buf
    }

    #[test]
    fn parse_entry_cache_type_none() {
        let ctx = ctx8();
        let scratch = [0u8; 16];
        let buf = build_entry_8(42, 0x3000, 0, &scratch);

        let entry = SymbolTableEntry::parse(&buf, &ctx).expect("parse must succeed");
        assert_eq!(entry.name_offset, 42);
        assert_eq!(entry.object_header_address, 0x3000);
        assert_eq!(entry.cache, SymbolCacheType::None);
    }

    #[test]
    fn parse_entry_cache_type_group() {
        let ctx = ctx8();
        let mut scratch = [0u8; 16];
        scratch[0..8].copy_from_slice(&0xAAAAu64.to_le_bytes());
        scratch[8..16].copy_from_slice(&0xBBBBu64.to_le_bytes());
        let buf = build_entry_8(10, 0x4000, 1, &scratch);

        let entry = SymbolTableEntry::parse(&buf, &ctx).expect("parse must succeed");
        assert_eq!(
            entry.cache,
            SymbolCacheType::Group {
                btree_address: 0xAAAA,
                heap_address: 0xBBBB,
            }
        );
    }

    #[test]
    fn parse_entry_cache_type_group_4byte() {
        let ctx = ctx4();
        let mut scratch = [0u8; 16];
        scratch[0..4].copy_from_slice(&0x1234u32.to_le_bytes());
        scratch[4..8].copy_from_slice(&0x5678u32.to_le_bytes());

        // Entry size for 4-byte = 2*4 + 24 = 32
        let mut buf = [0u8; 32];
        buf[0..4].copy_from_slice(&5u32.to_le_bytes()); // name_offset
        buf[4..8].copy_from_slice(&0x9000u32.to_le_bytes()); // obj addr
        buf[8..12].copy_from_slice(&1u32.to_le_bytes()); // cache type = group
        // buf[12..16] reserved
        buf[16..32].copy_from_slice(&scratch);

        let entry = SymbolTableEntry::parse(&buf, &ctx).expect("parse must succeed");
        assert_eq!(entry.name_offset, 5);
        assert_eq!(entry.object_header_address, 0x9000);
        assert_eq!(
            entry.cache,
            SymbolCacheType::Group {
                btree_address: 0x1234,
                heap_address: 0x5678,
            }
        );
    }

    #[test]
    fn parse_entry_cache_type_symbolic_link() {
        let ctx = ctx8();
        let mut scratch = [0u8; 16];
        scratch[0..4].copy_from_slice(&99u32.to_le_bytes());
        let buf = build_entry_8(7, 0x5000, 2, &scratch);

        let entry = SymbolTableEntry::parse(&buf, &ctx).expect("parse must succeed");
        assert_eq!(
            entry.cache,
            SymbolCacheType::SymbolicLink {
                link_target_offset: 99,
            }
        );
    }

    #[test]
    fn parse_entry_unknown_cache_type_treated_as_none() {
        let ctx = ctx8();
        let scratch = [0u8; 16];
        let buf = build_entry_8(0, 0x6000, 99, &scratch);

        let entry = SymbolTableEntry::parse(&buf, &ctx).expect("parse must succeed");
        assert_eq!(entry.cache, SymbolCacheType::None);
    }

    #[test]
    fn parse_entry_too_short() {
        let ctx = ctx8();
        let buf = [0u8; 39]; // need 40
        assert!(SymbolTableEntry::parse(&buf, &ctx).is_err());
    }

    // -- SymbolTableNode (requires alloc + ReadAt mock) -----------------------

    #[cfg(feature = "alloc")]
    mod node_tests {
        use super::*;

        /// Minimal in-memory `ReadAt` implementation for testing.
        struct MemReader {
            data: Vec<u8>,
        }

        impl MemReader {
            fn new(data: Vec<u8>) -> Self {
                Self { data }
            }
        }

        impl ReadAt for MemReader {
            fn read_at(&self, pos: u64, buf: &mut [u8]) -> Result<()> {
                let start = pos as usize;
                let end = start + buf.len();
                if end > self.data.len() {
                    return Err(Error::BufferTooSmall {
                        required: end,
                        provided: self.data.len(),
                    });
                }
                buf.copy_from_slice(&self.data[start..end]);
                Ok(())
            }
        }

        /// Build a complete SNOD blob at offset 0 with the given entries.
        fn build_snod(ctx: &ParseContext, entries: &[SymbolTableEntry]) -> Vec<u8> {
            let entry_size = SymbolTableEntry::entry_size(ctx);
            let total = SNOD_HEADER_SIZE + entries.len() * entry_size;
            let mut data = vec![0u8; total];

            // Signature
            data[0..4].copy_from_slice(&SNOD_SIGNATURE);
            // Version
            data[4] = SNOD_VERSION;
            // Reserved
            data[5] = 0;
            // Number of symbols
            let count = entries.len() as u16;
            data[6..8].copy_from_slice(&count.to_le_bytes());

            let s = ctx.offset_bytes();
            for (i, entry) in entries.iter().enumerate() {
                let base = SNOD_HEADER_SIZE + i * entry_size;
                // name_offset
                match s {
                    4 => data[base..base + 4]
                        .copy_from_slice(&(entry.name_offset as u32).to_le_bytes()),
                    8 => data[base..base + 8].copy_from_slice(&entry.name_offset.to_le_bytes()),
                    _ => {}
                }
                // object_header_address
                match s {
                    4 => data[base + s..base + 2 * s]
                        .copy_from_slice(&(entry.object_header_address as u32).to_le_bytes()),
                    8 => data[base + s..base + 2 * s]
                        .copy_from_slice(&entry.object_header_address.to_le_bytes()),
                    _ => {}
                }
                // cache type
                let ct: u32 = match entry.cache {
                    SymbolCacheType::None => 0,
                    SymbolCacheType::Group { .. } => 1,
                    SymbolCacheType::SymbolicLink { .. } => 2,
                };
                data[base + 2 * s..base + 2 * s + 4].copy_from_slice(&ct.to_le_bytes());
                // reserved 4 bytes already zero
                // scratch-pad
                let scratch_base = base + 2 * s + 8;
                match entry.cache {
                    SymbolCacheType::None => {}
                    SymbolCacheType::Group {
                        btree_address,
                        heap_address,
                    } => match s {
                        4 => {
                            data[scratch_base..scratch_base + 4]
                                .copy_from_slice(&(btree_address as u32).to_le_bytes());
                            data[scratch_base + 4..scratch_base + 8]
                                .copy_from_slice(&(heap_address as u32).to_le_bytes());
                        }
                        8 => {
                            data[scratch_base..scratch_base + 8]
                                .copy_from_slice(&btree_address.to_le_bytes());
                            data[scratch_base + 8..scratch_base + 16]
                                .copy_from_slice(&heap_address.to_le_bytes());
                        }
                        _ => {}
                    },
                    SymbolCacheType::SymbolicLink { link_target_offset } => {
                        data[scratch_base..scratch_base + 4]
                            .copy_from_slice(&link_target_offset.to_le_bytes());
                    }
                }
            }

            data
        }

        #[test]
        fn parse_node_two_entries() {
            let ctx = ctx8();
            let entries = [
                SymbolTableEntry {
                    name_offset: 0,
                    object_header_address: 0x1000,
                    cache: SymbolCacheType::None,
                },
                SymbolTableEntry {
                    name_offset: 8,
                    object_header_address: 0x2000,
                    cache: SymbolCacheType::Group {
                        btree_address: 0x3000,
                        heap_address: 0x4000,
                    },
                },
            ];
            let blob = build_snod(&ctx, &entries);
            let reader = MemReader::new(blob);

            let node = SymbolTableNode::parse(&reader, 0, &ctx).expect("node parse must succeed");
            assert_eq!(node.entries.len(), 2);
            assert_eq!(node.entries[0].object_header_address, 0x1000);
            assert_eq!(node.entries[0].cache, SymbolCacheType::None);
            assert_eq!(node.entries[1].object_header_address, 0x2000);
            assert_eq!(
                node.entries[1].cache,
                SymbolCacheType::Group {
                    btree_address: 0x3000,
                    heap_address: 0x4000,
                }
            );
        }

        #[test]
        fn parse_node_filters_deleted_entries() {
            let ctx = ctx8();
            let entries = [
                SymbolTableEntry {
                    name_offset: 0,
                    object_header_address: 0x1000,
                    cache: SymbolCacheType::None,
                },
                SymbolTableEntry {
                    name_offset: 16,
                    object_header_address: UNDEFINED_ADDRESS, // deleted
                    cache: SymbolCacheType::None,
                },
                SymbolTableEntry {
                    name_offset: 32,
                    object_header_address: 0x5000,
                    cache: SymbolCacheType::SymbolicLink {
                        link_target_offset: 48,
                    },
                },
            ];
            let blob = build_snod(&ctx, &entries);
            let reader = MemReader::new(blob);

            let node = SymbolTableNode::parse(&reader, 0, &ctx).expect("node parse must succeed");
            assert_eq!(node.entries.len(), 2, "deleted entry must be filtered out");
            assert_eq!(node.entries[0].object_header_address, 0x1000);
            assert_eq!(node.entries[1].object_header_address, 0x5000);
            assert_eq!(
                node.entries[1].cache,
                SymbolCacheType::SymbolicLink {
                    link_target_offset: 48,
                }
            );
        }

        #[test]
        fn parse_node_empty() {
            let ctx = ctx8();
            let blob = build_snod(&ctx, &[]);
            let reader = MemReader::new(blob);

            let node = SymbolTableNode::parse(&reader, 0, &ctx).expect("empty node must parse");
            assert!(node.entries.is_empty());
        }

        #[test]
        fn parse_node_bad_signature() {
            let mut blob = vec![0u8; SNOD_HEADER_SIZE];
            blob[0..4].copy_from_slice(b"BAAD");
            blob[4] = SNOD_VERSION;
            let reader = MemReader::new(blob);
            assert!(SymbolTableNode::parse(&reader, 0, &ctx8()).is_err());
        }

        #[test]
        fn parse_node_bad_version() {
            let mut blob = vec![0u8; SNOD_HEADER_SIZE];
            blob[0..4].copy_from_slice(&SNOD_SIGNATURE);
            blob[4] = 99; // bad version
            let reader = MemReader::new(blob);
            assert!(SymbolTableNode::parse(&reader, 0, &ctx8()).is_err());
        }

        #[test]
        fn parse_node_at_nonzero_address() {
            let ctx = ctx4();
            let entries = [SymbolTableEntry {
                name_offset: 0,
                object_header_address: 0xABCD,
                cache: SymbolCacheType::None,
            }];
            let snod_blob = build_snod(&ctx, &entries);

            // Place the SNOD at address 256 within a larger buffer.
            let base_addr: u64 = 256;
            let total = base_addr as usize + snod_blob.len();
            let mut data = vec![0xFFu8; total];
            data[base_addr as usize..].copy_from_slice(&snod_blob);
            let reader = MemReader::new(data);

            let node = SymbolTableNode::parse(&reader, base_addr, &ctx)
                .expect("parse at offset must succeed");
            assert_eq!(node.entries.len(), 1);
            assert_eq!(node.entries[0].object_header_address, 0xABCD);
        }
    }
}
