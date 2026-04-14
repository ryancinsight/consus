//! Local and global heap implementations.
//!
//! ## Specification
//!
//! Heaps store variable-length data (names, VL strings, etc.).
//!
//! ### Local Heap (signature "HEAP")
//!
//! Used for group member names. Contains a data segment and a free list.
//!
//! ### Global Heap (signature "GCOL")
//!
//! Stores collections of variable-length objects. Each collection has
//! a header followed by heap objects indexed by ID.

/// Local heap signature.
pub const LOCAL_HEAP_SIGNATURE: [u8; 4] = *b"HEAP";

/// Global heap collection signature.
pub const GLOBAL_HEAP_SIGNATURE: [u8; 4] = *b"GCOL";

/// Parsed local heap header.
#[derive(Debug, Clone)]
pub struct LocalHeap {
    /// Version (0).
    pub version: u8,
    /// Total data segment size.
    pub data_segment_size: u64,
    /// Offset of the free list head within the data segment.
    pub free_list_offset: u64,
    /// Address of the data segment.
    pub data_address: u64,
}

/// A single object within a global heap collection.
#[derive(Debug, Clone)]
pub struct GlobalHeapObject {
    /// Object index (1-based; 0 is the free-space marker).
    pub index: u16,
    /// Reference count.
    pub reference_count: u32,
    /// Object data.
    #[cfg(feature = "alloc")]
    pub data: alloc::vec::Vec<u8>,
}
