//! Global heap: stores collections of variable-length objects.
//!
//! ## Specification (HDF5 File Format Specification, Section IV.B)
//!
//! A global heap collection stores a set of variable-length objects
//! (VL strings, VL sequences, etc.) indexed by a 1-based integer ID.
//! Object 0 is the free-space marker and terminates iteration.
//!
//! ### Global Heap Collection Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0      | 4    | Signature "GCOL" |
//! | 4      | 1    | Version (1) |
//! | 5      | 3    | Reserved |
//! | 8      | L    | Collection size (length_size bytes) |
//! | 8+L    | var  | Heap objects |
//!
//! ### Global Heap Object Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0      | 2    | Heap object index (0 = free space / end marker) |
//! | 2      | 2    | Reference count |
//! | 4      | 4    | Reserved |
//! | 8      | L    | Object size (length_size bytes) |
//! | 8+L    | size | Object data (padded to 8-byte boundary) |
//!
//! ## Invariants
//!
//! - Object index 0 marks the end of the used object list.
//! - Object data is padded to an 8-byte boundary on disk; the
//!   `object_size` field gives the unpadded logical size.
//! - Reference counts track how many links point to this object;
//!   a count of 0 means the object slot can be reclaimed.

mod collection;
mod reference;

#[cfg(test)]
mod tests;

pub use collection::GLOBAL_HEAP_SIGNATURE;
#[cfg(feature = "alloc")]
pub use collection::GlobalHeapCollection;
pub use collection::GlobalHeapObject;
#[cfg(feature = "alloc")]
pub use reference::resolve_vl_references;
