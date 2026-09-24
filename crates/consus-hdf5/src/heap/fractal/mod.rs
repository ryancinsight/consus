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
//! | …      | 4 (if FRHP flags bit 1 set)    | Checksum (before data)    |
//! | …      | variable                       | Object data               |
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

mod header;
mod huge;
mod ids;
mod read;
#[cfg(test)]
mod tests;

pub use header::{DIRECT_BLOCK_SIGNATURE, FRACTAL_HEAP_SIGNATURE, FractalHeapHeader};
pub use huge::read_huge_object;
pub(crate) use huge::{read_bounded_bytes, read_uint_le};
#[cfg(feature = "alloc")]
pub use ids::FractalHeapId;
pub use ids::decode_heap_id;
pub use read::read_managed_object;
