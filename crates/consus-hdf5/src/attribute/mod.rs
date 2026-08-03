//! HDF5 attribute message parsing (header message type 0x000C).
//!
//! ## Specification (HDF5 File Format Specification, Section IV.A.2.m)
//!
//! Attributes are small named datasets attached to any HDF5 object.
//! Each attribute is stored as a header message containing the name,
//! datatype, dataspace, and raw data inline.
//!
//! ### Version 1 Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 1 | Version (1) |
//! | 1 | 1 | Reserved |
//! | 2 | 2 | Name size (including null terminator) |
//! | 4 | 2 | Datatype size |
//! | 6 | 2 | Dataspace size |
//! | 8 | var | Name (null-terminated, padded to 8-byte boundary) |
//! | var | var | Datatype (padded to 8-byte boundary) |
//! | var | var | Dataspace (padded to 8-byte boundary) |
//! | var | var | Data |
//!
//! ### Version 2 Layout
//!
//! Identical field order to v1 but components are NOT padded to
//! 8-byte boundaries. A flags byte replaces the reserved byte.
//!
//! ### Version 3 Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 1 | Version (3) |
//! | 1 | 1 | Flags (bit 0: shared datatype, bit 1: shared dataspace) |
//! | 2 | 2 | Name size (byte count, NOT null-terminated) |
//! | 4 | 2 | Datatype size |
//! | 6 | 2 | Dataspace size |
//! | 8 | 1 | Character encoding (0=ASCII, 1=UTF-8) |
//! | 9 | var | Name (length from name_size, no null terminator) |
//! | var | var | Datatype |
//! | var | var | Dataspace |
//! | var | var | Data |

pub mod info;

mod decode;
mod types;

#[cfg(feature = "alloc")]
pub use decode::decode_attribute_value;
#[cfg(feature = "alloc")]
pub use types::Hdf5Attribute;
