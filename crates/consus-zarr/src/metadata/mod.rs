//! Zarr array and group metadata.
//!
//! This module defines the canonical in-memory representation of Zarr metadata
//! for both v2 and v3, along with JSON serialization/deserialization.
//!
//! ## Module Hierarchy
//!
//! ```text
//! metadata/
//! ├── mod.rs          # Manifest + re-exports
//! ├── array.rs        # Array metadata, fill values, chunk-key encoding
//! ├── group.rs        # Group metadata and attribute values
//! ├── codec.rs        # Canonical codec representation
//! ├── dtype.rs        # dtype_to_element_size utility
//! ├── v2.rs           # .zarray, .zgroup, .zattrs parse + serialize
//! ├── v3.rs           # zarr.json parse + serialize
//! └── consolidated.rs # .zmetadata consolidated format
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
mod array;
#[cfg(feature = "alloc")]
mod codec;
#[cfg(feature = "alloc")]
mod consolidated;
#[cfg(feature = "alloc")]
mod dtype;
#[cfg(feature = "alloc")]
mod group;
#[cfg(feature = "alloc")]
mod v2;
#[cfg(feature = "alloc")]
mod v3;

#[cfg(feature = "alloc")]
pub use array::{ArrayMetadata, ChunkKeyEncoding, FillValue, ZarrVersion};
#[cfg(feature = "alloc")]
pub use codec::Codec;
#[cfg(feature = "alloc")]
pub use consolidated::ConsolidatedMetadata;
#[cfg(feature = "alloc")]
pub use consolidated::{
    ConsolidatedMetadataV2, ConsolidatedMetadataV3, MetadataEntryV2, MetadataEntryV3,
    ParseError as ConsolidatedParseError, SerializeError as ConsolidatedSerializeError,
};
#[cfg(feature = "alloc")]
pub use dtype::dtype_to_element_size;
#[cfg(feature = "alloc")]
pub use group::{AttributeValue, GroupMetadata};
#[cfg(feature = "alloc")]
pub use v2::{
    ArrayMetadataV2, CompressorConfig, FilterId, GroupMetadataV2, parse_zattrs, serialize_zattrs,
};
#[cfg(feature = "alloc")]
pub use v3::ZarrJson;
#[cfg(feature = "std")]
pub use v3::{WriteZarrJsonError, write_group_json, write_zarr_json};
