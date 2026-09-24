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
//! Object headers are written in v2 format with Jenkins lookup3 checksums.
//! Messages are packed sequentially within a single header chunk.
//! Continuation chunks are not emitted by the writer; all messages
//! for an object must fit in the initial allocation.
//!
//! ## Dependencies (DIP)
//!
//! - `consus_core`: Error types, canonical data model types.
//! - `consus_io`: `WriteAt` trait for positioned byte output.
//! - `consus_compression`: `Lookup3` (Jenkins) checksum for v2 metadata integrity.

#[cfg(feature = "alloc")]
mod attributes;
mod builder;
mod builder_specs;
mod builder_subgroup;
mod chunk_data;
mod chunk_index;
mod dataset;
mod dataspace;
mod datatype;
mod filters;
mod group;
mod layout;
mod links;
mod state;
#[cfg(test)]
mod tests;

#[cfg(feature = "alloc")]
pub(crate) use attributes::write_group_node;
#[cfg(feature = "alloc")]
pub use attributes::{encode_attribute, write_object_header_v2};
#[cfg(feature = "alloc")]
pub use builder::Hdf5FileBuilder;
#[cfg(feature = "alloc")]
pub use builder_specs::{ChildDatasetSpec, ChildGroupSpec};
#[cfg(feature = "alloc")]
pub use builder_subgroup::SubGroupBuilder;
#[cfg(feature = "alloc")]
pub(crate) use chunk_data::{write_chunked_data, write_chunked_data_with_element_size};
#[cfg(feature = "alloc")]
pub(crate) use chunk_index::{
    ChunkIndexEntry, write_chunk_btree_v1, write_chunk_btree_v2, write_chunk_farray,
};
#[cfg(feature = "alloc")]
pub use dataset::{write_contiguous_data, write_dataset_header};
#[cfg(feature = "alloc")]
pub use dataspace::encode_dataspace;
#[cfg(feature = "alloc")]
pub use datatype::encode_datatype;
#[cfg(feature = "alloc")]
pub(crate) use filters::{dataset_filter_ids, encode_filter_pipeline};
#[cfg(feature = "alloc")]
pub(crate) use group::{encode_group_info, encode_link_info};
#[cfg(feature = "alloc")]
pub use group::{write_group_header, write_local_heap, write_v1_group_header};
#[cfg(feature = "alloc")]
pub use layout::{encode_layout, encode_layout_with_chunk_index};
#[cfg(feature = "alloc")]
pub use links::{encode_hard_link, encode_soft_link};
#[cfg(feature = "alloc")]
pub use state::{WriteState, update_superblock_eof, write_superblock};
#[cfg(feature = "alloc")]
pub(crate) use state::{chunk_size_encoding, write_chunk_size, write_offset, write_v2_message};

// Re-export property list types so consumers can import them via this module.
#[cfg(feature = "alloc")]
pub use crate::property_list::{DatasetCreationProps, FileCreationProps};
