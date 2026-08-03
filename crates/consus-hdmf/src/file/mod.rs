//! HDF5-backed reader and writer for HDMF DynamicTable objects.
//!
//! ## Reader — [`HdmfFile`]
//!
//! Opens an HDF5 byte slice, validates that the root group carries
//! `data_type = "DynamicTable"`, and exposes [`HdmfFile::read_table`] to
//! extract the full [`DynamicTable`] with all columns.
//!
//! ## Writer — [`HdmfFileBuilder`]
//!
//! Accumulates columns via the builder API, then serialises a conformant
//! HDMF HDF5 image with the required `VectorData`, `ElementIdentifiers`, and
//! optional `VectorIndex` datasets and all mandatory root-group attributes.
//!
//! ## HDF5 file layout written by [`HdmfFileBuilder`]
//!
//! ```text
//! / (root group)
//!   attrs: data_type="DynamicTable", namespace="hdmf-common",
//!          description=..., colnames=[...], object_id=UUID
//!   id          — int64 1-D  [ElementIdentifiers]
//!   <col>       — typed 1-D  [VectorData]
//!   <col>_index — uint64 1-D [VectorIndex]  (ragged columns only)
//! ```

mod builder;
mod reader;

pub use builder::HdmfFileBuilder;
pub use reader::HdmfFile;

const HDMF_COMMON_NS: &str = "hdmf-common";
const TYPE_DYNAMIC_TABLE: &str = "DynamicTable";
const TYPE_VECTOR_DATA: &str = "VectorData";
const TYPE_VECTOR_INDEX: &str = "VectorIndex";
const TYPE_ELEMENT_IDENTIFIERS: &str = "ElementIdentifiers";
