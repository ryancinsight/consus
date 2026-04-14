//! HDF5 dataset operations.
//!
//! ## Specification
//!
//! A dataset is an N-dimensional array with:
//! - Datatype (header message 0x0003)
//! - Dataspace (header message 0x0001)
//! - Storage layout (header message 0x0008): contiguous, chunked, or compact
//! - Optional filter pipeline (header message 0x000B)
//! - Optional fill value (header message 0x0005)
//!
//! ### Storage Layouts
//!
//! | Layout | Class | Description |
//! |--------|-------|-------------|
//! | Compact | 0 | Data stored in object header |
//! | Contiguous | 1 | Single contiguous block |
//! | Chunked | 2 | B-tree indexed chunks |
//! | Virtual | 3 | Maps to regions of other datasets |

use consus_core::datatype::Datatype;
use consus_core::dimension::{ChunkShape, Shape};

/// Storage layout class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageLayout {
    /// Data stored directly in the object header.
    Compact,
    /// Single contiguous block in the file.
    Contiguous,
    /// B-tree indexed chunks.
    Chunked,
    /// Virtual dataset (mapped regions).
    Virtual,
}

/// Resolved dataset metadata.
///
/// Parsed from object header messages during file traversal.
#[derive(Debug, Clone)]
pub struct Hdf5Dataset {
    /// Absolute path of this dataset.
    #[cfg(feature = "alloc")]
    pub path: alloc::string::String,
    /// Address of the object header.
    pub object_header_address: u64,
    /// Element datatype.
    pub datatype: Datatype,
    /// Shape of the dataset.
    pub shape: Shape,
    /// Storage layout.
    pub layout: StorageLayout,
    /// Chunk shape (only for chunked layout).
    pub chunk_shape: Option<ChunkShape>,
    /// Address of contiguous data (for contiguous layout).
    pub data_address: Option<u64>,
    /// Filter pipeline IDs applied to this dataset.
    #[cfg(feature = "alloc")]
    pub filters: alloc::vec::Vec<u16>,
}
