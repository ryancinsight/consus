//!
//! Chunk I/O operations for Zarr arrays.
//!
//! This module provides functions for reading and writing Zarr chunks,
//! including compression/decompression support and selection-based access.

mod error;
pub mod key_encoding;
mod ops;
mod selection;

pub use error::ChunkError;
pub use key_encoding::{ChunkKeySeparator, chunk_key};
pub use ops::{
    expand_fill_value, read_array, read_chunk, write_array, write_array_selection, write_chunk,
};
pub use selection::{Selection, SelectionStep};
