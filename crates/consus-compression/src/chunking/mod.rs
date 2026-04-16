//! Chunk indexing and iteration for N-dimensional chunked storage.
//!
//! This module is the SSOT for all chunk coordinate computation in Consus.
//! HDF5, Zarr, and other chunked formats use these utilities.
//!
//! ## Hierarchy
//!
//! ```text
//! chunking/
//! ├── index     # Chunk coordinate ↔ linear index conversion
//! └── iterator  # Iterate chunk ranges over a dataspace
//! ```
//!
//! ## Definitions
//!
//! - **Dataset shape**: The dimensions of the full N-dimensional array `[D₀, D₁, ..., Dₙ₋₁]`.
//! - **Chunk shape**: The dimensions of each chunk `[C₀, C₁, ..., Cₙ₋₁]`.
//! - **Chunk grid**: The number of chunks along each axis `[⌈D₀/C₀⌉, ..., ⌈Dₙ₋₁/Cₙ₋₁⌉]`.
//! - **Chunk coordinate**: An N-dimensional index into the chunk grid.
//! - **Linear index**: A scalar index into a flattened (row-major) chunk grid.

pub mod index;
pub mod iterator;

pub use index::{
    chunk_coord_to_linear, chunk_element_range, chunk_grid_shape_fixed, linear_to_chunk_coord,
    total_chunks,
};

#[cfg(feature = "alloc")]
pub use index::chunk_grid_shape;

#[cfg(feature = "alloc")]
pub use iterator::ChunkIterator;
