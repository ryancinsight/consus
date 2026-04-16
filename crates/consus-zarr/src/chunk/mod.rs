//! Chunk indexing and key generation.
//!
//! ## Specification
//!
//! Zarr chunks are identified by their grid coordinates.
//! For a dataset of shape `S` with chunk shape `C`, chunk `(i, j, k)`
//! covers the region `[i*C[0]..(i+1)*C[0], j*C[1]..(j+1)*C[1], ...]`.

pub mod key_encoding;

pub use key_encoding::{ChunkKeySeparator, chunk_key};
