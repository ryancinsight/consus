//! Single-chunk read/write drivers against the store and codec pipeline.

#[cfg(feature = "alloc")]
use super::{chunk_key_for_array, validate_chunk_coords};
#[cfg(feature = "alloc")]
use crate::chunk::error::ChunkError;
#[cfg(feature = "alloc")]
use crate::codec::{CodecPipeline, default_registry};
#[cfg(feature = "alloc")]
use crate::metadata::{ArrayMetadata, dtype_to_element_size};
#[cfg(feature = "alloc")]
use crate::store::Store;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Reads and decompresses a single chunk from the store.
#[cfg(feature = "alloc")]
pub fn read_chunk<S: Store>(
    store: &S,
    array_key: &str,
    coords: &[u64],
    meta: &ArrayMetadata,
) -> Result<Vec<u8>, ChunkError> {
    validate_chunk_coords(coords, meta)?;

    let key = chunk_key_for_array(array_key, coords, &meta.chunk_key_encoding);

    let data = match store.get(&key) {
        Ok(data) => data,
        Err(consus_core::Error::NotFound { .. }) => return Err(ChunkError::Uninitialized),
        Err(e) => return Err(ChunkError::StoreError(e.to_string())),
    };

    if data.is_empty() {
        return Err(ChunkError::Uninitialized);
    }

    // Apply codec pipeline for decompression
    if !meta.codecs.is_empty() {
        #[cfg(not(feature = "std"))]
        return Err(ChunkError::DecompressFailed);
        #[cfg(feature = "std")]
        return CodecPipeline::new(meta.codecs.clone())
            .with_element_size(dtype_to_element_size(&meta.dtype).unwrap_or(1))
            .decompress(&data, default_registry())
            .map_err(|_| ChunkError::DecompressFailed);
    }
    Ok(data)
}

/// Writes and compresses a single chunk to the store.
#[cfg(feature = "alloc")]
pub fn write_chunk<S: Store>(
    store: &mut S,
    array_key: &str,
    coords: &[u64],
    meta: &ArrayMetadata,
    data: &[u8],
) -> Result<(), ChunkError> {
    validate_chunk_coords(coords, meta)?;

    let key = chunk_key_for_array(array_key, coords, &meta.chunk_key_encoding);

    let encoded_data = if !meta.codecs.is_empty() {
        #[cfg(not(feature = "std"))]
        return Err(ChunkError::CompressFailed);
        #[cfg(feature = "std")]
        CodecPipeline::new(meta.codecs.clone())
            .with_element_size(dtype_to_element_size(&meta.dtype).unwrap_or(1))
            .compress(data, default_registry())
            .map_err(|_| ChunkError::CompressFailed)?
    } else {
        data.to_vec()
    };

    store
        .set(&key, &encoded_data)
        .map_err(|e| ChunkError::StoreError(e.to_string()))
}
