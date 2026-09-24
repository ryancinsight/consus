//! Chunk key derivation, coordinate checks, stride math, and stored shape helpers.

#[cfg(feature = "alloc")]
use crate::chunk::error::ChunkError;
#[cfg(feature = "alloc")]
use crate::metadata::ArrayMetadata;
#[cfg(feature = "alloc")]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

/// Internal helper: generates the chunk key for a given array and coordinates.
///
/// This computes the hierarchical chunk key based on the array key and
/// chunk grid coordinates using the specified key encoding.
#[cfg(feature = "alloc")]
pub(crate) fn chunk_key_for_array(
    array_key: &str,
    coords: &[u64],
    chunk_key_encoding: &crate::metadata::ChunkKeyEncoding,
) -> String {
    let coord_parts: Vec<String> = coords.iter().map(|c| c.to_string()).collect();

    if chunk_key_encoding.name == "v2" || chunk_key_encoding.separator == '.' {
        if array_key.is_empty() || array_key == "." {
            coord_parts.join(".")
        } else {
            format!("{}/{}", array_key, coord_parts.join("."))
        }
    } else {
        let chunk_suffix = if coord_parts.is_empty() {
            String::from("c")
        } else {
            format!("c/{}", coord_parts.join("/"))
        };

        if array_key.is_empty() || array_key == "." {
            chunk_suffix
        } else {
            format!("{}/{}", array_key, chunk_suffix)
        }
    }
}

#[cfg(feature = "alloc")]
pub(crate) fn validate_chunk_coords(
    coords: &[u64],
    meta: &ArrayMetadata,
) -> Result<(), ChunkError> {
    if coords.len() != meta.shape.len() || meta.chunks.len() != meta.shape.len() {
        return Err(ChunkError::ChunkOutOfBounds);
    }

    for ((&coord, &shape_dim), &chunk_dim) in
        coords.iter().zip(meta.shape.iter()).zip(meta.chunks.iter())
    {
        if chunk_dim == 0 {
            return Err(ChunkError::ChunkOutOfBounds);
        }

        let chunk_grid_extent = u64::try_from(shape_dim.div_ceil(chunk_dim))
            .map_err(|_| ChunkError::ChunkOutOfBounds)?;
        if coord >= chunk_grid_extent {
            return Err(ChunkError::ChunkOutOfBounds);
        }
    }

    Ok(())
}

/// Internal helper: computes row-major strides for a shape.
pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![1usize; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() {
        strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    strides
}

/// Returns the shape to use for stride computation when indexing into `chunk_data`.
///
/// Zarr-python writes boundary chunks as full padded chunks (padded with fill value
/// to the full chunk shape). When `chunk_data` has
/// `product(full_chunk) * element_size` bytes but the valid region is
/// only `product(chunk_extent) * element_size` bytes, strides must use
/// `full_chunk`.
pub(crate) fn stored_shape_for_chunk<'a>(
    chunk_data_len: usize,
    element_size: usize,
    chunk_extent: &'a [usize],
    full_chunk: &'a [usize],
) -> &'a [usize] {
    let extent_elements: usize = if chunk_extent.is_empty() {
        1
    } else {
        chunk_extent.iter().product()
    };
    let full_elements: usize = if full_chunk.is_empty() {
        1
    } else {
        full_chunk.iter().product()
    };
    if full_elements != extent_elements
        && chunk_data_len == full_elements.saturating_mul(element_size)
    {
        full_chunk
    } else {
        chunk_extent
    }
}
