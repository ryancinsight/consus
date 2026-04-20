//!
//! Chunk I/O operations for Zarr arrays.
//!
//! This module provides functions for reading and writing Zarr chunks,
//! including compression/decompression support and selection-based access.

use core::fmt;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

#[cfg(feature = "alloc")]
use crate::codec::{CodecPipeline, default_registry};

#[cfg(feature = "alloc")]
use crate::metadata::{ArrayMetadata, FillValue};

#[cfg(feature = "alloc")]
use crate::store::Store;

pub mod key_encoding;
pub use key_encoding::{ChunkKeySeparator, chunk_key};

// ---------------------------------------------------------------------------
// Selection types
// ---------------------------------------------------------------------------

/// A step in a multi-dimensional selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionStep {
    /// Starting index of the selection.
    pub start: u64,
    /// Number of elements in this step.
    pub count: u64,
    /// Stride between elements (spacing).
    pub stride: u64,
}

impl SelectionStep {
    /// Returns true if this step represents a contiguous range.
    ///
    /// A step is contiguous when stride equals 1.
    pub fn contiguous(&self) -> bool {
        self.stride == 1
    }

    /// Returns the exclusive end index of this step.
    pub fn end(&self) -> u64 {
        self.start + (self.count.saturating_sub(1)) * self.stride + 1
    }

    /// Returns an iterator over the indices covered by this step.
    pub fn indices(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.count).map(move |i| self.start + i * self.stride)
    }
}

/// A multi-dimensional selection for array indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// The selection steps for each dimension.
    pub steps: Vec<SelectionStep>,
}

impl Selection {
    /// Creates a selection that covers the entire array.
    pub fn full(_dims: usize) -> Self {
        Self { steps: Vec::new() }
    }

    /// Creates a selection from explicit steps.
    pub fn from_steps(steps: Vec<SelectionStep>) -> Self {
        Self { steps }
    }

    /// Returns the total number of elements selected.
    pub fn num_elements(&self) -> u64 {
        if self.steps.is_empty() {
            return 0;
        }
        self.steps.iter().map(|s| s.count).product()
    }

    /// Returns true if this selection covers the full array.
    pub fn is_full(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns the selection step for each array dimension.
    ///
    /// For full-array selections, this materializes one contiguous step per
    /// dimension spanning the entire array extent.
    pub fn normalized_steps(&self, shape: &[usize]) -> Result<Vec<SelectionStep>, ChunkError> {
        if self.is_full() {
            return Ok(shape
                .iter()
                .map(|&extent| SelectionStep {
                    start: 0,
                    count: extent as u64,
                    stride: 1,
                })
                .collect());
        }

        if self.steps.len() != shape.len() {
            return Err(ChunkError::UnexpectedLength);
        }

        for (step, &extent) in self.steps.iter().zip(shape.iter()) {
            if step.stride == 0 {
                return Err(ChunkError::UnexpectedLength);
            }
            if step.count == 0 {
                return Err(ChunkError::UnexpectedLength);
            }
            if step.end() > extent as u64 {
                return Err(ChunkError::ChunkOutOfBounds);
            }
        }

        Ok(self.steps.clone())
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during chunk operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkError {
    /// The requested chunk is out of bounds of the array.
    ChunkOutOfBounds,
    /// The chunk has not been initialized (no data written yet).
    Uninitialized,
    /// Decompression of chunk data failed.
    DecompressFailed,
    /// Compression of chunk data failed.
    CompressFailed,
    /// Unexpected length when reading or writing data.
    UnexpectedLength,
    /// Store operation failed.
    StoreError(String),
}

impl fmt::Display for ChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkError::ChunkOutOfBounds => write!(f, "chunk is out of bounds"),
            ChunkError::Uninitialized => write!(f, "chunk is uninitialized"),
            ChunkError::DecompressFailed => write!(f, "failed to decompress chunk"),
            ChunkError::CompressFailed => write!(f, "failed to compress chunk"),
            ChunkError::UnexpectedLength => write!(f, "unexpected data length"),
            ChunkError::StoreError(msg) => write!(f, "store error: {}", msg),
        }
    }
}

impl std::error::Error for ChunkError {}

#[cfg(feature = "alloc")]
impl From<ChunkError> for consus_core::Error {
    fn from(err: ChunkError) -> Self {
        consus_core::Error::InvalidFormat {
            message: err.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Internal helper: generates the chunk key for a given array and coordinates.
///
/// This computes the hierarchical chunk key based on the array key and
/// chunk grid coordinates using the specified key encoding.
#[cfg(feature = "alloc")]
fn chunk_key_for_array(
    array_key: &str,
    coords: &[u64],
    chunk_key_encoding: &crate::metadata::ChunkKeyEncoding,
) -> String {
    if chunk_key_encoding.name == "v2" || chunk_key_encoding.separator == '.' {
        // v2 style: array.c/0/0/0
        let parts: Vec<String> = std::iter::once(array_key.to_string())
            .chain(coords.iter().map(|c| c.to_string()))
            .collect();
        format!("{}.c/{}", array_key, &parts[1..].join("/"))
    } else {
        // v3 default style: array/c/0/0/0
        let parts: Vec<String> = std::iter::once(array_key.to_string())
            .chain(coords.iter().map(|c| c.to_string()))
            .collect();
        format!("{}/c/{}", array_key, &parts[1..].join("/"))
    }
}

/// Internal helper: generates the metadata key for an array.
fn metadata_key(array_key: &str) -> String {
    format!("{}/zarr.json", array_key)
}

#[cfg(feature = "alloc")]
fn validate_chunk_coords(coords: &[u64], meta: &ArrayMetadata) -> Result<(), ChunkError> {
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
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![1usize; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() {
        strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    strides
}

/// Internal helper: computes the realized indices for each selection dimension.
fn selection_indices(steps: &[SelectionStep]) -> Vec<Vec<u64>> {
    steps.iter().map(|step| step.indices().collect()).collect()
}

/// Internal helper: checks if a chunk intersects with a selection.
fn chunk_intersects_selection(
    chunk_origin: &[u64],
    chunk_extent: &[usize],
    selection_steps: &[SelectionStep],
) -> bool {
    for dim in 0..selection_steps.len() {
        let chunk_start = chunk_origin[dim];
        let chunk_end = chunk_start + chunk_extent[dim] as u64;
        let step = &selection_steps[dim];

        let intersects = step
            .indices()
            .any(|index| index >= chunk_start && index < chunk_end);
        if !intersects {
            return false;
        }
    }

    true
}

/// Internal helper: copies selected elements from a chunk into the output buffer.
fn copy_chunk_selection_to_output(
    chunk_data: &[u8],
    chunk_origin: &[u64],
    chunk_extent: &[usize],
    selection_indices: &[Vec<u64>],
    output: &mut [u8],
    element_size: usize,
) -> Result<(), ChunkError> {
    let chunk_strides = compute_strides(chunk_extent);
    let selection_shape: Vec<usize> = selection_indices.iter().map(Vec::len).collect();
    let output_strides = compute_strides(&selection_shape);

    if selection_indices.is_empty() {
        if chunk_data.len() != element_size || output.len() != element_size {
            return Err(ChunkError::UnexpectedLength);
        }
        output.copy_from_slice(chunk_data);
        return Ok(());
    }

    let mut selection_position = vec![0usize; selection_indices.len()];

    loop {
        let mut in_chunk = true;
        let mut chunk_linear = 0usize;
        let mut output_linear = 0usize;

        for dim in 0..selection_indices.len() {
            let absolute_index = selection_indices[dim][selection_position[dim]];
            let chunk_start = chunk_origin[dim];
            let chunk_end = chunk_start + chunk_extent[dim] as u64;
            if absolute_index < chunk_start || absolute_index >= chunk_end {
                in_chunk = false;
                break;
            }

            let local_index = (absolute_index - chunk_start) as usize;
            chunk_linear += local_index * chunk_strides[dim];
            output_linear += selection_position[dim] * output_strides[dim];
        }

        if in_chunk {
            let chunk_byte_start = chunk_linear * element_size;
            let chunk_byte_end = chunk_byte_start + element_size;
            let output_byte_start = output_linear * element_size;
            let output_byte_end = output_byte_start + element_size;

            if chunk_byte_end > chunk_data.len() || output_byte_end > output.len() {
                return Err(ChunkError::UnexpectedLength);
            }

            output[output_byte_start..output_byte_end]
                .copy_from_slice(&chunk_data[chunk_byte_start..chunk_byte_end]);
        }

        let mut advanced = false;
        for dim in (0..selection_position.len()).rev() {
            selection_position[dim] += 1;
            if selection_position[dim] < selection_indices[dim].len() {
                advanced = true;
                break;
            }
            selection_position[dim] = 0;
        }

        if !advanced {
            break;
        }
    }

    Ok(())
}

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
        let codec_pipeline = CodecPipeline::new(meta.codecs.clone());
        let registry = default_registry();
        codec_pipeline
            .decompress(&data, registry)
            .map_err(|_| ChunkError::DecompressFailed)
    } else {
        Ok(data)
    }
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
        let codec_pipeline = CodecPipeline::new(meta.codecs.clone());
        let registry = default_registry();
        codec_pipeline
            .compress(data, registry)
            .map_err(|_| ChunkError::CompressFailed)?
    } else {
        data.to_vec()
    };

    store
        .set(&key, &encoded_data)
        .map_err(|e| ChunkError::StoreError(e.to_string()))
}

/// Expands a fill value to a byte vector of the specified length.
#[cfg(feature = "alloc")]
pub fn expand_fill_value(fill_value: &FillValue, dtype: &str, num_elements: u64) -> Vec<u8> {
    let element_size = crate::metadata::dtype_to_element_size(dtype).unwrap_or(8);
    let total_size = (num_elements as usize) * element_size;

    let fill_bytes: Vec<u8> = match fill_value {
        FillValue::Default => vec![0u8; element_size],
        FillValue::Null => vec![0u8; element_size],
        FillValue::Bool(b) => {
            let mut bytes = vec![0u8; element_size];
            if element_size >= 1 {
                bytes[0] = if *b { 1 } else { 0 };
            }
            bytes
        }
        FillValue::Int(i) => {
            let mut bytes = vec![0u8; element_size];
            let val = *i;
            for (idx, byte) in bytes.iter_mut().enumerate() {
                *byte = (val >> (idx * 8)) as u8;
            }
            bytes
        }
        FillValue::Uint(u) => {
            let mut bytes = vec![0u8; element_size];
            let val = *u;
            for (idx, byte) in bytes.iter_mut().enumerate() {
                *byte = (val >> (idx * 8)) as u8;
            }
            bytes
        }
        FillValue::Float(s) => {
            let val: f64 = s.parse().unwrap_or(f64::NAN);
            let raw = val.to_le_bytes();
            let mut bytes = vec![0u8; element_size];
            let copy_len = core::cmp::min(element_size, raw.len());
            bytes[..copy_len].copy_from_slice(&raw[..copy_len]);
            bytes
        }
        FillValue::String(_) => vec![0u8; element_size],
        FillValue::Bytes(b) => {
            let mut bytes = b.clone();
            bytes.resize(element_size, 0);
            bytes
        }
    };

    let mut result = vec![0u8; total_size];
    for i in (0..total_size).step_by(element_size) {
        result[i..i + element_size].copy_from_slice(&fill_bytes);
    }
    result
}

/// Reads data from an array using a selection.
#[cfg(feature = "alloc")]
pub fn read_array<S: Store>(
    store: &S,
    array_key: &str,
    selection: &Selection,
    meta: &ArrayMetadata,
) -> Result<Vec<u8>, ChunkError> {
    let selection_steps = selection.normalized_steps(&meta.shape)?;
    let num_elements = if selection_steps.is_empty() {
        1usize
    } else {
        selection_steps
            .iter()
            .map(|step| step.count as usize)
            .product()
    };

    let element_size = crate::metadata::dtype_to_element_size(&meta.dtype).unwrap_or(8);
    let mut output = expand_fill_value(&meta.fill_value, &meta.dtype, num_elements as u64);

    let chunk_grid: Vec<u64> = meta
        .shape
        .iter()
        .zip(meta.chunks.iter())
        .map(|(&shape, &chunk)| shape.div_ceil(chunk) as u64)
        .collect();
    let selection_indices = selection_indices(&selection_steps);
    let mut chunk_indices: Vec<u64> = vec![0; meta.shape.len()];

    loop {
        let chunk_origin: Vec<u64> = chunk_indices
            .iter()
            .zip(meta.chunks.iter())
            .map(|(&index, &chunk)| index * chunk as u64)
            .collect();

        let mut chunk_extent = vec![0usize; meta.shape.len()];
        for dim in 0..meta.shape.len() {
            let remaining = meta.shape[dim].saturating_sub(chunk_origin[dim] as usize);
            chunk_extent[dim] = remaining.min(meta.chunks[dim]);
        }

        if chunk_intersects_selection(&chunk_origin, &chunk_extent, &selection_steps) {
            match read_chunk(store, array_key, &chunk_indices, meta) {
                Ok(chunk_data) => {
                    let chunk_elements = if chunk_extent.is_empty() {
                        1
                    } else {
                        chunk_extent.iter().product()
                    };
                    let expected_chunk_bytes = chunk_elements * element_size;
                    if chunk_data.len() != expected_chunk_bytes {
                        return Err(ChunkError::UnexpectedLength);
                    }

                    if selection.is_full() {
                        let chunk_strides = compute_strides(&chunk_extent);
                        let mut local_position = vec![0usize; chunk_extent.len()];

                        loop {
                            let mut absolute_linear = 0usize;
                            for dim in 0..chunk_extent.len() {
                                let absolute_index =
                                    chunk_origin[dim] as usize + local_position[dim];
                                absolute_linear =
                                    absolute_linear * meta.shape[dim] + absolute_index;
                            }

                            let mut chunk_linear = 0usize;
                            for dim in 0..chunk_extent.len() {
                                chunk_linear += local_position[dim] * chunk_strides[dim];
                            }

                            let output_byte_start = absolute_linear * element_size;
                            let chunk_byte_start = chunk_linear * element_size;
                            let output_byte_end = output_byte_start + element_size;
                            let chunk_byte_end = chunk_byte_start + element_size;

                            if output_byte_end > output.len() || chunk_byte_end > chunk_data.len() {
                                return Err(ChunkError::UnexpectedLength);
                            }

                            output[output_byte_start..output_byte_end]
                                .copy_from_slice(&chunk_data[chunk_byte_start..chunk_byte_end]);

                            let mut advanced_local = false;
                            for dim in (0..local_position.len()).rev() {
                                local_position[dim] += 1;
                                if local_position[dim] < chunk_extent[dim] {
                                    advanced_local = true;
                                    break;
                                }
                                local_position[dim] = 0;
                            }

                            if !advanced_local {
                                break;
                            }
                        }
                    } else {
                        copy_chunk_selection_to_output(
                            &chunk_data,
                            &chunk_origin,
                            &chunk_extent,
                            &selection_indices,
                            &mut output,
                            element_size,
                        )?;
                    }
                }
                Err(ChunkError::Uninitialized) => {}
                Err(e) => return Err(e),
            }
        }

        let mut advanced = false;
        for dim in (0..chunk_indices.len()).rev() {
            chunk_indices[dim] += 1;
            if chunk_indices[dim] < chunk_grid[dim] {
                advanced = true;
                break;
            }
            chunk_indices[dim] = 0;
        }

        if !advanced {
            break;
        }
    }

    Ok(output)
}

/// Writes data to an entire array.
#[cfg(feature = "alloc")]
pub fn write_array<S: Store>(
    store: &mut S,
    array_key: &str,
    meta: &ArrayMetadata,
    data: &[u8],
) -> Result<(), ChunkError> {
    let element_size = crate::metadata::dtype_to_element_size(&meta.dtype).unwrap_or(8);
    let total_elements = if meta.shape.is_empty() {
        1
    } else {
        meta.shape.iter().product()
    };
    let expected_len = total_elements * element_size;
    if data.len() != expected_len {
        return Err(ChunkError::UnexpectedLength);
    }

    let chunk_grid: Vec<u64> = meta
        .shape
        .iter()
        .zip(meta.chunks.iter())
        .map(|(&shape, &chunk)| shape.div_ceil(chunk) as u64)
        .collect();
    let array_strides = compute_strides(&meta.shape);

    let mut chunk_indices: Vec<u64> = vec![0; meta.shape.len()];

    loop {
        let chunk_origin: Vec<u64> = chunk_indices
            .iter()
            .zip(meta.chunks.iter())
            .map(|(&index, &chunk)| index * chunk as u64)
            .collect();

        let mut chunk_extent = vec![0usize; meta.shape.len()];
        for dim in 0..meta.shape.len() {
            let remaining = meta.shape[dim].saturating_sub(chunk_origin[dim] as usize);
            chunk_extent[dim] = remaining.min(meta.chunks[dim]);
        }

        let chunk_elements = if chunk_extent.is_empty() {
            1
        } else {
            chunk_extent.iter().product()
        };
        let chunk_bytes = chunk_elements * element_size;
        let chunk_strides = compute_strides(&chunk_extent);
        let mut chunk_data = vec![0u8; chunk_bytes];

        if chunk_extent.is_empty() {
            chunk_data.copy_from_slice(&data[..element_size]);
        } else {
            let mut local_position = vec![0usize; chunk_extent.len()];

            loop {
                let mut array_linear = 0usize;
                let mut chunk_linear = 0usize;

                for dim in 0..chunk_extent.len() {
                    let absolute_index = chunk_origin[dim] as usize + local_position[dim];
                    array_linear += absolute_index * array_strides[dim];
                    chunk_linear += local_position[dim] * chunk_strides[dim];
                }

                let array_byte_start = array_linear * element_size;
                let array_byte_end = array_byte_start + element_size;
                let chunk_byte_start = chunk_linear * element_size;
                let chunk_byte_end = chunk_byte_start + element_size;

                if array_byte_end > data.len() || chunk_byte_end > chunk_data.len() {
                    return Err(ChunkError::UnexpectedLength);
                }

                chunk_data[chunk_byte_start..chunk_byte_end]
                    .copy_from_slice(&data[array_byte_start..array_byte_end]);

                let mut advanced_local = false;
                for dim in (0..local_position.len()).rev() {
                    local_position[dim] += 1;
                    if local_position[dim] < chunk_extent[dim] {
                        advanced_local = true;
                        break;
                    }
                    local_position[dim] = 0;
                }

                if !advanced_local {
                    break;
                }
            }
        }

        write_chunk(store, array_key, &chunk_indices, meta, &chunk_data)?;

        let mut advanced = false;
        for dim in (0..chunk_indices.len()).rev() {
            chunk_indices[dim] += 1;
            if chunk_indices[dim] < chunk_grid[dim] {
                advanced = true;
                break;
            }
            chunk_indices[dim] = 0;
        }

        if !advanced {
            break;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{ChunkKeyEncoding, Codec, FillValue, ZarrVersion};
    use crate::store::InMemoryStore;

    #[test]
    fn selection_full() {
        let sel = Selection::full(3);
        assert!(sel.is_full());
        assert_eq!(sel.num_elements(), 0);
    }

    #[test]
    fn selection_is_full() {
        let sel = Selection::full(2);
        assert!(sel.is_full());

        let sel = Selection::from_steps(vec![
            SelectionStep {
                start: 0,
                count: 10,
                stride: 1,
            },
            SelectionStep {
                start: 0,
                count: 10,
                stride: 1,
            },
        ]);
        assert!(!sel.is_full());
    }

    #[test]
    fn chunk_key_v2() {
        // Using actual ChunkKeyEncoding structure
        let encoding = ChunkKeyEncoding {
            name: "v2".to_string(),
            separator: '.',
        };
        let key = chunk_key_for_array("arr", &[0u64, 0u64, 0u64], &encoding);
        assert_eq!(key, "arr.c/0/0/0");

        let key = chunk_key_for_array("myarray", &[1u64, 2u64, 3u64], &encoding);
        assert_eq!(key, "myarray.c/1/2/3");
    }

    #[test]
    fn chunk_key_default() {
        // Default encoding uses slash separator
        let encoding = ChunkKeyEncoding::default();
        let key = chunk_key_for_array("arr", &[0u64, 0u64, 0u64], &encoding);
        assert_eq!(key, "arr/c/0/0/0");
    }

    #[test]
    fn read_write_chunk() {
        let mut store = InMemoryStore::new();
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: vec![100, 100],
            chunks: vec![10, 10],
            dtype: "<f8".to_string(),
            fill_value: FillValue::Float("NaN".to_string()),
            order: 'C',
            codecs: vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
        };

        let chunk_data: Vec<u8> = (0..100)
            .flat_map(|i| {
                let val = i as f64;
                val.to_le_bytes()
            })
            .collect();

        // Write chunk
        write_chunk(&mut store, "test_array", &[0u64, 0u64], &meta, &chunk_data).unwrap();

        // Read chunk
        let read_data = read_chunk(&store, "test_array", &[0u64, 0u64], &meta).unwrap();
        assert_eq!(read_data, chunk_data);
    }

    #[test]
    fn read_write_chunk_with_compression() {
        let mut store = InMemoryStore::new();
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: vec![100, 100],
            chunks: vec![10, 10],
            dtype: "<f8".to_string(),
            fill_value: FillValue::Float("NaN".to_string()),
            order: 'C',
            codecs: vec![Codec {
                name: "gzip".to_string(),
                configuration: vec![("level".to_string(), "1".to_string())],
            }],
            chunk_key_encoding: ChunkKeyEncoding::default(),
        };

        let chunk_data: Vec<u8> = (0..100)
            .flat_map(|i| {
                let val = i as f64;
                val.to_le_bytes()
            })
            .collect();

        // Write chunk with compression
        write_chunk(&mut store, "test_array", &[0u64, 0u64], &meta, &chunk_data).unwrap();

        // Read chunk and decompress
        let read_data = read_chunk(&store, "test_array", &[0u64, 0u64], &meta).unwrap();
        assert_eq!(read_data, chunk_data);
    }

    #[test]
    fn read_chunk_out_of_bounds() {
        let store = InMemoryStore::new();
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: vec![10, 10],
            chunks: vec![5, 5],
            dtype: "<f8".to_string(),
            fill_value: FillValue::Float("NaN".to_string()),
            order: 'C',
            codecs: vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
        };

        let result = read_chunk(&store, "test_array", &[2u64, 0u64], &meta);
        assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));

        let result = read_chunk(&store, "test_array", &[0u64, 2u64], &meta);
        assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));

        let result = read_chunk(&store, "test_array", &[2u64, 2u64], &meta);
        assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));
    }

    #[test]
    fn write_chunk_out_of_bounds() {
        let mut store = InMemoryStore::new();
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: vec![10, 10],
            chunks: vec![5, 5],
            dtype: "<f8".to_string(),
            fill_value: FillValue::Float("NaN".to_string()),
            order: 'C',
            codecs: vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
        };

        let chunk_data: Vec<u8> = (0..25)
            .flat_map(|i| {
                let val = i as f64;
                val.to_le_bytes()
            })
            .collect();

        let result = write_chunk(&mut store, "test_array", &[2u64, 0u64], &meta, &chunk_data);
        assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));

        let result = write_chunk(&mut store, "test_array", &[0u64, 2u64], &meta, &chunk_data);
        assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));

        let result = write_chunk(&mut store, "test_array", &[2u64, 2u64], &meta, &chunk_data);
        assert!(matches!(result, Err(ChunkError::ChunkOutOfBounds)));
    }

    #[test]
    fn uninitialized_chunk_returns_empty() {
        let store = InMemoryStore::new();
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: vec![10, 10],
            chunks: vec![5, 5],
            dtype: "<f8".to_string(),
            fill_value: FillValue::Float("NaN".to_string()),
            order: 'C',
            codecs: vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
        };

        // Chunk [0, 0] was never written, should return Uninitialized
        let result = read_chunk(&store, "test_array", &[0u64, 0u64], &meta);
        assert!(matches!(result, Err(ChunkError::Uninitialized)));
    }

    #[test]
    fn selection_step_contiguous() {
        let step = SelectionStep {
            start: 0,
            count: 10,
            stride: 1,
        };
        assert!(step.contiguous());

        let step = SelectionStep {
            start: 0,
            count: 10,
            stride: 2,
        };
        assert!(!step.contiguous());
    }

    #[test]
    fn selection_step_end() {
        let step = SelectionStep {
            start: 5,
            count: 3,
            stride: 2,
        };
        assert_eq!(step.end(), 5 + 2 * 2 + 1);
    }

    #[test]
    fn selection_step_indices() {
        let step = SelectionStep {
            start: 0,
            count: 3,
            stride: 2,
        };
        let indices: Vec<u64> = step.indices().collect();
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn expand_fill_value_test() {
        let fill = FillValue::Float("42.0".to_string());
        let dtype = "<f8";
        let expanded = expand_fill_value(&fill, dtype, 4);
        assert_eq!(expanded.len(), 32); // 4 * 8 bytes
    }

    #[test]
    fn selection_num_elements() {
        let sel = Selection::from_steps(vec![
            SelectionStep {
                start: 0,
                count: 10,
                stride: 1,
            },
            SelectionStep {
                start: 0,
                count: 5,
                stride: 1,
            },
        ]);
        assert_eq!(sel.num_elements(), 50);
    }

    #[test]
    fn read_array_partial_selection_contiguous_across_chunks() {
        let mut store = InMemoryStore::new();
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: vec![4, 4],
            chunks: vec![2, 2],
            dtype: "<i4".to_string(),
            fill_value: FillValue::Int(0),
            order: 'C',
            codecs: vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
        };

        let data: Vec<u8> = (0..16i32).flat_map(|value| value.to_le_bytes()).collect();
        write_array(&mut store, "test_array", &meta, &data).unwrap();

        let selection = Selection::from_steps(vec![
            SelectionStep {
                start: 1,
                count: 2,
                stride: 1,
            },
            SelectionStep {
                start: 1,
                count: 3,
                stride: 1,
            },
        ]);

        let read_data = read_array(&store, "test_array", &selection, &meta).unwrap();
        let values: Vec<i32> = read_data
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        assert_eq!(values, vec![5, 6, 7, 9, 10, 11]);
    }

    #[test]
    fn read_array_partial_selection_strided_across_chunks() {
        let mut store = InMemoryStore::new();
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: vec![4, 4],
            chunks: vec![2, 2],
            dtype: "<i4".to_string(),
            fill_value: FillValue::Int(0),
            order: 'C',
            codecs: vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
        };

        let data: Vec<u8> = (0..16i32).flat_map(|value| value.to_le_bytes()).collect();
        write_array(&mut store, "test_array", &meta, &data).unwrap();

        let selection = Selection::from_steps(vec![
            SelectionStep {
                start: 0,
                count: 2,
                stride: 2,
            },
            SelectionStep {
                start: 1,
                count: 2,
                stride: 2,
            },
        ]);

        let read_data = read_array(&store, "test_array", &selection, &meta).unwrap();
        let values: Vec<i32> = read_data
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        assert_eq!(values, vec![1, 3, 9, 11]);
    }

    #[test]
    fn read_array_partial_selection_uninitialized_chunk_uses_fill_value() {
        let mut store = InMemoryStore::new();
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: vec![4, 4],
            chunks: vec![2, 2],
            dtype: "<i4".to_string(),
            fill_value: FillValue::Int(-1),
            order: 'C',
            codecs: vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
        };

        let initialized_chunk: Vec<u8> = [0i32, 1, 4, 5]
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        write_chunk(
            &mut store,
            "test_array",
            &[0u64, 0u64],
            &meta,
            &initialized_chunk,
        )
        .unwrap();

        let selection = Selection::from_steps(vec![
            SelectionStep {
                start: 0,
                count: 2,
                stride: 1,
            },
            SelectionStep {
                start: 0,
                count: 4,
                stride: 1,
            },
        ]);

        let read_data = read_array(&store, "test_array", &selection, &meta).unwrap();
        let values: Vec<i32> = read_data
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        assert_eq!(values, vec![0, 1, -1, -1, 4, 5, -1, -1]);
    }

    #[test]
    fn write_array_and_read_back() {
        let mut store = InMemoryStore::new();
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: vec![20, 20],
            chunks: vec![10, 10],
            dtype: "<f4".to_string(),
            fill_value: FillValue::Float("0.0".to_string()),
            order: 'C',
            codecs: vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
        };

        let data: Vec<u8> = (0..400)
            .flat_map(|i| {
                let val = i as f32;
                val.to_le_bytes()
            })
            .collect();

        write_array(&mut store, "test_array", &meta, &data).unwrap();

        let selection = Selection::full(2);
        let read_data = read_array(&store, "test_array", &selection, &meta).unwrap();
        assert_eq!(read_data, data);
    }
}
