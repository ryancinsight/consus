//! Selection index iteration and selection <-> chunk copy helpers.

#[cfg(feature = "alloc")]
use super::compute_strides;
#[cfg(feature = "alloc")]
use crate::chunk::error::ChunkError;
#[cfg(feature = "alloc")]
use crate::chunk::selection::SelectionStep;
#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

/// CSR-shaped selection indices: realized indices for all selection
/// dimensions stored in one contiguous buffer, with an offset table locating
/// each dimension's slice.
///
/// Replaces a `Vec<Vec<u64>>` so the traversal hot path performs a single
/// indirection (`flat[offsets[dim] + position]`) instead of two.
pub(crate) struct SelectionIndices {
    flat: Vec<u64>,
    offsets: Vec<usize>,
}

impl SelectionIndices {
    pub(crate) fn build(steps: &[SelectionStep]) -> Self {
        let mut flat = Vec::new();
        let mut offsets = Vec::with_capacity(steps.len());
        for step in steps {
            offsets.push(flat.len());
            flat.extend(step.indices());
        }
        Self { flat, offsets }
    }

    /// The realized indices for one selection dimension.
    fn dim(&self, dim: usize) -> &[u64] {
        let start = self.offsets[dim];
        let end = self
            .offsets
            .get(dim + 1)
            .copied()
            .unwrap_or(self.flat.len());
        &self.flat[start..end]
    }

    /// Number of selection dimensions.
    pub(crate) fn len(&self) -> usize {
        self.offsets.len()
    }

    /// True when the selection has no dimensions (scalar access).
    pub(crate) fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// Internal helper: checks if a chunk intersects with a selection.
pub(crate) fn chunk_intersects_selection(
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
pub(crate) fn copy_chunk_selection_to_output(
    chunk_data: &[u8],
    chunk_origin: &[u64],
    chunk_extent: &[usize],
    stored_shape: &[usize],
    selection_indices: &SelectionIndices,
    output: &mut [u8],
    element_size: usize,
) -> Result<(), ChunkError> {
    let chunk_strides = compute_strides(stored_shape);
    let dims: Vec<&[u64]> = (0..selection_indices.len())
        .map(|dim| selection_indices.dim(dim))
        .collect();
    let selection_shape: Vec<usize> = dims.iter().map(|dim| dim.len()).collect();
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
            let absolute_index = dims[dim][selection_position[dim]];
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
            if selection_position[dim] < dims[dim].len() {
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

/// Internal helper: copies selected elements from the input buffer into a chunk buffer.
pub(crate) fn copy_selection_input_to_chunk(
    input: &[u8],
    selection_indices: &SelectionIndices,
    chunk_origin: &[u64],
    chunk_extent: &[usize],
    stored_shape: &[usize],
    chunk_data: &mut [u8],
    element_size: usize,
) -> Result<(), ChunkError> {
    let chunk_strides = compute_strides(stored_shape);
    let dims: Vec<&[u64]> = (0..selection_indices.len())
        .map(|dim| selection_indices.dim(dim))
        .collect();
    let selection_shape: Vec<usize> = dims.iter().map(|dim| dim.len()).collect();
    let input_strides = compute_strides(&selection_shape);

    if selection_indices.is_empty() {
        if input.len() != element_size || chunk_data.len() != element_size {
            return Err(ChunkError::UnexpectedLength);
        }
        chunk_data.copy_from_slice(input);
        return Ok(());
    }

    let mut selection_position = vec![0usize; selection_indices.len()];

    loop {
        let mut in_chunk = true;
        let mut chunk_linear = 0usize;
        let mut input_linear = 0usize;

        for dim in 0..selection_indices.len() {
            let absolute_index = dims[dim][selection_position[dim]];
            let chunk_start = chunk_origin[dim];
            let chunk_end = chunk_start + chunk_extent[dim] as u64;
            if absolute_index < chunk_start || absolute_index >= chunk_end {
                in_chunk = false;
                break;
            }

            let local_index = (absolute_index - chunk_start) as usize;
            chunk_linear += local_index * chunk_strides[dim];
            input_linear += selection_position[dim] * input_strides[dim];
        }

        if in_chunk {
            let chunk_byte_start = chunk_linear * element_size;
            let chunk_byte_end = chunk_byte_start + element_size;
            let input_byte_start = input_linear * element_size;
            let input_byte_end = input_byte_start + element_size;

            if chunk_byte_end > chunk_data.len() || input_byte_end > input.len() {
                return Err(ChunkError::UnexpectedLength);
            }

            chunk_data[chunk_byte_start..chunk_byte_end]
                .copy_from_slice(&input[input_byte_start..input_byte_end]);
        }

        let mut advanced = false;
        for dim in (0..selection_position.len()).rev() {
            selection_position[dim] += 1;
            if selection_position[dim] < dims[dim].len() {
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
