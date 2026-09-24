//! Whole-array and selection read/write paths.

#[cfg(feature = "alloc")]
use super::{
    SelectionIndices, checked_chunk_bytes, chunk_intersects_selection, chunk_key_for_array,
    compute_strides, copy_chunk_selection_to_output, copy_selection_input_to_chunk,
    read_array_sharded, read_chunk, stored_shape_for_chunk, try_expand_fill_value,
    write_array_sharded, write_chunk,
};
#[cfg(feature = "alloc")]
use crate::chunk::error::ChunkError;
#[cfg(feature = "alloc")]
use crate::chunk::selection::Selection;
#[cfg(feature = "alloc")]
use crate::metadata::{ArrayMetadata, dtype_to_element_size};
#[cfg(feature = "alloc")]
use crate::store::Store;
#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

/// Reads data from an array using a selection.
#[cfg(feature = "alloc")]
pub fn read_array<S: Store>(
    store: &S,
    array_key: &str,
    selection: &Selection,
    meta: &ArrayMetadata,
) -> Result<Vec<u8>, ChunkError> {
    if let Some(shard_cfg) = crate::shard::extract_sharding_config(&meta.codecs) {
        return read_array_sharded(store, array_key, selection, meta, &shard_cfg);
    }
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
    let mut output = try_expand_fill_value(&meta.fill_value, &meta.dtype, num_elements as u64)?;

    let chunk_grid: Vec<u64> = meta
        .shape
        .iter()
        .zip(meta.chunks.iter())
        .map(|(&shape, &chunk)| shape.div_ceil(chunk) as u64)
        .collect();
    let selection_indices = SelectionIndices::build(&selection_steps);
    let mut chunk_indices: Vec<u64> = vec![0; meta.shape.len()];

    let mut chunk_indices_list = Vec::new();
    let mut extents_list = Vec::new();
    let mut origins_list = Vec::new();
    let mut keys_list = Vec::new();

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
            let key = chunk_key_for_array(array_key, &chunk_indices, &meta.chunk_key_encoding);
            chunk_indices_list.push(chunk_indices.clone());
            extents_list.push(chunk_extent);
            origins_list.push(chunk_origin);
            keys_list.push(key);
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

    let keys_ref: Vec<&str> = keys_list.iter().map(|s| s.as_str()).collect();
    let raw_chunks = store.get_many(&keys_ref);

    for (i, raw_result) in raw_chunks.into_iter().enumerate() {
        let chunk_extent = &extents_list[i];
        let chunk_origin = &origins_list[i];

        let chunk_data_result = match raw_result {
            Ok(data) => {
                if data.is_empty() {
                    Err(ChunkError::Uninitialized)
                } else if !meta.codecs.is_empty() {
                    #[cfg(not(feature = "std"))]
                    {
                        Err(ChunkError::DecompressFailed)
                    }
                    #[cfg(feature = "std")]
                    {
                        use crate::codec::{CodecPipeline, default_registry};
                        CodecPipeline::new(meta.codecs.clone())
                            .with_element_size(dtype_to_element_size(&meta.dtype).unwrap_or(1))
                            .decompress(&data, default_registry())
                            .map_err(|_| ChunkError::DecompressFailed)
                    }
                } else {
                    Ok(data)
                }
            }
            Err(consus_core::Error::NotFound { .. }) => Err(ChunkError::Uninitialized),
            Err(e) => Err(ChunkError::StoreError(e.to_string())),
        };

        match chunk_data_result {
            Ok(chunk_data) => {
                let padded_chunk_elements = if meta.chunks.is_empty() {
                    1
                } else {
                    meta.chunks.iter().product()
                };
                let padded_chunk_bytes = checked_chunk_bytes(padded_chunk_elements, element_size)?;
                let chunk_elements = if chunk_extent.is_empty() {
                    1
                } else {
                    chunk_extent.iter().product()
                };
                let expected_chunk_bytes = checked_chunk_bytes(chunk_elements, element_size)?;
                if chunk_data.len() != expected_chunk_bytes
                    && chunk_data.len() != padded_chunk_bytes
                {
                    return Err(ChunkError::UnexpectedLength);
                }

                if selection.is_full() {
                    let index_shape = stored_shape_for_chunk(
                        chunk_data.len(),
                        element_size,
                        chunk_extent,
                        &meta.chunks,
                    );
                    let chunk_strides = compute_strides(index_shape);
                    let mut local_position = vec![0usize; chunk_extent.len()];

                    loop {
                        let mut absolute_linear = 0usize;
                        for dim in 0..chunk_extent.len() {
                            let absolute_index = chunk_origin[dim] as usize + local_position[dim];
                            absolute_linear = absolute_linear * meta.shape[dim] + absolute_index;
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
                    let stored = stored_shape_for_chunk(
                        chunk_data.len(),
                        element_size,
                        chunk_extent,
                        &meta.chunks,
                    );
                    copy_chunk_selection_to_output(
                        &chunk_data,
                        chunk_origin,
                        chunk_extent,
                        stored,
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

    Ok(output)
}

/// Writes data to an array using a selection.
#[cfg(feature = "alloc")]
pub fn write_array_selection<S: Store>(
    store: &mut S,
    array_key: &str,
    selection: &Selection,
    meta: &ArrayMetadata,
    data: &[u8],
) -> Result<(), ChunkError> {
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
    let expected_len = checked_chunk_bytes(num_elements, element_size)?;
    if data.len() != expected_len {
        return Err(ChunkError::UnexpectedLength);
    }

    let chunk_grid: Vec<u64> = meta
        .shape
        .iter()
        .zip(meta.chunks.iter())
        .map(|(&shape, &chunk)| shape.div_ceil(chunk) as u64)
        .collect();
    let selection_indices = SelectionIndices::build(&selection_steps);
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
            let chunk_elements = if chunk_extent.is_empty() {
                1
            } else {
                chunk_extent.iter().product()
            };
            let chunk_bytes = checked_chunk_bytes(chunk_elements, element_size)?;
            let padded_chunk_elements = if meta.chunks.is_empty() {
                1
            } else {
                meta.chunks.iter().product()
            };
            let padded_chunk_bytes = checked_chunk_bytes(padded_chunk_elements, element_size)?;

            let mut chunk_data = match read_chunk(store, array_key, &chunk_indices, meta) {
                Ok(existing) => {
                    if existing.len() == chunk_bytes || existing.len() == padded_chunk_bytes {
                        existing
                    } else {
                        return Err(ChunkError::UnexpectedLength);
                    }
                }
                Err(ChunkError::Uninitialized) => try_expand_fill_value(
                    &meta.fill_value,
                    &meta.dtype,
                    padded_chunk_elements as u64,
                )?,
                Err(e) => return Err(e),
            };

            if chunk_data.len() != chunk_bytes && chunk_data.len() != padded_chunk_bytes {
                return Err(ChunkError::UnexpectedLength);
            }

            let stored =
                stored_shape_for_chunk(chunk_data.len(), element_size, &chunk_extent, &meta.chunks);
            copy_selection_input_to_chunk(
                data,
                &selection_indices,
                &chunk_origin,
                &chunk_extent,
                stored,
                &mut chunk_data,
                element_size,
            )?;

            write_chunk(store, array_key, &chunk_indices, meta, &chunk_data)?;
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

    Ok(())
}

/// Writes data to an entire array.
#[cfg(feature = "alloc")]
pub fn write_array<S: Store>(
    store: &mut S,
    array_key: &str,
    meta: &ArrayMetadata,
    data: &[u8],
) -> Result<(), ChunkError> {
    if let Some(shard_cfg) = crate::shard::extract_sharding_config(&meta.codecs) {
        return write_array_sharded(store, array_key, meta, data, &shard_cfg);
    }
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
        let chunk_bytes = checked_chunk_bytes(chunk_elements, element_size)?;
        let chunk_strides = compute_strides(&chunk_extent);
        let mut chunk_data = consus_core::ParseBudget::DEFAULT
            .zeroed(chunk_bytes as u64, "zarr shard chunk buffer")
            .map_err(|e| ChunkError::StoreError(e.to_string()))?;

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
