//! Sharded (v3) read/write paths, decomposing shards into inner chunks.

#[cfg(feature = "alloc")]
use super::{
    SelectionIndices, chunk_intersects_selection, chunk_key_for_array, compute_strides,
    copy_chunk_selection_to_output, stored_shape_for_chunk, try_expand_fill_value,
};
#[cfg(feature = "alloc")]
use crate::chunk::error::ChunkError;
#[cfg(feature = "alloc")]
use crate::chunk::selection::Selection;
#[cfg(feature = "alloc")]
use crate::metadata::ArrayMetadata;
#[cfg(feature = "alloc")]
use crate::store::Store;
#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
pub(crate) fn read_array_sharded<S: Store>(
    store: &S,
    array_key: &str,
    selection: &Selection,
    meta: &ArrayMetadata,
    shard_cfg: &crate::shard::ShardingConfig,
) -> Result<Vec<u8>, ChunkError> {
    use crate::shard::{inner_linear_index, read_inner_chunk_from_shard};

    let selection_steps = selection.normalized_steps(&meta.shape)?;
    let num_elements: usize = if selection_steps.is_empty() {
        1
    } else {
        selection_steps.iter().map(|s| s.count as usize).product()
    };
    let element_size = crate::metadata::dtype_to_element_size(&meta.dtype).unwrap_or(8);
    let mut output = try_expand_fill_value(&meta.fill_value, &meta.dtype, num_elements as u64)?;

    let shard_grid: Vec<u64> = meta
        .shape
        .iter()
        .zip(meta.chunks.iter())
        .map(|(&shape, &chunk)| shape.div_ceil(chunk) as u64)
        .collect();
    let inner_per_dim = shard_cfg.inner_chunks_per_dim(&meta.chunks);
    let sel_indices = SelectionIndices::build(&selection_steps);
    let mut shard_coords: Vec<u64> = vec![0; meta.shape.len()];

    loop {
        let shard_origin: Vec<u64> = shard_coords
            .iter()
            .zip(meta.chunks.iter())
            .map(|(&idx, &chunk)| idx * chunk as u64)
            .collect();
        let shard_key = chunk_key_for_array(array_key, &shard_coords, &meta.chunk_key_encoding);
        let shard_data = match store.get(&shard_key) {
            Ok(d) => d,
            Err(consus_core::Error::NotFound { .. }) => {
                let mut advanced = false;
                for dim in (0..shard_coords.len()).rev() {
                    shard_coords[dim] += 1;
                    if shard_coords[dim] < shard_grid[dim] {
                        advanced = true;
                        break;
                    }
                    shard_coords[dim] = 0;
                }
                if !advanced {
                    break;
                }
                continue;
            }
            Err(e) => return Err(ChunkError::StoreError(e.to_string())),
        };
        let total_inner = shard_cfg.total_inner_chunks(&meta.chunks);
        let mut inner_coords: Vec<usize> = vec![0; meta.shape.len()];
        loop {
            let inner_origin: Vec<u64> = inner_coords
                .iter()
                .zip(shard_cfg.inner_chunk_shape.iter())
                .zip(shard_origin.iter())
                .map(|((&ic, &is), &so)| so + (ic * is) as u64)
                .collect();
            let mut inner_extent = vec![0usize; meta.shape.len()];
            for dim in 0..meta.shape.len() {
                let remaining = meta.shape[dim].saturating_sub(inner_origin[dim] as usize);
                inner_extent[dim] = remaining.min(shard_cfg.inner_chunk_shape[dim]);
            }
            let intersects = if selection.is_full() {
                true
            } else {
                chunk_intersects_selection(&inner_origin, &inner_extent, &selection_steps)
            };
            if intersects {
                let linear = inner_linear_index(&inner_coords, &inner_per_dim);
                match read_inner_chunk_from_shard(
                    &shard_data,
                    linear,
                    total_inner,
                    &shard_cfg.inner_codecs,
                    element_size,
                ) {
                    Ok(chunk_data) if !chunk_data.is_empty() => {
                        let stored = stored_shape_for_chunk(
                            chunk_data.len(),
                            element_size,
                            &inner_extent,
                            &shard_cfg.inner_chunk_shape,
                        );
                        if selection.is_full() {
                            let chunk_strides = compute_strides(stored);
                            let mut local_pos = vec![0usize; inner_extent.len()];
                            loop {
                                let mut abs_linear = 0usize;
                                for dim in 0..inner_extent.len() {
                                    let abs_idx = inner_origin[dim] as usize + local_pos[dim];
                                    abs_linear = abs_linear * meta.shape[dim] + abs_idx;
                                }
                                let mut chunk_linear = 0usize;
                                for dim in 0..inner_extent.len() {
                                    chunk_linear += local_pos[dim] * chunk_strides[dim];
                                }
                                let out_start = abs_linear * element_size;
                                let chnk_start = chunk_linear * element_size;
                                if out_start + element_size <= output.len()
                                    && chnk_start + element_size <= chunk_data.len()
                                {
                                    output[out_start..out_start + element_size].copy_from_slice(
                                        &chunk_data[chnk_start..chnk_start + element_size],
                                    );
                                }
                                let mut adv = false;
                                for dim in (0..local_pos.len()).rev() {
                                    local_pos[dim] += 1;
                                    if local_pos[dim] < inner_extent[dim] {
                                        adv = true;
                                        break;
                                    }
                                    local_pos[dim] = 0;
                                }
                                if !adv {
                                    break;
                                }
                            }
                        } else {
                            copy_chunk_selection_to_output(
                                &chunk_data,
                                &inner_origin,
                                &inner_extent,
                                stored,
                                &sel_indices,
                                &mut output,
                                element_size,
                            )?;
                        }
                    }
                    Ok(_) => {}
                    Err(_) => {
                        return Err(ChunkError::DecompressFailed);
                    }
                }
            }
            let mut adv = false;
            for dim in (0..inner_coords.len()).rev() {
                inner_coords[dim] += 1;
                if inner_coords[dim] < inner_per_dim[dim] {
                    adv = true;
                    break;
                }
                inner_coords[dim] = 0;
            }
            if !adv {
                break;
            }
        }
        let mut advanced = false;
        for dim in (0..shard_coords.len()).rev() {
            shard_coords[dim] += 1;
            if shard_coords[dim] < shard_grid[dim] {
                advanced = true;
                break;
            }
            shard_coords[dim] = 0;
        }
        if !advanced {
            break;
        }
    }
    Ok(output)
}

#[cfg(feature = "alloc")]
pub(crate) fn write_array_sharded<S: Store>(
    store: &mut S,
    array_key: &str,
    meta: &ArrayMetadata,
    data: &[u8],
    shard_cfg: &crate::shard::ShardingConfig,
) -> Result<(), ChunkError> {
    use crate::shard::{inner_linear_index, write_shard};

    let element_size = crate::metadata::dtype_to_element_size(&meta.dtype).unwrap_or(8);
    let total_elements: usize = if meta.shape.is_empty() {
        1
    } else {
        meta.shape.iter().product()
    };
    if data.len() != total_elements * element_size {
        return Err(ChunkError::UnexpectedLength);
    }
    let shard_grid: Vec<u64> = meta
        .shape
        .iter()
        .zip(meta.chunks.iter())
        .map(|(&shape, &chunk)| shape.div_ceil(chunk) as u64)
        .collect();
    let inner_per_dim = shard_cfg.inner_chunks_per_dim(&meta.chunks);
    let array_strides = compute_strides(&meta.shape);
    let mut shard_coords: Vec<u64> = vec![0; meta.shape.len()];
    loop {
        let shard_origin: Vec<u64> = shard_coords
            .iter()
            .zip(meta.chunks.iter())
            .map(|(&idx, &chunk)| idx * chunk as u64)
            .collect();
        let total_inner = shard_cfg.total_inner_chunks(&meta.chunks);
        let mut inner_chunks: alloc::collections::BTreeMap<usize, Vec<u8>> =
            alloc::collections::BTreeMap::new();
        let mut inner_coords: Vec<usize> = vec![0; meta.shape.len()];
        loop {
            let inner_origin: Vec<u64> = inner_coords
                .iter()
                .zip(shard_cfg.inner_chunk_shape.iter())
                .zip(shard_origin.iter())
                .map(|((&ic, &is), &so)| so + (ic * is) as u64)
                .collect();
            let mut inner_extent = vec![0usize; meta.shape.len()];
            for dim in 0..meta.shape.len() {
                let remaining = meta.shape[dim].saturating_sub(inner_origin[dim] as usize);
                inner_extent[dim] = remaining.min(shard_cfg.inner_chunk_shape[dim]);
            }
            let inner_elements: usize = if inner_extent.is_empty() {
                1
            } else {
                inner_extent.iter().product()
            };
            let inner_strides = compute_strides(&inner_extent);
            let mut chunk_data = vec![0u8; inner_elements * element_size];
            let mut local_pos = vec![0usize; inner_extent.len()];
            loop {
                let mut array_linear = 0usize;
                let mut chunk_linear = 0usize;
                for dim in 0..inner_extent.len() {
                    let abs_idx = inner_origin[dim] as usize + local_pos[dim];
                    array_linear += abs_idx * array_strides[dim];
                    chunk_linear += local_pos[dim] * inner_strides[dim];
                }
                let arr_start = array_linear * element_size;
                let chnk_start = chunk_linear * element_size;
                if arr_start + element_size <= data.len()
                    && chnk_start + element_size <= chunk_data.len()
                {
                    chunk_data[chnk_start..chnk_start + element_size]
                        .copy_from_slice(&data[arr_start..arr_start + element_size]);
                }
                let mut adv = false;
                for dim in (0..local_pos.len()).rev() {
                    local_pos[dim] += 1;
                    if local_pos[dim] < inner_extent[dim] {
                        adv = true;
                        break;
                    }
                    local_pos[dim] = 0;
                }
                if !adv {
                    break;
                }
            }
            let compressed = if shard_cfg.inner_codecs.is_empty() {
                chunk_data
            } else {
                #[cfg(not(feature = "std"))]
                return Err(ChunkError::CompressFailed);
                #[cfg(feature = "std")]
                crate::codec::CodecPipeline::new(shard_cfg.inner_codecs.clone())
                    .with_element_size(element_size)
                    .compress(&chunk_data, crate::codec::default_registry())
                    .map_err(|_| ChunkError::CompressFailed)?
            };
            let linear = inner_linear_index(&inner_coords, &inner_per_dim);
            inner_chunks.insert(linear, compressed);
            let mut adv = false;
            for dim in (0..inner_coords.len()).rev() {
                inner_coords[dim] += 1;
                if inner_coords[dim] < inner_per_dim[dim] {
                    adv = true;
                    break;
                }
                inner_coords[dim] = 0;
            }
            if !adv {
                break;
            }
        }
        let shard_bytes = write_shard(&inner_chunks, total_inner);
        let shard_key = chunk_key_for_array(array_key, &shard_coords, &meta.chunk_key_encoding);
        store
            .set(&shard_key, &shard_bytes)
            .map_err(|e| ChunkError::StoreError(e.to_string()))?;
        let mut advanced = false;
        for dim in (0..shard_coords.len()).rev() {
            shard_coords[dim] += 1;
            if shard_coords[dim] < shard_grid[dim] {
                advanced = true;
                break;
            }
            shard_coords[dim] = 0;
        }
        if !advanced {
            break;
        }
    }
    Ok(())
}
