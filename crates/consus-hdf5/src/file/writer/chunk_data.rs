//! Chunk coordinate math, byte extraction, and chunked dataset data emission.

#[cfg(feature = "alloc")]
use super::{
    ChunkIndexEntry, DatasetCreationProps, WriteState, dataset_filter_ids, write_chunk_btree_v1,
    write_chunk_btree_v2, write_chunk_farray,
};
#[cfg(feature = "alloc")]
use crate::dataset::chunk::{ChunkLocation, edge_chunk_dims, write_chunk_raw};
#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};
#[cfg(feature = "alloc")]
use consus_core::{Datatype, Error, Result, Shape};
#[cfg(feature = "alloc")]
use consus_io::WriteAt;

#[cfg(feature = "alloc")]
fn linear_index(coords: &[usize], dims: &[usize]) -> usize {
    let mut index = 0usize;
    for (&coord, &dim) in coords.iter().zip(dims.iter()) {
        index = index * dim + coord;
    }
    index
}

#[cfg(feature = "alloc")]
fn increment_chunk_coord(coord: &mut [usize], grid_dims: &[usize]) -> bool {
    if coord.is_empty() {
        return false;
    }

    for dim in (0..coord.len()).rev() {
        coord[dim] += 1;
        if coord[dim] < grid_dims[dim] {
            return true;
        }
        coord[dim] = 0;
    }

    false
}

#[cfg(feature = "alloc")]
fn extract_chunk_bytes(
    raw_data: &[u8],
    dataset_dims: &[usize],
    chunk_coord: &[usize],
    chunk_dims: &[usize],
    element_size: usize,
) -> Result<Vec<u8>> {
    let actual_chunk_dims = edge_chunk_dims(chunk_coord, chunk_dims, dataset_dims);
    let chunk_elements = actual_chunk_dims.iter().product::<usize>();
    let mut chunk = vec![0u8; chunk_elements * element_size];

    if dataset_dims.is_empty() {
        if raw_data.len() != element_size {
            return Err(Error::InvalidFormat {
                message: String::from("scalar dataset raw byte length does not match element size"),
            });
        }
        chunk.copy_from_slice(raw_data);
        return Ok(chunk);
    }

    let rank = dataset_dims.len();
    let chunk_origin: Vec<usize> = chunk_coord
        .iter()
        .zip(chunk_dims.iter())
        .map(|(&coord, &dim)| coord.checked_mul(dim).ok_or(Error::Overflow))
        .collect::<Result<Vec<usize>>>()?;

    let mut local_coord = vec![0usize; rank];
    let mut done = false;

    while !done {
        let mut dataset_coord = Vec::with_capacity(rank);
        for d in 0..rank {
            dataset_coord.push(
                chunk_origin[d]
                    .checked_add(local_coord[d])
                    .ok_or(Error::Overflow)?,
            );
        }

        let dataset_linear = linear_index(&dataset_coord, dataset_dims);
        let chunk_linear = linear_index(&local_coord, &actual_chunk_dims);

        let src_start = dataset_linear
            .checked_mul(element_size)
            .ok_or(Error::Overflow)?;
        let src_end = src_start.checked_add(element_size).ok_or(Error::Overflow)?;
        let dst_start = chunk_linear
            .checked_mul(element_size)
            .ok_or(Error::Overflow)?;
        let dst_end = dst_start.checked_add(element_size).ok_or(Error::Overflow)?;

        chunk[dst_start..dst_end].copy_from_slice(&raw_data[src_start..src_end]);

        for dim in (0..rank).rev() {
            local_coord[dim] += 1;
            if local_coord[dim] < actual_chunk_dims[dim] {
                break;
            }
            local_coord[dim] = 0;
            if dim == 0 {
                done = true;
            }
        }
    }

    Ok(chunk)
}

#[cfg(feature = "alloc")]
pub(crate) fn write_chunked_data<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    datatype: &Datatype,
    shape: &Shape,
    raw_data: &[u8],
    props: &DatasetCreationProps,
) -> Result<u64> {
    let element_size = datatype
        .element_size()
        .ok_or_else(|| Error::UnsupportedFeature {
            feature: String::from("chunked write requires fixed-size element datatype"),
        })?;
    write_chunked_data_with_element_size(sink, state, element_size, shape, raw_data, props)
}

#[cfg(feature = "alloc")]
pub(crate) fn write_chunked_data_with_element_size<W: WriteAt>(
    sink: &mut W,
    state: &mut WriteState,
    element_size: usize,
    shape: &Shape,
    raw_data: &[u8],
    props: &DatasetCreationProps,
) -> Result<u64> {
    let chunk_dims = props
        .chunk_dims
        .as_ref()
        .ok_or_else(|| Error::InvalidFormat {
            message: String::from("chunked dataset write requires chunk dimensions"),
        })?;

    if chunk_dims.len() != shape.rank() {
        return Err(Error::ShapeError {
            message: alloc::format!(
                "chunk rank mismatch: dataset rank {}, chunk rank {}",
                shape.rank(),
                chunk_dims.len()
            ),
        });
    }
    let dataset_dims = shape.current_dims();
    let expected_len = shape
        .num_elements()
        .checked_mul(element_size)
        .ok_or(Error::Overflow)?;
    if raw_data.len() != expected_len {
        return Err(Error::ShapeError {
            message: alloc::format!(
                "dataset payload byte length mismatch: expected {expected_len}, found {}",
                raw_data.len()
            ),
        });
    }

    let grid_dims: Vec<usize> = if dataset_dims.is_empty() {
        Vec::new()
    } else {
        dataset_dims
            .iter()
            .zip(chunk_dims.iter())
            .map(|(&dataset_dim, &chunk_dim)| dataset_dim.div_ceil(chunk_dim))
            .collect()
    };

    let filter_ids = dataset_filter_ids(props);
    let registry = consus_compression::DefaultCodecRegistry::new();
    let mut entries = Vec::new();

    if dataset_dims.is_empty() {
        let location = write_chunk_raw(
            sink,
            state.eof,
            raw_data,
            &filter_ids,
            element_size,
            &registry,
        )?;
        state.allocate_aligned(location.size);
        entries.push(ChunkIndexEntry {
            chunk_offsets: Vec::new(),
            filter_mask: location.filter_mask,
            chunk_size: location.size as u32,
            chunk_address: location.address,
        });
    } else {
        let mut chunk_coord = vec![0usize; grid_dims.len()];
        loop {
            let chunk_bytes = extract_chunk_bytes(
                raw_data,
                &dataset_dims,
                &chunk_coord,
                chunk_dims,
                element_size,
            )?;
            let location: ChunkLocation = write_chunk_raw(
                sink,
                state.eof,
                &chunk_bytes,
                &filter_ids,
                element_size,
                &registry,
            )?;
            state.allocate_aligned(location.size);

            let chunk_offsets: Vec<u64> = chunk_coord
                .iter()
                .zip(chunk_dims.iter())
                .map(|(&coord, &dim)| {
                    let elem_offset = coord.checked_mul(dim).ok_or(Error::Overflow)?;
                    u64::try_from(elem_offset).map_err(|_| Error::Overflow)
                })
                .collect::<Result<Vec<u64>>>()?;

            entries.push(ChunkIndexEntry {
                chunk_offsets,
                filter_mask: location.filter_mask,
                chunk_size: location.size as u32,
                chunk_address: location.address,
            });

            if !increment_chunk_coord(&mut chunk_coord, &grid_dims) {
                break;
            }
        }
    }

    let has_filters = !filter_ids.is_empty();
    if props.layout_version == Some(4) && !has_filters {
        // Fixed Array (FARRAY) indexing for fixed-dimension datasets without
        // filters.  Write FAHD + FADB and return the FAHD address.
        write_chunk_farray(sink, state, &entries)
    } else if props.layout_version == Some(4) {
        write_chunk_btree_v2(sink, state, chunk_dims, &entries, has_filters)
    } else {
        write_chunk_btree_v1(
            sink,
            state,
            chunk_dims,
            &entries,
            &dataset_dims,
            element_size,
        )
    }
}
