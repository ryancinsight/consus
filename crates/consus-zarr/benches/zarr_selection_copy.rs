//! Criterion comparison: jagged `Vec<Vec<u64>>` selection indices vs
//! CSR-shaped flat buffer + offset table on the chunk-selection copy
//! traversal.
//!
//! The pre-conversion jagged formulation is kept inline as a baseline so the
//! comparison is reproducible from the committed tree (moirai/ritk
//! precedent). Both variants drive the full public read path against a
//! populated `InMemoryStore`, so the measured surface is the production
//! traversal, not a synthetic inner loop.
//!
//! A parity assertion in the setup phase verifies the two implementations
//! produce byte-identical output on the same inputs before timing begins.

use consus_zarr::chunk::{Selection, SelectionStep, read_array};
use consus_zarr::metadata::{
    ArrayMetadata, ChunkKeyEncoding, FillValue, ZarrVersion, dtype_to_element_size,
};
use consus_zarr::store::InMemoryStore;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

/// Row-major strides for a shape (faithful replica of the private
/// `compute_strides`).
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

/// Stored shape to use for stride computation (faithful replica of the
/// private `stored_shape_for_chunk`).
fn stored_shape_for_chunk(
    chunk_data_len: usize,
    element_size: usize,
    chunk_extent: &[usize],
    full_chunk: &[usize],
) -> Vec<usize> {
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
        full_chunk.to_vec()
    } else {
        chunk_extent.to_vec()
    }
}

/// Pre-conversion `selection_indices`: one `Vec<u64>` per dimension.
fn jagged_selection_indices(steps: &[SelectionStep]) -> Vec<Vec<u64>> {
    steps.iter().map(|step| step.indices().collect()).collect()
}

/// Pre-conversion `copy_chunk_selection_to_output` over jagged indices.
fn jagged_copy_chunk_selection_to_output(
    chunk_data: &[u8],
    chunk_origin: &[u64],
    chunk_extent: &[usize],
    stored_shape: &[usize],
    selection_indices: &[Vec<u64>],
    output: &mut [u8],
    element_size: usize,
) -> Result<(), consus_zarr::chunk::ChunkError> {
    let chunk_strides = compute_strides(stored_shape);
    let selection_shape: Vec<usize> = selection_indices.iter().map(Vec::len).collect();
    let output_strides = compute_strides(&selection_shape);

    if selection_indices.is_empty() {
        if chunk_data.len() != element_size || output.len() != element_size {
            return Err(consus_zarr::chunk::ChunkError::UnexpectedLength);
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
                return Err(consus_zarr::chunk::ChunkError::UnexpectedLength);
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

/// Pre-conversion full read path over jagged selection indices.
///
/// A line-faithful replica of the non-sharded `read_array` path (including
/// the batch `get_many` store access), so the only difference from the
/// production function is the selection-index representation.
fn read_array_jagged(
    store: &InMemoryStore,
    array_key: &str,
    selection: &Selection,
    meta: &ArrayMetadata,
) -> Vec<u8> {
    use consus_zarr::store::Store;

    let selection_steps = selection.normalized_steps(&meta.shape).expect("normalize");
    let num_elements = if selection_steps.is_empty() {
        1usize
    } else {
        selection_steps
            .iter()
            .map(|step| step.count as usize)
            .product()
    };

    let element_size = dtype_to_element_size(&meta.dtype).unwrap_or(8);
    let mut output =
        consus_zarr::chunk::expand_fill_value(&meta.fill_value, &meta.dtype, num_elements as u64);

    let chunk_grid: Vec<u64> = meta
        .shape
        .iter()
        .zip(meta.chunks.iter())
        .map(|(&shape, &chunk)| shape.div_ceil(chunk) as u64)
        .collect();
    let selection_indices = jagged_selection_indices(&selection_steps);
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

        let mut intersects = true;
        for dim in 0..selection_steps.len() {
            let chunk_start = chunk_origin[dim];
            let chunk_end = chunk_start + chunk_extent[dim] as u64;
            let step = &selection_steps[dim];
            let hit = step
                .indices()
                .any(|index| index >= chunk_start && index < chunk_end);
            if !hit {
                intersects = false;
                break;
            }
        }

        if intersects {
            let coord_parts: Vec<String> = chunk_indices.iter().map(|c| c.to_string()).collect();
            let key = format!("{}/c/{}", array_key, coord_parts.join("/"));
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
                    Err(consus_zarr::chunk::ChunkError::Uninitialized)
                } else {
                    Ok(data)
                }
            }
            Err(consus_core::Error::NotFound { .. }) => {
                Err(consus_zarr::chunk::ChunkError::Uninitialized)
            }
            Err(e) => Err(consus_zarr::chunk::ChunkError::StoreError(e.to_string())),
        };

        match chunk_data_result {
            Ok(chunk_data) => {
                let padded_chunk_elements = if meta.chunks.is_empty() {
                    1
                } else {
                    meta.chunks.iter().product()
                };
                let padded_chunk_bytes = padded_chunk_elements * element_size;
                let chunk_elements = if chunk_extent.is_empty() {
                    1
                } else {
                    chunk_extent.iter().product()
                };
                let expected_chunk_bytes = chunk_elements * element_size;
                if chunk_data.len() != expected_chunk_bytes
                    && chunk_data.len() != padded_chunk_bytes
                {
                    return Vec::new();
                }

                let stored = stored_shape_for_chunk(
                    chunk_data.len(),
                    element_size,
                    chunk_extent,
                    &meta.chunks,
                );
                jagged_copy_chunk_selection_to_output(
                    &chunk_data,
                    chunk_origin,
                    chunk_extent,
                    &stored,
                    &selection_indices,
                    &mut output,
                    element_size,
                )
                .expect("copy");
            }
            Err(consus_zarr::chunk::ChunkError::Uninitialized) => {}
            Err(_) => return Vec::new(),
        }
    }

    output
}

/// Populates a store with a full array of deterministic `f64` bytes.
fn populate(shape: &[usize], chunks: &[usize]) -> (InMemoryStore, ArrayMetadata) {
    let meta = ArrayMetadata {
        version: ZarrVersion::V3,
        shape: shape.to_vec(),
        chunks: chunks.to_vec(),
        dtype: "<f8".to_string(),
        fill_value: FillValue::Float("NaN".to_string()),
        order: 'C',
        codecs: vec![],
        chunk_key_encoding: ChunkKeyEncoding::default(),
        dimension_names: None,
    };
    let total: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total * 8);
    for i in 0..total {
        let val = i as f64 * 0.5;
        data.extend_from_slice(&val.to_le_bytes());
    }
    let mut store = InMemoryStore::new();
    consus_zarr::chunk::write_array(&mut store, "bench", &meta, &data).expect("write array");
    (store, meta)
}

struct BenchCase {
    name: &'static str,
    shape: Vec<usize>,
    chunks: Vec<usize>,
    steps: Vec<SelectionStep>,
}

fn selection_copy_benchmarks(c: &mut Criterion) {
    let cases: &[BenchCase] = &[
        BenchCase {
            name: "2d_strided_256x256",
            shape: vec![512, 512],
            chunks: vec![64, 64],
            steps: vec![
                SelectionStep {
                    start: 0,
                    count: 256,
                    stride: 2,
                },
                SelectionStep {
                    start: 0,
                    count: 256,
                    stride: 2,
                },
            ],
        },
        BenchCase {
            name: "2d_strided_512x512",
            shape: vec![1024, 1024],
            chunks: vec![128, 128],
            steps: vec![
                SelectionStep {
                    start: 0,
                    count: 512,
                    stride: 2,
                },
                SelectionStep {
                    start: 0,
                    count: 512,
                    stride: 2,
                },
            ],
        },
        BenchCase {
            name: "3d_strided_64_cubed",
            shape: vec![128, 128, 128],
            chunks: vec![32, 32, 32],
            steps: vec![
                SelectionStep {
                    start: 0,
                    count: 64,
                    stride: 2,
                },
                SelectionStep {
                    start: 0,
                    count: 64,
                    stride: 2,
                },
                SelectionStep {
                    start: 0,
                    count: 64,
                    stride: 2,
                },
            ],
        },
    ];

    for case in cases {
        let (store, meta) = populate(&case.shape, &case.chunks);
        let selection = Selection::from_steps(case.steps.clone());
        let selected: usize = case.steps.iter().map(|s| s.count as usize).product();

        let production = read_array(&store, "bench", &selection, &meta).expect("read_array");
        let jagged = read_array_jagged(&store, "bench", &selection, &meta);
        assert_eq!(
            &production, &jagged,
            "parity: production and jagged replica must produce identical output for {}",
            case.name
        );

        let mut group = c.benchmark_group("zarr_selection_copy");
        group.throughput(Throughput::Bytes((selected * 8) as u64));
        let input = (store, selection);
        group.bench_with_input(
            BenchmarkId::new("jagged_baseline", case.name),
            &input,
            |b, (store, selection)| {
                b.iter(|| black_box(read_array_jagged(store, "bench", selection, &meta)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("csr_flat", case.name),
            &input,
            |b, (store, selection)| {
                b.iter(|| black_box(read_array(store, "bench", selection, &meta)));
            },
        );
        group.finish();
    }
}

criterion_group!(benches, selection_copy_benchmarks);
criterion_main!(benches);
