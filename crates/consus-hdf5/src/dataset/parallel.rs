//! Parallel chunk read and filter-pipeline decompression for HDF5 chunked datasets.
//!
//! ## Parallelism Model
//!
//! Execution is split into two phases:
//!
//! **Phase 1 (parallel when `parallel-io` is enabled)**: Each `ChunkTask` reads
//! `location.size` bytes from the source and applies the full filter-pipeline
//! decompression pipeline. Tasks are independent because the HDF5 chunk grid
//! partitions the dataset space into non-overlapping byte ranges; therefore
//! concurrent execution introduces no data races.
//!
//! **Phase 2 (always serial)**: `ChunkResult`s are assembled into the caller's
//! output buffer via `copy_chunk_into_dataset`. Although output regions are
//! provably disjoint (by the HDF5 chunk grid invariant), the Rust borrow checker
//! cannot verify non-overlap across arbitrary-rank index expressions at compile
//! time, so assembly remains serial.
//!
//! ## Thread Safety Contract
//!
//! `execute_parallel` requires `R: ReadAt + Sync` so that `&R` can be shared
//! across Rayon worker threads without data races. The `CompressionRegistry`
//! trait already requires `Send + Sync` as supertraits, so no extra bound is
//! needed on the registry at call sites.
//!
//! ## Correctness Invariant
//!
//! For all pairs `(i, j)` where `i != j`:
//!   `tasks[i].location` and `tasks[j].location` refer to disjoint byte ranges
//! in the source (guaranteed by the HDF5 spec's chunk address uniqueness).
//!   `results[i].chunk_coord` and `results[j].chunk_coord` map to disjoint
//! regions in the output buffer (guaranteed by the chunk grid partition property).
//!
//! Therefore `execute_parallel` output is semantically identical to `execute_serial`
//! output regardless of Rayon scheduling order.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use consus_core::Result;

#[cfg(feature = "alloc")]
use consus_io::ReadAt;

#[cfg(feature = "alloc")]
use crate::dataset::chunk::{read_chunk_raw, ChunkLocation};

/// Fully describes the I/O and decompression work for one chunk.
///
/// All fields are computed before any I/O begins; construction performs
/// no reads. This separation allows the caller to build a complete task
/// list from cheap metadata operations, then dispatch all reads at once.
#[cfg(feature = "alloc")]
pub struct ChunkTask {
    /// N-dimensional chunk coordinate in the chunk grid.
    pub chunk_coord: Vec<usize>,
    /// On-disk location (address, compressed size, filter mask) of the chunk.
    pub location: ChunkLocation,
    /// Actual (possibly edge-clipped) chunk dimensions after dataset-boundary
    /// clamping.
    pub actual_chunk_dims: Vec<usize>,
    /// Expected byte count after full filter-pipeline decompression.
    pub uncompressed_size: usize,
}

/// Decoded output from processing one [`ChunkTask`].
#[cfg(feature = "alloc")]
pub struct ChunkResult {
    /// N-dimensional chunk coordinate (preserved from the originating `ChunkTask`).
    pub chunk_coord: Vec<usize>,
    /// Actual chunk dimensions (preserved from the originating `ChunkTask`).
    pub actual_chunk_dims: Vec<usize>,
    /// Decompressed chunk data. Length equals `uncompressed_size` from the task.
    pub data: Vec<u8>,
}

/// Read and decompress all tasks serially, in input order.
///
/// Available unconditionally (regardless of the `parallel-io` feature). Used as
/// the serial fallback and for single-chunk special cases.
///
/// # Errors
///
/// Propagates the first I/O or decompression error encountered.
#[cfg(feature = "alloc")]
pub fn execute_serial<R: ReadAt>(
    source: &R,
    tasks: Vec<ChunkTask>,
    filter_ids: &[u16],
    registry: &dyn consus_compression::CompressionRegistry,
    fill_value: Option<&[u8]>,
) -> Result<Vec<ChunkResult>> {
    tasks
        .into_iter()
        .map(|task| {
            let data = read_chunk_raw(
                source,
                &task.location,
                task.uncompressed_size,
                filter_ids,
                registry,
                fill_value,
            )?;
            Ok(ChunkResult {
                chunk_coord: task.chunk_coord,
                actual_chunk_dims: task.actual_chunk_dims,
                data,
            })
        })
        .collect()
}

/// Read and decompress all tasks concurrently using the Rayon thread pool.
///
/// Output order matches input order (`rayon::collect` preserves parallel
/// iterator ordering for `Vec` targets).
///
/// # Thread Safety
///
/// `R: Sync` is required so that `&R` can be shared across Rayon worker threads
/// without a data race. `CompressionRegistry: Send + Sync` is already guaranteed
/// by the trait's supertrait bound; no additional constraint is needed at call
/// sites.
///
/// # Errors
///
/// On error in any task, Rayon short-circuits and returns the first error
/// encountered. Partial results are discarded.
#[cfg(all(feature = "parallel-io", feature = "alloc"))]
pub fn execute_parallel<R: ReadAt + Sync>(
    source: &R,
    tasks: Vec<ChunkTask>,
    filter_ids: &[u16],
    registry: &dyn consus_compression::CompressionRegistry,
    fill_value: Option<&[u8]>,
) -> Result<Vec<ChunkResult>> {
    use rayon::prelude::*;
    tasks
        .into_par_iter()
        .map(|task| {
            let data = read_chunk_raw(
                source,
                &task.location,
                task.uncompressed_size,
                filter_ids,
                registry,
                fill_value,
            )?;
            Ok(ChunkResult {
                chunk_coord: task.chunk_coord,
                actual_chunk_dims: task.actual_chunk_dims,
                data,
            })
        })
        .collect()
}
