#[cfg(feature = "alloc")]
use alloc::{collections::BTreeMap, vec, vec::Vec};

#[cfg(feature = "std")]
use crate::codec::CodecPipeline;
#[cfg(feature = "std")]
use crate::codec::default_registry;
use crate::metadata::Codec;
use consus_core::Result;

use super::ShardError;

/// Compute the row-major linear index for inner chunk coordinates within a shard.
///
/// `linear = sum_i( coords[i] * product(inner_per_dim[i+1..]) )`
#[must_use]
pub fn inner_linear_index(coords: &[usize], inner_per_dim: &[usize]) -> usize {
    let mut linear = 0usize;
    for (i, &coord) in coords.iter().enumerate() {
        let stride: usize = if i + 1 < inner_per_dim.len() {
            inner_per_dim[i + 1..].iter().product()
        } else {
            1
        };
        linear += coord * stride;
    }
    linear
}

/// Read and decompress a single inner chunk from a shard byte slice.
///
/// ## Layout
///
/// The shard index occupies the last `total_inner_chunks * 16` bytes of `shard_data`.
/// Each index entry is `(offset: u64 LE, nbytes: u64 LE)`.
/// Uninitialized chunks have `(u64::MAX, u64::MAX)` and return an empty `Vec`.
///
/// ## Returns
///
/// The decompressed inner chunk bytes, or an empty `Vec` for uninitialized chunks.
#[cfg(feature = "alloc")]
pub fn read_inner_chunk_from_shard(
    shard_data: &[u8],
    inner_linear_idx: usize,
    total_inner_chunks: usize,
    inner_codecs: &[Codec],
) -> Result<Vec<u8>> {
    let index_size = total_inner_chunks.saturating_mul(16);
    if shard_data.len() < index_size {
        return Err(ShardError::ShardTooSmall {
            shard_size: shard_data.len() as u64,
            index_size: index_size as u64,
        }
        .into());
    }
    let index_start = shard_data.len() - index_size;
    let entry_offset = index_start + inner_linear_idx.saturating_mul(16);
    if entry_offset + 16 > shard_data.len() {
        return Err(ShardError::ChunkOutOfBounds {
            coords: vec![inner_linear_idx],
            grid: vec![total_inner_chunks],
        }
        .into());
    }
    let offset = u64::from_le_bytes(
        shard_data[entry_offset..entry_offset + 8]
            .try_into()
            .expect("8-byte slice for u64 deserialization"),
    );
    let length = u64::from_le_bytes(
        shard_data[entry_offset + 8..entry_offset + 16]
            .try_into()
            .expect("8-byte slice for u64 deserialization"),
    );
    if offset == u64::MAX && length == u64::MAX {
        return Ok(Vec::new());
    }
    let start = offset as usize;
    let end = start.saturating_add(length as usize);
    if end > index_start {
        return Err(ShardError::InvalidChunkEntry {
            coords: vec![inner_linear_idx],
            offset,
            length,
            shard_size: shard_data.len() as u64,
        }
        .into());
    }
    let compressed = &shard_data[start..end];
    if inner_codecs.is_empty() {
        return Ok(compressed.to_vec());
    }
    #[cfg(not(feature = "std"))]
    return Err(consus_core::Error::UnsupportedFeature {
        feature: alloc::string::String::from("shard_codec_requires_std"),
    });
    #[cfg(feature = "std")]
    return CodecPipeline::new(inner_codecs.to_vec()).decompress(compressed, default_registry());
}

/// Assemble a shard file from a map of compressed inner chunks.
///
/// ## Layout
///
/// `[inner_chunk_0][inner_chunk_1]...[inner_chunk_N-1][shard_index]`
///
/// Index entries use absolute byte offsets from the start of the shard file.
/// Uninitialized entries use `(u64::MAX, u64::MAX)`.
///
/// ## Parameters
///
/// - `inner_chunks`: `linear_idx -> compressed_chunk_bytes`.
/// - `total_inner_chunks`: total number of inner chunk slots (index entries).
#[cfg(feature = "alloc")]
pub fn write_shard(inner_chunks: &BTreeMap<usize, Vec<u8>>, total_inner_chunks: usize) -> Vec<u8> {
    let mut data_section: Vec<u8> = Vec::new();
    let mut index_entries: Vec<(u64, u64)> = vec![(u64::MAX, u64::MAX); total_inner_chunks];
    let mut current_offset: u64 = 0;
    for (&linear_idx, chunk_bytes) in inner_chunks {
        if linear_idx < total_inner_chunks {
            index_entries[linear_idx] = (current_offset, chunk_bytes.len() as u64);
            data_section.extend_from_slice(chunk_bytes);
            current_offset += chunk_bytes.len() as u64;
        }
    }
    let index_size = total_inner_chunks * 16;
    let mut shard = Vec::with_capacity(data_section.len() + index_size);
    shard.extend_from_slice(&data_section);
    for (offset, length) in &index_entries {
        shard.extend_from_slice(&offset.to_le_bytes());
        shard.extend_from_slice(&length.to_le_bytes());
    }
    shard
}
