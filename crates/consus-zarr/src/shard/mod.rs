//! Zarr v3 sharded array support.
//!
//! ## Specification
//!
//! This module implements the Zarr v3 sharding codec specification, enabling
//! efficient storage of many small chunks in a single shard file with an
//! embedded index.
//!
//! ## Invariants
//!
//! - Shard files contain an index followed by chunk data.
//! - The index maps chunk coordinates to byte offsets and lengths.
//! - Uninitialized chunks have offset=0 and length=0 in the index.
//! - All offsets are aligned according to the shard configuration.

#[cfg(feature = "alloc")]
use alloc::{collections::BTreeMap, string::String, vec::Vec};

use crate::codec::{CodecPipeline, default_registry};
use crate::metadata::ArrayMetadata;
use crate::metadata::Codec;
use crate::store::Store;
use consus_core::Result;

// ---------------------------------------------------------------------------
// Error types for shard operations
// ---------------------------------------------------------------------------

/// Errors that can occur during shard operations.
#[derive(Debug, Clone)]
pub enum ShardError {
    /// The shard file is too small to contain the index.
    ShardTooSmall { shard_size: u64, index_size: u64 },
    /// The requested chunk coordinates are out of bounds.
    ChunkOutOfBounds {
        coords: Vec<usize>,
        grid: Vec<usize>,
    },
    /// The chunk entry in the index is invalid.
    InvalidChunkEntry {
        coords: Vec<usize>,
        offset: u64,
        length: u64,
        shard_size: u64,
    },
}

impl core::fmt::Display for ShardError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShardTooSmall {
                shard_size,
                index_size,
            } => {
                write!(
                    f,
                    "shard too small: {} bytes, need at least {} bytes for index",
                    shard_size, index_size
                )
            }
            Self::ChunkOutOfBounds { coords, grid } => {
                write!(
                    f,
                    "chunk coordinates {:?} out of bounds for grid {:?}",
                    coords, grid
                )
            }
            Self::InvalidChunkEntry {
                coords,
                offset,
                length,
                shard_size,
            } => {
                write!(
                    f,
                    "invalid chunk entry at {:?}: offset={}, length={}, shard_size={}",
                    coords, offset, length, shard_size
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ShardError {}

impl From<ShardError> for consus_core::Error {
    fn from(err: ShardError) -> Self {
        match err {
            ShardError::ShardTooSmall {
                shard_size,
                index_size,
            } => consus_core::Error::InvalidFormat {
                message: String::from(format!(
                    "shard too small: {} bytes, need at least {} bytes for index",
                    shard_size, index_size
                )),
            },
            ShardError::ChunkOutOfBounds { coords, grid } => consus_core::Error::InvalidFormat {
                message: String::from(format!(
                    "chunk coordinates {:?} out of bounds for grid {:?}",
                    coords, grid
                )),
            },
            ShardError::InvalidChunkEntry {
                coords,
                offset,
                length,
                shard_size,
            } => consus_core::Error::Corrupted {
                message: String::from(format!(
                    "invalid chunk entry at {:?}: offset={}, length={}, shard_size={}",
                    coords, offset, length, shard_size
                )),
            },
        }
    }
}

/// Shard configuration parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardConfig {
    /// Index chunk grid dimensions.
    pub index_chunk_grid: Vec<usize>,
    /// Codec for index compression (optional).
    pub index_codec: Option<Codec>,
    /// Chunk alignment requirement in bytes.
    pub chunk_alignment: usize,
}

impl ShardConfig {
    /// Create a new shard configuration.
    #[must_use]
    pub fn new(
        index_chunk_grid: Vec<usize>,
        index_codec: Option<Codec>,
        chunk_alignment: usize,
    ) -> Self {
        Self {
            index_chunk_grid,
            index_codec,
            chunk_alignment,
        }
    }

    /// Compute the index chunk size in bytes.
    #[must_use]
    pub fn index_chunk_size(&self, meta: &ArrayMetadata) -> Option<u64> {
        let chunks_per_shard = self
            .index_chunk_grid
            .iter()
            .zip(meta.chunks.iter())
            .map(|(idx, grid)| idx / grid)
            .fold(1usize, |acc, x| acc.saturating_mul(x));

        // Each chunk entry is 16 bytes (8-byte offset + 8-byte length)
        let index_bytes = chunks_per_shard.saturating_mul(16);
        Some(index_bytes as u64)
    }
}

/// Reader for shard index data.
#[derive(Debug, Clone)]
pub struct ShardIndexReader<'a> {
    data: &'a [u8],
    chunk_grid: &'a [usize],
}

impl<'a> ShardIndexReader<'a> {
    /// Create a new shard index reader.
    #[must_use]
    pub fn new(data: &'a [u8], chunk_grid: &'a [usize]) -> Self {
        Self { data, chunk_grid }
    }

    /// Get a chunk entry from the index.
    /// Returns (offset, length) for the chunk, or None if out of bounds.
    #[must_use]
    pub fn get_chunk_entry(&self, coords: &[usize]) -> Option<(u64, u64)> {
        if coords.len() != self.chunk_grid.len() {
            return None;
        }

        // Compute linear index
        let mut linear_idx = 0usize;
        for (i, (coord, grid)) in coords.iter().zip(self.chunk_grid.iter()).enumerate() {
            if coord >= grid {
                return None;
            }
            if i > 0 {
                linear_idx = linear_idx.saturating_mul(*grid);
            }
            linear_idx = linear_idx.saturating_add(*coord);
        }

        // Each entry is 16 bytes
        let entry_offset = linear_idx.saturating_mul(16);
        if entry_offset.saturating_add(16) > self.data.len() {
            return None;
        }

        // Read offset and length (little-endian)
        let offset = u64::from_le_bytes(
            self.data[entry_offset..entry_offset.saturating_add(8)]
                .try_into()
                .ok()?,
        );
        let length = u64::from_le_bytes(
            self.data[entry_offset.saturating_add(8)..entry_offset.saturating_add(16)]
                .try_into()
                .ok()?,
        );

        Some((offset, length))
    }
}

/// Writer for shard index data.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ShardIndexWriter {
    entries: BTreeMap<usize, (u64, u64)>,
    total_chunks: usize,
}

#[cfg(feature = "alloc")]
impl ShardIndexWriter {
    /// Create a new shard index writer.
    #[must_use]
    pub fn new(total_chunks: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            total_chunks,
        }
    }

    /// Set a chunk entry in the index.
    pub fn set_chunk_entry(&mut self, linear_idx: usize, offset: u64, length: u64) {
        if linear_idx < self.total_chunks {
            self.entries.insert(linear_idx, (offset, length));
        }
    }

    /// Serialize the index to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = vec![0u8; self.total_chunks.saturating_mul(16)];
        for (&idx, &(offset, length)) in &self.entries {
            let entry_offset = idx.saturating_mul(16);
            if entry_offset.saturating_add(16) <= data.len() {
                data[entry_offset..entry_offset.saturating_add(8)]
                    .copy_from_slice(&offset.to_le_bytes());
                data[entry_offset.saturating_add(8)..entry_offset.saturating_add(16)]
                    .copy_from_slice(&length.to_le_bytes());
            }
        }
        data
    }
}

/// Read a single chunk from a sharded array.
///
/// ## Parameters
///
/// - `store`: The key-value store
/// - `shard_key`: Key for the shard file
/// - `coords`: Chunk coordinates within the shard
/// - `meta`: Array metadata
/// - `shard_config`: Shard configuration
///
/// ## Returns
///
/// The decompressed chunk data, or an empty vector for uninitialized chunks.
pub fn read_sharded_chunk(
    store: &dyn Store,
    shard_key: &str,
    coords: &[usize],
    meta: &ArrayMetadata,
    shard_config: &ShardConfig,
) -> Result<Vec<u8>> {
    // Read the full shard
    let shard_data = store.get(shard_key)?;
    let shard_size = shard_data.len() as u64;

    // Determine index size
    let index_size =
        shard_config
            .index_chunk_size(meta)
            .ok_or_else(|| consus_core::Error::InvalidFormat {
                message: String::from("invalid shard configuration"),
            })?;

    if shard_size < index_size {
        return Err(ShardError::ShardTooSmall {
            shard_size,
            index_size,
        }
        .into());
    }

    // Decompress index if needed
    let index_bytes = if let Some(ref codec) = shard_config.index_codec {
        let index_compressed = &shard_data[..index_size as usize];
        let pipeline = CodecPipeline::single(codec.clone());
        pipeline.decompress(index_compressed, default_registry())?
    } else {
        shard_data[..index_size as usize].to_vec()
    };

    // Parse index
    let reader = ShardIndexReader::new(&index_bytes, &meta.chunks);
    let (offset, length) =
        reader
            .get_chunk_entry(coords)
            .ok_or_else(|| ShardError::ChunkOutOfBounds {
                coords: coords.to_vec(),
                grid: meta.chunks.clone(),
            })?;

    // Uninitialized chunk
    if offset == 0 && length == 0 {
        return Ok(Vec::new());
    }

    // Validate bounds
    if offset.saturating_add(length) > shard_size {
        return Err(ShardError::InvalidChunkEntry {
            coords: coords.to_vec(),
            offset,
            length,
            shard_size,
        }
        .into());
    }

    // Extract chunk data
    let start = offset as usize;
    let end = start.saturating_add(length as usize);
    Ok(shard_data[start..end].to_vec())
}

/// Write a sharded array to a store.
///
/// This function writes the complete shard file containing all chunks,
/// building the index and writing data chunks with appropriate alignment.
///
/// ## Parameters
///
/// - `store`: The key-value store
/// - `shard_key`: Key for the shard file
/// - `chunks`: Map from chunk coordinates to chunk data
/// - `meta`: Array metadata
/// - `shard_config`: Shard configuration
#[cfg(feature = "alloc")]
pub fn write_sharded_array(
    store: &mut dyn Store,
    shard_key: &str,
    chunks: &BTreeMap<Vec<usize>, Vec<u8>>,
    meta: &ArrayMetadata,
    shard_config: &ShardConfig,
) -> Result<()> {
    let index_size =
        shard_config
            .index_chunk_size(meta)
            .ok_or_else(|| consus_core::Error::InvalidFormat {
                message: String::from("invalid shard configuration"),
            })? as usize;

    // Build index
    let total_chunks = meta
        .chunks
        .iter()
        .fold(1usize, |acc, &x| acc.saturating_mul(x));
    let mut index_writer = ShardIndexWriter::new(total_chunks);

    // Build chunk data section
    let mut data_section = Vec::new();
    let mut current_offset = index_size as u64;

    for (coords, chunk_data) in chunks {
        // Compute linear index
        let mut linear_idx = 0usize;
        for (i, coord) in coords.iter().enumerate() {
            if i > 0 {
                linear_idx = linear_idx.saturating_mul(meta.chunks[i]);
            }
            linear_idx = linear_idx.saturating_add(*coord);
        }

        // Align offset
        let alignment = shard_config.chunk_alignment;
        let misalignment = current_offset % alignment as u64;
        if misalignment != 0 {
            let padding = (alignment as u64).saturating_sub(misalignment);
            current_offset = current_offset.saturating_add(padding);
            data_section.extend(alloc::vec![0u8; padding as usize]);
        }

        // Record entry in index
        let length = chunk_data.len() as u64;
        index_writer.set_chunk_entry(linear_idx, current_offset, length);

        // Append chunk data
        data_section.extend(chunk_data);
        current_offset = current_offset.saturating_add(length);
    }

    // Serialize index
    let index_data = index_writer.to_bytes();

    // Compress index if configured
    let index_bytes = if let Some(ref codec) = shard_config.index_codec {
        let pipeline = CodecPipeline::single(codec.clone());
        pipeline.compress(&index_data, default_registry())?
    } else {
        index_data
    };

    // Build final shard file
    let mut shard_file = Vec::with_capacity(index_bytes.len().saturating_add(data_section.len()));
    shard_file.extend(&index_bytes);
    shard_file.extend(&data_section);

    // Write to store
    store.set(shard_key, &shard_file)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{ChunkKeyEncoding, FillValue, ZarrVersion};

    #[test]
    fn shard_config_index_size() {
        let config = ShardConfig::new(vec![4, 4], None, 1);
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: vec![8, 8],
            chunks: vec![2, 2],
            dtype: String::from("float64"),
            fill_value: FillValue::Float(alloc::string::String::from("0.0")),
            order: 'C',
            codecs: vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
        };

        let size = config.index_chunk_size(&meta);
        assert!(size.is_some());
        // Current layout uses 64-byte index chunks for this configuration.
        assert_eq!(size.unwrap(), 64);
    }

    #[test]
    fn shard_index_reader_get_entry() {
        let chunk_grid = vec![2, 2];
        let mut data = vec![0u8; 64];

        // Set entry at (1, 0) -> linear index 2 in row-major order.
        let offset = 100u64;
        let length = 50u64;
        data[32..40].copy_from_slice(&offset.to_le_bytes());
        data[40..48].copy_from_slice(&length.to_le_bytes());

        let reader = ShardIndexReader::new(&data, &chunk_grid);

        // Get entry at (1, 0)
        let entry = reader.get_chunk_entry(&[1, 0]);
        assert!(entry.is_some());
        let (off, len) = entry.unwrap();
        assert_eq!(off, offset);
        assert_eq!(len, length);

        // Out of bounds
        let entry = reader.get_chunk_entry(&[0, 2]);
        assert!(entry.is_none());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn shard_index_writer_roundtrip() {
        let total_chunks = 4;
        let mut writer = ShardIndexWriter::new(total_chunks);
        writer.set_chunk_entry(0, 100, 200);
        writer.set_chunk_entry(2, 300, 400);

        let data = writer.to_bytes();
        assert_eq!(data.len(), 64);

        // Read back
        let chunk_grid = vec![2, 2];
        let reader = ShardIndexReader::new(&data, &chunk_grid);

        let (off1, len1) = reader.get_chunk_entry(&[0, 0]).unwrap();
        assert_eq!(off1, 100);
        assert_eq!(len1, 200);

        let (off2, len2) = reader.get_chunk_entry(&[1, 0]).unwrap();
        assert_eq!(off2, 300);
        assert_eq!(len2, 400);
    }
}
