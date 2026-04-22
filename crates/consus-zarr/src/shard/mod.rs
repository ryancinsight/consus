//! Zarr v3 sharded array support.
//!
//! ## Specification
//!
//! Implements the Zarr v3 sharding codec (`sharding_indexed`) per:
//! <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/v1.0.html>
//!
//! ## Shard File Layout
//!
//! ```text
//! [inner_chunk_0_bytes][inner_chunk_1_bytes]...[inner_chunk_N-1_bytes][shard_index]
//! ```
//!
//! The shard index is at the END of the file. Each entry is 16 bytes:
//! `(offset: u64 LE, nbytes: u64 LE)`. Uninitialized chunks use `(u64::MAX, u64::MAX)`.
//! Inner chunk offsets are absolute byte positions from the start of the shard file.

#[cfg(feature = "alloc")]
use alloc::{collections::BTreeMap, format, string::String, vec, vec::Vec};

use crate::codec::{CodecPipeline, default_registry};
use crate::metadata::Codec;
use consus_core::Result;

// ---------------------------------------------------------------------------
// Error types
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
    /// A chunk index entry points outside the valid data section.
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
            } => write!(
                f,
                "shard too small: {} bytes, need at least {} for index",
                shard_size, index_size
            ),
            Self::ChunkOutOfBounds { coords, grid } => write!(
                f,
                "chunk coords {:?} out of bounds for grid {:?}",
                coords, grid
            ),
            Self::InvalidChunkEntry {
                coords,
                offset,
                length,
                shard_size,
            } => write!(
                f,
                "invalid chunk entry {:?}: offset={}, length={}, shard_size={}",
                coords, offset, length, shard_size
            ),
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
                message: format!(
                    "shard too small: {} bytes, need at least {} for index",
                    shard_size, index_size
                ),
            },
            ShardError::ChunkOutOfBounds { coords, grid } => consus_core::Error::InvalidFormat {
                message: format!(
                    "chunk coords {:?} out of bounds for grid {:?}",
                    coords, grid
                ),
            },
            ShardError::InvalidChunkEntry {
                coords,
                offset,
                length,
                shard_size,
            } => consus_core::Error::Corrupted {
                message: format!(
                    "invalid chunk entry {:?}: offset={}, length={}, shard_size={}",
                    coords, offset, length, shard_size
                ),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// ShardingConfig
// ---------------------------------------------------------------------------

/// Configuration extracted from a `sharding_indexed` codec entry in the codec chain.
///
/// ## Relation to `ArrayMetadata`
///
/// - `meta.chunks` is the outer chunk (shard) shape.
/// - `inner_chunk_shape` is the shape of each inner chunk (sub-chunk) within a shard.
/// - `inner_chunks_per_dim[i] = ceil(meta.chunks[i] / inner_chunk_shape[i])`
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct ShardingConfig {
    /// Shape of inner chunks (sub-chunks) within each shard.
    pub inner_chunk_shape: Vec<usize>,
    /// Codec chain applied to each inner chunk for compress/decompress.
    pub inner_codecs: Vec<Codec>,
    /// Codec chain applied to the shard index.
    pub index_codecs: Vec<Codec>,
}

#[cfg(feature = "alloc")]
impl ShardingConfig {
    /// Number of inner chunks along each shard dimension.
    ///
    /// `inner_chunks_per_dim[i] = ceil(shard_shape[i] / inner_chunk_shape[i])`
    #[must_use]
    pub fn inner_chunks_per_dim(&self, shard_shape: &[usize]) -> Vec<usize> {
        shard_shape
            .iter()
            .zip(self.inner_chunk_shape.iter())
            .map(
                |(&shard, &inner)| {
                    if inner == 0 { 0 } else { shard.div_ceil(inner) }
                },
            )
            .collect()
    }

    /// Total number of inner chunks per shard.
    ///
    /// `total = product(inner_chunks_per_dim)`
    #[must_use]
    pub fn total_inner_chunks(&self, shard_shape: &[usize]) -> usize {
        let per_dim = self.inner_chunks_per_dim(shard_shape);
        if per_dim.is_empty() {
            1
        } else {
            per_dim.iter().product()
        }
    }

    /// Size of the shard index in bytes.
    ///
    /// `index_size = total_inner_chunks * 16`  (8-byte offset + 8-byte length per entry)
    #[must_use]
    pub fn index_size_bytes(&self, shard_shape: &[usize]) -> usize {
        self.total_inner_chunks(shard_shape).saturating_mul(16)
    }
}

// ---------------------------------------------------------------------------
// Config extraction
// ---------------------------------------------------------------------------

/// Extract `ShardingConfig` from a codec chain.
///
/// Returns `Some(ShardingConfig)` when the chain contains a `sharding_indexed` codec,
/// `None` otherwise.
#[cfg(feature = "alloc")]
pub fn extract_sharding_config(codecs: &[Codec]) -> Option<ShardingConfig> {
    for codec in codecs {
        if codec.name == "sharding_indexed" {
            let inner_chunk_shape = extract_usize_vec(codec, "chunk_shape")?;
            let inner_codecs = extract_codec_array(codec, "codecs").unwrap_or_default();
            let index_codecs = extract_codec_array(codec, "index_codecs").unwrap_or_default();
            return Some(ShardingConfig {
                inner_chunk_shape,
                inner_codecs,
                index_codecs,
            });
        }
    }
    None
}

#[cfg(feature = "alloc")]
fn extract_usize_vec(codec: &Codec, key: &str) -> Option<Vec<usize>> {
    let val = codec
        .configuration
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())?;
    let json: serde_json::Value = serde_json::from_str(val).ok()?;
    json.as_array()?
        .iter()
        .map(|v| v.as_u64().map(|n| n as usize))
        .collect()
}

#[cfg(feature = "alloc")]
fn extract_codec_array(codec: &Codec, key: &str) -> Option<Vec<Codec>> {
    let val = codec
        .configuration
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())?;
    let json: serde_json::Value = serde_json::from_str(val).ok()?;
    Some(
        json.as_array()?
            .iter()
            .filter_map(|v| {
                let name = v.get("name")?.as_str()?.to_string();
                let config = v
                    .get("configuration")
                    .and_then(|c| c.as_object())
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, v)| {
                                let s = match v {
                                    serde_json::Value::String(s) => s.clone(),
                                    serde_json::Value::Number(n) => n.to_string(),
                                    serde_json::Value::Bool(b) => b.to_string(),
                                    serde_json::Value::Null => String::new(),
                                    _ => v.to_string(),
                                };
                                Some((k.clone(), s))
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                Some(Codec {
                    name,
                    configuration: config,
                })
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Linear index
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Read inner chunk from shard
// ---------------------------------------------------------------------------

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
            .unwrap(),
    );
    let length = u64::from_le_bytes(
        shard_data[entry_offset + 8..entry_offset + 16]
            .try_into()
            .unwrap(),
    );
    // Uninitialized chunk sentinel per the Zarr v3 sharding spec.
    if offset == u64::MAX && length == u64::MAX {
        return Ok(Vec::new());
    }
    let start = offset as usize;
    let end = start.saturating_add(length as usize);
    // Chunk data must lie entirely within the data section (before the index).
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
    let pipeline = CodecPipeline::new(inner_codecs.to_vec());
    pipeline.decompress(compressed, default_registry())
}

// ---------------------------------------------------------------------------
// Write shard
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{ArrayMetadata, ChunkKeyEncoding, FillValue, ZarrVersion};

    fn make_meta(shape: Vec<usize>, chunks: Vec<usize>) -> ArrayMetadata {
        ArrayMetadata {
            version: ZarrVersion::V3,
            shape,
            chunks,
            dtype: alloc::string::String::from("float64"),
            fill_value: FillValue::Float(alloc::string::String::from("0.0")),
            order: 'C',
            codecs: alloc::vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
            dimension_names: None,
        }
    }

    #[test]
    fn sharding_config_total_inner_chunks() {
        let cfg = ShardingConfig {
            inner_chunk_shape: vec![2, 2],
            inner_codecs: vec![],
            index_codecs: vec![],
        };
        assert_eq!(cfg.total_inner_chunks(&[4, 4]), 4);
        assert_eq!(cfg.total_inner_chunks(&[4, 6]), 6);
    }

    #[test]
    fn sharding_config_index_size_bytes() {
        let cfg = ShardingConfig {
            inner_chunk_shape: vec![2, 2],
            inner_codecs: vec![],
            index_codecs: vec![],
        };
        assert_eq!(cfg.index_size_bytes(&[4, 4]), 64);
    }

    #[test]
    fn sharding_config_inner_chunks_per_dim_partial() {
        let cfg = ShardingConfig {
            inner_chunk_shape: vec![2, 3],
            inner_codecs: vec![],
            index_codecs: vec![],
        };
        // ceil(5/2)=3, ceil(7/3)=3
        assert_eq!(cfg.inner_chunks_per_dim(&[5, 7]), vec![3, 3]);
    }

    #[test]
    fn inner_linear_index_correctness() {
        // 2D grid [2, 3]: (1, 2) -> 1*3 + 2 = 5
        assert_eq!(inner_linear_index(&[1, 2], &[2, 3]), 5);
        assert_eq!(inner_linear_index(&[0, 0], &[2, 3]), 0);
        assert_eq!(inner_linear_index(&[1, 0], &[2, 3]), 3);
    }

    #[test]
    fn write_shard_and_read_inner_chunk() {
        let mut inner_chunks = BTreeMap::new();
        inner_chunks.insert(0usize, vec![42u8, 0, 0, 0, 43, 0, 0, 0]);
        inner_chunks.insert(2usize, vec![99u8, 0, 0, 0, 100, 0, 0, 0]);
        let total = 4usize;
        let shard = write_shard(&inner_chunks, total);
        // data section: 8 + 8 = 16 bytes; index: 4 * 16 = 64 bytes -> total 80
        assert_eq!(shard.len(), 16 + 64);
        let c0 = read_inner_chunk_from_shard(&shard, 0, total, &[]).unwrap();
        assert_eq!(c0, vec![42u8, 0, 0, 0, 43, 0, 0, 0]);
        let c1 = read_inner_chunk_from_shard(&shard, 1, total, &[]).unwrap();
        assert!(c1.is_empty(), "uninitialized chunk must return empty vec");
        let c2 = read_inner_chunk_from_shard(&shard, 2, total, &[]).unwrap();
        assert_eq!(c2, vec![99u8, 0, 0, 0, 100, 0, 0, 0]);
        let c3 = read_inner_chunk_from_shard(&shard, 3, total, &[]).unwrap();
        assert!(c3.is_empty(), "uninitialized chunk must return empty vec");
    }

    #[test]
    fn write_shard_all_uninitialized() {
        let inner_chunks = BTreeMap::new();
        let total = 4usize;
        let shard = write_shard(&inner_chunks, total);
        assert_eq!(shard.len(), 64);
        for i in 0..total {
            let c = read_inner_chunk_from_shard(&shard, i, total, &[]).unwrap();
            assert!(c.is_empty());
        }
    }

    #[test]
    fn extract_sharding_config_basic() {
        let codec = Codec {
            name: alloc::string::String::from("sharding_indexed"),
            configuration: vec![
                (
                    alloc::string::String::from("chunk_shape"),
                    alloc::string::String::from("[2,2]"),
                ),
                (
                    alloc::string::String::from("codecs"),
                    alloc::string::String::from(
                        r#"[{"name":"bytes","configuration":{"endian":"little"}}]"#,
                    ),
                ),
                (
                    alloc::string::String::from("index_codecs"),
                    alloc::string::String::from("[]"),
                ),
            ],
        };
        let cfg = extract_sharding_config(&[codec]);
        assert!(cfg.is_some());
        let cfg = cfg.unwrap();
        assert_eq!(cfg.inner_chunk_shape, vec![2, 2]);
        assert_eq!(cfg.inner_codecs.len(), 1);
        assert_eq!(cfg.inner_codecs[0].name, "bytes");
        assert_eq!(cfg.index_codecs.len(), 0);
    }

    #[test]
    fn extract_sharding_config_returns_none_for_non_sharding() {
        let codec = Codec {
            name: alloc::string::String::from("bytes"),
            configuration: vec![],
        };
        assert!(extract_sharding_config(&[codec]).is_none());
        assert!(extract_sharding_config(&[]).is_none());
    }

    #[test]
    fn make_meta_helper_compiles() {
        let meta = make_meta(vec![8, 8], vec![4, 4]);
        assert_eq!(meta.shape, vec![8, 8]);
        assert_eq!(meta.chunks, vec![4, 4]);
    }
}
