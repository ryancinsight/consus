#[cfg(feature = "alloc")]
use alloc::{format, vec::Vec};

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
