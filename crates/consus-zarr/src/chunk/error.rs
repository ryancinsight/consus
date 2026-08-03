use core::fmt;

#[cfg(feature = "alloc")]
use alloc::string::{String, ToString};

/// Errors that can occur during chunk operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkError {
    /// The requested chunk is out of bounds of the array.
    ChunkOutOfBounds,
    /// The chunk has not been initialized (no data written yet).
    Uninitialized,
    /// Decompression of chunk data failed.
    DecompressFailed,
    /// Compression of chunk data failed.
    CompressFailed,
    /// Unexpected length when reading or writing data.
    UnexpectedLength,
    /// Store operation failed.
    StoreError(String),
}

impl fmt::Display for ChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkError::ChunkOutOfBounds => write!(f, "chunk is out of bounds"),
            ChunkError::Uninitialized => write!(f, "chunk is uninitialized"),
            ChunkError::DecompressFailed => write!(f, "failed to decompress chunk"),
            ChunkError::CompressFailed => write!(f, "failed to compress chunk"),
            ChunkError::UnexpectedLength => write!(f, "unexpected data length"),
            ChunkError::StoreError(msg) => write!(f, "store error: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ChunkError {}

#[cfg(feature = "alloc")]
impl From<ChunkError> for consus_core::Error {
    fn from(err: ChunkError) -> Self {
        consus_core::Error::InvalidFormat {
            message: err.to_string(),
        }
    }
}
