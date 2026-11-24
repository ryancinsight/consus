use core::fmt;

#[cfg(feature = "alloc")]
use alloc::string::{String, ToString};

/// Errors that can occur during chunk operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChunkError {
    /// The requested chunk is out of bounds of the array.
    ChunkOutOfBounds,
    /// The chunk has not been initialized (no data written yet).
    Uninitialized,
    /// Decompression of chunk data failed.
    DecompressFailed,
    /// Compression of chunk data failed.
    CompressFailed,
    /// A codec in the array metadata declares a configuration it cannot
    /// honor, such as a `gzip` or `zlib` level outside `0..=9`.
    ///
    /// Carries the codec's own description of the rejected value.
    InvalidCodecConfiguration(String),
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
            ChunkError::InvalidCodecConfiguration(msg) => {
                write!(f, "invalid codec configuration: {}", msg)
            }
            ChunkError::UnexpectedLength => write!(f, "unexpected data length"),
            ChunkError::StoreError(msg) => write!(f, "store error: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ChunkError {}

#[cfg(all(feature = "std", feature = "alloc"))]
impl ChunkError {
    /// Classify a codec-pipeline compression failure.
    ///
    /// [`consus_core::Error::InvalidFormat`] is the pipeline's report that the
    /// declared codec configuration is unusable, so the message is kept;
    /// every other failure is an encoder failure.
    pub(crate) fn from_compression(error: consus_core::Error) -> Self {
        match error {
            consus_core::Error::InvalidFormat { message } => {
                Self::InvalidCodecConfiguration(message)
            }
            _ => Self::CompressFailed,
        }
    }
}

#[cfg(feature = "alloc")]
impl From<ChunkError> for consus_core::Error {
    fn from(err: ChunkError) -> Self {
        consus_core::Error::InvalidFormat {
            message: err.to_string(),
        }
    }
}
