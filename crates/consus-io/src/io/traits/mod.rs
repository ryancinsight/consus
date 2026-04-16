//! Abstract I/O trait definitions forming the Dependency Inversion boundary.
//!
//! ## Architecture
//!
//! Format backends and I/O implementations depend on these traits,
//! never on concrete types. This enables:
//!
//! - In-memory buffers for testing
//! - Memory-mapped I/O for performance
//! - Object store backends (S3, GCS) via async adapters
//! - `no_std` environments with custom I/O providers
//!
//! ## Trait Hierarchy
//!
//! ```text
//! ReadAt            (positioned read, stateless, concurrent-safe)
//! WriteAt           (positioned write)
//! Length            (byte length query)
//! Truncate          (resize source/sink)
//! SeekFrom          (seek position enum, no_std)
//! Seekable          (cursor-based seeking)
//! RandomAccess      = ReadAt + WriteAt + Length + Truncate
//!
//! [async-io feature]
//! AsyncReadAt       (async positioned read)
//! AsyncWriteAt      (async positioned write)
//! AsyncLength       (async length query)
//! AsyncTruncate     (async resize)
//! AsyncSeekable     (async cursor seeking)
//! AsyncRandomAccess = AsyncReadAt + AsyncWriteAt + AsyncLength + AsyncTruncate
//! ```

pub mod combined;
pub mod extent;
pub mod read;
pub mod seek;
pub mod write;

#[cfg(feature = "async-io")]
pub mod async_traits;

// ── Sync trait re-exports ──────────────────────────────────────────

pub use combined::RandomAccess;
pub use extent::{Length, Truncate};
pub use read::ReadAt;
pub use seek::{SeekFrom, Seekable};
pub use write::WriteAt;

// ── Async trait re-exports ─────────────────────────────────────────

#[cfg(feature = "async-io")]
pub use async_traits::{
    AsyncLength, AsyncRandomAccess, AsyncReadAt, AsyncSeekable, AsyncTruncate, AsyncWriteAt,
};
