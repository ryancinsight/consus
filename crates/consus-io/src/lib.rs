//! # consus-io
//!
//! Sync and async I/O abstractions for the Consus storage library.
//!
//! ## Architecture
//!
//! This crate decouples format logic from physical I/O by defining
//! position-aware read/write traits. Format backends operate on these
//! traits rather than `std::fs::File` directly, enabling:
//!
//! - In-memory buffers for testing
//! - Memory-mapped I/O
//! - Object store backends (S3, GCS) via async adapters
//! - `no_std` environments with custom I/O
//!
//! ## Module Hierarchy
//!
//! ```text
//! consus-io
//! ├── io/
//! │   ├── traits/      # ReadAt, WriteAt, Length, Truncate, Seekable, SeekFrom, RandomAccess
//! │   ├── sync/        # Synchronous positioned I/O implementations
//! │   │   ├── cursor   # In-memory I/O source (MemCursor)
//! │   │   ├── slice    # Read-only &[u8] adapter (SliceReader)
//! │   │   ├── stream   # Cursor-based sequential reader (StreamReader)
//! │   │   ├── file     # std::fs::File positioned I/O implementation
//! │   │   └── mmap     # Read-only memory-mapped file reader (MmapReader)
//! │   └── async_io/    # Asynchronous positioned I/O (feature-gated)
//! ```
//!
//! ### Trait Hierarchy
//!
//! ```text
//! ReadAt ──┐
//!          ├── RandomAccess (read + write + length + truncate)
//! WriteAt ─┘
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod io;

// ── Convenience re-exports at crate root ───────────────────────────

// Sync traits
pub use io::traits::{Length, RandomAccess, ReadAt, SeekFrom, Seekable, Truncate, WriteAt};

// Sync implementations
#[cfg(feature = "alloc")]
pub use io::sync::cursor::MemCursor;
#[cfg(feature = "mmap")]
pub use io::sync::mmap::MmapReader;
pub use io::sync::slice::SliceReader;
#[cfg(feature = "alloc")]
pub use io::sync::stream::StreamReader;

// Async trait re-exports
#[cfg(feature = "async-io")]
pub use io::traits::{
    AsyncLength, AsyncRandomAccess, AsyncReadAt, AsyncSeekable, AsyncTruncate, AsyncWriteAt,
};

// Async implementation re-exports
#[cfg(feature = "async-io")]
pub use io::async_io::AsyncMemCursor;

#[cfg(all(feature = "async-io", feature = "s3"))]
pub use io::async_io::s3::S3Reader;
