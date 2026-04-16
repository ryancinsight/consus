//! Core I/O module bridging trait definitions and implementations.
//!
//! ## Module Hierarchy
//!
//! ```text
//! io/
//! ├── traits/      # ReadAt, WriteAt, Length, Truncate, Seekable, SeekFrom, RandomAccess
//! ├── sync/        # Synchronous implementations (MemCursor, SliceReader, StreamReader, File)
//! └── async_io/    # Asynchronous implementations (feature-gated)
//! ```

pub mod traits;

pub mod sync;

#[cfg(feature = "async-io")]
pub mod async_io;
