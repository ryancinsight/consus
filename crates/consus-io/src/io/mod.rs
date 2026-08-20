//! Core I/O module bridging trait definitions and implementations.
//!
//! ## Module Hierarchy
//!
//! ```text
//! io/
//! ├── traits/      # ReadAt, WriteAt, Length, Truncate, Seekable, SeekFrom, RandomAccess
//! ├── sync/        # Synchronous implementations (MemCursor, SliceReader, StreamReader, File)
//! ```

pub mod traits;

pub mod sync;
