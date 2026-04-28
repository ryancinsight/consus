//! Synchronous I/O implementations.
//!
//! ## Module Hierarchy
//!
//! ```text
//! sync/
//! ├── cursor    # MemCursor: in-memory Vec<u8> buffer (alloc-gated)
//! ├── slice     # SliceReader: read-only &[u8] adapter (no_std)
//! ├── stream    # StreamReader<R>: cursor-based sequential reader
//! └── file      # std::fs::File impl (std-gated)
//! ```

#[cfg(feature = "alloc")]
pub mod cursor;

pub mod slice;

#[cfg(feature = "alloc")]
pub mod stream;

#[cfg(feature = "std")]
pub mod file;

#[cfg(feature = "mmap")]
pub mod mmap;
