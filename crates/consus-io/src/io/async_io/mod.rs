//! Asynchronous I/O implementations.
//!
//! This module is available only when the `async-io` feature is enabled.
//!
//! ## Module Hierarchy
//!
//! ```text
//! async_io/
//! ├── cursor    # AsyncMemCursor: async in-memory buffer
//! └── s3        # S3Reader: async S3 object store backend
//! ```

pub mod cursor;
pub use cursor::AsyncMemCursor;

#[cfg(feature = "s3")]
pub mod s3;
#[cfg(feature = "s3")]
pub use s3::S3Reader;
