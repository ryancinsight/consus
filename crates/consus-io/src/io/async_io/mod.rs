//! Asynchronous I/O implementations.
//!
//! This module is available only when the `async-io` feature is enabled.
//!
//! ## Module Hierarchy
//!
//! ```text
//! async_io/
//! └── cursor    # AsyncMemCursor: async in-memory buffer
//! ```

pub mod cursor;

pub use cursor::AsyncMemCursor;
