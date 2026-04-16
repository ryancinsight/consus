//! Low-level binary serialization primitives.
//!
//! This module is the SSOT for binary encoding/decoding utilities
//! shared across format backends. No other crate may duplicate
//! these implementations.
//!
//! ## Hierarchy
//!
//! ```text
//! serialization/
//! └── primitives   # LEB128, NUL-terminated strings, alignment
//! ```

pub mod primitives;

pub use primitives::*;
