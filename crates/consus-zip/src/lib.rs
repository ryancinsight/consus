#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

mod error;
mod format;
mod read;
mod write;

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    reason = "tests patch fixed header offsets of archives they just built"
)]
mod tests;

pub use error::{Error, Result};
pub use format::Method;
pub use read::{Entry, ZipReader};
pub use write::ZipWriter;
