#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod format;
mod read;
mod write;

#[cfg(test)]
mod tests;

pub use error::{Error, Result};
pub use format::Method;
pub use read::{Entry, ZipReader};
pub use write::ZipWriter;
