//! MAT v4 file reader.
//!
//! Parses a complete MAT v4 byte stream into a vector of
//! `(name, MatArray)` pairs.

pub mod element;
pub mod header;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use crate::error::MatError;
use crate::model::MatArray;

/// Parse all variables from a MAT v4 byte stream.
///
/// MAT v4 files have no file-level header. The stream is a sequence of
/// variable records, each starting with a 20-byte header followed by the
/// variable name and data bytes.
#[cfg(feature = "alloc")]
pub fn read_mat_v4(data: &[u8]) -> Result<Vec<(String, MatArray)>, MatError> {
    let mut pos = 0usize;
    let mut variables = Vec::new();

    while pos < data.len() {
        match element::read_v4_variable(data, &mut pos)? {
            Some(var) => variables.push(var),
            None => break,
        }
    }

    Ok(variables)
}
