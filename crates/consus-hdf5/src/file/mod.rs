//! HDF5 file-level API.
//!
//! ## Design
//!
//! `Hdf5File` is the entry point for reading and writing HDF5 files.
//! It owns a reference to the I/O source and the parsed superblock,
//! providing navigation through the object hierarchy.
//!
//! ### Lifecycle
//!
//! 1. Open: locate superblock, parse root group
//! 2. Navigate: traverse groups via B-tree/heap
//! 3. Read: resolve dataset metadata, read raw data through selection
//! 4. Close: flush and release resources

#[cfg(all(feature = "async", feature = "alloc"))]
pub mod async_file;
#[cfg(all(feature = "async", feature = "alloc"))]
pub mod async_reader;
pub mod reader;
#[cfg(feature = "alloc")]
pub mod writer;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use consus_core::Result;
use consus_io::ReadAt;

use crate::address::ParseContext;
use crate::superblock::Superblock;

#[cfg(feature = "alloc")]
use consus_core::Error;

/// An open HDF5 file for reading.
///
/// Parameterized over the I/O source to support both file and in-memory backends.
pub struct Hdf5File<R: ReadAt> {
    /// Underlying I/O source.
    source: R,
    /// Parsed superblock.
    superblock: Superblock,
    /// Parsing context derived from the superblock.
    ctx: ParseContext,
}

/// Calculate a dataset payload size without trusting a file-supplied shape.
///
/// `Shape::num_elements` is intentionally a plain value operation for
/// already-validated in-memory shapes. HDF5 shapes cross an untrusted file
/// boundary, so this path must detect multiplication overflow and apply
/// the parser's byte and element ceilings before allocating output.
#[cfg(feature = "alloc")]
pub(crate) fn checked_dataset_byte_count(
    ctx: &ParseContext,
    shape: &consus_core::Shape,
    element_size: usize,
) -> Result<usize> {
    checked_element_payload_byte_count(
        ctx,
        &shape.current_dims(),
        element_size,
        "dataset element count",
    )
}

/// Calculate a bounded byte payload from file-supplied element extents.
#[cfg(feature = "alloc")]
pub(crate) fn checked_element_payload_byte_count(
    ctx: &ParseContext,
    extents: &[usize],
    element_size: usize,
    what: &'static str,
) -> Result<usize> {
    let element_count = extents
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or(Error::Overflow)?;
    let bounded_count = ctx.budget.checked_elements(
        u64::try_from(element_count).map_err(|_| Error::Overflow)?,
        element_size,
        what,
    )?;
    bounded_count
        .checked_mul(element_size)
        .ok_or(Error::Overflow)
}

/// Calculate the bounded record region of a v1 raw-data chunk B-tree leaf.
#[cfg(feature = "alloc")]
pub(crate) fn checked_v1_chunk_btree_data_size(
    ctx: &ParseContext,
    rank: usize,
    entries_used: u16,
) -> Result<usize> {
    let key_size = 4usize
        .checked_add(4)
        .and_then(|size| {
            rank.checked_add(1)
                .and_then(|dimensions| dimensions.checked_mul(8))
                .and_then(|dimensions| size.checked_add(dimensions))
        })
        .ok_or(Error::Overflow)?;
    let pair_size = ctx
        .offset_bytes()
        .checked_add(key_size)
        .ok_or(Error::Overflow)?;
    let entry_count = usize::from(entries_used);
    let data_size = entry_count
        .checked_mul(pair_size)
        .and_then(|size| size.checked_add(key_size))
        .ok_or(Error::Overflow)?;
    ctx.budget.checked_bytes(
        u64::try_from(data_size).map_err(|_| Error::Overflow)?,
        "HDF5 v1 chunk B-tree records",
    )
}

mod access;
mod chunk_read;
mod dataset_io;
mod navigation;

#[derive(Debug)]
pub(crate) struct ChunkIndexEntry {
    pub(crate) dimension_offsets: Vec<u64>,
    pub(crate) filter_mask: u32,
    pub(crate) chunk_size: u32,
    pub(crate) chunk_address: u64,
}

#[cfg(feature = "alloc")]
fn linear_index(coords: &[usize], dims: &[usize]) -> usize {
    let mut index = 0usize;
    for (&coord, &dim) in coords.iter().zip(dims.iter()) {
        index = index * dim + coord;
    }
    index
}

#[cfg(test)]
mod tests;
