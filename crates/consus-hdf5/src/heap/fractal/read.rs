//! Managed-object reads: direct/indirect block walking and row sizing.

use super::{FractalHeapHeader, read_bounded_bytes};
#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use consus_core::{Error, Result};
#[cfg(feature = "alloc")]
use consus_io::ReadAt;

// ---------------------------------------------------------------------------
// Managed object reading
// ---------------------------------------------------------------------------

/// Read a managed object from a fractal heap.
///
/// Locates and reads the object at the given `offset` and `length` (decoded
/// from a [`FractalHeapId::Managed`]).  Handles both the simple case where
/// the root block is a direct block (`root_indirect_rows == 0`) and the
/// general case where the root block is an indirect block.
///
/// # Indirect block traversal
///
/// When `root_indirect_rows > 0` the root block is a fractal heap indirect
/// block (`"FHIB"`).  Each row of the table covers a range of heap addresses;
/// rows 0..max_direct_block_rows contain direct-block children and rows
/// `max_direct_block_rows..nrows` contain indirect-block children.  The
/// function recursively descends until it locates the direct block that
/// contains `offset`, then reads the data from it.
///
/// # Direct Block Overhead
///
/// ```text
/// overhead = 5 (sig + ver)
///          + offset_size               (heap header address)
///          + ⌈max_heap_size_bits / 8⌉  (block offset field)
///          + 4 (if FRHP flags bit 1)   (per-block checksum, stored before data)
/// ```
///
/// # Errors
///
/// - [`Error::InvalidFormat`] if a block has an invalid signature.
/// - [`Error::Overflow`] on address arithmetic overflow.
/// - Propagates I/O errors from `source.read_at`.
#[cfg(feature = "alloc")]
pub fn read_managed_object<R: ReadAt>(
    source: &R,
    header: &FractalHeapHeader,
    offset: u64,
    length: u64,
    ctx: &ParseContext,
) -> Result<Vec<u8>> {
    if header.root_indirect_rows == 0 {
        // Root is a direct block.
        // managed_off (offset) encodes the absolute byte position from block
        // start, so no overhead term is needed here.
        let data_address = header
            .root_block_address
            .checked_add(offset)
            .ok_or(Error::Overflow)?;
        return read_bounded_bytes(
            source,
            data_address,
            length,
            ctx,
            "fractal heap managed object",
        );
    }

    // Root is an indirect block: traverse the table to find the containing
    // direct block.
    find_in_indirect_block(
        source,
        header,
        header.root_block_address,
        header.root_indirect_rows,
        0, // base heap offset for the root indirect block
        offset,
        length,
        ctx,
    )
}

/// Recursively traverse a fractal heap indirect block to locate and read a
/// managed object.
///
/// - `iblock_addr` — file address of the `"FHIB"` indirect block.
/// - `nrows` — number of rows in this indirect block.
/// - `base_offset` — the heap address where this indirect block's table starts.
/// - `target` — managed heap offset of the object to read.
/// - `length` — object byte length.
#[cfg(feature = "alloc")]
fn find_in_indirect_block<R: ReadAt>(
    source: &R,
    header: &FractalHeapHeader,
    iblock_addr: u64,
    nrows: u16,
    base_offset: u64,
    target: u64,
    length: u64,
    ctx: &ParseContext,
) -> Result<Vec<u8>> {
    let o = ctx.offset_bytes();
    let heap_offset_field_bytes = (header.max_heap_size_bits as usize).div_ceil(8);
    // Indirect block overhead: sig(4) + ver(1) + heap_header_addr(O) + block_offset(variable)
    let iblock_overhead = 4 + 1 + o + heap_offset_field_bytes;
    let width = header.table_width as usize;
    let nrows_u = nrows as usize;
    let max_dblock_rows = max_direct_block_rows(header);

    // Per-child entry size in the indirect block:
    // direct-block rows: O bytes address + optional filter info
    // indirect-block rows: O bytes address only
    let filter_extra = if header.io_filter_size > 0 {
        ctx.length_bytes() + 4
    } else {
        0
    };
    let direct_entry_size = o.checked_add(filter_extra).ok_or(Error::Overflow)?;
    let n_direct_children = nrows_u.min(max_dblock_rows);
    let n_indirect_children = nrows_u.saturating_sub(max_dblock_rows);
    let direct_bytes = n_direct_children
        .checked_mul(width)
        .and_then(|children| children.checked_mul(direct_entry_size))
        .ok_or(Error::Overflow)?;
    let indirect_bytes = n_indirect_children
        .checked_mul(width)
        .and_then(|children| children.checked_mul(o))
        .ok_or(Error::Overflow)?;
    let buf_size = iblock_overhead
        .checked_add(direct_bytes)
        .and_then(|size| size.checked_add(indirect_bytes))
        .and_then(|size| size.checked_add(4)) // checksum
        .ok_or(Error::Overflow)?;

    let ibuf = read_bounded_bytes(
        source,
        iblock_addr,
        u64::try_from(buf_size).map_err(|_| Error::Overflow)?,
        ctx,
        "fractal heap indirect block",
    )?;

    if ibuf[0..4] != *b"FHIB" {
        return Err(Error::InvalidFormat {
            message: alloc::format!("invalid indirect block signature at {iblock_addr:#x}"),
        });
    }

    let mut pos = iblock_overhead;
    let mut heap_off = base_offset;

    for row in 0..nrows_u {
        let bsize = row_block_size(row, header)?;

        for _col in 0..width {
            let child_addr = ctx.read_offset(&ibuf[pos..]);
            pos += o;

            if row < max_dblock_rows {
                // Direct block child.
                if header.io_filter_size > 0 {
                    pos += ctx.length_bytes() + 4; // skip filtered_size + filter_mask
                }

                let child_end = heap_off.checked_add(bsize).ok_or(Error::Overflow)?;
                if target >= heap_off && target < child_end {
                    let local_offset = target - heap_off;
                    // managed_off (target) encodes the absolute byte offset
                    // from block start — no overhead term in the address.
                    let data_addr = child_addr
                        .checked_add(local_offset)
                        .ok_or(Error::Overflow)?;
                    return read_bounded_bytes(
                        source,
                        data_addr,
                        length,
                        ctx,
                        "fractal heap managed object",
                    );
                }
                heap_off = child_end;
            } else {
                // Indirect block child: compute its total heap coverage and
                // recurse if the target falls within it.
                let child_nrows = indirect_child_nrows(row, header)?;
                let child_coverage = indirect_block_coverage(child_nrows, header)?;
                let child_end = heap_off
                    .checked_add(child_coverage)
                    .ok_or(Error::Overflow)?;
                if target >= heap_off && target < child_end {
                    return find_in_indirect_block(
                        source,
                        header,
                        child_addr,
                        child_nrows,
                        heap_off,
                        target,
                        length,
                        ctx,
                    );
                }
                heap_off = child_end;
            }
        }
    }

    Err(Error::InvalidFormat {
        message: alloc::format!(
            "managed offset {target} not found in indirect block at {iblock_addr:#x} (nrows={nrows})"
        ),
    })
}

/// Block size (in bytes) for row `row` of the fractal heap doubling table.
///
/// Rows `0..starting_rows` all have `starting_block_size`; row
/// `starting_rows + k` has `starting_block_size << (k + 1)`.
#[cfg(feature = "alloc")]
fn row_block_size(row: usize, header: &FractalHeapHeader) -> Result<u64> {
    let start = header.starting_rows as usize;
    if row < start {
        Ok(header.starting_block_size)
    } else {
        let shift = row - start;
        if shift >= 64 {
            Err(Error::ResourceLimit {
                what: "fractal heap block row",
                requested: shift as u64,
                limit: 63,
            })
        } else {
            header
                .starting_block_size
                .checked_shl(shift as u32)
                .ok_or(Error::Overflow)
        }
    }
}

/// Maximum number of direct-block rows in any indirect block.
///
/// `max_dblock_rows = log₂(max_direct_block_size) − log₂(starting_block_size) + starting_rows + 1`
#[cfg(feature = "alloc")]
fn max_direct_block_rows(header: &FractalHeapHeader) -> usize {
    if header.starting_block_size == 0 || header.max_direct_block_size == 0 {
        return 1;
    }
    let log2_max = (u64::BITS - 1 - header.max_direct_block_size.leading_zeros()) as usize;
    let log2_start = (u64::BITS - 1 - header.starting_block_size.leading_zeros()) as usize;
    (log2_max.saturating_sub(log2_start)) + header.starting_rows as usize + 1
}

/// Number of rows in a child indirect block at parent row `parent_row`.
///
/// Child indirect blocks that appear at rows ≥ `max_dblock_rows` of the
/// parent each have `max_dblock_rows` rows.
#[cfg(feature = "alloc")]
fn indirect_child_nrows(_parent_row: usize, header: &FractalHeapHeader) -> Result<u16> {
    u16::try_from(max_direct_block_rows(header)).map_err(|_| Error::ResourceLimit {
        what: "fractal heap indirect row count",
        requested: max_direct_block_rows(header) as u64,
        limit: u16::MAX as u64,
    })
}

/// Total heap-address-space coverage of an indirect block with `nrows` rows.
///
/// Sums `table_width * row_block_size(r)` for each row `r`.
#[cfg(feature = "alloc")]
fn indirect_block_coverage(nrows: u16, header: &FractalHeapHeader) -> Result<u64> {
    let width = header.table_width as u64;
    (0..nrows as usize).try_fold(0u64, |coverage, row| {
        let row_bytes = width
            .checked_mul(row_block_size(row, header)?)
            .ok_or(Error::Overflow)?;
        coverage.checked_add(row_bytes).ok_or(Error::Overflow)
    })
}
