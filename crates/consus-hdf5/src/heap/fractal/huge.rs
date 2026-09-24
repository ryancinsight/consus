//! Huge-object reads, bounded reads, and little-endian decode helpers.

use super::FractalHeapHeader;
#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use consus_core::{Error, Result};
#[cfg(feature = "alloc")]
use consus_io::ReadAt;

// ---------------------------------------------------------------------------
// HUGE object reading
// ---------------------------------------------------------------------------

/// Read a HUGE object from a fractal heap.
///
/// HUGE objects are stored outside the managed address space and indexed
/// by a v2 B-tree. The `btree_key` from [`FractalHeapId::Huge`][super::FractalHeapId::Huge] is either:
/// - A **direct file address** (when header flag bit 0 is SET), or
/// - A **B-tree key** to search (when header flag bit 0 is CLEAR).
///
/// ## Arguments
///
/// - `source`: I/O source.
/// - `header`: Parsed fractal heap header.
/// - `btree_key`: The key from `FractalHeapId::Huge`.
/// - `ctx`: Parse context.
///
/// ## Returns
///
/// The raw object bytes, or an error on I/O or parse failure.
/// Filtered huge objects (flag bit 2 set) currently return
/// [`Error::UnsupportedFeature`] — I/O filter pipeline integration
/// is deferred.
///
/// ## Errors
///
/// - [`Error::UnsupportedFeature`] if flag bit 2 (huge-objects-filtered)
///   is set (filtered data requires I/O filter pipeline).
/// - Propagates errors from B-tree search or direct address read.
#[cfg(feature = "alloc")]
pub fn read_huge_object<R: ReadAt>(
    source: &R,
    header: &FractalHeapHeader,
    btree_key: u64,
    ctx: &ParseContext,
) -> Result<Vec<u8>> {
    use crate::btree::v2::find_huge_object_record;

    // Case 1: Direct address mode (flag bit 0 set)
    if header.flags & 0x01 != 0 {
        // btree_key IS the direct file address of the object.
        // We still need to know the length. For direct-mode HUGE objects,
        // the length is stored separately in the B-tree record — we must
        // search the B-tree to get the length even in direct mode.
        if header.huge_object_btree_address == crate::constants::UNDEFINED_ADDRESS {
            return Err(Error::InvalidFormat {
                message: String::from(
                    "huge object direct-address mode but B-tree address is undefined",
                ),
            });
        }
        let btree_header =
            crate::btree::v2::BTreeV2Header::parse(source, header.huge_object_btree_address, ctx)?;
        let location = find_huge_object_record(
            source,
            header.huge_object_btree_address,
            &btree_header,
            btree_key,
            ctx,
        )?
        .ok_or_else(|| Error::InvalidFormat {
            message: String::from("huge object record not found in B-tree"),
        })?;

        return read_bounded_bytes(
            source,
            location.address,
            location.length,
            ctx,
            "fractal heap huge object",
        );
    }

    // Case 2: B-tree lookup mode (flag bit 0 clear)
    // btree_key is the search key in the HUGE object B-tree.
    if header.huge_object_btree_address == crate::constants::UNDEFINED_ADDRESS {
        return Err(Error::InvalidFormat {
            message: String::from("huge object B-tree address is undefined"),
        });
    }

    // Check if huge objects are filtered (flag bit 2)
    if header.flags & 0x04 != 0 {
        // HUGE objects are filtered — would need I/O filter pipeline.
        // This is not yet implemented.
        return Err(Error::UnsupportedFeature {
            feature: String::from("filtered huge fractal heap objects require I/O filter pipeline"),
        });
    }

    let btree_header =
        crate::btree::v2::BTreeV2Header::parse(source, header.huge_object_btree_address, ctx)?;
    let location = find_huge_object_record(
        source,
        header.huge_object_btree_address,
        &btree_header,
        btree_key,
        ctx,
    )?
    .ok_or_else(|| Error::InvalidFormat {
        message: String::from("huge object record not found in B-tree"),
    })?;

    read_bounded_bytes(
        source,
        location.address,
        location.length,
        ctx,
        "fractal heap huge object",
    )
}

/// Read a file region only after applying the parser byte ceiling and a
/// fallible allocation. All managed and huge heap objects use this boundary;
/// no file-supplied length reaches `vec![0; length]` directly.
#[cfg(feature = "alloc")]
pub(crate) fn read_bounded_bytes<R: ReadAt>(
    source: &R,
    address: u64,
    length: u64,
    ctx: &ParseContext,
    what: &'static str,
) -> Result<Vec<u8>> {
    let mut buffer = ctx.budget.zeroed(length, what)?;
    source.read_at(address, &mut buffer)?;
    Ok(buffer)
}

// ---------------------------------------------------------------------------
// Helpers (crate-private)
// ---------------------------------------------------------------------------

/// Read an unsigned little-endian integer of 1–8 bytes.
///
/// Delegates to [`consus_core::read_uint_width`], the single home for
/// fixed-width byte-order decoding.
#[cfg(feature = "alloc")]
pub(crate) fn read_uint_le(data: &[u8], size: usize) -> u64 {
    consus_core::read_uint_width(data, size.min(8), consus_core::ByteOrder::LittleEndian)
        .expect("fractal-heap integer width is 0..=8 and the caller supplies the bytes")
}
