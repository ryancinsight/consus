#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use consus_core::{Error, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;

use super::collection::GlobalHeapCollection;

// ---------------------------------------------------------------------------
// VL Reference Resolution
// ---------------------------------------------------------------------------

/// Resolve variable-length data references from raw dataset bytes.
///
/// Each VL reference occupies `4 + offset_size + 4` bytes:
/// - 4-byte sequence length (u32 LE, number of elements in the VL object)
/// - `offset_size` bytes: file address of the GCOL collection
/// - 4-byte object index (u32 LE, 1-based)
///
/// ## Algorithm
///
/// 1. Compute `ref_size = 4 + ctx.offset_bytes() + 4`.
/// 2. Validate `raw_data.len() % ref_size == 0`.
/// 3. For each `ref_size`-byte chunk in `raw_data`:
///    a. Read `sequence_length` (u32 LE, bytes `0..4`).
///    b. Read `heap_collection_address` via `ctx.read_offset(&chunk[4..])`.
///    c. Read `object_index` (u32 LE, bytes `4+offset_bytes..4+offset_bytes+4`).
///    d. If `heap_collection_address` equals the offset-size-specific undefined
///       sentinel or `sequence_length == 0`, push an empty `Vec<u8>`.
///    e. Otherwise call [`GlobalHeapCollection::parse`], then `get_data`, and
///       clone the returned slice.
///       Returns [`Error::InvalidFormat`] if the object index is absent.
/// 4. Return `Vec<Vec<u8>>` - one entry per VL element.
///
/// ## Undefined Address Sentinel
///
/// The sentinel is offset-size-dependent: all bits set within the
/// `offset_bytes`-wide address field.
///
/// | offset_bytes | sentinel value |
/// |---|---|
/// | 4 | `0xFFFF_FFFF` |
/// | 8 | `0xFFFF_FFFF_FFFF_FFFF` |
///
/// This matches the HDF5 specification and the `UNDEFINED_ADDRESS` constant
/// for 8-byte offset files.
///
/// ## Errors
///
/// - [`Error::InvalidFormat`] if `raw_data.len()` is not a multiple of `ref_size`.
/// - [`Error::InvalidFormat`] if `object_index` does not fit in `u16`.
/// - [`Error::InvalidFormat`] if the heap object is absent from the collection.
/// - Propagates I/O and format errors from [`GlobalHeapCollection::parse`].
#[cfg(feature = "alloc")]
pub fn resolve_vl_references<R: ReadAt>(
    source: &R,
    raw_data: &[u8],
    ctx: &ParseContext,
) -> Result<Vec<Vec<u8>>> {
    let offset_bytes = ctx.offset_bytes();
    let ref_size = 4 + offset_bytes + 4;

    if !raw_data.len().is_multiple_of(ref_size) {
        return Err(Error::InvalidFormat {
            message: alloc::format!(
                "VL raw data length {} is not a multiple of ref_size {} (offset_size={})",
                raw_data.len(),
                ref_size,
                offset_bytes,
            ),
        });
    }

    // Offset-size-specific undefined address: all `offset_bytes` bits set.
    // For 4-byte offsets: 0xFFFF_FFFF; for 8-byte offsets: u64::MAX.
    // The guard `offset_bytes == 8` prevents the `1u64 << 64` overflow.
    let undef_addr: u64 = if offset_bytes == 8 {
        u64::MAX
    } else {
        // offset_bytes in {2, 4} => shift in {16, 32} => no overflow.
        (1u64 << (offset_bytes * 8)).wrapping_sub(1)
    };

    let n = raw_data.len() / ref_size;
    let mut result = ctx.budget.vec_with_capacity::<Vec<u8>>(
        u64::try_from(n).map_err(|_| Error::Overflow)?,
        "variable-length reference count",
    )?;

    for (i, chunk) in raw_data.chunks_exact(ref_size).enumerate() {
        let sequence_length = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let heap_collection_address = ctx.read_offset(&chunk[4..]);
        let idx_base = 4 + offset_bytes;
        let object_index = u32::from_le_bytes([
            chunk[idx_base],
            chunk[idx_base + 1],
            chunk[idx_base + 2],
            chunk[idx_base + 3],
        ]);

        if heap_collection_address == undef_addr || sequence_length == 0 {
            result.push(Vec::new());
            continue;
        }

        // object_index is stored as u32 in the VL reference but the GCOL
        // indexes objects with a u16 field; a value > u16::MAX indicates
        // file corruption.
        let obj_idx = u16::try_from(object_index).map_err(|_| Error::InvalidFormat {
            message: alloc::format!(
                "VL object index {} exceeds u16::MAX at element {} (heap addr {})",
                object_index,
                i,
                heap_collection_address,
            ),
        })?;

        let coll = GlobalHeapCollection::parse(source, heap_collection_address, ctx)?;

        let data = coll.get_data(obj_idx).ok_or_else(|| Error::InvalidFormat {
            message: alloc::format!(
                "VL object index {} not found in heap collection at address {} (element {})",
                obj_idx,
                heap_collection_address,
                i,
            ),
        })?;

        result.push(Vec::from(data));
    }

    Ok(result)
}
