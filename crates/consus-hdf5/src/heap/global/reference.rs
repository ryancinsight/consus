//! Variable-length reference resolution.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use consus_core::{Error, Result};
use consus_io::ReadAt;

use crate::address::ParseContext;

#[cfg(feature = "alloc")]
use super::collection::GlobalHeapCollection;

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy)]
struct PendingReference {
    address: u64,
    output: usize,
    object: u16,
}

/// Resolve variable-length data references from raw dataset bytes.
///
/// Each VL reference occupies `4 + offset_size + 4` bytes:
/// - 4-byte sequence length (u32 LE, number of elements in the VL object)
/// - `offset_size` bytes: file address of the GCOL collection
/// - 4-byte object index (u32 LE, 1-based)
///
/// The resolver groups references by collection address, parses each global
/// heap collection once, and copies each referenced object into the returned
/// value at its original element position.
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
    context: &ParseContext,
) -> Result<Vec<Vec<u8>>> {
    let offset_bytes = context.offset_bytes();
    let reference_size = 4 + offset_bytes + 4;

    if !raw_data.len().is_multiple_of(reference_size) {
        return Err(Error::InvalidFormat {
            message: alloc::format!(
                "VL raw data length {} is not a multiple of ref_size {} (offset_size={})",
                raw_data.len(),
                reference_size,
                offset_bytes,
            ),
        });
    }

    let element_count = raw_data.len() / reference_size;
    let mut values = context.budget.vec_with_capacity::<Vec<u8>>(
        u64::try_from(element_count).map_err(|_| Error::Overflow)?,
        "variable-length reference count",
    )?;
    let mut pending = context.budget.vec_with_capacity::<PendingReference>(
        u64::try_from(element_count).map_err(|_| Error::Overflow)?,
        "variable-length reference metadata",
    )?;

    let mut failure: Option<(usize, Error)> = None;
    for (output_index, chunk) in raw_data.chunks_exact(reference_size).enumerate() {
        values.push(Vec::new());

        let sequence_length = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let collection_address = context.read_offset(&chunk[4..]);
        let object_index_offset = 4 + offset_bytes;
        let object_index = u32::from_le_bytes([
            chunk[object_index_offset],
            chunk[object_index_offset + 1],
            chunk[object_index_offset + 2],
            chunk[object_index_offset + 3],
        ]);

        if collection_address == undefined_address(offset_bytes) || sequence_length == 0 {
            continue;
        }

        let Ok(object_index) = u16::try_from(object_index) else {
            // Earlier references may still fail during collection lookup.
            // Keep this error as an upper bound on the failing element and
            // avoid collecting references that cannot change the result.
            failure = Some((
                output_index,
                Error::InvalidFormat {
                    message: alloc::format!(
                        "VL object index {} exceeds u16::MAX at element {} (heap addr {})",
                        object_index,
                        output_index,
                        collection_address,
                    ),
                },
            ));
            break;
        };

        pending.push(PendingReference {
            address: collection_address,
            output: output_index,
            object: object_index,
        });
    }

    pending.sort_by_key(|reference| reference.address);

    let mut pending_offset = 0_usize;
    while pending_offset < pending.len() {
        let collection_address = pending.get(pending_offset).ok_or(Error::Overflow)?.address;
        let mut group_end = pending_offset.checked_add(1).ok_or(Error::Overflow)?;
        while pending
            .get(group_end)
            .is_some_and(|reference| reference.address == collection_address)
        {
            group_end = group_end.checked_add(1).ok_or(Error::Overflow)?;
        }

        let references = pending
            .get(pending_offset..group_end)
            .ok_or(Error::Overflow)?;
        match GlobalHeapCollection::parse(source, collection_address, context) {
            Ok(collection) => {
                for reference in references {
                    match collection.get_data(reference.object) {
                        Some(data) => {
                            let output = values.get_mut(reference.output).ok_or(Error::Overflow)?;
                            *output = Vec::from(data);
                        }
                        None => {
                            let error = Error::InvalidFormat {
                                message: alloc::format!(
                                    "VL object index {} not found in heap collection at address {} (element {})",
                                    reference.object,
                                    collection_address,
                                    reference.output,
                                ),
                            };
                            record_failure(&mut failure, reference.output, error);
                        }
                    }
                }
            }
            Err(error) => {
                let first_output = references
                    .iter()
                    .map(|reference| reference.output)
                    .min()
                    .ok_or(Error::Overflow)?;
                record_failure(&mut failure, first_output, error);
            }
        }
        pending_offset = group_end;
    }

    if let Some((_, error)) = failure {
        return Err(error);
    }

    Ok(values)
}

/// Keep the failure belonging to the earliest element, matching element-order
/// resolution: the first invalid reference reports its own error.
#[cfg(feature = "alloc")]
fn record_failure(failure: &mut Option<(usize, Error)>, output: usize, error: Error) {
    match failure {
        Some((earliest, _)) if *earliest <= output => {}
        _ => *failure = Some((output, error)),
    }
}

#[cfg(feature = "alloc")]
const fn undefined_address(offset_bytes: usize) -> u64 {
    if offset_bytes == 8 {
        u64::MAX
    } else {
        (1_u64 << (offset_bytes * 8)).wrapping_sub(1)
    }
}
