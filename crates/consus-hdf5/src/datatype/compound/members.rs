//! Compound datatype member parsing across layout versions.

use super::{parse_datatype_inner, read_uint_le};
use alloc::{format, string::String, vec::Vec};
use consus_core::{CompoundField, Datatype, Error, ParseBudget, Result};

/// Smallest number of property bytes one compound member can occupy: a name
/// null terminator, a 1-byte member offset, and an 8-byte datatype header.
const MIN_COMPOUND_MEMBER_BYTES: usize = 10;

// ---------------------------------------------------------------------------
// Class 6 — Compound
// ---------------------------------------------------------------------------

/// Parse a compound (struct-like) datatype.
///
/// ### Class Bit Fields
///
/// Bytes 0-1 (LE u16): number of members.
///
/// ### Properties
///
/// Repeated member definitions.  The encoding varies by version:
///
/// **Version 1 / 2** (per member):
///
/// | Field          | Size                                        |
/// |----------------|---------------------------------------------|
/// | Name           | null-terminated, padded to 8-byte boundary  |
/// | Byte offset    | 4 bytes (u32 LE)                            |
/// | Dimensionality | 12 + rank×4 bytes (deprecated, typically 0) |
/// | Datatype msg   | recursive, variable                         |
///
/// **Version 3** (per member):
///
/// | Field        | Size                                       |
/// |--------------|--------------------------------------------|
/// | Name         | null-terminated, NO padding                |
/// | Byte offset  | 1–4 bytes depending on compound total size |
/// | Datatype msg | recursive, variable                        |
pub(crate) fn parse_compound(
    compound_size: usize,
    flags: [u8; 3],
    props: &[u8],
    version: u8,
    budget: &ParseBudget,
    depth: u16,
) -> Result<(Datatype, usize)> {
    let num_members = u16::from_le_bytes([flags[0], flags[1]]) as usize;
    // The u16 width caps the count at 65 535, but each member still needs a
    // name terminator plus an 8-byte datatype header inside `props`; the
    // available bytes are the tighter bound on the reservation.
    let mut fields = Vec::with_capacity(
        budget
            .capacity_hint(num_members as u64, size_of::<CompoundField>())
            .min(props.len() / MIN_COMPOUND_MEMBER_BYTES + 1),
    );
    let mut pos: usize = 0;

    for i in 0..num_members {
        let remaining = props.get(pos..).ok_or_else(|| Error::InvalidFormat {
            message: format!("compound member {i} extends past properties"),
        })?;
        let (field, consumed) =
            parse_compound_member(remaining, compound_size, version, i, budget, depth)?;
        fields.push(field);
        pos += consumed;
    }

    Ok((
        Datatype::Compound {
            fields,
            size: compound_size,
        },
        pos,
    ))
}

/// Parse a single compound member definition starting at `data[0]`.
///
/// Returns `(CompoundField, bytes_consumed)`.
fn parse_compound_member(
    data: &[u8],
    compound_size: usize,
    version: u8,
    member_index: usize,
    budget: &ParseBudget,
    depth: u16,
) -> Result<(CompoundField, usize)> {
    let mut pos: usize = 0;

    // -- Name (null-terminated) ----------------------------------------------
    let name_start = pos;
    while pos < data.len() && data[pos] != 0 {
        pos += 1;
    }
    if pos >= data.len() {
        return Err(Error::InvalidFormat {
            message: format!("unterminated compound member name at index {member_index}"),
        });
    }
    let name = core::str::from_utf8(&data[name_start..pos])
        .map_err(|_| Error::InvalidFormat {
            message: format!("compound member {member_index} name is not valid UTF-8"),
        })
        .map(String::from)?;
    pos += 1; // skip null terminator

    // For version 1/2: name field (including null) is padded to 8-byte boundary.
    if version < 3 {
        let name_field_len = pos - name_start;
        pos = name_start + ((name_field_len + 7) & !7);
    }

    // -- Byte offset of member within the compound ---------------------------
    let member_offset = if version < 3 {
        if pos + 4 > data.len() {
            return Err(Error::InvalidFormat {
                message: format!("compound member {member_index} offset truncated"),
            });
        }
        let off =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        off
    } else {
        let off_size = member_offset_byte_count(compound_size);
        if pos + off_size > data.len() {
            return Err(Error::InvalidFormat {
                message: format!("compound member {member_index} v3 offset truncated"),
            });
        }
        let off = read_uint_le(&data[pos..], off_size) as usize;
        pos += off_size;
        off
    };

    // Dimensionality (version 1/2 only, deprecated) -----------------------
    //
    // | Size | Field                                              |
    // |------|----------------------------------------------------- |
    // | 1    | Number of dimensions (rank; typically 0 for scalar)|
    // | 3    | Reserved                                           |
    // | 4    | Dimension permutation index (deprecated)           |
    // | 4    | Reserved                                           |
    // | 4×4  | Dimension sizes — ALWAYS 4 slots × 4 bytes (16 B) |
    //
    // HDF5 spec III.4.5: versions 1 and 2 reserve space for exactly 4
    // dimension size entries (16 bytes) regardless of the actual rank.
    // The total fixed overhead is 1+3+4+4+16 = 28 bytes.
    if version < 3 {
        if pos + 28 > data.len() {
            return Err(Error::InvalidFormat {
                message: format!("compound member {member_index} dimensionality section truncated"),
            });
        }
        // Read rank for documentation purposes; it is not used to size the
        // array because the 4-slot layout is fixed by the spec.
        let _rank = data[pos];
        // 1 (rank) + 3 (reserved) + 4 (perm) + 4 (reserved) + 4×4 (dim_sizes) = 28
        pos += 28;
    }

    // -- Member datatype (recursive) -----------------------------------------
    if pos >= data.len() {
        return Err(Error::InvalidFormat {
            message: format!("compound member {member_index} datatype missing"),
        });
    }
    let (datatype, dt_consumed) = parse_datatype_inner(&data[pos..], budget, depth)?;
    pos += dt_consumed;

    Ok((
        CompoundField {
            name,
            datatype,
            offset: member_offset,
        },
        pos,
    ))
}

/// Compute the byte count used for member offsets in compound datatype
/// version 3.
///
/// The byte count is the minimum number of bytes required to represent any
/// byte offset within the compound's total `size`:
///
/// | Compound size range | Bytes |
/// |---------------------|-------|
/// | 0 ..= 255          | 1     |
/// | 256 ..= 65 535     | 2     |
/// | 65 536 ..= 16 777 215 | 3  |
/// | ≥ 16 777 216       | 4     |
pub(crate) fn member_offset_byte_count(compound_size: usize) -> usize {
    if compound_size < 256 {
        1
    } else if compound_size < 65_536 {
        2
    } else if compound_size < 16_777_216 {
        3
    } else {
        4
    }
}
