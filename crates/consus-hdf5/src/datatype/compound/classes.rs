//! Reference, enum, variable-length, and array datatype parsers.

use super::{charset_to_encoding, parse_datatype_inner, read_uint_be, read_uint_le, sign_extend};
use alloc::{boxed::Box, format, string::String, vec::Vec};
use consus_core::{ByteOrder, Datatype, EnumMember, Error, ParseBudget, ReferenceType, Result};

// ---------------------------------------------------------------------------
// Class 7 — Reference
// ---------------------------------------------------------------------------

/// Parse a reference datatype.
///
/// ### Class Bit Fields (byte 0)
///
/// | Bit | Meaning                                |
/// |-----|----------------------------------------|
/// | 0   | 0 = object reference, 1 = region reference |
///
/// ### Properties
///
/// None.
pub(crate) fn parse_reference(flags: [u8; 3]) -> Result<(Datatype, usize)> {
    let ref_type = if flags[0] & 0x01 == 0 {
        ReferenceType::Object
    } else {
        ReferenceType::Region
    };
    Ok((Datatype::Reference(ref_type), 0))
}

// ---------------------------------------------------------------------------
// Class 8 — Enum
// ---------------------------------------------------------------------------

/// Parse an enumeration datatype.
///
/// ### Class Bit Fields
///
/// Bytes 0-1 (LE u16): number of members.
///
/// ### Properties
///
/// 1. **Base type** — a complete datatype message (must be an integer type).
/// 2. **Member names** — null-terminated; padded to 8-byte boundary for
///    version < 3, no padding for version 3.
/// 3. **Member values** — packed contiguously, each `base_element_size`
///    bytes in the same encoding as the base type.
pub(crate) fn parse_enum(
    flags: [u8; 3],
    props: &[u8],
    version: u8,
    budget: &ParseBudget,
    depth: u16,
) -> Result<(Datatype, usize)> {
    let num_members = u16::from_le_bytes([flags[0], flags[1]]) as usize;
    let mut pos: usize = 0;

    // -- Base type -----------------------------------------------------------
    let (base_dt, base_consumed) = parse_datatype_inner(props, budget, depth)?;
    pos += base_consumed;

    let base_size = base_dt.element_size().ok_or_else(|| Error::InvalidFormat {
        message: String::from("enum base type must be fixed-size"),
    })?;

    let signed = matches!(&base_dt, Datatype::Integer { signed: true, .. });
    let base_be = matches!(
        &base_dt,
        Datatype::Integer {
            byte_order: ByteOrder::BigEndian,
            ..
        }
    );

    // -- Member names --------------------------------------------------------
    // Each name needs at least its null terminator inside `props`.
    let mut names = Vec::with_capacity(
        budget
            .capacity_hint(num_members as u64, size_of::<String>())
            .min(props.len() + 1),
    );
    for i in 0..num_members {
        let name_start = pos;
        while pos < props.len() && props[pos] != 0 {
            pos += 1;
        }
        if pos >= props.len() {
            return Err(Error::InvalidFormat {
                message: format!("unterminated enum member name at index {i}"),
            });
        }
        let name = core::str::from_utf8(&props[name_start..pos])
            .map_err(|_| Error::InvalidFormat {
                message: format!("enum member {i} name is not valid UTF-8"),
            })
            .map(String::from)?;
        pos += 1; // skip null

        // Version 1/2: each name is padded to 8-byte boundary.
        if version < 3 {
            let name_field_len = pos - name_start;
            pos = name_start + ((name_field_len + 7) & !7);
        }

        names.push(name);
    }

    // -- Member values (packed, base_size bytes each) ------------------------
    // `names` was bounded by the bytes actually consumed above, so its length
    // is the real member count rather than the declared one.
    let mut members = Vec::with_capacity(names.len());
    for (i, name) in names.into_iter().enumerate() {
        if pos + base_size > props.len() {
            return Err(Error::InvalidFormat {
                message: format!("enum member {i} value truncated"),
            });
        }
        let raw = if base_be {
            read_uint_be(&props[pos..], base_size)
        } else {
            read_uint_le(&props[pos..], base_size)
        };
        let value = if signed {
            sign_extend(raw, base_size)
        } else {
            raw as i64
        };
        pos += base_size;
        members.push(EnumMember { name, value });
    }

    Ok((
        Datatype::Enum {
            base: Box::new(base_dt),
            members,
        },
        pos,
    ))
}

// ---------------------------------------------------------------------------
// Class 9 — Variable-Length
// ---------------------------------------------------------------------------

/// Parse a variable-length datatype.
///
/// ### Class Bit Fields
///
/// | Byte | Bits | Meaning                                        |
/// |------|------|------------------------------------------------|
/// | 0    | 0-3  | Type: 0 = sequence, 1 = string                 |
/// | 0    | 4-7  | Padding type (strings only)                    |
/// | 1    | 0-3  | Character set: 0 = ASCII, 1 = UTF-8 (strings) |
///
/// ### Properties
///
/// - **Sequence (type 0):** base datatype message (recursive).
/// - **String (type 1):** none (charset is in the class bit fields).
pub(crate) fn parse_variable_length(
    flags: [u8; 3],
    props: &[u8],
    budget: &ParseBudget,
    depth: u16,
) -> Result<(Datatype, usize)> {
    let vl_type = flags[0] & 0x0F;

    match vl_type {
        // -- Sequence --------------------------------------------------------
        0 => {
            if props.is_empty() {
                return Err(Error::InvalidFormat {
                    message: String::from("variable-length sequence missing base type properties"),
                });
            }
            let (base_dt, base_consumed) = parse_datatype_inner(props, budget, depth)?;
            Ok((
                Datatype::VarLen {
                    base: Box::new(base_dt),
                },
                base_consumed,
            ))
        }

        // -- String ----------------------------------------------------------
        1 => {
            let charset = flags[1] & 0x0F;
            let encoding = charset_to_encoding(charset)?;
            // HDF5 spec: all VL types carry an embedded base type message.
            // For VL STRING the base type is typically a 1-byte FIXED_POINT char
            // (8 header + 4 properties = 12 bytes). We parse it to advance the
            // position correctly when this type appears inside a compound member,
            // but discard the base type in favour of the canonical VariableString.
            // If props is empty (written by an older encoder that omitted the base
            // type), fall back to consuming 0 bytes.
            let base_consumed = if !props.is_empty() {
                match parse_datatype_inner(props, budget, depth) {
                    Ok((_, consumed)) => consumed,
                    Err(_) => 0,
                }
            } else {
                0
            };
            Ok((Datatype::VariableString { encoding }, base_consumed))
        }

        _ => Err(Error::UnsupportedFeature {
            feature: format!("variable-length sub-type {vl_type}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Class 10 — Array
// ---------------------------------------------------------------------------

/// Parse a fixed-size array datatype.
///
/// ### Properties
///
/// **Version 2:**
///
/// | Offset       | Size   | Field                            |
/// |--------------|--------|----------------------------------|
/// | 0            | 1      | Number of dimensions (rank)      |
/// | 1            | 3      | Reserved                         |
/// | 4            | 4×rank | Dimension sizes (u32 LE each)    |
/// | 4+4×rank     | 4×rank | Permutation indices (deprecated) |
/// | 4+8×rank     | var    | Base datatype message            |
///
/// **Version 3:**
///
/// | Offset       | Size   | Field                         |
/// |--------------|--------|-------------------------------|
/// | 0            | 1      | Number of dimensions (rank)   |
/// | 1            | 4×rank | Dimension sizes (u32 LE each) |
/// | 1+4×rank     | var    | Base datatype message         |
pub(crate) fn parse_array(
    props: &[u8],
    version: u8,
    budget: &ParseBudget,
    depth: u16,
) -> Result<(Datatype, usize)> {
    if props.is_empty() {
        return Err(Error::InvalidFormat {
            message: String::from("array datatype properties missing"),
        });
    }

    let rank = props[0] as usize;
    let mut pos: usize = 1;

    // Version < 3: 3 reserved bytes after rank.
    if version < 3 {
        pos += 3;
    }

    // Dimension sizes (4 bytes each, u32 LE).
    // `rank` is a single byte, but each dimension still needs 4 bytes of
    // properties; the available bytes are the tighter bound.
    let mut dims = Vec::with_capacity(rank.min(props.len() / 4 + 1));
    for i in 0..rank {
        if pos + 4 > props.len() {
            return Err(Error::InvalidFormat {
                message: format!("array dimension {i} truncated"),
            });
        }
        let dim = u32::from_le_bytes([props[pos], props[pos + 1], props[pos + 2], props[pos + 3]])
            as usize;
        dims.push(dim);
        pos += 4;
    }

    // Version < 3: skip deprecated permutation indices (4 bytes × rank).
    if version < 3 {
        let perm_size = rank * 4;
        if pos + perm_size > props.len() {
            return Err(Error::InvalidFormat {
                message: String::from("array permutation indices truncated"),
            });
        }
        pos += perm_size;
    }

    // Base datatype (recursive).
    if pos >= props.len() {
        return Err(Error::InvalidFormat {
            message: String::from("array base datatype missing"),
        });
    }
    let (base_dt, base_consumed) = parse_datatype_inner(&props[pos..], budget, depth)?;
    pos += base_consumed;

    Ok((
        Datatype::Array {
            base: Box::new(base_dt),
            dims,
        },
        pos,
    ))
}
