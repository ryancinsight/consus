//! Complete HDF5 datatype parsing for all datatype classes.
//!
//! ## Specification (HDF5 File Format Specification, Section IV.A.2.d)
//!
//! Every HDF5 datatype message shares an 8-byte header followed by
//! class-specific properties:
//!
//! | Offset | Size | Field                                              |
//! |--------|------|----------------------------------------------------|
//! | 0      | 1    | Class (bits 0-3) + version (bits 4-7)              |
//! | 1      | 3    | Class bit fields (class-specific flags)             |
//! | 4      | 4    | Size of the datatype element in bytes (LE u32)      |
//! | 8      | var  | Class-specific properties                          |
//!
//! ## Supported Classes
//!
//! | Class | Value | Properties                                              |
//! |-------|-------|---------------------------------------------------------|
//! | Fixed-point    | 0  | bit_offset(2) + bit_precision(2)               |
//! | Floating-point | 1  | bit_offset(2) + precision(2) + exp(4) + man(4) |
//! | Time           | 2  | (deprecated, unsupported)                      |
//! | String         | 3  | (none; padding + charset in flags)             |
//! | Bitfield       | 4  | bit_offset(2) + bit_precision(2)               |
//! | Opaque         | 5  | null-padded ASCII tag                          |
//! | Compound       | 6  | ordered member definitions (recursive)         |
//! | Reference      | 7  | (none; ref type in flags)                      |
//! | Enum           | 8  | base type + member names + packed values        |
//! | Variable-len   | 9  | optional base type (sequence only)             |
//! | Array          | 10 | rank + dims + base type (recursive)            |
//!
//! ## Module Gate
//!
//! This module requires the `alloc` feature because `Datatype` variants
//! (`Compound`, `Enum`, `Array`, `VarLen`) and helper types (`CompoundField`,
//! `EnumMember`) use heap-allocated collections.

use alloc::{boxed::Box, format, string::String, vec::Vec};
use core::num::NonZeroUsize;

use consus_core::{
    ByteOrder, CompoundField, Datatype, EnumMember, Error, ReferenceType, Result, StringEncoding,
};

use super::classes::{
    ARRAY, BITFIELD, COMPOUND, ENUM, FIXED_POINT, FLOATING_POINT, OPAQUE, REFERENCE, STRING, TIME,
    VARIABLE_LENGTH,
};
use super::{byte_order_from_flags, map_fixed_point, map_floating_point};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a complete HDF5 datatype message and return the canonical
/// [`Datatype`].
///
/// `data` must begin at byte 0 of the datatype message (the class+version
/// byte).  The slice may be longer than the message; only the consumed
/// portion is read.
///
/// # Errors
///
/// - [`Error::InvalidFormat`] on truncated or structurally invalid data.
/// - [`Error::UnsupportedFeature`] for the deprecated TIME class or
///   unknown class values.
pub fn parse_datatype(data: &[u8]) -> Result<Datatype> {
    let (dt, _consumed) = parse_datatype_inner(data)?;
    Ok(dt)
}

// ---------------------------------------------------------------------------
// Internal recursive parser
// ---------------------------------------------------------------------------

/// Parse a datatype message and return `(Datatype, total_bytes_consumed)`.
///
/// `total_bytes_consumed` includes the 8-byte header plus class-specific
/// properties, enabling callers (compound member parsing, enum base type,
/// array base type, VL base type) to advance past the embedded message.
fn parse_datatype_inner(data: &[u8]) -> Result<(Datatype, usize)> {
    if data.len() < 8 {
        return Err(Error::InvalidFormat {
            message: String::from("datatype message shorter than 8-byte header"),
        });
    }

    let class_and_version = data[0];
    let class = class_and_version & 0x0F;
    let version = (class_and_version >> 4) & 0x0F;
    let flags = [data[1], data[2], data[3]];
    let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let props = &data[8..];

    let (dt, props_consumed) = match class {
        FIXED_POINT => parse_fixed_point(size, flags, props)?,
        FLOATING_POINT => parse_floating_point(size, flags, props)?,
        TIME => {
            return Err(Error::UnsupportedFeature {
                feature: String::from("TIME datatype class (deprecated)"),
            });
        }
        STRING => parse_string(size, flags)?,
        BITFIELD => parse_bitfield(size, flags, props)?,
        OPAQUE => parse_opaque(size, props)?,
        COMPOUND => parse_compound(size, flags, props, version)?,
        REFERENCE => parse_reference(flags)?,
        ENUM => parse_enum(flags, props, version)?,
        VARIABLE_LENGTH => parse_variable_length(flags, props)?,
        ARRAY => parse_array(props, version)?,
        _ => {
            return Err(Error::UnsupportedFeature {
                feature: format!("datatype class {class}"),
            });
        }
    };

    Ok((dt, 8 + props_consumed))
}

// ---------------------------------------------------------------------------
// Class 0 — Fixed-Point (integer)
// ---------------------------------------------------------------------------

/// Parse a fixed-point (integer) datatype.
///
/// ### Class Bit Fields (byte 1 of header)
///
/// | Bit | Meaning                       |
/// |-----|-------------------------------|
/// | 0   | Byte order: 0 = LE, 1 = BE   |
/// | 3   | Signed: 0 = unsigned, 1 = yes |
///
/// ### Properties (4 bytes)
///
/// | Offset | Size | Field         |
/// |--------|------|---------------|
/// | 0      | 2    | Bit offset    |
/// | 2      | 2    | Bit precision |
fn parse_fixed_point(size: usize, flags: [u8; 3], props: &[u8]) -> Result<(Datatype, usize)> {
    if props.len() < 4 {
        return Err(Error::InvalidFormat {
            message: String::from("fixed-point properties truncated"),
        });
    }
    let dt = map_fixed_point(size, flags[0]);
    Ok((dt, 4))
}

// ---------------------------------------------------------------------------
// Class 1 — Floating-Point
// ---------------------------------------------------------------------------

/// Parse a floating-point datatype.
///
/// ### Class Bit Fields (byte 1 of header)
///
/// | Bit | Meaning                     |
/// |-----|-----------------------------|
/// | 0   | Byte order: 0 = LE, 1 = BE |
///
/// ### Properties (12 bytes)
///
/// | Offset | Size | Field              |
/// |--------|------|--------------------|
/// | 0      | 2    | Bit offset         |
/// | 2      | 2    | Bit precision      |
/// | 4      | 1    | Exponent location  |
/// | 5      | 1    | Exponent size      |
/// | 6      | 1    | Mantissa location  |
/// | 7      | 1    | Mantissa size      |
/// | 8      | 4    | Exponent bias      |
fn parse_floating_point(size: usize, flags: [u8; 3], props: &[u8]) -> Result<(Datatype, usize)> {
    if props.len() < 12 {
        return Err(Error::InvalidFormat {
            message: String::from("floating-point properties truncated"),
        });
    }
    let dt = map_floating_point(size, flags[0]);
    Ok((dt, 12))
}

// ---------------------------------------------------------------------------
// Class 3 — String
// ---------------------------------------------------------------------------

/// Parse a fixed-length string datatype.
///
/// ### Class Bit Fields
///
/// | Byte | Bits | Meaning                                           |
/// |------|------|---------------------------------------------------|
/// | 0    | 0-3  | Padding: 0 = null-terminate, 1 = null-pad, 2 = space-pad |
/// | 1    | 0-3  | Character set: 0 = ASCII, 1 = UTF-8              |
///
/// ### Properties
///
/// None.  The element size from the header gives the fixed string length.
///
/// If `size == 0` this indicates a variable-length string that is only
/// meaningful inside a variable-length (class 9) wrapper; the returned
/// `FixedString` with `length = 0` should be interpreted accordingly by
/// the caller.
fn parse_string(size: usize, flags: [u8; 3]) -> Result<(Datatype, usize)> {
    let charset = flags[1] & 0x0F;
    let encoding = charset_to_encoding(charset)?;
    Ok((
        Datatype::FixedString {
            length: size,
            encoding,
        },
        0,
    ))
}

// ---------------------------------------------------------------------------
// Class 4 — Bitfield
// ---------------------------------------------------------------------------

/// Parse a bitfield datatype, mapped to an unsigned integer of the same size.
///
/// ### Class Bit Fields (byte 1 of header)
///
/// | Bit | Meaning                     |
/// |-----|-----------------------------|
/// | 0   | Byte order: 0 = LE, 1 = BE |
///
/// ### Properties (4 bytes)
///
/// | Offset | Size | Field         |
/// |--------|------|---------------|
/// | 0      | 2    | Bit offset    |
/// | 2      | 2    | Bit precision |
fn parse_bitfield(size: usize, flags: [u8; 3], props: &[u8]) -> Result<(Datatype, usize)> {
    if props.len() < 4 {
        return Err(Error::InvalidFormat {
            message: String::from("bitfield properties truncated"),
        });
    }
    let byte_order = byte_order_from_flags(flags[0]);
    let bits = NonZeroUsize::new(size * 8).ok_or_else(|| Error::InvalidFormat {
        message: String::from("bitfield size must be > 0"),
    })?;
    Ok((
        Datatype::Integer {
            bits,
            byte_order,
            signed: false,
        },
        4,
    ))
}

// ---------------------------------------------------------------------------
// Class 5 — Opaque
// ---------------------------------------------------------------------------

/// Parse an opaque datatype.
///
/// ### Properties
///
/// A null-padded ASCII tag string.  The HDF5 specification states the tag
/// is NOT null-terminated, but in practice the C library writes it as a
/// null-padded string aligned to an 8-byte boundary.  This parser scans
/// for the first null byte to determine the tag extent.
///
/// When no tag is present (empty properties), the returned tag is `None`.
fn parse_opaque(size: usize, props: &[u8]) -> Result<(Datatype, usize)> {
    if props.is_empty() {
        return Ok((Datatype::Opaque { size, tag: None }, 0));
    }

    let null_pos = props.iter().position(|&b| b == 0);
    let (tag, consumed) = match null_pos {
        Some(0) => {
            // Empty tag; consume null + padding (up to 8-byte boundary).
            let padded = 8.min(props.len());
            (None, padded)
        }
        Some(n) => {
            let tag_str = core::str::from_utf8(&props[..n]).map_err(|_| Error::InvalidFormat {
                message: String::from("opaque tag is not valid UTF-8"),
            })?;
            // Consume tag + null, rounded up to 8-byte boundary.
            let padded = ((n + 1) + 7) & !7;
            (Some(tag_str.to_string()), padded.min(props.len()))
        }
        None => {
            // No null found; consume all available bytes as the tag.
            let tag_str = core::str::from_utf8(props).map_err(|_| Error::InvalidFormat {
                message: String::from("opaque tag is not valid UTF-8"),
            })?;
            (Some(tag_str.to_string()), props.len())
        }
    };

    Ok((Datatype::Opaque { size, tag }, consumed))
}

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
fn parse_compound(
    compound_size: usize,
    flags: [u8; 3],
    props: &[u8],
    version: u8,
) -> Result<(Datatype, usize)> {
    let num_members = u16::from_le_bytes([flags[0], flags[1]]) as usize;
    let mut fields = Vec::with_capacity(num_members);
    let mut pos: usize = 0;

    for i in 0..num_members {
        let remaining = props.get(pos..).ok_or_else(|| Error::InvalidFormat {
            message: format!("compound member {i} extends past properties"),
        })?;
        let (field, consumed) = parse_compound_member(remaining, compound_size, version, i)?;
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
        })?
        .to_string();
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

    // -- Dimensionality (version 1/2 only, deprecated) -----------------------
    //
    // | Size | Field                                              |
    // |------|----------------------------------------------------|
    // | 1    | Number of dimensions (rank; typically 0 for scalar)|
    // | 3    | Reserved                                           |
    // | 4    | Dimension permutation index (deprecated)           |
    // | 4    | Reserved                                           |
    // | 4×r  | Dimension sizes (u32 LE each)                     |
    if version < 3 {
        if pos >= data.len() {
            return Err(Error::InvalidFormat {
                message: format!("compound member {member_index} dimensionality truncated"),
            });
        }
        let rank = data[pos] as usize;
        // 1 (rank) + 3 (reserved) + 4 (perm) + 4 (reserved) = 12 fixed
        let dim_overhead = 12 + rank * 4;
        if pos + dim_overhead > data.len() {
            return Err(Error::InvalidFormat {
                message: format!("compound member {member_index} dimensionality section truncated"),
            });
        }
        pos += dim_overhead;
    }

    // -- Member datatype (recursive) -----------------------------------------
    if pos >= data.len() {
        return Err(Error::InvalidFormat {
            message: format!("compound member {member_index} datatype missing"),
        });
    }
    let (datatype, dt_consumed) = parse_datatype_inner(&data[pos..])?;
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
fn parse_reference(flags: [u8; 3]) -> Result<(Datatype, usize)> {
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
fn parse_enum(flags: [u8; 3], props: &[u8], version: u8) -> Result<(Datatype, usize)> {
    let num_members = u16::from_le_bytes([flags[0], flags[1]]) as usize;
    let mut pos: usize = 0;

    // -- Base type -----------------------------------------------------------
    let (base_dt, base_consumed) = parse_datatype_inner(props)?;
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
    let mut names = Vec::with_capacity(num_members);
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
            })?
            .to_string();
        pos += 1; // skip null

        // Version 1/2: each name is padded to 8-byte boundary.
        if version < 3 {
            let name_field_len = pos - name_start;
            pos = name_start + ((name_field_len + 7) & !7);
        }

        names.push(name);
    }

    // -- Member values (packed, base_size bytes each) ------------------------
    let mut members = Vec::with_capacity(num_members);
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
fn parse_variable_length(flags: [u8; 3], props: &[u8]) -> Result<(Datatype, usize)> {
    let vl_type = flags[0] & 0x0F;

    match vl_type {
        // -- Sequence --------------------------------------------------------
        0 => {
            if props.is_empty() {
                return Err(Error::InvalidFormat {
                    message: String::from("variable-length sequence missing base type properties"),
                });
            }
            let (base_dt, base_consumed) = parse_datatype_inner(props)?;
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
            Ok((Datatype::VariableString { encoding }, 0))
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
fn parse_array(props: &[u8], version: u8) -> Result<(Datatype, usize)> {
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
    let mut dims = Vec::with_capacity(rank);
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
    let (base_dt, base_consumed) = parse_datatype_inner(&props[pos..])?;
    pos += base_consumed;

    Ok((
        Datatype::Array {
            base: Box::new(base_dt),
            dims,
        },
        pos,
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map an HDF5 character-set code to the canonical [`StringEncoding`].
///
/// | Code | Encoding |
/// |------|----------|
/// | 0    | ASCII    |
/// | 1    | UTF-8    |
fn charset_to_encoding(charset: u8) -> Result<StringEncoding> {
    match charset {
        0 => Ok(StringEncoding::Ascii),
        1 => Ok(StringEncoding::Utf8),
        _ => Err(Error::UnsupportedFeature {
            feature: format!("character set code {charset}"),
        }),
    }
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
fn member_offset_byte_count(compound_size: usize) -> usize {
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

/// Read an unsigned little-endian integer of 0–8 bytes.
fn read_uint_le(data: &[u8], size: usize) -> u64 {
    match size {
        0 => 0,
        1 => data[0] as u64,
        2 => u16::from_le_bytes([data[0], data[1]]) as u64,
        3 => u32::from_le_bytes([data[0], data[1], data[2], 0]) as u64,
        4 => u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64,
        5 => u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], 0, 0, 0]),
        6 => u64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], data[5], 0, 0]),
        7 => u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], 0,
        ]),
        _ => u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
    }
}

/// Read an unsigned big-endian integer of 0–8 bytes.
fn read_uint_be(data: &[u8], size: usize) -> u64 {
    match size {
        0 => 0,
        1 => data[0] as u64,
        2 => u16::from_be_bytes([data[0], data[1]]) as u64,
        3 => u32::from_be_bytes([0, data[0], data[1], data[2]]) as u64,
        4 => u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as u64,
        5 => u64::from_be_bytes([0, 0, 0, data[0], data[1], data[2], data[3], data[4]]),
        6 => u64::from_be_bytes([0, 0, data[0], data[1], data[2], data[3], data[4], data[5]]),
        7 => u64::from_be_bytes([
            0, data[0], data[1], data[2], data[3], data[4], data[5], data[6],
        ]),
        _ => u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]),
    }
}

/// Sign-extend an unsigned value of `size` bytes to `i64`.
///
/// If the most-significant bit of the `size`-byte value is set, the upper
/// bits of the returned `i64` are filled with ones (arithmetic shift).
fn sign_extend(val: u64, size: usize) -> i64 {
    let bits = size * 8;
    if bits == 0 || bits >= 64 {
        return val as i64;
    }
    let shift = 64 - bits;
    ((val as i64) << shift) >> shift
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers for building datatype messages ------------------------------

    /// Build a datatype message header.
    fn dt_header(class: u8, version: u8, flags: [u8; 3], size: u32) -> [u8; 8] {
        let cv = (version << 4) | (class & 0x0F);
        let s = size.to_le_bytes();
        [cv, flags[0], flags[1], flags[2], s[0], s[1], s[2], s[3]]
    }

    /// Build a complete integer (fixed-point) datatype message.
    fn int_msg(size: u32, signed: bool, le: bool) -> Vec<u8> {
        let mut flags_byte: u8 = 0;
        if !le {
            flags_byte |= 0x01;
        }
        if signed {
            flags_byte |= 0x08;
        }
        let hdr = dt_header(FIXED_POINT, 1, [flags_byte, 0, 0], size);
        let mut msg = hdr.to_vec();
        // Properties: bit_offset(2) + bit_precision(2)
        msg.extend_from_slice(&0u16.to_le_bytes()); // bit offset = 0
        msg.extend_from_slice(&((size * 8) as u16).to_le_bytes()); // precision = size*8
        msg
    }

    // -- Class 0: Fixed-point ------------------------------------------------

    #[test]
    fn parse_u32_le() {
        let msg = int_msg(4, false, true);
        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Integer {
                bits,
                byte_order,
                signed,
            } => {
                assert_eq!(bits.get(), 32);
                assert_eq!(byte_order, ByteOrder::LittleEndian);
                assert!(!signed);
            }
            other => panic!("expected Integer, got: {other:?}"),
        }
    }

    #[test]
    fn parse_i16_be() {
        let msg = int_msg(2, true, false);
        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Integer {
                bits,
                byte_order,
                signed,
            } => {
                assert_eq!(bits.get(), 16);
                assert_eq!(byte_order, ByteOrder::BigEndian);
                assert!(signed);
            }
            other => panic!("expected Integer, got: {other:?}"),
        }
    }

    // -- Class 1: Floating-point ---------------------------------------------

    #[test]
    fn parse_f64_le() {
        let hdr = dt_header(FLOATING_POINT, 1, [0x00, 0, 0], 8);
        let mut msg = hdr.to_vec();
        // 12 bytes of properties (content irrelevant for mapping)
        msg.extend_from_slice(&[0u8; 12]);
        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Float { bits, byte_order } => {
                assert_eq!(bits.get(), 64);
                assert_eq!(byte_order, ByteOrder::LittleEndian);
            }
            other => panic!("expected Float, got: {other:?}"),
        }
    }

    // -- Class 3: String -----------------------------------------------------

    #[test]
    fn parse_fixed_string_ascii() {
        // Padding = 0 (null-terminate), charset = 0 (ASCII), size = 10
        let hdr = dt_header(STRING, 1, [0x00, 0x00, 0], 10);
        let msg = hdr.to_vec();
        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::FixedString { length, encoding } => {
                assert_eq!(length, 10);
                assert_eq!(encoding, StringEncoding::Ascii);
            }
            other => panic!("expected FixedString, got: {other:?}"),
        }
    }

    #[test]
    fn parse_fixed_string_utf8() {
        // charset = 1 (UTF-8)
        let hdr = dt_header(STRING, 1, [0x00, 0x01, 0], 32);
        let msg = hdr.to_vec();
        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::FixedString { length, encoding } => {
                assert_eq!(length, 32);
                assert_eq!(encoding, StringEncoding::Utf8);
            }
            other => panic!("expected FixedString, got: {other:?}"),
        }
    }

    // -- Class 4: Bitfield ---------------------------------------------------

    #[test]
    fn parse_bitfield_2byte_le() {
        let hdr = dt_header(BITFIELD, 1, [0x00, 0, 0], 2);
        let mut msg = hdr.to_vec();
        msg.extend_from_slice(&[0u8; 4]); // properties
        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Integer {
                bits,
                byte_order,
                signed,
            } => {
                assert_eq!(bits.get(), 16);
                assert_eq!(byte_order, ByteOrder::LittleEndian);
                assert!(!signed);
            }
            other => panic!("expected Integer from bitfield, got: {other:?}"),
        }
    }

    // -- Class 5: Opaque -----------------------------------------------------

    #[test]
    fn parse_opaque_with_tag() {
        let hdr = dt_header(OPAQUE, 1, [0, 0, 0], 100);
        let mut msg = hdr.to_vec();
        // Tag "mytype" + null + padding to 8 bytes
        let tag = b"mytype\0\0"; // 8 bytes total
        msg.extend_from_slice(tag);
        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Opaque { size, tag } => {
                assert_eq!(size, 100);
                assert_eq!(tag.as_deref(), Some("mytype"));
            }
            other => panic!("expected Opaque, got: {other:?}"),
        }
    }

    #[test]
    fn parse_opaque_no_tag() {
        let hdr = dt_header(OPAQUE, 1, [0, 0, 0], 8);
        let msg = hdr.to_vec(); // empty properties
        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Opaque { size, tag } => {
                assert_eq!(size, 8);
                assert!(tag.is_none());
            }
            other => panic!("expected Opaque, got: {other:?}"),
        }
    }

    // -- Class 6: Compound ---------------------------------------------------

    #[test]
    fn parse_compound_v3_two_members() {
        // Compound with 2 members, total size = 12:
        //   member "x": u32 LE at offset 0
        //   member "y": i64 LE at offset 4

        let num_members: u16 = 2;
        let compound_size: u32 = 12;
        let hdr = dt_header(
            COMPOUND,
            3,
            [
                num_members.to_le_bytes()[0],
                num_members.to_le_bytes()[1],
                0,
            ],
            compound_size,
        );
        let mut msg = hdr.to_vec();

        // Member "x": name, offset (1 byte for size<256), datatype
        msg.extend_from_slice(b"x\0");
        msg.push(0x00); // offset = 0 (1 byte since compound_size=12 < 256)
        msg.extend_from_slice(&int_msg(4, false, true));

        // Member "y": name, offset, datatype
        msg.extend_from_slice(b"y\0");
        msg.push(0x04); // offset = 4
        msg.extend_from_slice(&int_msg(8, true, true));

        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Compound { ref fields, size } => {
                assert_eq!(size, 12);
                assert_eq!(fields.len(), 2);

                assert_eq!(fields[0].name, "x");
                assert_eq!(fields[0].offset, 0);
                assert!(matches!(
                    fields[0].datatype,
                    Datatype::Integer {
                        bits,
                        signed: false,
                        ..
                    } if bits.get() == 32
                ));

                assert_eq!(fields[1].name, "y");
                assert_eq!(fields[1].offset, 4);
                assert!(matches!(
                    fields[1].datatype,
                    Datatype::Integer {
                        bits,
                        signed: true,
                        ..
                    } if bits.get() == 64
                ));
            }
            other => panic!("expected Compound, got: {other:?}"),
        }
    }

    #[test]
    fn parse_compound_v1_with_dimensionality() {
        // Version 1 compound with one scalar member (rank=0 dimensionality).
        let num_members: u16 = 1;
        let compound_size: u32 = 4;
        let hdr = dt_header(
            COMPOUND,
            1,
            [
                num_members.to_le_bytes()[0],
                num_members.to_le_bytes()[1],
                0,
            ],
            compound_size,
        );
        let mut msg = hdr.to_vec();

        // Member name "val" + null = 4 bytes, padded to 8.
        msg.extend_from_slice(b"val\0");
        msg.extend_from_slice(&[0u8; 4]); // padding to 8 bytes

        // Byte offset (4 bytes, u32 LE): 0
        msg.extend_from_slice(&0u32.to_le_bytes());

        // Dimensionality: rank=0, reserved(3), perm(4), reserved(4) = 12 bytes
        msg.push(0); // rank
        msg.extend_from_slice(&[0u8; 3]); // reserved
        msg.extend_from_slice(&[0u8; 4]); // dimension permutation
        msg.extend_from_slice(&[0u8; 4]); // reserved
        // No dimension sizes (rank=0)

        // Member datatype: u32 LE
        msg.extend_from_slice(&int_msg(4, false, true));

        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Compound { ref fields, size } => {
                assert_eq!(size, 4);
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].name, "val");
                assert_eq!(fields[0].offset, 0);
                assert!(matches!(
                    fields[0].datatype,
                    Datatype::Integer {
                        bits,
                        signed: false,
                        ..
                    } if bits.get() == 32
                ));
            }
            other => panic!("expected Compound, got: {other:?}"),
        }
    }

    // -- Class 7: Reference --------------------------------------------------

    #[test]
    fn parse_object_reference() {
        let hdr = dt_header(REFERENCE, 1, [0x00, 0, 0], 8);
        let msg = hdr.to_vec();
        let dt = parse_datatype(&msg).unwrap();
        assert_eq!(dt, Datatype::Reference(ReferenceType::Object));
    }

    #[test]
    fn parse_region_reference() {
        let hdr = dt_header(REFERENCE, 1, [0x01, 0, 0], 12);
        let msg = hdr.to_vec();
        let dt = parse_datatype(&msg).unwrap();
        assert_eq!(dt, Datatype::Reference(ReferenceType::Region));
    }

    // -- Class 8: Enum -------------------------------------------------------

    #[test]
    fn parse_enum_v3_two_members() {
        // Enum with u8 base, 2 members: RED=0, GREEN=1
        let num_members: u16 = 2;
        let hdr = dt_header(
            ENUM,
            3,
            [
                num_members.to_le_bytes()[0],
                num_members.to_le_bytes()[1],
                0,
            ],
            1, // enum element size = base size = 1
        );
        let mut msg = hdr.to_vec();

        // Base type: u8 unsigned LE
        msg.extend_from_slice(&int_msg(1, false, true));

        // Names (version 3: no padding)
        msg.extend_from_slice(b"RED\0");
        msg.extend_from_slice(b"GREEN\0");

        // Values: 0, 1 (each 1 byte)
        msg.push(0x00);
        msg.push(0x01);

        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Enum {
                ref base,
                ref members,
            } => {
                assert!(matches!(
                    base.as_ref(),
                    Datatype::Integer {
                        bits,
                        signed: false,
                        ..
                    } if bits.get() == 8
                ));
                assert_eq!(members.len(), 2);
                assert_eq!(members[0].name, "RED");
                assert_eq!(members[0].value, 0);
                assert_eq!(members[1].name, "GREEN");
                assert_eq!(members[1].value, 1);
            }
            other => panic!("expected Enum, got: {other:?}"),
        }
    }

    #[test]
    fn parse_enum_signed_base() {
        // Enum with i32 base, 1 member: NEGATIVE = -42
        let num_members: u16 = 1;
        let hdr = dt_header(
            ENUM,
            3,
            [
                num_members.to_le_bytes()[0],
                num_members.to_le_bytes()[1],
                0,
            ],
            4,
        );
        let mut msg = hdr.to_vec();
        msg.extend_from_slice(&int_msg(4, true, true)); // i32 LE
        msg.extend_from_slice(b"NEGATIVE\0");
        msg.extend_from_slice(&(-42i32).to_le_bytes());

        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Enum { ref members, .. } => {
                assert_eq!(members.len(), 1);
                assert_eq!(members[0].name, "NEGATIVE");
                assert_eq!(members[0].value, -42);
            }
            other => panic!("expected Enum, got: {other:?}"),
        }
    }

    // -- Class 9: Variable-length --------------------------------------------

    #[test]
    fn parse_vl_string_utf8() {
        // VL string: type=1, charset=1 (UTF-8) in flags byte 1
        let hdr = dt_header(VARIABLE_LENGTH, 1, [0x01, 0x01, 0], 16);
        let msg = hdr.to_vec();
        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::VariableString { encoding } => {
                assert_eq!(encoding, StringEncoding::Utf8);
            }
            other => panic!("expected VariableString, got: {other:?}"),
        }
    }

    #[test]
    fn parse_vl_sequence() {
        // VL sequence: type=0, base type = u32 LE
        let hdr = dt_header(VARIABLE_LENGTH, 1, [0x00, 0, 0], 16);
        let mut msg = hdr.to_vec();
        msg.extend_from_slice(&int_msg(4, false, true));
        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::VarLen { ref base } => {
                assert!(matches!(
                    base.as_ref(),
                    Datatype::Integer {
                        bits,
                        signed: false,
                        ..
                    } if bits.get() == 32
                ));
            }
            other => panic!("expected VarLen, got: {other:?}"),
        }
    }

    // -- Class 10: Array -----------------------------------------------------

    #[test]
    fn parse_array_v3_2d() {
        // 2-D array [3][5] of f64 LE
        let hdr = dt_header(ARRAY, 3, [0, 0, 0], 120); // 3*5*8=120
        let mut msg = hdr.to_vec();
        msg.push(2); // rank = 2
        msg.extend_from_slice(&3u32.to_le_bytes()); // dim 0
        msg.extend_from_slice(&5u32.to_le_bytes()); // dim 1
        // Base type: f64 LE (8 bytes + 12 props = 20 bytes)
        let base_hdr = dt_header(FLOATING_POINT, 1, [0x00, 0, 0], 8);
        msg.extend_from_slice(&base_hdr);
        msg.extend_from_slice(&[0u8; 12]);

        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Array { ref base, ref dims } => {
                assert_eq!(dims, &[3, 5]);
                assert!(matches!(
                    base.as_ref(),
                    Datatype::Float { bits, .. } if bits.get() == 64
                ));
            }
            other => panic!("expected Array, got: {other:?}"),
        }
    }

    #[test]
    fn parse_array_v2_with_reserved() {
        // Version 2: rank + 3 reserved + dims + perm_indices + base
        let hdr = dt_header(ARRAY, 2, [0, 0, 0], 40); // [10] of u32 = 40
        let mut msg = hdr.to_vec();
        msg.push(1); // rank = 1
        msg.extend_from_slice(&[0u8; 3]); // reserved
        msg.extend_from_slice(&10u32.to_le_bytes()); // dim 0
        msg.extend_from_slice(&0u32.to_le_bytes()); // perm index (deprecated)
        msg.extend_from_slice(&int_msg(4, false, true)); // base: u32 LE

        let dt = parse_datatype(&msg).unwrap();
        match dt {
            Datatype::Array { ref base, ref dims } => {
                assert_eq!(dims, &[10]);
                assert!(matches!(
                    base.as_ref(),
                    Datatype::Integer {
                        bits,
                        signed: false,
                        ..
                    } if bits.get() == 32
                ));
            }
            other => panic!("expected Array, got: {other:?}"),
        }
    }

    // -- Error paths ---------------------------------------------------------

    #[test]
    fn truncated_header_rejected() {
        let err = parse_datatype(&[0x00, 0x00, 0x00]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat { .. }));
    }

    #[test]
    fn unknown_class_rejected() {
        let hdr = dt_header(15, 1, [0, 0, 0], 4);
        let err = parse_datatype(&hdr).unwrap_err();
        assert!(matches!(err, Error::UnsupportedFeature { .. }));
    }

    #[test]
    fn time_class_rejected() {
        let hdr = dt_header(TIME, 1, [0, 0, 0], 4);
        let err = parse_datatype(&hdr).unwrap_err();
        assert!(matches!(err, Error::UnsupportedFeature { .. }));
    }

    // -- Helpers unit tests --------------------------------------------------

    #[test]
    fn read_uint_le_widths() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(read_uint_le(&data, 0), 0);
        assert_eq!(read_uint_le(&data, 1), 0x01);
        assert_eq!(read_uint_le(&data, 2), 0x0201);
        assert_eq!(read_uint_le(&data, 3), 0x0003_0201);
        assert_eq!(read_uint_le(&data, 4), 0x0403_0201);
        assert_eq!(read_uint_le(&data, 8), 0x0807_0605_0403_0201);
    }

    #[test]
    fn read_uint_be_widths() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(read_uint_be(&data, 0), 0);
        assert_eq!(read_uint_be(&data, 1), 0x01);
        assert_eq!(read_uint_be(&data, 2), 0x0102);
        assert_eq!(read_uint_be(&data, 4), 0x0102_0304);
        assert_eq!(read_uint_be(&data, 8), 0x0102_0304_0506_0708);
    }

    #[test]
    fn sign_extend_values() {
        // Positive i8 (0x7F = 127)
        assert_eq!(sign_extend(0x7F, 1), 127);
        // Negative i8 (0xFF = -1)
        assert_eq!(sign_extend(0xFF, 1), -1);
        // Negative i16 (0xFFFE = -2)
        assert_eq!(sign_extend(0xFFFE, 2), -2);
        // Positive i32
        assert_eq!(sign_extend(42, 4), 42);
        // Negative i32 (0xFFFF_FFD6 = -42)
        assert_eq!(sign_extend(0xFFFF_FFD6, 4), -42);
    }

    #[test]
    fn member_offset_byte_count_thresholds() {
        assert_eq!(member_offset_byte_count(0), 1);
        assert_eq!(member_offset_byte_count(255), 1);
        assert_eq!(member_offset_byte_count(256), 2);
        assert_eq!(member_offset_byte_count(65_535), 2);
        assert_eq!(member_offset_byte_count(65_536), 3);
        assert_eq!(member_offset_byte_count(16_777_215), 3);
        assert_eq!(member_offset_byte_count(16_777_216), 4);
    }
}
