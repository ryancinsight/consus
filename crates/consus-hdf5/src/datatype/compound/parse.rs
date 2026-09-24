//! Top-level datatype message dispatch (`parse_datatype`) and shared byte helpers.

use super::super::classes::{
    ARRAY, BITFIELD, COMPOUND, ENUM, FIXED_POINT, FLOATING_POINT, OPAQUE, REFERENCE, STRING, TIME,
    VARIABLE_LENGTH,
};
use super::{
    parse_array, parse_bitfield, parse_compound, parse_enum, parse_fixed_point,
    parse_floating_point, parse_opaque, parse_reference, parse_string, parse_variable_length,
};
use alloc::{format, string::String};
use consus_core::{Datatype, Error, ParseBudget, Result};

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
pub fn parse_datatype(data: &[u8], budget: &ParseBudget) -> Result<Datatype> {
    let (dt, _consumed) = parse_datatype_inner(data, budget, 0)?;
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
pub(crate) fn parse_datatype_inner(
    data: &[u8],
    budget: &ParseBudget,
    depth: u16,
) -> Result<(Datatype, usize)> {
    // Compound, enum, variable-length, and array datatypes each re-enter this
    // function, so a nested message is input-driven recursion. Rust performs
    // no tail-call elimination and a stack overflow is an uncatchable abort,
    // so every arm of the cycle passes through this one bound.
    let depth = budget.descend(depth, "datatype nesting depth")?;

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
        COMPOUND => parse_compound(size, flags, props, version, budget, depth)?,
        REFERENCE => parse_reference(flags)?,
        ENUM => parse_enum(flags, props, version, budget, depth)?,
        VARIABLE_LENGTH => parse_variable_length(flags, props, budget, depth)?,
        ARRAY => parse_array(props, version, budget, depth)?,
        _ => {
            return Err(Error::UnsupportedFeature {
                feature: format!("datatype class {class}"),
            });
        }
    };

    Ok((dt, 8 + props_consumed))
}
