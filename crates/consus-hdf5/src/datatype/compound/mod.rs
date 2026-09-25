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

mod classes;
mod members;
mod parse;
mod scalar;
#[cfg(test)]
mod tests;

pub(crate) use classes::{parse_array, parse_enum, parse_reference, parse_variable_length};
#[cfg(test)]
pub(crate) use members::member_offset_byte_count;
pub(crate) use members::parse_compound;
pub use parse::parse_datatype;
pub(crate) use parse::parse_datatype_inner;
pub(crate) use scalar::{
    charset_to_encoding, parse_bitfield, parse_fixed_point, parse_floating_point, parse_opaque,
    parse_string, read_uint_be, read_uint_le, sign_extend,
};
