//! Unit tests for the datatype parsers, split from the former monolith.

mod scalar;
use super::super::classes::{
    ARRAY, BITFIELD, COMPOUND, ENUM, FIXED_POINT, FLOATING_POINT, OPAQUE, REFERENCE, STRING, TIME,
    VARIABLE_LENGTH,
};
use super::*;
use consus_core::{ByteOrder, Datatype, Error, ParseBudget, ReferenceType, StringEncoding};

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
mod classes;
mod members;
