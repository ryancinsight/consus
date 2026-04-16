//! HDF5 datatype <-> canonical datatype mapping.
//!
//! ## Specification
//!
//! HDF5 datatypes are encoded in header messages (type 0x0003).
//! The encoding begins with a 4-byte class+version+flags field
//! followed by a 4-byte size field, then class-specific properties.

pub mod classes;

#[cfg(feature = "alloc")]
pub mod compound;

use consus_core::{ByteOrder, Datatype};

/// Extract the datatype class from the first 4 bytes of a datatype message.
///
/// Bits 0-3 of byte 0 contain the class.
pub fn datatype_class(header_byte: u8) -> u8 {
    header_byte & 0x0F
}

/// Extract the byte order from a fixed-point or floating-point datatype.
///
/// Bit 0 of the class bit field (bits 8-31 of the 4-byte header):
/// 0 = little-endian, 1 = big-endian.
pub fn byte_order_from_flags(flags_byte: u8) -> ByteOrder {
    if flags_byte & 0x01 == 0 {
        ByteOrder::LittleEndian
    } else {
        ByteOrder::BigEndian
    }
}

/// Map an HDF5 fixed-point (integer) datatype to canonical form.
pub fn map_fixed_point(size_bytes: usize, flags: u8) -> Datatype {
    let signed = (flags & 0x08) != 0;
    let byte_order = byte_order_from_flags(flags);
    let bits = core::num::NonZeroUsize::new(size_bytes * 8).expect("HDF5 integer size must be > 0");
    Datatype::Integer {
        bits,
        byte_order,
        signed,
    }
}

/// Map an HDF5 floating-point datatype to canonical form.
pub fn map_floating_point(size_bytes: usize, flags: u8) -> Datatype {
    let byte_order = byte_order_from_flags(flags);
    let bits = core::num::NonZeroUsize::new(size_bytes * 8).expect("HDF5 float size must be > 0");
    Datatype::Float { bits, byte_order }
}
