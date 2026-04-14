//! HDF5 datatype ↔ canonical datatype mapping.
//!
//! ## Specification
//!
//! HDF5 datatypes are encoded in header messages (type 0x0003).
//! The encoding begins with a 4-byte class+version+flags field
//! followed by a 4-byte size field, then class-specific properties.
//!
//! ### Datatype Classes
//!
//! | Class | Value | Description |
//! |-------|-------|-------------|
//! | Fixed-point | 0 | Integer types |
//! | Floating-point | 1 | IEEE 754 |
//! | Time | 2 | (deprecated) |
//! | String | 3 | Fixed or variable length |
//! | Bitfield | 4 | Bit-packed |
//! | Opaque | 5 | Raw bytes |
//! | Compound | 6 | Struct-like |
//! | Reference | 7 | Object/region reference |
//! | Enum | 8 | Enumerated integers |
//! | Variable-length | 9 | VL sequences/strings |
//! | Array | 10 | Fixed-size arrays |

use consus_core::datatype::{ByteOrder, Datatype};

/// HDF5 datatype class values.
pub mod classes {
    pub const FIXED_POINT: u8 = 0;
    pub const FLOATING_POINT: u8 = 1;
    pub const TIME: u8 = 2;
    pub const STRING: u8 = 3;
    pub const BITFIELD: u8 = 4;
    pub const OPAQUE: u8 = 5;
    pub const COMPOUND: u8 = 6;
    pub const REFERENCE: u8 = 7;
    pub const ENUM: u8 = 8;
    pub const VARIABLE_LENGTH: u8 = 9;
    pub const ARRAY: u8 = 10;
}

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
