//! HDF5 datatype class constants.
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
