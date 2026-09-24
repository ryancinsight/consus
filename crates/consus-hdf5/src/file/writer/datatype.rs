//! HDF5 datatype message encoding from canonical `Datatype` values.

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};
#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};
#[cfg(feature = "alloc")]
use consus_core::{Datatype, Error, Result};

// ---------------------------------------------------------------------------
// Datatype encoding
// ---------------------------------------------------------------------------

/// Encode a canonical `Datatype` into HDF5 binary datatype message bytes.
///
/// ## Supported types
///
/// | Canonical type | HDF5 class | Encoded |
/// |---------------|-----------|---------|
/// | `Boolean` | 0 (fixed-point) | 1-byte unsigned integer |
/// | `Integer` | 0 (fixed-point) | Matching width/sign/order |
/// | `Float` | 1 (floating-point) | IEEE 754 f32 or f64 |
/// | `FixedString` | 3 (string) | Null-padded fixed-length |
///
/// Returns `Error::UnsupportedFeature` for types not yet supported
/// in the write path.
#[cfg(feature = "alloc")]
pub fn encode_datatype(dt: &Datatype) -> Result<Vec<u8>> {
    match dt {
        Datatype::Boolean => {
            // 1-byte unsigned integer
            let mut buf = vec![0u8; 12]; // header(8) + properties(4)
            buf[0] = 0x10; // class=0 (fixed-point), version=1
            buf[1] = 0x00; // LE, unsigned
            LittleEndian::write_u32(&mut buf[4..8], 1); // size = 1 byte
            LittleEndian::write_u16(&mut buf[8..10], 0); // bit offset
            LittleEndian::write_u16(&mut buf[10..12], 8); // bit precision
            Ok(buf)
        }
        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            let size = bits.get() / 8;
            let mut buf = vec![0u8; 12]; // header(8) + properties(4)
            buf[0] = 0x10; // class=0, version=1
            let mut flags = 0u8;
            if *byte_order == consus_core::ByteOrder::BigEndian {
                flags |= 0x01;
            }
            if *signed {
                flags |= 0x08;
            }
            buf[1] = flags;
            LittleEndian::write_u32(&mut buf[4..8], size as u32);
            LittleEndian::write_u16(&mut buf[8..10], 0); // bit offset
            LittleEndian::write_u16(&mut buf[10..12], (size * 8) as u16); // bit precision
            Ok(buf)
        }
        Datatype::Float { bits, byte_order } => {
            let size = bits.get() / 8;
            let mut buf = vec![0u8; 20]; // header(8) + properties(12)
            buf[0] = 0x11; // class=1 (float), version=1
            // Class bit field byte 0 (buf[1]):
            //   bit 0:   byte order (0=LE, 1=BE)
            //   bits 4-5: mantissa normalization (2 = implicit leading 1 bit, per IEEE 754)
            // Class bit field byte 1 (buf[2]): sign bit position within element.
            let be_flag: u8 = if *byte_order == consus_core::ByteOrder::BigEndian {
                0x01
            } else {
                0x00
            };
            // mantissa norm = 2 (implicit, 0x20) combined with byte-order flag.
            buf[1] = 0x20 | be_flag;
            // Sign bit location = most significant bit of the element (size*8 - 1).
            buf[2] = (size * 8 - 1) as u8;
            LittleEndian::write_u32(&mut buf[4..8], size as u32);

            // IEEE 754 properties:
            //   buf[8..10]:  bit offset of element lsb in storage (always 0)
            //   buf[10..12]: bit precision (total bits = size * 8)
            //   buf[12]:     exponent position (lsb of exponent field)
            //   buf[13]:     exponent size (number of exponent bits)
            //   buf[14]:     mantissa position (lsb of mantissa field, always 0)
            //   buf[15]:     mantissa size (number of mantissa bits)
            //   buf[16..20]: exponent bias
            match size {
                4 => {
                    // f32: exponent at bit 23, 8 bits; mantissa at bit 0, 23 bits; bias 127
                    LittleEndian::write_u16(&mut buf[8..10], 0); // bit offset
                    LittleEndian::write_u16(&mut buf[10..12], 32); // bit precision
                    buf[12] = 23; // exponent position
                    buf[13] = 8; // exponent size
                    buf[14] = 0; // mantissa position
                    buf[15] = 23; // mantissa size
                    LittleEndian::write_u32(&mut buf[16..20], 127); // exponent bias
                }
                8 => {
                    // f64: exponent at bit 52, 11 bits; mantissa at bit 0, 52 bits; bias 1023
                    LittleEndian::write_u16(&mut buf[8..10], 0);
                    LittleEndian::write_u16(&mut buf[10..12], 64);
                    buf[12] = 52;
                    buf[13] = 11;
                    buf[14] = 0;
                    buf[15] = 52;
                    LittleEndian::write_u32(&mut buf[16..20], 1023);
                }
                2 => {
                    // f16: exponent at bit 10, 5 bits; mantissa at bit 0, 10 bits; bias 15
                    LittleEndian::write_u16(&mut buf[8..10], 0);
                    LittleEndian::write_u16(&mut buf[10..12], 16);
                    buf[12] = 10;
                    buf[13] = 5;
                    buf[14] = 0;
                    buf[15] = 10;
                    LittleEndian::write_u32(&mut buf[16..20], 15);
                }
                _ => {
                    return Err(Error::UnsupportedFeature {
                        feature: alloc::format!(
                            "write-path float encoding for {}-byte floats not supported",
                            size
                        ),
                    });
                }
            }
            Ok(buf)
        }
        Datatype::FixedString { length, encoding } => {
            let mut buf = vec![0u8; 8]; // header only, no additional properties
            buf[0] = 0x13; // class=3 (string), version=1
            // flags byte 0: padding type = 1 (null-pad)
            // flags byte 1: charset (0=ASCII, 1=UTF-8)
            buf[1] = 0x01; // null-pad
            buf[2] = match encoding {
                consus_core::StringEncoding::Ascii => 0x00,
                consus_core::StringEncoding::Utf8 => 0x01,
            };
            LittleEndian::write_u32(&mut buf[4..8], *length as u32);
            Ok(buf)
        }
        Datatype::Reference(ref_type) => {
            // HDF5 reference datatype: class=7, version=1.
            // Byte 0: (version=1 << 4) | class=7 = 0x17
            // Byte 1: 0 = object reference, 1 = region reference
            // Bytes 2–3: reserved (0)
            // Bytes 4–7: size in bytes = 8 (all HDF5 object/region references are 8 bytes)
            let mut buf = vec![0u8; 8];
            buf[0] = 0x17; // class=7 (Reference), version=1
            buf[1] = match ref_type {
                consus_core::ReferenceType::Object => 0x00,
                consus_core::ReferenceType::Region => 0x01,
            };
            LittleEndian::write_u32(&mut buf[4..8], 8);
            Ok(buf)
        }
        _ => Err(Error::UnsupportedFeature {
            feature: alloc::format!("write-path encoding for datatype {dt:?} not yet supported"),
        }),
    }
}
