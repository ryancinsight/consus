#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};

#[cfg(feature = "alloc")]
use consus_core::{Error, Result, Shape};

#[cfg(feature = "alloc")]
use super::decode_attribute_value;

/// Parsed HDF5 attribute from an attribute header message.
///
/// Contains the attribute name, decoded datatype descriptor, dataspace
/// (shape), and the raw attribute data bytes. The caller is responsible
/// for interpreting `raw_data` according to `datatype` and `shape`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct Hdf5Attribute {
    /// Attribute name.
    pub name: String,
    /// Attribute datatype (canonical representation).
    pub datatype: consus_core::Datatype,
    /// Attribute dataspace (shape).
    pub shape: Shape,
    /// Raw attribute data bytes (uninterpreted).
    pub raw_data: Vec<u8>,
    /// Character encoding of the name (0 = ASCII, 1 = UTF-8).
    /// Versions 1 and 2 default to 0 (ASCII).
    pub name_encoding: u8,
    /// Creation order index, if tracked by the containing object header.
    /// Populated by the caller; not present in the attribute message itself.
    pub creation_order: Option<u16>,
}

/// Alignment boundary for version 1 attribute message components.
const V1_ALIGNMENT: usize = 8;

/// Round `offset` up to the next multiple of `alignment`.
///
/// ## Invariant
///
/// `align_up(n, a) % a == 0` for all `n` and power-of-two `a`.
const fn align_up(offset: usize, alignment: usize) -> usize {
    let mask = alignment - 1;
    (offset + mask) & !mask
}

/// Minimum header size shared by all attribute message versions.
/// version(1) + flags/reserved(1) + name_size(2) + dt_size(2) + ds_size(2) = 8.
const MIN_HEADER_SIZE: usize = 8;

/// Extended header size for version 3 (includes encoding byte).
/// version(1) + flags(1) + name_size(2) + dt_size(2) + ds_size(2) + encoding(1) = 9.
const V3_HEADER_SIZE: usize = 9;

#[cfg(feature = "alloc")]
impl Hdf5Attribute {
    /// Parse an attribute from raw header message bytes.
    ///
    /// Dispatches to the version-specific parser based on the first byte.
    ///
    /// ## Arguments
    ///
    /// - `data`: raw bytes of the attribute header message payload.
    /// - `ctx`: parsing context (offset/length sizes from the superblock).
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the version byte is unsupported (not 1, 2, or 3).
    /// - [`Error::InvalidFormat`] if the message is truncated.
    pub fn parse(data: &[u8], ctx: &crate::address::ParseContext) -> Result<Self> {
        if data.len() < MIN_HEADER_SIZE {
            return Err(Error::InvalidFormat {
                message: String::from("attribute message too short for header"),
            });
        }

        let version = data[0];
        match version {
            1 => Self::parse_v1(data, ctx),
            2 => Self::parse_v2(data, ctx),
            3 => Self::parse_v3(data, ctx),
            _ => Err(Error::InvalidFormat {
                message: alloc::format!("unsupported attribute message version: {version}"),
            }),
        }
    }

    /// Parse a version 1 attribute message.
    ///
    /// Components (name, datatype, dataspace) are each padded to 8-byte boundaries.
    fn parse_v1(data: &[u8], ctx: &crate::address::ParseContext) -> Result<Self> {
        let name_size = LittleEndian::read_u16(&data[2..4]) as usize;
        let dt_size = LittleEndian::read_u16(&data[4..6]) as usize;
        let ds_size = LittleEndian::read_u16(&data[6..8]) as usize;

        let mut cursor = MIN_HEADER_SIZE;

        let name = Self::read_null_terminated_name(data, cursor, name_size)?;
        cursor += align_up(name_size, V1_ALIGNMENT);

        let datatype = Self::read_datatype(data, cursor, dt_size, &ctx.budget)?;
        cursor += align_up(dt_size, V1_ALIGNMENT);

        let shape = Self::read_dataspace(data, cursor, ds_size, ctx)?;
        cursor += align_up(ds_size, V1_ALIGNMENT);

        let raw_data = Self::read_raw_data(data, cursor);

        Ok(Self {
            name,
            datatype,
            shape,
            raw_data,
            name_encoding: 0,
            creation_order: None,
        })
    }

    /// Parse a version 2 attribute message.
    ///
    /// Identical field layout to v1 but without padding between components.
    /// Byte 1 is a flags byte instead of reserved.
    fn parse_v2(data: &[u8], ctx: &crate::address::ParseContext) -> Result<Self> {
        let _flags = data[1];
        let name_size = LittleEndian::read_u16(&data[2..4]) as usize;
        let dt_size = LittleEndian::read_u16(&data[4..6]) as usize;
        let ds_size = LittleEndian::read_u16(&data[6..8]) as usize;

        let mut cursor = MIN_HEADER_SIZE;

        let name = Self::read_null_terminated_name(data, cursor, name_size)?;
        cursor += name_size;

        let datatype = Self::read_datatype(data, cursor, dt_size, &ctx.budget)?;
        cursor += dt_size;

        let shape = Self::read_dataspace(data, cursor, ds_size, ctx)?;
        cursor += ds_size;

        let raw_data = Self::read_raw_data(data, cursor);

        Ok(Self {
            name,
            datatype,
            shape,
            raw_data,
            name_encoding: 0,
            creation_order: None,
        })
    }

    /// Parse a version 3 attribute message.
    ///
    /// Name is NOT null-terminated; length is exact byte count.
    /// Includes a character encoding byte (0=ASCII, 1=UTF-8).
    /// No padding between components.
    fn parse_v3(data: &[u8], ctx: &crate::address::ParseContext) -> Result<Self> {
        if data.len() < V3_HEADER_SIZE {
            return Err(Error::InvalidFormat {
                message: String::from("attribute v3 message too short for header"),
            });
        }

        let _flags = data[1];
        let name_size = LittleEndian::read_u16(&data[2..4]) as usize;
        let dt_size = LittleEndian::read_u16(&data[4..6]) as usize;
        let ds_size = LittleEndian::read_u16(&data[6..8]) as usize;
        let name_encoding = data[8];

        let mut cursor = V3_HEADER_SIZE;

        if cursor + name_size > data.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "attribute v3 name overflows message: need {} bytes at offset {}, have {}",
                    name_size,
                    cursor,
                    data.len(),
                ),
            });
        }
        let name_bytes = &data[cursor..cursor + name_size];
        let end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        let name = core::str::from_utf8(&name_bytes[..end])
            .map_err(|_| Error::InvalidFormat {
                message: String::from("attribute v3 name is not valid UTF-8"),
            })?
            .into();
        cursor += name_size;

        let datatype = Self::read_datatype(data, cursor, dt_size, &ctx.budget)?;
        cursor += dt_size;

        let shape = Self::read_dataspace(data, cursor, ds_size, ctx)?;
        cursor += ds_size;

        let raw_data = Self::read_raw_data(data, cursor);

        Ok(Self {
            name,
            datatype,
            shape,
            raw_data,
            name_encoding,
            creation_order: None,
        })
    }

    /// Read a null-terminated name from `data[offset..offset+size]`.
    ///
    /// The `size` field includes the null terminator. The returned string
    /// excludes the terminator. Validates UTF-8 encoding.
    fn read_null_terminated_name(data: &[u8], offset: usize, size: usize) -> Result<String> {
        if size == 0 {
            return Ok(String::new());
        }
        if offset + size > data.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "attribute name overflows message: need {} bytes at offset {}, have {}",
                    size,
                    offset,
                    data.len(),
                ),
            });
        }
        let name_bytes = &data[offset..offset + size];
        let end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        core::str::from_utf8(&name_bytes[..end])
            .map(String::from)
            .map_err(|_| Error::InvalidFormat {
                message: String::from("attribute name is not valid UTF-8"),
            })
    }

    /// Read and parse a datatype from `data[offset..offset+size]`.
    ///
    /// Delegates to `crate::datatype::compound::parse_datatype` which handles
    /// all HDF5 datatype classes.
    fn read_datatype(
        data: &[u8],
        offset: usize,
        size: usize,
        budget: &consus_core::ParseBudget,
    ) -> Result<consus_core::Datatype> {
        if size == 0 {
            return Err(Error::InvalidFormat {
                message: String::from("attribute datatype has zero size"),
            });
        }
        if offset + size > data.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "attribute datatype overflows message: need {} bytes at offset {}, have {}",
                    size,
                    offset,
                    data.len(),
                ),
            });
        }
        let dt_bytes = &data[offset..offset + size];
        crate::datatype::compound::parse_datatype(dt_bytes, budget)
    }

    /// Read and parse a dataspace from `data[offset..offset+size]`.
    ///
    /// Delegates to `crate::dataspace::parse_dataspace`. A zero-size dataspace
    /// is treated as scalar (rank-0).
    fn read_dataspace(
        data: &[u8],
        offset: usize,
        size: usize,
        ctx: &crate::address::ParseContext,
    ) -> Result<Shape> {
        if size == 0 {
            return Ok(Shape::scalar());
        }
        if offset + size > data.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "attribute dataspace overflows message: need {} bytes at offset {}, have {}",
                    size,
                    offset,
                    data.len(),
                ),
            });
        }
        let ds_bytes = &data[offset..offset + size];
        crate::dataspace::parse_dataspace(ds_bytes, ctx.offset_size)
    }

    /// Read raw data bytes from `data[offset..]`.
    ///
    /// Returns all remaining bytes as the attribute's raw data payload.
    fn read_raw_data(data: &[u8], offset: usize) -> Vec<u8> {
        if offset >= data.len() {
            Vec::new()
        } else {
            Vec::from(&data[offset..])
        }
    }

    /// Decode the raw attribute data bytes into a typed [`consus_core::AttributeValue`].
    ///
    /// Interprets `raw_data` according to `datatype` and `shape`.
    ///
    /// ## Errors
    ///
    /// - [`consus_core::Error::UnsupportedFeature`] for variable-length or
    ///   compound types that require heap traversal.
    /// - [`consus_core::Error::InvalidFormat`] if `raw_data` is too short.
    pub fn decode_value(&self) -> consus_core::Result<consus_core::AttributeValue> {
        decode_attribute_value(&self.raw_data, &self.datatype, &self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_cases() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(16, 8), 16);
        assert_eq!(align_up(13, 8), 16);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn reject_too_short() {
        let data = [0u8; 4];
        let ctx = crate::address::ParseContext::new(8, 8);
        let err = Hdf5Attribute::parse(&data, &ctx).unwrap_err();
        match err {
            consus_core::Error::InvalidFormat { message } => {
                assert!(message.contains("too short"));
            }
            _ => panic!("expected InvalidFormat, got: {err:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn reject_unsupported_version() {
        let mut data = [0u8; 8];
        data[0] = 4;
        let ctx = crate::address::ParseContext::new(8, 8);
        let err = Hdf5Attribute::parse(&data, &ctx).unwrap_err();
        match err {
            consus_core::Error::InvalidFormat { message } => {
                assert!(message.contains("unsupported"));
            }
            _ => panic!("expected InvalidFormat, got: {err:?}"),
        }
    }
}
