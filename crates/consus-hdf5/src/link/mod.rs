//! HDF5 link message parsing (header message type 0x0006).
//!
//! ## Specification
//!
//! Link messages appear in v2-style group object headers and encode named
//! references to other objects. Each link message is self-contained: it
//! carries the link name, type, optional creation order, and the
//! type-specific target value.
//!
//! Reference: *HDF5 File Format Specification Version 3.0*, Section IV.A.2.g
//! (<https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html>)
//!
//! ## Link Message Layout
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 1 | Version (must be 1) |
//! | 1 | 1 | Flags (see below) |
//! | var | 1 | Link type (optional, present if flags bit 3 set) |
//! | var | 8 | Creation order (optional, present if flags bit 2 set) |
//! | var | 1 | Link name character encoding (optional, present if flags bit 4 set) |
//! | var | 1/2/4/8 | Length of link name (size determined by flags bits 0–1) |
//! | var | N | Link name bytes (NOT null-terminated, N = name length) |
//! | var | var | Link value (type-specific, see below) |
//!
//! ### Flags Bits
//!
//! | Bit(s) | Meaning |
//! |--------|---------|
//! | 0–1 | Size of the "link name length" field: 0→1 byte, 1→2 bytes, 2→4 bytes, 3→8 bytes |
//! | 2 | Creation order field is present (8 bytes, little-endian u64) |
//! | 3 | Link type field is present (1 byte). If absent, the link type is hard (0). |
//! | 4 | Link name character encoding field is present (1 byte: 0=ASCII, 1=UTF-8) |
//!
//! ### Link Values
//!
//! - **Hard link (type 0):** `offset_size` bytes encoding the target object header address.
//! - **Soft link (type 1):** 2-byte little-endian length prefix followed by the target path
//!   string (NOT null-terminated).
//! - **External link (type 64):** 2-byte little-endian length prefix followed by a payload of
//!   1 flag byte, a null-terminated filename, and a null-terminated object path.

pub mod external;

#[cfg(feature = "alloc")]
use alloc::string::String;

use byteorder::{ByteOrder, LittleEndian};
use consus_core::{Error, LinkType, Result};

use crate::address::ParseContext;

// ---------------------------------------------------------------------------
// Flag-bit masks
// ---------------------------------------------------------------------------

/// Mask for the link-name-length field size (bits 0–1).
const FLAG_NAME_LENGTH_SIZE_MASK: u8 = 0x03;

/// Bit 2: creation order field is present.
const FLAG_CREATION_ORDER_PRESENT: u8 = 0x04;

/// Bit 3: link type field is present.
const FLAG_LINK_TYPE_PRESENT: u8 = 0x08;

/// Bit 4: link name character encoding field is present.
const FLAG_CHAR_ENCODING_PRESENT: u8 = 0x10;

/// Expected link message version.
const LINK_MESSAGE_VERSION: u8 = 1;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Parsed HDF5 link from a link header message (message type 0x0006).
///
/// Each instance represents a single named link in a v2-style group.
/// The variant-specific target data is stored in exactly one of the
/// optional fields (`hard_link_address`, `soft_link_target`, or
/// `external_link`), determined by `link_type`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct Hdf5Link {
    /// Link name (decoded from the name bytes according to `name_encoding`).
    pub name: String,
    /// Discriminant identifying the link variant.
    pub link_type: LinkType,
    /// Target object header address. Present only for hard links.
    pub hard_link_address: Option<u64>,
    /// Target path within this file. Present only for soft links.
    pub soft_link_target: Option<String>,
    /// External link target. Present only for external links.
    pub external_link: Option<ExternalLinkData>,
    /// Creation order index, if tracked by the group.
    pub creation_order: Option<u64>,
    /// Name character encoding byte (0 = ASCII, 1 = UTF-8).
    /// Defaults to 0 (ASCII) when the encoding field is absent.
    pub name_encoding: u8,
}

/// External link target data embedded in a link message.
///
/// The payload consists of a 1-byte flags/version field followed by two
/// null-terminated strings: the external filename and the object path
/// within that file.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ExternalLinkData {
    /// Path to the external HDF5 file.
    pub filename: String,
    /// Object path within the external file.
    pub object_path: String,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl Hdf5Link {
    /// Parse a link from raw link message bytes.
    ///
    /// # Arguments
    ///
    /// * `data` — Raw bytes of the link header message payload (after the
    ///   standard header-message envelope has been stripped).
    /// * `ctx`  — Parsing context carrying `offset_size` (needed for hard
    ///   link target addresses).
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidFormat` when:
    /// - The buffer is too short for any mandatory or flagged-present field.
    /// - The version byte is not 1.
    /// - The link type byte is in the reserved range (2–63).
    /// - A string field contains invalid UTF-8.
    pub fn parse(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        // -- Version -----------------------------------------------------------
        if data.is_empty() {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("link message is empty"),
            });
        }
        let version = data[0];
        if version != LINK_MESSAGE_VERSION {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "unsupported link message version {version}, expected {LINK_MESSAGE_VERSION}"
                ),
            });
        }

        // -- Flags -------------------------------------------------------------
        if data.len() < 2 {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("link message truncated at flags byte"),
            });
        }
        let flags = data[1];
        let mut cursor: usize = 2;

        // -- Optional: link type -----------------------------------------------
        let link_type_byte = if flags & FLAG_LINK_TYPE_PRESENT != 0 {
            ensure_remaining(data, cursor, 1, "link type")?;
            let b = data[cursor];
            cursor += 1;
            b
        } else {
            0 // absent → hard link
        };

        let link_type = decode_link_type(link_type_byte)?;

        // -- Optional: creation order ------------------------------------------
        let creation_order = if flags & FLAG_CREATION_ORDER_PRESENT != 0 {
            ensure_remaining(data, cursor, 8, "creation order")?;
            let val = LittleEndian::read_u64(&data[cursor..cursor + 8]);
            cursor += 8;
            Some(val)
        } else {
            None
        };

        // -- Optional: name encoding -------------------------------------------
        let name_encoding = if flags & FLAG_CHAR_ENCODING_PRESENT != 0 {
            ensure_remaining(data, cursor, 1, "name character encoding")?;
            let enc = data[cursor];
            cursor += 1;
            enc
        } else {
            0 // default: ASCII
        };

        // -- Link name length --------------------------------------------------
        let name_len_field_size = name_length_field_size(flags);
        ensure_remaining(data, cursor, name_len_field_size, "link name length")?;
        let name_length = read_name_length(&data[cursor..], name_len_field_size)?;
        cursor += name_len_field_size;

        // -- Link name bytes ---------------------------------------------------
        let name_length_usize = name_length as usize;
        ensure_remaining(data, cursor, name_length_usize, "link name")?;
        let name =
            core::str::from_utf8(&data[cursor..cursor + name_length_usize]).map_err(|_| {
                Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: String::from("link name is not valid UTF-8"),
                }
            })?;
        let name = String::from(name);
        cursor += name_length_usize;

        // -- Link value (type-specific) ----------------------------------------
        let remaining = &data[cursor..];
        let (hard_link_address, soft_link_target, external_link) = match link_type {
            LinkType::Hard => {
                let addr = parse_hard_link_value(remaining, ctx)?;
                (Some(addr), None, None)
            }
            LinkType::Soft => {
                let target = parse_soft_link_value(remaining)?;
                (None, Some(target), None)
            }
            LinkType::External => {
                let ext = parse_external_link_value(remaining)?;
                (None, None, Some(ext))
            }
        };

        Ok(Self {
            name,
            link_type,
            hard_link_address,
            soft_link_target,
            external_link,
            creation_order,
            name_encoding,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map the 2-bit name-length-size field (flags bits 0–1) to the byte width
/// of the "link name length" integer.
///
/// | Field value | Width |
/// |-------------|-------|
/// | 0           | 1     |
/// | 1           | 2     |
/// | 2           | 4     |
/// | 3           | 8     |
fn name_length_field_size(flags: u8) -> usize {
    match flags & FLAG_NAME_LENGTH_SIZE_MASK {
        0 => 1,
        1 => 2,
        2 => 4,
        _ => 8, // 3
    }
}

/// Read the link name length from `buf` using the determined field width.
fn read_name_length(buf: &[u8], size: usize) -> Result<u64> {
    let val = match size {
        1 => u64::from(buf[0]),
        2 => u64::from(LittleEndian::read_u16(buf)),
        4 => u64::from(LittleEndian::read_u32(buf)),
        8 => LittleEndian::read_u64(buf),
        _ => {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!("invalid link name length field size: {size}"),
            });
        }
    };
    Ok(val)
}

/// Decode the 1-byte link type field into `LinkType`.
///
/// | Byte value | Type     |
/// |------------|----------|
/// | 0          | Hard     |
/// | 1          | Soft     |
/// | 2–63       | Reserved |
/// | 64         | External |
/// | 65–255     | Reserved |
fn decode_link_type(byte: u8) -> Result<LinkType> {
    match byte {
        0 => Ok(LinkType::Hard),
        1 => Ok(LinkType::Soft),
        64 => Ok(LinkType::External),
        other => Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!("reserved or unknown link type: {other}"),
        }),
    }
}

/// Parse a hard link value: `offset_size` bytes encoding the target object
/// header address.
fn parse_hard_link_value(data: &[u8], ctx: &ParseContext) -> Result<u64> {
    let s = ctx.offset_bytes();
    if data.len() < s {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "hard link value too short: need {s} bytes, have {}",
                data.len()
            ),
        });
    }
    Ok(ctx.read_offset(data))
}

/// Parse a soft link value: 2-byte little-endian length prefix followed by
/// the target path string (NOT null-terminated).
#[cfg(feature = "alloc")]
fn parse_soft_link_value(data: &[u8]) -> Result<String> {
    if data.len() < 2 {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: String::from("soft link value too short for length prefix"),
        });
    }
    let target_len = LittleEndian::read_u16(data) as usize;
    if data.len() < 2 + target_len {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "soft link value truncated: declared {target_len} bytes, have {}",
                data.len() - 2
            ),
        });
    }
    let target =
        core::str::from_utf8(&data[2..2 + target_len]).map_err(|_| Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: String::from("soft link target path is not valid UTF-8"),
        })?;
    Ok(String::from(target))
}

/// Parse an external link value: 2-byte little-endian length prefix followed
/// by a payload containing 1 flag byte, a null-terminated filename, and a
/// null-terminated object path.
#[cfg(feature = "alloc")]
fn parse_external_link_value(data: &[u8]) -> Result<ExternalLinkData> {
    if data.len() < 2 {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: String::from("external link value too short for length prefix"),
        });
    }
    let payload_len = LittleEndian::read_u16(data) as usize;
    let payload_start = 2;
    if data.len() < payload_start + payload_len {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "external link value truncated: declared {payload_len} bytes, have {}",
                data.len() - payload_start
            ),
        });
    }
    let payload = &data[payload_start..payload_start + payload_len];
    let target = external::parse_external_link_value(payload)?;
    Ok(ExternalLinkData {
        filename: target.file_path,
        object_path: target.object_path,
    })
}

/// Assert that `data[cursor..cursor+need]` is within bounds, returning
/// `Error::InvalidFormat` with a contextual message on failure.
fn ensure_remaining(data: &[u8], cursor: usize, need: usize, field: &str) -> Result<()> {
    if data.len() < cursor + need {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::format!(
                "link message truncated at {field}: need {need} bytes at offset {cursor}, \
                 buffer length is {}",
                data.len()
            ),
        });
    }
    Ok(())
}
