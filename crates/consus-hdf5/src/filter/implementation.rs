#![cfg(feature = "alloc")]
//! HDF5 filter pipeline message parsing implementation.

use alloc::string::String;
use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};
use consus_core::{Error, Result};

use super::*;

/// A single filter in the pipeline.
///
/// Represents one stage of the filter pipeline applied to chunked data.
/// Filters are applied in order during writes and in reverse during reads.
#[derive(Debug, Clone)]
pub struct Hdf5Filter {
    /// HDF5 filter identification value.
    ///
    /// Standard IDs: 1=deflate, 2=shuffle, 3=fletcher32, 4=szip,
    /// 5=nbit, 6=scaleoffset. Values ≥ 256 are user-defined.
    pub filter_id: u16,
    /// Optional filter name (present for user-defined filters, optional
    /// for predefined filters).
    pub name: Option<String>,
    /// Filter flags. Bit 0: filter is optional (dataset is accessible
    /// even if the filter is unavailable).
    pub flags: u16,
    /// Client data parameters passed to the filter function.
    ///
    /// Interpretation is filter-specific. For deflate, `client_data[0]`
    /// is the compression level (0–9).
    pub client_data: Vec<u32>,
}

/// Parsed filter pipeline message.
///
/// Contains the pipeline version and the ordered sequence of filters.
#[derive(Debug, Clone)]
pub struct Hdf5FilterPipeline {
    /// Pipeline message version (1 or 2).
    pub version: u8,
    /// Ordered filter sequence. Filters are applied in this order on
    /// write and in reverse order on read.
    pub filters: Vec<Hdf5Filter>,
}

impl Hdf5FilterPipeline {
    /// Parse a filter pipeline from raw filter pipeline message bytes.
    ///
    /// The input `data` is the raw payload of a header message with
    /// type [`FILTER_PIPELINE`](crate::object_header::message_types::FILTER_PIPELINE).
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the version is unsupported.
    /// - [`Error::InvalidFormat`] if the data is truncated or structurally
    ///   inconsistent with the declared field sizes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidFormat {
                message: String::from("filter pipeline message too short for version and count"),
            });
        }

        let version = data[0];
        let num_filters = data[1] as usize;

        match version {
            1 => Self::parse_v1(data, num_filters),
            2 => Self::parse_v2(data, num_filters),
            _ => Err(Error::InvalidFormat {
                message: alloc::format!("unsupported filter pipeline version: {version}"),
            }),
        }
    }

    /// Parse version 1 filter pipeline.
    ///
    /// Version 1 has a 6-byte reserved region after the 2-byte header,
    /// names padded to 8-byte boundaries, and client data padding when
    /// the count is odd.
    fn parse_v1(data: &[u8], num_filters: usize) -> Result<Self> {
        if data.len() < V1_PIPELINE_HEADER_SIZE {
            return Err(Error::InvalidFormat {
                message: String::from("filter pipeline v1 message truncated in header"),
            });
        }

        // Bytes 2..8 are reserved in v1.
        let mut cursor = V1_PIPELINE_HEADER_SIZE;
        let mut filters = Vec::with_capacity(num_filters);

        for i in 0..num_filters {
            let filter = parse_filter_desc_v1(data, &mut cursor, i)?;
            filters.push(filter);
        }

        Ok(Self {
            version: 1,
            filters,
        })
    }

    /// Parse version 2 filter pipeline.
    ///
    /// Version 2 omits the reserved region, name padding, and client
    /// data padding. Names are omitted entirely for predefined filters
    /// (ID < 256) when name length is 0.
    fn parse_v2(data: &[u8], num_filters: usize) -> Result<Self> {
        let mut cursor = V2_PIPELINE_HEADER_SIZE;
        let mut filters = Vec::with_capacity(num_filters);

        for i in 0..num_filters {
            let filter = parse_filter_desc_v2(data, &mut cursor, i)?;
            filters.push(filter);
        }

        Ok(Self {
            version: 2,
            filters,
        })
    }
}

/// Parse a single version 1 filter description starting at `cursor`.
///
/// Advances `cursor` past the consumed bytes including all padding.
fn parse_filter_desc_v1(data: &[u8], cursor: &mut usize, index: usize) -> Result<Hdf5Filter> {
    let pos = *cursor;

    if pos + FILTER_DESC_HEADER_SIZE > data.len() {
        return Err(Error::InvalidFormat {
            message: alloc::format!(
                "filter pipeline v1: filter {index} description header \
                     truncated at offset {pos}"
            ),
        });
    }

    let filter_id = LittleEndian::read_u16(&data[pos..pos + 2]);
    let name_length = LittleEndian::read_u16(&data[pos + 2..pos + 4]) as usize;
    let flags = LittleEndian::read_u16(&data[pos + 4..pos + 6]);
    let num_client_data = LittleEndian::read_u16(&data[pos + 6..pos + 8]) as usize;

    *cursor = pos + FILTER_DESC_HEADER_SIZE;

    // Parse name (null-terminated, padded to 8-byte boundary).
    let name = if name_length > 0 {
        if *cursor + name_length > data.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "filter pipeline v1: filter {index} name truncated \
                         (need {name_length} bytes at offset {})",
                    *cursor
                ),
            });
        }

        let name_bytes = &data[*cursor..*cursor + name_length];
        // Find null terminator; the name_length includes it.
        let name_end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_length);
        let name_str =
            core::str::from_utf8(&name_bytes[..name_end]).map_err(|_| Error::InvalidFormat {
                message: alloc::format!(
                    "filter pipeline v1: filter {index} name is not valid UTF-8"
                ),
            })?;

        // Advance past the name padded to 8-byte boundary.
        let padded_name_length = align_up(name_length, V1_NAME_ALIGNMENT);
        *cursor += padded_name_length;

        Some(String::from(name_str))
    } else {
        None
    };

    // Parse client data values (N × u32 little-endian).
    let client_data_bytes = num_client_data * 4;
    if *cursor + client_data_bytes > data.len() {
        return Err(Error::InvalidFormat {
            message: alloc::format!(
                "filter pipeline v1: filter {index} client data truncated \
                     (need {client_data_bytes} bytes at offset {})",
                *cursor
            ),
        });
    }

    let mut client_data = Vec::with_capacity(num_client_data);
    for j in 0..num_client_data {
        let off = *cursor + j * 4;
        client_data.push(LittleEndian::read_u32(&data[off..off + 4]));
    }
    *cursor += client_data_bytes;

    // Version 1: pad client data to even count (4 bytes padding if odd).
    if num_client_data % 2 != 0 {
        *cursor += 4;
    }

    Ok(Hdf5Filter {
        filter_id,
        name,
        flags,
        client_data,
    })
}

/// Parse a single version 2 filter description starting at `cursor`.
///
/// Version 2 has no name padding and no client data padding.
/// Predefined filters (ID < 256) omit the name when name_length is 0.
fn parse_filter_desc_v2(data: &[u8], cursor: &mut usize, index: usize) -> Result<Hdf5Filter> {
    let pos = *cursor;

    if pos + FILTER_DESC_HEADER_SIZE > data.len() {
        return Err(Error::InvalidFormat {
            message: alloc::format!(
                "filter pipeline v2: filter {index} description header \
                     truncated at offset {pos}"
            ),
        });
    }

    let filter_id = LittleEndian::read_u16(&data[pos..pos + 2]);
    let name_length = LittleEndian::read_u16(&data[pos + 2..pos + 4]) as usize;
    let flags = LittleEndian::read_u16(&data[pos + 4..pos + 6]);
    let num_client_data = LittleEndian::read_u16(&data[pos + 6..pos + 8]) as usize;

    *cursor = pos + FILTER_DESC_HEADER_SIZE;

    // Parse name (no padding in v2).
    let name = if name_length > 0 {
        if *cursor + name_length > data.len() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "filter pipeline v2: filter {index} name truncated \
                         (need {name_length} bytes at offset {})",
                    *cursor
                ),
            });
        }

        let name_bytes = &data[*cursor..*cursor + name_length];
        let name_end = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_length);
        let name_str =
            core::str::from_utf8(&name_bytes[..name_end]).map_err(|_| Error::InvalidFormat {
                message: alloc::format!(
                    "filter pipeline v2: filter {index} name is not valid UTF-8"
                ),
            })?;

        *cursor += name_length;
        Some(String::from(name_str))
    } else {
        None
    };

    // Parse client data values.
    let client_data_bytes = num_client_data * 4;
    if *cursor + client_data_bytes > data.len() {
        return Err(Error::InvalidFormat {
            message: alloc::format!(
                "filter pipeline v2: filter {index} client data truncated \
                     (need {client_data_bytes} bytes at offset {})",
                *cursor
            ),
        });
    }

    let mut client_data = Vec::with_capacity(num_client_data);
    for j in 0..num_client_data {
        let off = *cursor + j * 4;
        client_data.push(LittleEndian::read_u32(&data[off..off + 4]));
    }
    *cursor += client_data_bytes;

    // No padding in version 2.

    Ok(Hdf5Filter {
        filter_id,
        name,
        flags,
        client_data,
    })
}

/// Align `value` upward to the nearest multiple of `alignment`.
///
/// ## Invariant
///
/// `alignment` must be a power of two and non-zero.
/// `align_up(n, a) % a == 0` for all valid inputs.
fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    let mask = alignment - 1;
    (value + mask) & !mask
}
