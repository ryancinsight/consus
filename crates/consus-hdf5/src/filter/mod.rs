//! HDF5 filter pipeline message parsing (header message type 0x000B).
//!
//! ## Specification (HDF5 File Format Specification, Section IV.A.2.k)
//!
//! The filter pipeline message describes a sequence of filters applied to
//! chunked dataset data. Each filter performs a transformation (compression,
//! checksum, shuffle, etc.) during write and the inverse during read.
//!
//! ### Filter Pipeline Message Layout
//!
//! | Offset | Size | Field                              |
//! |--------|------|------------------------------------|
//! | 0      | 1    | Version (1 or 2)                   |
//! | 1      | 1    | Number of filters                  |
//! | 2      | 6    | Reserved (version 1 only)          |
//! | 8/2    | var  | Filter descriptions (concatenated) |
//!
//! ### Version 1 Filter Description
//!
//! | Offset | Size   | Field                                                  |
//! |--------|--------|--------------------------------------------------------|
//! | 0      | 2      | Filter identification value                            |
//! | 2      | 2      | Name length (including null terminator; 0 if unnamed)  |
//! | 4      | 2      | Flags (bit 0: optional filter)                         |
//! | 6      | 2      | Number of client data values (N)                       |
//! | 8      | var    | Name (null-terminated, padded to 8-byte boundary)      |
//! | var    | 4 × N  | Client data (little-endian u32 values)                 |
//! | var    | 0 or 4 | Padding if N is odd (version 1 only)                   |
//!
//! ### Version 2 Filter Description
//!
//! | Offset | Size  | Field                                                      |
//! |--------|-------|------------------------------------------------------------|
//! | 0      | 2     | Filter identification value                                |
//! | 2      | 2     | Name length (0 for predefined filters with ID < 256)       |
//! | 4      | 2     | Flags                                                      |
//! | 6      | 2     | Number of client data values (N)                           |
//! | 8      | var   | Name (if name length > 0; NOT padded)                      |
//! | var    | 4 × N | Client data (little-endian u32 values)                     |
//!
//! ### Standard HDF5 Filter IDs
//!
//! | ID | Name             |
//! |----|------------------|
//! | 1  | Deflate (zlib)   |
//! | 2  | Shuffle          |
//! | 3  | Fletcher32       |
//! | 4  | Szip             |
//! | 5  | Nbit             |
//! | 6  | ScaleOffset      |

/// Well-known HDF5 filter identification values.
pub mod filter_ids {
    /// Deflate (zlib) compression.
    pub const DEFLATE: u16 = 1;
    /// Byte shuffle for improved compression ratios.
    pub const SHUFFLE: u16 = 2;
    /// Fletcher32 checksum.
    pub const FLETCHER32: u16 = 3;
    /// Szip compression.
    pub const SZIP: u16 = 4;
    /// N-bit packing.
    pub const NBIT: u16 = 5;
    /// Scale-offset encoding.
    pub const SCALE_OFFSET: u16 = 6;
}

/// Version 1 filter pipeline message alignment boundary.
const V1_NAME_ALIGNMENT: usize = 8;

/// Version 1 filter description fixed header size (before name).
///
/// Layout: filter_id(2) + name_length(2) + flags(2) + num_client_data(2) = 8.
const FILTER_DESC_HEADER_SIZE: usize = 8;

/// Minimum filter pipeline message header size.
///
/// Version 1: version(1) + num_filters(1) + reserved(6) = 8.
/// Version 2: version(1) + num_filters(1) = 2.
const V1_PIPELINE_HEADER_SIZE: usize = 8;
const V2_PIPELINE_HEADER_SIZE: usize = 2;

/// Threshold below which filter IDs are predefined (unnamed in v2).
#[allow(dead_code)]
const PREDEFINED_FILTER_ID_LIMIT: u16 = 256;

#[cfg(feature = "alloc")]
mod implementation {
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
                    message: String::from(
                        "filter pipeline message too short for version and count",
                    ),
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
            let name_str = core::str::from_utf8(&name_bytes[..name_end]).map_err(|_| {
                Error::InvalidFormat {
                    message: alloc::format!(
                        "filter pipeline v1: filter {index} name is not valid UTF-8"
                    ),
                }
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
            let name_str = core::str::from_utf8(&name_bytes[..name_end]).map_err(|_| {
                Error::InvalidFormat {
                    message: alloc::format!(
                        "filter pipeline v2: filter {index} name is not valid UTF-8"
                    ),
                }
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
}

#[cfg(feature = "alloc")]
pub use implementation::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_id_constants() {
        assert_eq!(filter_ids::DEFLATE, 1);
        assert_eq!(filter_ids::SHUFFLE, 2);
        assert_eq!(filter_ids::FLETCHER32, 3);
        assert_eq!(filter_ids::SZIP, 4);
        assert_eq!(filter_ids::NBIT, 5);
        assert_eq!(filter_ids::SCALE_OFFSET, 6);
    }

    #[test]
    fn predefined_limit() {
        assert_eq!(PREDEFINED_FILTER_ID_LIMIT, 256);
    }

    #[cfg(feature = "alloc")]
    mod alloc_tests {
        use super::super::*;

        /// Construct a version 1 filter pipeline with one deflate filter
        /// (level 6) and verify all parsed fields.
        #[test]
        fn parse_v1_single_deflate() {
            // Pipeline header: version=1, num_filters=1, reserved=6 zeros.
            let mut data = alloc::vec![0u8; 0];

            // Version 1 header.
            data.push(1); // version
            data.push(1); // num_filters
            data.extend_from_slice(&[0u8; 6]); // reserved

            // Filter description: deflate, no name, 1 client data value (level=6).
            // filter_id = 1 (deflate)
            data.extend_from_slice(&1u16.to_le_bytes());
            // name_length = 0
            data.extend_from_slice(&0u16.to_le_bytes());
            // flags = 0
            data.extend_from_slice(&0u16.to_le_bytes());
            // num_client_data = 1
            data.extend_from_slice(&1u16.to_le_bytes());
            // client_data[0] = 6 (compression level)
            data.extend_from_slice(&6u32.to_le_bytes());
            // Padding: num_client_data=1 is odd, so 4 bytes padding.
            data.extend_from_slice(&[0u8; 4]);

            let pipeline = Hdf5FilterPipeline::parse(&data).unwrap();
            assert_eq!(pipeline.version, 1);
            assert_eq!(pipeline.filters.len(), 1);

            let f = &pipeline.filters[0];
            assert_eq!(f.filter_id, filter_ids::DEFLATE);
            assert!(f.name.is_none());
            assert_eq!(f.flags, 0);
            assert_eq!(f.client_data.len(), 1);
            assert_eq!(f.client_data[0], 6);
        }

        /// Construct a version 1 filter pipeline with a named user-defined
        /// filter and verify name parsing with 8-byte alignment.
        #[test]
        fn parse_v1_named_filter() {
            let mut data = alloc::vec![0u8; 0];

            // Version 1 header.
            data.push(1); // version
            data.push(1); // num_filters
            data.extend_from_slice(&[0u8; 6]); // reserved

            // Filter description: user-defined filter ID=300, name="myfilter\0"
            // (9 bytes including null, padded to 16 bytes for 8-byte alignment).
            data.extend_from_slice(&300u16.to_le_bytes()); // filter_id
            data.extend_from_slice(&9u16.to_le_bytes()); // name_length (includes null)
            data.extend_from_slice(&0u16.to_le_bytes()); // flags
            data.extend_from_slice(&2u16.to_le_bytes()); // num_client_data

            // Name: "myfilter\0" = 9 bytes, padded to 16.
            data.extend_from_slice(b"myfilter\0");
            data.extend_from_slice(&[0u8; 7]); // pad to 16 bytes

            // Client data: two u32 values.
            data.extend_from_slice(&42u32.to_le_bytes());
            data.extend_from_slice(&99u32.to_le_bytes());
            // num_client_data=2 is even, no padding.

            let pipeline = Hdf5FilterPipeline::parse(&data).unwrap();
            assert_eq!(pipeline.version, 1);
            assert_eq!(pipeline.filters.len(), 1);

            let f = &pipeline.filters[0];
            assert_eq!(f.filter_id, 300);
            assert_eq!(f.name.as_deref(), Some("myfilter"));
            assert_eq!(f.flags, 0);
            assert_eq!(f.client_data, &[42, 99]);
        }

        /// Construct a version 2 filter pipeline with shuffle + deflate
        /// (two filters) and verify parse order and fields.
        #[test]
        fn parse_v2_shuffle_and_deflate() {
            let mut data = alloc::vec![0u8; 0];

            // Version 2 header (no reserved bytes).
            data.push(2); // version
            data.push(2); // num_filters

            // Filter 0: shuffle (ID=2), no name (predefined, ID < 256),
            // no client data.
            data.extend_from_slice(&2u16.to_le_bytes()); // filter_id
            data.extend_from_slice(&0u16.to_le_bytes()); // name_length
            data.extend_from_slice(&0u16.to_le_bytes()); // flags
            data.extend_from_slice(&0u16.to_le_bytes()); // num_client_data

            // Filter 1: deflate (ID=1), no name, 1 client data (level=4).
            data.extend_from_slice(&1u16.to_le_bytes()); // filter_id
            data.extend_from_slice(&0u16.to_le_bytes()); // name_length
            data.extend_from_slice(&0u16.to_le_bytes()); // flags
            data.extend_from_slice(&1u16.to_le_bytes()); // num_client_data
            data.extend_from_slice(&4u32.to_le_bytes()); // level=4

            let pipeline = Hdf5FilterPipeline::parse(&data).unwrap();
            assert_eq!(pipeline.version, 2);
            assert_eq!(pipeline.filters.len(), 2);

            assert_eq!(pipeline.filters[0].filter_id, filter_ids::SHUFFLE);
            assert!(pipeline.filters[0].name.is_none());
            assert!(pipeline.filters[0].client_data.is_empty());

            assert_eq!(pipeline.filters[1].filter_id, filter_ids::DEFLATE);
            assert!(pipeline.filters[1].name.is_none());
            assert_eq!(pipeline.filters[1].client_data, &[4]);
        }

        /// Version 2 with a user-defined named filter (ID >= 256).
        #[test]
        fn parse_v2_named_user_filter() {
            let mut data = alloc::vec![0u8; 0];

            data.push(2); // version
            data.push(1); // num_filters

            data.extend_from_slice(&512u16.to_le_bytes()); // filter_id
            // Name: "custom\0" = 7 bytes (including null).
            data.extend_from_slice(&7u16.to_le_bytes()); // name_length
            data.extend_from_slice(&1u16.to_le_bytes()); // flags (optional)
            data.extend_from_slice(&0u16.to_le_bytes()); // num_client_data

            data.extend_from_slice(b"custom\0"); // name (no padding in v2)

            let pipeline = Hdf5FilterPipeline::parse(&data).unwrap();
            assert_eq!(pipeline.filters.len(), 1);

            let f = &pipeline.filters[0];
            assert_eq!(f.filter_id, 512);
            assert_eq!(f.name.as_deref(), Some("custom"));
            assert_eq!(f.flags, 1); // optional bit set
            assert!(f.client_data.is_empty());
        }

        /// Reject unsupported pipeline version.
        #[test]
        fn reject_unsupported_version() {
            let data = [3u8, 0]; // version=3, num_filters=0
            let err = Hdf5FilterPipeline::parse(&data).unwrap_err();
            match err {
                consus_core::Error::InvalidFormat { message } => {
                    assert!(message.contains("unsupported filter pipeline version"));
                }
                other => panic!("expected InvalidFormat, got: {other:?}"),
            }
        }

        /// Reject truncated pipeline header.
        #[test]
        fn reject_truncated_header() {
            let data = [1u8]; // only 1 byte
            let err = Hdf5FilterPipeline::parse(&data).unwrap_err();
            match err {
                consus_core::Error::InvalidFormat { .. } => {}
                other => panic!("expected InvalidFormat, got: {other:?}"),
            }
        }

        /// Parse empty filter pipeline (zero filters).
        #[test]
        fn parse_v2_empty_pipeline() {
            let data = [2u8, 0]; // version=2, num_filters=0
            let pipeline = Hdf5FilterPipeline::parse(&data).unwrap();
            assert_eq!(pipeline.version, 2);
            assert!(pipeline.filters.is_empty());
        }

        /// Parse empty version 1 pipeline.
        #[test]
        fn parse_v1_empty_pipeline() {
            let mut data = alloc::vec![0u8; 8];
            data[0] = 1; // version
            data[1] = 0; // num_filters
            // bytes 2..8 reserved

            let pipeline = Hdf5FilterPipeline::parse(&data).unwrap();
            assert_eq!(pipeline.version, 1);
            assert!(pipeline.filters.is_empty());
        }

        /// Verify fletcher32 filter (no client data, no name).
        #[test]
        fn parse_v2_fletcher32() {
            let mut data = alloc::vec![0u8; 0];
            data.push(2); // version
            data.push(1); // num_filters

            data.extend_from_slice(&3u16.to_le_bytes()); // filter_id = fletcher32
            data.extend_from_slice(&0u16.to_le_bytes()); // name_length
            data.extend_from_slice(&0u16.to_le_bytes()); // flags
            data.extend_from_slice(&0u16.to_le_bytes()); // num_client_data

            let pipeline = Hdf5FilterPipeline::parse(&data).unwrap();
            assert_eq!(pipeline.filters.len(), 1);
            assert_eq!(pipeline.filters[0].filter_id, filter_ids::FLETCHER32);
            assert!(pipeline.filters[0].client_data.is_empty());
        }
    }
}
