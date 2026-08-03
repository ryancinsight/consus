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
#[cfg(feature = "alloc")]
mod implementation;

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
