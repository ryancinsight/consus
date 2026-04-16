//! HDF5 data layout message parser (header message type 0x0008).
//!
//! ## Specification (HDF5 File Format Specification, Section IV.A.2.i)
//!
//! The data layout message describes how raw data for a dataset is
//! organized within the file. It specifies the storage strategy
//! (compact, contiguous, or chunked) and the associated addresses
//! and dimensions.
//!
//! ### Version 3 Layout (current standard)
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0      | 1    | Version (3) |
//! | 1      | 1    | Layout class (0=compact, 1=contiguous, 2=chunked) |
//! | 2      | var  | Class-specific data |
//!
//! #### Compact Layout (class 0)
//!
//! | Offset | Size      | Field |
//! |--------|-----------|-------|
//! | 2      | 2         | Data size (u16) |
//! | 4      | data_size | Raw data bytes |
//!
//! #### Contiguous Layout (class 1)
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 2      | S    | Data address (offset_size bytes) |
//! | 2+S    | L    | Data size (length_size bytes) |
//!
//! #### Chunked Layout (class 2, version 3)
//!
//! | Offset | Size           | Field |
//! |--------|----------------|-------|
//! | 2      | 1              | Dimensionality (rank + 1) |
//! | 3      | S              | B-tree v1 address |
//! | 3+S    | 4 × dimensionality | Chunk dimension sizes (u32 each) |
//!
//! The extra dimension in chunked layout stores the dataset element
//! size as the innermost (fastest-varying) chunk dimension.
//!
//! ### Version 4 Layout
//!
//! | Offset | Size           | Field |
//! |--------|----------------|-------|
//! | 0      | 1              | Version (4) |
//! | 1      | 1              | Layout class |
//! | 2      | var            | Class-specific data |
//!
//! #### Chunked Layout (class 2, version 4)
//!
//! | Offset | Size              | Field |
//! |--------|-------------------|-------|
//! | 2      | 1                 | Flags |
//! | 3      | 1                 | Dimensionality (rank) |
//! | 4      | 1                 | Encoded size of filtered chunk (bytes) |
//! | 5      | 4 × rank          | Chunk dimension sizes (u32 each) |
//! | var    | 1                 | Chunk indexing type |
//! | var    | var               | Index-type-specific data |
//!
//! Chunk indexing types:
//! - 1: Single Chunk
//! - 2: Implicit (no index needed, all chunks allocated)
//! - 3: Fixed Array
//! - 4: Extensible Array
//! - 5: Version 2 B-tree

#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec::Vec};

#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};

#[cfg(feature = "alloc")]
use consus_core::{Error, Result};

#[cfg(feature = "alloc")]
use crate::address::ParseContext;

#[cfg(feature = "alloc")]
use crate::dataset::StorageLayout;

/// Chunk indexing type constants for version 4 layout messages.
pub mod chunk_index_type {
    /// Single chunk — entire dataset is one chunk.
    pub const SINGLE_CHUNK: u8 = 1;
    /// Implicit — all chunks allocated sequentially, no index structure.
    pub const IMPLICIT: u8 = 2;
    /// Fixed array index.
    pub const FIXED_ARRAY: u8 = 3;
    /// Extensible array index.
    pub const EXTENSIBLE_ARRAY: u8 = 4;
    /// Version 2 B-tree index.
    pub const BTREE_V2: u8 = 5;
}

/// Parsed data layout from a data layout header message.
///
/// Consolidates all layout variants into a single struct. Fields that
/// are not applicable to the current layout class are `None`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DataLayout {
    /// Layout message version (3 or 4).
    pub version: u8,
    /// Storage class.
    pub layout: StorageLayout,
    /// Data address for contiguous storage.
    ///
    /// Set to `Some(addr)` for contiguous layout. The value
    /// `u64::MAX` (undefined address) indicates unallocated storage.
    pub data_address: Option<u64>,
    /// Data size in bytes for contiguous or compact storage.
    pub data_size: Option<u64>,
    /// Compact data bytes (stored inline in the object header).
    ///
    /// Present only for compact layout (class 0).
    pub compact_data: Option<Vec<u8>>,
    /// B-tree address for chunked storage.
    ///
    /// For version 3: address of B-tree v1.
    /// For version 4 with B-tree v2 indexing: address of B-tree v2 header.
    pub chunk_btree_address: Option<u64>,
    /// Chunk dimensions (one entry per dataset dimension).
    ///
    /// For version 3: the dimensionality field includes an extra element
    /// for the dataset element size. This field strips that extra element,
    /// returning only the spatial chunk dimensions.
    pub chunk_dims: Option<Vec<u32>>,
    /// Chunk indexing type (version 4 only).
    ///
    /// See [`chunk_index_type`] module for constants.
    pub chunk_index_type: Option<u8>,
    /// Dataset element size extracted from the v3 chunked layout.
    ///
    /// The last dimension in the v3 chunked dimensionality encodes the
    /// element size. Stored here for downstream verification.
    pub chunk_element_size: Option<u32>,
    /// Single-chunk filter info for v4 single-chunk indexing.
    pub single_chunk_filtered_size: Option<u64>,
    /// Single-chunk filter mask for v4 single-chunk indexing.
    pub single_chunk_filter_mask: Option<u32>,
    /// Index address for v4 chunked layouts (extensible array, fixed array, B-tree v2).
    pub chunk_index_address: Option<u64>,
}

#[cfg(feature = "alloc")]
impl DataLayout {
    /// Parse a data layout from raw layout header message bytes.
    ///
    /// ## Arguments
    ///
    /// - `data`: Raw bytes of the data layout message payload.
    /// - `ctx`: Parsing context (offset/length sizes from superblock).
    ///
    /// ## Errors
    ///
    /// - [`Error::InvalidFormat`] if the version or layout class is invalid.
    /// - [`Error::InvalidFormat`] if the message is truncated.
    /// - [`Error::UnsupportedFeature`] for virtual dataset layout (class 3).
    pub fn parse(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidFormat {
                message: String::from("data layout message too short"),
            });
        }

        let version = data[0];
        match version {
            3 => Self::parse_v3(data, ctx),
            4 => Self::parse_v4(data, ctx),
            _ => Err(Error::InvalidFormat {
                message: format!("unsupported data layout message version: {version}"),
            }),
        }
    }

    /// Parse version 3 layout message.
    fn parse_v3(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        let layout_class = data[1];

        match layout_class {
            // Compact
            0 => Self::parse_v3_compact(data),
            // Contiguous
            1 => Self::parse_v3_contiguous(data, ctx),
            // Chunked
            2 => Self::parse_v3_chunked(data, ctx),
            // Virtual
            3 => Err(Error::UnsupportedFeature {
                feature: String::from("virtual dataset layout"),
            }),
            _ => Err(Error::InvalidFormat {
                message: format!("unknown layout class: {layout_class}"),
            }),
        }
    }

    /// Parse version 3 compact layout.
    ///
    /// Data is stored inline in the layout message.
    fn parse_v3_compact(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(Error::InvalidFormat {
                message: String::from("compact layout message truncated at data size"),
            });
        }

        let data_size = LittleEndian::read_u16(&data[2..4]) as u64;
        let compact_data = if data_size > 0 {
            let end = 4 + data_size as usize;
            if data.len() < end {
                return Err(Error::InvalidFormat {
                    message: format!(
                        "compact layout data truncated: need {end} bytes, have {}",
                        data.len()
                    ),
                });
            }
            Some(Vec::from(&data[4..end]))
        } else {
            Some(Vec::new())
        };

        Ok(Self {
            version: 3,
            layout: StorageLayout::Compact,
            data_address: None,
            data_size: Some(data_size),
            compact_data,
            chunk_btree_address: None,
            chunk_dims: None,
            chunk_index_type: None,
            chunk_element_size: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
            chunk_index_address: None,
        })
    }

    /// Parse version 3 contiguous layout.
    fn parse_v3_contiguous(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        let s = ctx.offset_bytes();
        let l = ctx.length_bytes();
        let min_size = 2 + s + l;

        if data.len() < min_size {
            return Err(Error::InvalidFormat {
                message: format!(
                    "contiguous layout message truncated: need {min_size} bytes, have {}",
                    data.len()
                ),
            });
        }

        let data_address = ctx.read_offset(&data[2..]);
        let data_size = ctx.read_length(&data[2 + s..]);

        Ok(Self {
            version: 3,
            layout: StorageLayout::Contiguous,
            data_address: Some(data_address),
            data_size: Some(data_size),
            compact_data: None,
            chunk_btree_address: None,
            chunk_dims: None,
            chunk_index_type: None,
            chunk_element_size: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
            chunk_index_address: None,
        })
    }

    /// Parse version 3 chunked layout.
    ///
    /// The dimensionality field is `rank + 1`; the extra dimension
    /// encodes the dataset element size.
    fn parse_v3_chunked(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        if data.len() < 3 {
            return Err(Error::InvalidFormat {
                message: String::from("chunked layout message truncated at dimensionality"),
            });
        }

        let ndims = data[2] as usize;
        if ndims == 0 {
            return Err(Error::InvalidFormat {
                message: String::from("chunked layout dimensionality must be >= 1"),
            });
        }

        let s = ctx.offset_bytes();
        let btree_offset = 3;
        let dims_offset = btree_offset + s;
        let min_size = dims_offset + 4 * ndims;

        if data.len() < min_size {
            return Err(Error::InvalidFormat {
                message: format!(
                    "chunked layout message truncated: need {min_size} bytes, have {}",
                    data.len()
                ),
            });
        }

        let btree_address = ctx.read_offset(&data[btree_offset..]);

        let mut all_dims = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let off = dims_offset + i * 4;
            all_dims.push(LittleEndian::read_u32(&data[off..off + 4]));
        }

        // Last dimension is the element size, not a spatial dimension.
        let element_size = all_dims.pop();
        let chunk_dims = all_dims;

        Ok(Self {
            version: 3,
            layout: StorageLayout::Chunked,
            data_address: None,
            data_size: None,
            compact_data: None,
            chunk_btree_address: Some(btree_address),
            chunk_dims: Some(chunk_dims),
            chunk_index_type: None,
            chunk_element_size: element_size,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
            chunk_index_address: None,
        })
    }

    /// Parse version 4 layout message.
    fn parse_v4(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        let layout_class = data[1];

        match layout_class {
            0 => Self::parse_v3_compact(data),
            1 => Self::parse_v3_contiguous(data, ctx),
            2 => Self::parse_v4_chunked(data, ctx),
            3 => Err(Error::UnsupportedFeature {
                feature: String::from("virtual dataset layout"),
            }),
            _ => Err(Error::InvalidFormat {
                message: format!("unknown layout class: {layout_class}"),
            }),
        }
    }

    /// Parse version 4 chunked layout.
    ///
    /// Version 4 uses a different encoding from v3: the dimensionality
    /// does NOT include an extra element-size dimension, and the chunk
    /// indexing type is explicitly specified.
    fn parse_v4_chunked(data: &[u8], ctx: &ParseContext) -> Result<Self> {
        if data.len() < 5 {
            return Err(Error::InvalidFormat {
                message: String::from("v4 chunked layout message truncated"),
            });
        }

        let flags = data[2];
        let ndims = data[3] as usize;
        let encoded_size_width = data[4] as usize;

        let dims_offset = 5;
        let index_type_offset = dims_offset + 4 * ndims;

        if data.len() < index_type_offset + 1 {
            return Err(Error::InvalidFormat {
                message: String::from("v4 chunked layout truncated at chunk dimensions"),
            });
        }

        let mut chunk_dims = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let off = dims_offset + i * 4;
            chunk_dims.push(LittleEndian::read_u32(&data[off..off + 4]));
        }

        let indexing_type = data[index_type_offset];
        let index_data_offset = index_type_offset + 1;

        // Parse index-type-specific fields.
        let mut index_address = None;
        let mut single_filtered_size = None;
        let mut single_filter_mask = None;

        match indexing_type {
            chunk_index_type::SINGLE_CHUNK => {
                // Single chunk: optional filtered chunk size + filter mask + address
                if flags & 0x01 != 0 {
                    // Filtered single chunk
                    let needed = index_data_offset + encoded_size_width + 4;
                    if data.len() >= needed {
                        let size_bytes =
                            &data[index_data_offset..index_data_offset + encoded_size_width];
                        let mut size = 0u64;
                        for (i, &b) in size_bytes.iter().enumerate() {
                            size |= (b as u64) << (i * 8);
                        }
                        single_filtered_size = Some(size);
                        single_filter_mask = Some(LittleEndian::read_u32(
                            &data[index_data_offset + encoded_size_width..],
                        ));
                    }
                }
                // No index address for single chunk (data address is in a separate field
                // or derived from the filtered chunk info). Callers use contiguous semantics.
            }
            chunk_index_type::IMPLICIT => {
                // Implicit: no additional index data.
            }
            chunk_index_type::FIXED_ARRAY
            | chunk_index_type::EXTENSIBLE_ARRAY
            | chunk_index_type::BTREE_V2 => {
                // These indexing types store an address to their header structure.
                let s = ctx.offset_bytes();
                if data.len() >= index_data_offset + s {
                    index_address = Some(ctx.read_offset(&data[index_data_offset..]));
                }
            }
            _ => {
                return Err(Error::InvalidFormat {
                    message: format!("unknown chunk indexing type: {indexing_type}"),
                });
            }
        }

        Ok(Self {
            version: 4,
            layout: StorageLayout::Chunked,
            data_address: None,
            data_size: None,
            compact_data: None,
            chunk_btree_address: None,
            chunk_dims: Some(chunk_dims),
            chunk_index_type: Some(indexing_type),
            chunk_element_size: None,
            single_chunk_filtered_size: single_filtered_size,
            single_chunk_filter_mask: single_filter_mask,
            chunk_index_address: index_address,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    mod alloc_tests {
        use super::*;

        fn ctx8() -> ParseContext {
            ParseContext::new(8, 8)
        }

        fn ctx4() -> ParseContext {
            ParseContext::new(4, 4)
        }

        #[test]
        fn parse_v3_compact_empty() {
            // version=3, class=0, size=0
            let data = [3u8, 0, 0, 0];
            let layout = DataLayout::parse(&data, &ctx8()).unwrap();
            assert_eq!(layout.version, 3);
            assert_eq!(layout.layout, StorageLayout::Compact);
            assert_eq!(layout.data_size, Some(0));
            assert!(layout.compact_data.as_ref().unwrap().is_empty());
        }

        #[test]
        fn parse_v3_compact_with_data() {
            // version=3, class=0, size=4, data=[0xDE, 0xAD, 0xBE, 0xEF]
            let data = [3u8, 0, 4, 0, 0xDE, 0xAD, 0xBE, 0xEF];
            let layout = DataLayout::parse(&data, &ctx8()).unwrap();
            assert_eq!(layout.layout, StorageLayout::Compact);
            assert_eq!(layout.data_size, Some(4));
            assert_eq!(
                layout.compact_data.as_ref().unwrap(),
                &[0xDE, 0xAD, 0xBE, 0xEF]
            );
        }

        #[test]
        fn parse_v3_contiguous_8byte_offsets() {
            // version=3, class=1, addr=0x1000 (LE 8 bytes), size=0x2000 (LE 8 bytes)
            let mut data = vec![3u8, 1];
            // Address = 0x1000
            data.extend_from_slice(&0x1000u64.to_le_bytes());
            // Size = 0x2000
            data.extend_from_slice(&0x2000u64.to_le_bytes());

            let layout = DataLayout::parse(&data, &ctx8()).unwrap();
            assert_eq!(layout.layout, StorageLayout::Contiguous);
            assert_eq!(layout.data_address, Some(0x1000));
            assert_eq!(layout.data_size, Some(0x2000));
        }

        #[test]
        fn parse_v3_contiguous_4byte_offsets() {
            // version=3, class=1, addr=0x100 (LE 4 bytes), size=0x200 (LE 4 bytes)
            let mut data = vec![3u8, 1];
            data.extend_from_slice(&0x100u32.to_le_bytes());
            data.extend_from_slice(&0x200u32.to_le_bytes());

            let layout = DataLayout::parse(&data, &ctx4()).unwrap();
            assert_eq!(layout.layout, StorageLayout::Contiguous);
            assert_eq!(layout.data_address, Some(0x100));
            assert_eq!(layout.data_size, Some(0x200));
        }

        #[test]
        fn parse_v3_chunked_2d() {
            // version=3, class=2, ndims=3 (rank 2 + 1 element size dim)
            // btree_addr=0x3000, chunk_dims=[10, 20], element_size=4
            let mut data = vec![3u8, 2, 3]; // version, class, ndims
            data.extend_from_slice(&0x3000u64.to_le_bytes()); // B-tree address (8 bytes)
            data.extend_from_slice(&10u32.to_le_bytes()); // dim 0
            data.extend_from_slice(&20u32.to_le_bytes()); // dim 1
            data.extend_from_slice(&4u32.to_le_bytes()); // element size

            let layout = DataLayout::parse(&data, &ctx8()).unwrap();
            assert_eq!(layout.layout, StorageLayout::Chunked);
            assert_eq!(layout.chunk_btree_address, Some(0x3000));
            assert_eq!(layout.chunk_dims.as_ref().unwrap(), &[10, 20]);
            assert_eq!(layout.chunk_element_size, Some(4));
            assert!(layout.chunk_index_type.is_none()); // v3 has no explicit index type
        }

        #[test]
        fn parse_v3_chunked_1d() {
            // ndims=2 (rank 1 + element size), btree=0x500, chunk=[100], elem_size=8
            let mut data = vec![3u8, 2, 2]; // version, class, ndims=2
            data.extend_from_slice(&0x500u64.to_le_bytes());
            data.extend_from_slice(&100u32.to_le_bytes());
            data.extend_from_slice(&8u32.to_le_bytes());

            let layout = DataLayout::parse(&data, &ctx8()).unwrap();
            assert_eq!(layout.chunk_dims.as_ref().unwrap(), &[100]);
            assert_eq!(layout.chunk_element_size, Some(8));
        }

        #[test]
        fn parse_v4_chunked_btree_v2() {
            // version=4, class=2, flags=0, ndims=2, encoded_size=0
            // chunk_dims=[64, 64], indexing_type=5 (B-tree v2), address=0x4000
            let mut data = vec![4u8, 2, 0, 2, 0]; // version, class, flags, ndims, enc_size
            data.extend_from_slice(&64u32.to_le_bytes()); // dim 0
            data.extend_from_slice(&64u32.to_le_bytes()); // dim 1
            data.push(chunk_index_type::BTREE_V2); // indexing type
            data.extend_from_slice(&0x4000u64.to_le_bytes()); // index address

            let layout = DataLayout::parse(&data, &ctx8()).unwrap();
            assert_eq!(layout.version, 4);
            assert_eq!(layout.layout, StorageLayout::Chunked);
            assert_eq!(layout.chunk_dims.as_ref().unwrap(), &[64, 64]);
            assert_eq!(layout.chunk_index_type, Some(chunk_index_type::BTREE_V2));
            assert_eq!(layout.chunk_index_address, Some(0x4000));
        }

        #[test]
        fn parse_v4_chunked_implicit() {
            // version=4, class=2, flags=0, ndims=1, encoded_size=0
            // chunk_dims=[1024], indexing_type=2 (Implicit)
            let mut data = vec![4u8, 2, 0, 1, 0];
            data.extend_from_slice(&1024u32.to_le_bytes());
            data.push(chunk_index_type::IMPLICIT);

            let layout = DataLayout::parse(&data, &ctx8()).unwrap();
            assert_eq!(layout.chunk_index_type, Some(chunk_index_type::IMPLICIT));
            assert!(layout.chunk_index_address.is_none());
        }

        #[test]
        fn parse_v4_chunked_single_chunk_filtered() {
            // version=4, class=2, flags=0x01 (filtered), ndims=2, encoded_size=4
            // chunk_dims=[100, 100], indexing_type=1 (Single Chunk)
            // filtered_size=512 (4 bytes LE), filter_mask=0
            let mut data = vec![4u8, 2, 0x01, 2, 4]; // flags bit 0 = filtered
            data.extend_from_slice(&100u32.to_le_bytes());
            data.extend_from_slice(&100u32.to_le_bytes());
            data.push(chunk_index_type::SINGLE_CHUNK);
            data.extend_from_slice(&512u32.to_le_bytes()); // filtered size (4 bytes)
            data.extend_from_slice(&0u32.to_le_bytes()); // filter mask

            let layout = DataLayout::parse(&data, &ctx8()).unwrap();
            assert_eq!(
                layout.chunk_index_type,
                Some(chunk_index_type::SINGLE_CHUNK)
            );
            assert_eq!(layout.single_chunk_filtered_size, Some(512));
            assert_eq!(layout.single_chunk_filter_mask, Some(0));
        }

        #[test]
        fn reject_virtual_layout() {
            let data = [3u8, 3]; // version=3, class=3 (virtual)
            let err = DataLayout::parse(&data, &ctx8()).unwrap_err();
            match err {
                Error::UnsupportedFeature { .. } => {}
                other => panic!("expected UnsupportedFeature, got: {other:?}"),
            }
        }

        #[test]
        fn reject_unknown_version() {
            let data = [5u8, 0]; // version=5
            let err = DataLayout::parse(&data, &ctx8()).unwrap_err();
            match err {
                Error::InvalidFormat { .. } => {}
                other => panic!("expected InvalidFormat, got: {other:?}"),
            }
        }

        #[test]
        fn reject_truncated_message() {
            let data = [3u8]; // too short
            let err = DataLayout::parse(&data, &ctx8()).unwrap_err();
            match err {
                Error::InvalidFormat { .. } => {}
                other => panic!("expected InvalidFormat, got: {other:?}"),
            }
        }

        #[test]
        fn reject_truncated_contiguous() {
            // version=3, class=1, but no address/size data
            let data = [3u8, 1];
            let err = DataLayout::parse(&data, &ctx8()).unwrap_err();
            match err {
                Error::InvalidFormat { .. } => {}
                other => panic!("expected InvalidFormat, got: {other:?}"),
            }
        }

        #[test]
        fn reject_zero_dimensionality_chunked() {
            // version=3, class=2, ndims=0
            let data = [3u8, 2, 0];
            let err = DataLayout::parse(&data, &ctx8()).unwrap_err();
            match err {
                Error::InvalidFormat { .. } => {}
                other => panic!("expected InvalidFormat, got: {other:?}"),
            }
        }
    }
}
