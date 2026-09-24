//! Data layout message encoders: contiguous, compact, chunked v3, and the v4 chunked indices.

#[cfg(feature = "alloc")]
use super::{DatasetCreationProps, dataset_filter_ids, write_offset};
#[cfg(feature = "alloc")]
use crate::address::ParseContext;
#[cfg(feature = "alloc")]
use crate::property_list::DatasetLayout;
#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};
#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};
#[cfg(feature = "alloc")]
use consus_core::{Error, Result};

// ---------------------------------------------------------------------------
// Layout encoding
// ---------------------------------------------------------------------------

/// Encode a data layout message (version 3).
///
/// ## Contiguous (class 1)
///
/// | 0 | 1 | Version (3) |
/// | 1 | 1 | Class (1) |
/// | 2 | S | Data address |
/// | 2+S | L | Data size |
///
/// ## Compact (class 0)
///
/// | 0 | 1 | Version (3) |
/// | 1 | 1 | Class (0) |
/// | 2 | 2 | Data size |
/// | 4 | N | Compact data |
///
/// ## Chunked (class 2)
///
/// | 0 | 1 | Version (3) |
/// | 1 | 1 | Class (2) |
/// | 2 | 1 | Dimensionality (rank + 1) |
/// | 3 | S | B-tree v1 address |
/// | 3+S | 4×(rank+1) | Chunk dimension sizes |
#[cfg(feature = "alloc")]
pub fn encode_layout(
    data_address: u64,
    props: &DatasetCreationProps,
    ctx: &ParseContext,
) -> Result<Vec<u8>> {
    encode_layout_with_chunk_index(data_address, props, ctx, None, None)
}

#[cfg(feature = "alloc")]
pub fn encode_layout_with_chunk_index(
    data_address: u64,
    props: &DatasetCreationProps,
    ctx: &ParseContext,
    chunk_index_address: Option<u64>,
    chunk_element_size: Option<u32>,
) -> Result<Vec<u8>> {
    let s = ctx.offset_bytes();

    match props.layout {
        DatasetLayout::Contiguous => {
            let l = ctx.length_bytes();
            let total = 2 + s + l;
            let mut buf = vec![0u8; total];
            buf[0] = 3; // version 3
            buf[1] = 1; // contiguous
            write_offset(&mut buf[2..], s, data_address);
            // Data size = 0 placeholder (computed from shape × element_size at write time)
            Ok(buf)
        }
        DatasetLayout::Compact => {
            let mut buf = vec![0u8; 4];
            buf[0] = 3; // version 3
            buf[1] = 0; // compact
            LittleEndian::write_u16(&mut buf[2..4], 0); // data size placeholder
            Ok(buf)
        }
        DatasetLayout::Chunked => {
            let chunk_dims = props
                .chunk_dims
                .as_ref()
                .ok_or_else(|| Error::InvalidFormat {
                    message: String::from(
                        "chunked layout requires chunk_dims in DatasetCreationProps",
                    ),
                })?;
            // V4 layout (FARRAY for unfiltered, BT2 for filtered)
            if props.layout_version == Some(4) {
                let index_address = chunk_index_address.ok_or_else(|| Error::InvalidFormat {
                    message: String::from(
                        "v4 chunked layout requires a materialized chunk index address",
                    ),
                })?;
                let element_size = chunk_element_size.ok_or_else(|| Error::InvalidFormat {
                    message: String::from("v4 chunked layout requires a resolved element size"),
                })?;
                let has_filters = !dataset_filter_ids(props).is_empty();
                return encode_layout_v4_chunked(
                    chunk_dims,
                    element_size,
                    index_address,
                    has_filters,
                    ctx,
                );
            }
            let element_size = chunk_element_size.ok_or_else(|| Error::InvalidFormat {
                message: String::from(
                    "chunked layout requires a resolved element size in the terminal dimension",
                ),
            })?;
            let btree_address = chunk_index_address.ok_or_else(|| Error::InvalidFormat {
                message: String::from("chunked layout requires a materialized chunk index address"),
            })?;
            // Dimensionality = rank + 1 (extra element for type size)
            let ndims = chunk_dims.len() + 1;
            let total = 3 + s + 4 * ndims;
            let mut buf = vec![0u8; total];
            buf[0] = 3; // version 3
            buf[1] = 2; // chunked
            buf[2] = ndims as u8; // dimensionality
            write_offset(&mut buf[3..], s, btree_address);
            let mut pos = 3 + s;
            for &d in chunk_dims {
                LittleEndian::write_u32(&mut buf[pos..], d as u32);
                pos += 4;
            }
            LittleEndian::write_u32(&mut buf[pos..], element_size);
            Ok(buf)
        }
        DatasetLayout::Virtual => {
            // Emit a minimal version 3 class 3 (virtual) layout message.
            // The HDF5 read path surfaces this as StorageLayout::Virtual.
            Ok(vec![3u8, 3u8])
        }
    }
}

/// Encode a v4 chunked layout message with B-tree v2 index reference.
///
/// ## V4 Layout Message Format (HDF5 spec §IV.A.2.l)
///
/// | Field | Size | Value |
/// |-------|------|-------|
/// | Version | 1 | 4 |
/// | Layout class | 1 | 2 (chunked) |
/// | Flags | 1 | 0 |
/// | Dimensionality | 1 | rank + 1 (extra element-size dimension) |
/// | Encoded dim size | 1 | min bytes to hold max dim value (1–4) |
/// | Chunk dimensions | (rank+1) × enc_size | spatial dims ++ element_size |
/// | Chunk index type | 1 | 4 (B-tree v2, H5D_CHUNK_IDX_BT2) |
/// | Index address | offset_size | B-tree v2 header address |
///
/// Total size = 6 + (rank+1) * enc_size + offset_size
#[cfg(feature = "alloc")]
fn encode_layout_v4_chunked(
    chunk_dims: &[usize],
    element_size: u32,
    index_address: u64,
    has_filters: bool,
    ctx: &ParseContext,
) -> Result<Vec<u8>> {
    let s = ctx.offset_bytes();
    // ndims = rank + 1: spatial dimensions plus the trailing element-size dimension.
    let ndims = chunk_dims.len() + 1;
    // Encoded dim size: minimum bytes to represent the largest dimension value.
    let max_val = chunk_dims
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(element_size as usize);
    let enc_size: usize = if max_val < 256 {
        1
    } else if max_val < 65_536 {
        2
    } else if max_val < 16_777_216 {
        3
    } else {
        4
    };
    // For FARRAY (no filters): 5 header + ndims*enc + idx_type(1) + param(1) + addr
    // For BT2 (filtered): 5 header + ndims*enc + idx_type(1) + addr
    let total = if has_filters {
        6 + ndims * enc_size + s
    } else {
        7 + ndims * enc_size + s
    };
    let mut buf = vec![0u8; total];

    buf[0] = 4; // version 4
    buf[1] = 2; // layout class = chunked
    buf[2] = 0; // flags = 0
    buf[3] = ndims as u8;
    buf[4] = enc_size as u8;

    let mut pos = 5;
    // Spatial chunk dimensions.
    for &d in chunk_dims {
        let bytes = (d as u64).to_le_bytes();
        buf[pos..pos + enc_size].copy_from_slice(&bytes[..enc_size]);
        pos += enc_size;
    }
    // Trailing dimension = element size in bytes.
    let bytes = (element_size as u64).to_le_bytes();
    buf[pos..pos + enc_size].copy_from_slice(&bytes[..enc_size]);
    pos += enc_size;

    if has_filters {
        // Index type 5 (H5D_CHUNK_IDX_BT2): B-tree v2, used for filtered datasets.
        buf[pos] = 5; // chunk index type = BT2
        pos += 1;
        write_offset(&mut buf[pos..], s, index_address);
    } else {
        // Index type 3 (H5D_CHUNK_IDX_FARRAY): Fixed Array chunk index, used by
        // libhdf5 for fixed-dimension datasets without filters.  The layout stores
        // one parameter byte (H5D_FARRAY_MAX_NELMTS_BITS = 10, hard-coded by
        // libhdf5) followed by the address of the Fixed Array header (FAHD).
        buf[pos] = 3; // chunk index type = FARRAY
        pos += 1;
        buf[pos] = 10; // H5D_FARRAY_MAX_NELMTS_BITS — hardcoded FA creation param
        pos += 1;
        write_offset(&mut buf[pos..], s, index_address); // FAHD address
    }

    Ok(buf)
}
