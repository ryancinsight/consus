//! HDF5 dataspace encoding and decoding.
//!
//! ## Specification
//!
//! A dataspace describes the shape of a dataset or attribute.
//! It is encoded as a header message (type 0x0001) with version,
//! rank, flags, and dimension sizes.
//!
//! ### Layout
//!
//! | Field | Size | Description |
//! |-------|------|-------------|
//! | Version | 1 | 1 or 2 |
//! | Dimensionality (rank) | 1 | 0-32 |
//! | Flags | 1 | bit 0: max dims present, bit 1: permutation present |
//! | (v1 only) Reserved | 5 | |
//! | Dimension sizes | 8 x rank | Current dimension sizes |
//! | Max dimension sizes | 8 x rank | (if flag bit 0 set; 0xFFFF...=unlimited) |

use consus_core::{Extent, Shape};

/// Undefined dimension size sentinel (unlimited dimension).
pub const UNLIMITED_DIM: u64 = u64::MAX;

/// Parse a dataspace from raw message bytes.
///
/// Returns the current shape with extent information (fixed vs. unlimited).
#[cfg(feature = "alloc")]
pub fn parse_dataspace(data: &[u8], _offset_size: u8) -> consus_core::Result<Shape> {
    use alloc::vec::Vec;
    use byteorder::{ByteOrder, LittleEndian};

    if data.len() < 4 {
        return Err(consus_core::Error::InvalidFormat {
            message: alloc::string::String::from("dataspace message too short"),
        });
    }

    let version = data[0];
    let rank = data[1] as usize;
    let flags = data[2];

    let has_max_dims = (flags & 0x01) != 0;

    // Determine offset of dimension sizes
    let dims_offset = match version {
        1 => 8, // 1 (version) + 1 (rank) + 1 (flags) + 5 (reserved)
        2 => 4, // 1 (version) + 1 (rank) + 1 (flags) + 1 (type)
        _ => {
            return Err(consus_core::Error::InvalidFormat {
                message: alloc::format!("unsupported dataspace version: {version}"),
            });
        }
    };

    let mut current_dims = Vec::with_capacity(rank);
    for i in 0..rank {
        let off = dims_offset + i * 8;
        if off + 8 > data.len() {
            return Err(consus_core::Error::InvalidFormat {
                message: alloc::string::String::from("dataspace truncated"),
            });
        }
        current_dims.push(LittleEndian::read_u64(&data[off..off + 8]));
    }

    let max_dims_offset = dims_offset + rank * 8;
    let mut extents = Vec::with_capacity(rank);

    for (i, &cur) in current_dims.iter().enumerate() {
        let extent = if has_max_dims {
            let off = max_dims_offset + i * 8;
            if off + 8 <= data.len() {
                let max = LittleEndian::read_u64(&data[off..off + 8]);
                if max == UNLIMITED_DIM {
                    Extent::Unlimited {
                        current: cur as usize,
                    }
                } else {
                    Extent::Fixed(cur as usize)
                }
            } else {
                Extent::Fixed(cur as usize)
            }
        } else {
            Extent::Fixed(cur as usize)
        };
        extents.push(extent);
    }

    Ok(Shape::new(&extents))
}
