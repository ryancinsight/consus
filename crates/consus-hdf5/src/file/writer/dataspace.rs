//! HDF5 v2 dataspace message encoding from `Shape`.

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};
#[cfg(feature = "alloc")]
use byteorder::{ByteOrder, LittleEndian};
#[cfg(feature = "alloc")]
use consus_core::{Extent, Result, Shape};

// ---------------------------------------------------------------------------
// Dataspace encoding
// ---------------------------------------------------------------------------

/// Encode a `Shape` into an HDF5 version 2 dataspace message.
///
/// ## Layout (version 2)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 1 | Version (2) |
/// | 1 | 1 | Dimensionality (rank) |
/// | 2 | 1 | Flags (bit 0: max dims present) |
/// | 3 | 1 | Type (0=scalar, 1=simple, 2=null) |
/// | 4 | 8×rank | Current dimension sizes |
/// | var | 8×rank | Maximum dimension sizes (if flags bit 0) |
#[cfg(feature = "alloc")]
pub fn encode_dataspace(shape: &Shape) -> Result<Vec<u8>> {
    let rank = shape.rank();
    let has_unlimited = shape.has_unlimited();
    let flags: u8 = if has_unlimited { 0x01 } else { 0x00 };
    let ds_type: u8 = if rank == 0 { 0 } else { 1 }; // scalar vs simple

    let max_dims_bytes = if has_unlimited { 8 * rank } else { 0 };
    let total = 4 + 8 * rank + max_dims_bytes;
    let mut buf = vec![0u8; total];

    buf[0] = 2; // version
    buf[1] = rank as u8;
    buf[2] = flags;
    buf[3] = ds_type;

    let mut pos = 4;
    for ext in shape.extents() {
        LittleEndian::write_u64(&mut buf[pos..], ext.current_size() as u64);
        pos += 8;
    }

    if has_unlimited {
        for ext in shape.extents() {
            let max = match ext {
                Extent::Fixed(n) => *n as u64,
                Extent::Unlimited { .. } => u64::MAX,
            };
            LittleEndian::write_u64(&mut buf[pos..], max);
            pos += 8;
        }
    }

    Ok(buf)
}
