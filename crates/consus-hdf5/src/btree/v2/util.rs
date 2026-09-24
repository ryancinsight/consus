//! Variable-width record-count and unsigned-integer decoding helpers.

use super::BTreeV2Header;

/// Compute the byte width of the num_records field in internal node
/// child pointer entries.
///
/// The width is `ceil(log2(max_leaf_records + 1)) / 8`, clamped to
/// at least 1 byte.
///
/// For most practical HDF5 files (node_size ≤ 64 KiB, record_size ≥ 1),
/// this is 1 or 2 bytes.
pub(crate) fn compute_num_records_width(header: &BTreeV2Header) -> usize {
    // Maximum records in a leaf node:
    // node_size - signature(4) - version(1) - type(1) - checksum(4) = usable
    // max_leaf_records = usable / record_size
    let usable = (header.node_size as usize).saturating_sub(10);
    let max_leaf_records = if header.record_size > 0 {
        usable / header.record_size as usize
    } else {
        0
    };

    // Width in bytes = ceil(ceil(log2(max_leaf_records + 1)) / 8)
    if max_leaf_records == 0 {
        return 1;
    }
    let bits_needed = u32::BITS - (max_leaf_records as u32).leading_zeros();
    let bytes_needed = (bits_needed as usize).div_ceil(8);
    bytes_needed.max(1)
}

/// Read a variable-width unsigned integer (1–8 bytes, little-endian).
pub(crate) fn read_variable_width_uint(buf: &[u8], width: usize) -> u64 {
    let mut value = 0u64;
    for i in 0..width.min(8).min(buf.len()) {
        value |= (buf[i] as u64) << (i * 8);
    }
    value
}
