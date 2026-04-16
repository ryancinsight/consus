//! Chunk-level I/O for HDF5 chunked datasets.
//!
//! Coordinates B-tree lookup, decompression, and data extraction
//! for individual chunks in chunked datasets.
//!
//! ## Specification
//!
//! Chunked datasets store data in fixed-size blocks indexed by a B-tree.
//! Each chunk may be independently compressed via a filter pipeline.
//! The on-disk size of a chunk may differ from its uncompressed size
//! when filters are applied.
//!
//! ### Filter Mask
//!
//! Each chunk carries a 32-bit filter mask. Bit `i` set means filter `i`
//! in the pipeline was **not** applied to that chunk. This allows partial
//! filter application (e.g., when a filter fails on a particular chunk).
//!
//! ### Decompression Pipeline
//!
//! Filters are applied in forward order during writes and reversed during
//! reads. The pipeline is:
//!
//! ```text
//! Write: data → filter[0] → filter[1] → ... → filter[N-1] → disk
//! Read:  disk → filter[N-1]⁻¹ → ... → filter[1]⁻¹ → filter[0]⁻¹ → data
//! ```
//!
//! ### Standard Filter IDs
//!
//! | ID | Name | Action |
//! |----|------|--------|
//! | 1 | Deflate | zlib compression |
//! | 2 | Shuffle | Byte transposition by element size |
//! | 3 | Fletcher32 | Append 4-byte checksum |
//! | 4 | Szip | Entropy coding |
//! | 5 | Nbit | Bit packing |
//! | 6 | ScaleOffset | Integer/float scaling |

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

#[cfg(feature = "alloc")]
use consus_compression::Checksum;

#[cfg(feature = "alloc")]
use consus_core::{Error, Result};

/// Address and size of a single chunk in the file.
///
/// Represents the on-disk location of chunk data, which may be
/// compressed. The `filter_mask` indicates which pipeline filters
/// were applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkLocation {
    /// File offset of the chunk data.
    pub address: u64,
    /// Size of the chunk data on disk in bytes.
    ///
    /// When filters are applied this may be smaller (compression)
    /// or larger (checksum appended) than the uncompressed size.
    pub size: u64,
    /// Filter mask: bit `i` set means filter `i` was **not** applied.
    ///
    /// A mask of 0 means all filters in the pipeline were applied.
    pub filter_mask: u32,
}

/// Read raw chunk data from the file, applying the decompression pipeline.
///
/// # Arguments
///
/// - `source`: positioned I/O source.
/// - `location`: on-disk chunk address, size, and filter mask.
/// - `uncompressed_size`: expected byte count after full decompression.
/// - `filter_ids`: ordered filter IDs from the dataset's filter pipeline
///   message. Index 0 is the first filter applied during writes.
/// - `registry`: compression codec registry for decompression lookups.
///
/// # Algorithm
///
/// 1. Read `location.size` bytes from `location.address`.
/// 2. If `filter_ids` is empty, return the raw bytes.
/// 3. Apply filters in **reverse** order (last applied during write is
///    first reversed during read). For each filter:
///    - If the corresponding bit in `filter_mask` is set, skip it.
///    - Filter ID 2 (shuffle): requires element size context; passed
///      through in this implementation (caller must handle externally
///      or provide a `ShuffleFilter` in the registry).
///    - Filter ID 3 (Fletcher32): strip the trailing 4-byte checksum.
///    - All other IDs: look up the codec by `FilterId` and decompress.
///
/// # Errors
///
/// - I/O errors from `source.read_at`.
/// - `Error::CompressionError` if decompression fails.
/// - `Error::InvalidFormat` if the codec is not found in the registry.
#[cfg(feature = "alloc")]
pub fn read_chunk_raw<R: consus_io::ReadAt>(
    source: &R,
    location: &ChunkLocation,
    uncompressed_size: usize,
    filter_ids: &[u16],
    registry: &dyn consus_compression::CompressionRegistry,
    fill_value: Option<&[u8]>,
) -> Result<Vec<u8>> {
    // Uninitialized chunk: address is undefined (chunk not yet written).
    // Return a buffer filled with the fill value pattern, or zeros if no fill value.
    if location.address == crate::constants::UNDEFINED_ADDRESS {
        let mut buf = vec![0u8; uncompressed_size];
        if let Some(fv) = fill_value {
            if !fv.is_empty() {
                // Tile the fill value pattern across the buffer.
                // Invariant: element_size divides uncompressed_size.
                for chunk in buf.chunks_mut(fv.len()) {
                    let copy_len = chunk.len().min(fv.len());
                    chunk[..copy_len].copy_from_slice(&fv[..copy_len]);
                }
            }
        }
        return Ok(buf);
    }

    // 1. Read raw on-disk data.
    let disk_size = location.size as usize;
    let mut compressed = vec![0u8; disk_size];
    source.read_at(location.address, &mut compressed)?;

    // 2. No filters → return raw bytes.
    if filter_ids.is_empty() {
        return Ok(compressed);
    }

    // 3. Apply filters in reverse order.
    let mut data = compressed;
    let n_filters = filter_ids.len();

    for reverse_idx in 0..n_filters {
        let pipeline_idx = n_filters - 1 - reverse_idx;
        let filter_id = filter_ids[pipeline_idx];

        // Check filter mask: bit set means this filter was NOT applied.
        if (location.filter_mask >> pipeline_idx) & 1 != 0 {
            continue;
        }

        data = apply_reverse_filter(filter_id, data, uncompressed_size, registry)?;
    }

    Ok(data)
}

/// Write chunk data to the file, applying the compression pipeline.
///
/// # Arguments
///
/// - `sink`: positioned I/O sink.
/// - `offset`: file offset at which to write the chunk.
/// - `data`: uncompressed chunk data.
/// - `filter_ids`: ordered filter IDs from the dataset's filter pipeline.
/// - `element_size`: size of one data element in bytes (needed for shuffle).
/// - `registry`: compression codec registry.
///
/// # Returns
///
/// A [`ChunkLocation`] describing where and how the chunk was written.
/// If a non-optional filter fails, its bit is set in the `filter_mask`
/// and the data is written without that filter.
///
/// # Errors
///
/// - I/O errors from `sink.write_at`.
/// - `Error::CompressionError` if a mandatory compression step fails
///   and cannot be skipped.
#[cfg(feature = "alloc")]
pub fn write_chunk_raw<W: consus_io::WriteAt>(
    sink: &mut W,
    offset: u64,
    data: &[u8],
    filter_ids: &[u16],
    element_size: usize,
    registry: &dyn consus_compression::CompressionRegistry,
) -> Result<ChunkLocation> {
    if filter_ids.is_empty() {
        sink.write_at(offset, data)?;
        return Ok(ChunkLocation {
            address: offset,
            size: data.len() as u64,
            filter_mask: 0,
        });
    }

    let mut processed = data.to_vec();
    let mut filter_mask = 0u32;

    // Apply filters in forward order.
    for (i, &filter_id) in filter_ids.iter().enumerate() {
        match apply_forward_filter(filter_id, &processed, element_size, registry) {
            Ok(output) => processed = output,
            Err(_) => {
                // Mark this filter as not applied.
                filter_mask |= 1 << i;
            }
        }
    }

    sink.write_at(offset, &processed)?;
    Ok(ChunkLocation {
        address: offset,
        size: processed.len() as u64,
        filter_mask,
    })
}

/// Apply a single filter in the reverse (read/decompress) direction.
///
/// # Standard Filters
///
/// | ID | Reverse Action |
/// |----|----------------|
/// | 1 | Deflate decompression via codec registry |
/// | 2 | Shuffle reverse (identity pass-through here) |
/// | 3 | Fletcher32 checksum strip (remove trailing 4 bytes) |
/// | 4 | Szip decompression via codec registry |
/// | ≥5 | Generic codec registry lookup |
#[cfg(feature = "alloc")]
fn apply_reverse_filter(
    filter_id: u16,
    data: Vec<u8>,
    uncompressed_size: usize,
    registry: &dyn consus_compression::CompressionRegistry,
) -> Result<Vec<u8>> {
    match filter_id {
        // Shuffle (ID 2): byte transposition.
        // The caller must handle shuffle externally with knowledge of element size.
        // Here we pass the data through unchanged; the higher-level reader must
        // apply ShuffleFilter from consus_compression if needed.
        2 => Ok(data),

        // Fletcher32 (ID 3): strip trailing 4-byte checksum.
        3 => {
            if data.len() < 4 {
                return Err(Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: String::from(
                        "Fletcher32 filter: chunk data too short to contain checksum",
                    ),
                });
            }
            let payload_len = data.len() - 4;
            // Optionally verify the checksum.
            let stored = u32::from_le_bytes([
                data[payload_len],
                data[payload_len + 1],
                data[payload_len + 2],
                data[payload_len + 3],
            ]);
            let computed = consus_compression::Fletcher32::compute(&data[..payload_len]);
            if stored != computed {
                return Err(Error::Corrupted {
                    #[cfg(feature = "alloc")]
                    message: alloc::format!(
                        "Fletcher32 checksum mismatch: stored 0x{:08x}, computed 0x{:08x}",
                        stored,
                        computed,
                    ),
                });
            }
            Ok(data[..payload_len].to_vec())
        }

        // All other filters: use the codec registry.
        _ => {
            let codec_id = consus_compression::CodecId::FilterId(filter_id);
            let codec = registry.get(&codec_id)?;
            codec.decompress(&data, uncompressed_size)
        }
    }
}

/// Apply a single filter in the forward (write/compress) direction.
///
/// # Standard Filters
///
/// | ID | Forward Action |
/// |----|----------------|
/// | 1 | Deflate compression via codec registry |
/// | 2 | Shuffle forward (identity pass-through here) |
/// | 3 | Fletcher32 checksum append (4 bytes) |
/// | 4 | Szip compression via codec registry |
/// | ≥5 | Generic codec registry lookup |
#[cfg(feature = "alloc")]
fn apply_forward_filter(
    filter_id: u16,
    data: &[u8],
    _element_size: usize,
    registry: &dyn consus_compression::CompressionRegistry,
) -> Result<Vec<u8>> {
    match filter_id {
        // Shuffle (ID 2): identity pass-through (see apply_reverse_filter docs).
        2 => Ok(data.to_vec()),

        // Fletcher32 (ID 3): append 4-byte checksum.
        3 => {
            use consus_compression::Checksum;
            let checksum = consus_compression::Fletcher32::compute(data);
            let mut output = Vec::with_capacity(data.len() + 4);
            output.extend_from_slice(data);
            output.extend_from_slice(&checksum.to_le_bytes());
            Ok(output)
        }

        // All other filters: use the codec registry.
        _ => {
            let codec_id = consus_compression::CodecId::FilterId(filter_id);
            let codec = registry.get(&codec_id)?;
            let level = consus_compression::CompressionLevel::default();
            codec.compress(data, level)
        }
    }
}

/// Compute the uncompressed size of a chunk in bytes.
///
/// # Formula
///
/// `uncompressed_chunk_bytes = element_size × ∏ chunk_dims[i]`
///
/// # Arguments
///
/// - `chunk_dims`: the chunk shape dimensions.
/// - `element_size`: the size of one dataset element in bytes.
///
/// # Returns
///
/// The total byte count for one full (non-edge) chunk.
pub fn chunk_uncompressed_size(chunk_dims: &[usize], element_size: usize) -> usize {
    chunk_dims.iter().product::<usize>() * element_size
}

/// Compute the actual element count for an edge chunk that may be
/// smaller than the full chunk shape along one or more dimensions.
///
/// # Arguments
///
/// - `chunk_coord`: N-dimensional chunk coordinate in the chunk grid.
/// - `chunk_dims`: the chunk shape dimensions.
/// - `dataset_dims`: the current dataset shape dimensions.
///
/// # Returns
///
/// Per-dimension element counts for this specific chunk, clamped to
/// the dataset boundary.
///
/// # Invariant
///
/// For all `d`: `result[d] = min(chunk_dims[d], dataset_dims[d] - chunk_coord[d] * chunk_dims[d])`
#[cfg(feature = "alloc")]
pub fn edge_chunk_dims(
    chunk_coord: &[usize],
    chunk_dims: &[usize],
    dataset_dims: &[usize],
) -> Vec<usize> {
    let rank = chunk_dims.len();
    let mut dims = Vec::with_capacity(rank);
    for d in 0..rank {
        let chunk_start = chunk_coord[d] * chunk_dims[d];
        let remaining = dataset_dims[d].saturating_sub(chunk_start);
        dims.push(remaining.min(chunk_dims[d]));
    }
    dims
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_uncompressed_size_basic() {
        // 10×20 chunk of 4-byte elements → 800 bytes.
        assert_eq!(chunk_uncompressed_size(&[10, 20], 4), 800);
    }

    #[test]
    fn chunk_uncompressed_size_scalar() {
        // Scalar (0-D) chunk: product of empty slice = 1.
        assert_eq!(chunk_uncompressed_size(&[], 8), 8);
    }

    #[test]
    fn chunk_uncompressed_size_1d() {
        assert_eq!(chunk_uncompressed_size(&[256], 2), 512);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn edge_chunk_dims_interior() {
        // Interior chunk: fully within dataset bounds.
        let coord = &[1, 1];
        let chunk = &[10, 10];
        let dataset = &[100, 100];
        let result = edge_chunk_dims(coord, chunk, dataset);
        assert_eq!(result, vec![10, 10]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn edge_chunk_dims_boundary() {
        // Edge chunk: dataset dimension 25 with chunk size 10.
        // Chunk at coord 2: starts at 20, remaining = 5.
        let coord = &[2, 0];
        let chunk = &[10, 10];
        let dataset = &[25, 10];
        let result = edge_chunk_dims(coord, chunk, dataset);
        assert_eq!(result, vec![5, 10]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn edge_chunk_dims_past_boundary() {
        // Chunk coordinate past the dataset: remaining = 0.
        let coord = &[10];
        let chunk = &[10];
        let dataset = &[95];
        let result = edge_chunk_dims(coord, chunk, dataset);
        assert_eq!(result, vec![0]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn edge_chunk_dims_exact_fit() {
        // Dataset is an exact multiple of chunk size.
        let coord = &[2];
        let chunk = &[10];
        let dataset = &[30];
        let result = edge_chunk_dims(coord, chunk, dataset);
        assert_eq!(result, vec![10]);
    }

    #[test]
    fn chunk_location_debug() {
        let loc = ChunkLocation {
            address: 4096,
            size: 512,
            filter_mask: 0,
        };
        let s = alloc::format!("{:?}", loc);
        assert!(s.contains("4096"));
        assert!(s.contains("512"));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn fletcher32_reverse_strips_checksum() {
        use consus_compression::Checksum;
        let payload = b"hello world";
        let checksum = consus_compression::Fletcher32::compute(payload);
        let mut data = Vec::from(&payload[..]);
        data.extend_from_slice(&checksum.to_le_bytes());

        let registry = consus_compression::DefaultCodecRegistry::new();
        let result = apply_reverse_filter(3, data, payload.len(), &registry).unwrap();
        assert_eq!(result, payload);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn fletcher32_reverse_rejects_bad_checksum() {
        let mut data = Vec::from(b"hello world" as &[u8]);
        // Append garbage checksum.
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);

        let registry = consus_compression::DefaultCodecRegistry::new();
        let err = apply_reverse_filter(3, data, 11, &registry).unwrap_err();
        match err {
            Error::Corrupted { .. } => {}
            other => panic!("expected Corrupted, got: {other:?}"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn fletcher32_forward_appends_checksum() {
        let payload = b"test data";
        let registry = consus_compression::DefaultCodecRegistry::new();
        let result = apply_forward_filter(3, payload, 1, &registry).unwrap();
        assert_eq!(result.len(), payload.len() + 4);
        // Verify round-trip.
        let round_trip = apply_reverse_filter(3, result, payload.len(), &registry).unwrap();
        assert_eq!(round_trip, payload);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn read_chunk_no_filters() {
        use consus_io::MemCursor;
        let data = b"uncompressed chunk data";
        let cursor = MemCursor::from_bytes(data.to_vec());
        let loc = ChunkLocation {
            address: 0,
            size: data.len() as u64,
            filter_mask: 0,
        };
        let registry = consus_compression::DefaultCodecRegistry::new();
        let result = read_chunk_raw(&cursor, &loc, data.len(), &[], &registry, None).unwrap();
        assert_eq!(result, data);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn write_chunk_no_filters() {
        use consus_io::MemCursor;
        let data = b"chunk payload";
        // Pre-allocate space.
        let mut cursor = MemCursor::from_bytes(vec![0u8; 256]);
        let registry = consus_compression::DefaultCodecRegistry::new();
        let loc = write_chunk_raw(&mut cursor, 0, data, &[], 1, &registry).unwrap();
        assert_eq!(loc.address, 0);
        assert_eq!(loc.size, data.len() as u64);
        assert_eq!(loc.filter_mask, 0);
        assert_eq!(&cursor.as_bytes()[..data.len()], data);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn write_read_roundtrip_with_fletcher32() {
        use consus_io::MemCursor;
        let data = b"round trip test data for checksumming";
        let mut cursor = MemCursor::from_bytes(vec![0u8; 256]);
        let registry = consus_compression::DefaultCodecRegistry::new();

        let loc = write_chunk_raw(&mut cursor, 0, data, &[3], 1, &registry).unwrap();
        assert_eq!(loc.filter_mask, 0);
        // On-disk size should be original + 4 checksum bytes.
        assert_eq!(loc.size, data.len() as u64 + 4);

        let result = read_chunk_raw(&cursor, &loc, data.len(), &[3], &registry, None).unwrap();
        assert_eq!(result, data);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn filter_mask_skips_filter() {
        use consus_io::MemCursor;
        let data = b"no filter applied";
        let cursor = MemCursor::from_bytes(data.to_vec());
        let loc = ChunkLocation {
            address: 0,
            size: data.len() as u64,
            filter_mask: 0x01, // bit 0 set → filter 0 not applied
        };
        let registry = consus_compression::DefaultCodecRegistry::new();
        // Even though filter_ids has deflate(1), the mask says it wasn't applied.
        let result = read_chunk_raw(&cursor, &loc, data.len(), &[1], &registry, None).unwrap();
        assert_eq!(result, data);
    }
    /// read_chunk_raw returns fill-value-tiled buffer for undefined address.
    ///
    /// Invariant: UNDEFINED_ADDRESS causes early return without I/O;
    /// the buffer is tiled with the fill value pattern.
    /// Also verifies that None fill value yields all-zero buffer.
    #[cfg(feature = "alloc")]
    #[test]
    fn read_chunk_fill_value_for_undefined_address() {
        use consus_io::MemCursor;
        // address = u64::MAX is UNDEFINED_ADDRESS per constants.
        let loc = ChunkLocation {
            address: u64::MAX,
            size: 0,
            filter_mask: 0,
        };
        let registry = consus_compression::DefaultCodecRegistry::new();
        let empty_source = MemCursor::from_bytes(vec![]);

        // With fill_value = Some(&[0xFF, 0x00]): buffer tiled with pattern.
        let fv: &[u8] = &[0xFF, 0x00];
        let result = read_chunk_raw(&empty_source, &loc, 8, &[], &registry, Some(fv)).unwrap();
        assert_eq!(
            result,
            vec![0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00],
            "fill pattern must tile across buffer"
        );

        // With fill_value = None: buffer is all zeros.
        let zeros = read_chunk_raw(&empty_source, &loc, 8, &[], &registry, None).unwrap();
        assert_eq!(
            zeros,
            vec![0u8; 8],
            "none fill value must yield zero buffer"
        );
    }

}
