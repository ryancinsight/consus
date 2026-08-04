#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use super::{codec::Codec, dtype::dtype_to_element_size};

/// Zarr format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZarrVersion {
    V2,
    V3,
}

/// Fill value representation.
///
/// Zarr fill values are JSON-serializable and may be primitive scalars
/// or special values like `NaN`, `Infinity`, `-Infinity`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum FillValue {
    /// The default fill value (zero for numeric types, empty string for string types).
    #[default]
    Default,
    /// Null / missing value.
    Null,
    /// Boolean fill value.
    Bool(bool),
    /// Integer fill value.
    Int(i64),
    /// Unsigned integer fill value.
    Uint(u64),
    /// Float fill value stored as raw JSON representation.
    Float(String),
    /// String fill value.
    String(String),
    /// Byte array fill value.
    Bytes(Vec<u8>),
}

/// Zarr array metadata covering both v2 and v3.
///
/// This is the canonical in-memory representation after parsing any Zarr
/// metadata file, independent of the source format version.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ArrayMetadata {
    /// Zarr format version.
    pub version: ZarrVersion,
    /// Array shape (dimension extents).
    pub shape: Vec<usize>,
    /// Chunk shape.
    pub chunks: Vec<usize>,
    /// Data type string.
    /// - v2: NumPy dtype string (e.g., `"<f8"`, `"|S10"`)
    /// - v3: Named type (e.g., `"float64"`, `"Uint32"`, `"VLen<Unicode>"`)
    pub dtype: String,
    /// Fill value.
    pub fill_value: FillValue,
    /// Memory order: `'C'` (row-major) or `'F'` (column-major).
    pub order: char,
    /// Codecs applied to each chunk.
    /// - v2: at most one compressor codec (and optional filters list)
    /// - v3: ordered codec chain
    pub codecs: Vec<Codec>,
    /// For v3 arrays: chunk key encoding configuration.
    pub chunk_key_encoding: ChunkKeyEncoding,
    /// Optional dimension names for Zarr v3 arrays.
    pub dimension_names: Option<Vec<String>>,
}

#[cfg(feature = "alloc")]
impl ArrayMetadata {
    /// Total number of elements in the array.
    pub fn num_elements(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Number of chunks along each dimension.
    pub fn chunk_grid(&self) -> Vec<usize> {
        self.shape
            .iter()
            .zip(self.chunks.iter())
            .map(|(&shape, &chunk)| shape.div_ceil(chunk))
            .collect()
    }

    /// Total number of chunks in the array.
    pub fn total_chunks(&self) -> usize {
        self.chunk_grid().iter().product()
    }

    /// Element size in bytes (None for variable-length types).
    pub fn element_size(&self) -> Option<usize> {
        dtype_to_element_size(&self.dtype)
    }
}

/// Chunk key encoding strategy.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkKeyEncoding {
    /// Encoding name: `"default"` or `"v2"`.
    pub name: String,
    /// Separator for the default encoding.
    /// v2 default: `"/"` (produces keys like `"c/0/0/0"`).
    /// v2 compat: `"."` (produces keys like `"0.0.0"`).
    pub separator: char,
}

#[cfg(feature = "alloc")]
impl Default for ChunkKeyEncoding {
    fn default() -> Self {
        Self {
            name: String::from("default"),
            separator: '/',
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_fill_value_default() {
        assert_eq!(FillValue::default(), FillValue::Default);
    }

    #[test]
    fn test_array_metadata_chunk_grid() {
        let meta = ArrayMetadata {
            version: ZarrVersion::V3,
            shape: alloc::vec![100, 100, 100],
            chunks: alloc::vec![10, 10, 10],
            dtype: alloc::string::String::from("float64"),
            fill_value: FillValue::default(),
            order: 'C',
            codecs: alloc::vec![],
            chunk_key_encoding: ChunkKeyEncoding::default(),
            dimension_names: None,
        };
        assert_eq!(meta.chunk_grid(), alloc::vec![10, 10, 10]);
        assert_eq!(meta.total_chunks(), 1000);
        assert_eq!(meta.num_elements(), 1_000_000);
    }
}
