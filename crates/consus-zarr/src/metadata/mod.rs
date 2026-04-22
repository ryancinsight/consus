//! Zarr array and group metadata.
//!
//! This module defines the canonical in-memory representation of Zarr metadata
//! for both v2 and v3, along with JSON serialization/deserialization.
//!
//! ## Zarr v2 `.zarray` Schema
//!
//! ```json
//! {
//!   "zarr_format": 2,
//!   "shape": [1000, 1000],
//!   "chunks": [100, 100],
//!   "dtype": "<f8",
//!   "compressor": {"id": "zlib", "level": 1},
//!   "fill_value": 0,
//!   "order": "C",
//!   "filters": null
//! }
//! ```
//!
//! ## Zarr v3 `zarr.json` Schema
//!
//! ```json
//! {
//!   "zarr_format": 3,
//!   "node_type": "array",
//   "shape": [1000, 1000],
//!   "data_type": "float64",
//!   "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100, 100]}},
//!   "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
//!   "codecs": [
//!     {"name": "bytes", "configuration": {"endian": "little"}},
//!     {"name": "gzip", "configuration": {"level": 1}}
//!   ],
//!   "fill_value": 0
//! }
//! ```
//!
//! ## Module Hierarchy
//!
//! ```text
//! metadata/
//! ├── mod.rs          # Shared types: ArrayMetadata, GroupMetadata, Codec, FillValue
//! ├── v2.rs           # .zarray, .zgroup, .zattrs parse + serialize
//! ├── v3.rs           # zarr.json parse + serialize
//! └── consolidated.rs # metadata.fo consolidated format
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
mod consolidated;
#[cfg(feature = "alloc")]
mod v2;
#[cfg(feature = "alloc")]
mod v3;

#[cfg(feature = "alloc")]
pub use consolidated::ConsolidatedMetadata;
#[cfg(feature = "alloc")]
pub use v2::{
    ArrayMetadataV2, CompressorConfig, FilterId, GroupMetadataV2, parse_zattrs, serialize_zattrs,
};
#[cfg(feature = "alloc")]
pub use v3::ZarrJson;
#[cfg(feature = "std")]
pub use v3::{WriteZarrJsonError, write_group_json, write_zarr_json};

// ---------------------------------------------------------------------------
// Zarr format version
// ---------------------------------------------------------------------------

/// Zarr format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZarrVersion {
    V2,
    V3,
}

// ---------------------------------------------------------------------------
// Fill value
// ---------------------------------------------------------------------------

/// Fill value representation.
///
/// Zarr fill values are JSON-serializable and may be primitive scalars
/// or special values like `NaN`, `Infinity`, `-Infinity`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub enum FillValue {
    /// The default fill value (zero for numeric types, empty string for string types).
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
    String(alloc::string::String),
    /// Byte array fill value.
    Bytes(alloc::vec::Vec<u8>),
}

#[cfg(feature = "alloc")]
impl Default for FillValue {
    fn default() -> Self {
        Self::Default
    }
}

// ---------------------------------------------------------------------------
// Array metadata (format-agnostic)
// ---------------------------------------------------------------------------

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
    pub shape: alloc::vec::Vec<usize>,
    /// Chunk shape.
    pub chunks: alloc::vec::Vec<usize>,
    /// Data type string.
    /// - v2: NumPy dtype string (e.g., `"<f8"`, `"|S10"`)
    /// - v3: Named type (e.g., `"float64"`, `"Uint32"`, `"VLen<Unicode>"`)
    pub dtype: alloc::string::String,
    /// Fill value.
    pub fill_value: FillValue,
    /// Memory order: `'C'` (row-major) or `'F'` (column-major).
    pub order: char,
    /// Codecs applied to each chunk.
    /// - v2: at most one compressor codec (and optional filters list)
    /// - v3: ordered codec chain
    pub codecs: alloc::vec::Vec<Codec>,
    /// For v3 arrays: chunk key encoding configuration.
    pub chunk_key_encoding: ChunkKeyEncoding,
    /// Optional dimension names for Zarr v3 arrays.
    pub dimension_names: Option<alloc::vec::Vec<alloc::string::String>>,
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
    pub fn chunk_grid(&self) -> alloc::vec::Vec<usize> {
        self.shape
            .iter()
            .zip(self.chunks.iter())
            .map(|(&s, &c)| s.div_ceil(c))
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
    pub name: alloc::string::String,
    /// Separator for the default encoding.
    /// v2 default: `"/"` (produces keys like `"c/0/0/0"`).
    /// v2 compat: `"."` (produces keys like `"0.0.0"`).
    pub separator: char,
}

#[cfg(feature = "alloc")]
impl Default for ChunkKeyEncoding {
    fn default() -> Self {
        Self {
            name: alloc::string::String::from("default"),
            separator: '/',
        }
    }
}

// ---------------------------------------------------------------------------
// Group metadata (format-agnostic)
// ---------------------------------------------------------------------------

/// Zarr group metadata for both v2 and v3.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct GroupMetadata {
    /// Zarr format version.
    pub version: ZarrVersion,
    /// Custom attributes (key-value pairs).
    pub attributes: alloc::vec::Vec<(alloc::string::String, AttributeValue)>,
    /// For v3: group codec chain.
    pub codecs: alloc::vec::Vec<Codec>,
}

#[cfg(feature = "alloc")]
impl Default for GroupMetadata {
    fn default() -> Self {
        Self {
            version: ZarrVersion::V3,
            attributes: alloc::vec::Vec::new(),
            codecs: alloc::vec::Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Attribute value
// ---------------------------------------------------------------------------

/// Attribute value for Zarr group attributes.
///
/// Zarr attributes are stored as JSON in `.zattrs`. Values may be scalars
/// or arrays of any JSON-compatible type.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    String(alloc::string::String),
    BoolArray(alloc::vec::Vec<bool>),
    IntArray(alloc::vec::Vec<i64>),
    UintArray(alloc::vec::Vec<u64>),
    FloatArray(alloc::vec::Vec<f64>),
    StringArray(alloc::vec::Vec<alloc::string::String>),
}

#[cfg(feature = "alloc")]
impl AttributeValue {
    /// Number of elements (1 for scalar, length for array).
    pub fn num_elements(&self) -> usize {
        match self {
            Self::Bool(_) | Self::Int(_) | Self::Uint(_) | Self::Float(_) | Self::String(_) => 1,
            Self::BoolArray(v) => v.len(),
            Self::IntArray(v) => v.len(),
            Self::UintArray(v) => v.len(),
            Self::FloatArray(v) => v.len(),
            Self::StringArray(v) => v.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------

/// A codec configuration entry.
///
/// Represents a codec in a Zarr v2 or v3 codec chain. Each codec has a
/// name and an optional configuration object.
///
/// ## Zarr v3 Codec Names
///
/// | Name | Description |
/// |------|-------------|
/// | `"bytes"` | Raw byte transport (endianness) |
/// | `"crc32"` | CRC-32 checksum |
/// | `"gzip"` | Gzip compression |
/// | `"zstd"` | Zstandard compression |
/// | `"lz4"` | LZ4 compression |
/// | `"blosc"` | Blosc meta-compressor |
/// | `"sharding"` | Sharding codec |
///
/// ## Zarr v2 Compressor IDs
///
/// | ID | Codec |
/// |----|-------|
/// | `"zlib"` | deflate |
/// | `"gzip"` | gzip |
/// | `"blosc"` | blosc |
/// | `"lz4"` | lz4 |
/// | `"zstd"` | zstd |
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Codec {
    /// Codec name (e.g., `"gzip"`, `"zstd"`, `"bytes"`).
    pub name: alloc::string::String,
    /// Optional codec configuration as key-value pairs.
    pub configuration: alloc::vec::Vec<(alloc::string::String, alloc::string::String)>,
}

#[cfg(feature = "alloc")]
impl Codec {
    /// Returns the gzip compression level if this is a gzip codec.
    pub fn gzip_level(&self) -> Option<u32> {
        if self.name == "gzip" {
            self.configuration
                .iter()
                .find(|(k, _)| k == "level")
                .and_then(|(_, v)| v.parse().ok())
        } else {
            None
        }
    }

    /// Returns the zstd compression level if this is a zstd codec.
    pub fn zstd_level(&self) -> Option<i32> {
        if self.name == "zstd" {
            self.configuration
                .iter()
                .find(|(k, _)| k == "level")
                .and_then(|(_, v)| v.parse().ok())
        } else {
            None
        }
    }

    /// Returns a boolean configuration flag for this codec.
    pub fn bool_flag(&self, key: &str) -> Option<bool> {
        self.configuration
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.parse().ok())
    }

    /// Returns the zstd checksum flag if this is a zstd codec.
    pub fn zstd_checksum(&self) -> Option<bool> {
        if self.name == "zstd" {
            self.bool_flag("checksum")
        } else {
            None
        }
    }

    /// Returns the lz4 compression level if this is an lz4 codec.
    pub fn lz4_level(&self) -> Option<i32> {
        if self.name == "lz4" {
            self.configuration
                .iter()
                .find(|(k, _)| k == "level")
                .and_then(|(_, v)| v.parse().ok())
        } else {
            None
        }
    }

    /// Returns the endianness configuration if this is a bytes codec.
    pub fn bytes_endian(&self) -> Option<&str> {
        self.configuration
            .iter()
            .find(|(k, _)| k == "endian")
            .map(|(_, v)| v.as_str())
    }

    /// Returns true if this codec is a no-op (identity).
    pub fn is_identity(&self) -> bool {
        self.name == "bytes"
            && self
                .configuration
                .iter()
                .all(|(k, v)| k == "endian" && v == "native")
    }
}

// ---------------------------------------------------------------------------
// dtype utilities
// ---------------------------------------------------------------------------

/// Compute the element size in bytes for a given dtype string.
///
/// Returns `None` for variable-length types (vlen, unicode, bytes).
///
/// ## Supported dtypes
///
/// - Boolean: 1 byte
/// - Integer: 1, 2, 4, 8 bytes (signed and unsigned)
/// - Float: 2, 4, 8, 16 bytes
/// - Complex: 8, 16 bytes (real+imag)
/// - String: fixed (size encoded in dtype) or variable (None)
/// - Bitfield: size encoded in dtype
/// - Time: 8 bytes
/// - Reference: 8 bytes
/// - Enum: base type size
/// - Array: product of dims × base size
/// - Compound: sum of field sizes
/// - Opaque: size encoded in dtype
#[cfg(feature = "alloc")]
pub fn dtype_to_element_size(dtype: &str) -> Option<usize> {
    // Normalize: lowercase, strip whitespace
    let dt = dtype.trim().to_lowercase();

    // ---- v3 named types ----
    match dt.as_str() {
        "bool" | "bool_" => return Some(1),
        "int8" => return Some(1),

        "uint8" | "i1" | "u1" => return Some(1),
        "int16" | "i2" => return Some(2),
        "uint16" | "u2" => return Some(2),
        "int32" | "i4" => return Some(4),
        "uint32" | "u4" => return Some(4),
        "int64" | "i8" => return Some(8),
        "uint64" | "u8" => return Some(8),
        "float16" | "f2" => return Some(2),
        "float32" | "f4" => return Some(4),
        "float64" | "f8" => return Some(8),
        "complex64" | "c8" => return Some(8),
        "complex128" | "c16" => return Some(16),
        "reference" | "object" => return Some(8),
        "string" | "utf8" | "unicode" => return None, // variable
        "bytes" | "binary" => return None,            // variable
        // vlen
        s if s.starts_with("vlen<") => return None,
        _ => {}
    }

    // ---- v2 numpy dtype strings ----
    // Format: <|> endianness, then type characters, then size for arrays
    // Examples: "<f8", "|S10", "<i4", ">u2", "|O8", "<c8"
    let bytes = dt.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    // Peek at first character for endianness
    let (rest, _) = match bytes[0] {
        b'<' | b'>' | b'=' | b'|' => (&bytes[1..], bytes[0]),
        _ => (bytes, b'|'), // default to native or no-endian (for byte strings)
    };

    if rest.is_empty() {
        return None;
    }

    match rest[0] {
        // Boolean
        b'b' => Some(1),
        // Signed integer
        b'i' => {
            if rest.len() >= 2 {
                let size_str: alloc::string::String = rest[1..]
                    .iter()
                    .take_while(|&&c| c.is_ascii_digit())
                    .map(|&c| c as char)
                    .collect();
                size_str.parse().ok()
            } else {
                None
            }
        }
        // Unsigned integer
        b'u' => {
            if rest.len() >= 2 {
                let size_str: alloc::string::String = rest[1..]
                    .iter()
                    .take_while(|&&c| c.is_ascii_digit())
                    .map(|&c| c as char)
                    .collect();
                size_str.parse().ok()
            } else {
                None
            }
        }
        // Float
        b'f' => {
            if rest.len() >= 2 {
                let size_str: alloc::string::String = rest[1..]
                    .iter()
                    .take_while(|&&c| c.is_ascii_digit())
                    .map(|&c| c as char)
                    .collect();
                size_str.parse().ok()
            } else {
                None
            }
        }
        // Complex
        b'c' => {
            if rest.len() >= 2 {
                let size_str: alloc::string::String = rest[1..]
                    .iter()
                    .take_while(|&&c| c.is_ascii_digit())
                    .map(|&c| c as char)
                    .collect();
                size_str.parse().ok()
            } else {
                None
            }
        }
        // String (fixed-length)
        b's' => {
            if rest.len() >= 2 {
                let size_str: alloc::string::String = rest[1..]
                    .iter()
                    .take_while(|&&c| c.is_ascii_digit())
                    .map(|&c| c as char)
                    .collect();
                size_str.parse().ok()
            } else {
                None
            }
        }
        // Void / opaque
        b'v' => {
            if rest.len() >= 2 {
                let size_str: alloc::string::String = rest[1..]
                    .iter()
                    .take_while(|&&c| c.is_ascii_digit())
                    .map(|&c| c as char)
                    .collect();
                size_str.parse().ok()
            } else {
                Some(1)
            }
        }
        // Object reference
        b'o' => Some(8),
        // Date/time (8 bytes)
        b'm' => Some(8),
        // Timedelta (8 bytes)
        b't' => Some(8),
        // Bitfield
        b'B' => {
            if rest.len() >= 2 {
                let size_str: alloc::string::String = rest[1..]
                    .iter()
                    .take_while(|&&c| c.is_ascii_digit())
                    .map(|&c| c as char)
                    .collect();
                size_str.parse().ok()
            } else {
                None
            }
        }
        // Enum base is the underlying integer type
        b'e' => {
            if rest.len() >= 2 {
                let size_str: alloc::string::String = rest[1..]
                    .iter()
                    .take_while(|&&c| c.is_ascii_digit())
                    .map(|&c| c as char)
                    .collect();
                size_str.parse().ok()
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_to_element_size_v3_named() {
        assert_eq!(dtype_to_element_size("bool"), Some(1));
        assert_eq!(dtype_to_element_size("int8"), Some(1));
        assert_eq!(dtype_to_element_size("uint16"), Some(2));
        assert_eq!(dtype_to_element_size("int32"), Some(4));
        assert_eq!(dtype_to_element_size("uint64"), Some(8));
        assert_eq!(dtype_to_element_size("float32"), Some(4));
        assert_eq!(dtype_to_element_size("float64"), Some(8));
        assert_eq!(dtype_to_element_size("complex64"), Some(8));
        assert_eq!(dtype_to_element_size("complex128"), Some(16));
        assert_eq!(dtype_to_element_size("float16"), Some(2));
    }

    #[test]
    fn test_dtype_to_element_size_v2_numpy() {
        assert_eq!(dtype_to_element_size("<f8"), Some(8));
        assert_eq!(dtype_to_element_size("<i4"), Some(4));
        assert_eq!(dtype_to_element_size(">u2"), Some(2));
        assert_eq!(dtype_to_element_size("<c8"), Some(8));
        assert_eq!(dtype_to_element_size("|S10"), Some(10));
        assert_eq!(dtype_to_element_size("<V16"), Some(16));
    }

    #[test]
    fn test_dtype_to_element_size_variable() {
        assert_eq!(dtype_to_element_size("string"), None);
        assert_eq!(dtype_to_element_size("utf8"), None);
        assert_eq!(dtype_to_element_size("vlen<unicode>"), None);
        assert_eq!(dtype_to_element_size("vlen<uint8>"), None);
        assert_eq!(dtype_to_element_size("bytes"), None);
    }

    #[test]
    fn test_fill_value_default() {
        assert_eq!(FillValue::default(), FillValue::Default);
    }

    #[test]
    fn test_codec_is_identity() {
        let bytes_native = Codec {
            name: alloc::string::String::from("bytes"),
            configuration: alloc::vec![(
                alloc::string::String::from("endian"),
                alloc::string::String::from("native")
            )],
        };
        assert!(bytes_native.is_identity());

        let gzip = Codec {
            name: alloc::string::String::from("gzip"),
            configuration: alloc::vec![(
                alloc::string::String::from("level"),
                alloc::string::String::from("1")
            )],
        };
        assert!(!gzip.is_identity());
    }

    #[test]
    fn test_codec_gzip_level() {
        let gzip = Codec {
            name: alloc::string::String::from("gzip"),
            configuration: alloc::vec![(
                alloc::string::String::from("level"),
                alloc::string::String::from("6")
            )],
        };
        assert_eq!(gzip.gzip_level(), Some(6));
    }

    #[test]
    fn test_codec_bool_flag_parses_true() {
        let codec = Codec {
            name: alloc::string::String::from("zstd"),
            configuration: alloc::vec![(
                alloc::string::String::from("checksum"),
                alloc::string::String::from("true")
            )],
        };

        assert_eq!(codec.bool_flag("checksum"), Some(true));
    }

    #[test]
    fn test_codec_bool_flag_parses_false() {
        let codec = Codec {
            name: alloc::string::String::from("zstd"),
            configuration: alloc::vec![(
                alloc::string::String::from("checksum"),
                alloc::string::String::from("false")
            )],
        };

        assert_eq!(codec.bool_flag("checksum"), Some(false));
    }

    #[test]
    fn test_zstd_checksum_extraction() {
        let codec = Codec {
            name: alloc::string::String::from("zstd"),
            configuration: alloc::vec![(
                alloc::string::String::from("checksum"),
                alloc::string::String::from("false")
            )],
        };

        assert_eq!(codec.zstd_checksum(), Some(false));
    }

    #[test]
    fn test_zstd_checksum_non_zstd_codec_returns_none() {
        let codec = Codec {
            name: alloc::string::String::from("gzip"),
            configuration: alloc::vec![(
                alloc::string::String::from("checksum"),
                alloc::string::String::from("true")
            )],
        };

        assert_eq!(codec.zstd_checksum(), None);
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

    #[test]
    fn test_attribute_value_num_elements() {
        assert_eq!(AttributeValue::Int(42).num_elements(), 1);
        assert_eq!(
            AttributeValue::IntArray(alloc::vec![1, 2, 3]).num_elements(),
            3
        );
    }
}
