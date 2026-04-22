//! Zarr v2 metadata parsing and serialization.
//!
//! ## Zarr v2 Specification
//!
//! Zarr v2 stores array metadata in `.zarray` files and group metadata
//! in `.zgroup` files. Both are JSON documents. Attributes are stored
//! separately in `.zattrs`.
//!
//! ## File Structure
//!
//! ```text
//! array.zarr/
//! ├── .zarray          # Array metadata (this module)
//! ├── .zattrs          # Custom attributes (JSON object)
//! ├── .zgroup          # Group metadata (this module)
//! └── c/               # Chunk data directory
//!     ├── 0.0.0        # Chunk at grid coordinate (0, 0)
//!     ├── 0.0.1
//!     └── ...
//! ```
//!
//! ## Compressor ID Mapping
//!
//! | `.zarray` compressor id | consus-compression codec |
//! |-------------------------|--------------------------|
//! | `"zlib"`                | deflate (filter ID 1)    |
//! | `"gzip"`                | gzip                     |
//! | `"blosc"`               | blosc (filter ID 32001)  |
//! | `"lz4"`                 | lz4 (filter ID 32004)    |
//! | `"zstd"`                | zstd (filter ID 32015)   |
//! | `null`                  | no compression           |
//!
//! ## Filters (deprecated in v3)
//!
//! Zarr v2 allowed a `filters` array in `.zarray`. Each filter entry has
//! an `id` field mapping to an HDF5-style filter number or a custom
//! codec name. This module handles the standard set used by zarr-python.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

#[cfg(feature = "alloc")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "alloc")]
use super::{
    ArrayMetadata, AttributeValue, ChunkKeyEncoding, Codec, FillValue, GroupMetadata, ZarrVersion,
};

// ---------------------------------------------------------------------------
// .zarray — Array metadata
// ---------------------------------------------------------------------------

/// Root metadata for a Zarr v2 array.
///
/// Parsed from the `.zarray` JSON file in the array directory.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayMetadataV2 {
    /// Must be 2 for Zarr v2.
    pub zarr_format: u32,

    /// Array shape as a list of dimension sizes.
    pub shape: Vec<usize>,

    /// Chunk shape as a list of chunk sizes per dimension.
    pub chunks: Vec<usize>,

    /// NumPy dtype string (e.g., `"<f8"`, `"|S10"`).
    pub dtype: String,

    /// Fill value. May be a scalar, `null`, or a special JSON value.
    #[serde(default)]
    pub fill_value: FillValueJson,

    /// Memory layout order: `"C"` (row-major) or `"F"` (column-major).
    #[serde(default = "default_order")]
    pub order: char,

    /// Compressor configuration. `null` means no compression.
    #[serde(default)]
    pub compressor: Option<CompressorConfig>,

    /// Deprecated filter list. `null` or empty in modern files.
    #[serde(default)]
    pub filters: Option<Vec<FilterConfig>>,

    /// Chunk dimension separator used by filesystem-backed stores.
    ///
    /// Zarr v2 commonly uses `"."` for keys like `"0.1"`, while some stores
    /// may use `"/"` for nested chunk paths.
    #[serde(default = "default_dimension_separator")]
    pub dimension_separator: String,
}

fn default_order() -> char {
    'C'
}

fn default_dimension_separator() -> String {
    ".".to_string()
}

/// JSON representation of a fill value in `.zarray`.
/// Zarr v2 allows `null`, scalars, and the special JSON literals
/// `Infinity`, `-Infinity`, and `NaN` (written as a string).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[derive(Default)]
pub enum FillValueJson {
    /// The literal JSON `null` — means "use default fill value".
    #[default]
    Null,
    /// Boolean fill value.
    Bool(bool),
    /// Integer fill value.
    Int(i64),
    /// Unsigned integer fill value.
    Uint(u64),
    /// Float fill value (may be a JSON number or a string like `"NaN"`).
    Float(f64),
    /// Special floating-point values encoded as strings by zarr-python.
    /// Must precede String(String) in declaration order so that serde's
    /// untagged deserialization tries SpecialFillValue before falling through
    /// to the generic String variant (which matches any JSON string).
    Special(SpecialFillValue),
    /// String fill value.
    String(String),
}

/// Special floating-point fill values serialized as strings by zarr-python.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecialFillValue {
    #[serde(rename = "Infinity")]
    Infinity,
    #[serde(rename = "-Infinity")]
    NegInfinity,
    #[serde(rename = "NaN")]
    NaN,
}

impl FillValueJson {
    /// Convert the JSON fill value to the canonical `FillValue` type.
    pub fn to_fill_value(&self) -> FillValue {
        match self {
            Self::Null => FillValue::Default,
            Self::Bool(b) => FillValue::Bool(*b),
            Self::Int(i) => FillValue::Int(*i),
            Self::Uint(u) => FillValue::Uint(*u),
            Self::Float(f) => FillValue::Float(
                serde_json::Number::from_f64(*f)
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "NaN".to_string()),
            ),
            Self::String(s) => FillValue::String(s.clone()),
            Self::Special(s) => FillValue::Float(match s {
                SpecialFillValue::Infinity => "Infinity".to_string(),
                SpecialFillValue::NegInfinity => "-Infinity".to_string(),
                SpecialFillValue::NaN => "NaN".to_string(),
            }),
        }
    }
}

/// Compressor configuration in Zarr v2.
///
/// Maps to a named codec in `consus-compression`. The `id` field
/// is either a string codec name (zarr-python style) or an HDF5 filter ID.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CompressorConfig {
    /// A compressor identified by a named string id (e.g., `"zlib"`, `"gzip"`).
    Named(CompressorNamed),
    /// An HDF5-style filter with an integer id.
    FilterId(FilterIdConfig),
    /// `null` means no compression.
    Null,
}

/// Named compressor with optional configuration.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressorNamed {
    /// Codec identifier (e.g., `"zlib"`, `"gzip"`, `"blosc"`, `"lz4"`, `"zstd"`).
    pub id: String,

    /// Optional codec configuration (level, blocksize, etc.).
    #[serde(default)]
    pub configuration: Option<CodecConfiguration>,

    /// Inline gzip/zlib level used by zarr-python v2 metadata.
    #[serde(default)]
    pub level: Option<i32>,
}

/// An HDF5 filter referenced by its numeric filter ID.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterIdConfig {
    /// HDF5 filter ID (e.g., `1` = deflate, `32004` = lz4, `32015` = zstd).
    pub id: u16,

    /// Optional filter parameters.
    #[serde(default)]
    pub configuration: Option<CodecConfiguration>,
}

/// Codec configuration as a list of key-value integer pairs.
///
/// Used by both compressors and filters in Zarr v2.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CodecConfiguration {
    /// Key-value configuration pairs.
    /// Each entry is a list `[key, value]` as required by the Zarr spec.
    #[serde(default)]
    pub elements: Vec<CodecConfigElement>,
}

/// A single key-value pair in a codec configuration.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CodecConfigElement {
    /// A two-element list `[key_name, value]`.
    Pair([String; 2]),
    /// A bare integer value (used for level, etc.).
    Value(i32),
}

/// Legacy filter configuration (Zarr v2 `filters` array entry).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Filter identifier (string name or integer HDF5 filter ID).
    pub id: FilterId,

    /// Optional filter parameters.
    #[serde(default)]
    pub configuration: Option<CodecConfiguration>,
}

/// A filter identifier: either a string name or an integer HDF5 filter ID.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterId {
    Name(String),
    Number(u16),
}

impl ArrayMetadataV2 {
    /// Parse a `.zarray` JSON document into an `ArrayMetadataV2`.
    #[cfg(feature = "std")]
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        serde_json::from_str(text).map_err(ParseError::Json)
    }

    /// Serialize this array metadata to a `.zarray` JSON document.
    #[cfg(feature = "std")]
    pub fn to_json(&self) -> Result<String, SerializeError> {
        serde_json::to_string_pretty(self).map_err(SerializeError::Json)
    }

    /// Convert to the canonical `ArrayMetadata` representation.
    pub fn to_canonical(&self) -> ArrayMetadata {
        let mut codecs = Vec::new();

        // Compressor -> codec entry
        if let Some(compressor) = &self.compressor {
            if let Some(codec) = compressor.to_codec() {
                codecs.push(codec);
            }
        }

        // Filters (for completeness; most v2 files have no filters)
        if let Some(filters) = &self.filters {
            for f in filters {
                if let Some(codec) = f.to_codec() {
                    codecs.push(codec);
                }
            }
        }

        ArrayMetadata {
            version: ZarrVersion::V2,
            shape: self.shape.clone(),
            chunks: self.chunks.clone(),
            dtype: self.dtype.clone(),
            fill_value: self.fill_value.to_fill_value(),
            order: self.order,
            codecs,
            chunk_key_encoding: ChunkKeyEncoding {
                name: if self.dimension_separator == "." {
                    String::from("v2")
                } else {
                    String::from("default")
                },
                separator: self.dimension_separator.chars().next().unwrap_or('.'),
            },
            dimension_names: None,
        }
    }
}

impl CompressorConfig {
    /// Convert to a canonical `Codec` if this compressor is not null.
    fn to_codec(&self) -> Option<Codec> {
        match self {
            Self::Named(named) => Some(named.to_codec()),
            Self::FilterId(fid) => Some(fid.to_codec()),
            Self::Null => None,
        }
    }
}

impl CompressorNamed {
    /// Convert to a canonical `Codec`.
    fn to_codec(&self) -> Codec {
        let mut configuration = Vec::new();
        if let Some(cfg) = &self.configuration {
            for elem in &cfg.elements {
                if let Some(pair) = elem.to_pair() {
                    configuration.push(pair);
                }
            }
        }
        if let Some(level) = self.level {
            configuration.push((String::from("level"), level.to_string()));
        }
        Codec {
            name: self.id.clone(),
            configuration,
        }
    }
}

impl FilterIdConfig {
    /// Convert an HDF5 filter ID to a canonical codec name.
    fn to_codec(&self) -> Codec {
        let name = hdf5_filter_id_to_name(self.id);
        let mut configuration = Vec::new();
        if let Some(cfg) = &self.configuration {
            for elem in &cfg.elements {
                if let Some(pair) = elem.to_pair() {
                    configuration.push(pair);
                }
            }
        }
        Codec {
            name,
            configuration,
        }
    }
}

impl FilterConfig {
    /// Convert to a canonical `Codec`.
    fn to_codec(&self) -> Option<Codec> {
        let name = match &self.id {
            FilterId::Name(s) => s.clone(),
            FilterId::Number(n) => hdf5_filter_id_to_name(*n),
        };
        let mut configuration = Vec::new();
        if let Some(cfg) = &self.configuration {
            for elem in &cfg.elements {
                if let Some(pair) = elem.to_pair() {
                    configuration.push(pair);
                }
            }
        }
        Some(Codec {
            name,
            configuration,
        })
    }
}

impl CodecConfigElement {
    /// Convert to an optional `(key, value)` string pair.
    fn to_pair(&self) -> Option<(String, String)> {
        match self {
            Self::Pair([k, v]) => Some((k.clone(), v.clone())),
            Self::Value(v) => Some(("level".to_string(), v.to_string())),
        }
    }
}

/// Map an HDF5 filter ID to a codec name string.
fn hdf5_filter_id_to_name(id: u16) -> String {
    match id {
        1 => String::from("deflate"),
        2 => String::from("shuffle"),
        3 => String::from("fletcher32"),
        4 => String::from("szip"),
        5 => String::from("nbit"),
        307 => String::from("scaleoffset"),
        32001 => String::from("blosc"),
        32004 => String::from("lz4"),
        32015 => String::from("zstd"),
        _ => format!("filter{}", id),
    }
}

// ---------------------------------------------------------------------------
// .zgroup — Group metadata
// ---------------------------------------------------------------------------

/// Root metadata for a Zarr v2 group.
///
/// Parsed from the `.zgroup` JSON file in the group directory.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMetadataV2 {
    /// Must be 2 for Zarr v2.
    pub zarr_format: u32,

    /// Custom attributes. Parsed separately from `.zattrs`.
    #[serde(default)]
    pub attributes: Option<serde_json::Map<String, serde_json::Value>>,
}

impl GroupMetadataV2 {
    /// Parse a `.zgroup` JSON document.
    #[cfg(feature = "std")]
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        serde_json::from_str(text).map_err(ParseError::Json)
    }

    /// Serialize to a `.zgroup` JSON document.
    #[cfg(feature = "std")]
    pub fn to_json(&self) -> Result<String, SerializeError> {
        serde_json::to_string_pretty(self).map_err(SerializeError::Json)
    }

    /// Convert to the canonical `GroupMetadata` representation.
    pub fn to_canonical(&self) -> GroupMetadata {
        let attributes = self
            .attributes
            .as_ref()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), json_value_to_attribute(v.clone())))
                    .collect()
            })
            .unwrap_or_default();
        GroupMetadata {
            version: ZarrVersion::V2,
            attributes,
            codecs: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// .zattrs — Attribute parsing helpers
// ---------------------------------------------------------------------------

/// Parse a `.zattrs` JSON document into a list of `(name, value)` pairs.
#[cfg(feature = "alloc")]
pub fn parse_zattrs(text: &str) -> Result<Vec<(String, AttributeValue)>, ParseError> {
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(text).map_err(ParseError::Json)?;
    Ok(map
        .into_iter()
        .map(|(k, v)| (k, json_value_to_attribute(v)))
        .collect())
}

/// Convert a serde JSON value to an `AttributeValue`.
pub(crate) fn json_value_to_attribute(v: serde_json::Value) -> AttributeValue {
    match v {
        serde_json::Value::Bool(b) => AttributeValue::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                AttributeValue::Int(i)
            } else if let Some(u) = n.as_u64() {
                AttributeValue::Uint(u)
            } else if let Some(f) = n.as_f64() {
                AttributeValue::Float(f)
            } else {
                AttributeValue::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => AttributeValue::String(s),
        serde_json::Value::Array(arr) => {
            // Try to determine the array element type from the first element.
            let values: Vec<AttributeValue> =
                arr.into_iter().map(json_value_to_attribute).collect();
            if values.is_empty() {
                return AttributeValue::StringArray(vec![]);
            }
            match &values[0] {
                AttributeValue::Bool(_) => {
                    if let Ok(parsed) = values
                        .into_iter()
                        .map(|av| match av {
                            AttributeValue::Bool(b) => Ok(b),
                            _ => Err(()),
                        })
                        .collect()
                    {
                        AttributeValue::BoolArray(parsed)
                    } else {
                        AttributeValue::StringArray(vec![])
                    }
                }
                AttributeValue::Int(_) => {
                    if let Ok(parsed) = values
                        .into_iter()
                        .map(|av| match av {
                            AttributeValue::Int(i) => Ok(i),
                            AttributeValue::Uint(u) => Ok(i64::try_from(u).unwrap_or(i64::MAX)),
                            _ => Err(()),
                        })
                        .collect()
                    {
                        AttributeValue::IntArray(parsed)
                    } else {
                        AttributeValue::StringArray(vec![])
                    }
                }
                AttributeValue::Uint(_) => {
                    if let Ok(parsed) = values
                        .into_iter()
                        .map(|av| match av {
                            AttributeValue::Uint(u) => Ok(u),
                            AttributeValue::Int(i) => Ok(u64::try_from(i).unwrap_or(u64::MAX)),
                            _ => Err(()),
                        })
                        .collect()
                    {
                        AttributeValue::UintArray(parsed)
                    } else {
                        AttributeValue::StringArray(vec![])
                    }
                }
                AttributeValue::Float(_) => {
                    if let Ok(parsed) = values
                        .into_iter()
                        .map(|av| match av {
                            AttributeValue::Float(f) => Ok(f),
                            AttributeValue::Int(i) => Ok(i as f64),
                            AttributeValue::Uint(u) => Ok(u as f64),
                            _ => Err(()),
                        })
                        .collect()
                    {
                        AttributeValue::FloatArray(parsed)
                    } else {
                        AttributeValue::StringArray(vec![])
                    }
                }
                AttributeValue::String(_) => {
                    if let Ok(parsed) = values
                        .into_iter()
                        .map(|av| match av {
                            AttributeValue::String(s) => Ok(s),
                            _ => Err(()),
                        })
                        .collect()
                    {
                        AttributeValue::StringArray(parsed)
                    } else {
                        AttributeValue::StringArray(vec![])
                    }
                }
                _ => AttributeValue::StringArray(vec![]),
            }
        }
        _ => AttributeValue::String(v.to_string()),
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Serialize a list of `(String, AttributeValue)` pairs to a `.zattrs` JSON document.
#[cfg(feature = "alloc")]
pub fn serialize_zattrs(attrs: &[(String, AttributeValue)]) -> Result<String, SerializeError> {
    let mut map = serde_json::Map::new();
    for (k, v) in attrs {
        map.insert(k.clone(), attribute_to_json_value(v));
    }
    serde_json::to_string_pretty(&map).map_err(SerializeError::Json)
}

/// Convert an `AttributeValue` to a serde JSON value.
fn attribute_to_json_value(av: &AttributeValue) -> serde_json::Value {
    match av {
        AttributeValue::Bool(b) => serde_json::Value::Bool(*b),
        AttributeValue::Int(i) => serde_json::json!(*i),
        AttributeValue::Uint(u) => serde_json::json!(*u),
        AttributeValue::Float(f) => serde_json::json!(*f),
        AttributeValue::String(s) => serde_json::Value::String(s.clone()),
        AttributeValue::BoolArray(arr) => {
            serde_json::Value::Array(arr.iter().map(|b| serde_json::Value::Bool(*b)).collect())
        }
        AttributeValue::IntArray(arr) => {
            serde_json::Value::Array(arr.iter().map(|i| serde_json::json!(i)).collect())
        }
        AttributeValue::UintArray(arr) => {
            serde_json::Value::Array(arr.iter().map(|u| serde_json::json!(u)).collect())
        }
        AttributeValue::FloatArray(arr) => {
            serde_json::Value::Array(arr.iter().map(|f| serde_json::json!(f)).collect())
        }
        AttributeValue::StringArray(arr) => serde_json::Value::Array(
            arr.iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur when parsing Zarr v2 metadata.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub enum ParseError {
    Json(serde_json::Error),
}

#[cfg(feature = "std")]
impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "JSON parse error: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
        }
    }
}

/// Errors that can occur when serializing Zarr v2 metadata.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub enum SerializeError {
    Json(serde_json::Error),
}

#[cfg(feature = "std")]
impl std::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "JSON serialize error: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SerializeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_zarray_minimal() {
        let json = r#"{
            "zarr_format": 2,
            "shape": [10, 10],
            "chunks": [5, 5],
            "dtype": "<f8",
            "fill_value": 0.0,
            "order": "C",
            "compressor": null,
            "filters": null
        }"#;
        let meta = ArrayMetadataV2::parse(json).expect("must parse");
        assert_eq!(meta.zarr_format, 2);
        assert_eq!(meta.shape, &[10, 10]);
        assert_eq!(meta.chunks, &[5, 5]);
        assert_eq!(meta.dtype, "<f8");
        assert!(meta.compressor.is_none());
    }

    #[test]
    fn test_parse_zarray_with_gzip() {
        let json = r#"{
            "zarr_format": 2,
            "shape": [100],
            "chunks": [10],
            "dtype": "<i4",
            "fill_value": -999,
            "order": "C",
            "compressor": {"id": "gzip", "configuration": {"level": 1}},
            "filters": null
        }"#;
        let meta = ArrayMetadataV2::parse(json).expect("must parse");
        let comp = meta.compressor.expect("compressor must be present");
        match comp {
            CompressorConfig::Named(n) => {
                assert_eq!(n.id, "gzip");
                let cfg = n.configuration.as_ref().expect("gzip config must exist");
                assert!(
                    cfg.elements.is_empty()
                        || cfg.elements
                            .iter()
                            .any(|e| matches!(e, CodecConfigElement::Pair(pair) if pair[0] == "level" && pair[1] == "1"))
                );
            }
            _ => panic!("expected Named compressor"),
        }
    }

    #[test]
    fn test_parse_zgroup() {
        let json = r#"{"zarr_format": 2}"#;
        let meta = GroupMetadataV2::parse(json).expect("must parse");
        assert_eq!(meta.zarr_format, 2);
    }

    #[test]
    fn test_array_metadata_v2_to_canonical() {
        let json = r#"{
  "zarr_format": 2,
  "shape": [100, 100],
  "chunks": [10, 10],
  "dtype": "<f8",
  "fill_value": 0.0,
  "order": "C",
  "compressor": {"id": "gzip", "configuration": {"level": 1}},
  "filters": null
}"#;
        let v2 = ArrayMetadataV2::parse(json).unwrap();
        let canon = v2.to_canonical();
        assert_eq!(canon.version, ZarrVersion::V2);
        assert_eq!(canon.shape, vec![100, 100]);
        assert_eq!(canon.chunks, vec![10, 10]);
        assert_eq!(canon.dtype, "<f8");
        assert_eq!(canon.order, 'C');
        assert_eq!(canon.codecs.len(), 1);
        assert_eq!(canon.codecs[0].name, "gzip");
        assert_eq!(canon.chunk_key_encoding.name, "v2");
        assert_eq!(canon.chunk_key_encoding.separator, '.');
    }

    #[test]
    fn test_array_metadata_v2_to_canonical_preserves_inline_gzip_level() {
        let json = r#"{
  "zarr_format": 2,
  "shape": [5, 4],
  "chunks": [2, 2],
  "dtype": "<f8",
  "fill_value": 0.0,
  "order": "C",
  "filters": null,
  "dimension_separator": ".",
  "compressor": {
    "id": "gzip",
    "level": 1
  }
}"#;
        let v2 = ArrayMetadataV2::parse(json).unwrap();
        let canon = v2.to_canonical();
        assert_eq!(canon.codecs.len(), 1);
        assert_eq!(canon.codecs[0].name, "gzip");
        assert_eq!(
            canon.codecs[0].configuration,
            vec![(String::from("level"), String::from("1"))]
        );
        assert_eq!(canon.chunk_key_encoding.name, "v2");
        assert_eq!(canon.chunk_key_encoding.separator, '.');
    }

    #[test]
    fn test_array_metadata_v2_to_canonical_preserves_dot_dimension_separator() {
        let json = r#"{
  "zarr_format": 2,
  "shape": [4, 6],
  "chunks": [2, 3],
  "dtype": "<i4",
  "fill_value": -1,
  "order": "C",
  "filters": null,
  "dimension_separator": "."
}"#;
        let v2 = ArrayMetadataV2::parse(json).unwrap();
        let canon = v2.to_canonical();
        assert_eq!(canon.chunk_key_encoding.name, "v2");
        assert_eq!(canon.chunk_key_encoding.separator, '.');
    }

    #[test]
    fn test_array_metadata_v2_to_canonical_preserves_slash_dimension_separator() {
        let json = r#"{
  "zarr_format": 2,
  "shape": [4, 6],
  "chunks": [2, 3],
  "dtype": "<i4",
  "fill_value": -1,
  "order": "C",
  "filters": null,
  "dimension_separator": "/"
}"#;
        let v2 = ArrayMetadataV2::parse(json).unwrap();
        let canon = v2.to_canonical();
        assert_eq!(canon.chunk_key_encoding.name, "default");
        assert_eq!(canon.chunk_key_encoding.separator, '/');
    }

    #[test]
    fn test_fill_value_special() {
        let json = r#"{
            "zarr_format": 2,
            "shape": [1],
            "chunks": [1],
            "dtype": "<f8",
            "fill_value": "NaN",
            "order": "C",
            "compressor": null,
            "filters": null
        }"#;
        let meta = ArrayMetadataV2::parse(json).expect("must parse");
        let is_nan = match meta.fill_value {
            FillValueJson::Special(SpecialFillValue::NaN) => true,
            FillValueJson::String(ref s) if s == "NaN" => true,
            _ => false,
        };
        assert!(
            is_nan,
            "fill value NaN must parse to a NaN-compatible canonical value"
        );
    }

    #[test]
    fn test_zattrs_roundtrip() {
        let attrs = vec![
            ("temperature".to_string(), AttributeValue::Float(298.15)),
            (
                "dimensions".to_string(),
                AttributeValue::IntArray(vec![3, 512, 512]),
            ),
            (
                "name".to_string(),
                AttributeValue::String("run_001".to_string()),
            ),
        ];
        let json = serialize_zattrs(&attrs).expect("must serialize");
        let parsed = parse_zattrs(&json).expect("must parse");
        assert_eq!(parsed.len(), 3);
        assert!(parsed.iter().any(|(k, _)| k == "temperature"));
        assert!(parsed.iter().any(|(k, _)| k == "dimensions"));
        assert!(parsed.iter().any(|(k, _)| k == "name"));
    }

    #[test]
    fn test_hdf5_filter_id_mapping() {
        assert_eq!(hdf5_filter_id_to_name(1), "deflate");
        assert_eq!(hdf5_filter_id_to_name(32004), "lz4");
        assert_eq!(hdf5_filter_id_to_name(32015), "zstd");
        assert_eq!(hdf5_filter_id_to_name(32001), "blosc");
        assert_eq!(hdf5_filter_id_to_name(999), "filter999");
    }

    #[test]
    fn test_dtype_v2_numpy() {
        // Regression: ensure dtype parsing is not broken by v2 module
        let json = r#"{
            "zarr_format": 2,
            "shape": [5],
            "chunks": [5],
            "dtype": "|S32",
            "fill_value": "",
            "order": "C",
            "compressor": null,
            "filters": null
        }"#;
        let meta = ArrayMetadataV2::parse(json).expect("must parse");
        assert_eq!(meta.dtype, "|S32");
    }
}
