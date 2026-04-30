//! Zarr consolidated metadata format (`.zmetadata`).
//!
//! ## Zarr v2 Consolidated Metadata
//!
//! Zarr v2 supports consolidating all metadata in a `.zmetadata` file at
//! the root of a hierarchy. This file contains a JSON object with a
//! `metadata` array listing all arrays and groups, along with their
//! metadata, enabling a single HTTP GET to retrieve the entire
//! directory structure.
//!
//! ## Zarr v3 Consolidated Metadata
//!
//! Zarr v3 uses a similar `.zmetadata` file, but the structure differs.
//! The v3 format includes a `metadata` field with the `zarr.json` content
//! of the root node and all leaf nodes (arrays and groups) referenced by
//! their relative paths.
//!
//! ## Python Interoperability
//!
//! Both formats are fully readable and writable by zarr-python via
//! `zarr.consolidate_metadata()` and `zarr.open_consolidated()`.
//! This module produces spec-compliant output to ensure cross-language
//! compatibility.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

#[cfg(feature = "alloc")]
use serde::{Deserialize, Serialize};

use super::{ArrayMetadata, AttributeValue, GroupMetadata, ZarrVersion};

// ---------------------------------------------------------------------------
// Zarr v2 consolidated metadata
// ---------------------------------------------------------------------------

/// Consolidated metadata for Zarr v2.
///
/// Parsed from the `.zmetadata` JSON file produced by
/// `zarr.consolidate_metadata()`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedMetadataV2 {
    /// Must be 2 for Zarr v2 consolidated format.
    pub zarr_format: u32,
    /// List of all nodes (arrays and groups) in the hierarchy.
    pub metadata: Vec<MetadataEntryV2>,
    /// zarr-python version that wrote this file.
    #[serde(default)]
    pub zarr_python_version: Option<String>,
    /// Consolidation timestamp (ISO 8601).
    #[serde(default, alias = "metadata写入时间")]
    pub metadata_write_time: Option<String>,
}

/// A single entry in the v2 consolidated metadata image.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEntryV2 {
    /// Path to the node (relative to the store root).
    pub path: String,
    /// Node type: `"array"` or `"group"`.
    #[serde(rename = "type")]
    pub node_type: String,
    /// The `.zarray` or `.zgroup` metadata as a JSON value.
    pub data: serde_json::Value,
    /// The `.zattrs` metadata as a JSON string (arrays only).
    #[serde(default)]
    pub attributes: Option<String>,
}

#[cfg(feature = "alloc")]
impl ConsolidatedMetadataV2 {
    /// Parse a `.zmetadata` v2 JSON document.
    #[cfg(feature = "std")]
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        serde_json::from_str(text).map_err(ParseError::Json)
    }

    /// Serialize to a `.zmetadata` v2 JSON document.
    #[cfg(feature = "std")]
    pub fn to_json(&self) -> Result<String, SerializeError> {
        serde_json::to_string_pretty(self).map_err(SerializeError::Json)
    }

    /// Build a `ConsolidatedMetadata` from this v2 representation.
    pub fn to_canonical(&self) -> ConsolidatedMetadata {
        let entries: Vec<ConsolidatedEntry> = self
            .metadata
            .iter()
            .map(|entry| {
                let node = if entry.node_type == "array" {
                    // Parse .zarray metadata from the JSON data
                    let array_meta: Option<super::v2::ArrayMetadataV2> =
                        serde_json::from_value(entry.data.clone()).ok();

                    let canonical =
                        array_meta
                            .map(|a| a.to_canonical())
                            .unwrap_or_else(|| ArrayMetadata {
                                version: ZarrVersion::V2,
                                shape: vec![],
                                chunks: vec![],
                                dtype: String::new(),
                                fill_value: super::FillValue::Default,
                                order: 'C',
                                codecs: vec![],
                                chunk_key_encoding: super::ChunkKeyEncoding::default(),
                                dimension_names: None,
                            });

                    NodeMetadata::Array(canonical)
                } else {
                    let group_meta: Option<super::v2::GroupMetadataV2> =
                        serde_json::from_value(entry.data.clone()).ok();

                    NodeMetadata::Group(
                        group_meta
                            .map(|g| g.to_canonical())
                            .unwrap_or_else(|| GroupMetadata::default()),
                    )
                };

                ConsolidatedEntry {
                    path: entry.path.clone(),
                    metadata: node,
                    attributes: entry.attributes.clone(),
                }
            })
            .collect();

        ConsolidatedMetadata {
            version: ZarrVersion::V2,
            zarr_python_version: self.zarr_python_version.clone(),
            entries,
        }
    }
}

// ---------------------------------------------------------------------------
// Zarr v3 consolidated metadata
// ---------------------------------------------------------------------------

/// Consolidated metadata for Zarr v3.
///
/// The v3 consolidated metadata format is structurally different from v2.
/// It stores each node's `zarr.json` content directly rather than wrapping
/// it in a `data` field.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedMetadataV3 {
    /// Must be 3 for Zarr v3 consolidated format.
    pub zarr_format: u32,
    /// List of all nodes in the hierarchy.
    pub metadata: Vec<MetadataEntryV3>,
    /// zarr-python version that wrote this file.
    #[serde(default)]
    pub zarr_python_version: Option<String>,
    /// Consolidation timestamp.
    #[serde(default, alias = "metadata写入时间")]
    pub metadata_write_time: Option<String>,
}

/// A single entry in the v3 consolidated metadata.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEntryV3 {
    /// Path to the node (relative to store root).
    pub path: String,
    /// The `zarr.json` content of the node as a JSON object.
    pub data: serde_json::Value,
}

#[cfg(feature = "alloc")]
impl ConsolidatedMetadataV3 {
    /// Parse a `.zmetadata` v3 JSON document.
    #[cfg(feature = "std")]
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        serde_json::from_str(text).map_err(ParseError::Json)
    }

    /// Serialize to a `.zmetadata` v3 JSON document.
    #[cfg(feature = "std")]
    pub fn to_json(&self) -> Result<String, SerializeError> {
        serde_json::to_string_pretty(self).map_err(SerializeError::Json)
    }

    /// Build a `ConsolidatedMetadata` from this v3 representation.
    pub fn to_canonical(&self) -> ConsolidatedMetadata {
        let entries: Vec<ConsolidatedEntry> = self
            .metadata
            .iter()
            .map(|entry| {
                let zarr_json: Option<super::v3::ZarrJson> =
                    serde_json::from_value(entry.data.clone()).ok();

                let node = zarr_json
                    .and_then(|zj| {
                        if let Some(arr) = zj.to_array_canonical() {
                            Some(NodeMetadata::Array(arr))
                        } else if let Some(grp) = zj.to_group_canonical() {
                            Some(NodeMetadata::Group(grp))
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| NodeMetadata::Group(GroupMetadata::default()));

                ConsolidatedEntry {
                    path: entry.path.clone(),
                    metadata: node,
                    attributes: None,
                }
            })
            .collect();

        ConsolidatedMetadata {
            version: ZarrVersion::V3,
            zarr_python_version: self.zarr_python_version.clone(),
            entries,
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical consolidated metadata (format-agnostic)
// ---------------------------------------------------------------------------

/// Canonical consolidated metadata representation.
///
/// This is the format-independent representation of consolidated metadata
/// used internally by consus-zarr.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ConsolidatedMetadata {
    /// Zarr format version of the consolidated file.
    pub version: ZarrVersion,
    /// zarr-python version that produced the consolidated file.
    pub zarr_python_version: Option<String>,
    /// All node entries in the hierarchy.
    pub entries: Vec<ConsolidatedEntry>,
}

#[cfg(feature = "alloc")]
impl ConsolidatedMetadata {
    /// Look up an entry by path.
    pub fn get(&self, path: &str) -> Option<&ConsolidatedEntry> {
        self.entries.iter().find(|e| e.path == path)
    }

    /// List all array paths in this consolidated metadata.
    pub fn array_paths(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|e| matches!(e.metadata, NodeMetadata::Array(_)))
            .map(|e| e.path.as_str())
            .collect()
    }

    /// List all group paths in this consolidated metadata.
    pub fn group_paths(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|e| matches!(e.metadata, NodeMetadata::Group(_)))
            .map(|e| e.path.as_str())
            .collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A single node entry in consolidated metadata.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ConsolidatedEntry {
    /// Path to the node (relative to store root, without leading slash).
    pub path: String,
    /// Node metadata (array or group).
    pub metadata: NodeMetadata,
    /// Raw `.zattrs` JSON string (only for v2 array entries).
    pub attributes: Option<String>,
}

#[cfg(feature = "alloc")]
impl ConsolidatedEntry {
    /// Get the attributes for this entry as `(name, value)` pairs.
    pub fn attributes(&self) -> Vec<(String, AttributeValue)> {
        if let Some(attrs_json) = &self.attributes {
            super::v2::parse_zattrs(attrs_json).unwrap_or_default()
        } else if let NodeMetadata::Group(g) = &self.metadata {
            g.attributes.clone()
        } else {
            vec![]
        }
    }

    /// The node type as a string.
    pub fn node_type_str(&self) -> &'static str {
        match &self.metadata {
            NodeMetadata::Array(_) => "array",
            NodeMetadata::Group(_) => "group",
        }
    }
}

/// Node metadata variant.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub enum NodeMetadata {
    Array(ArrayMetadata),
    Group(GroupMetadata),
}

#[cfg(feature = "alloc")]
impl NodeMetadata {
    /// The Zarr format version of this node.
    pub fn version(&self) -> ZarrVersion {
        match self {
            Self::Array(a) => a.version,
            Self::Group(g) => g.version,
        }
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur when parsing consolidated metadata.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub enum ParseError {
    Json(serde_json::Error),
}

#[cfg(feature = "std")]
impl core::fmt::Display for ParseError {
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

/// Errors that can occur when serializing consolidated metadata.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub enum SerializeError {
    Json(serde_json::Error),
}

#[cfg(feature = "std")]
impl core::fmt::Display for SerializeError {
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
    fn test_consolidated_v2_parse() {
        let json = r#"{
            "zarr_format": 2,
            "metadata": [
                {
                    "path": ".",
                    "type": "group",
                    "data": {"zarr_format": 2}
                },
                {
                    "path": "array",
                    "type": "array",
                    "data": {
                        "zarr_format": 2,
                        "shape": [10, 10],
                        "chunks": [5, 5],
                        "dtype": "<f8",
                        "fill_value": 0.0,
                        "order": "C",
                        "compressor": null,
                        "filters": null
                    },
                    "attributes": "{\"units\": \"K\"}"
                }
            ],
            "zarr_python_version": "2.14.0"
        }"#;
        let meta = ConsolidatedMetadataV2::parse(json).expect("must parse");
        assert_eq!(meta.zarr_format, 2);
        assert_eq!(meta.metadata.len(), 2);
        assert_eq!(meta.metadata[0].path, ".");
        assert_eq!(meta.metadata[1].path, "array");
    }

    #[test]
    fn test_consolidated_v3_parse() {
        let json = r#"{
            "zarr_format": 3,
            "metadata": [
                {
                    "path": ".",
                    "data": {
                        "zarr_format": 3,
                        "node_type": "group",
                        "codecs": []
                    }
                },
                {
                    "path": "data/array",
                    "data": {
                        "zarr_format": 3,
                        "node_type": "array",
                        "shape": [100],
                        "data_type": "float32",
                        "chunk_grid": {
                            "name": "regular",
                            "configuration": {"chunk_shape": [10]}
                        },
                        "chunk_key_encoding": {
                            "name": "default",
                            "configuration": {"separator": "/"}
                        },
                        "codecs": [],
                        "fill_value": 0.0,
                        "order": "C"
                    }
                }
            ]
        }"#;
        let meta = ConsolidatedMetadataV3::parse(json).expect("must parse");
        assert_eq!(meta.zarr_format, 3);
        assert_eq!(meta.metadata.len(), 2);
    }

    #[test]
    fn test_canonical_entry_attributes_v2() {
        let json = r#"{
            "zarr_format": 2,
            "metadata": [
                {
                    "path": ".",
                    "type": "array",
                    "data": {
                        "zarr_format": 2,
                        "shape": [5],
                        "chunks": [5],
                        "dtype": "<f4",
                        "fill_value": 0.0,
                        "order": "C",
                        "compressor": null,
                        "filters": null
                    },
                    "attributes": "{\"temperature\": 298.15}"
                }
            ]
        }"#;
        let v2 = ConsolidatedMetadataV2::parse(json).expect("must parse");
        let canon = v2.to_canonical();
        let entry = canon.get(".").expect("entry must exist");
        let attrs = entry.attributes();
        assert_eq!(attrs.len(), 1);
        assert_eq!(attrs[0].0, "temperature");
    }

    #[test]
    fn test_consolidated_entry_lookup() {
        let json = r#"{
            "zarr_format": 2,
            "metadata": [
                {"path": "a", "type": "group", "data": {"zarr_format": 2}},
                {"path": "b", "type": "group", "data": {"zarr_format": 2}}
            ]
        }"#;
        let v2 = ConsolidatedMetadataV2::parse(json).expect("must parse");
        let canon = v2.to_canonical();
        assert!(canon.get("a").is_some());
        assert!(canon.get("b").is_some());
        assert!(canon.get("c").is_none());
    }

    #[test]
    fn test_array_group_paths() {
        let json = r#"{
            "zarr_format": 3,
            "metadata": [
                {
                    "path": ".",
                    "data": {"zarr_format": 3, "node_type": "group", "codecs": []}
                },
                {
                    "path": "x",
                    "data": {
                        "zarr_format": 3,
                        "node_type": "array",
                        "shape": [10],
                        "data_type": "int32",
                        "chunk_grid": {
                            "name": "regular",
                            "configuration": {"chunk_shape": [10]}
                        },
                        "chunk_key_encoding": {
                            "name": "default",
                            "configuration": {"separator": "/"}
                        },
                        "codecs": [],
                        "fill_value": 0,
                        "order": "C"
                    }
                },
                {
                    "path": "y",
                    "data": {
                        "zarr_format": 3,
                        "node_type": "group",
                        "codecs": []
                    }
                }
            ]
        }"#;
        let v3 = ConsolidatedMetadataV3::parse(json).expect("must parse");
        let canon = v3.to_canonical();
        assert_eq!(canon.array_paths(), &["x"]);
        assert_eq!(canon.group_paths(), &[".", "y"]);
    }

    #[test]
    fn test_v3_group_to_canonical() {
        let json = r#"{
            "zarr_format": 3,
            "metadata": [
                {
                    "path": ".",
                    "data": {
                        "zarr_format": 3,
                        "node_type": "group",
                        "codecs": [],
                        "attributes": {"author": "test"}
                    }
                }
            ]
        }"#;
        let v3 = ConsolidatedMetadataV3::parse(json).expect("must parse");
        let canon = v3.to_canonical();
        let entry = canon.get(".").expect("entry must exist");
        if let NodeMetadata::Group(g) = &entry.metadata {
            assert_eq!(g.version, ZarrVersion::V3);
        } else {
            panic!("expected group");
        }
    }

    #[test]
    fn test_empty_consolidated_metadata() {
        let json = r#"{
            "zarr_format": 3,
            "metadata": []
        }"#;
        let v3 = ConsolidatedMetadataV3::parse(json).expect("must parse");
        let canon = v3.to_canonical();
        assert!(canon.is_empty());
        assert!(canon.array_paths().is_empty());
        assert!(canon.group_paths().is_empty());
    }

    #[test]
    fn test_consolidated_v2_with_attributes() {
        let json = r#"{
            "zarr_format": 2,
            "metadata": [
                {
                    "path": "arr",
                    "type": "array",
                    "data": {
                        "zarr_format": 2,
                        "shape": [5],
                        "chunks": [5],
                        "dtype": "<f4",
                        "fill_value": 0.0,
                        "order": "C",
                        "compressor": null,
                        "filters": null
                    },
                    "attributes": "{\"units\": \"K\", \"scale\": 1.0}"
                }
            ]
        }"#;
        let v2 = ConsolidatedMetadataV2::parse(json).expect("must parse");
        let canon = v2.to_canonical();
        let entry = canon.get("arr").expect("entry must exist");
        let attrs = entry.attributes();
        assert_eq!(attrs.len(), 2);
        assert!(attrs.iter().any(|(k, _)| k == "units"));
        assert!(attrs.iter().any(|(k, _)| k == "scale"));
    }
}
