#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Error returned by conservative NWB namespace YAML parsing.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NwbNamespaceYamlError {
    /// A required field was not present.
    MissingField(&'static str),
    /// A field value was malformed or nested content was encountered.
    InvalidField(&'static str),
    /// A field appeared more than once.
    DuplicateField(&'static str),
}

// ---------------------------------------------------------------------------
// NwbNamespace
// ---------------------------------------------------------------------------

/// NWB specification namespace descriptor.
///
/// Holds the identifying name, specification version string, and documentation
/// URL for a single NWB namespace. The two canonical namespaces in every
/// NWB 2.x file are `"core"` and `"hdmf-common"`; both are provided as
/// compile-time constructors.
///
/// ## Invariants
///
/// - `name` and `version` are non-empty for all constructors defined here.
/// - `doc_url` is a `'static` reference to a compile-time string literal;
///   no heap allocation is required to access it.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NwbNamespace {
    /// Namespace identifier (e.g. `"core"`, `"hdmf-common"`).
    pub name: String,
    /// Specification version string (e.g. `"2.8.0"`).
    pub version: String,
    /// Documentation URL for this namespace.
    pub doc_url: &'static str,
}

#[cfg(feature = "alloc")]
impl NwbNamespace {
    /// The identifier of the NWB core namespace.
    pub const CORE_NAME: &'static str = "core";

    /// Parse a conservative NWB namespace descriptor from YAML text.
    ///
    /// The parser accepts only top-level `key: value` entries for
    /// `name`, `version`, and `doc_url`. It ignores comments and blank
    /// lines, rejects duplicate keys, and rejects nested structures.
    ///
    /// Failure mode is explicit:
    /// - missing required field → `MissingField`
    /// - duplicate field → `DuplicateField`
    /// - malformed or nested content → `InvalidField`
    pub fn parse_yaml(text: &str) -> Result<Self, NwbNamespaceYamlError> {
        let mut name: Option<String> = None;
        let mut version: Option<String> = None;
        let mut doc_url: Option<&'static str> = None;

        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with('-') || !line.contains(':') {
                return Err(NwbNamespaceYamlError::InvalidField("mapping"));
            }

            let (key, value) = match line.split_once(':') {
                Some((k, v)) => (k.trim(), v.trim()),
                None => return Err(NwbNamespaceYamlError::InvalidField("mapping")),
            };

            if key.is_empty() || value.is_empty() {
                return Err(NwbNamespaceYamlError::InvalidField("scalar value"));
            }

            let parsed_value = if value.starts_with('"') {
                if value.len() < 2 || !value.ends_with('"') {
                    return Err(NwbNamespaceYamlError::InvalidField("doc_url"));
                }
                let inner = &value[1..value.len() - 1];
                if inner.contains('\n') {
                    return Err(NwbNamespaceYamlError::InvalidField("doc_url"));
                }
                inner
            } else {
                value
            };

            match key {
                "name" => {
                    if name.is_some() {
                        return Err(NwbNamespaceYamlError::DuplicateField("name"));
                    }
                    name = Some(String::from(parsed_value));
                }
                "version" => {
                    if version.is_some() {
                        return Err(NwbNamespaceYamlError::DuplicateField("version"));
                    }
                    version = Some(String::from(parsed_value));
                }
                "doc_url" => {
                    if doc_url.is_some() {
                        return Err(NwbNamespaceYamlError::DuplicateField("doc_url"));
                    }
                    doc_url = Some(match parsed_value {
                        "https://nwb-schema.readthedocs.io/en/latest/" => {
                            "https://nwb-schema.readthedocs.io/en/latest/"
                        }
                        "https://hdmf-common-schema.readthedocs.io/" => {
                            "https://hdmf-common-schema.readthedocs.io/"
                        }
                        _ => return Err(NwbNamespaceYamlError::InvalidField("doc_url")),
                    });
                }
                _ => {}
            }
        }

        let name = name.ok_or(NwbNamespaceYamlError::MissingField("name"))?;
        let version = version.ok_or(NwbNamespaceYamlError::MissingField("version"))?;
        let doc_url = doc_url.ok_or(NwbNamespaceYamlError::MissingField("doc_url"))?;

        Ok(Self {
            name,
            version,
            doc_url,
        })
    }

    /// Return the descriptor for the NWB 2.x core namespace.
    ///
    /// Constructs the descriptor from compile-time constant data; no I/O
    /// is performed. The version string `"2.8.0"` matches the current stable
    /// NWB specification release at the time this crate was authored.
    ///
    /// Reference: <https://nwb-schema.readthedocs.io/en/latest/>
    pub fn core() -> Self {
        Self {
            name: String::from("core"),
            version: String::from("2.8.0"),
            doc_url: "https://nwb-schema.readthedocs.io/en/latest/",
        }
    }

    /// Return the descriptor for the HDMF common namespace.
    ///
    /// HDMF-common defines shared data types (tables, vector data, etc.)
    /// that are used by NWB core and third-party extensions.
    ///
    /// Reference: <https://hdmf-common-schema.readthedocs.io/>
    pub fn hdmf_common() -> Self {
        Self {
            name: String::from("hdmf-common"),
            version: String::from("1.8.0"),
            doc_url: "https://hdmf-common-schema.readthedocs.io/",
        }
    }
}

// ---------------------------------------------------------------------------
// NwbTypeSpec
// ---------------------------------------------------------------------------

/// Per-type entry in a NWB namespace specification.
///
/// Records the declared type name and, when present, the single direct
/// parent type from which it inherits (`neurodata_type_inc`).
///
/// ## Invariants
///
/// - `name` is non-empty.
/// - `neurodata_type_inc` is `None` when the type extends no other type.
/// - The inheritance chain is acyclic (enforced by the NWB specification;
///   the Consus model trusts spec-provided data on this invariant).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NwbTypeSpec {
    /// Declared neurodata type name.
    pub name: String,
    /// Direct parent type (`neurodata_type_inc`), if any.
    pub neurodata_type_inc: Option<String>,
}

// ---------------------------------------------------------------------------
// NwbNamespaceSpec
// ---------------------------------------------------------------------------

/// NWB namespace specification parsed from an HDF5 `/specifications/` group.
///
/// Holds the type-system metadata for one namespace at one version, as stored
/// under `/specifications/{name}/{version}/namespace` in NWB 2.x HDF5 files.
///
/// ## Invariants
///
/// - `name` and `version` are non-empty for specs produced by `parse_nwb_spec_yaml`.
/// - `neurodata_types` records per-type entries with optional `neurodata_type_inc`
///   inheritance chains, enabling arbitrary-depth resolution.
/// - `doc_url` is `None` when the YAML source did not carry a `doc_url` key.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NwbNamespaceSpec {
    /// Namespace identifier (e.g. `"core"`, `"hdmf-common"`).
    pub name: String,
    /// Specification version string (e.g. `"2.8.0"`).
    pub version: String,
    /// Documentation URL, if present in the YAML source.
    pub doc_url: Option<String>,
    /// Per-type entries declared in this namespace, each carrying an optional
    /// `neurodata_type_inc` parent for arbitrary-depth inheritance resolution.
    pub neurodata_types: Vec<NwbTypeSpec>,
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    // ── NwbNamespace::CORE_NAME ───────────────────────────────────────────

    #[test]
    fn core_name_constant_is_core() {
        assert_eq!(NwbNamespace::CORE_NAME, "core");
    }

    #[test]
    fn parse_yaml_extracts_core_metadata() {
        let text = r#"
name: core
version: 2.8.0
doc_url: https://nwb-schema.readthedocs.io/en/latest/
"#;
        let ns = NwbNamespace::parse_yaml(text).expect("valid core namespace YAML");
        assert_eq!(ns.name, "core");
        assert_eq!(ns.version, "2.8.0");
        assert_eq!(ns.doc_url, "https://nwb-schema.readthedocs.io/en/latest/");
    }

    #[test]
    fn parse_yaml_extracts_hdmf_common_metadata() {
        let text = r#"
name: hdmf-common
version: 1.8.0
doc_url: https://hdmf-common-schema.readthedocs.io/
"#;
        let ns = NwbNamespace::parse_yaml(text).expect("valid hdmf-common namespace YAML");
        assert_eq!(ns.name, "hdmf-common");
        assert_eq!(ns.version, "1.8.0");
        assert_eq!(ns.doc_url, "https://hdmf-common-schema.readthedocs.io/");
    }

    #[test]
    fn parse_yaml_rejects_missing_fields() {
        let text = r#"
name: core
version: 2.8.0
"#;
        let err = NwbNamespace::parse_yaml(text).expect_err("missing doc_url must fail");
        assert_eq!(err, NwbNamespaceYamlError::MissingField("doc_url"));
    }

    #[test]
    fn parse_yaml_rejects_malformed_content() {
        let text = r#"
name: core
version
doc_url: https://nwb-schema.readthedocs.io/en/latest/
"#;
        let err = NwbNamespace::parse_yaml(text).expect_err("malformed YAML must fail");
        assert_eq!(err, NwbNamespaceYamlError::InvalidField("mapping"));
    }

    // ── NwbNamespace::core() ──────────────────────────────────────────────

    #[test]
    fn core_name_field_equals_core() {
        let ns = NwbNamespace::core();
        assert_eq!(ns.name, NwbNamespace::CORE_NAME);
    }

    #[test]
    fn core_version_is_2_8_0() {
        let ns = NwbNamespace::core();
        assert_eq!(ns.version, "2.8.0");
    }

    #[test]
    fn core_doc_url_is_nwb_schema_readthedocs() {
        let ns = NwbNamespace::core();
        assert_eq!(ns.doc_url, "https://nwb-schema.readthedocs.io/en/latest/");
    }

    // ── NwbNamespace::hdmf_common() ───────────────────────────────────────

    #[test]
    fn hdmf_common_name_is_hdmf_common() {
        let ns = NwbNamespace::hdmf_common();
        assert_eq!(ns.name, "hdmf-common");
    }

    #[test]
    fn hdmf_common_version_is_1_8_0() {
        let ns = NwbNamespace::hdmf_common();
        assert_eq!(ns.version, "1.8.0");
    }

    #[test]
    fn hdmf_common_doc_url_is_hdmf_common_readthedocs() {
        let ns = NwbNamespace::hdmf_common();
        assert_eq!(ns.doc_url, "https://hdmf-common-schema.readthedocs.io/");
    }

    // ── Clone / PartialEq ────────────────────────────────────────────────

    #[test]
    fn clone_core_equals_original() {
        let original = NwbNamespace::core();
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn clone_hdmf_common_equals_original() {
        let original = NwbNamespace::hdmf_common();
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn core_and_hdmf_common_are_not_equal() {
        assert_ne!(NwbNamespace::core(), NwbNamespace::hdmf_common());
    }

    #[test]
    fn two_core_instances_are_equal() {
        assert_eq!(NwbNamespace::core(), NwbNamespace::core());
    }

    #[test]
    fn two_hdmf_common_instances_are_equal() {
        assert_eq!(NwbNamespace::hdmf_common(), NwbNamespace::hdmf_common());
    }

    // ── Field mutation independence ───────────────────────────────────────

    #[test]
    fn mutating_clone_does_not_affect_original() {
        let original = NwbNamespace::core();
        let mut mutated = original.clone();
        mutated.version = String::from("99.0.0");

        assert_eq!(original.version, "2.8.0");
        assert_eq!(mutated.version, "99.0.0");
    }

    // ── CORE_NAME consistency with core() ────────────────────────────────

    #[test]
    fn core_instance_name_matches_core_name_constant() {
        let ns = NwbNamespace::core();
        assert_eq!(ns.name.as_str(), NwbNamespace::CORE_NAME);
    }

    // ── NwbNamespaceSpec construction ─────────────────────────────────────

    #[test]
    fn namespace_spec_new_stores_all_fields() {
        let spec = NwbNamespaceSpec {
            name: String::from("core"),
            version: String::from("2.8.0"),
            doc_url: Some(String::from("https://nwb-schema.readthedocs.io/en/latest/")),
            neurodata_types: vec![
                NwbTypeSpec {
                    name: String::from("TimeSeries"),
                    neurodata_type_inc: None,
                },
                NwbTypeSpec {
                    name: String::from("ElectricalSeries"),
                    neurodata_type_inc: None,
                },
            ],
        };
        assert_eq!(spec.name, "core");
        assert_eq!(spec.version, "2.8.0");
        assert_eq!(
            spec.doc_url.as_deref(),
            Some("https://nwb-schema.readthedocs.io/en/latest/")
        );
        assert_eq!(spec.neurodata_types[0].name, "TimeSeries");
        assert_eq!(spec.neurodata_types[1].name, "ElectricalSeries");
        assert_eq!(spec.neurodata_types.len(), 2);
    }

    #[test]
    fn namespace_spec_without_doc_url() {
        let spec = NwbNamespaceSpec {
            name: String::from("hdmf-common"),
            version: String::from("1.8.0"),
            doc_url: None,
            neurodata_types: vec![NwbTypeSpec {
                name: String::from("VectorData"),
                neurodata_type_inc: None,
            }],
        };
        assert!(spec.doc_url.is_none());
        assert_eq!(spec.neurodata_types.len(), 1);
    }

    #[test]
    fn namespace_spec_clone_equals_original() {
        let spec = NwbNamespaceSpec {
            name: String::from("core"),
            version: String::from("2.8.0"),
            doc_url: None,
            neurodata_types: vec![NwbTypeSpec {
                name: String::from("TimeSeries"),
                neurodata_type_inc: None,
            }],
        };
        let cloned = spec.clone();
        assert_eq!(spec, cloned);
    }

    // ── NwbTypeSpec construction ──────────────────────────────────────────

    #[test]
    fn nwb_type_spec_with_inc_stores_parent() {
        let ts = NwbTypeSpec {
            name: String::from("TimeSeries"),
            neurodata_type_inc: Some(String::from("NWBDataInterface")),
        };
        assert_eq!(ts.name, "TimeSeries");
        assert_eq!(ts.neurodata_type_inc.as_deref(), Some("NWBDataInterface"));
    }

    #[test]
    fn nwb_type_spec_without_inc_has_none_parent() {
        let ts = NwbTypeSpec {
            name: String::from("X"),
            neurodata_type_inc: None,
        };
        assert_eq!(ts.name, "X");
        assert!(ts.neurodata_type_inc.is_none());
    }
}
