//! NWB namespace registry and type system.
//!
//! Manages NWB specification namespaces. In NWB 2.x files, namespace
//! definitions are stored as YAML-serialised spec files under
//! `/specifications/{namespace_name}/{version}/`.
//!
//! This module provides:
//! - the registry model and hard-coded core namespace entry
//! - a conservative YAML text parser for extracting NWB namespace metadata
//!
//! ## Invariants
//!
//! - `NwbNamespace::CORE_NAME` is always `"core"`.
//! - `NwbNamespace::core()` returns the canonical NWB 2.x core namespace
//!   descriptor without I/O.
//! - `parse_namespace_yaml_text()` only accepts explicit scalar fields for
//!   `name`, `version`, and `doc_url`; unknown or malformed inputs fail
//!   deterministically.

#[cfg(feature = "alloc")]
use alloc::string::String;

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

/// Parse a NWB namespace specification YAML document into namespace specs.
///
/// The expected format is the compact flat-key representation produced by
/// [`format_nwb_spec_yaml`] and compatible with the NWB reference implementation:
///
/// ```text
/// namespaces:
/// - name: core
///   version: 2.8.0
///   doc_url: https://nwb-schema.readthedocs.io/en/latest/
///   neurodata_types:
///   - NWBDataInterface
///   - name: TimeSeries
///     inc: NWBDataInterface
///   - name: ElectricalSeries
///     inc: TimeSeries
/// ```
///
/// Multiple namespace objects may appear as separate `- ` list items.
/// Unknown YAML keys are silently ignored for forward compatibility.
///
/// ## Parser invariants
///
/// - A list item starting with `- ` at indent 0 opens a new namespace entry.
/// - Keys indented by exactly 2 spaces are namespace-level (`name`, `version`,
///   `doc_url`, `neurodata_types:`).
/// - After `neurodata_types:`, `  - TypeName` (bare) or `  - name: TypeName` at
///   indent 2 starts a type entry; `    inc: ParentName` at indent 4 sets the
///   parent. Bare entries are backward-compatible with the previous format.
/// - A non-list-item indent-2 key terminates the neurodata type sub-list.
///
/// ## Errors
///
/// - `MissingField("namespaces")` — the `namespaces:` root key is absent.
/// - `MissingField("name")` — a namespace entry lacks a `name` key.
/// - `MissingField("version")` — a namespace entry lacks a `version` key.
#[cfg(feature = "alloc")]
pub fn parse_nwb_spec_yaml(text: &str) -> Result<Vec<NwbNamespaceSpec>, NwbNamespaceYamlError> {
    let mut in_namespaces = false;
    let mut in_neuro_types = false;
    let mut pending_type: Option<NwbTypeSpec> = None;
    let mut current: Option<NwbNamespaceSpec> = None;
    let mut specs: Vec<NwbNamespaceSpec> = Vec::new();

    for raw_line in text.lines() {
        let indent = raw_line.len() - raw_line.trim_start_matches(' ').len();
        let trimmed = raw_line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if !in_namespaces {
            if trimmed == "namespaces:" {
                in_namespaces = true;
            }
            continue;
        }

        // New namespace list item: `- ` at indent 0.
        if indent == 0 && trimmed.starts_with("- ") {
            // Flush any pending type entry before switching namespace.
            if let Some(pt) = pending_type.take() {
                if let Some(ref mut ns) = current {
                    ns.neurodata_types.push(pt);
                }
            }
            if let Some(ns) = current.take() {
                specs.push(ns);
            }
            in_neuro_types = false;
            let mut ns = NwbNamespaceSpec {
                name: String::new(),
                version: String::new(),
                doc_url: None,
                neurodata_types: Vec::new(),
            };
            // Optional inline key: `- name: core`
            let rest = trimmed[2..].trim();
            if !rest.is_empty() {
                if let Some((key, value)) = rest.split_once(':') {
                    let k = key.trim();
                    let v = value.trim();
                    if !k.is_empty() && !v.is_empty() {
                        apply_spec_key(&mut ns, k, v);
                    }
                }
            }
            current = Some(ns);
            continue;
        }

        // Sub-keys at indent 4 within a named type entry in neurodata_types.
        if indent == 4 && in_neuro_types {
            if let Some((key, value)) = trimmed.split_once(':') {
                let k = key.trim();
                let v = value.trim();
                if k == "inc" && !v.is_empty() {
                    if let Some(ref mut pt) = pending_type {
                        pt.neurodata_type_inc = Some(String::from(v));
                    }
                }
            }
            continue;
        }

        // Keys within a namespace item at indent 2.
        if indent == 2 {
            if trimmed == "neurodata_types:" {
                in_neuro_types = true;
                continue;
            }
            if in_neuro_types && trimmed.starts_with("- ") {
                // Flush any pending type entry before starting a new one.
                if let Some(pt) = pending_type.take() {
                    if let Some(ref mut ns) = current {
                        ns.neurodata_types.push(pt);
                    }
                }
                let rest = trimmed[2..].trim();
                if !rest.is_empty() {
                    if rest.contains(':') {
                        if let Some((key, value)) = rest.split_once(':') {
                            let k = key.trim();
                            let v = value.trim();
                            if k == "name" && !v.is_empty() {
                                // `- name: TypeName` — start a pending named entry.
                                pending_type = Some(NwbTypeSpec {
                                    name: String::from(v),
                                    neurodata_type_inc: None,
                                });
                            }
                            // Unknown key format: silently ignored.
                        }
                    } else {
                        // Bare type name (no colon): backward-compatible format.
                        if let Some(ref mut ns) = current {
                            ns.neurodata_types.push(NwbTypeSpec {
                                name: String::from(rest),
                                neurodata_type_inc: None,
                            });
                        }
                    }
                }
                continue;
            }
            // Non-list-item key at indent 2 terminates the neurodata_types sub-list.
            // Flush any pending type entry first.
            if let Some(pt) = pending_type.take() {
                if let Some(ref mut ns) = current {
                    ns.neurodata_types.push(pt);
                }
            }
            in_neuro_types = false;
            if let Some((key, value)) = trimmed.split_once(':') {
                let k = key.trim();
                let v = value.trim();
                if !k.is_empty() {
                    if let Some(ref mut ns) = current {
                        apply_spec_key(ns, k, v);
                    }
                }
            }
        }
    }

    // Finalize: flush pending type then finalize last namespace.
    if let Some(pt) = pending_type.take() {
        if let Some(ref mut ns) = current {
            ns.neurodata_types.push(pt);
        }
    }
    if let Some(ns) = current.take() {
        specs.push(ns);
    }

    if !in_namespaces {
        return Err(NwbNamespaceYamlError::MissingField("namespaces"));
    }

    // Validate that each spec has name and version.
    for spec in &specs {
        if spec.name.is_empty() {
            return Err(NwbNamespaceYamlError::MissingField("name"));
        }
        if spec.version.is_empty() {
            return Err(NwbNamespaceYamlError::MissingField("version"));
        }
    }

    Ok(specs)
}

#[cfg(feature = "alloc")]
fn apply_spec_key(ns: &mut NwbNamespaceSpec, key: &str, value: &str) {
    match key {
        "name" => ns.name = String::from(value),
        "version" => ns.version = String::from(value),
        "doc_url" => {
            if !value.is_empty() {
                ns.doc_url = Some(String::from(value));
            }
        }
        _ => {}
    }
}

/// Serialize a slice of namespace specifications to the canonical NWB YAML format.
///
/// Produces the flat-key YAML representation consumed by [`parse_nwb_spec_yaml`].
/// The output is stable and deterministic: keys appear in the order
/// `name`, `version`, `doc_url` (if present), `neurodata_types` (if non-empty).
///
/// ## Format
///
/// ```text
/// namespaces:
/// - name: core
///   version: 2.8.0
///   doc_url: https://nwb-schema.readthedocs.io/en/latest/
///   neurodata_types:
///   - TimeSeries
///   - ElectricalSeries
/// ```
#[cfg(feature = "alloc")]
pub fn format_nwb_spec_yaml(specs: &[NwbNamespaceSpec]) -> String {
    let mut out = String::from("namespaces:\n");
    for spec in specs {
        out.push_str("- name: ");
        out.push_str(&spec.name);
        out.push('\n');
        out.push_str("  version: ");
        out.push_str(&spec.version);
        out.push('\n');
        if let Some(ref url) = spec.doc_url {
            out.push_str("  doc_url: ");
            out.push_str(url);
            out.push('\n');
        }
        if !spec.neurodata_types.is_empty() {
            out.push_str("  neurodata_types:\n");
            for t in &spec.neurodata_types {
                if t.neurodata_type_inc.is_none() {
                    out.push_str("  - ");
                    out.push_str(&t.name);
                    out.push('\n');
                } else {
                    out.push_str("  - name: ");
                    out.push_str(&t.name);
                    out.push('\n');
                    out.push_str("    inc: ");
                    out.push_str(t.neurodata_type_inc.as_deref().unwrap());
                    out.push('\n');
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

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

    // ── format_nwb_spec_yaml ──────────────────────────────────────────────

    #[test]
    fn format_nwb_spec_yaml_empty_slice_produces_namespaces_header() {
        let yaml = format_nwb_spec_yaml(&[]);
        assert_eq!(yaml, "namespaces:\n");
    }

    #[test]
    fn format_nwb_spec_yaml_single_spec_no_doc_url_no_types() {
        let spec = NwbNamespaceSpec {
            name: String::from("core"),
            version: String::from("2.8.0"),
            doc_url: None,
            neurodata_types: vec![],
        };
        let yaml = format_nwb_spec_yaml(&[spec]);
        assert!(
            yaml.contains("namespaces:"),
            "must have namespaces key: {yaml}"
        );
        assert!(yaml.contains("- name: core"), "must have name: {yaml}");
        assert!(
            yaml.contains("  version: 2.8.0"),
            "must have version: {yaml}"
        );
        assert!(
            !yaml.contains("doc_url"),
            "must omit absent doc_url: {yaml}"
        );
        assert!(
            !yaml.contains("neurodata_types"),
            "must omit empty type list: {yaml}"
        );
    }

    #[test]
    fn format_nwb_spec_yaml_includes_doc_url_when_present() {
        let spec = NwbNamespaceSpec {
            name: String::from("core"),
            version: String::from("2.8.0"),
            doc_url: Some(String::from("https://nwb-schema.readthedocs.io/en/latest/")),
            neurodata_types: vec![],
        };
        let yaml = format_nwb_spec_yaml(&[spec]);
        assert!(
            yaml.contains("  doc_url: https://nwb-schema.readthedocs.io/en/latest/"),
            "must include doc_url: {yaml}"
        );
    }

    #[test]
    fn format_nwb_spec_yaml_includes_neurodata_types_when_present() {
        let spec = NwbNamespaceSpec {
            name: String::from("core"),
            version: String::from("2.8.0"),
            doc_url: None,
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
        let yaml = format_nwb_spec_yaml(&[spec]);
        assert!(
            yaml.contains("  neurodata_types:"),
            "must have neurodata_types key: {yaml}"
        );
        assert!(
            yaml.contains("  - TimeSeries"),
            "must have TimeSeries type: {yaml}"
        );
        assert!(
            yaml.contains("  - ElectricalSeries"),
            "must have ElectricalSeries type: {yaml}"
        );
    }

    // ── parse_nwb_spec_yaml ───────────────────────────────────────────────

    #[test]
    fn parse_nwb_spec_yaml_rejects_missing_namespaces_key() {
        let text = "name: core\nversion: 2.8.0\n";
        let err = parse_nwb_spec_yaml(text).expect_err("must fail without namespaces key");
        assert_eq!(err, NwbNamespaceYamlError::MissingField("namespaces"));
    }

    #[test]
    fn parse_nwb_spec_yaml_empty_namespaces_list_returns_empty_vec() {
        let text = "namespaces:\n";
        let specs = parse_nwb_spec_yaml(text).expect("empty list is valid");
        assert!(specs.is_empty());
    }

    #[test]
    fn parse_nwb_spec_yaml_single_spec_minimal() {
        let text = "namespaces:\n- name: core\n  version: 2.8.0\n";
        let specs = parse_nwb_spec_yaml(text).expect("valid single spec");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "core");
        assert_eq!(specs[0].version, "2.8.0");
        assert!(specs[0].doc_url.is_none());
        assert!(specs[0].neurodata_types.is_empty());
    }

    #[test]
    fn parse_nwb_spec_yaml_single_spec_with_doc_url() {
        let text =
            "namespaces:\n- name: core\n  version: 2.8.0\n  doc_url: https://nwb-schema.readthedocs.io/en/latest/\n";
        let specs = parse_nwb_spec_yaml(text).expect("valid spec with doc_url");
        assert_eq!(specs.len(), 1);
        assert_eq!(
            specs[0].doc_url.as_deref(),
            Some("https://nwb-schema.readthedocs.io/en/latest/")
        );
    }

    #[test]
    fn parse_nwb_spec_yaml_extracts_neurodata_types() {
        let text = "namespaces:\n- name: core\n  version: 2.8.0\n  neurodata_types:\n  - TimeSeries\n  - ElectricalSeries\n";
        let specs = parse_nwb_spec_yaml(text).expect("valid spec with types");
        assert_eq!(specs[0].neurodata_types.len(), 2);
        assert_eq!(specs[0].neurodata_types[0].name, "TimeSeries");
        assert!(specs[0].neurodata_types[0].neurodata_type_inc.is_none());
        assert_eq!(specs[0].neurodata_types[1].name, "ElectricalSeries");
        assert!(specs[0].neurodata_types[1].neurodata_type_inc.is_none());
    }

    #[test]
    fn parse_nwb_spec_yaml_two_namespaces() {
        let text =
            "namespaces:\n- name: core\n  version: 2.8.0\n- name: hdmf-common\n  version: 1.8.0\n";
        let specs = parse_nwb_spec_yaml(text).expect("valid two-spec YAML");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "core");
        assert_eq!(specs[1].name, "hdmf-common");
    }

    #[test]
    fn parse_nwb_spec_yaml_rejects_spec_without_name() {
        let text = "namespaces:\n- version: 2.8.0\n";
        let err = parse_nwb_spec_yaml(text).expect_err("missing name must fail");
        assert_eq!(err, NwbNamespaceYamlError::MissingField("name"));
    }

    #[test]
    fn parse_nwb_spec_yaml_rejects_spec_without_version() {
        let text = "namespaces:\n- name: core\n";
        let err = parse_nwb_spec_yaml(text).expect_err("missing version must fail");
        assert_eq!(err, NwbNamespaceYamlError::MissingField("version"));
    }

    #[test]
    fn parse_nwb_spec_yaml_skips_comments_and_blank_lines() {
        let text = "# comment\n\nnamespaces:\n# another comment\n- name: core\n  version: 2.8.0\n";
        let specs = parse_nwb_spec_yaml(text).expect("comments and blanks must be skipped");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "core");
    }

    // ── format/parse roundtrip ────────────────────────────────────────────

    #[test]
    fn format_parse_roundtrip_single_spec_with_types() {
        let original = NwbNamespaceSpec {
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
                NwbTypeSpec {
                    name: String::from("SpatialSeries"),
                    neurodata_type_inc: None,
                },
            ],
        };
        let yaml = format_nwb_spec_yaml(core::slice::from_ref(&original));
        let restored = parse_nwb_spec_yaml(&yaml).expect("roundtrip must succeed");
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0], original);
    }

    #[test]
    fn format_parse_roundtrip_two_specs() {
        let specs = vec![
            NwbNamespaceSpec {
                name: String::from("core"),
                version: String::from("2.8.0"),
                doc_url: None,
                neurodata_types: vec![NwbTypeSpec {
                    name: String::from("TimeSeries"),
                    neurodata_type_inc: None,
                }],
            },
            NwbNamespaceSpec {
                name: String::from("hdmf-common"),
                version: String::from("1.8.0"),
                doc_url: None,
                neurodata_types: vec![
                    NwbTypeSpec {
                        name: String::from("VectorData"),
                        neurodata_type_inc: None,
                    },
                    NwbTypeSpec {
                        name: String::from("DynamicTable"),
                        neurodata_type_inc: None,
                    },
                ],
            },
        ];
        let yaml = format_nwb_spec_yaml(&specs);
        let restored = parse_nwb_spec_yaml(&yaml).expect("two-spec roundtrip must succeed");
        assert_eq!(restored, specs);
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

    #[test]
    fn parse_nwb_spec_yaml_bare_type_name_has_no_inc() {
        let text =
            "namespaces:\n- name: core\n  version: 2.8.0\n  neurodata_types:\n  - TimeSeries\n";
        let specs = parse_nwb_spec_yaml(text).expect("bare type entry must parse");
        assert_eq!(specs[0].neurodata_types.len(), 1);
        assert_eq!(specs[0].neurodata_types[0].name, "TimeSeries");
        assert!(specs[0].neurodata_types[0].neurodata_type_inc.is_none());
    }

    #[test]
    fn parse_nwb_spec_yaml_named_type_with_inc_parses_chain() {
        let text = "namespaces:\n- name: core\n  version: 2.8.0\n  neurodata_types:\n  - name: ElectricalSeries\n    inc: TimeSeries\n";
        let specs = parse_nwb_spec_yaml(text).expect("named type with inc must parse");
        assert_eq!(specs[0].neurodata_types.len(), 1);
        assert_eq!(specs[0].neurodata_types[0].name, "ElectricalSeries");
        assert_eq!(
            specs[0].neurodata_types[0].neurodata_type_inc.as_deref(),
            Some("TimeSeries")
        );
    }

    #[test]
    fn format_parse_roundtrip_type_with_inc() {
        let original = NwbNamespaceSpec {
            name: String::from("core"),
            version: String::from("2.8.0"),
            doc_url: None,
            neurodata_types: vec![
                NwbTypeSpec {
                    name: String::from("NWBDataInterface"),
                    neurodata_type_inc: None,
                },
                NwbTypeSpec {
                    name: String::from("TimeSeries"),
                    neurodata_type_inc: Some(String::from("NWBDataInterface")),
                },
                NwbTypeSpec {
                    name: String::from("ElectricalSeries"),
                    neurodata_type_inc: Some(String::from("TimeSeries")),
                },
            ],
        };
        let yaml = format_nwb_spec_yaml(core::slice::from_ref(&original));
        let restored = parse_nwb_spec_yaml(&yaml).expect("roundtrip with inc must succeed");
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0], original);
    }
}
