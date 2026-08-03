#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use super::types::{NwbNamespaceSpec, NwbNamespaceYamlError, NwbTypeSpec};

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
        "doc_url" if !value.is_empty() => {
            ns.doc_url = Some(String::from(value));
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
