//! NWB namespace registry and type system.
//!
//! Manages NWB specification namespaces. In NWB 2.x files, namespace
//! definitions are stored as YAML-serialised spec files under
//! `/specifications/{namespace_name}/{version}/`. Reading and parsing those
//! YAML specs is a future roadmap item; this module provides the
//! registry model and the hard-coded core namespace entry as a stable
//! compile-time constant.
//!
//! ## Invariants
//!
//! - `NwbNamespace::CORE_NAME` is always `"core"`.
//! - `NwbNamespace::core()` returns the canonical NWB 2.x core namespace
//!   descriptor without I/O.

#[cfg(feature = "alloc")]
use alloc::string::String;

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
        // Equality is value-semantic: two independent calls produce equal descriptors.
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
        mutated.version = alloc::string::String::from("99.0.0");

        // Original is unchanged; clone owns its data.
        assert_eq!(original.version, "2.8.0");
        assert_eq!(mutated.version, "99.0.0");
    }

    // ── CORE_NAME consistency with core() ────────────────────────────────

    #[test]
    fn core_instance_name_matches_core_name_constant() {
        let ns = NwbNamespace::core();
        // The name field of core() must equal the CORE_NAME compile-time constant.
        assert_eq!(ns.name.as_str(), NwbNamespace::CORE_NAME);
    }
}
