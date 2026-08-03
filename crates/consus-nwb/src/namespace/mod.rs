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

mod types;
mod yaml;

pub use self::types::*;
pub use self::yaml::{format_nwb_spec_yaml, parse_nwb_spec_yaml};
