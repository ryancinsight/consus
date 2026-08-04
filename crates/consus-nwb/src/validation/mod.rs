//! NWB schema conformance and constraint checking.
//!
//! Validates NWB files against the NWB 2.x specification: required group
//! presence, required attribute values, and type-compatibility checks.
//!
//! ## Specification
//!
//! An NWB 2.x file must satisfy the following root-group constraints:
//!
//! | Attribute             | Required value | Enforcement          |
//! |-----------------------|----------------|----------------------|
//! | `neurodata_type_def`  | `"NWBFile"`    | hard error           |
//! | `nwb_version`         | any string     | hard error (absence) |
//!
//! Additional required attributes (`identifier`, `session_description`,
//! `session_start_time`) are validated by the session-metadata reader in
//! [`crate::file`]; they are not re-checked here to avoid duplicate I/O.
//!
//! ## Invariants
//!
//! - [`validate_root_attributes`] reads the root group's object header
//!   attributes exactly once and checks both constraints in a single pass.
//! - A file that passes validation is guaranteed to have
//!   `neurodata_type_def == "NWBFile"` on its root HDF5 group.
//! - Validation is read-only: no bytes are written to the source.

mod basic;
#[cfg(feature = "alloc")]
mod report;

pub use basic::is_valid_iso8601;
#[cfg(feature = "alloc")]
pub use basic::{validate_root_attributes, validate_time_series_for_write};
#[cfg(feature = "alloc")]
pub use report::{
    check_dynamic_table_colnames, check_dynamic_table_column_content, check_root_session_attrs,
    ConformanceViolation, NwbConformanceReport,
};
