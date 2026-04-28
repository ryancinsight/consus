//! NWB version detection and compatibility.
//!
//! Reads the `nwb_version` attribute from the root HDF5 group to determine
//! the NWB specification version and selects the appropriate parsing path.
//!
//! ## Specification
//!
//! NWB 2.x files store a scalar string attribute `nwb_version` on the root
//! group (e.g. `"2.7.0"`).  The major and minor version components determine
//! the set of neurodata types and required fields that the file must satisfy.
//!
//! ## Invariants
//!
//! - `NwbVersion::parse` is total: every string maps to exactly one variant.
//! - `detect_version` returns `Error::NotFound` when the attribute is absent.
//! - `is_supported` is false only for `Unknown`, which represents a future or
//!   unrecognised version string.

#[cfg(feature = "alloc")]
use alloc::string::String;

use consus_core::{AttributeValue, Error, Result};
use consus_hdf5::file::Hdf5File;
use consus_io::ReadAt;

/// NWB 2.x specification version.
///
/// Encodes the recognised minor-version series of the NWB 2 specification.
/// Any version string that does not match a known series is preserved as
/// `Unknown(String)` so callers can inspect it without information loss.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NwbVersion {
    /// NWB 2.0.x
    V2_0,
    /// NWB 2.1.x
    V2_1,
    /// NWB 2.2.x
    V2_2,
    /// NWB 2.3.x
    V2_3,
    /// NWB 2.4.x
    V2_4,
    /// NWB 2.5.x
    V2_5,
    /// NWB 2.6.x
    V2_6,
    /// NWB 2.7.x
    V2_7,
    /// Unrecognised version string, preserved verbatim.
    #[cfg(feature = "alloc")]
    Unknown(String),
}

impl NwbVersion {
    /// Parse an `nwb_version` attribute string into a `NwbVersion`.
    ///
    /// Matches on the major.minor prefix of `s` (e.g. `"2.7.0"` → `V2_7`).
    /// Trailing patch components and whitespace are ignored.
    ///
    /// ## Invariant
    ///
    /// `parse` is total: every input produces exactly one variant.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn parse(s: &str) -> Self {
        let s = s.trim();
        if s.starts_with("2.0") {
            NwbVersion::V2_0
        } else if s.starts_with("2.1") {
            NwbVersion::V2_1
        } else if s.starts_with("2.2") {
            NwbVersion::V2_2
        } else if s.starts_with("2.3") {
            NwbVersion::V2_3
        } else if s.starts_with("2.4") {
            NwbVersion::V2_4
        } else if s.starts_with("2.5") {
            NwbVersion::V2_5
        } else if s.starts_with("2.6") {
            NwbVersion::V2_6
        } else if s.starts_with("2.7") {
            NwbVersion::V2_7
        } else {
            NwbVersion::Unknown(String::from(s))
        }
    }

    /// Returns `true` when the version is in the set of recognised NWB 2.x series.
    ///
    /// `Unknown` variants return `false` regardless of content.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        !matches!(self, NwbVersion::Unknown(_))
    }

    /// Returns the canonical two-component version string for display purposes.
    ///
    /// `Unknown` variants return the raw string they were constructed with.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            NwbVersion::V2_0 => "2.0",
            NwbVersion::V2_1 => "2.1",
            NwbVersion::V2_2 => "2.2",
            NwbVersion::V2_3 => "2.3",
            NwbVersion::V2_4 => "2.4",
            NwbVersion::V2_5 => "2.5",
            NwbVersion::V2_6 => "2.6",
            NwbVersion::V2_7 => "2.7",
            #[cfg(feature = "alloc")]
            NwbVersion::Unknown(s) => s.as_str(),
        }
    }
}

/// Detect the NWB specification version from the root group's `nwb_version` attribute.
///
/// ## Algorithm
///
/// 1. Read all attributes attached to the root group object header.
/// 2. Find the attribute named `nwb_version`.
/// 3. Decode its scalar string value.
/// 4. Parse the version string via [`NwbVersion::parse`].
///
/// ## Errors
///
/// - [`Error::NotFound`] when the `nwb_version` attribute is absent.
/// - [`Error::InvalidFormat`] when the attribute value cannot be decoded as a string.
/// - Propagates any I/O or format errors from `Hdf5File::attributes_at`.
#[cfg(feature = "alloc")]
pub fn detect_version<R: ReadAt + Sync>(file: &Hdf5File<R>) -> Result<NwbVersion> {
    let root_addr = file.superblock().root_group_address;
    let attrs = file.attributes_at(root_addr)?;
    for attr in &attrs {
        if attr.name == "nwb_version" {
            return match attr.decode_value() {
                Ok(AttributeValue::String(v)) => Ok(NwbVersion::parse(&v)),
                Ok(_) => Err(Error::InvalidFormat {
                    message: String::from("NWB: nwb_version attribute is not a string"),
                }),
                Err(e) => Err(e),
            };
        }
    }
    Err(Error::NotFound {
        path: String::from("nwb_version"),
    })
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    // ── NwbVersion::parse ──────────────────────────────────────────────────

    #[test]
    fn parse_2_7_0_returns_v2_7() {
        assert_eq!(NwbVersion::parse("2.7.0"), NwbVersion::V2_7);
    }

    #[test]
    fn parse_2_0_x_returns_v2_0() {
        assert_eq!(NwbVersion::parse("2.0.2"), NwbVersion::V2_0);
    }

    #[test]
    fn parse_2_6_x_returns_v2_6() {
        assert_eq!(NwbVersion::parse("2.6.0"), NwbVersion::V2_6);
    }

    #[test]
    fn parse_whitespace_trimmed() {
        assert_eq!(NwbVersion::parse("  2.5.1  "), NwbVersion::V2_5);
    }

    #[test]
    fn parse_unknown_version_preserved() {
        let ver = NwbVersion::parse("3.0.0");
        assert!(!ver.is_supported());
        assert_eq!(ver.as_str(), "3.0.0");
    }

    #[test]
    fn parse_empty_string_is_unknown() {
        let ver = NwbVersion::parse("");
        assert!(!ver.is_supported());
    }

    // ── NwbVersion::is_supported ───────────────────────────────────────────

    #[test]
    fn all_known_versions_are_supported() {
        let known = [
            NwbVersion::V2_0,
            NwbVersion::V2_1,
            NwbVersion::V2_2,
            NwbVersion::V2_3,
            NwbVersion::V2_4,
            NwbVersion::V2_5,
            NwbVersion::V2_6,
            NwbVersion::V2_7,
        ];
        for v in &known {
            assert!(v.is_supported(), "expected {:?} to be supported", v);
        }
    }

    #[test]
    fn unknown_version_is_not_supported() {
        let ver = NwbVersion::Unknown(String::from("99.0.0"));
        assert!(!ver.is_supported());
    }

    // ── NwbVersion::as_str ─────────────────────────────────────────────────

    #[test]
    fn as_str_v2_7_returns_2_7() {
        assert_eq!(NwbVersion::V2_7.as_str(), "2.7");
    }

    #[test]
    fn as_str_unknown_returns_raw() {
        let ver = NwbVersion::Unknown(String::from("3.1.0"));
        assert_eq!(ver.as_str(), "3.1.0");
    }
}
