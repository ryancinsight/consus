//! External link resolution and validation.
//!
//! ## Specification
//!
//! External links (link type byte 64) reference objects in other HDF5 files.
//! The link value payload (after the 2-byte length prefix handled by the
//! parent [`super::Hdf5Link::parse`]) has the following layout:
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0 | 1 | Version / flags byte (currently ignored by readers) |
//! | 1 | var | Null-terminated filename string (UTF-8) |
//! | var | var | Null-terminated object path string (UTF-8) |
//!
//! Both strings are encoded as UTF-8 and terminated by a `0x00` byte.
//! The object path is expected to be absolute (starting with `'/'`).
//!
//! Reference: *HDF5 File Format Specification Version 3.0*, Section IV.A.2.g
//! (<https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html>)

#[cfg(feature = "alloc")]
use alloc::string::String;

use consus_core::{Error, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Resolved external link target.
///
/// Carries the two components needed to open an object in another HDF5 file:
/// the filesystem path to the file and the absolute object path within it.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalTarget {
    /// Filesystem path to the external HDF5 file.
    ///
    /// May be absolute or relative; interpretation is caller-defined.
    pub file_path: String,
    /// Absolute object path within the external file (must start with `'/'`).
    pub object_path: String,
}

#[cfg(feature = "alloc")]
impl ExternalTarget {
    /// Construct a new external target from validated components.
    ///
    /// No validation is performed at construction time; call [`validate`]
    /// to enforce the absolute-path invariant.
    ///
    /// [`validate`]: Self::validate
    pub fn new(file_path: String, object_path: String) -> Self {
        Self {
            file_path,
            object_path,
        }
    }

    /// Validate that the object path is absolute (starts with `'/'`).
    ///
    /// The HDF5 specification requires external link object paths to be
    /// absolute within the target file.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidFormat` if `object_path` does not begin
    /// with `'/'`.
    pub fn validate(&self) -> Result<()> {
        if !self.object_path.starts_with('/') {
            return Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: alloc::format!(
                    "external link object path must be absolute, got: {}",
                    self.object_path
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse an external link value from raw payload bytes.
///
/// The input `data` must point to the start of the external link payload
/// (i.e., the first byte is the version/flags byte, followed by the two
/// null-terminated strings). The 2-byte length prefix that precedes the
/// payload in the link message is **not** included in `data`; it is
/// consumed by the caller ([`super::Hdf5Link::parse`]).
///
/// ## Layout
///
/// ```text
/// +------+----------------------------+----------------------------+
/// | 0x00 | filename\0                 | object_path\0              |
/// | flags| (null-terminated UTF-8)    | (null-terminated UTF-8)    |
/// +------+----------------------------+----------------------------+
/// ```
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` when:
/// - `data` is empty (no version/flags byte).
/// - Either null terminator is missing.
/// - Either string is not valid UTF-8.
#[cfg(feature = "alloc")]
pub fn parse_external_link_value(data: &[u8]) -> Result<ExternalTarget> {
    if data.is_empty() {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: String::from("empty external link data"),
        });
    }

    // Byte 0: version/flags — reserved for future use, currently ignored.
    let _flags = data[0];
    let rest = &data[1..];

    // --- Filename (null-terminated) -------------------------------------------
    let filename_end = rest
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: String::from("external link filename not null-terminated"),
        })?;

    let filename =
        core::str::from_utf8(&rest[..filename_end]).map_err(|_| Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: String::from("external link filename is not valid UTF-8"),
        })?;

    // --- Object path (null-terminated) ----------------------------------------
    let obj_start = filename_end + 1; // skip the null terminator
    let obj_rest = &rest[obj_start..];

    let obj_end = obj_rest
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: String::from("external link object path not null-terminated"),
        })?;

    let object_path =
        core::str::from_utf8(&obj_rest[..obj_end]).map_err(|_| Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: String::from("external link object path is not valid UTF-8"),
        })?;

    Ok(ExternalTarget::new(
        String::from(filename),
        String::from(object_path),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;

    /// Golden-path: well-formed external link payload with absolute object path.
    #[test]
    fn parse_valid_external_link() {
        // flags=0x00, filename="other.h5\0", object_path="/group/dataset\0"
        let mut data: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
        data.push(0x00); // flags
        data.extend_from_slice(b"other.h5\0");
        data.extend_from_slice(b"/group/dataset\0");

        let target = parse_external_link_value(&data).expect("parse must succeed");
        assert_eq!(target.file_path, "other.h5");
        assert_eq!(target.object_path, "/group/dataset");
        assert!(target.validate().is_ok());
    }

    /// Relative object path fails validation.
    #[test]
    fn validate_rejects_relative_path() {
        let target = ExternalTarget::new(String::from("file.h5"), String::from("group/dataset"));
        let err = target.validate().unwrap_err();
        let msg = alloc::format!("{err}");
        assert!(
            msg.contains("absolute"),
            "error message must mention 'absolute', got: {msg}"
        );
    }

    /// Empty payload is rejected.
    #[test]
    fn parse_empty_payload_fails() {
        let result = parse_external_link_value(&[]);
        assert!(result.is_err());
    }

    /// Missing null terminator on filename is rejected.
    #[test]
    fn parse_missing_filename_terminator_fails() {
        let data = [0x00, b'a', b'b', b'c']; // no null terminator
        let result = parse_external_link_value(&data);
        assert!(result.is_err());
    }

    /// Missing null terminator on object path is rejected.
    #[test]
    fn parse_missing_object_path_terminator_fails() {
        let mut data: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
        data.push(0x00);
        data.extend_from_slice(b"file.h5\0");
        data.extend_from_slice(b"/no_terminator"); // no trailing \0
        let result = parse_external_link_value(&data);
        assert!(result.is_err());
    }

    /// Flags byte is accepted but currently ignored.
    #[test]
    fn parse_nonzero_flags_accepted() {
        let mut data: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
        data.push(0x01); // non-zero flags
        data.extend_from_slice(b"f.h5\0");
        data.extend_from_slice(b"/root\0");

        let target = parse_external_link_value(&data).expect("non-zero flags must be accepted");
        assert_eq!(target.file_path, "f.h5");
        assert_eq!(target.object_path, "/root");
    }
}
