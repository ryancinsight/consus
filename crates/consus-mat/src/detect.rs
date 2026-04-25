//! MATLAB .mat file version detection.
//!
//! ## Algorithm
//!
//! 1. Bytes 0-7 == HDF5 magic -> [`MatVersion::V73`].
//! 2. Bytes 124-127 match MAT v5 header -> [`MatVersion::V5`].
//! 3. Fallback -> [`MatVersion::V4`].

/// MATLAB .mat file format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatVersion {
    /// Level 4 format (MATLAB 4): plain binary variable records.
    V4,
    /// Level 5 format (MATLAB 5-7.2): structured binary with element tags.
    V5,
    /// Level 7.3 format (MATLAB 7.3+): HDF5-based.
    V73,
}

/// HDF5 file signature (8 bytes at file offset 0 for v7.3 MAT files).
const HDF5_SIG: &[u8; 8] = b"\x89HDF\r\n\x1a\n";

/// Detect the MAT file version from the file's leading bytes.
///
/// Returns `None` if `header_bytes` is empty.
pub fn detect_version(header_bytes: &[u8]) -> Option<MatVersion> {
    if header_bytes.is_empty() {
        return None;
    }

    // Check HDF5 magic -> v7.3.
    if header_bytes.len() >= 8 && header_bytes[..8] == *HDF5_SIG {
        return Some(MatVersion::V73);
    }

    // Check MAT v5 endian indicator at bytes 126-127.
    if header_bytes.len() >= 128 {
        let ver = [header_bytes[124], header_bytes[125]];
        let endian = [header_bytes[126], header_bytes[127]];
        // LE file: version = 0x0100 in LE = [0x00, 0x01], endian indicator = "IM"
        // BE file: version = 0x0100 in BE = [0x01, 0x00], endian indicator = "MI"
        if (ver == [0x00, 0x01] && endian == *b"IM")
            || (ver == [0x01, 0x00] && endian == *b"MI")
        {
            return Some(MatVersion::V5);
        }
    }

    // Default: treat as v4 (no magic header).
    Some(MatVersion::V4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_hdf5_magic_returns_v73() {
        let mut data = [0u8; 128];
        data[..8].copy_from_slice(b"\x89HDF\r\n\x1a\n");
        assert_eq!(detect_version(&data), Some(MatVersion::V73));
    }

    #[test]
    fn detect_v5_le_header() {
        let mut data = [0u8; 128];
        data[124] = 0x00;
        data[125] = 0x01;
        data[126] = b'I';
        data[127] = b'M';
        assert_eq!(detect_version(&data), Some(MatVersion::V5));
    }

    #[test]
    fn detect_v5_be_header() {
        let mut data = [0u8; 128];
        data[124] = 0x01;
        data[125] = 0x00;
        data[126] = b'M';
        data[127] = b'I';
        assert_eq!(detect_version(&data), Some(MatVersion::V5));
    }

    #[test]
    fn detect_unknown_falls_back_to_v4() {
        let data = [0u8; 128];
        assert_eq!(detect_version(&data), Some(MatVersion::V4));
    }

    #[test]
    fn detect_empty_returns_none() {
        assert_eq!(detect_version(&[]), None);
    }
}
