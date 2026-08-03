#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

/// A codec configuration entry.
///
/// Represents a codec in a Zarr v2 or v3 codec chain. Each codec has a
/// name and an optional configuration object.
///
/// ## Zarr v3 Codec Names
///
/// | Name | Description |
/// |------|-------------|
/// | `"bytes"` | Raw byte transport (endianness) |
/// | `"crc32"` | CRC-32 checksum |
/// | `"gzip"` | Gzip compression |
/// | `"zstd"` | Zstandard compression |
/// | `"lz4"` | LZ4 compression |
/// | `"blosc"` | Blosc meta-compressor |
/// | `"sharding"` | Sharding codec |
///
/// ## Zarr v2 Compressor IDs
///
/// | ID | Codec |
/// |----|-------|
/// | `"zlib"` | deflate |
/// | `"gzip"` | gzip |
/// | `"blosc"` | blosc |
/// | `"lz4"` | lz4 |
/// | `"zstd"` | zstd |
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Codec {
    /// Codec name (e.g., `"gzip"`, `"zstd"`, `"bytes"`).
    pub name: String,
    /// Optional codec configuration as key-value pairs.
    pub configuration: Vec<(String, String)>,
}

#[cfg(feature = "alloc")]
impl Codec {
    /// Returns the gzip compression level if this is a gzip codec.
    pub fn gzip_level(&self) -> Option<u32> {
        if self.name == "gzip" {
            self.configuration
                .iter()
                .find(|(key, _)| key == "level")
                .and_then(|(_, value)| value.parse().ok())
        } else {
            None
        }
    }

    /// Returns the zstd compression level if this is a zstd codec.
    pub fn zstd_level(&self) -> Option<i32> {
        if self.name == "zstd" {
            self.configuration
                .iter()
                .find(|(key, _)| key == "level")
                .and_then(|(_, value)| value.parse().ok())
        } else {
            None
        }
    }

    /// Returns a boolean configuration flag for this codec.
    pub fn bool_flag(&self, key: &str) -> Option<bool> {
        self.configuration
            .iter()
            .find(|(candidate, _)| candidate == key)
            .and_then(|(_, value)| value.parse().ok())
    }

    /// Returns the zstd checksum flag if this is a zstd codec.
    pub fn zstd_checksum(&self) -> Option<bool> {
        if self.name == "zstd" {
            self.bool_flag("checksum")
        } else {
            None
        }
    }

    /// Returns the lz4 compression level if this is an lz4 codec.
    pub fn lz4_level(&self) -> Option<i32> {
        if self.name == "lz4" {
            self.configuration
                .iter()
                .find(|(key, _)| key == "level")
                .and_then(|(_, value)| value.parse().ok())
        } else {
            None
        }
    }

    /// Returns the endianness configuration if this is a bytes codec.
    pub fn bytes_endian(&self) -> Option<&str> {
        self.configuration
            .iter()
            .find(|(key, _)| key == "endian")
            .map(|(_, value)| value.as_str())
    }

    /// Returns true if this codec is a no-op (identity).
    pub fn is_identity(&self) -> bool {
        self.name == "bytes"
            && self
                .configuration
                .iter()
                .all(|(key, value)| key == "endian" && value == "native")
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_codec_is_identity() {
        let bytes_native = Codec {
            name: alloc::string::String::from("bytes"),
            configuration: alloc::vec![(
                alloc::string::String::from("endian"),
                alloc::string::String::from("native")
            )],
        };
        assert!(bytes_native.is_identity());

        let gzip = Codec {
            name: alloc::string::String::from("gzip"),
            configuration: alloc::vec![(
                alloc::string::String::from("level"),
                alloc::string::String::from("1")
            )],
        };
        assert!(!gzip.is_identity());
    }

    #[test]
    fn test_codec_gzip_level() {
        let gzip = Codec {
            name: alloc::string::String::from("gzip"),
            configuration: alloc::vec![(
                alloc::string::String::from("level"),
                alloc::string::String::from("6")
            )],
        };
        assert_eq!(gzip.gzip_level(), Some(6));
    }

    #[test]
    fn test_codec_bool_flag_parses_true() {
        let codec = Codec {
            name: alloc::string::String::from("zstd"),
            configuration: alloc::vec![(
                alloc::string::String::from("checksum"),
                alloc::string::String::from("true")
            )],
        };

        assert_eq!(codec.bool_flag("checksum"), Some(true));
    }

    #[test]
    fn test_codec_bool_flag_parses_false() {
        let codec = Codec {
            name: alloc::string::String::from("zstd"),
            configuration: alloc::vec![(
                alloc::string::String::from("checksum"),
                alloc::string::String::from("false")
            )],
        };

        assert_eq!(codec.bool_flag("checksum"), Some(false));
    }

    #[test]
    fn test_zstd_checksum_extraction() {
        let codec = Codec {
            name: alloc::string::String::from("zstd"),
            configuration: alloc::vec![(
                alloc::string::String::from("checksum"),
                alloc::string::String::from("false")
            )],
        };

        assert_eq!(codec.zstd_checksum(), Some(false));
    }

    #[test]
    fn test_zstd_checksum_non_zstd_codec_returns_none() {
        let codec = Codec {
            name: alloc::string::String::from("gzip"),
            configuration: alloc::vec![(
                alloc::string::String::from("checksum"),
                alloc::string::String::from("true")
            )],
        };

        assert_eq!(codec.zstd_checksum(), None);
    }
}
