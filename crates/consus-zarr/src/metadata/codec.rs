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
    /// Returns a raw configuration value by key.
    pub fn config_value(&self, key: &str) -> Option<&str> {
        self.configuration
            .iter()
            .find(|(candidate, _)| candidate == key)
            .map(|(_, value)| value.as_str())
    }

    fn parsed_config<T: core::str::FromStr>(&self, key: &str) -> Option<T> {
        self.config_value(key).and_then(|value| value.parse().ok())
    }

    fn config_json(&self, key: &str) -> Option<serde_json::Value> {
        serde_json::from_str(self.config_value(key)?).ok()
    }

    /// Returns a vector-valued JSON configuration field as `usize`s.
    pub fn usize_vec(&self, key: &str) -> Option<Vec<usize>> {
        self.config_json(key)?
            .as_array()?
            .iter()
            .map(|value| value.as_u64().map(|n| n as usize))
            .collect()
    }

    /// Returns a nested codec array stored inside this codec's configuration.
    pub fn codec_array(&self, key: &str) -> Option<Vec<Codec>> {
        Some(
            self.config_json(key)?
                .as_array()?
                .iter()
                .filter_map(|value| {
                    let name = value.get("name")?.as_str()?.to_owned();
                    let configuration = value
                        .get("configuration")
                        .and_then(|config| config.as_object())
                        .map(|config| {
                            config
                                .iter()
                                .map(|(config_key, config_value)| {
                                    (
                                        config_key.clone(),
                                        match config_value {
                                            serde_json::Value::String(s) => s.clone(),
                                            serde_json::Value::Number(n) => n.to_string(),
                                            serde_json::Value::Bool(b) => b.to_string(),
                                            serde_json::Value::Null => String::new(),
                                            _ => config_value.to_string(),
                                        },
                                    )
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    Some(Codec {
                        name,
                        configuration,
                    })
                })
                .collect(),
        )
    }

    /// Returns the gzip compression level if this is a gzip codec.
    pub fn gzip_level(&self) -> Option<u32> {
        if self.name == "gzip" {
            self.parsed_config("level")
        } else {
            None
        }
    }

    /// Returns the zstd compression level if this is a zstd codec.
    pub fn zstd_level(&self) -> Option<i32> {
        if self.name == "zstd" {
            self.parsed_config("level")
        } else {
            None
        }
    }

    /// Returns a boolean configuration flag for this codec.
    pub fn bool_flag(&self, key: &str) -> Option<bool> {
        self.parsed_config(key)
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
            self.parsed_config("level")
        } else {
            None
        }
    }

    /// Returns the endianness configuration if this is a bytes codec.
    pub fn bytes_endian(&self) -> Option<&str> {
        self.config_value("endian")
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

    #[test]
    fn test_usize_vec_parses_json_array() {
        let codec = Codec {
            name: alloc::string::String::from("sharding_indexed"),
            configuration: alloc::vec![(
                alloc::string::String::from("chunk_shape"),
                alloc::string::String::from("[2, 4, 8]"),
            )],
        };

        assert_eq!(codec.usize_vec("chunk_shape"), Some(vec![2, 4, 8]));
    }

    #[test]
    fn test_codec_array_parses_nested_codecs() {
        let codec = Codec {
            name: alloc::string::String::from("sharding_indexed"),
            configuration: alloc::vec![(
                alloc::string::String::from("codecs"),
                alloc::string::String::from(
                    r#"[{"name":"bytes","configuration":{"endian":"little","level":1}}]"#,
                ),
            )],
        };

        let nested = codec.codec_array("codecs").expect("nested codecs must parse");
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0].name, "bytes");
        assert_eq!(nested[0].bytes_endian(), Some("little"));
        assert_eq!(nested[0].config_value("level"), Some("1"));
    }
}
