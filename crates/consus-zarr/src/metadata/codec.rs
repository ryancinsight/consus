#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

/// A compression level for the deflate-family `gzip` and `zlib` codecs.
///
/// zlib defines levels `0..=9` (`zlib.h`, `deflateInit`: 0 stores, 1 is
/// fastest, 9 is smallest) and the sentinel `-1` (`Z_DEFAULT_COMPRESSION`),
/// "currently equivalent to level 6". CPython's `zlib.compress` and
/// `gzip.compress` accept exactly this set and raise on every other value.
/// The sentinel resolves to [`Self::DEFAULT`] here, so the stored value is
/// always a concrete level.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeflateLevel(u8);

#[cfg(feature = "alloc")]
impl DeflateLevel {
    /// The level used when a codec declares none, and the meaning of `-1`.
    pub(crate) const DEFAULT: Self = Self(6);

    /// The level as the signed hint the codec registry takes.
    pub(crate) fn get(self) -> i32 {
        i32::from(self.0)
    }

    /// Parse the declared text of a `level` configuration value.
    fn parse(codec_name: &str, declared: &str) -> consus_core::Result<Self> {
        let level = declared.parse::<i32>().ok().and_then(|n| match n {
            -1 => Some(Self::DEFAULT),
            0..=9 => u8::try_from(n).ok().map(Self),
            _ => None,
        });
        level.ok_or_else(|| consus_core::Error::InvalidFormat {
            message: alloc::format!(
                "zarr `{codec_name}` codec: level {declared:?} is not an integer                  in 0..=9 or the zlib default -1"
            ),
        })
    }
}

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
/// | `"zlib"` | zlib (RFC 1950) |
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

    /// The validated compression level of a `gzip` or `zlib` codec.
    ///
    /// An absent `level` is [`DeflateLevel::DEFAULT`]. A present value is
    /// never defaulted or clamped.
    ///
    /// # Errors
    ///
    /// [`consus_core::Error::InvalidFormat`] when `level` is present and is
    /// not an integer in `0..=9` or `-1`.
    pub(crate) fn deflate_level(&self) -> consus_core::Result<DeflateLevel> {
        self.config_value("level")
            .map_or(Ok(DeflateLevel::DEFAULT), |declared| {
                DeflateLevel::parse(&self.name, declared)
            })
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

    fn level_codec(name: &str, level: Option<&str>) -> Codec {
        Codec {
            name: alloc::string::String::from(name),
            configuration: level
                .map(|text| {
                    alloc::vec![(
                        alloc::string::String::from("level"),
                        alloc::string::String::from(text)
                    )]
                })
                .unwrap_or_default(),
        }
    }

    #[test]
    fn deflate_level_accepts_zero_to_nine_and_default_sentinel() {
        for n in 0..=9_i32 {
            let level = level_codec("zlib", Some(&n.to_string()))
                .deflate_level()
                .expect("level in range");
            assert_eq!(level.get(), n);
        }
        let sentinel = level_codec("gzip", Some("-1"))
            .deflate_level()
            .expect("zlib default sentinel");
        assert_eq!(sentinel, DeflateLevel::DEFAULT);
        assert_eq!(sentinel.get(), 6);
        assert_eq!(
            level_codec("zlib", None).deflate_level().expect("absent"),
            DeflateLevel::DEFAULT
        );
    }

    #[test]
    fn deflate_level_rejects_every_other_declaration() {
        for declared in [
            "10",
            "-2",
            "2147483647",
            "-2147483648",
            "4294967295",
            "99999999999999999999",
            "1.5",
            "",
            "true",
            "six",
        ] {
            let error = level_codec("zlib", Some(declared))
                .deflate_level()
                .expect_err("out-of-range or non-integer level");
            assert!(
                matches!(
                    &error,
                    consus_core::Error::InvalidFormat { message }
                        if message.contains(&alloc::format!("{declared:?}"))
                            && message.contains("`zlib`")
                ),
                "level {declared:?}: {error:?}"
            );
        }
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

        let nested = codec
            .codec_array("codecs")
            .expect("nested codecs must parse");
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0].name, "bytes");
        assert_eq!(nested[0].bytes_endian(), Some("little"));
        assert_eq!(nested[0].config_value("level"), Some("1"));
    }
}
