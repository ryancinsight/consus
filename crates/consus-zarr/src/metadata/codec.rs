#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

/// A compression level for the deflate-family `gzip` and `zlib` codecs.
///
/// zlib defines levels `0..=9` and the sentinel `-1`
/// (`Z_DEFAULT_COMPRESSION`, "currently equivalent to level 6", `zlib.h`).
/// CPython's `zlib.compress` and `gzip.compress` and numcodecs' `Zlib` and
/// `GZip` (0.16.5, run with 10, -2 and -1) accept exactly this set. The
/// sentinel resolves to [`Self::DEFAULT`], so the stored value is always a
/// concrete level.
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
                "zarr `{codec_name}` codec: level {declared:?} is not an integer in 0..=9 or the zlib default -1"
            ),
        })
    }
}

/// The configuration text of a JSON value: strings unquoted, `null` empty,
/// every other value in its JSON form.
#[cfg(feature = "alloc")]
pub(crate) fn config_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
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
                                    (config_key.clone(), config_text(config_value))
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

    /// The text of this codec's declared `level`, `None` when absent.
    ///
    /// # Errors
    ///
    /// [`consus_core::Error::InvalidFormat`] when `level` is declared more
    /// than once with differing text (a Zarr v2 inline `level` beside a
    /// nested `configuration.level`). Every compression codec reads one
    /// level, so the declarations must agree.
    pub(crate) fn declared_level(&self) -> consus_core::Result<Option<&str>> {
        let mut declared = self
            .configuration
            .iter()
            .filter(|(key, _)| key == "level")
            .map(|(_, value)| value.as_str());
        let Some(first) = declared.next() else {
            return Ok(None);
        };
        if let Some(other) = declared.find(|value| *value != first) {
            return Err(consus_core::Error::InvalidFormat {
                message: alloc::format!(
                    "zarr `{}` codec: level is declared as both {first:?} and {other:?}",
                    self.name
                ),
            });
        }
        Ok(Some(first))
    }

    /// The validated compression level of a `gzip` or `zlib` codec.
    ///
    /// An absent `level` is [`DeflateLevel::DEFAULT`]. A present value is
    /// never defaulted or clamped.
    ///
    /// # Errors
    ///
    /// [`consus_core::Error::InvalidFormat`] when `level` is present and is
    /// not an integer in `0..=9` or `-1`, or when [`Self::declared_level`]
    /// rejects its declarations.
    pub(crate) fn deflate_level(&self) -> consus_core::Result<DeflateLevel> {
        self.declared_level()?
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

    fn level_codec(name: &str, levels: &[&str]) -> Codec {
        Codec {
            name: name.into(),
            configuration: levels
                .iter()
                .map(|text| ("level".into(), (*text).into()))
                .collect(),
        }
    }

    #[test]
    fn deflate_level_accepts_zero_to_nine_and_default_sentinel() {
        for n in 0..=9_i32 {
            let level = level_codec("zlib", &[&n.to_string()]).deflate_level();
            assert_eq!(level.expect("level in range").get(), n);
        }
        let accepted: [(&[&str], i32); 3] = [(&["-1"], 6), (&[], 6), (&["3", "3"], 3)];
        for (levels, expected) in accepted {
            let level = level_codec("gzip", levels).deflate_level();
            assert_eq!(level.expect("accepted").get(), expected, "{levels:?}");
        }
    }

    #[test]
    fn declared_level_requires_agreeing_declarations() {
        let agreeing: [(&[&str], Option<&str>); 3] =
            [(&[], None), (&["4"], Some("4")), (&["4", "4"], Some("4"))];
        for (levels, expected) in agreeing {
            let codec = level_codec("zstd", levels);
            let declared = codec.declared_level().expect("agreeing declarations");
            assert_eq!(declared, expected);
        }
        let error = level_codec("lz4", &["9", "1"])
            .declared_level()
            .expect_err("differing declarations");
        assert!(
            matches!(&error, consus_core::Error::InvalidFormat { message }
                if message == "zarr `lz4` codec: level is declared as both \"9\" and \"1\""),
            "{error:?}"
        );
    }

    #[test]
    fn deflate_level_rejects_every_other_declaration() {
        let rejected = [
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
        ];
        for declared in rejected {
            let error = level_codec("zlib", &[declared])
                .deflate_level()
                .expect_err("out-of-range or non-integer level");
            let message = alloc::format!(
                "zarr `zlib` codec: level {declared:?} is not an integer in 0..=9 or the zlib default -1"
            );
            assert!(
                matches!(&error, consus_core::Error::InvalidFormat { message: m } if *m == message),
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
