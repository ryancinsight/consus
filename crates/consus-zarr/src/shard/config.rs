#[cfg(feature = "alloc")]
use alloc::{
    string::{String, ToString},
    vec::Vec,
};

use crate::metadata::Codec;

/// Configuration extracted from a `sharding_indexed` codec entry in the codec chain.
///
/// ## Relation to `ArrayMetadata`
///
/// - `meta.chunks` is the outer chunk (shard) shape.
/// - `inner_chunk_shape` is the shape of each inner chunk (sub-chunk) within a shard.
/// - `inner_chunks_per_dim[i] = ceil(meta.chunks[i] / inner_chunk_shape[i])`
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct ShardingConfig {
    /// Shape of inner chunks (sub-chunks) within each shard.
    pub inner_chunk_shape: Vec<usize>,
    /// Codec chain applied to each inner chunk for compress/decompress.
    pub inner_codecs: Vec<Codec>,
    /// Codec chain applied to the shard index.
    pub index_codecs: Vec<Codec>,
}

#[cfg(feature = "alloc")]
impl ShardingConfig {
    /// Number of inner chunks along each shard dimension.
    ///
    /// `inner_chunks_per_dim[i] = ceil(shard_shape[i] / inner_chunk_shape[i])`
    #[must_use]
    pub fn inner_chunks_per_dim(&self, shard_shape: &[usize]) -> Vec<usize> {
        shard_shape
            .iter()
            .zip(self.inner_chunk_shape.iter())
            .map(
                |(&shard, &inner)| {
                    if inner == 0 { 0 } else { shard.div_ceil(inner) }
                },
            )
            .collect()
    }

    /// Total number of inner chunks per shard.
    ///
    /// `total = product(inner_chunks_per_dim)`
    #[must_use]
    pub fn total_inner_chunks(&self, shard_shape: &[usize]) -> usize {
        let per_dim = self.inner_chunks_per_dim(shard_shape);
        if per_dim.is_empty() {
            1
        } else {
            per_dim.iter().product()
        }
    }

    /// Size of the shard index in bytes.
    ///
    /// `index_size = total_inner_chunks * 16`  (8-byte offset + 8-byte length per entry)
    #[must_use]
    pub fn index_size_bytes(&self, shard_shape: &[usize]) -> usize {
        self.total_inner_chunks(shard_shape).saturating_mul(16)
    }
}

/// Extract `ShardingConfig` from a codec chain.
///
/// Returns `Some(ShardingConfig)` when the chain contains a `sharding_indexed` codec,
/// `None` otherwise.
#[cfg(feature = "alloc")]
pub fn extract_sharding_config(codecs: &[Codec]) -> Option<ShardingConfig> {
    for codec in codecs {
        if codec.name == "sharding_indexed" {
            let inner_chunk_shape = extract_usize_vec(codec, "chunk_shape")?;
            let inner_codecs = extract_codec_array(codec, "codecs").unwrap_or_default();
            let index_codecs = extract_codec_array(codec, "index_codecs").unwrap_or_default();
            return Some(ShardingConfig {
                inner_chunk_shape,
                inner_codecs,
                index_codecs,
            });
        }
    }
    None
}

#[cfg(feature = "alloc")]
fn extract_usize_vec(codec: &Codec, key: &str) -> Option<Vec<usize>> {
    let val = codec
        .configuration
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())?;
    let json: serde_json::Value = serde_json::from_str(val).ok()?;
    json.as_array()?
        .iter()
        .map(|v| v.as_u64().map(|n| n as usize))
        .collect()
}

#[cfg(feature = "alloc")]
fn extract_codec_array(codec: &Codec, key: &str) -> Option<Vec<Codec>> {
    let val = codec
        .configuration
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())?;
    let json: serde_json::Value = serde_json::from_str(val).ok()?;
    Some(
        json.as_array()?
            .iter()
            .filter_map(|v| {
                let name = v.get("name")?.as_str()?.to_string();
                let config = v
                    .get("configuration")
                    .and_then(|c| c.as_object())
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, v)| {
                                let s = match v {
                                    serde_json::Value::String(s) => s.clone(),
                                    serde_json::Value::Number(n) => n.to_string(),
                                    serde_json::Value::Bool(b) => b.to_string(),
                                    serde_json::Value::Null => String::new(),
                                    _ => v.to_string(),
                                };
                                Some((k.clone(), s))
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                Some(Codec {
                    name,
                    configuration: config,
                })
            })
            .collect(),
    )
}
