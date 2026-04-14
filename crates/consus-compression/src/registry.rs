//! Codec registry for runtime codec lookup.
//!
//! ## Design
//!
//! The registry maps `CodecId` → `&dyn Codec`. Format backends use the
//! registry to resolve filter pipelines at runtime. A default registry
//! is pre-populated with all codecs enabled by cargo features.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{Codec, CodecId};
use consus_core::error::{Error, Result};

/// Runtime codec registry.
///
/// Pre-populated with all codecs enabled by feature flags.
pub struct CodecRegistry {
    #[cfg(feature = "alloc")]
    codecs: Vec<(CodecId, &'static dyn Codec)>,
}

impl CodecRegistry {
    /// Create a new registry pre-populated with default codecs.
    #[cfg(feature = "alloc")]
    pub fn new() -> Self {
        let mut codecs: Vec<(CodecId, &'static dyn Codec)> = Vec::new();

        #[cfg(feature = "deflate")]
        {
            use crate::codecs::deflate::DeflateCodec;
            static DEFLATE: DeflateCodec = DeflateCodec;
            codecs.push((CodecId::FilterId(1), &DEFLATE));
        }

        #[cfg(feature = "zstd")]
        {
            use crate::codecs::zstd_codec::ZstdCodec;
            static ZSTD: ZstdCodec = ZstdCodec;
            codecs.push((CodecId::FilterId(32015), &ZSTD));
        }

        #[cfg(feature = "lz4")]
        {
            use crate::codecs::lz4_codec::Lz4Codec;
            static LZ4: Lz4Codec = Lz4Codec;
            codecs.push((CodecId::FilterId(32004), &LZ4));
        }

        Self { codecs }
    }

    /// Look up a codec by ID.
    #[cfg(feature = "alloc")]
    pub fn get(&self, id: &CodecId) -> Result<&dyn Codec> {
        self.codecs
            .iter()
            .find(|(cid, _)| cid == id)
            .map(|(_, codec)| *codec)
            .ok_or_else(|| Error::UnsupportedFeature {
                feature: alloc::format!("codec {:?}", id),
            })
    }

    /// Register an additional codec.
    #[cfg(feature = "alloc")]
    pub fn register(&mut self, id: CodecId, codec: &'static dyn Codec) {
        self.codecs.push((id, codec));
    }
}

#[cfg(feature = "alloc")]
impl Default for CodecRegistry {
    fn default() -> Self {
        Self::new()
    }
}
