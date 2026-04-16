//! Codec registry trait and default implementation.
//!
//! ## Design
//!
//! The [`CompressionRegistry`] trait defines the interface for codec lookup.
//! Format backends depend on this trait (DIP) rather than a concrete registry,
//! enabling custom registries (e.g., with format-specific codecs).
//!
//! [`DefaultCodecRegistry`] provides the standard implementation, pre-populated
//! with all codecs enabled by cargo feature flags.
//!
//! ## Module Hierarchy
//!
//! ```text
//! registry/
//! ├── mod.rs      # CompressionRegistry trait
//! └── default.rs  # DefaultCodecRegistry implementation
//! ```

#[cfg(feature = "alloc")]
pub mod default;

#[cfg(feature = "alloc")]
pub use default::DefaultCodecRegistry;

#[cfg(feature = "alloc")]
use crate::codec::traits::{Codec, CodecId};

#[cfg(feature = "alloc")]
use consus_core::Result;

/// Trait for pluggable codec registries.
///
/// Format backends depend on this trait (DIP) rather than a concrete registry.
/// This enables custom registries (e.g., with format-specific codecs or
/// restricted codec sets for security-sensitive contexts).
///
/// ## Invariant
///
/// For any registered codec with id `id`:
///   `registry.get(&id).is_ok()` iff the codec was previously registered or
///   included in the default set.
///
/// ## Contract
///
/// - [`get`](CompressionRegistry::get) returns `Ok` for any previously
///   registered `CodecId`, and `Err(Error::UnsupportedFeature)` otherwise.
/// - [`register`](CompressionRegistry::register) makes `id` immediately
///   resolvable by subsequent calls to [`get`](CompressionRegistry::get).
/// - [`get_by_name`](CompressionRegistry::get_by_name) searches by the
///   codec's [`Codec::name`] value, independent of the `CodecId` key.
/// - [`codec_ids`](CompressionRegistry::codec_ids) returns all registered
///   `(CodecId, &dyn Codec)` pairs in registration order.
#[cfg(feature = "alloc")]
pub trait CompressionRegistry: Send + Sync {
    /// Look up a codec by its identifier.
    ///
    /// # Errors
    ///
    /// Returns `Error::UnsupportedFeature` if no codec is registered
    /// under the given `id`.
    fn get(&self, id: &CodecId) -> Result<&dyn Codec>;

    /// Register an additional codec, keyed by the given identifier.
    ///
    /// If a codec with the same `id` already exists, both entries remain
    /// in the registry; the first registered entry wins on lookup.
    fn register(&mut self, id: CodecId, codec: &'static dyn Codec);

    /// Returns a slice over all registered `(CodecId, &dyn Codec)` pairs
    /// in registration order.
    fn codec_ids(&self) -> &[(CodecId, &'static dyn Codec)];

    /// Check whether a codec with the given identifier is registered.
    ///
    /// Default implementation delegates to [`get`](CompressionRegistry::get).
    fn contains(&self, id: &CodecId) -> bool {
        self.get(id).is_ok()
    }

    /// Look up a codec by its human-readable name.
    ///
    /// Searches by comparing [`Codec::name`] against `name`, independent
    /// of the `CodecId` key used during registration.
    ///
    /// # Errors
    ///
    /// Returns `Error::UnsupportedFeature` if no codec with the given
    /// name is registered.
    fn get_by_name(&self, name: &str) -> Result<&dyn Codec>;
}

/// Backward-compatible type alias for [`DefaultCodecRegistry`].
///
/// Existing code using `CodecRegistry` continues to work. New code should
/// depend on the [`CompressionRegistry`] trait instead.
#[cfg(feature = "alloc")]
pub type CodecRegistry = DefaultCodecRegistry;
