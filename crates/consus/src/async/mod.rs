//! Asynchronous facade placeholder.
//!
//! The synchronous public API is being established first. This module remains
//! intentionally minimal until the corresponding backend-neutral async
//! abstraction boundary is defined coherently.

#![allow(dead_code)]

/// Marker type reserving the async facade namespace.
///
/// This preserves the public module path without exposing an invalid or
/// incomplete async API surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AsyncFacadeUnavailable;

impl AsyncFacadeUnavailable {
    /// Canonical explanation for the current async facade status.
    pub const fn message() -> &'static str {
        "async facade is temporarily unavailable until the sync API is finalized"
    }
}
