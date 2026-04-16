//! High-level public facade for format-agnostic container access.
//!
//! This module defines the canonical `consus::File`, `consus::Group`, and
//! `consus::Dataset` facade types together with the backend abstraction
//! boundary they depend on.
//!
//! This crate contains zero format-specific logic. Concrete format crates are
//! expected to provide adapter implementations for the traits re-exported here.

pub(crate) mod dataset;
pub(crate) mod file;
pub(crate) mod group;

pub use dataset::Dataset;
pub use file::{
    BackendFactory, BackendRegistry, DatasetCreateSpec, File, FileOptions, UnifiedBackend,
    ZeroCopyBytes,
};
pub use group::Group;
