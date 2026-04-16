//! Fluent builder APIs for the format-agnostic `consus` facade.
//!
//! This module defines the public builder surface only. It contains no
//! format-specific logic.

pub(crate) mod dataset;
pub(crate) mod file;
pub(crate) mod group;

pub use dataset::{DatasetBuilder, DatasetBuilderSpec};
pub use file::{FileBuilder, FileOpenOptions};
pub use group::GroupBuilder;
