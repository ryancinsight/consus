//! TIFF image-file-directory traversal for EXIF presentation metadata.
//!
//! The traversal uses fixed-capacity pending and visited arrays, so encoded
//! metadata cannot cause allocation. Every directory, entry table, and
//! out-of-line value range is checked against the input slice before use.
//! Repeated offsets reject cycles, and the shared 16-directory bound limits
//! both traversal depth and breadth. The module tests cover both byte orders,
//! cycles, the directory bound, extent failures, and supported presentation
//! metadata.

#![deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

mod field;
mod metadata;
mod traversal;

pub(super) use traversal::parse;
