//! Untrusted-input parsing policy shared by every format backend.
//!
//! ```text
//! parse/
//! └── budget    # ParseBudget: byte, element, and depth ceilings
//! ```

// Trust-boundary floor: this module is already clean under the two
// restriction lints the standard requires of parser code, so they are
// denied here rather than left to the workspace-level ratchet.
#![deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
#![deny(clippy::unwrap_used)]

pub mod budget;

pub use budget::ParseBudget;
