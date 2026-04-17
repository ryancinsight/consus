//! FITS header card model boundary.
//!
//! ## Scope
//!
//! This module is the authoritative namespace for FITS header card concepts.
//! A FITS card is the canonical 80-byte logical record used to encode header
//! keywords, values, and comments inside a Header Data Unit (HDU) header.
//!
//! The module currently defines only the architectural boundary and public
//! re-export surface. It intentionally contains no parsing, serialization, or
//! validation logic yet.
//!
//! ## Specification Context
//!
//! FITS headers are composed of fixed-width card images with these invariants:
//! - each card occupies exactly 80 bytes on disk
//! - card ordering is semantically significant
//! - reserved keywords such as `SIMPLE`, `BITPIX`, `NAXIS`, `END`, and
//!   extension-specific keywords participate in HDU-level invariants
//! - keyword/value/comment structure must remain representable without leaking
//!   wire-format concerns into higher-level APIs
//!
//! ## Architecture
//!
//! ```text
//! card/
//! ├── keyword     # keyword identity and reserved-keyword classification
//! ├── value       # typed FITS scalar and textual value model
//! ├── comment     # comment and history payload model
//! └── image       # complete card image descriptor and formatting boundary
//! ```
//!
//! ## Design Constraints
//!
//! - This module is the single source of truth for card-domain types.
//! - Parsing and formatting remain isolated from semantic card modeling.
//! - HDU, header, and validation layers depend on this module, not the reverse.
//! - No FITS I/O behavior is implemented in this scaffold.
//!
//! ## Status

//!
//! This is a documentation-only facade module for the initial crate skeleton.

pub mod comment;
pub mod image;
pub mod keyword;
pub mod value;

pub use comment::CardComment;
pub use image::FitsCardImage;
pub use keyword::{CardKeyword, ReservedKeywordClass};
pub use value::CardValue;
