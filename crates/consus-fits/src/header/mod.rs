//! FITS header parsing, keyword normalization, and semantic validation.
//!
//! ## Scope
//!
//! This module is the single source of truth for FITS header handling in
//! `consus-fits`. It owns:
//! - 80-character card parsing
//! - canonical keyword classification and normalization
//! - typed header value representation
//! - header assembly from ordered card sequences
//! - mandatory primary-header validation for `SIMPLE`, `BITPIX`, and `NAXIS`
//!
//! ## Architectural role
//!
//! The module preserves SSOT by centralizing all header keyword handling in one
//! bounded context. Higher-level HDU and file logic depend on this module for
//! header semantics rather than re-implementing keyword parsing or validation.
//!
//! ## FITS Standard alignment
//!
//! The implementation surface is designed around FITS Standard 4.0 concepts:
//! - each card image occupies exactly 80 bytes
//! - headers are ordered sequences of card images
//! - long string values may be continued with `CONTINUE` cards
//! - hierarchical keywords are represented explicitly
//! - primary headers require `SIMPLE`, `BITPIX`, and `NAXIS`
//!
//! ## Module hierarchy
//!
//! ```text
//! header/
//! ├── card.rs      # 80-character card image model
//! ├── keyword.rs   # canonical keyword normalization and classification
//! ├── value.rs     # typed FITS header value model
//! └── parser.rs    # ordered header parsing and mandatory-key validation
//! ```
//!
//! ## Design constraints
//!
//! - Header keyword handling is authoritative in this subtree only.
//! - Parsing remains deterministic and order-preserving.
//! - Validation is value-semantic and does not depend on file I/O.
//! - Public exports are limited to canonical header-facing types and functions.

pub mod card;
pub mod keyword;
pub mod parser;
pub mod value;

pub use card::{FITS_CARD_LEN, FitsCard};
pub use keyword::{
    classify_mandatory_primary_keyword, classify_reserved_keyword, is_mandatory_primary_keyword,
    is_reserved_keyword, is_valid_hierarchical_segment, mandatory_primary_keywords,
    FitsKeyword, KeywordError, MandatoryPrimaryKeyword, ReservedKeywordClass, BITPIX_KEYWORD,
    COMMENT_KEYWORD, CONTINUE_KEYWORD, END_KEYWORD, HIERARCH_KEYWORD, HISTORY_KEYWORD,
    NAXIS_KEYWORD, SIMPLE_KEYWORD, STANDARD_KEYWORD_WIDTH,
};
pub use parser::{parse_header_bytes, parse_header_cards, FitsHeader};
pub use value::{ComplexValue, HeaderValue, IntegerValue, RealValue};
