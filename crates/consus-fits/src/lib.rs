//! # consus-fits
//!
//! Pure-Rust FITS format support for the Consus scientific storage workspace.
//!
//! ## Scope
//!
//! This crate defines the authoritative FITS-facing abstractions for:
//! - header card parsing and normalization
//! - canonical FITS keyword handling
//! - FITS scalar/header value modeling
//! - FITS file scanning and HDU indexing
//! - FITS image and table metadata extraction
//! - FITS file-level integration with `consus-core` traits
//!
//! ## Architecture
//!
//! ```text
//! consus-fits
//! ├── header/         # Card parsing, keyword SSOT, header value model, validation
//! ├── types/          # BITPIX mapping and HDU classification
//! ├── datastructure/  # 2880-byte blocking, logical records, payload spans
//! ├── hdu/            # HDU descriptors, sequencing, primary/extension semantics
//! ├── image/          # Image metadata extraction and raw image access
//! ├── table/          # ASCII/Binary table metadata extraction and raw row access
//! ├── file/           # FITS file wrapper and consus-core trait integration
//! └── validation/     # File/HDU/path invariant validation
//! ```
//!
//! ## Design constraints
//!
//! - The crate root is a facade only.
//! - FITS keyword handling is centralized under `header`.
//! - Public type mappings are defined in exactly one authoritative module.
//! - FITS file traversal is deterministic and index-stable.
//! - The crate remains `no_std`-compatible outside `alloc`-gated functionality.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

/// FITS header parsing, keyword handling, and header value abstractions.
pub mod header;

/// FITS `BITPIX` mappings and HDU classification types.
pub mod types;

/// FITS logical-record, blocking, and payload layout descriptors.
pub mod datastructure;

/// FITS Header/Data Unit descriptors and sequencing.
pub mod hdu;

/// FITS image metadata extraction and raw image access.
pub mod image;

/// FITS ASCII and binary table metadata extraction and raw row access.
pub mod table;

/// FITS file wrapper and `consus-core` trait integration.
pub mod file;

/// FITS facade integration for `consus-core` and top-level `consus`.
pub mod fits;

/// FITS file-, HDU-, and path-level invariant validation.
pub mod validation;

pub use datastructure::{
    FitsBlockAlignment, FitsDataSpan, FitsHeaderBlock, FitsHeaderCardCount, FitsLogicalRecord,
};
pub use file::FitsFile;

pub use hdu::{FitsHdu, FitsHduIndex, FitsHduKind, FitsHduSequence};
pub use header::{
    card::{FITS_CARD_LEN, FitsCard},
    keyword::{
        BITPIX_KEYWORD, COMMENT_KEYWORD, CONTINUE_KEYWORD, END_KEYWORD, FitsKeyword,
        HIERARCH_KEYWORD, HISTORY_KEYWORD, KeywordError, MandatoryPrimaryKeyword, NAXIS_KEYWORD,
        ReservedKeywordClass, SIMPLE_KEYWORD, STANDARD_KEYWORD_WIDTH,
        classify_mandatory_primary_keyword, classify_reserved_keyword,
        is_mandatory_primary_keyword, is_reserved_keyword, is_valid_hierarchical_segment,
        mandatory_primary_keywords,
    },
    parser::{FitsHeader, parse_header_bytes, parse_header_cards},
    value::{ComplexValue, HeaderValue, IntegerValue, RealValue},
};
pub use image::FitsImageDescriptor;
pub use table::{FitsAsciiTableDescriptor, FitsBinaryTableDescriptor, FitsColumnValue, FitsTableColumn};
pub use types::{BinaryFormatCode, Bitpix, HduType, tform_to_datatype};
