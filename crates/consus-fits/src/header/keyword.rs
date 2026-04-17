//! FITS header keyword single source of truth.
//!
//! ## Specification
//!
//! This module defines the authoritative keyword model for FITS header parsing.
//! All keyword classification, normalization, and mandatory-keyword handling
//! flows through this module.
//!
//! The implementation follows FITS Standard 4.0 card-keyword constraints:
//! - standard keywords occupy columns 1-8 and are ASCII uppercase by convention
//! - the `HIERARCH` convention permits extended hierarchical keyword names
//! - `CONTINUE` is a reserved keyword used for long-string continuation cards
//! - mandatory primary-header keywords are `SIMPLE`, `BITPIX`, and `NAXIS`
//!
//! ## Invariants
//!
//! - Standard keywords are trimmed, ASCII, non-empty, and at most 8 characters.
//! - Hierarchical keywords are represented canonically without the `HIERARCH`
//!   prefix and preserve segment order.
//! - Reserved keyword classification is centralized here.
//! - Mandatory primary-header keyword handling is defined here and re-used by
//!   higher-level parsers and validators.
//!
//! ## Architectural role
//!
//! This module is the SSOT for FITS keyword semantics. Parsing code should not
//! duplicate reserved-keyword tables or mandatory-keyword logic.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use core::fmt;

/// Canonical FITS keyword representation.
///
/// Standard FITS keywords are limited to 8 characters. Hierarchical keywords
/// are represented without the leading `HIERARCH` marker.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FitsKeyword {
    /// Standard FITS keyword occupying the 8-character keyword field.
    Standard(String),
    /// Hierarchical keyword represented as ordered path segments.
    Hierarchical(Vec<String>),
}

#[cfg(feature = "alloc")]
impl FitsKeyword {
    /// Construct a canonical keyword from a raw card keyword field or a raw
    /// hierarchical keyword payload.
    ///
    /// ## Accepted forms
    ///
    /// - `"SIMPLE"` -> `Standard("SIMPLE")`
    /// - `"NAXIS1"` -> `Standard("NAXIS1")`
    /// - `"HIERARCH ESO DET CHIP ID"` -> `Hierarchical(["ESO","DET","CHIP","ID"])`
    /// - `"ESO DET CHIP ID"` with `hierarchical = true` ->
    ///   `Hierarchical(["ESO","DET","CHIP","ID"])`
    ///
    /// ## Errors
    ///
    /// Returns [`KeywordError`] if the keyword is empty, contains non-ASCII
    /// bytes, exceeds the standard keyword width for non-hierarchical keywords,
    /// or contains invalid hierarchical segments.
    pub fn parse(raw: &str, hierarchical: bool) -> Result<Self, KeywordError> {
        if hierarchical {
            return Self::parse_hierarchical(raw);
        }

        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(KeywordError::EmptyKeyword);
        }
        if !trimmed.is_ascii() {
            return Err(KeywordError::NonAsciiKeyword);
        }
        if trimmed.len() > STANDARD_KEYWORD_WIDTH {
            return Err(KeywordError::KeywordTooLong {
                length: trimmed.len(),
                max: STANDARD_KEYWORD_WIDTH,
            });
        }
        if trimmed.contains(' ') {
            return Err(KeywordError::InvalidStandardKeyword {
                keyword: trimmed.into(),
            });
        }

        Ok(Self::Standard(trimmed.to_ascii_uppercase()))
    }

    /// Parse a hierarchical keyword.
    ///
    /// The input may include or omit the leading `HIERARCH` marker.
    pub fn parse_hierarchical(raw: &str) -> Result<Self, KeywordError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(KeywordError::EmptyKeyword);
        }
        if !trimmed.is_ascii() {
            return Err(KeywordError::NonAsciiKeyword);
        }

        let without_prefix = trimmed
            .strip_prefix(HIERARCH_KEYWORD)
            .map(str::trim_start)
            .unwrap_or(trimmed);

        if without_prefix.is_empty() {
            return Err(KeywordError::EmptyHierarchicalKeyword);
        }

        let mut segments = Vec::new();
        for segment in without_prefix.split('.') {
            for token in segment.split_whitespace() {
                if token.is_empty() {
                    continue;
                }
                if !is_valid_hierarchical_segment(token) {
                    return Err(KeywordError::InvalidHierarchicalSegment {
                        segment: token.into(),
                    });
                }
                segments.push(token.to_ascii_uppercase());
            }
        }

        if segments.is_empty() {
            return Err(KeywordError::EmptyHierarchicalKeyword);
        }

        Ok(Self::Hierarchical(segments))
    }

    /// Returns the canonical display form of the keyword.
    ///
    /// - Standard keywords return the keyword itself.
    /// - Hierarchical keywords return `HIERARCH ` followed by space-separated
    ///   segments.
    pub fn canonical_name(&self) -> String {
        match self {
            Self::Standard(name) => name.clone(),
            Self::Hierarchical(segments) => {
                let mut rendered = String::from(HIERARCH_KEYWORD);
                rendered.push(' ');
                rendered.push_str(&segments.join(" "));
                rendered
            }
        }
    }

    /// Returns the normalized lookup key used for equality and map indexing.
    ///
    /// This is identical to [`Self::canonical_name`] for standard keywords and
    /// hierarchical keywords.
    pub fn lookup_key(&self) -> String {
        self.canonical_name()
    }

    /// Returns `true` if this is a standard 8-character FITS keyword.
    pub fn is_standard(&self) -> bool {
        matches!(self, Self::Standard(_))
    }

    /// Returns `true` if this is a hierarchical keyword.
    pub fn is_hierarchical(&self) -> bool {
        matches!(self, Self::Hierarchical(_))
    }

    /// Returns the reserved keyword class for this keyword, if any.
    pub fn reserved_class(&self) -> Option<ReservedKeywordClass> {
        match self {
            Self::Standard(name) => classify_reserved_keyword(name),
            Self::Hierarchical(_) => None,
        }
    }

    /// Returns `true` if this keyword is one of the mandatory primary-header
    /// keywords.
    pub fn is_mandatory_primary(&self) -> bool {
        matches!(
            self.reserved_class(),
            Some(ReservedKeywordClass::MandatoryPrimary(_))
        )
    }

    /// Returns the mandatory primary keyword kind if this keyword is mandatory
    /// in a primary header.
    pub fn mandatory_primary_kind(&self) -> Option<MandatoryPrimaryKeyword> {
        match self.reserved_class() {
            Some(ReservedKeywordClass::MandatoryPrimary(kind)) => Some(kind),
            _ => None,
        }
    }

    /// Returns `true` if this keyword is `CONTINUE`.
    pub fn is_continue(&self) -> bool {
        matches!(
            self.reserved_class(),
            Some(ReservedKeywordClass::ContinueString)
        )
    }

    /// Returns `true` if this keyword is `END`.
    pub fn is_end(&self) -> bool {
        matches!(self.reserved_class(), Some(ReservedKeywordClass::End))
    }

    /// Returns `true` if this keyword is `HISTORY`.
    pub fn is_history(&self) -> bool {
        matches!(self.reserved_class(), Some(ReservedKeywordClass::History))
    }

    /// Returns `true` if this keyword is `COMMENT`.
    pub fn is_comment(&self) -> bool {
        matches!(self.reserved_class(), Some(ReservedKeywordClass::Comment))
    }

    /// Returns the standard keyword text if this is a standard keyword.
    pub fn as_standard(&self) -> Option<&str> {
        match self {
            Self::Standard(name) => Some(name.as_str()),
            Self::Hierarchical(_) => None,
        }
    }

    /// Returns the hierarchical segments if this is a hierarchical keyword.
    pub fn as_hierarchical_segments(&self) -> Option<&[String]> {
        match self {
            Self::Standard(_) => None,
            Self::Hierarchical(segments) => Some(segments.as_slice()),
        }
    }
}

#[cfg(feature = "alloc")]
impl fmt::Display for FitsKeyword {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard(name) => f.write_str(name),
            Self::Hierarchical(segments) => {
                f.write_str(HIERARCH_KEYWORD)?;
                f.write_str(" ")?;
                f.write_str(&segments.join(" "))
            }
        }
    }
}

/// Width of the standard FITS keyword field in card columns 1-8.
pub const STANDARD_KEYWORD_WIDTH: usize = 8;

/// Reserved keyword used for hierarchical keyword conventions.
pub const HIERARCH_KEYWORD: &str = "HIERARCH";

/// Reserved keyword used for long-string continuation cards.
pub const CONTINUE_KEYWORD: &str = "CONTINUE";

/// Reserved keyword terminating a FITS header.
pub const END_KEYWORD: &str = "END";

/// Reserved keyword for free-form comment cards.
pub const COMMENT_KEYWORD: &str = "COMMENT";

/// Reserved keyword for history cards.
pub const HISTORY_KEYWORD: &str = "HISTORY";

/// Mandatory primary-header keyword `SIMPLE`.
pub const SIMPLE_KEYWORD: &str = "SIMPLE";

/// Mandatory primary-header keyword `BITPIX`.
pub const BITPIX_KEYWORD: &str = "BITPIX";

/// Mandatory primary-header keyword `NAXIS`.
pub const NAXIS_KEYWORD: &str = "NAXIS";

/// Mandatory primary-header keyword classification.
///
/// FITS primary headers require `SIMPLE`, `BITPIX`, and `NAXIS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MandatoryPrimaryKeyword {
    /// `SIMPLE`
    Simple,
    /// `BITPIX`
    Bitpix,
    /// `NAXIS`
    Naxis,
}

impl MandatoryPrimaryKeyword {
    /// Returns the canonical FITS keyword spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Simple => SIMPLE_KEYWORD,
            Self::Bitpix => BITPIX_KEYWORD,
            Self::Naxis => NAXIS_KEYWORD,
        }
    }
}

/// Reserved FITS keyword classification.
///
/// This enum centralizes special handling for reserved keywords.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReservedKeywordClass {
    /// Mandatory primary-header keyword.
    MandatoryPrimary(MandatoryPrimaryKeyword),
    /// `CONTINUE`
    ContinueString,
    /// `END`
    End,
    /// `COMMENT`
    Comment,
    /// `HISTORY`
    History,
    /// `HIERARCH`
    HierarchMarker,
}

/// Keyword parsing and normalization errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeywordError {
    /// The keyword field is empty after trimming.
    EmptyKeyword,
    /// The hierarchical payload is empty after removing `HIERARCH`.
    EmptyHierarchicalKeyword,
    /// The keyword contains non-ASCII bytes.
    NonAsciiKeyword,
    /// A standard keyword exceeds 8 characters.
    KeywordTooLong {
        /// Actual keyword length.
        length: usize,
        /// Maximum permitted length.
        max: usize,
    },
    /// A standard keyword contains invalid characters or spacing.
    InvalidStandardKeyword {
        /// The offending keyword text.
        keyword: String,
    },
    /// A hierarchical segment is invalid.
    InvalidHierarchicalSegment {
        /// The offending segment.
        segment: String,
    },
}

impl fmt::Display for KeywordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyKeyword => f.write_str("empty FITS keyword"),
            Self::EmptyHierarchicalKeyword => {
                f.write_str("empty FITS hierarchical keyword payload")
            }
            Self::NonAsciiKeyword => f.write_str("FITS keyword must be ASCII"),
            Self::KeywordTooLong { length, max } => {
                write!(
                    f,
                    "FITS standard keyword length {length} exceeds maximum {max}"
                )
            }
            Self::InvalidStandardKeyword { keyword } => {
                write!(f, "invalid FITS standard keyword: {keyword}")
            }
            Self::InvalidHierarchicalSegment { segment } => {
                write!(f, "invalid FITS hierarchical keyword segment: {segment}")
            }
        }
    }
}

/// Returns the reserved keyword class for a canonical standard keyword.
///
/// The input is normalized case-insensitively.
pub fn classify_reserved_keyword(keyword: &str) -> Option<ReservedKeywordClass> {
    let normalized = keyword.trim().to_ascii_uppercase();
    match normalized.as_str() {
        SIMPLE_KEYWORD => Some(ReservedKeywordClass::MandatoryPrimary(
            MandatoryPrimaryKeyword::Simple,
        )),
        BITPIX_KEYWORD => Some(ReservedKeywordClass::MandatoryPrimary(
            MandatoryPrimaryKeyword::Bitpix,
        )),
        NAXIS_KEYWORD => Some(ReservedKeywordClass::MandatoryPrimary(
            MandatoryPrimaryKeyword::Naxis,
        )),
        CONTINUE_KEYWORD => Some(ReservedKeywordClass::ContinueString),
        END_KEYWORD => Some(ReservedKeywordClass::End),
        COMMENT_KEYWORD => Some(ReservedKeywordClass::Comment),
        HISTORY_KEYWORD => Some(ReservedKeywordClass::History),
        HIERARCH_KEYWORD => Some(ReservedKeywordClass::HierarchMarker),
        _ => None,
    }
}

/// Returns the mandatory primary keyword kind for a keyword, if any.
pub fn classify_mandatory_primary_keyword(keyword: &str) -> Option<MandatoryPrimaryKeyword> {
    match classify_reserved_keyword(keyword) {
        Some(ReservedKeywordClass::MandatoryPrimary(kind)) => Some(kind),
        _ => None,
    }
}

/// Returns the canonical ordered list of mandatory primary-header keywords.
pub const fn mandatory_primary_keywords() -> [MandatoryPrimaryKeyword; 3] {
    [
        MandatoryPrimaryKeyword::Simple,
        MandatoryPrimaryKeyword::Bitpix,
        MandatoryPrimaryKeyword::Naxis,
    ]
}

/// Returns `true` if the keyword is mandatory in a primary header.
pub fn is_mandatory_primary_keyword(keyword: &str) -> bool {
    classify_mandatory_primary_keyword(keyword).is_some()
}

/// Returns `true` if the keyword is reserved by this module's SSOT table.
pub fn is_reserved_keyword(keyword: &str) -> bool {
    classify_reserved_keyword(keyword).is_some()
}

/// Returns `true` if a token is a valid hierarchical keyword segment.
///
/// This implementation accepts visible ASCII except `=` and preserves the
/// common FITS `HIERARCH` convention used by ESO and related tooling.
pub fn is_valid_hierarchical_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment.is_ascii()
        && segment
            .bytes()
            .all(|byte| matches!(byte, b'!'..=b'<' | b'>'..=b'~'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_keyword_simple() {
        let keyword = FitsKeyword::parse("SIMPLE  ", false).unwrap();
        assert_eq!(keyword.as_standard(), Some("SIMPLE"));
        assert!(keyword.is_standard());
        assert!(keyword.is_mandatory_primary());
        assert_eq!(
            keyword.mandatory_primary_kind(),
            Some(MandatoryPrimaryKeyword::Simple)
        );
    }

    #[test]
    fn parses_standard_keyword_naxis1() {
        let keyword = FitsKeyword::parse("NAXIS1", false).unwrap();
        assert_eq!(keyword.as_standard(), Some("NAXIS1"));
        assert!(!keyword.is_mandatory_primary());
        assert_eq!(keyword.reserved_class(), None);
    }

    #[test]
    fn rejects_standard_keyword_longer_than_eight_columns() {
        let error = FitsKeyword::parse("TOO_LONG9", false).unwrap_err();
        assert_eq!(
            error,
            KeywordError::KeywordTooLong {
                length: 9,
                max: STANDARD_KEYWORD_WIDTH,
            }
        );
    }

    #[test]
    fn rejects_standard_keyword_with_internal_space() {
        let error = FitsKeyword::parse("NA XIS", false).unwrap_err();
        assert_eq!(
            error,
            KeywordError::InvalidStandardKeyword {
                keyword: String::from("NA XIS"),
            }
        );
    }

    #[test]
    fn parses_hierarch_keyword_with_prefix() {
        let keyword = FitsKeyword::parse("HIERARCH ESO DET CHIP ID", true).unwrap();
        assert!(keyword.is_hierarchical());
        assert_eq!(
            keyword.as_hierarchical_segments().unwrap(),
            &[
                String::from("ESO"),
                String::from("DET"),
                String::from("CHIP"),
                String::from("ID"),
            ]
        );
        assert_eq!(keyword.canonical_name(), "HIERARCH ESO DET CHIP ID");
    }

    #[test]
    fn parses_hierarch_keyword_without_prefix() {
        let keyword = FitsKeyword::parse_hierarchical("eso det read noise").unwrap();
        assert_eq!(
            keyword.as_hierarchical_segments().unwrap(),
            &[
                String::from("ESO"),
                String::from("DET"),
                String::from("READ"),
                String::from("NOISE"),
            ]
        );
        assert_eq!(keyword.to_string(), "HIERARCH ESO DET READ NOISE");
    }

    #[test]
    fn parses_hierarch_keyword_with_dot_segments() {
        let keyword = FitsKeyword::parse_hierarchical("HIERARCH INSTRUME.DETECTOR.GAIN").unwrap();
        assert_eq!(
            keyword.as_hierarchical_segments().unwrap(),
            &[
                String::from("INSTRUME"),
                String::from("DETECTOR"),
                String::from("GAIN"),
            ]
        );
    }

    #[test]
    fn rejects_empty_hierarch_payload() {
        let error = FitsKeyword::parse_hierarchical("HIERARCH").unwrap_err();
        assert_eq!(error, KeywordError::EmptyHierarchicalKeyword);
    }

    #[test]
    fn continue_keyword_is_reserved() {
        let keyword = FitsKeyword::parse("CONTINUE", false).unwrap();
        assert!(keyword.is_continue());
        assert_eq!(
            keyword.reserved_class(),
            Some(ReservedKeywordClass::ContinueString)
        );
    }

    #[test]
    fn end_comment_and_history_are_reserved() {
        let end = FitsKeyword::parse("END", false).unwrap();
        let comment = FitsKeyword::parse("COMMENT", false).unwrap();
        let history = FitsKeyword::parse("HISTORY", false).unwrap();

        assert!(end.is_end());
        assert!(comment.is_comment());
        assert!(history.is_history());
    }

    #[test]
    fn mandatory_primary_keyword_table_is_ordered() {
        let mandatory = mandatory_primary_keywords();
        assert_eq!(mandatory[0], MandatoryPrimaryKeyword::Simple);
        assert_eq!(mandatory[1], MandatoryPrimaryKeyword::Bitpix);
        assert_eq!(mandatory[2], MandatoryPrimaryKeyword::Naxis);
        assert_eq!(mandatory[0].as_str(), "SIMPLE");
        assert_eq!(mandatory[1].as_str(), "BITPIX");
        assert_eq!(mandatory[2].as_str(), "NAXIS");
    }

    #[test]
    fn reserved_keyword_classification_is_case_insensitive() {
        assert_eq!(
            classify_reserved_keyword("simple"),
            Some(ReservedKeywordClass::MandatoryPrimary(
                MandatoryPrimaryKeyword::Simple
            ))
        );
        assert_eq!(
            classify_reserved_keyword("continue"),
            Some(ReservedKeywordClass::ContinueString)
        );
        assert_eq!(
            classify_reserved_keyword("history"),
            Some(ReservedKeywordClass::History)
        );
    }

    #[test]
    fn fits_standard_reference_example_keywords_are_classified() {
        let simple = FitsKeyword::parse("SIMPLE", false).unwrap();
        let bitpix = FitsKeyword::parse("BITPIX", false).unwrap();
        let naxis = FitsKeyword::parse("NAXIS", false).unwrap();
        let extend = FitsKeyword::parse("EXTEND", false).unwrap();
        let end = FitsKeyword::parse("END", false).unwrap();

        assert!(simple.is_mandatory_primary());
        assert!(bitpix.is_mandatory_primary());
        assert!(naxis.is_mandatory_primary());
        assert_eq!(extend.reserved_class(), None);
        assert!(end.is_end());
    }

    #[test]
    fn hierarchical_segment_validation_accepts_common_reference_forms() {
        assert!(is_valid_hierarchical_segment("ESO"));
        assert!(is_valid_hierarchical_segment("DET"));
        assert!(is_valid_hierarchical_segment("CHIP1"));
        assert!(is_valid_hierarchical_segment("READ-NOISE"));
        assert!(is_valid_hierarchical_segment("A_B"));
    }

    #[test]
    fn hierarchical_segment_validation_rejects_invalid_forms() {
        assert!(!is_valid_hierarchical_segment(""));
        assert!(!is_valid_hierarchical_segment("A=B"));
        assert!(!is_valid_hierarchical_segment("μ"));
        assert!(!is_valid_hierarchical_segment("A B"));
    }
}
