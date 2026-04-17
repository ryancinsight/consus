//! FITS header card model and 80-character card parsing.
//!
//! ## Specification
//!
//! A FITS header is a sequence of fixed-width 80-byte card images. Each card
//! contains:
//! - a keyword field in columns 1-8 for standard keywords, or a `HIERARCH`
//!   marker followed by an extended keyword payload
//! - an optional value indicator `= `
//! - an optional value/comment field
//!
//! This module is the authoritative card-level boundary for `consus-fits`.
//! Keyword normalization is delegated to `keyword.rs`. Scalar value parsing is
//! delegated to `value.rs`.
//!
//! ## Supported foundational features
//!
//! - standard 8-character keywords
//! - hierarchical keywords via `HIERARCH`
//! - commentary cards (`COMMENT`, `HISTORY`)
//! - `END` cards
//! - `CONTINUE` cards for long-string continuation
//! - value/comment splitting for ordinary value cards
//!
//! ## Invariants
//!
//! - `raw` is always exactly 80 bytes.
//! - `keyword` is the canonical parsed keyword for the card.
//! - `has_value` is true iff the card contains an explicit FITS value field.
//! - `comment` excludes the separating slash for value cards.
//! - `continue_fragment` is populated only for `CONTINUE` cards carrying a
//!   string fragment.
//! - `append_string_fragment` mutates only string-valued cards.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::string::{String, ToString};

use consus_core::{Error, Result};

use super::keyword::{FitsKeyword, HIERARCH_KEYWORD, KeywordError};
use super::value::HeaderValue;

/// Exact byte width of a FITS card image.
pub const FITS_CARD_LEN: usize = 80;

/// Parsed FITS header card.
///
/// This type is value-semantic and stores both the raw 80-byte card image and
/// the canonical parsed fields derived from it.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitsCard {
    raw: [u8; FITS_CARD_LEN],
    keyword: FitsKeyword,
    has_value: bool,
    value: Option<HeaderValue>,
    comment: Option<String>,
    continue_fragment: Option<String>,
}

#[cfg(feature = "alloc")]
impl FitsCard {
    /// Construct a parsed FITS card from canonical fields.
    pub fn new(
        raw: [u8; FITS_CARD_LEN],
        keyword: FitsKeyword,
        has_value: bool,
        value: Option<HeaderValue>,
        comment: Option<String>,
        continue_fragment: Option<String>,
    ) -> Self {
        Self {
            raw,
            keyword,
            has_value,
            value,
            comment,
            continue_fragment,
        }
    }

    /// Parse a FITS card from an 80-byte card image.
    ///
    /// ## Errors
    ///
    /// Returns `Error::InvalidFormat` if:
    /// - the input length is not exactly 80 bytes
    /// - the card is not valid ASCII
    /// - the keyword is invalid
    /// - the value field is syntactically invalid
    pub fn parse(raw: &[u8]) -> Result<Self> {
        if raw.len() != FITS_CARD_LEN {
            return invalid_format("FITS card image must be exactly 80 bytes");
        }
        if !raw.is_ascii() {
            return invalid_format("FITS card image must be ASCII");
        }

        let mut raw_array = [b' '; FITS_CARD_LEN];
        raw_array.copy_from_slice(raw);

        let text = match core::str::from_utf8(raw) {
            Ok(value) => value,
            Err(_) => {
                return invalid_format("FITS card image must be valid UTF-8 compatible ASCII");
            }
        };

        if text.starts_with(HIERARCH_KEYWORD) {
            return Self::parse_hierarch(raw_array, text);
        }

        let keyword_field = &text[..8];
        let keyword = parse_keyword(keyword_field, false)?;

        if keyword.is_end() {
            return Ok(Self::new(raw_array, keyword, false, None, None, None));
        }

        if keyword.is_comment() || keyword.is_history() {
            let comment = trim_right_spaces(&text[8..]).trim_start();
            let comment = if comment.is_empty() {
                None
            } else {
                Some(comment.to_string())
            };
            return Ok(Self::new(raw_array, keyword, false, None, comment, None));
        }

        if keyword.is_continue() {
            return Self::parse_continue(raw_array, text, keyword);
        }

        if text.as_bytes()[8] == b'=' && text.as_bytes()[9] == b' ' {
            let (value, comment) = parse_value_and_comment(&text[10..])?;
            return Ok(Self::new(
                raw_array,
                keyword,
                true,
                Some(value),
                comment,
                None,
            ));
        }

        let trailing = trim_right_spaces(&text[8..]);
        let comment = if trailing.is_empty() {
            None
        } else {
            Some(trailing.to_string())
        };

        Ok(Self::new(raw_array, keyword, false, None, comment, None))
    }

    fn parse_hierarch(raw: [u8; FITS_CARD_LEN], text: &str) -> Result<Self> {
        let after_prefix = &text[HIERARCH_KEYWORD.len()..];
        let trimmed = after_prefix.trim_start();

        let Some(eq_index) = trimmed.find("= ") else {
            let keyword = parse_keyword(trimmed, true)?;
            return Ok(Self::new(raw, keyword, false, None, None, None));
        };

        let keyword_text = trimmed[..eq_index].trim_end();
        let value_text = &trimmed[eq_index + 1..];
        let keyword = parse_keyword(keyword_text, true)?;
        let (value, comment) = parse_value_and_comment(value_text)?;
        Ok(Self::new(raw, keyword, true, Some(value), comment, None))
    }

    fn parse_continue(raw: [u8; FITS_CARD_LEN], text: &str, keyword: FitsKeyword) -> Result<Self> {
        let payload = trim_right_spaces(&text[8..]).trim_start();
        let (fragment, comment) = parse_continue_payload(payload)?;
        Ok(Self::new(
            raw,
            keyword,
            false,
            Some(HeaderValue::String(fragment.clone())),
            comment,
            Some(fragment),
        ))
    }

    /// Return the original 80-byte card image.
    pub const fn raw(&self) -> &[u8; FITS_CARD_LEN] {
        &self.raw
    }

    /// Return the original 80-byte card image as a string slice.
    pub fn raw_str(&self) -> &str {
        match core::str::from_utf8(&self.raw) {
            Ok(value) => value,
            Err(_) => unreachable!("FITS card image must remain ASCII"),
        }
    }

    /// Return the canonical parsed keyword.
    pub const fn keyword(&self) -> &FitsKeyword {
        &self.keyword
    }

    /// Return whether the card contains a FITS value field.
    pub const fn has_value(&self) -> bool {
        self.has_value
    }

    /// Return the parsed value, if present.
    pub const fn value(&self) -> Option<&HeaderValue> {
        self.value.as_ref()
    }

    /// Return the parsed comment text, if present.
    pub fn comment(&self) -> Option<&str> {
        self.comment.as_deref()
    }

    /// Return whether this is a `CONTINUE` card.
    pub fn is_continue(&self) -> bool {
        self.keyword.is_continue()
    }

    /// Return whether this is a hierarchical keyword card.
    pub fn is_hierarchical(&self) -> bool {
        self.keyword.is_hierarchical()
    }

    /// Return whether this is a commentary card.
    pub fn is_commentary(&self) -> bool {
        self.keyword.is_comment() || self.keyword.is_history()
    }

    /// Return whether this card terminates the header.
    pub fn is_end(&self) -> bool {
        self.keyword.is_end()
    }

    /// Return whether this card is one of the mandatory primary-header cards.
    pub fn is_primary_mandatory(&self) -> bool {
        self.keyword.is_mandatory_primary()
    }

    /// Return the `CONTINUE` string fragment, if this is a continuation card.
    pub fn continue_fragment(&self) -> Option<&str> {
        self.continue_fragment.as_deref()
    }

    /// Append a `CONTINUE` fragment to a preceding string-valued card.
    ///
    /// If the current string ends with `&`, the trailing ampersand is removed
    /// before appending the fragment, following the FITS long-string convention.
    pub fn append_string_fragment(&mut self, fragment: &str) -> Result<()> {
        let Some(HeaderValue::String(value)) = self.value.as_mut() else {
            return invalid_format("CONTINUE requires a preceding string-valued card");
        };

        if value.ends_with('&') {
            value.pop();
        }
        value.push_str(fragment);
        Ok(())
    }

    /// Consume the card and return its raw image.
    pub fn into_raw(self) -> [u8; FITS_CARD_LEN] {
        self.raw
    }
}

#[cfg(feature = "alloc")]
fn parse_keyword(raw: &str, hierarchical: bool) -> Result<FitsKeyword> {
    FitsKeyword::parse(raw, hierarchical).map_err(keyword_error_to_invalid_format)
}

#[cfg(feature = "alloc")]
fn parse_value_and_comment(field: &str) -> Result<(HeaderValue, Option<String>)> {
    let trimmed = trim_right_spaces(field);
    let (value_text, comment) = split_value_comment(trimmed)?;
    let value = HeaderValue::parse(value_text)?;
    Ok((value, comment))
}

#[cfg(feature = "alloc")]
fn parse_continue_payload(payload: &str) -> Result<(String, Option<String>)> {
    let (value_text, comment) = split_value_comment(payload)?;
    let value = HeaderValue::parse(value_text)?;
    match value {
        HeaderValue::String(fragment) => Ok((fragment, comment)),
        _ => invalid_format("CONTINUE card must contain a FITS string literal"),
    }
}

#[cfg(feature = "alloc")]
fn split_value_comment(field: &str) -> Result<(&str, Option<String>)> {
    let mut in_string = false;
    let mut index = 0usize;
    let bytes = field.as_bytes();

    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'\'' {
            if in_string && index + 1 < bytes.len() && bytes[index + 1] == b'\'' {
                index += 2;
                continue;
            }
            in_string = !in_string;
            index += 1;
            continue;
        }

        if !in_string && byte == b'/' {
            let value = trim_right_spaces(&field[..index]);
            let comment = trim_right_spaces(&field[index + 1..])
                .trim_start()
                .to_string();
            let comment = if comment.is_empty() {
                None
            } else {
                Some(comment)
            };
            return Ok((value, comment));
        }

        index += 1;
    }

    Ok((trim_right_spaces(field), None))
}

fn trim_right_spaces(value: &str) -> &str {
    value.trim_end_matches(' ')
}

#[cfg(feature = "alloc")]
fn keyword_error_to_invalid_format(error: KeywordError) -> Error {
    Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: error.to_string(),
    }
}

#[cfg(feature = "alloc")]
fn invalid_format<T>(message: &str) -> Result<T> {
    Err(Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: message.to_string(),
    })
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::header::keyword::FitsKeyword;
    use crate::header::value::HeaderValue;

    fn card_bytes(text: &str) -> [u8; FITS_CARD_LEN] {
        assert!(text.len() <= FITS_CARD_LEN);
        let mut raw = [b' '; FITS_CARD_LEN];
        raw[..text.len()].copy_from_slice(text.as_bytes());
        raw
    }

    #[test]
    fn parses_simple_reference_card() {
        let raw = card_bytes("SIMPLE  =                    T / file does conform to FITS standard");
        let card = FitsCard::parse(&raw).unwrap();

        assert_eq!(
            card.keyword(),
            &FitsKeyword::parse("SIMPLE", false).unwrap()
        );
        assert!(card.has_value());
        assert_eq!(card.value(), Some(&HeaderValue::Logical(true)));
        assert_eq!(card.comment(), Some("file does conform to FITS standard"));
        assert!(card.is_primary_mandatory());
    }

    #[test]
    fn parses_bitpix_reference_card() {
        let raw = card_bytes("BITPIX  =                   16 / number of bits per data pixel");
        let card = FitsCard::parse(&raw).unwrap();

        assert_eq!(
            card.keyword(),
            &FitsKeyword::parse("BITPIX", false).unwrap()
        );
        match card.value() {
            Some(HeaderValue::Integer(value)) => assert_eq!(value.to_i64().unwrap(), 16),
            other => panic!("expected integer value, found {other:?}"),
        }
    }

    #[test]
    fn parses_naxis_reference_card() {
        let raw = card_bytes("NAXIS   =                    0 / number of data axes");
        let card = FitsCard::parse(&raw).unwrap();

        assert_eq!(card.keyword(), &FitsKeyword::parse("NAXIS", false).unwrap());
        match card.value() {
            Some(HeaderValue::Integer(value)) => assert_eq!(value.to_i64().unwrap(), 0),
            other => panic!("expected integer value, found {other:?}"),
        }
    }

    #[test]
    fn parses_commentary_card() {
        let raw =
            card_bytes("COMMENT   This FITS header uses commentary cards for descriptive text");
        let card = FitsCard::parse(&raw).unwrap();

        assert!(card.is_commentary());
        assert!(!card.has_value());
        assert_eq!(
            card.comment(),
            Some("This FITS header uses commentary cards for descriptive text")
        );
    }

    #[test]
    fn parses_end_card() {
        let raw = card_bytes("END");
        let card = FitsCard::parse(&raw).unwrap();

        assert!(card.is_end());
        assert!(!card.has_value());
        assert_eq!(card.value(), None);
    }

    #[test]
    fn parses_hierarch_card() {
        let raw = card_bytes("HIERARCH ESO DET CHIP1 ID = 'CCD-42' / hierarchical keyword example");
        let card = FitsCard::parse(&raw).unwrap();

        assert!(card.is_hierarchical());
        match card.value() {
            Some(HeaderValue::String(value)) => assert_eq!(value, "CCD-42"),
            other => panic!("expected string value, found {other:?}"),
        }
        assert_eq!(card.comment(), Some("hierarchical keyword example"));
    }

    #[test]
    fn parses_continue_card_fragment() {
        let raw = card_bytes("CONTINUE  ' across multiple cards and remains ordered.'");
        let card = FitsCard::parse(&raw).unwrap();

        assert!(card.is_continue());
        assert_eq!(
            card.continue_fragment(),
            Some(" across multiple cards and remains ordered.")
        );
        assert_eq!(
            card.value(),
            Some(&HeaderValue::String(
                " across multiple cards and remains ordered.".to_string()
            ))
        );
    }

    #[test]
    fn append_string_fragment_removes_trailing_ampersand() {
        let raw = card_bytes("OBJECT  = 'This is a long string value that continues &'");
        let mut card = FitsCard::parse(&raw).unwrap();

        card.append_string_fragment(" across multiple cards.")
            .unwrap();

        assert_eq!(
            card.value(),
            Some(&HeaderValue::String(
                "This is a long string value that continues  across multiple cards.".to_string()
            ))
        );
    }

    #[test]
    fn append_string_fragment_rejects_non_string_card() {
        let raw = card_bytes("BITPIX  =                    8");
        let mut card = FitsCard::parse(&raw).unwrap();

        let error = card.append_string_fragment("x").unwrap_err();
        assert!(matches!(error, Error::InvalidFormat { .. }));
    }

    #[test]
    fn rejects_non_ascii_card() {
        let mut raw = [b' '; FITS_CARD_LEN];
        raw[0] = 0xFF;
        let error = FitsCard::parse(&raw).unwrap_err();
        assert!(matches!(error, Error::InvalidFormat { .. }));
    }

    #[test]
    fn rejects_wrong_card_length() {
        let error = FitsCard::parse(b"SIMPLE").unwrap_err();
        assert!(matches!(error, Error::InvalidFormat { .. }));
    }
}
