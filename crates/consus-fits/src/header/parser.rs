//! FITS header parser and validator.
//!
//! ## Specification
//!
//! This module implements foundational FITS header parsing for ordered 80-byte
//! card images. It supports:
//! - standard keyword cards
//! - commentary cards (`COMMENT`, `HISTORY`)
//! - hierarchical keywords via the `HIERARCH` convention
//! - long-string continuation via `CONTINUE`
//! - mandatory primary-header validation for `SIMPLE`, `BITPIX`, and `NAXIS`
//!
//! ## Architectural role
//!
//! Keyword normalization is delegated to `keyword.rs` and scalar value parsing
//! is delegated to `value.rs`. This module owns only:
//! - card-sequence parsing
//! - header assembly
//! - continuation stitching
//! - mandatory primary-header validation
//!
//! ## Invariants
//!
//! - Input card images are exactly 80 bytes.
//! - Parsing stops at the first `END` card.
//! - `CONTINUE` cards attach only to a preceding string-valued card.
//! - Mandatory primary-header keywords appear in order: `SIMPLE`, `BITPIX`,
//!   `NAXIS`.
//! - `BITPIX` is one of `8`, `16`, `32`, `64`, `-32`, `-64`.
//! - `NAXIS` is a non-negative integer.

#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec::Vec};

use consus_core::{Error, Result};

use super::card::{FITS_CARD_LEN, FitsCard};
use super::keyword::{FitsKeyword, MandatoryPrimaryKeyword, ReservedKeywordClass};
use super::value::{HeaderValue, IntegerValue};

/// Ordered FITS header representation.
///
/// The header stores semantic cards only. The terminating `END` card is not
/// retained as a semantic entry.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitsHeader {
    cards: Vec<FitsCard>,
}

#[cfg(feature = "alloc")]
impl FitsHeader {
    /// Construct a header from ordered semantic cards.
    pub fn new(cards: Vec<FitsCard>) -> Self {
        Self { cards }
    }

    /// Return the ordered semantic cards.
    pub fn cards(&self) -> &[FitsCard] {
        &self.cards
    }

    /// Return the number of semantic cards.
    pub fn len(&self) -> usize {
        self.cards.len()
    }

    /// Return whether the header contains no semantic cards.
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }

    /// Return the first card matching `keyword`.
    pub fn get(&self, keyword: &FitsKeyword) -> Option<&FitsCard> {
        self.cards.iter().find(|card| card.keyword() == keyword)
    }

    /// Return the first card matching a standard keyword.
    pub fn get_standard(&self, keyword: &str) -> Option<&FitsCard> {
        let keyword = FitsKeyword::parse(keyword, false).ok()?;
        self.get(&keyword)
    }

    /// Return the parsed `BITPIX` integer value.
    pub fn bitpix_i64(&self) -> Option<i64> {
        self.get_standard("BITPIX")
            .and_then(FitsCard::value)
            .and_then(|value| match value {
                HeaderValue::Integer(integer) => integer.to_i64().ok(),
                _ => None,
            })
    }

    /// Return the parsed `NAXIS` integer value.
    pub fn naxis_i64(&self) -> Option<i64> {
        self.get_standard("NAXIS")
            .and_then(FitsCard::value)
            .and_then(|value| match value {
                HeaderValue::Integer(integer) => integer.to_i64().ok(),
                _ => None,
            })
    }
}

/// Parse a FITS header from a contiguous byte slice containing one or more
/// 80-byte card images.
///
/// Parsing stops at the first `END` card.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` when:
/// - the input length is not a multiple of 80
/// - a card image is not valid ASCII
/// - a card cannot be parsed
/// - `END` is missing
/// - mandatory primary-header keywords are missing or out of order
#[cfg(feature = "alloc")]
pub fn parse_header_bytes(bytes: &[u8]) -> Result<FitsHeader> {
    if bytes.len() % FITS_CARD_LEN != 0 {
        return invalid_format("FITS header byte length is not a multiple of 80");
    }

    let mut parsed_cards = Vec::new();
    let mut saw_end = false;

    for chunk in bytes.chunks_exact(FITS_CARD_LEN) {
        let card = FitsCard::parse(chunk)?;

        if card.keyword().is_end() {
            saw_end = true;
            break;
        }

        if card.is_continue() {
            attach_continue_card(&mut parsed_cards, &card)?;
        } else {
            parsed_cards.push(card);
        }
    }

    if !saw_end {
        return invalid_format("FITS header is missing END card");
    }

    validate_primary_mandatory_keywords(&parsed_cards)?;
    Ok(FitsHeader::new(parsed_cards))
}

/// Parse a FITS header from an iterator of 80-byte card images.
///
/// Parsing stops at the first `END` card.
///
/// ## Errors
///
/// Returns `Error::InvalidFormat` when:
/// - any card image is not exactly 80 bytes
/// - a card cannot be parsed
/// - `END` is missing
/// - mandatory primary-header keywords are missing or out of order
#[cfg(feature = "alloc")]
pub fn parse_header_cards<'a, I>(cards: I) -> Result<FitsHeader>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let mut parsed_cards = Vec::new();
    let mut saw_end = false;

    for raw in cards {
        let card = FitsCard::parse(raw)?;

        if card.keyword().is_end() {
            saw_end = true;
            break;
        }

        if card.is_continue() {
            attach_continue_card(&mut parsed_cards, &card)?;
        } else {
            parsed_cards.push(card);
        }
    }

    if !saw_end {
        return invalid_format("FITS header is missing END card");
    }

    validate_primary_mandatory_keywords(&parsed_cards)?;
    Ok(FitsHeader::new(parsed_cards))
}

#[cfg(feature = "alloc")]
fn attach_continue_card(cards: &mut Vec<FitsCard>, continue_card: &FitsCard) -> Result<()> {
    let previous = cards.last_mut().ok_or_else(|| {
        invalid_format_error("CONTINUE card cannot appear before a string-valued card")
    })?;

    let fragment = continue_card.continue_fragment().ok_or_else(|| {
        invalid_format_error("CONTINUE card does not contain a valid string continuation fragment")
    })?;

    previous.append_string_fragment(fragment)
}

#[cfg(feature = "alloc")]
fn validate_primary_mandatory_keywords(cards: &[FitsCard]) -> Result<()> {
    let expected = [
        MandatoryPrimaryKeyword::Simple,
        MandatoryPrimaryKeyword::Bitpix,
        MandatoryPrimaryKeyword::Naxis,
    ];

    if cards.len() < expected.len() {
        return invalid_format("FITS primary header is missing mandatory keywords");
    }

    for (index, mandatory) in expected.into_iter().enumerate() {
        let card = &cards[index];
        match card.keyword().reserved_class() {
            Some(ReservedKeywordClass::MandatoryPrimary(found)) if found == mandatory => {}
            _ => {
                return invalid_format(&format!(
                    "mandatory FITS primary-header keyword out of order: expected {} at card index {}, found {}",
                    mandatory.as_str(),
                    index,
                    card.keyword().canonical_name()
                ));
            }
        }

        validate_mandatory_card_value(card, mandatory)?;
    }

    Ok(())
}

#[cfg(feature = "alloc")]
fn validate_mandatory_card_value(card: &FitsCard, keyword: MandatoryPrimaryKeyword) -> Result<()> {
    match keyword {
        MandatoryPrimaryKeyword::Simple => match card.value() {
            Some(HeaderValue::Logical(true)) => Ok(()),
            Some(HeaderValue::Logical(false)) => {
                invalid_format("SIMPLE must be the logical value T for a standard FITS primary HDU")
            }
            _ => invalid_format("SIMPLE must contain a logical value"),
        },
        MandatoryPrimaryKeyword::Bitpix => match card.value() {
            Some(HeaderValue::Integer(value)) if is_valid_bitpix(value) => Ok(()),
            Some(HeaderValue::Integer(_)) => {
                invalid_format("BITPIX must be one of 8, 16, 32, 64, -32, or -64")
            }
            _ => invalid_format("BITPIX must contain an integer value"),
        },
        MandatoryPrimaryKeyword::Naxis => match card.value() {
            Some(HeaderValue::Integer(value)) => {
                let parsed = value.to_i64()?;
                if parsed >= 0 {
                    Ok(())
                } else {
                    invalid_format("NAXIS must be a non-negative integer")
                }
            }
            _ => invalid_format("NAXIS must contain an integer value"),
        },
    }
}

#[cfg(feature = "alloc")]
fn is_valid_bitpix(value: &IntegerValue) -> bool {
    matches!(value.to_i64(), Ok(8 | 16 | 32 | 64 | -32 | -64))
}

#[cfg(feature = "alloc")]
fn invalid_format<T>(message: &str) -> Result<T> {
    Err(invalid_format_error(message))
}

#[cfg(feature = "alloc")]
fn invalid_format_error(message: &str) -> Error {
    Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: String::from(message),
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    fn card(text: &str) -> [u8; 80] {
        assert!(text.len() <= 80);
        let mut buf = [b' '; 80];
        buf[..text.len()].copy_from_slice(text.as_bytes());
        buf
    }

    fn header_bytes(cards: &[&str]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for card_text in cards {
            bytes.extend_from_slice(&card(card_text));
        }
        bytes
    }

    #[test]
    fn parses_minimal_primary_header() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T / file conforms to FITS standard",
            "BITPIX  =                   16 / number of bits per data pixel",
            "NAXIS   =                    0 / number of data axes",
            "END",
        ]);

        let header = parse_header_bytes(&bytes).expect("header should parse");
        assert_eq!(header.len(), 3);
        assert_eq!(header.bitpix_i64(), Some(16));
        assert_eq!(header.naxis_i64(), Some(0));
    }

    #[test]
    fn rejects_missing_end_card() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
        ]);

        let error = parse_header_bytes(&bytes).expect_err("missing END must fail");
        let message = error.to_string();
        assert!(message.contains("missing END"));
    }

    #[test]
    fn rejects_non_multiple_of_eighty() {
        let bytes = b"SIMPLE".to_vec();
        let error = parse_header_bytes(&bytes).expect_err("invalid length must fail");
        let message = error.to_string();
        assert!(message.contains("multiple of 80"));
    }

    #[test]
    fn rejects_out_of_order_mandatory_keywords() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "NAXIS   =                    0",
            "BITPIX  =                    8",
            "END",
        ]);

        let error = parse_header_bytes(&bytes).expect_err("out-of-order mandatory cards must fail");
        let message = error.to_string();
        assert!(message.contains("out of order"));
    }

    #[test]
    fn rejects_invalid_bitpix_value() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                   12",
            "NAXIS   =                    0",
            "END",
        ]);

        let error = parse_header_bytes(&bytes).expect_err("invalid BITPIX must fail");
        let message = error.to_string();
        assert!(message.contains("BITPIX"));
    }

    #[test]
    fn continue_card_appends_to_previous_string_value() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "LONGSTRN= 'OGIP 1.0'           / The OGIP long string convention may be used",
            "COMMENT   FITS Standard 4.0 long-string continuation example",
            "HISTORY   This card is not semantically relevant to continuation parsing",
            "OBJECT  = 'This is a long string value that continues &'",
            "CONTINUE  ' across multiple cards and remains ordered.'",
            "END",
        ]);

        let header = parse_header_bytes(&bytes).expect("header should parse");
        let object_keyword = FitsKeyword::parse("OBJECT", false).expect("keyword");
        let object_card = header.get(&object_keyword).expect("OBJECT card");

        match object_card.value() {
            Some(HeaderValue::String(value)) => {
                assert_eq!(
                    value,
                    "This is a long string value that continues  across multiple cards and remains ordered."
                );
            }
            other => panic!("expected string value, found {other:?}"),
        }
    }

    #[test]
    fn rejects_continue_without_preceding_string_card() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "CONTINUE  'orphaned fragment'",
            "END",
        ]);

        let error = parse_header_bytes(&bytes).expect_err("orphan CONTINUE must fail");
        let message = error.to_string();
        assert!(message.contains("CONTINUE"));
    }

    #[test]
    fn parses_hierarch_keyword_card() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "HIERARCH ESO DET CHIP1 ID = 'CCD-42' / hierarchical keyword example",
            "END",
        ]);

        let header = parse_header_bytes(&bytes).expect("header should parse");
        let keyword = FitsKeyword::parse_hierarchical("HIERARCH ESO DET CHIP1 ID")
            .expect("hierarchical keyword should parse");
        let card = header
            .get(&keyword)
            .expect("hierarchical card should exist");

        match card.value() {
            Some(HeaderValue::String(value)) => assert_eq!(value, "CCD-42"),
            other => panic!("expected string value, found {other:?}"),
        }
    }

    #[test]
    fn parse_header_cards_accepts_card_iterator() {
        let cards = vec![
            card("SIMPLE  =                    T"),
            card("BITPIX  =                  -32"),
            card("NAXIS   =                    2"),
            card("END"),
        ];

        let slices: Vec<&[u8]> = cards.iter().map(|card| card.as_slice()).collect();
        let header = parse_header_cards(slices).expect("iterator-based parse should succeed");

        assert_eq!(header.bitpix_i64(), Some(-32));
        assert_eq!(header.naxis_i64(), Some(2));
    }

    #[test]
    fn parses_commentary_cards_without_values() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "COMMENT   This is a commentary card",
            "HISTORY   This is a history card",
            "END",
        ]);

        let header = parse_header_bytes(&bytes).expect("header should parse");
        let comment = header
            .get_standard("COMMENT")
            .expect("COMMENT card should exist");
        let history = header
            .get_standard("HISTORY")
            .expect("HISTORY card should exist");

        assert_eq!(comment.comment(), Some("This is a commentary card"));
        assert_eq!(history.comment(), Some("This is a history card"));
    }

    #[test]
    fn rejects_simple_false() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    F",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "END",
        ]);

        let error = parse_header_bytes(&bytes).expect_err("SIMPLE = F must fail");
        let message = error.to_string();
        assert!(message.contains("SIMPLE"));
    }

    #[test]
    fn rejects_negative_naxis() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                   -1",
            "END",
        ]);

        let error = parse_header_bytes(&bytes).expect_err("negative NAXIS must fail");
        let message = error.to_string();
        assert!(message.contains("NAXIS"));
    }

    #[test]
    fn parses_reference_style_primary_header_example() {
        let bytes = header_bytes(&[
            "SIMPLE  =                    T / file does conform to FITS standard",
            "BITPIX  =                    8 / number of bits per data pixel",
            "NAXIS   =                    0 / number of data axes",
            "EXTEND  =                    T / FITS dataset may contain extensions",
            "END",
        ]);

        let header = parse_header_bytes(&bytes).expect("reference-style header should parse");
        assert_eq!(header.len(), 4);
        assert_eq!(header.bitpix_i64(), Some(8));
        assert_eq!(header.naxis_i64(), Some(0));
        assert!(header.get_standard("EXTEND").is_some());
    }
}
