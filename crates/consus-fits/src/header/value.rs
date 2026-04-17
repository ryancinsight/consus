//! FITS header value model.
//!
//! ## Specification
//!
//! FITS header values occupy columns 11-80 of a card image when the card
//! contains a value field (`= ` in columns 9-10). This module defines the
//! canonical in-memory representation for parsed FITS header values.
//!
//! The representation is intentionally semantic rather than byte-oriented:
//! - logical values are modeled as booleans
//! - integer and floating-point values preserve their textual source
//! - string values preserve their decoded content
//! - complex values preserve both components
//! - undefined values are represented explicitly
//!
//! ## Standard references
//!
//! FITS Standard 4.0, Section 4.2:
//! - 4.2.1 Character string values
//! - 4.2.2 Logical values
//! - 4.2.3 Integer values
//! - 4.2.4 Real floating values
//! - 4.2.5 Complex values
//! - 4.2.6 Array values
//! - 4.2.7 Undefined values
//!
//! ## Design constraints
//!
//! - This module is the single source of truth for FITS header value typing.
//! - Parsing logic is delegated to `parser`; this module owns the value model.
//! - Numeric values preserve lexical fidelity through `raw` fields.
//! - No I/O behavior is implemented here.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::string::{String, ToString};

use consus_core::{Error, Result};

/// Canonical FITS header value.
///
/// The enum preserves semantic category and, for numeric values, the original
/// lexical representation. This supports round-trip formatting and precise
/// diagnostics without introducing floating-point canonicalization at parse
/// time.
///
/// ## Invariants
///
/// - `Logical` corresponds to FITS `T` or `F`.
/// - `Integer.raw` is a valid FITS integer token.
/// - `Real.raw` is a valid FITS real token.
/// - `Complex.real.raw` and `Complex.imaginary.raw` are valid FITS real tokens.
/// - `String` contains the decoded string payload without surrounding quotes.
/// - `Undefined` corresponds to an explicitly present but undefined value.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderValue {
    /// FITS character string value without surrounding quotes.
    String(String),

    /// FITS logical value (`T` or `F`).
    Logical(bool),

    /// FITS integer value.
    Integer(IntegerValue),

    /// FITS real floating-point value.
    Real(RealValue),

    /// FITS complex value `(real, imaginary)`.
    Complex(ComplexValue),

    /// FITS undefined value.
    Undefined,
}

/// FITS integer value preserving lexical source.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerValue {
    /// Original FITS token after trimming field padding.
    pub raw: String,
}

/// FITS real value preserving lexical source.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealValue {
    /// Original FITS token after trimming field padding.
    pub raw: String,
}

/// FITS complex value preserving lexical source for both components.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComplexValue {
    /// Real component.
    pub real: RealValue,
    /// Imaginary component.
    pub imaginary: RealValue,
}

#[cfg(feature = "alloc")]
impl HeaderValue {
    /// Parse a FITS header value token.
    ///
    /// The input must be the value field only, not the full card image.
    /// Leading and trailing FITS padding is ignored.
    ///
    /// ## Accepted forms
    ///
    /// - `'text'`
    /// - `T`, `F`
    /// - `123`, `-42`, `+7`
    /// - `1.0`, `-1.23E+04`, `6.02D23`
    /// - `(1.0,2.0)`, `( -1.0E+2 , +3.5D-1 )`
    /// - empty or all-space field => `Undefined`
    ///
    /// ## Errors
    ///
    /// Returns [`Error::InvalidFormat`] when the token is not a valid FITS
    /// header value according to the supported foundational subset.
    pub fn parse(token: &str) -> Result<Self> {
        let trimmed = token.trim();

        if trimmed.is_empty() {
            return Ok(Self::Undefined);
        }

        if let Some(value) = parse_string(trimmed)? {
            return Ok(Self::String(value));
        }

        if let Some(value) = parse_logical(trimmed) {
            return Ok(Self::Logical(value));
        }

        if let Some(value) = parse_complex(trimmed)? {
            return Ok(Self::Complex(value));
        }

        if is_integer_token(trimmed) {
            return Ok(Self::Integer(IntegerValue {
                raw: trimmed.to_string(),
            }));
        }

        if is_real_token(trimmed) {
            return Ok(Self::Real(RealValue {
                raw: trimmed.to_string(),
            }));
        }

        Err(invalid_format("unrecognized FITS header value"))
    }

    /// Returns the value as a FITS string payload.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Returns the value as a FITS logical.
    pub fn as_logical(&self) -> Option<bool> {
        match self {
            Self::Logical(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the value as a FITS integer token.
    pub fn as_integer(&self) -> Option<&IntegerValue> {
        match self {
            Self::Integer(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the value as a FITS real token.
    pub fn as_real(&self) -> Option<&RealValue> {
        match self {
            Self::Real(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the value as a FITS complex token.
    pub fn as_complex(&self) -> Option<&ComplexValue> {
        match self {
            Self::Complex(value) => Some(value),
            _ => None,
        }
    }

    /// Returns `true` when the value is explicitly undefined.
    pub fn is_undefined(&self) -> bool {
        matches!(self, Self::Undefined)
    }
}

#[cfg(feature = "alloc")]
impl IntegerValue {
    /// Parse the integer token as `i64`.
    pub fn to_i64(&self) -> Result<i64> {
        self.raw
            .parse::<i64>()
            .map_err(|_| invalid_format("invalid FITS integer value"))
    }
}

#[cfg(feature = "alloc")]
impl RealValue {
    /// Parse the real token as `f64`.
    ///
    /// FITS permits both `E` and `D` exponent markers. `D` is normalized to `E`
    /// before parsing.
    pub fn to_f64(&self) -> Result<f64> {
        let normalized = normalize_real_token(&self.raw);
        normalized
            .parse::<f64>()
            .map_err(|_| invalid_format("invalid FITS real value"))
    }
}

#[cfg(feature = "alloc")]
impl ComplexValue {
    /// Parse the complex value as `(f64, f64)`.
    pub fn to_f64_pair(&self) -> Result<(f64, f64)> {
        Ok((self.real.to_f64()?, self.imaginary.to_f64()?))
    }
}

#[cfg(feature = "alloc")]
fn parse_string(token: &str) -> Result<Option<String>> {
    if !token.starts_with('\'') {
        return Ok(None);
    }

    if !token.ends_with('\'') || token.len() < 2 {
        return Err(invalid_format("unterminated FITS string literal"));
    }

    let inner = &token[1..token.len() - 1];
    let mut decoded = String::new();
    let mut chars = inner.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\'' {
            match chars.peek() {
                Some('\'') => {
                    decoded.push('\'');
                    chars.next();
                }
                _ => return Err(invalid_format("invalid embedded quote in FITS string literal")),
            }
        } else {
            decoded.push(ch);
        }
    }

    Ok(Some(decoded))
}

fn parse_logical(token: &str) -> Option<bool> {
    match token {
        "T" => Some(true),
        "F" => Some(false),
        _ => None,
    }
}

#[cfg(feature = "alloc")]
fn parse_complex(token: &str) -> Result<Option<ComplexValue>> {
    if !(token.starts_with('(') && token.ends_with(')')) {
        return Ok(None);
    }

    let inner = &token[1..token.len() - 1];
    let Some((left, right)) = split_complex_components(inner) else {
        return Err(invalid_format("invalid FITS complex value"));
    };

    let real = left.trim();
    let imaginary = right.trim();

    if !is_real_token(real) || !is_real_token(imaginary) {
        return Err(invalid_format("invalid FITS complex component"));
    }

    Ok(Some(ComplexValue {
        real: RealValue {
            raw: real.to_string(),
        },
        imaginary: RealValue {
            raw: imaginary.to_string(),
        },
    }))
}

fn split_complex_components(inner: &str) -> Option<(&str, &str)> {
    let mut comma_index = None;
    for (index, ch) in inner.char_indices() {
        if ch == ',' {
            if comma_index.is_some() {
                return None;
            }
            comma_index = Some(index);
        }
    }

    let index = comma_index?;
    Some((&inner[..index], &inner[index + 1..]))
}

fn is_integer_token(token: &str) -> bool {
    let bytes = token.as_bytes();
    if bytes.is_empty() {
        return false;
    }

    let start = if bytes[0] == b'+' || bytes[0] == b'-' {
        if bytes.len() == 1 {
            return false;
        }
        1
    } else {
        0
    };

    bytes[start..].iter().all(u8::is_ascii_digit)
}

fn is_real_token(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }

    let normalized = normalize_real_token(token);
    let bytes = normalized.as_bytes();

    let mut index = 0;
    if bytes[index] == b'+' || bytes[index] == b'-' {
        index += 1;
        if index == bytes.len() {
            return false;
        }
    }

    let mut seen_digit = false;
    let mut seen_dot = false;
    let mut seen_exp = false;

    while index < bytes.len() {
        match bytes[index] {
            b'0'..=b'9' => {
                seen_digit = true;
                index += 1;
            }
            b'.' if !seen_dot && !seen_exp => {
                seen_dot = true;
                index += 1;
            }
            b'E' | b'e' if !seen_exp && seen_digit => {
                seen_exp = true;
                index += 1;
                if index == bytes.len() {
                    return false;
                }
                if bytes[index] == b'+' || bytes[index] == b'-' {
                    index += 1;
                    if index == bytes.len() {
                        return false;
                    }
                }
                let exp_start = index;
                while index < bytes.len() && bytes[index].is_ascii_digit() {
                    index += 1;
                }
                if exp_start == index {
                    return false;
                }
            }
            _ => return false,
        }
    }

    seen_dot || seen_exp
}

fn normalize_real_token(token: &str) -> String {
    token
        .chars()
        .map(|ch| if ch == 'D' || ch == 'd' { 'E' } else { ch })
        .collect()
}

fn invalid_format(message: &str) -> Error {
    Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: message.to_string(),
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn parses_string_value() {
        let value = HeaderValue::parse("'ABC'").unwrap();
        assert_eq!(value, HeaderValue::String("ABC".to_string()));
        assert_eq!(value.as_string(), Some("ABC"));
    }

    #[test]
    fn parses_string_with_escaped_quote() {
        let value = HeaderValue::parse("'O''HARA'").unwrap();
        assert_eq!(value, HeaderValue::String("O'HARA".to_string()));
    }

    #[test]
    fn parses_logical_values() {
        assert_eq!(HeaderValue::parse("T").unwrap(), HeaderValue::Logical(true));
        assert_eq!(HeaderValue::parse("F").unwrap(), HeaderValue::Logical(false));
    }

    #[test]
    fn parses_integer_value() {
        let value = HeaderValue::parse("-32").unwrap();
        let integer = value.as_integer().unwrap();
        assert_eq!(integer.raw, "-32");
        assert_eq!(integer.to_i64().unwrap(), -32);
    }

    #[test]
    fn parses_real_value_with_e_exponent() {
        let value = HeaderValue::parse("-1.234E+03").unwrap();
        let real = value.as_real().unwrap();
        assert_eq!(real.raw, "-1.234E+03");
        assert_eq!(real.to_f64().unwrap(), -1234.0);
    }

    #[test]
    fn parses_real_value_with_d_exponent() {
        let value = HeaderValue::parse("6.02D23").unwrap();
        let real = value.as_real().unwrap();
        assert_eq!(real.raw, "6.02D23");
        assert_eq!(real.to_f64().unwrap(), 6.02e23);
    }

    #[test]
    fn parses_complex_value() {
        let value = HeaderValue::parse("(1.0,-2.5D+01)").unwrap();
        let complex = value.as_complex().unwrap();
        assert_eq!(complex.real.raw, "1.0");
        assert_eq!(complex.imaginary.raw, "-2.5D+01");
        assert_eq!(complex.to_f64_pair().unwrap(), (1.0, -25.0));
    }

    #[test]
    fn parses_undefined_value_from_empty_field() {
        let value = HeaderValue::parse("   ").unwrap();
        assert!(value.is_undefined());
    }

    #[test]
    fn rejects_unterminated_string() {
        let error = HeaderValue::parse("'ABC").unwrap_err();
        assert!(matches!(error, Error::InvalidFormat { .. }));
    }

    #[test]
    fn rejects_invalid_complex_value() {
        let error = HeaderValue::parse("(1.0,XYZ)").unwrap_err();
        assert!(matches!(error, Error::InvalidFormat { .. }));
    }

    #[test]
    fn rejects_unrecognized_token() {
        let error = HeaderValue::parse("ABC").unwrap_err();
        assert!(matches!(error, Error::InvalidFormat { .. }));
    }
}
