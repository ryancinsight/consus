//! Shared parsing for null-terminated compound and enum member names.

use alloc::{format, string::String};
use consus_core::{Error, Result};

/// Parse a null-terminated member name from `data[start..]`.
///
/// Datatype versions 1 and 2 align each name field, including its terminator,
/// to an 8-byte boundary. Version 3 and later consume exactly the name and
/// its terminator.
pub(super) fn parse_member_name(
    data: &[u8],
    start: usize,
    version: u8,
    kind: &str,
    index: usize,
) -> Result<(String, usize)> {
    let mut pos = start;
    while pos < data.len() && data[pos] != 0 {
        pos += 1;
    }
    if pos >= data.len() {
        return Err(Error::InvalidFormat {
            message: format!("unterminated {kind} name at index {index}"),
        });
    }

    let name = core::str::from_utf8(&data[start..pos])
        .map_err(|_| Error::InvalidFormat {
            message: format!("{kind} {index} name is not valid UTF-8"),
        })
        .map(String::from)?;
    pos += 1;

    if version < 3 {
        let name_field_len = pos - start;
        let padded_len = name_field_len
            .checked_add(7)
            .ok_or_else(|| Error::InvalidFormat {
                message: format!("{kind} {index} name padding overflows"),
            })?
            & !7;
        pos = start
            .checked_add(padded_len)
            .ok_or_else(|| Error::InvalidFormat {
                message: format!("{kind} {index} name padding overflows"),
            })?;
    }

    Ok((name, pos))
}

#[cfg(test)]
mod tests {
    use super::parse_member_name;
    use consus_core::Error;

    #[test]
    fn parses_empty_and_utf8_names() {
        assert_eq!(
            parse_member_name(b"\0tail", 0, 3, "enum member", 2).unwrap(),
            (String::new(), 1)
        );
        assert_eq!(
            parse_member_name("δ\0tail".as_bytes(), 0, 3, "compound member", 0).unwrap(),
            ("δ".to_owned(), 3)
        );
    }

    #[test]
    fn consumes_version_dependent_padding() {
        assert_eq!(
            parse_member_name(b"abc\0xxxx", 0, 2, "enum member", 1).unwrap(),
            ("abc".to_owned(), 8)
        );
        assert_eq!(
            parse_member_name(b"abc\0xxxx", 0, 3, "enum member", 1).unwrap(),
            ("abc".to_owned(), 4)
        );
    }

    #[test]
    fn reports_missing_terminator_and_invalid_utf8() {
        let missing = parse_member_name(b"abc", 0, 3, "compound member", 4).unwrap_err();
        assert!(
            matches!(missing, Error::InvalidFormat { message } if message == "unterminated compound member name at index 4")
        );

        let invalid = parse_member_name(&[0xff, 0], 0, 3, "enum member", 5).unwrap_err();
        assert!(
            matches!(invalid, Error::InvalidFormat { message } if message == "enum member 5 name is not valid UTF-8")
        );
    }
}
