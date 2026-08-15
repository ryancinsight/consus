//! FITS table parsing helpers.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "alloc")]
use consus_core::{Datatype, Error, Result};

#[cfg(feature = "alloc")]
use crate::header::{FitsHeader, HeaderValue};

#[cfg(feature = "alloc")]
use super::types::{FitsTableColumn, FitsTableDescriptorCore};

#[cfg(feature = "alloc")]
pub(super) fn parse_table_core(
    header: &FitsHeader,
    binary: bool,
) -> Result<FitsTableDescriptorCore> {
    let row_len = parse_required_non_negative_integer(header, "NAXIS1")?;
    let rows = parse_required_non_negative_integer(header, "NAXIS2")?;
    let fields = parse_required_non_negative_integer(header, "TFIELDS")?;
    let heap_size = if binary {
        parse_optional_non_negative_integer(header, "PCOUNT")?.unwrap_or(0)
    } else {
        0
    };
    let mut columns = Vec::with_capacity(fields);
    for index in 1..=fields {
        columns.push(parse_column(header, index, binary)?);
    }
    if binary {
        let mut offset = 0usize;
        for col in &mut columns {
            col.set_col_offset(offset);
            offset = offset
                .checked_add(col.byte_width())
                .ok_or(Error::Overflow)?;
        }
    } else {
        let mut prefix_offset = 0usize;
        for (i, col) in columns.iter_mut().enumerate() {
            let keyword = indexed_keyword("TBCOL", i + 1)?;
            let tbcol_opt = parse_optional_non_negative_integer(header, &keyword)?;
            let col_offset = match tbcol_opt {
                Some(tbcol) if tbcol > 0 => tbcol - 1,
                _ => prefix_offset,
            };
            col.set_col_offset(col_offset);
            prefix_offset = prefix_offset
                .checked_add(col.byte_width())
                .ok_or(Error::Overflow)?;
        }
    }
    Ok(FitsTableDescriptorCore::new(
        row_len, rows, columns, heap_size,
    ))
}

/// Extract the column width from an ASCII TFORM string.
///
/// ASCII TFORM values use formats like `A8`, `I10`, `E16.7`, `F12.5`, etc.
/// The width is the numeric portion before any decimal point.
///
/// ## Derivation
///
/// FITS Standard 4.0 §7.2: ASCII table TFORM is `<rTa<n>.<m>` where `<n>`
/// is the column width in characters. For this function, `<n>` is extracted
/// as the digits before the optional decimal point.
#[cfg(feature = "alloc")]
fn parse_ascii_column_width(tform: &str) -> usize {
    let trimmed = tform.trim_end();
    let digits_start = trimmed
        .char_indices()
        .find(|(_, c)| c.is_ascii_alphabetic())
        .map(|(i, _)| i + 1)
        .unwrap_or(0);
    let rest = &trimmed[digits_start..];
    let width_str: &str = rest.split('.').next().unwrap_or(rest);
    width_str.parse::<usize>().unwrap_or(0)
}

#[cfg(feature = "alloc")]
fn parse_column(header: &FitsHeader, index: usize, binary: bool) -> Result<FitsTableColumn> {
    let tform = parse_required_string(header, indexed_keyword("TFORM", index)?.as_str())?
        .trim_end()
        .to_owned();
    let name = parse_optional_string(header, indexed_keyword("TTYPE", index)?.as_str())?
        .map(str::trim_end)
        .map(str::to_owned);
    let unit = parse_optional_string(header, indexed_keyword("TUNIT", index)?.as_str())?
        .map(str::trim_end)
        .map(str::to_owned);
    let display = parse_optional_string(header, indexed_keyword("TDISP", index)?.as_str())?
        .map(str::trim_end)
        .map(str::to_owned);
    let null = parse_optional_string(header, indexed_keyword("TNULL", index)?.as_str())?
        .map(str::trim_end)
        .map(str::to_owned);
    let scale = parse_optional_real(header, indexed_keyword("TSCAL", index)?.as_str())?;
    let zero = parse_optional_real(header, indexed_keyword("TZERO", index)?.as_str())?;

    if binary {
        FitsTableColumn::from_binary_tform(index, name, tform, unit, display, null, scale, zero)
    } else {
        let width = parse_ascii_column_width(&tform);
        Ok(FitsTableColumn::new(
            index,
            name,
            tform,
            Datatype::FixedString {
                length: width,
                encoding: consus_core::StringEncoding::Ascii,
            },
            width,
            0,
            unit,
            display,
            null,
            scale,
            zero,
        ))
    }
}

#[cfg(feature = "alloc")]
pub(super) fn validate_xtension(header: &FitsHeader, expected: &str) -> Result<()> {
    let xtension = parse_required_string(header, "XTENSION")?.trim_end();
    if xtension == expected {
        Ok(())
    } else {
        invalid_format("unexpected FITS XTENSION value for table descriptor")
    }
}

#[cfg(feature = "alloc")]
fn parse_required_non_negative_integer(header: &FitsHeader, keyword: &str) -> Result<usize> {
    parse_optional_non_negative_integer(header, keyword)?.ok_or_else(|| Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: alloc::format!("missing required FITS table keyword: {keyword}"),
    })
}

#[cfg(feature = "alloc")]
fn parse_optional_non_negative_integer(
    header: &FitsHeader,
    keyword: &str,
) -> Result<Option<usize>> {
    let Some(value) = parse_optional_integer(header, keyword)? else {
        return Ok(None);
    };
    if value < 0 {
        return invalid_format("FITS table keyword must be a non-negative integer");
    }
    usize::try_from(value)
        .map(Some)
        .map_err(|_| Error::Overflow)
}

#[cfg(feature = "alloc")]
fn parse_optional_integer(header: &FitsHeader, keyword: &str) -> Result<Option<i64>> {
    let Some(card) = header.get_standard(keyword) else {
        return Ok(None);
    };
    match card.value() {
        Some(HeaderValue::Integer(value)) => value.to_i64().map(Some),
        Some(_) => invalid_format("FITS table keyword must contain an integer value"),
        None => invalid_format("FITS table keyword is missing a value"),
    }
}

#[cfg(feature = "alloc")]
fn parse_optional_real(header: &FitsHeader, keyword: &str) -> Result<Option<f64>> {
    let Some(card) = header.get_standard(keyword) else {
        return Ok(None);
    };
    match card.value() {
        Some(HeaderValue::Integer(value)) => Ok(Some(value.to_i64()? as f64)),
        Some(HeaderValue::Real(value)) => value.to_f64().map(Some),
        Some(_) => invalid_format("FITS table keyword must contain a numeric value"),
        None => invalid_format("FITS table keyword is missing a value"),
    }
}

#[cfg(feature = "alloc")]
pub(super) fn parse_required_string<'a>(header: &'a FitsHeader, keyword: &str) -> Result<&'a str> {
    parse_optional_string(header, keyword)?.ok_or_else(|| Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: alloc::format!("missing required FITS table keyword: {keyword}"),
    })
}

#[cfg(feature = "alloc")]
fn parse_optional_string<'a>(header: &'a FitsHeader, keyword: &str) -> Result<Option<&'a str>> {
    let Some(card) = header.get_standard(keyword) else {
        return Ok(None);
    };
    match card.value() {
        Some(HeaderValue::String(value)) => Ok(Some(value.as_str())),
        Some(_) => invalid_format("FITS table keyword must contain a string value"),
        None => invalid_format("FITS table keyword is missing a value"),
    }
}

#[cfg(feature = "alloc")]
fn indexed_keyword(prefix: &str, index: usize) -> Result<String> {
    if index == 0 {
        return invalid_format("FITS indexed keywords are 1-based");
    }
    Ok(alloc::format!("{prefix}{index}"))
}

#[cfg(feature = "alloc")]
pub(super) fn invalid_format<T>(message: &str) -> Result<T> {
    Err(Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: message.into(),
    })
}
