use consus_core::{Error, Result};

use crate::header::{FitsHeader, HeaderValue};

pub(super) fn parse_xtension(header: &FitsHeader) -> Result<Option<&str>> {
    let Some(card) = header.get_standard("XTENSION") else {
        return Ok(None);
    };

    match card.value() {
        Some(HeaderValue::String(value)) => Ok(Some(value.as_str())),
        Some(_) => invalid_format("XTENSION must contain a string value"),
        None => invalid_format("XTENSION is missing a value"),
    }
}

pub(super) fn invalid_format<T>(message: &str) -> Result<T> {
    Err(Error::InvalidFormat {
        #[cfg(feature = "alloc")]
        message: message.into(),
    })
}
