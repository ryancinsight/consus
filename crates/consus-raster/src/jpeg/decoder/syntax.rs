use crate::DecodeError;

use super::malformed;

pub(super) fn read_word(bytes: &[u8]) -> Result<u16, DecodeError> {
    let value: [u8; 2] = bytes
        .get(..2)
        .ok_or_else(malformed)?
        .try_into()
        .map_err(|_| malformed())?;
    Ok(u16::from_be_bytes(value))
}
