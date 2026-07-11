//! Bounded protobuf wire primitives.

use super::parse::ParseError;

pub(crate) const VARINT: u8 = 0;
pub(crate) const LENGTH_DELIMITED: u8 = 2;

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
    max_field_bytes: usize,
}

impl<'a> Reader<'a> {
    pub(crate) const fn new(bytes: &'a [u8], max_field_bytes: usize) -> Self {
        Self {
            bytes,
            position: 0,
            max_field_bytes,
        }
    }

    pub(crate) fn next(&mut self) -> Result<Option<(u32, u8)>, ParseError> {
        if self.position == self.bytes.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let number = u32::try_from(key >> 3).map_err(|_| ParseError::InvalidFieldKey {
            offset: self.position,
        })?;
        let wire = (key & 0x07) as u8;
        if number == 0 {
            return Err(ParseError::InvalidFieldKey {
                offset: self.position,
            });
        }
        Ok(Some((number, wire)))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }

    pub(crate) fn varint(&mut self) -> Result<u64, ParseError> {
        let start = self.position;
        let mut value = 0_u64;
        for shift in (0..70).step_by(7) {
            let byte = *self
                .bytes
                .get(self.position)
                .ok_or(ParseError::Truncated { offset: start })?;
            self.position += 1;
            if shift == 63 && byte > 1 {
                return Err(ParseError::VarintOverflow { offset: start });
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(ParseError::VarintOverflow { offset: start })
    }

    pub(crate) fn bytes(&mut self) -> Result<&'a [u8], ParseError> {
        let start = self.position;
        let length = usize::try_from(self.varint()?)
            .map_err(|_| ParseError::LengthOverflow { offset: start })?;
        if length > self.max_field_bytes {
            return Err(ParseError::LimitExceeded {
                resource: "field bytes",
                limit: self.max_field_bytes,
                actual: length,
            });
        }
        let end = self
            .position
            .checked_add(length)
            .ok_or(ParseError::LengthOverflow { offset: start })?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or(ParseError::Truncated { offset: start })?;
        self.position = end;
        Ok(value)
    }

    pub(crate) fn string(&mut self) -> Result<&'a str, ParseError> {
        let offset = self.position;
        core::str::from_utf8(self.bytes()?).map_err(|_| ParseError::InvalidUtf8 { offset })
    }

    pub(crate) fn message(&mut self) -> Result<Self, ParseError> {
        Ok(Self::new(self.bytes()?, self.max_field_bytes))
    }

    pub(crate) fn skip(&mut self, wire: u8) -> Result<(), ParseError> {
        match wire {
            VARINT => {
                self.varint()?;
            }
            1 => self.advance(8)?,
            LENGTH_DELIMITED => {
                self.bytes()?;
            }
            5 => self.advance(4)?,
            _ => {
                return Err(ParseError::UnsupportedWireType {
                    wire,
                    offset: self.position,
                });
            }
        }
        Ok(())
    }

    fn advance(&mut self, length: usize) -> Result<(), ParseError> {
        let start = self.position;
        self.position = self
            .position
            .checked_add(length)
            .filter(|&end| end <= self.bytes.len())
            .ok_or(ParseError::Truncated { offset: start })?;
        Ok(())
    }
}
