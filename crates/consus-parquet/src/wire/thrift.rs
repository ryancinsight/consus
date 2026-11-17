//! Minimal Thrift Compact Protocol decoder.
//!
//! ## Specification
//!
//! Implements the subset of the Thrift Compact Protocol (Apache Thrift v0.16)
//! required to decode Apache Parquet footer metadata and page headers.
//!
//! Reference: <https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md>
//!
//! ## Protocol invariants
//!
//! - `zigzag_decode_i32(n: u32) -> i32 = (n >> 1) as i32 ^ -((n & 1) as i32)`
//! - `zigzag_decode_i64(n: u64) -> i64 = (n >> 1) as i64 ^ -((n & 1) as i64)`
//! - Unsigned varint: bytes are 7-bit LSB-first groups; high bit = continuation
//! - Field headers encode the delta from the previous field ID in the upper nibble
//!   and the type code in the lower nibble; delta == 0 means an absolute i16 field
//!   ID follows as a zigzag varint
//! - Lists/sets encode short counts (<= 14) in the upper nibble of the header byte;
//!   long counts use `0xF0 | elem_type` followed by a varint count
//! - Boolean fields carry their value in the field type nibble (0x01 = true, 0x02 = false);
//!   no additional byte is read for booleans in struct context
//! - A byte with value 0x00 terminates a struct (field stop)

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

use consus_core::{Error, Result};

/// Thrift Compact Protocol type codes; discriminants match the wire encoding.
/// `BoolTrue`/`BoolFalse` are struct-field-only: value is in the header nibble.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThriftType {
    BoolTrue = 0x01,
    BoolFalse = 0x02,
    Byte = 0x03,
    I16 = 0x04,
    I32 = 0x05,
    I64 = 0x06,
    Double = 0x07,
    Binary = 0x08,
    List = 0x09,
    Set = 0x0A,
    Map = 0x0B,
    Struct = 0x0C,
    Uuid = 0x0D,
}

impl ThriftType {
    /// Maps a raw type-code byte to a `ThriftType`, or returns `None`.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0x01 => Some(Self::BoolTrue),
            0x02 => Some(Self::BoolFalse),
            0x03 => Some(Self::Byte),
            0x04 => Some(Self::I16),
            0x05 => Some(Self::I32),
            0x06 => Some(Self::I64),
            0x07 => Some(Self::Double),
            0x08 => Some(Self::Binary),
            0x09 => Some(Self::List),
            0x0A => Some(Self::Set),
            0x0B => Some(Self::Map),
            0x0C => Some(Self::Struct),
            0x0D => Some(Self::Uuid),
            _ => None,
        }
    }
}

/// Zero-copy cursor over a Thrift Compact Protocol-encoded byte slice.
pub struct ThriftReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ThriftReader<'a> {
    /// Creates a reader at position 0.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    /// Returns bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
    /// Returns `true` when all bytes have been consumed.
    pub fn is_exhausted(&self) -> bool {
        self.pos >= self.data.len()
    }
    /// Returns the current byte offset.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Takes a bounded byte range and advances the reader past it.
    fn take_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(len).ok_or(Error::Overflow)?;
        if end > self.data.len() {
            return Err(Error::BufferTooSmall {
                required: end,
                provided: self.data.len(),
            });
        }
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    /// Reads one byte; returns `Error::BufferTooSmall` when exhausted.
    pub fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(Error::BufferTooSmall {
                required: self.pos + 1,
                provided: self.data.len(),
            });
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    /// 7-bit LSB-first unsigned varint; errors on shift >= 64.
    pub fn read_varint_u64(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            if shift >= 64 {
                return Err(Error::InvalidFormat {
                    #[cfg(feature = "alloc")]
                    message: String::from("thrift: varint overflow"),
                });
            }
            let b = self.read_byte()?;
            result |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
    }

    /// Zigzag i16: `decode(w) = (w >> 1) as i16 ^ -((w & 1) as i16)`.
    pub fn read_i16(&mut self) -> Result<i16> {
        let w = self.read_varint_u64()? as u16;
        Ok((w >> 1) as i16 ^ -((w & 1) as i16))
    }

    /// Zigzag i32: `decode(w) = (w >> 1) as i32 ^ -((w & 1) as i32)`.
    pub fn read_i32(&mut self) -> Result<i32> {
        let w = self.read_varint_u64()? as u32;
        Ok((w >> 1) as i32 ^ -((w & 1) as i32))
    }

    /// Zigzag i64: `decode(w) = (w >> 1) as i64 ^ -((w & 1) as i64)`.
    pub fn read_i64(&mut self) -> Result<i64> {
        let w = self.read_varint_u64()?;
        Ok((w >> 1) as i64 ^ -((w & 1) as i64))
    }

    /// 8-byte little-endian IEEE 754 f64.
    pub fn read_double(&mut self) -> Result<f64> {
        let b: [u8; 8] = self
            .take_bytes(8)?
            .try_into()
            .expect("invariant: take_bytes returned exactly eight bytes");
        Ok(f64::from_le_bytes(b))
    }

    /// Varint length-prefixed binary blob.
    #[cfg(feature = "alloc")]
    pub fn read_binary(&mut self) -> Result<Vec<u8>> {
        let len = usize::try_from(self.read_varint_u64()?).map_err(|_| Error::Overflow)?;
        Ok(self.take_bytes(len)?.to_vec())
    }

    /// UTF-8 string decoded from a binary blob.
    #[cfg(feature = "alloc")]
    pub fn read_string(&mut self) -> Result<String> {
        let b = self.read_binary()?;
        String::from_utf8(b).map_err(|_| Error::InvalidFormat {
            message: String::from("thrift: invalid UTF-8 in string"),
        })
    }

    /// Field header: returns `None` on stop byte (0x00), or `Some((field_id, type_code))`.
    /// Updates `last_field_id` in place. `type_code` preserves BoolTrue/BoolFalse nibble semantics.
    pub fn read_field_header(&mut self, last_field_id: &mut i16) -> Result<Option<(i16, u8)>> {
        let byte = self.read_byte()?;
        if byte == 0x00 {
            return Ok(None);
        }
        let delta = (byte >> 4) & 0x0F;
        let type_code = byte & 0x0F;
        let field_id = if delta != 0 {
            (*last_field_id).wrapping_add(delta as i16)
        } else {
            self.read_i16()?
        };
        *last_field_id = field_id;
        Ok(Some((field_id, type_code)))
    }

    /// List/set header: short form (count <= 14) or long form (0xF0|elem + varint count).
    /// Returns `(element_type_code, count)`.
    pub fn read_list_header(&mut self) -> Result<(u8, usize)> {
        let byte = self.read_byte()?;
        let count_nibble = (byte >> 4) & 0x0F;
        let elem_type = byte & 0x0F;
        let count = if count_nibble == 0x0F {
            usize::try_from(self.read_varint_u64()?).map_err(|_| Error::Overflow)?
        } else {
            count_nibble as usize
        };
        if count > self.remaining() {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "thrift: list count {count} exceeds remaining bytes {}",
                    self.remaining()
                ),
            });
        }
        Ok((elem_type, count))
    }

    /// Map header: varint count; if > 0, one byte (key_type<<4)|val_type.
    /// Returns `(key_type, value_type, count)`; count 0 returns `(0,0,0)`.
    pub fn read_map_header(&mut self) -> Result<(u8, u8, usize)> {
        let count = usize::try_from(self.read_varint_u64()?).map_err(|_| Error::Overflow)?;
        if count == 0 {
            return Ok((0, 0, 0));
        }
        if count > self.remaining() / 2 {
            return Err(Error::InvalidFormat {
                message: alloc::format!(
                    "thrift: map count {count} exceeds remaining bytes {}",
                    self.remaining()
                ),
            });
        }
        let tb = self.read_byte()?;
        Ok(((tb >> 4) & 0x0F, tb & 0x0F, count))
    }

    /// Skips one value of `type_code` without decoding it.
    /// Bool (0x01/0x02): 0 bytes (value in header). UUID: 16 fixed bytes.
    /// Structs, lists, sets, maps are recursively skipped.
    pub fn skip(&mut self, type_code: u8) -> Result<()> {
        self.skip_depth(type_code, 0)
    }

    /// Depth-bounded variant of [`Self::skip`].
    ///
    /// Thrift structs/list/maps nest self-similarly: a struct containing a
    /// list of structs containing a list of structs... recurses one frame per
    /// input byte with no format-enforced lower bound. Rust performs no
    /// tail-call elimination, so unbounded recursion overflows the stack — an
    /// abort no `Result` can express. The [`consus_core::ParseBudget`] descent ceiling
    /// bounds it here, matching the HDF5 B-tree and MAT v5 fixes.
    fn skip_depth(&mut self, type_code: u8, depth: u16) -> Result<()> {
        let depth =
            consus_core::ParseBudget::DEFAULT.descend(depth, "thrift nested-type skip depth")?;
        match type_code {
            0x01 | 0x02 => Ok(()),
            0x03 => {
                self.read_byte()?;
                Ok(())
            }
            0x04 => {
                self.read_i16()?;
                Ok(())
            }
            0x05 => {
                self.read_i32()?;
                Ok(())
            }
            0x06 => {
                self.read_i64()?;
                Ok(())
            }
            0x07 => {
                self.read_double()?;
                Ok(())
            }
            0x08 => {
                let len = usize::try_from(self.read_varint_u64()?).map_err(|_| Error::Overflow)?;
                self.take_bytes(len)?;
                Ok(())
            }
            0x09 | 0x0A => {
                let (et, n) = self.read_list_header()?;
                for _ in 0..n {
                    self.skip_depth(et, depth)?;
                }
                Ok(())
            }
            0x0B => {
                let (kt, vt, n) = self.read_map_header()?;
                for _ in 0..n {
                    self.skip_depth(kt, depth)?;
                    self.skip_depth(vt, depth)?;
                }
                Ok(())
            }
            0x0C => {
                let mut id: i16 = 0;
                loop {
                    match self.read_field_header(&mut id)? {
                        None => return Ok(()),
                        Some((_, ft)) => self.skip_depth(ft, depth)?,
                    }
                }
            }
            0x0D => {
                self.take_bytes(16)?;
                Ok(())
            }
            _ => Err(Error::InvalidFormat {
                #[cfg(feature = "alloc")]
                message: String::from("thrift: unknown type code"),
            }),
        }
    }
}

#[cfg(test)]
mod tests;
