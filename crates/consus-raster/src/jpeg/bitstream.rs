use crate::{DecodeError, DecodeErrorKind};

#[derive(Clone, Copy)]
pub(super) struct HuffmanTable {
    first_code: [u16; 17],
    symbol_offset: [u16; 17],
    counts: [u8; 17],
    symbols: [u8; 256],
}

impl HuffmanTable {
    pub(super) fn parse(counts: &[u8], symbols: &[u8]) -> Result<Self, DecodeError> {
        let mut first_code = [0_u16; 17];
        let mut symbol_offset = [0_u16; 17];
        let mut table_counts = [0_u8; 17];
        let mut code = 0_u32;
        let mut offset = 0_u16;
        for length in 1..=16 {
            code = code.checked_shl(1).ok_or_else(malformed)?;
            let count = *counts.get(length - 1).ok_or_else(malformed)?;
            let code_limit = 1_u32 << length;
            if count != 0 && code + u32::from(count) >= code_limit {
                return Err(malformed());
            }
            first_code[length] = u16::try_from(code).map_err(|_| malformed())?;
            symbol_offset[length] = offset;
            table_counts[length] = count;
            code += u32::from(count);
            offset = offset.checked_add(u16::from(count)).ok_or_else(malformed)?;
        }
        if offset == 0 || usize::from(offset) != symbols.len() || symbols.len() > 256 {
            return Err(malformed());
        }
        let mut table_symbols = [0_u8; 256];
        table_symbols
            .get_mut(..symbols.len())
            .ok_or_else(malformed)?
            .copy_from_slice(symbols);
        Ok(Self {
            first_code,
            symbol_offset,
            counts: table_counts,
            symbols: table_symbols,
        })
    }

    pub(super) fn decode(self, reader: &mut BitReader<'_>) -> Result<u8, DecodeError> {
        let mut code = 0_u16;
        for length in 1..=16 {
            code = (code << 1) | u16::from(reader.bit()?);
            let first = self.first_code[length];
            let count = u16::from(self.counts[length]);
            if code >= first && code - first < count {
                let index = self.symbol_offset[length] + (code - first);
                return self
                    .symbols
                    .get(usize::from(index))
                    .copied()
                    .ok_or_else(malformed);
            }
        }
        Err(malformed())
    }
}

pub(super) struct Tables {
    pub(super) dc: [Option<HuffmanTable>; 4],
    pub(super) ac: [Option<HuffmanTable>; 4],
}

impl Tables {
    pub(super) const fn new() -> Self {
        Self {
            dc: [None; 4],
            ac: [None; 4],
        }
    }
}

pub(super) struct BitReader<'input> {
    bytes: &'input [u8],
    pub(super) cursor: usize,
    current: u8,
    remaining: u8,
}

impl<'input> BitReader<'input> {
    pub(super) const fn new(bytes: &'input [u8], cursor: usize) -> Self {
        Self {
            bytes,
            cursor,
            current: 0,
            remaining: 0,
        }
    }

    pub(super) fn bit(&mut self) -> Result<u8, DecodeError> {
        if self.remaining == 0 {
            self.current = self.data_byte()?;
            self.remaining = 8;
        }
        self.remaining -= 1;
        Ok((self.current >> self.remaining) & 1)
    }

    pub(super) fn bits(&mut self, count: u8) -> Result<u16, DecodeError> {
        if count > 16 {
            return Err(malformed());
        }
        let mut value = 0_u16;
        for _ in 0..count {
            value = (value << 1) | u16::from(self.bit()?);
        }
        Ok(value)
    }

    fn data_byte(&mut self) -> Result<u8, DecodeError> {
        let byte = *self.bytes.get(self.cursor).ok_or_else(malformed)?;
        self.cursor = self.cursor.checked_add(1).ok_or_else(too_large)?;
        if byte != 0xff {
            return Ok(byte);
        }
        match self.bytes.get(self.cursor) {
            Some(0x00) => {
                self.cursor = self.cursor.checked_add(1).ok_or_else(too_large)?;
                Ok(0xff)
            }
            _ => Err(malformed()),
        }
    }

    pub(super) fn finish_byte(&mut self) -> Result<(), DecodeError> {
        if self.remaining != 0 {
            let mask = u8::MAX >> (8 - self.remaining);
            if self.current & mask != mask {
                return Err(malformed());
            }
            self.remaining = 0;
        }
        Ok(())
    }

    pub(super) fn marker(&self) -> Result<(usize, u8, usize), DecodeError> {
        if self.remaining != 0 || self.bytes.get(self.cursor) != Some(&0xff) {
            return Err(malformed());
        }
        let start = self.cursor;
        let mut code_index = start.checked_add(1).ok_or_else(too_large)?;
        while self.bytes.get(code_index) == Some(&0xff) {
            code_index = code_index.checked_add(1).ok_or_else(too_large)?;
        }
        let code = *self.bytes.get(code_index).ok_or_else(malformed)?;
        if code == 0 {
            return Err(malformed());
        }
        Ok((
            start,
            code,
            code_index.checked_add(1).ok_or_else(too_large)?,
        ))
    }
}

pub(super) fn parse_tables(payload: &[u8], tables: &mut Tables) -> Result<(), DecodeError> {
    let mut cursor = 0_usize;
    while cursor < payload.len() {
        let selector = *payload.get(cursor).ok_or_else(malformed)?;
        cursor = cursor.checked_add(1).ok_or_else(too_large)?;
        let class = selector >> 4;
        let index = usize::from(selector & 0x0f);
        if class > 1 || index >= 4 {
            return Err(malformed());
        }
        let count_end = cursor.checked_add(16).ok_or_else(too_large)?;
        let counts = payload.get(cursor..count_end).ok_or_else(malformed)?;
        cursor = count_end;
        let symbol_count = counts.iter().try_fold(0_usize, |sum, count| {
            sum.checked_add(usize::from(*count)).ok_or_else(too_large)
        })?;
        let symbol_end = cursor.checked_add(symbol_count).ok_or_else(too_large)?;
        let symbols = payload.get(cursor..symbol_end).ok_or_else(malformed)?;
        cursor = symbol_end;
        let table = HuffmanTable::parse(counts, symbols)?;
        if class == 0 {
            tables.dc[index] = Some(table);
        } else {
            tables.ac[index] = Some(table);
        }
    }
    if cursor == 0 {
        return Err(malformed());
    }
    Ok(())
}

pub(super) fn receive_extend(reader: &mut BitReader<'_>, count: u8) -> Result<i32, DecodeError> {
    if count == 0 {
        return Ok(0);
    }
    let value = i32::from(reader.bits(count)?);
    let threshold = 1_i32
        .checked_shl(u32::from(count - 1))
        .ok_or_else(malformed)?;
    if value >= threshold {
        Ok(value)
    } else {
        let range = 1_i32.checked_shl(u32::from(count)).ok_or_else(malformed)?;
        value
            .checked_add(1)
            .and_then(|v| v.checked_sub(range))
            .ok_or_else(malformed)
    }
}

const fn malformed() -> DecodeError {
    DecodeError::new(DecodeErrorKind::Malformed)
}

const fn too_large() -> DecodeError {
    DecodeError::new(DecodeErrorKind::TooLarge)
}
