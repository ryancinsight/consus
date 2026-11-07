//! Parse the scalar NPY dictionary by structure, never by substring lookup.
//!
//! The three entries may appear in any order and use either quote delimiter.
//! Each token must end at the next dictionary/tuple separator. In particular,
//! `Falsehood` is not `False`, `(n)` is not a tuple, and `(n,,)` has a missing
//! element. Parenthesized dictionaries, comments between tokens and adjacent
//! strings retain Python literal semantics. Consuming all trailing trivia also
//! prevents field-looking text outside the dictionary from becoming metadata.

use crate::{Error, Result};
use std::borrow::Cow;

pub(super) struct Header {
    pub dtype: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

impl Header {
    pub(super) fn parse(text: &str) -> Result<Self> {
        let mut parser = Parser { rest: text };
        let mut groups = 0;
        while parser.take('(') {
            groups += 1;
        }
        parser.expect('{')?;
        let (mut dtype, mut fortran_order, mut shape) = (None, None, None);
        if !parser.take('}') {
            loop {
                let key = parser.string()?;
                parser.expect(':')?;
                match key.as_str() {
                    // Python dictionary literals use the last value of a
                    // repeated key. Preserve that behavior, not the old first
                    // substring match, without accepting unknown entries.
                    "descr" => dtype = Some(parser.string()?),
                    "fortran_order" => fortran_order = Some(parser.boolean()?),
                    "shape" => shape = Some(parser.shape()?),
                    _ => return Err(invalid(format!("unexpected header key {key:?}"))),
                }
                if parser.take('}') {
                    break;
                }
                parser.expect(',')?;
                if parser.take('}') {
                    break;
                }
            }
        }
        for _ in 0..groups {
            parser.expect(')')?;
        }
        parser.skip_trivia();
        if !parser.rest.is_empty() {
            return Err(invalid("content follows the header dictionary"));
        }
        Ok(Self {
            dtype: dtype.ok_or_else(|| invalid("header missing descr"))?,
            fortran_order: fortran_order.ok_or_else(|| invalid("header missing fortran_order"))?,
            shape: shape.ok_or_else(|| invalid("header missing shape"))?,
        })
    }
}

fn whitespace(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\n' | '\r' | '\x0c')
}

fn dimension(axis: &str) -> Result<usize> {
    let failure = || invalid(format!("invalid shape axis {axis:?}"));
    // Shape components are integer tokens, so a comment cannot be quoted
    // within a valid component. Preserve line boundaries when removing it:
    // `2 # comment\n 0` must not turn into the integer `20`.
    let uncommented = if axis.contains('#') {
        let mut text = String::with_capacity(axis.len());
        for line in axis.lines() {
            text.push_str(line.split('#').next().unwrap_or_default());
            text.push('\n');
        }
        Cow::Owned(text)
    } else {
        Cow::Borrowed(axis)
    };
    let mut text = uncommented.trim_matches(whitespace);
    while text.starts_with('(') && text.ends_with(')') {
        text = text[1..text.len() - 1].trim_matches(whitespace);
    }
    let negative = text.starts_with('-');
    if text.starts_with(['+', '-']) {
        text = text[1..].trim_start_matches(whitespace);
    }
    let (radix, digits) = if text.starts_with("0x") || text.starts_with("0X") {
        (16, &text[2..])
    } else if text.starts_with("0o") || text.starts_with("0O") {
        (8, &text[2..])
    } else if text.starts_with("0b") || text.starts_with("0B") {
        (2, &text[2..])
    } else {
        (10, text)
    };
    // An underscore can follow a base prefix, then only separate digits.
    let digits = if radix == 10 {
        digits
    } else {
        digits.strip_prefix('_').unwrap_or(digits)
    };
    let multiplier = usize::try_from(radix).map_err(|_| failure())?;
    let mut value = 0usize;
    let mut previous_digit = false;
    for character in digits.chars() {
        if character == '_' && previous_digit {
            previous_digit = false;
            continue;
        }
        let digit = character.to_digit(radix).ok_or_else(failure)?;
        let digit = usize::try_from(digit).map_err(|_| failure())?;
        value = value
            .checked_mul(multiplier)
            .and_then(|value| value.checked_add(digit))
            .ok_or_else(failure)?;
        previous_digit = true;
    }
    if !previous_digit
        || (negative && value != 0)
        || (radix == 10 && digits.starts_with('0') && value != 0)
    {
        return Err(failure());
    }
    Ok(value)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidFormat(message.into())
}

struct Parser<'a> {
    rest: &'a str,
}

impl Parser<'_> {
    // Called only between tokens; '#' inside a quoted string is data.
    fn skip_trivia(&mut self) {
        loop {
            self.rest = self.rest.trim_start_matches(whitespace);
            if !self.rest.starts_with('#') {
                return;
            }
            self.rest = self.rest.find('\n').map_or("", |end| &self.rest[end + 1..]);
        }
    }

    fn take(&mut self, expected: char) -> bool {
        self.skip_trivia();
        if let Some(rest) = self.rest.strip_prefix(expected) {
            self.rest = rest;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: char) -> Result<()> {
        if self.take(expected) {
            Ok(())
        } else {
            Err(invalid(format!("expected {expected:?} in NPY header")))
        }
    }

    fn next(&mut self) -> Result<char> {
        let value = self
            .rest
            .chars()
            .next()
            .ok_or_else(|| invalid("unterminated header string"))?;
        self.rest = &self.rest[value.len_utf8()..];
        Ok(value)
    }

    fn string(&mut self) -> Result<String> {
        let mut value = self.quoted_string()?;
        loop {
            self.skip_trivia();
            if !self.string_start() {
                return Ok(value);
            }
            value.push_str(&self.quoted_string()?);
        }
    }

    fn string_start(&self) -> bool {
        let text = self
            .rest
            .strip_prefix(['r', 'R', 'u', 'U'])
            .unwrap_or(self.rest);
        text.starts_with(['\'', '"'])
    }

    fn quoted_string(&mut self) -> Result<String> {
        self.skip_trivia();
        let raw = self.rest.starts_with(['r', 'R']);
        if self.rest.starts_with(['r', 'R', 'u', 'U']) {
            self.next()?;
        }
        let quote = self.next()?;
        if quote != '\'' && quote != '"' {
            return Err(invalid("header key or dtype must be a quoted string"));
        }
        let pair = if quote == '\'' { "''" } else { "\"\"" };
        let triple = self.rest.starts_with(pair);
        if triple {
            self.rest = &self.rest[2..];
        }
        let mut value = String::new();
        loop {
            match self.next()? {
                character if character == quote => {
                    if !triple {
                        return Ok(value);
                    }
                    if let Some(rest) = self.rest.strip_prefix(pair) {
                        self.rest = rest;
                        return Ok(value);
                    }
                    value.push(character);
                }
                '\n' | '\r' if !triple => {
                    return Err(invalid("unescaped newline in header string"));
                }
                '\\' => {
                    let escaped = self.next()?;
                    if raw {
                        value.push('\\');
                        value.push(escaped);
                        continue;
                    }
                    match escaped {
                        '\\' | '\'' | '"' => value.push(escaped),
                        'n' => value.push('\n'),
                        'r' => value.push('\r'),
                        't' => value.push('\t'),
                        'a' => value.push('\x07'),
                        'b' => value.push('\x08'),
                        'f' => value.push('\x0c'),
                        'v' => value.push('\x0b'),
                        '\n' => {}
                        '\r' => {
                            if let Some(rest) = self.rest.strip_prefix('\n') {
                                self.rest = rest;
                            }
                        }
                        'x' => value.push(self.escape(2)?),
                        'u' => value.push(self.escape(4)?),
                        'U' => value.push(self.escape(8)?),
                        // Python retains unknown escapes verbatim. They then
                        // fail exact key/dtype comparison rather than silently
                        // changing the value.
                        _ => {
                            value.push('\\');
                            value.push(escaped);
                        }
                    }
                }
                character => value.push(character),
            }
        }
    }

    fn escape(&mut self, digits: usize) -> Result<char> {
        let mut value = 0_u32;
        for _ in 0..digits {
            let digit = self
                .next()?
                .to_digit(16)
                .ok_or_else(|| invalid("invalid hexadecimal string escape"))?;
            value = value
                .checked_mul(16)
                .and_then(|value| value.checked_add(digit))
                .ok_or_else(|| invalid("string escape overflows Unicode"))?;
        }
        char::from_u32(value).ok_or_else(|| invalid("invalid Unicode string escape"))
    }

    fn boolean(&mut self) -> Result<bool> {
        self.skip_trivia();
        for (token, value) in [("True", true), ("False", false)] {
            if let Some(rest) = self.rest.strip_prefix(token) {
                if rest
                    .starts_with(|character: char| character.is_alphanumeric() || character == '_')
                {
                    return Err(invalid("fortran_order is not a boolean"));
                }
                self.rest = rest;
                return Ok(value);
            }
        }
        Err(invalid("fortran_order is not a boolean"))
    }

    fn shape(&mut self) -> Result<Vec<usize>> {
        self.expect('(')?;
        let mut shape = Vec::new();
        if self.take(')') {
            return Ok(shape);
        }
        loop {
            self.skip_trivia();
            let mut depth = 0usize;
            let mut comment = false;
            let end = self
                .rest
                .char_indices()
                .find_map(|(index, character)| {
                    if comment {
                        comment = character != '\n';
                        return None;
                    }
                    match character {
                        '#' => comment = true,
                        '(' => depth += 1,
                        ')' if depth > 0 => depth -= 1,
                        ')' | ',' if depth == 0 => return Some(index),
                        _ => {}
                    }
                    None
                })
                .ok_or_else(|| invalid("shape tuple is unterminated"))?;
            shape.push(dimension(&self.rest[..end])?);
            self.rest = &self.rest[end..];
            if self.take(')') {
                if shape.len() == 1 {
                    return Err(invalid("one-dimensional shape needs a trailing comma"));
                }
                return Ok(shape);
            }
            self.expect(',')?;
            if self.take(')') {
                return Ok(shape);
            }
        }
    }
}
