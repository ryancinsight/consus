//! Cursor-based sequential reader over positioned I/O.
//!
//! ## Design
//!
//! `StreamReader<R>` wraps any `ReadAt + Length` source and provides
//! `Seekable` access via an internal cursor position. This enables
//! format parsers to read sequentially without tracking offsets manually.
//!
//! ## Invariants
//!
//! - The cursor position is always in `[0, u64::MAX]`.
//! - `read_stream(buf)` advances the cursor by `buf.len()` on success.
//! - `seek(SeekFrom::Start(n))` sets cursor to `n`.
//! - `seek(SeekFrom::End(n))` sets cursor to `source.len() + n` (signed).
//! - `seek(SeekFrom::Current(n))` sets cursor to `pos + n` (signed).
//! - Seeking to a negative absolute position returns `Error::Overflow`.

use consus_core::{Error, Result};

use crate::io::traits::{Length, ReadAt, SeekFrom, Seekable};

/// Cursor-based sequential reader wrapping a positioned I/O source.
///
/// Converts any `ReadAt + Length` into a seekable stream. The internal
/// cursor tracks the current read position.
///
/// # Type Parameters
///
/// - `R`: The underlying positioned I/O source.
///
/// # Examples
///
/// ```
/// use consus_io::io::sync::cursor::MemCursor;
/// use consus_io::io::sync::stream::StreamReader;
/// use consus_io::io::traits::{WriteAt, SeekFrom, Seekable};
///
/// let mut mc = MemCursor::from_bytes(vec![10, 20, 30, 40, 50]);
/// let mut stream = StreamReader::new(mc);
///
/// let mut buf = [0u8; 2];
/// stream.read_stream(&mut buf).unwrap();
/// assert_eq!(buf, [10, 20]);
///
/// stream.seek(SeekFrom::Current(1)).unwrap();
/// stream.read_stream(&mut buf).unwrap();
/// assert_eq!(buf, [40, 50]);
/// ```
#[derive(Debug)]
pub struct StreamReader<R> {
    /// Underlying positioned I/O source.
    source: R,
    /// Current cursor position in bytes.
    pos: u64,
}

impl<R> StreamReader<R> {
    /// Create a new stream reader with the cursor at position 0.
    #[must_use]
    pub fn new(source: R) -> Self {
        Self { source, pos: 0 }
    }

    /// Create a new stream reader with the cursor at `pos`.
    #[must_use]
    pub fn with_position(source: R, pos: u64) -> Self {
        Self { source, pos }
    }

    /// Borrow the underlying source.
    #[must_use]
    pub fn inner(&self) -> &R {
        &self.source
    }

    /// Mutably borrow the underlying source.
    pub fn inner_mut(&mut self) -> &mut R {
        &mut self.source
    }

    /// Consume the stream reader and return the underlying source.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.source
    }

    /// Current cursor position.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.pos
    }
}

impl<R: ReadAt> StreamReader<R> {
    /// Read exactly `buf.len()` bytes at the current cursor position
    /// and advance the cursor by `buf.len()`.
    ///
    /// # Errors
    ///
    /// Returns `Error::BufferTooSmall` if the source has insufficient
    /// bytes from the current position.
    pub fn read_stream(&mut self, buf: &mut [u8]) -> Result<()> {
        self.source.read_at(self.pos, buf)?;
        self.pos = self
            .pos
            .checked_add(buf.len() as u64)
            .ok_or(Error::Overflow)?;
        Ok(())
    }
}

impl<R: ReadAt + Length> Seekable for StreamReader<R> {
    fn seek(&mut self, from: SeekFrom) -> Result<u64> {
        let new_pos = match from {
            SeekFrom::Start(offset) => offset,
            SeekFrom::End(offset) => {
                let len = self.source.len()?;
                if offset >= 0 {
                    len.checked_add(offset as u64).ok_or(Error::Overflow)?
                } else {
                    let abs = (-offset) as u64;
                    if abs > len {
                        return Err(Error::Overflow);
                    }
                    len - abs
                }
            }
            SeekFrom::Current(offset) => {
                if offset >= 0 {
                    self.pos.checked_add(offset as u64).ok_or(Error::Overflow)?
                } else {
                    let abs = (-offset) as u64;
                    if abs > self.pos {
                        return Err(Error::Overflow);
                    }
                    self.pos - abs
                }
            }
        };
        self.pos = new_pos;
        Ok(new_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::sync::cursor::MemCursor;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    /// Sequential read advances cursor.
    #[test]
    fn sequential_read_advances_cursor() {
        let mc = MemCursor::from_bytes(vec![10, 20, 30, 40, 50]);
        let mut stream = StreamReader::new(mc);

        assert_eq!(stream.position(), 0);

        let mut buf = [0u8; 2];
        stream.read_stream(&mut buf).unwrap();
        assert_eq!(buf, [10, 20]);
        assert_eq!(stream.position(), 2);

        stream.read_stream(&mut buf).unwrap();
        assert_eq!(buf, [30, 40]);
        assert_eq!(stream.position(), 4);
    }

    /// Seek to absolute position.
    #[test]
    fn seek_start() {
        let mc = MemCursor::from_bytes(vec![0; 100]);
        let mut stream = StreamReader::new(mc);
        let pos = stream.seek(SeekFrom::Start(42)).unwrap();
        assert_eq!(pos, 42);
        assert_eq!(stream.position(), 42);
    }

    /// Seek relative to current position (forward).
    #[test]
    fn seek_current_forward() {
        let mc = MemCursor::from_bytes(vec![0; 100]);
        let mut stream = StreamReader::with_position(mc, 10);
        let pos = stream.seek(SeekFrom::Current(5)).unwrap();
        assert_eq!(pos, 15);
    }

    /// Seek relative to current position (backward).
    #[test]
    fn seek_current_backward() {
        let mc = MemCursor::from_bytes(vec![0; 100]);
        let mut stream = StreamReader::with_position(mc, 20);
        let pos = stream.seek(SeekFrom::Current(-5)).unwrap();
        assert_eq!(pos, 15);
    }

    /// Seek before start returns Overflow error.
    #[test]
    fn seek_before_start_fails() {
        let mc = MemCursor::from_bytes(vec![0; 100]);
        let mut stream = StreamReader::with_position(mc, 3);
        let err = stream.seek(SeekFrom::Current(-10)).unwrap_err();
        match err {
            Error::Overflow => {}
            other => panic!("expected Overflow, got: {other}"),
        }
    }

    /// Seek from end (negative offset).
    #[test]
    fn seek_end_negative() {
        let mc = MemCursor::from_bytes(vec![0; 100]);
        let mut stream = StreamReader::new(mc);
        let pos = stream.seek(SeekFrom::End(-10)).unwrap();
        assert_eq!(pos, 90);
    }

    /// Seek from end (zero offset = at end).
    #[test]
    fn seek_end_zero() {
        let mc = MemCursor::from_bytes(vec![0; 50]);
        let mut stream = StreamReader::new(mc);
        let pos = stream.seek(SeekFrom::End(0)).unwrap();
        assert_eq!(pos, 50);
    }

    /// Seek from end past start returns Overflow.
    #[test]
    fn seek_end_past_start_fails() {
        let mc = MemCursor::from_bytes(vec![0; 10]);
        let mut stream = StreamReader::new(mc);
        let err = stream.seek(SeekFrom::End(-20)).unwrap_err();
        match err {
            Error::Overflow => {}
            other => panic!("expected Overflow, got: {other}"),
        }
    }

    /// `stream_position` returns current position without moving.
    #[test]
    fn stream_position_idempotent() {
        let mc = MemCursor::from_bytes(vec![0; 100]);
        let mut stream = StreamReader::with_position(mc, 42);
        assert_eq!(stream.stream_position().unwrap(), 42);
        assert_eq!(stream.stream_position().unwrap(), 42);
    }

    /// Read then seek then read.
    #[test]
    fn read_seek_read() {
        let mc = MemCursor::from_bytes(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let mut stream = StreamReader::new(mc);

        let mut buf = [0u8; 2];
        stream.read_stream(&mut buf).unwrap();
        assert_eq!(buf, [1, 2]);

        stream.seek(SeekFrom::Start(6)).unwrap();
        stream.read_stream(&mut buf).unwrap();
        assert_eq!(buf, [7, 8]);
    }

    /// Read past end fails.
    #[test]
    fn read_past_end_fails() {
        let mc = MemCursor::from_bytes(vec![1, 2, 3]);
        let mut stream = StreamReader::with_position(mc, 2);

        let mut buf = [0u8; 4];
        let result = stream.read_stream(&mut buf);
        assert!(result.is_err());
        // Cursor must not advance on failure.
        assert_eq!(stream.position(), 2);
    }

    /// `into_inner` returns the wrapped source.
    #[test]
    fn into_inner_returns_source() {
        let mc = MemCursor::from_bytes(vec![1, 2, 3]);
        let stream = StreamReader::new(mc);
        let recovered = stream.into_inner();
        assert_eq!(recovered.as_bytes(), &[1, 2, 3]);
    }
}
