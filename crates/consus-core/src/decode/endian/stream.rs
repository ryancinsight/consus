//! Fixed-width scalars read from and written to `std::io` streams.

use std::io::{self, Read, Write};

use super::{EndianScalar, MAX_SCALAR_WIDTH, decode_extend};
use crate::types::datatype::ByteOrder;

/// Bytes [`read_extend`] reads per step: a multiple of every scalar width,
/// small enough for the stack, and large enough that each `Read` call
/// carries thousands of scalars.
const STREAM_CHUNK_BYTES: usize = 16 * 1024;

/// Reads one fixed-width scalar from a stream.
///
/// # Errors
///
/// Returns the reader's error, including `UnexpectedEof` when the stream
/// ends before the scalar's width.
pub fn read_from<T: EndianScalar, R: Read + ?Sized>(
    reader: &mut R,
    byte_order: ByteOrder,
) -> io::Result<T> {
    let mut buf = [0_u8; MAX_SCALAR_WIDTH];
    let bytes = &mut buf[..T::BYTE_WIDTH];
    reader.read_exact(bytes)?;
    Ok(T::from_bytes(bytes, byte_order)
        .expect("invariant: the buffer holds exactly BYTE_WIDTH bytes"))
}

/// Writes one fixed-width scalar to a stream.
///
/// # Errors
///
/// Returns the writer's error.
pub fn write_to<T: EndianScalar, W: Write + ?Sized>(
    writer: &mut W,
    value: T,
    byte_order: ByteOrder,
) -> io::Result<()> {
    let mut buf = [0_u8; MAX_SCALAR_WIDTH];
    let bytes = &mut buf[..T::BYTE_WIDTH];
    value
        .to_bytes(bytes, byte_order)
        .expect("invariant: the buffer holds exactly BYTE_WIDTH bytes");
    writer.write_all(bytes)
}

/// Reads `count` scalars of `T` in `byte_order` from `reader` and appends
/// them to `out`: the streaming form of [`decode_extend`].
///
/// The stream is read in fixed steps, and `out` grows only by scalars
/// already read, so a `count` taken from an untrusted header never reserves
/// more than the stream supplies. The byte order is resolved once per step.
///
/// # Errors
///
/// Returns `UnexpectedEof` when the stream ends before `count` scalars, with
/// `out` holding every whole scalar the stream supplied, so `out.len()` is
/// the index of the first missing one. Returns the reader's other errors,
/// with `out` holding the scalars of the steps completed before, and
/// `OutOfMemory` when `out` cannot grow.
pub fn read_extend<T: EndianScalar, R: Read + ?Sized>(
    reader: &mut R,
    byte_order: ByteOrder,
    count: usize,
    out: &mut Vec<T>,
) -> io::Result<()> {
    let per_step = STREAM_CHUNK_BYTES / T::BYTE_WIDTH;
    let mut step = [0_u8; STREAM_CHUNK_BYTES];
    let mut remaining = count;
    while remaining > 0 {
        let wanted = remaining.min(per_step) * T::BYTE_WIDTH;
        let filled = fill(reader, &mut step[..wanted])?;
        let whole = filled / T::BYTE_WIDTH;
        out.try_reserve(whole)
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        decode_extend(&step[..whole * T::BYTE_WIDTH], byte_order, out, |value| {
            value
        })
        .expect("invariant: the slice holds whole scalars");
        if filled < wanted {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "stream ended before the requested scalars",
            ));
        }
        remaining -= whole;
    }
    Ok(())
}

/// Reads into `buf` until it is full or the stream ends, returning the bytes
/// read; an interrupted read is retried.
fn fill<R: Read + ?Sized>(reader: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

#[cfg(test)]
mod tests;
