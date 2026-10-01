//! Tests for the stream scalar reads and writes.

use std::io::{self, Read};

use super::{STREAM_CHUNK_BYTES, read_extend, read_from, write_to};
use crate::types::datatype::ByteOrder;

#[test]
fn streams_read_back_what_they_write() {
    let mut stream = Vec::new();
    write_to(&mut stream, 0x1234_u16, ByteOrder::BigEndian).expect("a Vec accepts writes");
    write_to(&mut stream, -1.5_f32, ByteOrder::LittleEndian).expect("a Vec accepts writes");
    write_to(&mut stream, -7_i64, ByteOrder::BigEndian).expect("a Vec accepts writes");
    assert_eq!(stream.len(), 2 + 4 + 8);
    assert_eq!(&stream[..2], &[0x12, 0x34]);

    let mut reader = stream.as_slice();
    assert_eq!(
        read_from::<u16, _>(&mut reader, ByteOrder::BigEndian).ok(),
        Some(0x1234)
    );
    assert_eq!(
        read_from::<f32, _>(&mut reader, ByteOrder::LittleEndian).ok(),
        Some(-1.5)
    );
    assert_eq!(
        read_from::<i64, _>(&mut reader, ByteOrder::BigEndian).ok(),
        Some(-7)
    );
    let end =
        read_from::<u8, _>(&mut reader, ByteOrder::BigEndian).expect_err("the stream is exhausted");
    assert_eq!(end.kind(), std::io::ErrorKind::UnexpectedEof);
}

/// Every `T` decoded from `values` written one scalar at a time.
fn written<T: super::EndianScalar + Copy>(values: &[T], order: ByteOrder) -> Vec<u8> {
    let mut stream = Vec::new();
    for &value in values {
        write_to(&mut stream, value, order).expect("a Vec accepts writes");
    }
    stream
}

/// A reader yielding at most `limit` bytes per call, as a gzip or socket
/// stream does.
struct Trickle<'a> {
    bytes: &'a [u8],
    limit: usize,
}

impl Read for Trickle<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = buf.len().min(self.limit).min(self.bytes.len());
        buf[..n].copy_from_slice(&self.bytes[..n]);
        self.bytes = &self.bytes[n..];
        Ok(n)
    }
}

/// Values spanning several read steps, so the step boundary is crossed with
/// a partial final step.
fn long_values() -> Vec<i32> {
    let count = 2 * (STREAM_CHUNK_BYTES / 4) + 3;
    (0..count)
        .map(|index| i32::try_from(index).expect("invariant: count fits i32") * -7 + 3)
        .collect()
}

#[test]
fn read_extend_matches_scalar_writes_across_steps() {
    let values = long_values();
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let stream = written(&values, order);
        let mut out = vec![99_i32];
        read_extend(
            &mut Trickle {
                bytes: &stream,
                limit: 1000,
            },
            order,
            values.len(),
            &mut out,
        )
        .expect("the stream holds every scalar");
        assert_eq!(out[0], 99, "{order:?}");
        assert_eq!(&out[1..], values.as_slice(), "{order:?}");
    }
}

#[test]
fn read_extend_leaves_the_rest_of_the_stream() {
    let stream = written(&[1.5_f64, -2.0, 4.25], ByteOrder::BigEndian);
    let mut reader = stream.as_slice();
    let mut out = Vec::new();
    read_extend::<f64, _>(&mut reader, ByteOrder::BigEndian, 2, &mut out)
        .expect("two scalars are present");
    assert_eq!(out, [1.5, -2.0]);
    assert_eq!(
        read_from::<f64, _>(&mut reader, ByteOrder::BigEndian).ok(),
        Some(4.25)
    );
}

#[test]
fn read_extend_of_nothing_reads_nothing() {
    let mut reader: &[u8] = &[7];
    let mut out: Vec<u16> = Vec::new();
    read_extend(&mut reader, ByteOrder::LittleEndian, 0, &mut out).expect("no scalars");
    assert!(out.is_empty());
    assert_eq!(reader, [7]);
}

/// A header claiming far more scalars than the stream holds fails at the end
/// of the stream, keeping every scalar read and reserving only what was read.
#[test]
fn read_extend_bounds_growth_by_the_stream() {
    let values = long_values();
    let stream = written(&values, ByteOrder::LittleEndian);
    let mut out = Vec::new();
    let err = read_extend::<i32, _>(
        &mut stream.as_slice(),
        ByteOrder::LittleEndian,
        usize::MAX / 4,
        &mut out,
    )
    .expect_err("the stream ends first");
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    assert_eq!(out, values);
    assert!(out.capacity() <= 2 * values.len(), "{}", out.capacity());
}

/// A stream ending inside a scalar keeps the whole scalars before it, so the
/// output length names the first missing scalar.
#[test]
fn read_extend_keeps_whole_scalars_before_a_partial_one() {
    let values = long_values();
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let mut stream = written(&values, order);
        stream.pop();
        let mut out = Vec::new();
        let err = read_extend::<i32, _>(
            &mut Trickle {
                bytes: &stream,
                limit: 333,
            },
            order,
            values.len(),
            &mut out,
        )
        .expect_err("one byte is missing");
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof, "{order:?}");
        assert_eq!(out, values[..values.len() - 1], "{order:?}");
    }
}
