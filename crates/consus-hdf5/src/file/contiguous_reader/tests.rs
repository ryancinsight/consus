//! Contract tests for the contiguous payload stream.

use std::error::Error as _;
use std::io::{self, Read};

use consus_core::{ByteOrder, Datatype, Error, Shape};
use consus_io::MemCursor;
use core::num::NonZeroUsize;

use super::io_error;
use crate::file::Hdf5File;
use crate::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};

const PAYLOAD_LEN: usize = 1000;

/// Bytes with period 251, a prime that divides neither the payload length nor
/// any tested step, so a shifted or repeated window never matches the source.
fn pattern() -> Vec<u8> {
    (0..PAYLOAD_LEN).map(|index| (index % 251) as u8).collect()
}

/// A file holding one contiguous `u8` dataset; returns the file, the payload
/// address, and the total file length in bytes.
fn file_with_payload(payload: &[u8]) -> (Hdf5File<MemCursor>, u64, u64) {
    let datatype = Datatype::Integer {
        bits: NonZeroUsize::new(8).expect("invariant: 8 is non-zero"),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    builder
        .add_dataset(
            "values",
            &datatype,
            &Shape::fixed(&[payload.len()]),
            payload,
            &DatasetCreationProps::default(),
        )
        .expect("add dataset");
    let bytes = builder.finish().expect("finish file");
    let file_len = bytes.len() as u64;
    let file = Hdf5File::open(MemCursor::from_bytes(bytes)).expect("open file");
    let header = file.open_path("values").expect("locate dataset");
    let data_address = file
        .dataset_at(header)
        .expect("read dataset metadata")
        .data_address
        .expect("contiguous dataset has a payload address");
    (file, data_address, file_len)
}

/// Drain `reader` through a fixed-size buffer, recording every read length.
fn drain_in_steps(reader: &mut impl Read, step: usize) -> (Vec<u8>, Vec<usize>) {
    let mut buf = vec![0u8; step];
    let mut bytes = Vec::new();
    let mut lengths = Vec::new();
    loop {
        let count = reader.read(&mut buf).expect("read step");
        if count == 0 {
            return (bytes, lengths);
        }
        bytes.extend_from_slice(&buf[..count]);
        lengths.push(count);
    }
}

#[test]
fn read_to_end_returns_the_whole_payload() {
    let payload = pattern();
    let (file, address, _) = file_with_payload(&payload);
    let mut streamed = Vec::new();
    file.contiguous_dataset_reader(address, payload.len() as u64)
        .read_to_end(&mut streamed)
        .expect("stream payload");
    assert_eq!(streamed, payload);
}

#[test]
fn one_byte_and_seven_byte_buffers_reassemble_the_payload() {
    let payload = pattern();
    let (file, address, _) = file_with_payload(&payload);
    for step in [1usize, 7] {
        let mut reader = file.contiguous_dataset_reader(address, payload.len() as u64);
        let (streamed, lengths) = drain_in_steps(&mut reader, step);
        assert_eq!(streamed, payload, "step {step}");
        // Every read but the last fills the buffer and the last carries the
        // remainder, so the read lengths are fully determined by the step.
        let mut expected = vec![step; payload.len() / step];
        if !payload.len().is_multiple_of(step) {
            expected.push(payload.len() % step);
        }
        assert_eq!(lengths, expected, "step {step}");
        assert_eq!(reader.remaining(), 0);
    }
}

#[test]
fn a_read_never_crosses_the_declared_length() {
    let payload = pattern();
    let (file, address, _) = file_with_payload(&payload);
    let mut reader = file.contiguous_dataset_reader(address, 10);
    let mut buf = [0xFFu8; 64];
    assert_eq!(reader.read(&mut buf).expect("read"), 10);
    assert_eq!(&buf[..10], &payload[..10]);
    assert_eq!(&buf[10..], &[0xFFu8; 54], "bytes past len stay untouched");
    assert_eq!(reader.read(&mut buf).expect("read at end"), 0);
}

#[test]
fn a_reader_over_a_window_yields_exactly_that_window() {
    let payload = pattern();
    let (file, address, _) = file_with_payload(&payload);
    let mut streamed = Vec::new();
    file.contiguous_dataset_reader(address + 100, 37)
        .read_to_end(&mut streamed)
        .expect("stream window");
    assert_eq!(streamed, &payload[100..137]);
}

#[test]
fn zero_length_reads_nothing_and_touches_no_bytes() {
    let payload = pattern();
    let (file, address, _) = file_with_payload(&payload);
    let mut reader = file.contiguous_dataset_reader(address, 0);
    let mut buf = [0xAAu8; 8];
    assert_eq!(reader.read(&mut buf).expect("read"), 0);
    assert_eq!(buf, [0xAAu8; 8]);
    let mut streamed = Vec::new();
    assert_eq!(reader.read_to_end(&mut streamed).expect("drain"), 0);
    assert!(streamed.is_empty());
}

#[test]
fn an_empty_buffer_reads_nothing_and_does_not_advance() {
    let payload = pattern();
    let (file, address, _) = file_with_payload(&payload);
    let mut reader = file.contiguous_dataset_reader(address, 5);
    assert_eq!(reader.read(&mut []).expect("read"), 0);
    assert_eq!(reader.remaining(), 5);
}

#[test]
fn a_range_past_the_file_end_keeps_the_consus_error_as_inner_error() {
    let payload = pattern();
    let (file, address, file_len) = file_with_payload(&payload);
    // Bytes from `address` to the file end exist; one more lies outside the
    // source.
    let len = file_len - address + 1;
    let mut streamed = Vec::new();
    let error = file
        .contiguous_dataset_reader(address, len)
        .read_to_end(&mut streamed)
        .expect_err("range exceeds the file");
    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    let inner = error
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<Error>())
        .expect("inner error is the consus error, not a string");
    assert!(
        matches!(inner, Error::BufferTooSmall { .. }),
        "unexpected inner error: {inner:?}"
    );
    // std defines `io::Error::source` as the inner error's own source, and a
    // `BufferTooSmall` has none.
    assert!(error.source().is_none());
}

#[test]
fn a_failed_read_leaves_the_position_unchanged() {
    let payload = pattern();
    let (file, address, file_len) = file_with_payload(&payload);
    let len = file_len - address + 1;
    let mut reader = file.contiguous_dataset_reader(address, len);
    let mut buf = vec![0u8; usize::try_from(len).expect("length fits usize")];
    assert!(reader.read(&mut buf).is_err());
    assert_eq!(reader.remaining(), len);
}

#[test]
fn an_address_range_overflowing_u64_is_an_error_not_a_panic() {
    let payload = pattern();
    let (file, _, _) = file_with_payload(&payload);
    let mut reader = file.contiguous_dataset_reader(u64::MAX - 3, 16);
    let mut buf = [0u8; 16];
    let error = reader.read(&mut buf).expect_err("range overflows");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(matches!(
        error
            .get_ref()
            .and_then(|inner| inner.downcast_ref::<Error>()),
        Some(Error::Overflow)
    ));
}

#[test]
fn io_error_kinds_follow_the_failure_class() {
    let cases = [
        (
            Error::Io(io::Error::from(io::ErrorKind::TimedOut)),
            io::ErrorKind::TimedOut,
        ),
        (
            Error::BufferTooSmall {
                required: 2,
                provided: 1,
            },
            io::ErrorKind::UnexpectedEof,
        ),
        (
            Error::NotFound {
                path: String::from("/a"),
            },
            io::ErrorKind::NotFound,
        ),
        (Error::ReadOnly, io::ErrorKind::PermissionDenied),
        (
            Error::UnsupportedFeature {
                feature: String::from("x"),
            },
            io::ErrorKind::Unsupported,
        ),
        (Error::Overflow, io::ErrorKind::InvalidData),
        (
            Error::Corrupted {
                message: String::from("x"),
            },
            io::ErrorKind::InvalidData,
        ),
        (Error::InvalidHandle, io::ErrorKind::Other),
    ];
    for (error, kind) in cases {
        let text = error.to_string();
        let converted = io_error(error);
        assert_eq!(converted.kind(), kind, "{text}");
        assert_eq!(converted.to_string(), text, "message is the consus message");
    }
}
