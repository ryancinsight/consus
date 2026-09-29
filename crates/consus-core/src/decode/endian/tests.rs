//! Tests for the fixed-width scalar reads and writes.

use super::{
    EndianScalar, read_int_width, read_integer, read_uint_arbitrary, read_uint_width, sign_extend,
    write_integer,
};
use crate::types::datatype::ByteOrder;

fn assert_round_trip<T>(little: &[u8], big: &[u8], expected: T)
where
    T: EndianScalar + Copy + PartialEq + core::fmt::Debug,
{
    assert_eq!(
        read_integer(little, ByteOrder::LittleEndian),
        Some(expected)
    );
    assert_eq!(read_integer(big, ByteOrder::BigEndian), Some(expected));
    assert_eq!(read_integer::<T>(&[], ByteOrder::LittleEndian), None);
}

#[test]
fn reads_all_supported_scalar_widths_and_orders() {
    assert_round_trip(&[0x34, 0x12], &[0x12, 0x34], 0x1234_u16);
    assert_round_trip(&[0xCC, 0xED], &[0xED, 0xCC], -4_660_i16);
    assert_round_trip(
        &[0x78, 0x56, 0x34, 0x12],
        &[0x12, 0x34, 0x56, 0x78],
        0x1234_5678_u32,
    );
    assert_round_trip(
        &[0x88, 0xA9, 0xCB, 0xED],
        &[0xED, 0xCB, 0xA9, 0x88],
        -305_419_896_i32,
    );
    assert_round_trip(
        &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
        &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        0x0102_0304_0506_0708_u64,
    );
    assert_round_trip(
        &[0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF],
        &[0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8],
        -283_686_952_306_184_i64,
    );
}

#[test]
fn reads_narrow_scalars_and_floats() {
    assert_eq!(
        read_integer::<u8>(&[0xAB], ByteOrder::LittleEndian),
        Some(0xAB)
    );
    assert_eq!(
        read_integer::<i8>(&[0xFF], ByteOrder::BigEndian),
        Some(-1_i8)
    );
    assert_round_trip(
        &[0x00, 0x00, 0xC0, 0x3F],
        &[0x3F, 0xC0, 0x00, 0x00],
        1.5_f32,
    );
    assert_round_trip(
        &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x40],
        &[0x40, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        2.5_f64,
    );
}

#[test]
fn read_uint_width_covers_arbitrary_widths() {
    let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    assert_eq!(read_uint_width(&data, 0, ByteOrder::LittleEndian), Some(0));
    assert_eq!(
        read_uint_width(&data, 3, ByteOrder::LittleEndian),
        Some(0x0003_0201)
    );
    assert_eq!(
        read_uint_width(&data, 3, ByteOrder::BigEndian),
        Some(0x0001_0203)
    );
    assert_eq!(
        read_uint_width(&data, 8, ByteOrder::LittleEndian),
        Some(0x0807_0605_0403_0201)
    );
    assert_eq!(read_uint_width(&data, 9, ByteOrder::LittleEndian), None);
    assert_eq!(
        read_uint_width(&data[..2], 4, ByteOrder::LittleEndian),
        None
    );
    assert_eq!(
        read_uint_arbitrary::<2>(&data, ByteOrder::LittleEndian),
        0x0201
    );
}

#[test]
fn read_int_width_sign_extends() {
    assert_eq!(
        read_int_width(&[0xFF], 1, ByteOrder::LittleEndian),
        Some(-1)
    );
    assert_eq!(
        read_int_width(&[0xFE, 0xFF], 2, ByteOrder::LittleEndian),
        Some(-2)
    );
    assert_eq!(
        read_int_width(&[0x00, 0x80], 2, ByteOrder::LittleEndian),
        Some(-32768)
    );
    assert_eq!(sign_extend(0, 2), 0);
}

fn assert_write_round_trip<T>(value: T, little: &[u8], big: &[u8])
where
    T: EndianScalar + Copy + PartialEq + core::fmt::Debug,
{
    for (order, expected) in [
        (ByteOrder::LittleEndian, little),
        (ByteOrder::BigEndian, big),
    ] {
        let mut out = [0xAA_u8; 10];
        assert_eq!(write_integer(&mut out, value, order), Some(()));
        assert_eq!(&out[..T::BYTE_WIDTH], expected);
        assert!(out[T::BYTE_WIDTH..].iter().all(|&byte| byte == 0xAA));
        assert_eq!(read_integer::<T>(&out, order), Some(value));
    }
    let mut short = [0xAA_u8; 10];
    let short = &mut short[..T::BYTE_WIDTH - 1];
    assert_eq!(write_integer(short, value, ByteOrder::BigEndian), None);
    assert!(short.iter().all(|&byte| byte == 0xAA));
}

#[test]
fn writes_every_scalar_in_both_orders() {
    assert_write_round_trip(0xAB_u8, &[0xAB], &[0xAB]);
    assert_write_round_trip(-2_i8, &[0xFE], &[0xFE]);
    assert_write_round_trip(0x1234_u16, &[0x34, 0x12], &[0x12, 0x34]);
    assert_write_round_trip(-2_i16, &[0xFE, 0xFF], &[0xFF, 0xFE]);
    assert_write_round_trip(
        0x1234_5678_u32,
        &[0x78, 0x56, 0x34, 0x12],
        &[0x12, 0x34, 0x56, 0x78],
    );
    assert_write_round_trip(-2_i32, &[0xFE, 0xFF, 0xFF, 0xFF], &[0xFF, 0xFF, 0xFF, 0xFE]);
    assert_write_round_trip(
        0x0102_0304_0506_0708_u64,
        &[8, 7, 6, 5, 4, 3, 2, 1],
        &[1, 2, 3, 4, 5, 6, 7, 8],
    );
    assert_write_round_trip(
        -2_i64,
        &[0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE],
    );
    assert_write_round_trip(1.5_f32, &[0, 0, 0xC0, 0x3F], &[0x3F, 0xC0, 0, 0]);
    assert_write_round_trip(
        -0.25_f64,
        &[0, 0, 0, 0, 0, 0, 0xD0, 0xBF],
        &[0xBF, 0xD0, 0, 0, 0, 0, 0, 0],
    );
}

#[cfg(feature = "std")]
#[test]
fn streams_read_back_what_they_write() {
    use super::{read_from, write_to};

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
