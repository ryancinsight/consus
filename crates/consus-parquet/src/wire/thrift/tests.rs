use super::*;

#[test]
fn read_byte_works() {
    let mut r = ThriftReader::new(&[0x42, 0x17]);
    assert_eq!(r.read_byte().unwrap(), 0x42);
    assert_eq!(r.read_byte().unwrap(), 0x17);
    assert_eq!(r.remaining(), 0);
}

#[test]
fn varint_single_byte() {
    let mut r = ThriftReader::new(&[0x05]);
    assert_eq!(r.read_varint_u64().unwrap(), 5);
}

#[test]
fn varint_multi_byte() {
    // 160 = 0b10100000; varint: low 7 bits = 32 with high set -> 0xA0, remaining 1 -> 0x01
    let mut r = ThriftReader::new(&[0xA0, 0x01]);
    assert_eq!(r.read_varint_u64().unwrap(), 160);
}

#[test]
fn zigzag_i32_positive() {
    // zigzag(2)=4 -> decode(4)=2; zigzag(1)=2 -> decode(2)=1
    let mut r = ThriftReader::new(&[0x04]);
    assert_eq!(r.read_i32().unwrap(), 2);
    let mut r2 = ThriftReader::new(&[0x02]);
    assert_eq!(r2.read_i32().unwrap(), 1);
}

#[test]
fn zigzag_i32_negative() {
    // zigzag(-1)=1 -> decode(1)=-1; zigzag(-2)=3 -> decode(3)=-2
    let mut r = ThriftReader::new(&[0x01]);
    assert_eq!(r.read_i32().unwrap(), -1);
    let mut r2 = ThriftReader::new(&[0x03]);
    assert_eq!(r2.read_i32().unwrap(), -2);
}

#[test]
fn zigzag_i64_positive() {
    // zigzag(10)=20=0x14
    let mut r = ThriftReader::new(&[0x14]);
    assert_eq!(r.read_i64().unwrap(), 10);
}

#[test]
fn read_binary_works() {
    let mut r = ThriftReader::new(&[0x03, b'a', b'b', b'c']);
    assert_eq!(r.read_binary().unwrap(), b"abc");
}

#[test]
fn read_string_works() {
    let mut r = ThriftReader::new(&[0x05, b'h', b'e', b'l', b'l', b'o']);
    assert_eq!(r.read_string().unwrap(), "hello");
}

#[test]
fn field_header_stop_returns_none() {
    let mut r = ThriftReader::new(&[0x00]);
    let mut last = 0i16;
    assert_eq!(r.read_field_header(&mut last).unwrap(), None);
}

#[test]
fn field_header_with_delta() {
    // 0x15 = (1<<4)|0x05 = delta=1, type=I32
    let mut r = ThriftReader::new(&[0x15]);
    let mut last = 0i16;
    let result = r.read_field_header(&mut last).unwrap();
    assert_eq!(result, Some((1, 0x05)));
    assert_eq!(last, 1);
}

#[test]
fn field_header_absolute() {
    // 0x05 = delta=0, type=I32; then zigzag i16: 0x04 = zigzag(2) = 2
    let mut r = ThriftReader::new(&[0x05, 0x04]);
    let mut last = 0i16;
    let result = r.read_field_header(&mut last).unwrap();
    assert_eq!(result, Some((2, 0x05)));
    assert_eq!(last, 2);
}

#[test]
fn list_header_short_form() {
    // 0x25 = (2<<4)|0x05 = count=2, elem_type=I32(0x05)
    let mut r = ThriftReader::new(&[0x25, 0x00, 0x00]);
    let (elem_type, count) = r.read_list_header().unwrap();
    assert_eq!(elem_type, 0x05);
    assert_eq!(count, 2);
}

#[test]
fn list_header_long_form() {
    // 0xF5 = (0xF<<4)|0x05 = long form, elem_type=I32; then varint 3
    let mut r = ThriftReader::new(&[0xF5, 0x03, 0x00, 0x00, 0x00]);
    let (elem_type, count) = r.read_list_header().unwrap();
    assert_eq!(elem_type, 0x05);
    assert_eq!(count, 3);
}

#[test]
fn list_header_rejects_count_larger_than_remaining_bytes() {
    let mut r = ThriftReader::new(&[0xF5, 0xF9, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);

    let error = r.read_list_header().unwrap_err();

    match error {
        consus_core::Error::InvalidFormat { message } => {
            assert!(message.contains("list count"));
            assert!(message.contains("exceeds remaining bytes 0"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn map_header_rejects_count_larger_than_remaining_entries() {
    let mut r = ThriftReader::new(&[0xF9, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);

    let error = r.read_map_header().unwrap_err();

    match error {
        consus_core::Error::InvalidFormat { message } => {
            assert!(message.contains("map count"));
            assert!(message.contains("exceeds remaining bytes 0"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn skip_i32() {
    // skip(0x05) reads the varint value: 0x04
    let mut r = ThriftReader::new(&[0x04]);
    r.skip(0x05).unwrap();
    assert_eq!(r.position(), 1);
}

#[test]
fn skip_binary() {
    // varint(3) = 0x03, then "abc"
    let mut r = ThriftReader::new(&[0x03, b'a', b'b', b'c']);
    r.skip(0x08).unwrap();
    assert_eq!(r.position(), 4);
}

#[test]
fn skip_binary_rejects_length_offset_overflow() {
    // Ten-byte u64 varint for usize::MAX; adding it to the current
    // position must return a typed overflow instead of wrapping.
    let mut r = ThriftReader::new(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]);
    assert!(matches!(r.skip(0x08), Err(consus_core::Error::Overflow)));
}

#[test]
fn skip_struct() {
    // struct with one i32 field (field 1) then stop
    // field header: 0x15 (delta=1, I32), value: 0x02 (zigzag(1)=1), stop: 0x00
    let mut r = ThriftReader::new(&[0x15, 0x02, 0x00]);
    r.skip(0x0C).unwrap();
    assert_eq!(r.position(), 3);
}

#[test]
fn read_byte_exhausted_returns_error() {
    let mut r = ThriftReader::new(&[]);
    let err = r.read_byte().unwrap_err();
    assert!(matches!(err, consus_core::Error::BufferTooSmall { .. }));
}
