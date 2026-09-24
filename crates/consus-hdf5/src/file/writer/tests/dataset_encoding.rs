//! Datatype, dataspace, and layout message encoding tests.

use super::super::*;
use crate::address::ParseContext;
use crate::property_list::{DatasetCreationProps, DatasetLayout};
use byteorder::{ByteOrder as _, LittleEndian};
use consus_core::ByteOrder;
use consus_core::{Datatype, Shape};
use core::num::NonZeroUsize;

#[cfg(feature = "alloc")]
#[test]
fn encode_integer_datatype_le_u32() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(32).unwrap(),
        byte_order: ByteOrder::LittleEndian,
        signed: false,
    };
    let bytes = encode_datatype(&dt).unwrap();
    assert_eq!(bytes.len(), 12);
    assert_eq!(bytes[0] & 0x0F, 0); // class 0
    assert_eq!(bytes[1] & 0x01, 0); // LE
    assert_eq!(bytes[1] & 0x08, 0); // unsigned
    assert_eq!(LittleEndian::read_u32(&bytes[4..8]), 4); // 4 bytes
}

#[cfg(feature = "alloc")]
#[test]
fn encode_integer_datatype_be_i16() {
    let dt = Datatype::Integer {
        bits: NonZeroUsize::new(16).unwrap(),
        byte_order: ByteOrder::BigEndian,
        signed: true,
    };
    let bytes = encode_datatype(&dt).unwrap();
    assert_eq!(bytes[1] & 0x01, 1); // BE
    assert_eq!(bytes[1] & 0x08, 0x08); // signed
    assert_eq!(LittleEndian::read_u32(&bytes[4..8]), 2); // 2 bytes
}

#[cfg(feature = "alloc")]
#[test]
fn encode_float_f64() {
    let dt = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ByteOrder::LittleEndian,
    };
    let bytes = encode_datatype(&dt).unwrap();
    assert_eq!(bytes.len(), 20);
    assert_eq!(bytes[0] & 0x0F, 1); // class 1 (float)
    assert_eq!(bytes[1], 0x20); // mantissa norm=2 (implicit), LE
    assert_eq!(bytes[2], 63); // sign bit at position 63
    assert_eq!(LittleEndian::read_u32(&bytes[4..8]), 8); // 8 bytes
    assert_eq!(bytes[12], 52); // exponent position
    assert_eq!(bytes[13], 11); // exponent size
    assert_eq!(LittleEndian::read_u32(&bytes[16..20]), 1023); // bias
}

#[cfg(feature = "alloc")]
#[test]
fn encode_dataspace_scalar() {
    let shape = Shape::scalar();
    let bytes = encode_dataspace(&shape).unwrap();
    assert_eq!(bytes.len(), 4);
    assert_eq!(bytes[0], 2); // version
    assert_eq!(bytes[1], 0); // rank 0
    assert_eq!(bytes[3], 0); // type: scalar
}

#[cfg(feature = "alloc")]
#[test]
fn encode_dataspace_2d() {
    let shape = Shape::fixed(&[10, 20]);
    let bytes = encode_dataspace(&shape).unwrap();
    assert_eq!(bytes.len(), 4 + 16); // header + 2×8 dims
    assert_eq!(bytes[1], 2); // rank 2
    assert_eq!(LittleEndian::read_u64(&bytes[4..12]), 10);
    assert_eq!(LittleEndian::read_u64(&bytes[12..20]), 20);
}

#[cfg(feature = "alloc")]
#[test]
fn encode_contiguous_layout() {
    let ctx = ParseContext::new(8, 8);
    let props = DatasetCreationProps::default(); // contiguous
    let bytes = encode_layout(0x1000, &props, &ctx).unwrap();
    assert_eq!(bytes[0], 3); // version 3
    assert_eq!(bytes[1], 1); // contiguous
    assert_eq!(LittleEndian::read_u64(&bytes[2..10]), 0x1000);
}

#[cfg(feature = "alloc")]
#[test]
fn encode_chunked_layout_with_materialized_index() {
    let ctx = ParseContext::new(8, 8);
    let props = DatasetCreationProps {
        layout: DatasetLayout::Chunked,
        chunk_dims: Some(vec![5, 7]),
        ..DatasetCreationProps::default()
    };
    let bytes = encode_layout_with_chunk_index(0, &props, &ctx, Some(0x4000), Some(4)).unwrap();
    assert_eq!(bytes[0], 3);
    assert_eq!(bytes[1], 2);
    assert_eq!(bytes[2], 3);
    assert_eq!(LittleEndian::read_u64(&bytes[3..11]), 0x4000);
    assert_eq!(LittleEndian::read_u32(&bytes[11..15]), 5);
    assert_eq!(LittleEndian::read_u32(&bytes[15..19]), 7);
    assert_eq!(LittleEndian::read_u32(&bytes[19..23]), 4);
}

#[cfg(feature = "alloc")]
#[test]
fn encode_reference_datatype_object_reference() {
    use consus_core::{Datatype, ReferenceType};
    let bytes = encode_datatype(&Datatype::Reference(ReferenceType::Object))
        .expect("encode_datatype(Reference(Object)) must succeed");
    assert_eq!(bytes.len(), 8, "reference datatype message must be 8 bytes");
    assert_eq!(
        bytes[0], 0x17,
        "byte[0]: class=7 (Reference), version=1 => 0x17"
    );
    assert_eq!(
        bytes[1], 0x00,
        "byte[1]: object reference discriminant must be 0"
    );
    assert_eq!(bytes[2], 0x00, "byte[2]: reserved must be 0");
    assert_eq!(bytes[3], 0x00, "byte[3]: reserved must be 0");
    let size = LittleEndian::read_u32(&bytes[4..8]);
    assert_eq!(size, 8, "reference element size must be 8 bytes");
}

#[cfg(feature = "alloc")]
#[test]
fn encode_reference_datatype_region_reference() {
    use consus_core::{Datatype, ReferenceType};
    let bytes = encode_datatype(&Datatype::Reference(ReferenceType::Region))
        .expect("encode_datatype(Reference(Region)) must succeed");
    assert_eq!(bytes.len(), 8, "reference datatype message must be 8 bytes");
    assert_eq!(
        bytes[0], 0x17,
        "byte[0]: class=7 (Reference), version=1 => 0x17"
    );
    assert_eq!(
        bytes[1], 0x01,
        "byte[1]: region reference discriminant must be 1"
    );
    let size = LittleEndian::read_u32(&bytes[4..8]);
    assert_eq!(size, 8, "reference element size must be 8 bytes");
}
