use super::{Orientation, parse};
use crate::DecodeErrorKind;

#[derive(Clone, Copy)]
enum FixtureOrder {
    Little,
    Big,
}

impl FixtureOrder {
    fn u16(self, value: u16) -> [u8; 2] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }

    fn u32(self, value: u32) -> [u8; 4] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }
}

fn tiff(orientation: u16, order: FixtureOrder) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(match order {
        FixtureOrder::Little => b"II",
        FixtureOrder::Big => b"MM",
    });
    bytes.extend_from_slice(&order.u16(42));
    bytes.extend_from_slice(&order.u32(8));
    bytes.extend_from_slice(&order.u16(1));
    entry(&mut bytes, order, 0x0112, 3, short(order, orientation));
    bytes.extend_from_slice(&order.u32(0));
    bytes
}

fn entry(bytes: &mut Vec<u8>, order: FixtureOrder, tag: u16, field_type: u16, value: [u8; 4]) {
    entry_with_count(bytes, order, tag, field_type, 1, value);
}

fn entry_with_count(
    bytes: &mut Vec<u8>,
    order: FixtureOrder,
    tag: u16,
    field_type: u16,
    count: u32,
    value: [u8; 4],
) {
    bytes.extend_from_slice(&order.u16(tag));
    bytes.extend_from_slice(&order.u16(field_type));
    bytes.extend_from_slice(&order.u32(count));
    bytes.extend_from_slice(&value);
}

fn short(order: FixtureOrder, value: u16) -> [u8; 4] {
    let [first, second] = order.u16(value);
    [first, second, 0, 0]
}

fn tiff_with_primary_and_thumbnail_density() -> Vec<u8> {
    let order = FixtureOrder::Little;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"II");
    bytes.extend_from_slice(&order.u16(42));
    bytes.extend_from_slice(&order.u32(8));
    bytes.extend_from_slice(&order.u16(4));
    entry(&mut bytes, order, 0x0112, 3, short(order, 6));
    entry(&mut bytes, order, 0x011a, 5, order.u32(62));
    entry(&mut bytes, order, 0x011b, 5, order.u32(70));
    entry(&mut bytes, order, 0x0128, 3, short(order, 2));
    bytes.extend_from_slice(&order.u32(78));
    bytes.extend_from_slice(&order.u32(300));
    bytes.extend_from_slice(&order.u32(1));
    bytes.extend_from_slice(&order.u32(300));
    bytes.extend_from_slice(&order.u32(1));
    bytes.extend_from_slice(&order.u16(4));
    entry(&mut bytes, order, 0x0112, 3, short(order, 1));
    entry(&mut bytes, order, 0x011a, 5, order.u32(132));
    entry(&mut bytes, order, 0x011b, 5, order.u32(140));
    entry(&mut bytes, order, 0x0128, 3, short(order, 2));
    bytes.extend_from_slice(&order.u32(0));
    bytes.extend_from_slice(&order.u32(72));
    bytes.extend_from_slice(&order.u32(1));
    bytes.extend_from_slice(&order.u32(72));
    bytes.extend_from_slice(&order.u32(1));
    bytes
}

fn tiff_with_thumbnail() -> Vec<u8> {
    let order = FixtureOrder::Little;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"II");
    bytes.extend_from_slice(&order.u16(42));
    bytes.extend_from_slice(&order.u32(8));
    bytes.extend_from_slice(&order.u16(1));
    entry(&mut bytes, order, 0x0112, 3, short(order, 4));
    bytes.extend_from_slice(&order.u32(26));
    bytes.extend_from_slice(&order.u16(2));
    entry(&mut bytes, order, 0x0201, 4, order.u32(56));
    entry(&mut bytes, order, 0x0202, 4, order.u32(4));
    bytes.extend_from_slice(&order.u32(0));
    bytes.extend_from_slice(&[0xff, 0xd8, 0xff, 0xd9]);
    bytes
}

fn big_endian_tiff_with_pointer_graph_and_density() -> Vec<u8> {
    let order = FixtureOrder::Big;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"MM");
    bytes.extend_from_slice(&order.u16(42));
    bytes.extend_from_slice(&order.u32(8));
    bytes.extend_from_slice(&order.u16(4));
    entry(&mut bytes, order, 0x0112, 3, short(order, 7));
    entry(&mut bytes, order, 0x011a, 5, order.u32(62));
    entry(&mut bytes, order, 0x011b, 5, order.u32(70));
    entry(&mut bytes, order, 0x8769, 4, order.u32(78));
    bytes.extend_from_slice(&order.u32(0));
    bytes.extend_from_slice(&order.u32(300));
    bytes.extend_from_slice(&order.u32(1));
    bytes.extend_from_slice(&order.u32(300));
    bytes.extend_from_slice(&order.u32(1));
    bytes.extend_from_slice(&order.u16(2));
    entry(&mut bytes, order, 0xa001, 3, short(order, 1));
    entry(&mut bytes, order, 0xa005, 4, order.u32(108));
    bytes.extend_from_slice(&order.u32(0));
    bytes.extend_from_slice(&order.u16(1));
    entry_with_count(&mut bytes, order, 0x0001, 2, 4, *b"R98\0");
    bytes.extend_from_slice(&order.u32(0));
    bytes
}

#[test]
fn both_byte_orders_map_all_orientation_values() {
    let expected = [
        Orientation::Normal,
        Orientation::MirrorHorizontal,
        Orientation::RotateHalf,
        Orientation::MirrorVertical,
        Orientation::Transpose,
        Orientation::RotateClockwise,
        Orientation::Transverse,
        Orientation::RotateCounterClockwise,
    ];
    for order in [FixtureOrder::Little, FixtureOrder::Big] {
        for (index, orientation) in expected.into_iter().enumerate() {
            let value = u16::try_from(index + 1).expect("eight orientations fit u16");
            assert_eq!(parse(&tiff(value, order)), Ok(orientation));
        }
    }
}

#[test]
fn source_coordinates_match_all_eight_grid_orders() {
    const EXPECTED: [[(u32, u32); 6]; 8] = [
        [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
        [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2)],
        [(1, 2), (0, 2), (1, 1), (0, 1), (1, 0), (0, 0)],
        [(0, 2), (1, 2), (0, 1), (1, 1), (0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 2), (0, 1), (0, 0), (1, 2), (1, 1), (1, 0)],
        [(1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)],
        [(1, 0), (1, 1), (1, 2), (0, 0), (0, 1), (0, 2)],
    ];
    let orientations = [
        Orientation::Normal,
        Orientation::MirrorHorizontal,
        Orientation::RotateHalf,
        Orientation::MirrorVertical,
        Orientation::Transpose,
        Orientation::RotateClockwise,
        Orientation::Transverse,
        Orientation::RotateCounterClockwise,
    ];
    let swaps_axes = [false, false, false, false, true, true, true, true];
    for ((orientation, expected), expected_swaps_axes) in
        orientations.into_iter().zip(EXPECTED).zip(swaps_axes)
    {
        assert_eq!(orientation.swaps_axes(), expected_swaps_axes);
        let (output_width, output_height) = if orientation.swaps_axes() {
            (3, 2)
        } else {
            (2, 3)
        };
        let actual: Vec<_> = (0..output_height)
            .flat_map(|y| (0..output_width).map(move |x| (x, y)))
            .map(|(x, y)| orientation.source_coordinate(2, 3, x, y))
            .collect::<Result<_, _>>()
            .expect("fixture coordinates are in range");
        assert_eq!(actual.as_slice(), expected);
    }
}

#[test]
fn coordinate_mapping_rejects_zero_dimensions_and_output_overflow() {
    for orientation in [Orientation::Normal, Orientation::RotateClockwise] {
        assert_eq!(
            orientation
                .source_coordinate(0, 3, 0, 0)
                .map_err(|error| error.kind()),
            Err(DecodeErrorKind::Malformed)
        );
        assert_eq!(
            orientation
                .source_coordinate(2, 0, 0, 0)
                .map_err(|error| error.kind()),
            Err(DecodeErrorKind::Malformed)
        );
    }
    assert_eq!(
        Orientation::Normal
            .source_coordinate(2, 3, 2, 0)
            .map_err(|error| error.kind()),
        Err(DecodeErrorKind::Malformed)
    );
    assert_eq!(
        Orientation::RotateClockwise
            .source_coordinate(2, 3, 3, 0)
            .map_err(|error| error.kind()),
        Err(DecodeErrorKind::Malformed)
    );
}

#[test]
fn primary_and_thumbnail_density_have_independent_square_resolutions() {
    assert_eq!(
        parse(&tiff_with_primary_and_thumbnail_density()),
        Ok(Orientation::RotateClockwise)
    );
}

#[test]
fn thumbnail_extent_and_markers_are_validated_without_nested_decoding() {
    let valid = tiff_with_thumbnail();
    assert_eq!(parse(&valid), Ok(Orientation::MirrorVertical));

    let mut dangling = valid.clone();
    dangling[40..42].copy_from_slice(&0x0100_u16.to_le_bytes());
    assert_eq!(
        parse(&dangling).map_err(|error| error.kind()),
        Err(DecodeErrorKind::Malformed)
    );

    let mut truncated = valid;
    truncated[48..52].copy_from_slice(&5_u32.to_le_bytes());
    assert_eq!(
        parse(&truncated).map_err(|error| error.kind()),
        Err(DecodeErrorKind::Malformed)
    );
}

#[test]
fn big_endian_rationals_and_pointer_graph_preserve_orientation() {
    assert_eq!(
        parse(&big_endian_tiff_with_pointer_graph_and_density()),
        Ok(Orientation::Transverse)
    );
}

#[test]
fn big_endian_rational_and_pointer_extents_are_bounded() {
    let valid = big_endian_tiff_with_pointer_graph_and_density();

    let mut zero_denominator = valid.clone();
    zero_denominator[66..70].copy_from_slice(&0_u32.to_be_bytes());
    assert_eq!(
        parse(&zero_denominator).map_err(|error| error.kind()),
        Err(DecodeErrorKind::Malformed)
    );

    let mut dangling_pointer = valid;
    dangling_pointer[54..58].copy_from_slice(&u32::MAX.to_be_bytes());
    assert_eq!(
        parse(&dangling_pointer).map_err(|error| error.kind()),
        Err(DecodeErrorKind::Malformed)
    );
}

#[test]
fn malformed_graph_fields_and_presentation_metadata_are_classified() {
    for malformed in [
        Vec::new(),
        b"II\x2a\0\xff\xff\xff\x7f".to_vec(),
        {
            let mut value = tiff(1, FixtureOrder::Little);
            value[12..14].copy_from_slice(&0_u16.to_le_bytes());
            value
        },
        {
            let mut value = tiff(1, FixtureOrder::Little);
            value[14..18].copy_from_slice(&u32::MAX.to_le_bytes());
            value
        },
        tiff(9, FixtureOrder::Little),
    ] {
        assert_eq!(
            parse(&malformed).map_err(|error| error.kind()),
            Err(DecodeErrorKind::Malformed)
        );
    }

    let mut cycle = Vec::new();
    cycle.extend_from_slice(b"II");
    cycle.extend_from_slice(&42_u16.to_le_bytes());
    cycle.extend_from_slice(&8_u32.to_le_bytes());
    cycle.extend_from_slice(&1_u16.to_le_bytes());
    entry(
        &mut cycle,
        FixtureOrder::Little,
        0x8769,
        4,
        8_u32.to_le_bytes(),
    );
    cycle.extend_from_slice(&0_u32.to_le_bytes());
    assert_eq!(
        parse(&cycle).map_err(|error| error.kind()),
        Err(DecodeErrorKind::Malformed)
    );

    let mut profile = tiff(1, FixtureOrder::Little);
    profile[10..12].copy_from_slice(&0x8773_u16.to_le_bytes());
    assert_eq!(
        parse(&profile).map_err(|error| error.kind()),
        Err(DecodeErrorKind::Unsupported)
    );

    let unequal_resolution = [
        0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x03, 0x00, 0x1a, 0x01, 0x05, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00, 0x1b, 0x01, 0x05, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x3a, 0x00, 0x00, 0x00, 0x28, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x90, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    ];
    assert_eq!(
        parse(&unequal_resolution).map_err(|error| error.kind()),
        Err(DecodeErrorKind::Unsupported)
    );

    let mut missing_vertical_density = unequal_resolution;
    missing_vertical_density[10..12].copy_from_slice(&0x0002_u16.to_le_bytes());
    assert_eq!(
        parse(&missing_vertical_density).map_err(|error| error.kind()),
        Err(DecodeErrorKind::Malformed)
    );
}

#[test]
fn directory_graph_is_limited_to_sixteen_nodes() {
    let mut graph = Vec::new();
    graph.extend_from_slice(b"II");
    graph.extend_from_slice(&42_u16.to_le_bytes());
    graph.extend_from_slice(&8_u32.to_le_bytes());
    for index in 0..17_u32 {
        graph.extend_from_slice(&1_u16.to_le_bytes());
        entry(
            &mut graph,
            FixtureOrder::Little,
            0x0100,
            4,
            1_u32.to_le_bytes(),
        );
        let next = if index == 16 { 0 } else { 8 + 18 * (index + 1) };
        graph.extend_from_slice(&next.to_le_bytes());
    }
    assert_eq!(
        parse(&graph).map_err(|error| error.kind()),
        Err(DecodeErrorKind::TooLarge)
    );
}
