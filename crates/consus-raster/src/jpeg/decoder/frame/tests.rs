use super::super::super::bitstream::parse_tables;
use super::*;
use crate::DecodeErrorKind;

fn frame() -> Frame {
    parse_frame(
        &[8, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0],
        0xc0,
        DecodeLimits {
            max_encoded_bytes: 1024,
            max_dimension: 8,
            max_pixels: 64,
            max_working_bytes: 8192,
        },
    )
    .expect("valid frame")
}

#[test]
fn quantization_is_bound_when_component_scan_begins() {
    let mut frame = frame();
    let mut tables = [None; 4];
    tables[0] = Some(QuantizationTable {
        values: [1; 64],
        wide: false,
    });
    capture_quantization(
        &mut frame,
        &Scan {
            components: vec![ScanComponent {
                frame_index: 0,
                dc_table: 0,
                ac_table: 0,
            }],
            start: 0,
            end: 63,
            high: 0,
            low: 0,
        },
        &tables,
    )
    .expect("first component table");
    tables[0] = Some(QuantizationTable {
        values: [2; 64],
        wide: false,
    });
    capture_quantization(
        &mut frame,
        &Scan {
            components: vec![ScanComponent {
                frame_index: 1,
                dc_table: 0,
                ac_table: 0,
            }],
            start: 0,
            end: 63,
            high: 0,
            low: 0,
        },
        &tables,
    )
    .expect("second component table");
    assert_eq!(frame.components[0].quantization_values, Some([1; 64]));
    assert_eq!(frame.components[1].quantization_values, Some([2; 64]));
}

#[test]
fn sequential_component_cannot_be_scanned_twice() {
    let mut frame = frame();
    frame.components[0].approximation[0] = 0;
    let mut tables = Tables::new();
    let mut definition = vec![0, 1];
    definition.extend_from_slice(&[0; 15]);
    definition.push(0);
    definition.extend_from_slice(&[0x10, 1]);
    definition.extend_from_slice(&[0; 15]);
    definition.push(0);
    parse_tables(&definition, &mut tables).expect("minimal Huffman tables");
    assert_eq!(
        parse_scan(&[1, 1, 0, 0, 63, 0], &frame, &tables)
            .map(|_| ())
            .expect_err("duplicate sequential scan")
            .kind(),
        DecodeErrorKind::Malformed
    );
}

#[test]
fn baseline_rejects_extended_huffman_destinations() {
    let mut frame = frame();
    for destination in 2..=3 {
        let mut tables = Tables::new();
        let mut definition = vec![destination, 1];
        definition.extend_from_slice(&[0; 15]);
        definition.push(0);
        definition.extend_from_slice(&[0x10 | destination, 1]);
        definition.extend_from_slice(&[0; 15]);
        definition.push(0);
        parse_tables(&definition, &mut tables).expect("extended Huffman table destinations");
        let payload = [1, 1, destination * 0x11, 0, 63, 0];
        frame.coding = Coding::Baseline;
        assert_eq!(
            parse_scan(&payload, &frame, &tables)
                .err()
                .expect("baseline destination restriction")
                .kind(),
            DecodeErrorKind::Malformed
        );
        frame.coding = Coding::Sequential;
        let scan = parse_scan(&payload, &frame, &tables).expect("extended sequential destinations");
        assert_eq!(scan.components[0].dc_table, usize::from(destination));
        assert_eq!(scan.components[0].ac_table, usize::from(destination));
    }
}

#[test]
fn progressive_quantization_cannot_change_between_scans() {
    let mut frame = frame();
    frame.coding = Coding::Progressive;
    let scan = Scan {
        components: vec![ScanComponent {
            frame_index: 0,
            dc_table: 0,
            ac_table: 0,
        }],
        start: 0,
        end: 0,
        high: 0,
        low: 0,
    };
    let mut tables = [None; 4];
    tables[0] = Some(QuantizationTable {
        values: [1; 64],
        wide: false,
    });
    capture_quantization(&mut frame, &scan, &tables).expect("initial progressive table");
    capture_quantization(&mut frame, &scan, &tables).expect("identical table remains valid");
    assert_eq!(frame.components[0].quantization_values, Some([1; 64]));
    tables[0] = Some(QuantizationTable {
        values: [2; 64],
        wide: false,
    });
    assert_eq!(
        capture_quantization(&mut frame, &scan, &tables)
            .unwrap_err()
            .kind(),
        DecodeErrorKind::Malformed
    );
}

#[test]
fn interleaved_selectors_preserve_frame_component_order() {
    let mut frame = frame();
    frame.entropy_coding = EntropyCoding::Arithmetic;
    frame.coding = Coding::Sequential;
    let tables = Tables::new();
    let ordered = parse_scan(&[2, 1, 0, 3, 0, 0, 63, 0], &frame, &tables)
        .expect("ordered subset of frame components");
    assert_eq!(
        ordered
            .components
            .iter()
            .map(|component| component.frame_index)
            .collect::<Vec<_>>(),
        [0, 2]
    );
    for payload in [[2, 3, 0, 1, 0, 0, 63, 0], [2, 1, 0, 1, 0, 0, 63, 0]] {
        assert_eq!(
            parse_scan(&payload, &frame, &tables)
                .err()
                .expect("unordered or repeated selector")
                .kind(),
            DecodeErrorKind::Malformed
        );
    }
}

#[test]
fn sampling_limit_applies_to_interleaved_scan_components() {
    for factors in [
        &[0x44][..],
        &[0x44, 0x44, 0x44][..],
        &[0x44, 0x44, 0x44, 0x44][..],
    ] {
        let mut payload = vec![8, 0, 8, 0, 8, u8::try_from(factors.len()).unwrap()];
        for (index, factor) in factors.iter().enumerate() {
            payload.extend_from_slice(&[u8::try_from(index + 1).unwrap(), *factor, 0]);
        }
        let frame = parse_frame(
            &payload,
            0xc9,
            DecodeLimits {
                max_encoded_bytes: 1024,
                max_dimension: 8,
                max_pixels: 64,
                max_working_bytes: crate::jpeg::working_storage_bound(8, 8).unwrap(),
            },
        )
        .expect("noninterleaved sampling factors");
        for component in &frame.components {
            let scan = parse_scan(&[1, component.id, 0, 0, 63, 0], &frame, &Tables::new())
                .expect("single-component scan has one block per MCU");
            assert_eq!(scan.components.len(), 1);
            assert_eq!((component.blocks_across, component.blocks_down), (1, 1));
            assert_eq!((component.stored_across, component.stored_down), (4, 4));
        }
        assert_eq!(frame.coefficients.len(), factors.len() * 16 * 64);
    }

    let mut frame = frame();
    frame.entropy_coding = EntropyCoding::Arithmetic;
    frame.components[0].horizontal = 3;
    frame.components[0].vertical = 3;
    let scan = parse_scan(&[2, 1, 0, 2, 0, 0, 63, 0], &frame, &Tables::new())
        .expect("ten-block interleaved subset");
    assert_eq!(
        scan.components
            .iter()
            .map(|component| component.frame_index)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(
        parse_scan(&[3, 1, 0, 2, 0, 3, 0, 0, 63, 0], &frame, &Tables::new())
            .err()
            .expect("eleven-block interleaved scan")
            .kind(),
        DecodeErrorKind::Malformed
    );
}

#[test]
fn noninterleaved_sampling_factors_preserve_pixels() {
    use crate::jpeg::{decode, encode_gray};
    let limits = DecodeLimits {
        max_encoded_bytes: 1 << 20,
        max_dimension: 8,
        max_pixels: 64,
        max_working_bytes: crate::jpeg::working_storage_bound(8, 8).unwrap(),
    };
    let encoded = encode_gray(&[73; 64], 8, 8, 90).expect("grayscale fixture");
    let expected = decode(&encoded, limits).expect("reference sampling");
    let frame = encoded
        .windows(2)
        .position(|bytes| bytes == [0xff, 0xc0])
        .expect("baseline frame marker");
    for horizontal in 1..=4 {
        for vertical in 1..=4 {
            let mut sampled = encoded.clone();
            sampled[frame + 11] = horizontal << 4 | vertical;
            let decoded = decode(&sampled, limits).expect("noninterleaved sampling");
            assert_eq!(decoded, expected);
            assert_eq!(decoded.pixels(), &[73; 64]);
        }
    }
}
