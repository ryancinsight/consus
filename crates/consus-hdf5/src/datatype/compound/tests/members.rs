//! Compound member parsing tests across layout versions.

use super::*;

#[test]
fn parse_compound_v3_two_members() {
    // Compound with 2 members, total size = 12:
    //   member "x": u32 LE at offset 0
    //   member "y": i64 LE at offset 4

    let num_members: u16 = 2;
    let compound_size: u32 = 12;
    let hdr = dt_header(
        COMPOUND,
        3,
        [
            num_members.to_le_bytes()[0],
            num_members.to_le_bytes()[1],
            0,
        ],
        compound_size,
    );
    let mut msg = hdr.to_vec();

    // Member "x": name, offset (1 byte for size<256), datatype
    msg.extend_from_slice(b"x\0");
    msg.push(0x00); // offset = 0 (1 byte since compound_size=12 < 256)
    msg.extend_from_slice(&int_msg(4, false, true));

    // Member "y": name, offset, datatype
    msg.extend_from_slice(b"y\0");
    msg.push(0x04); // offset = 4
    msg.extend_from_slice(&int_msg(8, true, true));

    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Compound { ref fields, size } => {
            assert_eq!(size, 12);
            assert_eq!(fields.len(), 2);

            assert_eq!(fields[0].name, "x");
            assert_eq!(fields[0].offset, 0);
            assert!(matches!(
                fields[0].datatype,
                Datatype::Integer {
                    bits,
                    signed: false,
                    ..
                } if bits.get() == 32
            ));

            assert_eq!(fields[1].name, "y");
            assert_eq!(fields[1].offset, 4);
            assert!(matches!(
                fields[1].datatype,
                Datatype::Integer {
                    bits,
                    signed: true,
                    ..
                } if bits.get() == 64
            ));
        }
        other => panic!("expected Compound, got: {other:?}"),
    }
}

#[test]
fn parse_compound_v1_with_dimensionality() {
    // Version 1 compound with one scalar member (rank=0 dimensionality).
    let num_members: u16 = 1;
    let compound_size: u32 = 4;
    let hdr = dt_header(
        COMPOUND,
        1,
        [
            num_members.to_le_bytes()[0],
            num_members.to_le_bytes()[1],
            0,
        ],
        compound_size,
    );
    let mut msg = hdr.to_vec();

    // Member name "val" + null = 4 bytes, padded to 8.
    msg.extend_from_slice(b"val\0");
    msg.extend_from_slice(&[0u8; 4]); // padding to 8 bytes

    // Byte offset (4 bytes, u32 LE): 0
    msg.extend_from_slice(&0u32.to_le_bytes());

    // Dimensionality: rank(1) + reserved(3) + perm(4) + reserved(4) + 4×dim_sizes(16) = 28 bytes
    // HDF5 spec: always 4 dimension size slots regardless of rank.
    msg.push(0); // rank = 0
    msg.extend_from_slice(&[0u8; 3]); // reserved
    msg.extend_from_slice(&[0u8; 4]); // dimension permutation
    msg.extend_from_slice(&[0u8; 4]); // reserved
    msg.extend_from_slice(&[0u8; 16]); // 4 × u32 dim_sizes (always present)

    // Member datatype: u32 LE
    msg.extend_from_slice(&int_msg(4, false, true));

    let dt = parse_datatype(&msg, &ParseBudget::DEFAULT).unwrap();
    match dt {
        Datatype::Compound { ref fields, size } => {
            assert_eq!(size, 4);
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].name, "val");
            assert_eq!(fields[0].offset, 0);
            assert!(matches!(
                fields[0].datatype,
                Datatype::Integer {
                    bits,
                    signed: false,
                    ..
                } if bits.get() == 32
            ));
        }
        other => panic!("expected Compound, got: {other:?}"),
    }
}

// -- Class 7: Reference --------------------------------------------------
