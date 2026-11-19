//! MAT v5 `miCOMPRESSED` integration coverage.
//!
//! This test file is compiled in both feature configurations:
//!
//! - with `compress`: compressed payloads must decode successfully
//! - without `compress`: compressed payloads must fail with
//!   `UnsupportedFeature("miCOMPRESSED requires the 'compress' feature")`

use consus_mat::loadmat_bytes;
#[cfg(feature = "compress")]
use consus_mat::{MatArray, MatNumericClass, MatVersion};

fn u32le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn element(mi_type: u32, data: &[u8]) -> Vec<u8> {
    let mut e = Vec::with_capacity(8 + data.len() + 7);
    u32le(&mut e, mi_type);
    u32le(&mut e, data.len() as u32);
    e.extend_from_slice(data);
    let pad = (8 - (data.len() % 8)) % 8;
    e.resize(e.len() + pad, 0u8);
    e
}

#[cfg(feature = "compress")]
fn matrix_payload(
    mx_class: u8,
    flags: u32,
    nzmax: u32,
    shape: &[i32],
    name: &[u8],
    extra_els: &[Vec<u8>],
) -> Vec<u8> {
    let mut p = Vec::new();

    let mut fdata = [0u8; 8];
    fdata[0..4].copy_from_slice(&(mx_class as u32 | flags).to_le_bytes());
    fdata[4..8].copy_from_slice(&nzmax.to_le_bytes());
    p.extend(element(6, &fdata)); // miUINT32 array flags

    let mut ddata = Vec::with_capacity(shape.len() * 4);
    for &d in shape {
        ddata.extend_from_slice(&d.to_le_bytes());
    }
    p.extend(element(5, &ddata)); // miINT32 dimensions
    p.extend(element(1, name)); // miINT8 name

    for ex in extra_els {
        p.extend_from_slice(ex);
    }

    p
}

#[cfg(feature = "compress")]
fn matrix_element(payload: Vec<u8>) -> Vec<u8> {
    element(14, &payload)
}

fn v5_file(elements: &[Vec<u8>]) -> Vec<u8> {
    let mut f = vec![0u8; 128];
    f[124] = 0x00;
    f[125] = 0x01;
    f[126] = b'I';
    f[127] = b'M';
    for e in elements {
        f.extend_from_slice(e);
    }
    f
}

#[cfg(feature = "compress")]
fn compressed_double_variable_file() -> Vec<u8> {
    use std::io::Write as _;

    let real_data: Vec<u8> = [1.0f64, 2.0f64, 3.0f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let real_el = element(9, &real_data); // miDOUBLE
    let payload = matrix_payload(6, 0, 0, &[1i32, 3i32], b"x", &[real_el]);
    let matrix = matrix_element(payload);

    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder
        .write_all(&matrix)
        .expect("zlib encoder write_all failed");
    let compressed = encoder.finish().expect("zlib encoder finish failed");
    v5_file(&[element(15, &compressed)]) // miCOMPRESSED
}

#[cfg(not(feature = "compress"))]
fn compressed_double_variable_file() -> Vec<u8> {
    // Precomputed zlib stream for the exact `miMATRIX` bytes for a
    // 1x3 double variable named `x` with values [1.0, 2.0, 3.0].
    const PRECOMPRESSED: &[u8] = &[
        0x78, 0x9c, 0xe3, 0x61, 0x60, 0x60, 0x70, 0x09, 0x62, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60,
        0x60, 0x60, 0x63, 0x60, 0x60, 0x70, 0x64, 0x60, 0x60, 0x48, 0xac, 0x60, 0x60, 0x60, 0x64,
        0x60, 0x60, 0x70, 0x04, 0x62, 0x26, 0x20, 0x66, 0x03, 0x62, 0x16, 0x20, 0x66, 0x07, 0x62,
        0x76, 0x20, 0x66, 0x00, 0x62, 0x4e, 0x20, 0x66, 0x04, 0x62, 0x2e, 0x20, 0x66, 0x02, 0x62,
        0x6e, 0x20, 0x66, 0x06, 0x62, 0x1e, 0x20, 0x66, 0x01, 0x62, 0x5e, 0x20, 0x66, 0x05, 0x62,
        0x3e, 0x20, 0x66, 0x03, 0x62, 0x7e, 0x20, 0x66, 0x07, 0x62, 0x00, 0x00, 0x8d, 0x8d, 0x0b,
        0x5d,
    ];
    v5_file(&[element(15, PRECOMPRESSED)]) // miCOMPRESSED
}

#[cfg(feature = "compress")]
fn decode_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect()
}

#[cfg(feature = "compress")]
#[test]
fn v5_micompressed_roundtrip_with_compress_feature() {
    let file = compressed_double_variable_file();
    let mat = loadmat_bytes(&file).expect("compressed v5 parse failed");

    assert_eq!(mat.version, MatVersion::V5);
    assert_eq!(mat.variables.len(), 1);

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "x");

    match arr {
        MatArray::Numeric(na) => {
            assert_eq!(na.class, MatNumericClass::Double);
            assert_eq!(na.shape, vec![1, 3]);
            assert_eq!(na.numel(), 3);
            assert_eq!(decode_f64s(&na.real_data), vec![1.0, 2.0, 3.0]);
            assert!(na.imag_data.is_none());
        }
        other => panic!("expected Numeric, got {:?}", other),
    }
}

#[cfg(not(feature = "compress"))]
#[test]
fn v5_micompressed_requires_compress_feature() {
    let file = compressed_double_variable_file();
    let err = loadmat_bytes(&file).expect_err("miCOMPRESSED must fail without compress feature");

    match err {
        consus_mat::MatError::UnsupportedFeature(message) => {
            assert_eq!(message, "miCOMPRESSED requires the 'compress' feature");
        }
        other => panic!("expected UnsupportedFeature, got {:?}", other),
    }
}

#[cfg(feature = "compress")]
fn traced_case<Input, Output>(
    arrange: impl FnOnce() -> Input,
    act: impl FnOnce(Input) -> Output,
    assert: impl FnOnce(Output),
) {
    let subscriber = tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let _test = tracing::info_span!("mat_level_five").entered();
        let input = tracing::info_span!("arrange").in_scope(arrange);
        let output = tracing::info_span!("act").in_scope(|| act(input));
        tracing::info_span!("assert").in_scope(|| assert(output));
    });
}

#[cfg(feature = "compress")]
fn compress_matrix(matrix: &[u8]) -> Vec<u8> {
    use std::io::Write as _;

    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(matrix).expect("compress complete matrix");
    encoder.finish().expect("flush zlib stream")
}

#[cfg(feature = "compress")]
#[test]
fn v5_compressed_double_stored_as_unsigned_integers_preserves_values() {
    traced_case(
        || {
            let real_data: Vec<u8> = [1u16, 256, u16::MAX, 17]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect();
            let matrix = matrix_element(matrix_payload(
                6,
                0,
                0,
                &[2, 1, 2],
                b"volume",
                &[element(4, &real_data)],
            ));
            v5_file(&[element(15, &compress_matrix(&matrix))])
        },
        |file| loadmat_bytes(&file).expect("read packed double matrix"),
        |mat| {
            let (name, MatArray::Numeric(array)) = &mat.variables[0] else {
                panic!("expected a numeric matrix");
            };
            assert_eq!(name, "volume");
            assert_eq!(array.class, MatNumericClass::Double);
            assert_eq!(array.shape, [2, 1, 2]);
            assert_eq!(decode_f64s(&array.real_data), [1.0, 256.0, 65_535.0, 17.0]);
        },
    );
}

#[cfg(feature = "compress")]
#[test]
fn v5_reads_consecutive_unpadded_compressed_variables() {
    traced_case(
        || {
            let variables: Vec<_> = [("first", -23i16), ("second", 147)]
                .into_iter()
                .map(|(name, value)| {
                    let payload = matrix_payload(
                        6,
                        0,
                        0,
                        &[1, 1],
                        name.as_bytes(),
                        &[element(3, &value.to_le_bytes())],
                    );
                    let compressed = compress_matrix(&matrix_element(payload));
                    let mut outer = element(15, &compressed);
                    outer.truncate(8 + compressed.len());
                    outer
                })
                .collect();
            assert!(
                !variables[0].len().is_multiple_of(8),
                "test must exercise an unaligned tag"
            );
            v5_file(&variables)
        },
        |file| {
            let mat = loadmat_bytes(&file).expect("read both matrices");
            let second = consus_mat::v5::read_mat_v5_variable(&file, "second")
                .expect("select second matrix");
            (mat, second)
        },
        |(mat, second)| {
            assert_eq!(
                mat.variables
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>(),
                ["first", "second"]
            );
            let MatArray::Numeric(first) = &mat.variables[0].1 else {
                panic!("expected numeric first matrix");
            };
            assert_eq!(decode_f64s(&first.real_data), [-23.0]);
            let Some(MatArray::Numeric(second)) = second else {
                panic!("expected numeric second matrix");
            };
            assert_eq!(second.shape, [1, 1]);
            assert_eq!(decode_f64s(&second.real_data), [147.0]);
        },
    );
}

#[cfg(feature = "compress")]
#[test]
fn v5_named_read_skips_unrelated_unsupported_matrix_values() {
    use consus_mat::v5::read_mat_v5_variable;

    traced_case(
        || {
            // An object is unsupported, but its validated name differs from the query.
            let preceding = matrix_element(matrix_payload(3, 0, 0, &[1, 1], b"unrelated", &[]));
            let target = matrix_element(matrix_payload(
                6,
                0,
                0,
                &[1, 1],
                b"target",
                &[element(9, &(-14.5f64).to_le_bytes())],
            ));
            v5_file(&[preceding, target])
        },
        |file| {
            let all = loadmat_bytes(&file);
            let target = read_mat_v5_variable(&file, "target").expect("select target");
            let absent = read_mat_v5_variable(&file, "absent").expect("search all names");
            (all, target, absent)
        },
        |(all, target, absent)| {
            assert!(matches!(
                all,
                Err(consus_mat::MatError::UnsupportedFeature(_))
            ));
            let Some(MatArray::Numeric(value)) = target else {
                panic!("expected selected numeric array");
            };
            assert_eq!(value.shape, [1, 1]);
            assert_eq!(decode_f64s(&value.real_data), [-14.5]);
            assert!(absent.is_none());
        },
    );
}
