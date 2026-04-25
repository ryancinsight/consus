//! MAT v5 read tests -- value-semantic coverage for all supported array classes.
use consus_mat::{MatArray, MatNumericClass, MatVersion, loadmat_bytes};

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
    p.extend(element(6, &fdata));
    let mut ddata = Vec::with_capacity(shape.len() * 4);
    for &d in shape {
        ddata.extend_from_slice(&d.to_le_bytes());
    }
    p.extend(element(5, &ddata));
    p.extend(element(1, name));
    for ex in extra_els {
        p.extend_from_slice(ex);
    }
    p
}

fn matrix_element(payload: Vec<u8>) -> Vec<u8> {
    element(14, &payload)
}

fn scalar_double_element(v: f64) -> Vec<u8> {
    let re_el = element(9, &v.to_le_bytes());
    let p = matrix_payload(6, 0, 0, &[1i32, 1i32], b"", &[re_el]);
    matrix_element(p)
}

fn v5_file(elements: &[Vec<u8>]) -> Vec<u8> {
    let mut f = vec![0u8; 128];
    f[124] = 0x00;
    f[125] = 0x01;
    f[126] = 73u8;
    f[127] = 77u8;
    for e in elements {
        f.extend_from_slice(e);
    }
    f
}

#[test]
fn v5_double_array_roundtrip() {
    let data = include_bytes!("test_v5.mat");
    let mat = loadmat_bytes(data).expect("v5 parse failed");
    assert_eq!(mat.version, MatVersion::V5);
    assert_eq!(mat.variables.len(), 1);
    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "x");
    if let MatArray::Numeric(na) = arr {
        assert_eq!(na.class, MatNumericClass::Double);
        assert_eq!(na.shape, vec![1, 3]);
        assert_eq!(na.numel(), 3);
        let vals: Vec<f64> = na
            .real_data
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    } else {
        panic!("expected Numeric, got {:?}", arr);
    }
}

#[test]
fn v5_truncated_element_returns_error() {
    let mut data = vec![0u8; 128];
    data[124] = 0x00;
    data[125] = 0x01;
    data[126] = 73u8;
    data[127] = 77u8;
    data.extend_from_slice(&14u32.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&[0u8; 4]);
    assert!(
        loadmat_bytes(&data).is_err(),
        "truncated top-level miMATRIX payload must return Err"
    );
}

#[test]
fn v5_invalid_endian_indicator_returns_error() {
    let mut data = [0u8; 128];
    data[124] = 0x00;
    data[125] = 0x01;
    data[126] = 0x20;
    data[127] = 0x20;
    assert!(loadmat_bytes(&data).is_err());
}

#[test]
fn v5_char_array_roundtrip() {
    let char_data: Vec<u8> = b"hello"
        .iter()
        .flat_map(|&b| (b as u16).to_le_bytes())
        .collect();
    let char_el = element(4, &char_data);
    let payload = matrix_payload(4, 0, 0, &[1i32, 5i32], b"s", &[char_el]);
    let file = v5_file(&[matrix_element(payload)]);
    let mat = loadmat_bytes(&file).expect("v5 char parse failed");
    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "s");
    if let MatArray::Char(ca) = arr {
        assert_eq!(ca.data, "hello");
        assert_eq!(ca.shape, vec![1, 5]);
    } else {
        panic!("expected Char, got {:?}", arr);
    }
}

#[test]
fn v5_logical_array_roundtrip() {
    const FLAG_LOGICAL: u32 = 1 << 9;
    let bool_data = vec![1u8, 0u8, 1u8, 1u8];
    let bool_el = element(2, &bool_data);
    let payload = matrix_payload(9, FLAG_LOGICAL, 0, &[1i32, 4i32], b"b", &[bool_el]);
    let file = v5_file(&[matrix_element(payload)]);
    let mat = loadmat_bytes(&file).expect("v5 logical parse failed");
    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "b");
    if let MatArray::Logical(la) = arr {
        assert_eq!(la.data, vec![true, false, true, true]);
        assert_eq!(la.shape, vec![1, 4]);
    } else {
        panic!("expected Logical, got {:?}", arr);
    }
}

#[test]
fn v5_complex_double_roundtrip() {
    const FLAG_COMPLEX: u32 = 1 << 11;
    let re: Vec<u8> = [1.0f64, 2.0f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let im: Vec<u8> = [3.0f64, 4.0f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let re_el = element(9, &re);
    let im_el = element(9, &im);
    let payload = matrix_payload(6, FLAG_COMPLEX, 0, &[1i32, 2i32], b"z", &[re_el, im_el]);
    let file = v5_file(&[matrix_element(payload)]);
    let mat = loadmat_bytes(&file).expect("v5 complex parse failed");
    let (_, arr) = &mat.variables[0];
    if let MatArray::Numeric(na) = arr {
        assert_eq!(na.class, MatNumericClass::Double);
        assert!(na.imag_data.is_some());
        assert_eq!(na.shape, vec![1, 2]);
        let re_v: Vec<f64> = na
            .real_data
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();
        let im_v: Vec<f64> = na
            .imag_data
            .as_ref()
            .unwrap()
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();
        assert_eq!(re_v, vec![1.0, 2.0]);
        assert_eq!(im_v, vec![3.0, 4.0]);
    } else {
        panic!("expected Numeric, got {:?}", arr);
    }
}

#[test]
fn v5_sparse_array_roundtrip() {
    let ir: Vec<u8> = [0i32, 2i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let jc: Vec<u8> = [0i32, 1i32, 1i32, 2i32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let pr: Vec<u8> = [5.0f64, 7.0f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let ir_el = element(5, &ir);
    let jc_el = element(5, &jc);
    let pr_el = element(9, &pr);
    let payload = matrix_payload(5, 0, 2, &[3i32, 3i32], b"sp", &[ir_el, jc_el, pr_el]);
    let file = v5_file(&[matrix_element(payload)]);
    let mat = loadmat_bytes(&file).expect("v5 sparse parse failed");
    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "sp");
    if let MatArray::Sparse(sa) = arr {
        assert_eq!(sa.nrows, 3);
        assert_eq!(sa.ncols, 3);
        assert_eq!(sa.row_indices, vec![0i32, 2i32]);
        assert_eq!(sa.col_ptrs, vec![0i32, 1i32, 1i32, 2i32]);
        let vals: Vec<f64> = sa
            .real_data
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();
        assert_eq!(vals, vec![5.0, 7.0]);
        assert!(sa.imag_data.is_none());
    } else {
        panic!("expected Sparse, got {:?}", arr);
    }
}

#[test]
fn v5_cell_array_roundtrip() {
    let elem0 = scalar_double_element(7.0);
    let elem1 = scalar_double_element(8.0);
    let payload = matrix_payload(1, 0, 0, &[1i32, 2i32], b"c", &[elem0, elem1]);
    let file = v5_file(&[matrix_element(payload)]);
    let mat = loadmat_bytes(&file).expect("v5 cell parse failed");
    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "c");
    if let MatArray::Cell(ca) = arr {
        assert_eq!(ca.shape, vec![1, 2]);
        assert_eq!(ca.cells.len(), 2);
        let val = |a: &MatArray| -> f64 {
            if let MatArray::Numeric(na) = a {
                f64::from_le_bytes([
                    na.real_data[0],
                    na.real_data[1],
                    na.real_data[2],
                    na.real_data[3],
                    na.real_data[4],
                    na.real_data[5],
                    na.real_data[6],
                    na.real_data[7],
                ])
            } else {
                panic!("expected Numeric cell element")
            }
        };
        assert_eq!(val(&ca.cells[0]), 7.0);
        assert_eq!(val(&ca.cells[1]), 8.0);
    } else {
        panic!("expected Cell, got {:?}", arr);
    }
}

#[test]
fn v5_struct_array_roundtrip() {
    let re_el = element(9, &42.0f64.to_le_bytes());
    let inner_p = matrix_payload(6, 0, 0, &[1i32, 1i32], b"x", &[re_el]);
    let inner_elem = matrix_element(inner_p);
    let fnl_el = element(5, &8i32.to_le_bytes());
    let mut fn_data = b"x".to_vec();
    fn_data.resize(8, 0u8);
    let fn_el = element(1, &fn_data);
    let payload = matrix_payload(2, 0, 0, &[1i32, 1i32], b"s", &[fnl_el, fn_el, inner_elem]);
    let file = v5_file(&[matrix_element(payload)]);
    let mat = loadmat_bytes(&file).expect("v5 struct parse failed");
    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "s");
    if let MatArray::Struct(sa) = arr {
        assert_eq!(sa.shape, vec![1, 1]);
        let names: Vec<&str> = sa.field_names().collect();
        assert_eq!(names, vec!["x"]);
        assert_eq!(sa.data.len(), 1);
        assert_eq!(sa.data[0].0, "x");
        let elems = &sa.data[0].1;
        assert_eq!(elems.len(), 1);
        if let MatArray::Numeric(na) = &elems[0] {
            let v = f64::from_le_bytes([
                na.real_data[0],
                na.real_data[1],
                na.real_data[2],
                na.real_data[3],
                na.real_data[4],
                na.real_data[5],
                na.real_data[6],
                na.real_data[7],
            ]);
            assert_eq!(v, 42.0);
        } else {
            panic!("expected Numeric field element");
        }
    } else {
        panic!("expected Struct, got {:?}", arr);
    }
}

#[test]
fn v5_sparse_nzmax_mismatch_returns_error() {
    let ir: Vec<u8> = [0i32, 2i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let jc: Vec<u8> = [0i32, 1i32, 1i32, 2i32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let pr: Vec<u8> = [5.0f64, 7.0f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let ir_el = element(5, &ir);
    let jc_el = element(5, &jc);
    let pr_el = element(9, &pr);
    let payload = matrix_payload(5, 0, 3, &[3i32, 3i32], b"sp", &[ir_el, jc_el, pr_el]);
    let file = v5_file(&[matrix_element(payload)]);
    assert!(
        loadmat_bytes(&file).is_err(),
        "nzmax mismatch must return Err"
    );
}

#[test]
fn v5_unknown_top_level_element_is_skipped() {
    let unknown = element(7, &[1u8, 2u8, 3u8, 4u8]);
    let re_el = element(9, &42.0f64.to_le_bytes());
    let payload = matrix_payload(6, 0, 0, &[1i32, 1i32], b"x", &[re_el]);
    let file = v5_file(&[unknown, matrix_element(payload)]);

    let mat = loadmat_bytes(&file).expect("unknown top-level element should be skipped");
    assert_eq!(mat.version, MatVersion::V5);
    assert_eq!(mat.variables.len(), 1);
    assert_eq!(mat.variables[0].0, "x");

    match &mat.variables[0].1 {
        MatArray::Numeric(na) => {
            assert_eq!(na.class, MatNumericClass::Double);
            assert_eq!(na.shape, vec![1, 1]);
            let value = f64::from_le_bytes([
                na.real_data[0],
                na.real_data[1],
                na.real_data[2],
                na.real_data[3],
                na.real_data[4],
                na.real_data[5],
                na.real_data[6],
                na.real_data[7],
            ]);
            assert_eq!(value, 42.0);
        }
        other => panic!("expected Numeric, got {:?}", other),
    }
}

#[test]
fn v5_object_class_returns_unsupported_feature_error() {
    let re_el = element(9, &42.0f64.to_le_bytes());
    let payload = matrix_payload(3, 0, 0, &[1i32, 1i32], b"obj", &[re_el]);
    let file = v5_file(&[matrix_element(payload)]);

    let err = loadmat_bytes(&file).expect_err("mxOBJECT_CLASS must return Err");
    match err {
        consus_mat::MatError::UnsupportedFeature(message) => {
            assert_eq!(message, "MAT v5 mxOBJECT_CLASS is not supported");
        }
        other => panic!("expected UnsupportedFeature, got {:?}", other),
    }
}

#[test]
fn v5_multiple_variables_roundtrip() {
    let re_x = element(9, &1.0f64.to_le_bytes());
    let payload_x = matrix_payload(6, 0, 0, &[1i32, 1i32], b"x", &[re_x]);
    let elem_x = matrix_element(payload_x);

    let re_y = element(9, &2.0f64.to_le_bytes());
    let payload_y = matrix_payload(6, 0, 0, &[1i32, 1i32], b"y", &[re_y]);
    let elem_y = matrix_element(payload_y);

    let file = v5_file(&[elem_x, elem_y]);
    let mat = loadmat_bytes(&file).expect("two-variable v5 parse failed");

    assert_eq!(mat.version, MatVersion::V5);
    assert_eq!(mat.variables.len(), 2);

    assert_eq!(mat.variables[0].0, "x");
    assert_eq!(mat.variables[1].0, "y");

    let read_scalar = |arr: &MatArray| -> f64 {
        if let MatArray::Numeric(na) = arr {
            f64::from_le_bytes([
                na.real_data[0], na.real_data[1], na.real_data[2], na.real_data[3],
                na.real_data[4], na.real_data[5], na.real_data[6], na.real_data[7],
            ])
        } else {
            panic!("expected Numeric")
        }
    };

    assert_eq!(read_scalar(&mat.variables[0].1), 1.0);
    assert_eq!(read_scalar(&mat.variables[1].1), 2.0);
}

#[cfg(feature = "std")]
#[test]
fn loadmat_from_reader_parses_test_fixture() {
    use std::fs::File;
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/test_v5.mat");
    let f = File::open(path).expect("test_v5.mat must be accessible");
    let mat = consus_mat::loadmat(f).expect("loadmat from file failed");

    assert_eq!(mat.version, MatVersion::V5);
    assert_eq!(mat.variables.len(), 1);
    assert_eq!(mat.variables[0].0, "x");

    if let MatArray::Numeric(na) = &mat.variables[0].1 {
        assert_eq!(na.class, MatNumericClass::Double);
        assert_eq!(na.shape, vec![1, 3]);
        let vals: Vec<f64> = na.real_data.chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]]))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    } else {
        panic!("expected Numeric, got {:?}", mat.variables[0].1);
    }
}
