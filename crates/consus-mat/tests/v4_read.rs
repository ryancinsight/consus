//! MAT v4 read tests.
use consus_mat::{loadmat_bytes, MatArray, MatNumericClass};

#[test]
fn v4_double_array_shape_and_values() {
    let data = include_bytes!("test_v4.mat");
    let mat = loadmat_bytes(data).expect("v4 parse failed");
    assert_eq!(mat.variables.len(), 1, "expected 1 variable");
    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "A", "variable name");
    if let MatArray::Numeric(na) = arr {
        assert_eq!(na.class, MatNumericClass::Double);
        assert_eq!(na.shape, vec![2, 3]);
        assert_eq!(na.numel(), 6);
        assert_eq!(na.real_data.len(), 6 * 8);
        let vals: Vec<f64> = na.real_data.chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]]))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    } else {
        panic!("expected Numeric array, got {:?}", arr);
    }
}

#[test]
fn v4_empty_slice_returns_error() {
    // An empty byte slice has no MAT header: detect_version returns None -> error.
    let data = [0u8; 0];
    assert!(loadmat_bytes(&data).is_err(), "empty slice should return error");
}

#[test]
fn v4_truncated_header_returns_error() {
    let data = [0u8; 10];
    let result = loadmat_bytes(&data);
    assert!(result.is_err(), "truncated v4 should fail");
}

#[test]
fn v4_sparse_matrix_returns_unsupported_feature_error() {
    // Minimal synthetic v4 sparse record.
    // type_code = 2 (M=0, P=0, T=2 → sparse), mrows=0, ncols=0, imagf=0, namlen=3.
    // With mrows=0 and ncols=0 the parser reads zero data bytes before reaching
    // the matrix_type==2 match arm, which returns UnsupportedFeature.
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(&2u32.to_le_bytes());   // type_code = 2
    data.extend_from_slice(&0u32.to_le_bytes());   // mrows    = 0
    data.extend_from_slice(&0u32.to_le_bytes());   // ncols    = 0
    data.extend_from_slice(&0u32.to_le_bytes());   // imagf    = 0
    data.extend_from_slice(&3u32.to_le_bytes());   // namlen   = 3
    data.extend_from_slice(b"sp\0");               // name

    let err = consus_mat::loadmat_bytes(&data)
        .expect_err("v4 sparse matrix must return Err");
    match err {
        consus_mat::MatError::UnsupportedFeature(message) => {
            assert_eq!(message, "MAT v4 sparse matrices are not supported");
        }
        other => panic!("expected UnsupportedFeature, got {:?}", other),
    }
}
