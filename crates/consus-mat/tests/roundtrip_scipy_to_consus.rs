//! Roundtrip tests: scipy.io / h5py writers → consus-mat reader.
//!
//! ## Purpose
//!
//! These tests verify that consus-mat correctly reads:
//!
//! 1. MAT v5 files produced by `scipy.io.savemat` (double array, int32 array,
//!    char array, logical array, struct, matrix).
//! 2. HDF5 files with MATLAB_class attributes produced by h5py — exercising the
//!    `loadmat_bytes` → `MatVersion::V73` → consus-hdf5 routing path.
//!
//! Every test skips gracefully when Python, scipy, or h5py is unavailable.
//!
//! ## Routing Invariant
//!
//! `loadmat_bytes` dispatches on the leading bytes of the input:
//! - HDF5 magic at byte 0 → `MatVersion::V73` → `v73::read_mat_v73` →
//!   `consus_hdf5::Hdf5File` (full routing through consus-hdf5)
//! - MATLAB v5 endian indicator at bytes 126-127 → `MatVersion::V5`
//! - Otherwise → `MatVersion::V4`
//!
//! ## Coverage
//!
//! | Test                                    | Format  | Variable | Class   | Shape  |
//! |-----------------------------------------|---------|----------|---------|--------|
//! | `scipy_v5_double_array`                 | MAT v5  | arr      | Double  | [1, 8] |
//! | `scipy_v5_int32_array`                  | MAT v5  | iarr     | Int32   | [1, 4] |
//! | `scipy_v5_char_array`                   | MAT v5  | s        | Char    | [1, 5] |
//! | `scipy_v5_logical_array`                | MAT v5  | logi     | Logical | [1, 4] |
//! | `scipy_v5_struct_simple`                | MAT v5  | st       | Struct  | [1, 1] |
//! | `scipy_v5_double_matrix`                | MAT v5  | mat      | Double  | [3, 4] |
//! | `h5py_mat_v73_routes_through_consus_hdf5` | HDF5  | arr      | Double  | [8]   |
//! | `h5py_mat_v73_int32_routes_through_consus_hdf5` | HDF5 | iarr | Int32  | [6]   |

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

use consus_mat::{MatArray, MatNumericClass, MatVersion, loadmat_bytes};

/// Serializes Python subprocess spawns to prevent concurrent HDF5/zlib DLL
/// initialization races on Windows when many processes start simultaneously.
static PYTHON_SPAWN_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Infrastructure helpers
// ---------------------------------------------------------------------------

fn gen_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data")
        .join("gen_mat_fixtures.py")
}

/// Locate a Python interpreter that has scipy available.
///
/// Returns `None` when no suitable interpreter is found, causing tests to
/// skip rather than fail.
fn find_python_with_scipy() -> Option<String> {
    for candidate in &[r"D:\miniforge3\python.exe", "python3", "python"] {
        let ok = Command::new(candidate)
            .args(["-c", "import scipy.io"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some((*candidate).to_string());
        }
    }
    None
}

/// Locate a Python interpreter that has h5py available.
fn find_python_with_h5py() -> Option<String> {
    for candidate in &[r"D:\miniforge3\python.exe", "python3", "python"] {
        let ok = Command::new(candidate)
            .args(["-c", "import h5py"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some((*candidate).to_string());
        }
    }
    None
}

/// Generate a MAT-file fixture via the Python script and return its bytes.
///
/// Returns `None` when Python/dependencies are unavailable or generation fails.
fn generate_fixture(case: &str, python: &str) -> Option<Vec<u8>> {
    let script = gen_script();
    if !script.exists() {
        eprintln!("Skipping {case}: gen_mat_fixtures.py not found at {script:?}");
        return None;
    }

    let tmp = tempfile::NamedTempFile::new().ok()?;
    let tmp_path = tmp.path().to_owned();

    let _guard = PYTHON_SPAWN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let output = Command::new(python)
        .args([
            script.to_str().unwrap(),
            "--case",
            case,
            "--file",
            tmp_path.to_str().unwrap(),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("Skipping {case}: fixture generation failed: {stderr}");
        return None;
    }

    std::fs::read(&tmp_path).ok()
}

// ---------------------------------------------------------------------------
// MAT v5 tests: scipy.io.savemat output
// ---------------------------------------------------------------------------

/// scipy.io.savemat writes a 1-D float64 array `arr = arange(8.0)`.
/// consus-mat reads it as MatVersion::V5, MatNumericClass::Double, shape [1,8].
///
/// ## Invariant
///
/// scipy-generated MAT v5 double arrays are correctly detected and decoded
/// by consus-mat's v5 reader.
#[test]
fn scipy_v5_double_array_parsed_by_consus_mat() {
    let python = match find_python_with_scipy() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: scipy not available");
            return;
        }
    };
    let bytes = match generate_fixture("v5_double_array", &python) {
        Some(b) => b,
        None => return,
    };

    let mat = loadmat_bytes(&bytes).expect("loadmat_bytes failed");

    assert_eq!(mat.version, MatVersion::V5, "version mismatch");
    assert_eq!(mat.variables.len(), 1, "expected exactly 1 variable");

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "arr", "variable name mismatch");

    let MatArray::Numeric(na) = arr else {
        panic!("expected MatArray::Numeric, got {arr:?}");
    };
    assert_eq!(na.class, MatNumericClass::Double, "class mismatch");
    // scipy saves 1-D arrays as MATLAB row vectors: shape [1, 8].
    assert_eq!(na.shape, vec![1usize, 8], "shape mismatch");
    assert_eq!(na.numel(), 8, "numel mismatch");

    // Check first and last values.
    let vals: Vec<f64> = na
        .real_data
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(vals[0], 0.0, "first value");
    assert_eq!(vals[7], 7.0, "last value");
}

/// scipy.io.savemat writes a 1-D int32 array `iarr = [10,20,30,40]`.
/// consus-mat reads it as MatVersion::V5, MatNumericClass::Int32, shape [1,4].
///
/// ## Invariant
///
/// scipy-generated MAT v5 int32 arrays are correctly typed and shaped.
#[test]
fn scipy_v5_int32_array_parsed_by_consus_mat() {
    let python = match find_python_with_scipy() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: scipy not available");
            return;
        }
    };
    let bytes = match generate_fixture("v5_int32_array", &python) {
        Some(b) => b,
        None => return,
    };

    let mat = loadmat_bytes(&bytes).expect("loadmat_bytes failed");

    assert_eq!(mat.version, MatVersion::V5, "version mismatch");

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "iarr", "variable name");

    let MatArray::Numeric(na) = arr else {
        panic!("expected MatArray::Numeric, got {arr:?}");
    };
    assert_eq!(na.class, MatNumericClass::Int32, "class mismatch");
    assert_eq!(na.shape, vec![1usize, 4], "shape mismatch");

    let vals: Vec<i32> = na
        .real_data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(vals, vec![10, 20, 30, 40], "values mismatch");
}

/// scipy.io.savemat writes a MATLAB char array `s = 'HELLO'`.
/// consus-mat reads it as MatVersion::V5, MatArray::Char, shape [1,5],
/// data "HELLO".
///
/// ## Invariant
///
/// scipy-generated MAT v5 char arrays are correctly decoded as UTF-8 strings.
#[test]
fn scipy_v5_char_array_parsed_by_consus_mat() {
    let python = match find_python_with_scipy() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: scipy not available");
            return;
        }
    };
    let bytes = match generate_fixture("v5_char_array", &python) {
        Some(b) => b,
        None => return,
    };

    let mat = loadmat_bytes(&bytes).expect("loadmat_bytes failed");

    assert_eq!(mat.version, MatVersion::V5, "version mismatch");

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "s", "variable name");

    let MatArray::Char(ca) = arr else {
        panic!("expected MatArray::Char, got {arr:?}");
    };
    // MATLAB char row vector of length 5: shape [1, 5].
    assert_eq!(ca.shape, vec![1usize, 5], "shape mismatch");
    assert_eq!(ca.data, "HELLO", "char data mismatch");
}

/// scipy.io.savemat writes a MATLAB logical array `logi = [T,F,T,F]`.
/// consus-mat reads it as MatVersion::V5, MatArray::Logical, shape [1,4].
///
/// ## Invariant
///
/// scipy-generated MAT v5 logical arrays are correctly decoded as bool vectors.
#[test]
fn scipy_v5_logical_array_parsed_by_consus_mat() {
    let python = match find_python_with_scipy() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: scipy not available");
            return;
        }
    };
    let bytes = match generate_fixture("v5_logical_array", &python) {
        Some(b) => b,
        None => return,
    };

    let mat = loadmat_bytes(&bytes).expect("loadmat_bytes failed");

    assert_eq!(mat.version, MatVersion::V5, "version mismatch");

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "logi", "variable name");

    let MatArray::Logical(la) = arr else {
        panic!("expected MatArray::Logical, got {arr:?}");
    };
    assert_eq!(la.shape, vec![1usize, 4], "shape mismatch");
    assert_eq!(
        la.data,
        vec![true, false, true, false],
        "logical data mismatch"
    );
}

/// scipy.io.savemat writes a MATLAB struct `st` with fields a=1.0, b=2.0.
/// consus-mat reads it as MatVersion::V5, MatArray::Struct, shape [1,1].
///
/// ## Invariant
///
/// scipy-generated MAT v5 struct arrays are correctly decoded by consus-mat.
#[test]
fn scipy_v5_struct_simple_parsed_by_consus_mat() {
    let python = match find_python_with_scipy() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: scipy not available");
            return;
        }
    };
    let bytes = match generate_fixture("v5_struct_simple", &python) {
        Some(b) => b,
        None => return,
    };

    let mat = loadmat_bytes(&bytes).expect("loadmat_bytes failed");

    assert_eq!(mat.version, MatVersion::V5, "version mismatch");

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "st", "variable name");

    let MatArray::Struct(sa) = arr else {
        panic!("expected MatArray::Struct, got {arr:?}");
    };
    // Scalar struct: shape [1,1].
    assert_eq!(sa.shape, vec![1usize, 1], "shape mismatch");

    // Fields 'a' and 'b' must be present (order may vary).
    let field_names: Vec<&str> = sa.data.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        field_names.contains(&"a"),
        "field 'a' missing; fields={field_names:?}"
    );
    assert!(
        field_names.contains(&"b"),
        "field 'b' missing; fields={field_names:?}"
    );

    // Field 'a' must decode to 1.0.
    let a_field = sa
        .data
        .iter()
        .find(|(n, _)| n == "a")
        .map(|(_, vals)| &vals[0])
        .expect("field 'a' not found");
    let MatArray::Numeric(a_na) = a_field else {
        panic!("field 'a' expected Numeric, got {a_field:?}");
    };
    assert_eq!(a_na.class, MatNumericClass::Double, "field 'a' class");
    let a_val = f64::from_le_bytes(a_na.real_data[..8].try_into().unwrap());
    assert!((a_val - 1.0).abs() < 1e-12, "field 'a' value: {a_val}");
}

/// scipy.io.savemat writes a 3×4 float64 matrix `mat = arange(12.0).reshape(3,4)`.
/// consus-mat reads it as MatVersion::V5, Double, shape [3,4], 12 elements.
///
/// ## Invariant
///
/// Multi-dimensional scipy-generated MAT v5 matrices are correctly shaped.
#[test]
fn scipy_v5_double_matrix_parsed_by_consus_mat() {
    let python = match find_python_with_scipy() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: scipy not available");
            return;
        }
    };
    let bytes = match generate_fixture("v5_double_matrix", &python) {
        Some(b) => b,
        None => return,
    };

    let mat = loadmat_bytes(&bytes).expect("loadmat_bytes failed");

    assert_eq!(mat.version, MatVersion::V5, "version mismatch");

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "mat", "variable name");

    let MatArray::Numeric(na) = arr else {
        panic!("expected MatArray::Numeric, got {arr:?}");
    };
    assert_eq!(na.class, MatNumericClass::Double, "class mismatch");
    // scipy saves np.arange(12).reshape(3,4) as MATLAB 3×4 matrix.
    assert_eq!(na.shape, vec![3usize, 4], "shape mismatch");
    assert_eq!(na.numel(), 12, "numel mismatch");

    // Data is stored column-major: [0,3,6,9, 1,4,7,10, 2,5,8,11] for C-ordered input.
    // Just verify total element count and that we can parse all bytes.
    let vals: Vec<f64> = na
        .real_data
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(vals.len(), 12, "element count");
    // All values in [0.0, 11.0].
    let mut sorted = vals.clone();
    sorted.sort_by(f64::total_cmp);
    for (i, &v) in sorted.iter().enumerate() {
        assert!((v - i as f64).abs() < 1e-12, "sorted[{i}] = {v}");
    }
}

// ---------------------------------------------------------------------------
// MAT v7.3 routing tests: h5py HDF5 with MATLAB_class attributes
// ---------------------------------------------------------------------------

/// h5py writes an HDF5 file with `arr` (float64, shape (8,)) and
/// `MATLAB_class = b'double'` attribute.  consus-mat detects HDF5 magic at
/// byte 0, routes to `MatVersion::V73`, and dispatches through consus-hdf5.
///
/// ## Routing Invariant
///
/// `loadmat_bytes` on a file with HDF5 magic → `MatVersion::V73` →
/// `v73::read_mat_v73` → `Hdf5File::open(SliceReader::new(data))`.
/// The variable 'arr' is read as Double, shape [8], values 0.0..7.0.
#[test]
fn h5py_mat_v73_routes_through_consus_hdf5() {
    let python = match find_python_with_h5py() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: h5py not available");
            return;
        }
    };
    let bytes = match generate_fixture("mat_v73_double", &python) {
        Some(b) => b,
        None => return,
    };

    // HDF5 magic must be at byte 0 — confirm the routing precondition.
    assert_eq!(
        &bytes[..8],
        b"\x89HDF\r\n\x1a\n",
        "expected HDF5 magic at byte 0"
    );

    let mat = loadmat_bytes(&bytes).expect("loadmat_bytes failed");

    // The routing must have selected V73 (not V5 or V4).
    assert_eq!(
        mat.version,
        MatVersion::V73,
        "expected MatVersion::V73 routing"
    );
    assert_eq!(mat.variables.len(), 1, "expected 1 variable");

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "arr", "variable name");

    let MatArray::Numeric(na) = arr else {
        panic!("expected MatArray::Numeric, got {arr:?}");
    };
    assert_eq!(na.class, MatNumericClass::Double, "class mismatch");
    // v73 reader reverses HDF5 dims: shape (8,) → [8].
    assert_eq!(na.shape, vec![8usize], "shape mismatch");
    assert_eq!(na.numel(), 8, "numel");

    let vals: Vec<f64> = na
        .real_data
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let expected: Vec<f64> = (0..8).map(|i| i as f64).collect();
    assert_eq!(vals, expected, "values mismatch");
}

/// h5py writes an HDF5 file with `iarr` (int32, shape (6,)) and
/// `MATLAB_class = b'int32'` attribute.  consus-mat routes to V73 and
/// reads it as MatNumericClass::Int32.
///
/// ## Invariant
///
/// consus-mat v73 routing dispatches non-double numeric classes correctly.
#[test]
fn h5py_mat_v73_int32_routes_through_consus_hdf5() {
    let python = match find_python_with_h5py() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: h5py not available");
            return;
        }
    };
    let bytes = match generate_fixture("mat_v73_int32", &python) {
        Some(b) => b,
        None => return,
    };

    let mat = loadmat_bytes(&bytes).expect("loadmat_bytes failed");

    assert_eq!(mat.version, MatVersion::V73, "expected V73 routing");

    let (name, arr) = &mat.variables[0];
    assert_eq!(name, "iarr", "variable name");

    let MatArray::Numeric(na) = arr else {
        panic!("expected MatArray::Numeric, got {arr:?}");
    };
    assert_eq!(na.class, MatNumericClass::Int32, "class mismatch");
    assert_eq!(na.shape, vec![6usize], "shape mismatch");

    let vals: Vec<i32> = na
        .real_data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(vals, vec![10, 20, 30, 40, 50, 60], "values mismatch");
}
