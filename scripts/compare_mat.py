"""compare_mat.py — MATLAB .mat compatibility tests between scipy/h5py and consus.

Tests
-----
1.  scipy writes f64 1-D array  → consus reads (array_class, shape, values)
2.  scipy writes i32 array      → consus reads
3.  scipy writes char array     → consus reads text
4.  scipy writes logical array  → consus reads booleans
5.  scipy writes 2-D f64 matrix → consus reads shape and values (column-major)
6.  scipy writes struct         → consus reads field names
7.  h5py writes v7.3 .mat       → consus reads (routes through consus-hdf5)
"""

import io
import struct
import sys

import numpy as np
import scipy.io
import h5py

import consus


PASS = "\u2713"
FAIL = "\u2717"

_results: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    _results.append((label, ok, detail))
    symbol = PASS if ok else FAIL
    line = f"  {symbol} {label}"
    if detail:
        line += f": {detail}"
    print(line)


def _scipy_to_bytes(variables: dict) -> bytes:
    """Write variables dict to a MAT v5 byte string via scipy.io.savemat."""
    buf = io.BytesIO()
    scipy.io.savemat(buf, variables)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test 1: scipy writes f64 1-D array → consus reads
# ---------------------------------------------------------------------------
def test_scipy_f64_1d() -> None:
    label = "scipy writes f64 1-D → consus reads"
    try:
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        raw = _scipy_to_bytes({"arr": arr})

        mat = consus.loadmat_bytes(raw)
        var = mat.get_variable("arr")
        assert var is not None, "variable 'arr' not found"

        # scipy wraps 1-D as a [1, n] row vector
        class_ok = var.array_class == "numeric"
        shape_ok = var.shape == [1, 4]
        nc_ok = var.numeric_class == "double"
        data = var.read_data()
        expected = struct.pack("<4d", 1.0, 2.0, 3.0, 4.0)
        data_ok = bytes(data) == expected
        check(label, class_ok and shape_ok and nc_ok and data_ok,
              f"class={var.array_class} shape={var.shape} nc={var.numeric_class}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 2: scipy writes i32 array → consus reads
# ---------------------------------------------------------------------------
def test_scipy_i32() -> None:
    label = "scipy writes i32 array → consus reads"
    try:
        arr = np.array([10, 20, 30], dtype=np.int32)
        raw = _scipy_to_bytes({"iarr": arr})

        mat = consus.loadmat_bytes(raw)
        var = mat.get_variable("iarr")
        assert var is not None, "variable 'iarr' not found"

        class_ok = var.array_class == "numeric"
        nc_ok = var.numeric_class == "int32"
        data = var.read_data()
        vals = list(struct.unpack("<3i", bytes(data)))
        vals_ok = vals == [10, 20, 30]
        check(label, class_ok and nc_ok and vals_ok,
              f"nc={var.numeric_class} shape={var.shape} vals={vals}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 3: scipy writes char array → consus reads text
# ---------------------------------------------------------------------------
def test_scipy_char() -> None:
    label = "scipy writes char array → consus reads text"
    try:
        raw = _scipy_to_bytes({"s": "hello"})

        mat = consus.loadmat_bytes(raw)
        var = mat.get_variable("s")
        assert var is not None, "variable 's' not found"

        class_ok = var.array_class == "char"
        text_ok = var.text == "hello"
        check(label, class_ok and text_ok,
              f"class={var.array_class} text={var.text!r}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 4: scipy writes logical array → consus reads booleans
# ---------------------------------------------------------------------------
def test_scipy_logical() -> None:
    label = "scipy writes logical array → consus reads booleans"
    try:
        arr = np.array([True, False, True, False], dtype=bool)
        raw = _scipy_to_bytes({"logi": arr})

        mat = consus.loadmat_bytes(raw)
        var = mat.get_variable("logi")
        assert var is not None, "variable 'logi' not found"

        class_ok = var.array_class == "logical"
        bools_ok = var.bools == [True, False, True, False]
        check(label, class_ok and bools_ok,
              f"class={var.array_class} bools={var.bools}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 5: scipy writes 2-D f64 matrix → consus reads shape and values
# ---------------------------------------------------------------------------
def test_scipy_f64_2d_matrix() -> None:
    label = "scipy writes 2-D f64 matrix → consus reads"
    try:
        mat_arr = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]], dtype=np.float64)
        raw = _scipy_to_bytes({"mat": mat_arr})

        mat = consus.loadmat_bytes(raw)
        var = mat.get_variable("mat")
        assert var is not None, "variable 'mat' not found"

        class_ok = var.array_class == "numeric"
        nc_ok = var.numeric_class == "double"
        shape_ok = var.shape == [2, 3]
        # MATLAB stores in column-major order: [1,4,2,5,3,6]
        data = var.read_data()
        col_major = list(struct.unpack("<6d", bytes(data)))
        expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        vals_ok = all(abs(a - b) < 1e-12 for a, b in zip(col_major, expected))
        check(label, class_ok and nc_ok and shape_ok and vals_ok,
              f"shape={var.shape} col_major={col_major}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 6: scipy writes struct → consus reads field names
# ---------------------------------------------------------------------------
def test_scipy_struct() -> None:
    label = "scipy writes struct → consus reads field names"
    try:
        st = np.zeros(1, dtype=[("x", np.float64), ("y", np.float64)])
        st["x"] = 1.0
        st["y"] = 2.0
        raw = _scipy_to_bytes({"st": st})

        mat = consus.loadmat_bytes(raw)
        var = mat.get_variable("st")
        assert var is not None, "variable 'st' not found"

        class_ok = var.array_class == "struct"
        fields_ok = sorted(var.field_names) == ["x", "y"]
        check(label, class_ok and fields_ok,
              f"class={var.array_class} fields={var.field_names}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 7: h5py writes v7.3 .mat → consus reads via consus-hdf5 routing
# ---------------------------------------------------------------------------
def test_h5py_v73_mat() -> None:
    label = "h5py writes v7.3 .mat → consus reads"
    try:
        # HDF5-backed .mat files use HDF5 magic.  consus detects HDF5 magic and
        # routes through consus-hdf5, reporting version "v7.3".
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            ds = f.create_dataset("arr", data=np.array([10.0, 20.0, 30.0], dtype=np.float64))
            ds.attrs["MATLAB_class"] = np.bytes_("double")
        raw = buf.getvalue()

        mat = consus.loadmat_bytes(raw)
        version_ok = mat.version == "v7.3"
        var = mat.get_variable("arr")
        assert var is not None, "variable 'arr' not found"

        class_ok = var.array_class == "numeric"
        nc_ok = var.numeric_class == "double"
        shape_ok = len(var.shape) >= 1 and 3 in var.shape
        data = var.read_data()
        vals = list(struct.unpack("<3d", bytes(data)))
        vals_ok = vals == [10.0, 20.0, 30.0]
        check(label, version_ok and class_ok and nc_ok and shape_ok and vals_ok,
              f"version={mat.version} shape={var.shape} vals={vals}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("compare_mat.py \u2014 MATLAB .mat compatibility tests\n")
    test_scipy_f64_1d()
    test_scipy_i32()
    test_scipy_char()
    test_scipy_logical()
    test_scipy_f64_2d_matrix()
    test_scipy_struct()
    test_h5py_v73_mat()

    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{passed}/{total} passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
