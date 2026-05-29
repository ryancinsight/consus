"""
Generate MAT-file test fixtures for the consus-mat scipy roundtrip tests.

Usage::

    D:\\miniforge3\\python.exe data/gen_mat_fixtures.py --case <name> --file <path>

Each ``--case`` writes one file at ``--file``.

| Case                 | Format | Contents                                            |
|----------------------|--------|-----------------------------------------------------|
| v5_double_array      | MAT v5 | scipy: arr float64 (1,8) = arange(8.0)             |
| v5_int32_array       | MAT v5 | scipy: iarr int32 (1,4) = [10,20,30,40]            |
| v5_char_array        | MAT v5 | scipy: s char 'HELLO'                               |
| v5_logical_array     | MAT v5 | scipy: logi logical (1,4) = [1,0,1,0]              |
| v5_struct_simple     | MAT v5 | scipy: st struct {a:1.0, b:2.0}                    |
| v5_double_matrix     | MAT v5 | scipy: mat float64 (3,4) = arange(12.0)            |
| mat_v73_double       | HDF5   | h5py: arr float64 (8,) MATLAB_class='double'        |
| mat_v73_int32        | HDF5   | h5py: iarr int32 (6,) MATLAB_class='int32'          |
"""

import argparse
import sys

try:
    import numpy as np
except ImportError as e:
    print(f"Dependency unavailable: {e}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# scipy.io v5 generators
# ---------------------------------------------------------------------------

def _require_scipy():
    try:
        import scipy.io
        return scipy.io
    except ImportError as e:
        print(f"scipy unavailable: {e}", file=sys.stderr)
        sys.exit(2)


def gen_v5_double_array(path: str) -> None:
    sio = _require_scipy()
    # scipy stores 1-D arrays as row vectors: shape (1, N) in MATLAB notation.
    sio.savemat(path, {"arr": np.arange(8.0, dtype=np.float64)})


def gen_v5_int32_array(path: str) -> None:
    sio = _require_scipy()
    sio.savemat(path, {"iarr": np.array([10, 20, 30, 40], dtype=np.int32)})


def gen_v5_char_array(path: str) -> None:
    sio = _require_scipy()
    sio.savemat(path, {"s": "HELLO"})


def gen_v5_logical_array(path: str) -> None:
    sio = _require_scipy()
    sio.savemat(path, {"logi": np.array([True, False, True, False], dtype=bool)})


def gen_v5_struct_simple(path: str) -> None:
    sio = _require_scipy()
    # scipy savemat encodes Python dicts as MATLAB structs.
    sio.savemat(path, {"st": {"a": np.float64(1.0), "b": np.float64(2.0)}})


def gen_v5_double_matrix(path: str) -> None:
    sio = _require_scipy()
    # 2-D matrix: MATLAB stores in column-major; scipy savemat handles transposition.
    sio.savemat(path, {"mat": np.arange(12.0, dtype=np.float64).reshape(3, 4)})


# ---------------------------------------------------------------------------
# h5py MAT-v7.3-compatible generators
# ---------------------------------------------------------------------------

def _require_h5py():
    try:
        import h5py
        return h5py
    except ImportError as e:
        print(f"h5py unavailable: {e}", file=sys.stderr)
        sys.exit(2)


def gen_mat_v73_double(path: str) -> None:
    """Write an HDF5 file with a single float64 dataset and MATLAB_class='double'.

    Produces HDF5 magic at offset 0, so consus-mat detects MatVersion::V73 and
    routes through consus-hdf5 to read the dataset.
    """
    h5py = _require_h5py()
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("arr", data=np.arange(8.0, dtype=np.float64))
        ds.attrs["MATLAB_class"] = np.bytes_("double")


def gen_mat_v73_int32(path: str) -> None:
    """HDF5 file with an int32 dataset and MATLAB_class='int32'."""
    h5py = _require_h5py()
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("iarr", data=np.array([10, 20, 30, 40, 50, 60], dtype=np.int32))
        ds.attrs["MATLAB_class"] = np.bytes_("int32")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "v5_double_array":   gen_v5_double_array,
    "v5_int32_array":    gen_v5_int32_array,
    "v5_char_array":     gen_v5_char_array,
    "v5_logical_array":  gen_v5_logical_array,
    "v5_struct_simple":  gen_v5_struct_simple,
    "v5_double_matrix":  gen_v5_double_matrix,
    "mat_v73_double":    gen_mat_v73_double,
    "mat_v73_int32":     gen_mat_v73_int32,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MAT-file fixtures for consus-mat roundtrip tests."
    )
    parser.add_argument(
        "--case", required=True,
        help=f"Case name. One of: {sorted(GENERATORS)}"
    )
    parser.add_argument("--file", required=True, help="Output file path.")
    args = parser.parse_args()

    if args.case not in GENERATORS:
        print(f"Unknown case: {args.case!r}. Known: {sorted(GENERATORS)}", file=sys.stderr)
        sys.exit(1)

    try:
        GENERATORS[args.case](args.file)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR [{args.case}]: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {args.case} → {args.file}")


if __name__ == "__main__":
    main()
