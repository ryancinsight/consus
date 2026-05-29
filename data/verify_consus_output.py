"""
Verify consus-hdf5 writer output using h5py.

Each ``--case`` value corresponds to one consus writer test scenario.
The script opens ``--file`` with h5py and asserts the expected structure
and values.  Exits 0 on success, 1 on verification failure, 2 when h5py
or numpy is unavailable.

Usage (from workspace root)::

    D:\\miniforge3\\python.exe data/verify_consus_output.py \\
        --file <path_to_consus_written_hdf5> --case <case_name>
"""

import argparse
import sys

try:
    import h5py
    import numpy as np
except ImportError as _e:
    print(f"Dependency unavailable: {_e}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Per-case verifiers
# ---------------------------------------------------------------------------

def _verify_scalar_i32(f: h5py.File) -> None:
    ds = f["scalar_i32"]
    assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
    assert ds.shape == (), f"shape: {ds.shape}"
    assert int(ds[()]) == 42, f"value: {ds[()]}"


def _verify_contiguous_1d_f64(f: h5py.File) -> None:
    ds = f["array_1d_f64"]
    expected = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    assert ds.dtype == np.dtype("float64"), f"dtype: {ds.dtype}"
    assert ds.shape == (4,), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values: {ds[:]}"


def _verify_contiguous_2d_i32(f: h5py.File) -> None:
    ds = f["array_2d_i32"]
    expected = np.arange(12, dtype=np.int32).reshape(3, 4)
    assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
    assert ds.shape == (3, 4), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values mismatch:\n{ds[:]}\nvs\n{expected}"


def _verify_contiguous_3d_f64(f: h5py.File) -> None:
    ds = f["array_3d_f64"]
    expected = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    assert ds.dtype == np.dtype("float64"), f"dtype: {ds.dtype}"
    assert ds.shape == (2, 3, 4), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values mismatch"


def _verify_chunked_1d_i32_v1(f: h5py.File) -> None:
    ds = f["chunked_1d_i32"]
    expected = np.arange(12, dtype=np.int32)
    assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
    assert ds.shape == (12,), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values: {ds[:]}"


def _verify_chunked_2d_f64_v1(f: h5py.File) -> None:
    ds = f["chunked_2d_f64"]
    expected = np.arange(24, dtype=np.float64).reshape(4, 6)
    assert ds.dtype == np.dtype("float64"), f"dtype: {ds.dtype}"
    assert ds.shape == (4, 6), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values mismatch"


def _verify_chunked_1d_i32_v4(f: h5py.File) -> None:
    ds = f["chunked_1d_i32_v4"]
    expected = np.arange(8, dtype=np.int32)
    assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
    assert ds.shape == (8,), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values: {ds[:]}"


def _verify_deflate_chunked_1d_i32(f: h5py.File) -> None:
    ds = f["deflate_1d_i32"]
    expected = np.arange(8, dtype=np.int32)
    assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
    assert ds.shape == (8,), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values: {ds[:]}"


def _verify_dataset_with_attr(f: h5py.File) -> None:
    ds = f["dataset_with_attr"]
    assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
    assert ds.shape == (), f"shape: {ds.shape}"
    assert int(ds[()]) == 7, f"dataset value: {ds[()]}"
    assert "answer" in ds.attrs, f"attribute 'answer' missing; attrs={list(ds.attrs)}"
    attr_val = int(ds.attrs["answer"])
    assert attr_val == 99, f"attribute 'answer' value: {attr_val}"


def _verify_group_nested_dataset(f: h5py.File) -> None:
    assert "grp" in f, f"group 'grp' missing; root keys={list(f.keys())}"
    grp = f["grp"]
    assert "nested_value" in grp, (
        f"dataset 'grp/nested_value' missing; grp keys={list(grp.keys())}"
    )
    ds = grp["nested_value"]
    assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
    assert ds.shape == (), f"shape: {ds.shape}"
    assert int(ds[()]) == 77, f"value: {ds[()]}"


def _verify_contiguous_1d_u8(f: h5py.File) -> None:
    ds = f["array_1d_u8"]
    expected = np.arange(8, dtype=np.uint8)
    assert ds.dtype == np.dtype("uint8"), f"dtype: {ds.dtype}"
    assert ds.shape == (8,), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values: {ds[:]}"


def _verify_contiguous_1d_f32(f: h5py.File) -> None:
    ds = f["array_1d_f32"]
    expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert ds.dtype == np.dtype("float32"), f"dtype: {ds.dtype}"
    assert ds.shape == (4,), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values: {ds[:]}"


def _verify_contiguous_1d_i16(f: h5py.File) -> None:
    ds = f["array_1d_i16"]
    expected = np.array([-100, 0, 100, 200], dtype=np.int16)
    assert ds.dtype == np.dtype("int16"), f"dtype: {ds.dtype}"
    assert ds.shape == (4,), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values: {ds[:]}"


def _verify_chunked_2d_f64_v4(f: h5py.File) -> None:
    ds = f["chunked_2d_f64_v4"]
    expected = np.arange(24, dtype=np.float64).reshape(4, 6)
    assert ds.dtype == np.dtype("float64"), f"dtype: {ds.dtype}"
    assert ds.shape == (4, 6), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values mismatch"


def _verify_multi_dataset_root(f: h5py.File) -> None:
    assert "ds_a" in f, f"dataset 'ds_a' missing; root keys={list(f.keys())}"
    assert "ds_b" in f, f"dataset 'ds_b' missing; root keys={list(f.keys())}"
    ds_a = f["ds_a"]
    ds_b = f["ds_b"]
    assert ds_a.dtype == np.dtype("int32"), f"ds_a dtype: {ds_a.dtype}"
    assert ds_b.dtype == np.dtype("float64"), f"ds_b dtype: {ds_b.dtype}"
    assert ds_a.shape == (), f"ds_a shape: {ds_a.shape}"
    assert ds_b.shape == (), f"ds_b shape: {ds_b.shape}"
    assert int(ds_a[()]) == 1, f"ds_a value: {ds_a[()]}"
    assert abs(float(ds_b[()]) - 2.5) < 1e-10, f"ds_b value: {ds_b[()]}"


def _verify_deflate_chunked_2d_i32(f: h5py.File) -> None:
    ds = f["deflate_2d_i32"]
    expected = np.arange(24, dtype=np.int32).reshape(3, 8)
    assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
    assert ds.shape == (3, 8), f"shape: {ds.shape}"
    assert np.array_equal(ds[:], expected), f"values mismatch"


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

CASES = {
    "scalar_i32":              _verify_scalar_i32,
    "contiguous_1d_f64":       _verify_contiguous_1d_f64,
    "contiguous_2d_i32":       _verify_contiguous_2d_i32,
    "contiguous_3d_f64":       _verify_contiguous_3d_f64,
    "chunked_1d_i32_v1":       _verify_chunked_1d_i32_v1,
    "chunked_2d_f64_v1":       _verify_chunked_2d_f64_v1,
    "chunked_1d_i32_v4":       _verify_chunked_1d_i32_v4,
    "deflate_chunked_1d_i32":  _verify_deflate_chunked_1d_i32,
    "dataset_with_attr":       _verify_dataset_with_attr,
    "group_nested_dataset":    _verify_group_nested_dataset,
    "contiguous_1d_u8":        _verify_contiguous_1d_u8,
    "contiguous_1d_f32":       _verify_contiguous_1d_f32,
    "contiguous_1d_i16":       _verify_contiguous_1d_i16,
    "chunked_2d_f64_v4":       _verify_chunked_2d_f64_v4,
    "multi_dataset_root":      _verify_multi_dataset_root,
    "deflate_chunked_2d_i32":  _verify_deflate_chunked_2d_i32,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a consus-hdf5-written HDF5 file using h5py."
    )
    parser.add_argument("--file", required=True, help="Path to the HDF5 file to verify.")
    parser.add_argument(
        "--case",
        required=True,
        help=f"Verification case name. One of: {sorted(CASES)}",
    )
    args = parser.parse_args()

    if args.case not in CASES:
        print(
            f"Unknown case: {args.case!r}. Known cases: {sorted(CASES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with h5py.File(args.file, "r") as f:
            CASES[args.case](f)
    except AssertionError as exc:
        print(f"FAIL [{args.case}]: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR [{args.case}]: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {args.case}")


if __name__ == "__main__":
    main()
