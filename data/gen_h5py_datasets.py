"""
Generate a deterministic HDF5 dataset fixture for consus integration tests.

Run from the workspace root:
    D:\\miniforge3\\python.exe data/gen_h5py_datasets.py

Output: data/h5py_datasets_sample.h5

Dataset manifest
----------------
/scalar_i32             int32 scalar,          value 42
/scalar_f64             float64 scalar,        value 3.14159265358979
/array_1d_i32           int32 1-D, shape=(8,), values 0..7
/array_1d_f64           float64 1-D, shape=(4,), values [1.5, 2.5, 3.5, 4.5]
/array_2d_f64           float64 2-D, shape=(3,4), row-major values 0.0..11.0
/array_3d_i32           int32 3-D, shape=(2,3,4), values 0..23
/array_1d_i16           int16 1-D, shape=(6,), values [10, 20, 30, 40, 50, 60]
/array_1d_u8            uint8 1-D, shape=(5,), values [255, 128, 64, 32, 16]
/array_1d_f32           float32 1-D, shape=(4,), values [1.0, 2.0, 3.0, 4.0]
/chunked_1d_i32         int32 1-D chunked (4,), shape=(12,), values 0..11
/chunked_2d_f64         float64 2-D chunked (2,3), shape=(4,6), row-major 0.0..23.0
/grp                    group
/grp/nested_i32         int32 scalar,          value 77
/grp/nested_f64         float64 1-D, shape=(3,), values [10.0, 20.0, 30.0]
"""

import os

import h5py
import numpy as np

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "h5py_datasets_sample.h5"
)

with h5py.File(OUT, "w") as f:
    # ── /scalar_i32 ──────────────────────────────────────────────────────
    f.create_dataset("scalar_i32", data=np.int32(42))

    # ── /scalar_f64 ──────────────────────────────────────────────────────
    f.create_dataset("scalar_f64", data=np.float64(3.14159265358979))

    # ── /array_1d_i32 ────────────────────────────────────────────────────
    f.create_dataset("array_1d_i32", data=np.arange(8, dtype=np.int32))

    # ── /array_1d_f64 ────────────────────────────────────────────────────
    f.create_dataset(
        "array_1d_f64",
        data=np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64),
    )

    # ── /array_2d_f64 ────────────────────────────────────────────────────
    f.create_dataset(
        "array_2d_f64",
        data=np.arange(12, dtype=np.float64).reshape(3, 4),
    )

    # ── /array_3d_i32 ────────────────────────────────────────────────────
    f.create_dataset(
        "array_3d_i32",
        data=np.arange(24, dtype=np.int32).reshape(2, 3, 4),
    )

    # ── /array_1d_i16 ────────────────────────────────────────────────────
    f.create_dataset(
        "array_1d_i16",
        data=np.array([10, 20, 30, 40, 50, 60], dtype=np.int16),
    )

    # ── /array_1d_u8 ─────────────────────────────────────────────────────
    f.create_dataset(
        "array_1d_u8",
        data=np.array([255, 128, 64, 32, 16], dtype=np.uint8),
    )

    # ── /array_1d_f32 ────────────────────────────────────────────────────
    f.create_dataset(
        "array_1d_f32",
        data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )

    # ── /chunked_1d_i32 (explicitly chunked) ─────────────────────────────
    f.create_dataset(
        "chunked_1d_i32",
        data=np.arange(12, dtype=np.int32),
        chunks=(4,),
    )

    # ── /chunked_2d_f64 (explicitly chunked) ─────────────────────────────
    f.create_dataset(
        "chunked_2d_f64",
        data=np.arange(24, dtype=np.float64).reshape(4, 6),
        chunks=(2, 3),
    )

    # ── /grp ─────────────────────────────────────────────────────────────
    grp = f.create_group("grp")

    # ── /grp/nested_i32 ──────────────────────────────────────────────────
    grp.create_dataset("nested_i32", data=np.int32(77))

    # ── /grp/nested_f64 ──────────────────────────────────────────────────
    grp.create_dataset(
        "nested_f64",
        data=np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )

print(f"Written: {OUT}")
print()

# ── Self-verification ─────────────────────────────────────────────────────
with h5py.File(OUT, "r") as f:
    print("=== Manifest ===")

    ds = f["scalar_i32"]
    v = int(ds[()])
    assert v == 42, f"scalar_i32 mismatch: {v}"
    print(f"  /scalar_i32      dtype={ds.dtype}  shape={ds.shape}  value={v}")

    ds = f["scalar_f64"]
    v = float(ds[()])
    assert abs(v - 3.14159265358979) < 1e-14, f"scalar_f64 mismatch: {v}"
    print(f"  /scalar_f64      dtype={ds.dtype}  shape={ds.shape}  value={v}")

    ds = f["array_1d_i32"]
    arr = ds[:]
    assert list(arr) == list(range(8)), f"array_1d_i32 mismatch: {arr}"
    print(f"  /array_1d_i32    dtype={ds.dtype}  shape={ds.shape}  values={list(arr)}")

    ds = f["array_1d_f64"]
    arr = ds[:]
    assert list(arr) == [1.5, 2.5, 3.5, 4.5], f"array_1d_f64 mismatch: {arr}"
    print(f"  /array_1d_f64    dtype={ds.dtype}  shape={ds.shape}  values={list(arr)}")

    ds = f["array_2d_f64"]
    arr = ds[:]
    assert arr.shape == (3, 4), f"array_2d_f64 shape mismatch: {arr.shape}"
    expected = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(arr, expected), f"array_2d_f64 values mismatch"
    print(f"  /array_2d_f64    dtype={ds.dtype}  shape={ds.shape}")

    ds = f["array_3d_i32"]
    arr = ds[:]
    assert arr.shape == (2, 3, 4), f"array_3d_i32 shape mismatch: {arr.shape}"
    expected = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    assert np.array_equal(arr, expected), f"array_3d_i32 values mismatch"
    print(f"  /array_3d_i32    dtype={ds.dtype}  shape={ds.shape}")

    ds = f["array_1d_i16"]
    arr = ds[:]
    assert list(arr) == [10, 20, 30, 40, 50, 60], f"array_1d_i16 mismatch: {arr}"
    print(f"  /array_1d_i16    dtype={ds.dtype}  shape={ds.shape}  values={list(arr)}")

    ds = f["array_1d_u8"]
    arr = ds[:]
    assert list(arr) == [255, 128, 64, 32, 16], f"array_1d_u8 mismatch: {arr}"
    print(f"  /array_1d_u8     dtype={ds.dtype}  shape={ds.shape}  values={list(arr)}")

    ds = f["array_1d_f32"]
    arr = ds[:]
    assert list(arr) == [1.0, 2.0, 3.0, 4.0], f"array_1d_f32 mismatch: {arr}"
    print(f"  /array_1d_f32    dtype={ds.dtype}  shape={ds.shape}  values={list(arr)}")

    ds = f["chunked_1d_i32"]
    arr = ds[:]
    assert list(arr) == list(range(12)), f"chunked_1d_i32 mismatch: {arr}"
    print(f"  /chunked_1d_i32  dtype={ds.dtype}  shape={ds.shape}  chunks={ds.chunks}  values={list(arr)}")

    ds = f["chunked_2d_f64"]
    arr = ds[:]
    assert arr.shape == (4, 6), f"chunked_2d_f64 shape mismatch: {arr.shape}"
    expected = np.arange(24, dtype=np.float64).reshape(4, 6)
    assert np.array_equal(arr, expected), f"chunked_2d_f64 values mismatch"
    print(f"  /chunked_2d_f64  dtype={ds.dtype}  shape={ds.shape}  chunks={ds.chunks}")

    grp = f["grp"]
    assert "nested_i32" in grp, "grp/nested_i32 missing"
    assert "nested_f64" in grp, "grp/nested_f64 missing"

    ds = grp["nested_i32"]
    v = int(ds[()])
    assert v == 77, f"grp/nested_i32 mismatch: {v}"
    print(f"  /grp/nested_i32  dtype={ds.dtype}  shape={ds.shape}  value={v}")

    ds = grp["nested_f64"]
    arr = ds[:]
    assert list(arr) == [10.0, 20.0, 30.0], f"grp/nested_f64 mismatch: {arr}"
    print(f"  /grp/nested_f64  dtype={ds.dtype}  shape={ds.shape}  values={list(arr)}")

    # Report storage layouts
    print()
    print("=== Storage layouts ===")
    for name in [
        "scalar_i32", "scalar_f64", "array_1d_i32", "array_1d_f64",
        "array_2d_f64", "array_3d_i32", "array_1d_i16", "array_1d_u8",
        "array_1d_f32", "chunked_1d_i32", "chunked_2d_f64",
    ]:
        ds = f[name]
        layout = "chunked" if ds.chunks is not None else "contiguous/compact"
        print(f"  /{name:<20}  {layout}")

print()
print("Self-verification passed.")
