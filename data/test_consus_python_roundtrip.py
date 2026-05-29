"""
End-to-end roundtrip test for the consus-python extension module.

Tests both directions:
  - consus-python writes HDF5 → h5py reads back and verifies
  - h5py writes HDF5 → consus-python reads back and verifies

Usage::

    D:\\miniforge3\\python.exe data/test_consus_python_roundtrip.py --pyd-dir <dir>

``--pyd-dir`` must be a directory containing ``consus.pyd`` (Windows) or
``consus.so`` (Linux/macOS) built by ``cargo build -p consus-python``.
"""

import argparse
import importlib.util
import io
import struct
import sys
import tempfile
import os

try:
    import h5py
    import numpy as np
except ImportError as e:
    print(f"Dependency unavailable: {e}", file=sys.stderr)
    sys.exit(2)


def load_consus(pyd_dir: str):
    """Load the consus extension module from ``pyd_dir``."""
    for ext in ("consus.pyd", "consus.so", "consus.dll"):
        path = os.path.join(pyd_dir, ext)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("consus", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError(
        f"consus extension module not found in {pyd_dir!r}; "
        "run `cargo build -p consus-python` first"
    )


# ---------------------------------------------------------------------------
# Direction 1: consus-python writes → h5py reads
# ---------------------------------------------------------------------------

def test_write_scalar_i32(consus, tmp_dir: str) -> None:
    """consus-python writes int32 scalar=42 → h5py reads 42."""
    b = consus.FileBuilder()
    b.add_dataset("scalar_i32", "<i4", [], struct.pack("<i", 42))
    data = bytes(b.finish())

    path = os.path.join(tmp_dir, "scalar_i32.h5")
    with open(path, "wb") as f:
        f.write(data)

    with h5py.File(path, "r") as f:
        ds = f["scalar_i32"]
        assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
        assert ds.shape == (), f"shape: {ds.shape}"
        assert int(ds[()]) == 42, f"value: {ds[()]}"
    print("PASS: write_scalar_i32")


def test_write_contiguous_f64(consus, tmp_dir: str) -> None:
    """consus-python writes float64 (4,) [1.5,2.5,3.5,4.5] → h5py reads same."""
    values = [1.5, 2.5, 3.5, 4.5]
    raw = struct.pack("<4d", *values)
    b = consus.FileBuilder()
    b.add_dataset("array_f64", "<f8", [4], raw)
    data = bytes(b.finish())

    path = os.path.join(tmp_dir, "contiguous_f64.h5")
    with open(path, "wb") as f:
        f.write(data)

    with h5py.File(path, "r") as f:
        ds = f["array_f64"]
        expected = np.array(values, dtype=np.float64)
        assert ds.dtype == np.dtype("float64"), f"dtype: {ds.dtype}"
        assert ds.shape == (4,), f"shape: {ds.shape}"
        assert np.array_equal(ds[:], expected), f"values: {ds[:]}"
    print("PASS: write_contiguous_f64")


def test_write_chunked_i32(consus, tmp_dir: str) -> None:
    """consus-python writes int32 chunked (12,) chunks=(4,) → h5py reads 0..11."""
    values = list(range(12))
    raw = struct.pack("<12i", *values)
    b = consus.FileBuilder()
    b.add_dataset("chunked_i32", "<i4", [12], raw,
                  layout="chunked", chunk_dims=[4])
    data = bytes(b.finish())

    path = os.path.join(tmp_dir, "chunked_i32.h5")
    with open(path, "wb") as f:
        f.write(data)

    with h5py.File(path, "r") as f:
        ds = f["chunked_i32"]
        expected = np.arange(12, dtype=np.int32)
        assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
        assert ds.shape == (12,), f"shape: {ds.shape}"
        assert np.array_equal(ds[:], expected), f"values: {ds[:]}"
    print("PASS: write_chunked_i32")


def test_write_deflate_chunked_f64(consus, tmp_dir: str) -> None:
    """consus-python writes deflate-compressed float64 (8,) → h5py reads same."""
    values = [float(i) for i in range(8)]
    raw = struct.pack("<8d", *values)
    b = consus.FileBuilder()
    b.add_dataset("deflate_f64", "<f8", [8], raw,
                  layout="chunked", chunk_dims=[4], compression="deflate:6")
    data = bytes(b.finish())

    path = os.path.join(tmp_dir, "deflate_f64.h5")
    with open(path, "wb") as f:
        f.write(data)

    with h5py.File(path, "r") as f:
        ds = f["deflate_f64"]
        expected = np.arange(8, dtype=np.float64)
        assert ds.dtype == np.dtype("float64"), f"dtype: {ds.dtype}"
        assert ds.shape == (8,), f"shape: {ds.shape}"
        assert np.array_equal(ds[:], expected), f"values: {ds[:]}"
    print("PASS: write_deflate_chunked_f64")


def test_write_2d_int32(consus, tmp_dir: str) -> None:
    """consus-python writes int32 (3,4) → h5py reads 0..11 row-major."""
    values = list(range(12))
    raw = struct.pack("<12i", *values)
    b = consus.FileBuilder()
    b.add_dataset("array_2d_i32", "<i4", [3, 4], raw)
    data = bytes(b.finish())

    path = os.path.join(tmp_dir, "array_2d.h5")
    with open(path, "wb") as f:
        f.write(data)

    with h5py.File(path, "r") as f:
        ds = f["array_2d_i32"]
        expected = np.arange(12, dtype=np.int32).reshape(3, 4)
        assert ds.dtype == np.dtype("int32"), f"dtype: {ds.dtype}"
        assert ds.shape == (3, 4), f"shape: {ds.shape}"
        assert np.array_equal(ds[:], expected), f"values mismatch"
    print("PASS: write_2d_int32")


# ---------------------------------------------------------------------------
# Direction 2: h5py writes → consus-python reads
# ---------------------------------------------------------------------------

def test_read_h5py_int32(consus, tmp_dir: str) -> None:
    """h5py writes int32 scalar=99 → consus-python reads same."""
    path = os.path.join(tmp_dir, "h5py_int32.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("val", data=np.int32(99))

    with open(path, "rb") as f:
        raw = f.read()

    hf = consus.open(raw)
    entries = hf.list_root_group()
    assert any(name == "val" for name, _, _ in entries), \
        f"'val' not found in root: {entries}"

    addr = hf.open_path("/val")
    info = hf.dataset_at(addr)
    assert info.dtype == "<int32", f"dtype: {info.dtype}"
    assert info.shape == [], f"shape: {info.shape}"

    data = hf.read_dataset(addr)
    value = struct.unpack("<i", bytes(data))[0]
    assert value == 99, f"value: {value}"
    print("PASS: read_h5py_int32")


def test_read_h5py_float64_array(consus, tmp_dir: str) -> None:
    """h5py writes float64 (4,) [1.5,2.5,3.5,4.5] → consus-python reads same."""
    path = os.path.join(tmp_dir, "h5py_f64.h5")
    arr = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    with h5py.File(path, "w") as f:
        f.create_dataset("arr", data=arr)

    with open(path, "rb") as f:
        raw = f.read()

    hf = consus.open(raw)
    addr = hf.open_path("/arr")
    info = hf.dataset_at(addr)
    assert info.shape == [4], f"shape: {info.shape}"

    data = hf.read_dataset(addr)
    values = struct.unpack("<4d", bytes(data))
    assert list(values) == [1.5, 2.5, 3.5, 4.5], f"values: {values}"
    print("PASS: read_h5py_float64_array")


def test_read_h5py_deflate_chunked(consus, tmp_dir: str) -> None:
    """h5py writes deflate-compressed int32 (8,) → consus-python reads and decompresses."""
    path = os.path.join(tmp_dir, "h5py_deflate.h5")
    data = np.arange(8, dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("dset", data=data, chunks=(4,), compression="gzip",
                         compression_opts=6)

    with open(path, "rb") as f:
        raw = f.read()

    hf = consus.open(raw)
    addr = hf.open_path("/dset")
    info = hf.dataset_at(addr)
    assert info.layout == "chunked", f"layout: {info.layout}"
    assert info.shape == [8], f"shape: {info.shape}"

    result = hf.read_dataset(addr)
    values = list(struct.unpack("<8i", bytes(result)))
    assert values == list(range(8)), f"values: {values}"
    print("PASS: read_h5py_deflate_chunked")


def test_read_h5py_group(consus, tmp_dir: str) -> None:
    """h5py writes nested group /grp/sub → consus-python navigates path and reads."""
    path = os.path.join(tmp_dir, "h5py_group.h5")
    with h5py.File(path, "w") as f:
        grp = f.create_group("grp")
        grp.create_dataset("sub", data=np.int32(55))

    with open(path, "rb") as f:
        raw = f.read()

    hf = consus.open(raw)
    addr = hf.open_path("/grp/sub")
    data = hf.read_dataset(addr)
    value = struct.unpack("<i", bytes(data))[0]
    assert value == 55, f"value: {value}"
    print("PASS: read_h5py_group")


def test_read_h5py_attributes(consus, tmp_dir: str) -> None:
    """h5py writes a dataset with int32 attribute → consus-python reads attribute."""
    path = os.path.join(tmp_dir, "h5py_attr.h5")
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("ds", data=np.int32(7))
        ds.attrs["answer"] = np.int32(99)

    with open(path, "rb") as f:
        raw = f.read()

    hf = consus.open(raw)
    addr = hf.open_path("/ds")
    attrs = hf.attributes_at(addr)
    names = [name for name, _, _ in attrs]
    assert "answer" in names, f"attribute 'answer' missing; found: {names}"

    for name, dtype, raw_bytes in attrs:
        if name == "answer":
            value = struct.unpack("<i", bytes(raw_bytes))[0]
            assert value == 99, f"attribute value: {value}"
    print("PASS: read_h5py_attributes")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end roundtrip tests for the consus-python extension."
    )
    parser.add_argument(
        "--pyd-dir",
        required=True,
        help="Directory containing consus.pyd / consus.so built by cargo.",
    )
    args = parser.parse_args()

    consus = load_consus(args.pyd_dir)

    failed: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tests = [
            test_write_scalar_i32,
            test_write_contiguous_f64,
            test_write_chunked_i32,
            test_write_deflate_chunked_f64,
            test_write_2d_int32,
            test_read_h5py_int32,
            test_read_h5py_float64_array,
            test_read_h5py_deflate_chunked,
            test_read_h5py_group,
            test_read_h5py_attributes,
        ]
        for test_fn in tests:
            try:
                test_fn(consus, tmp_dir)
            except Exception as exc:
                name = test_fn.__name__
                print(f"FAIL: {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed.append(name)

    if failed:
        print(f"\n{len(failed)} test(s) FAILED: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll {len(tests)} tests PASSED")


if __name__ == "__main__":
    main()
