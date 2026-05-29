"""compare_hdf5.py — HDF5 compatibility tests between h5py and consus.

Tests
-----
1.  h5py writes f64 scalar → consus reads (dtype, shape, value)
2.  h5py writes i32 1-D array → consus reads
3.  h5py writes f32 2-D array → consus reads
4.  consus writes f32 1-D → h5py reads (dtype, shape, values)
5.  consus writes i32 1-D → h5py reads
6.  consus writes chunked f64 → h5py reads
7.  consus writes gzip-compressed f32 → h5py reads
8.  h5py writes chunked gzip i32 → consus reads
9.  h5py writes multiple datasets → consus lists root group
10. consus roundtrip: write → read, verify byte-exact values
"""

import io
import struct
import sys
import tempfile

import h5py
import numpy as np

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


def _h5py_to_bytes(tmp: tempfile.SpooledTemporaryFile) -> bytes:
    tmp.seek(0)
    return tmp.read()


# ---------------------------------------------------------------------------
# Test 1: h5py writes f64 scalar → consus reads
# ---------------------------------------------------------------------------
def test_h5py_f64_scalar_to_consus() -> None:
    label = "h5py writes f64 scalar → consus reads"
    try:
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            f.create_dataset("x", data=np.float64(3.14))
        raw = buf.getvalue()

        hf = consus.Hdf5File.open_path(raw)
        info = hf.dataset_at("x")
        data = hf.read_dataset("x")

        shape_ok = info.shape == []
        dtype_ok = info.dtype == "<f8"
        val = struct.unpack("<d", data)[0]
        val_ok = abs(val - 3.14) < 1e-12
        check(label, shape_ok and dtype_ok and val_ok,
              f"shape={info.shape} dtype={info.dtype} val={val:.6f}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 2: h5py writes i32 1-D array → consus reads
# ---------------------------------------------------------------------------
def test_h5py_i32_1d_to_consus() -> None:
    label = "h5py writes i32 1-D → consus reads"
    try:
        arr = np.array([10, 20, 30, 40], dtype=np.int32)
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            f.create_dataset("arr", data=arr)
        raw = buf.getvalue()

        hf = consus.Hdf5File.open_path(raw)
        info = hf.dataset_at("arr")
        data = hf.read_dataset("arr")

        shape_ok = info.shape == [4]
        dtype_ok = "<i4" in info.dtype
        vals = list(struct.unpack("<4i", data))
        vals_ok = vals == [10, 20, 30, 40]
        check(label, shape_ok and dtype_ok and vals_ok,
              f"shape={info.shape} dtype={info.dtype} vals={vals}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 3: h5py writes f32 2-D array → consus reads
# ---------------------------------------------------------------------------
def test_h5py_f32_2d_to_consus() -> None:
    label = "h5py writes f32 2-D → consus reads"
    try:
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            f.create_dataset("mat", data=arr)
        raw = buf.getvalue()

        hf = consus.Hdf5File.open_path(raw)
        info = hf.dataset_at("mat")
        data = hf.read_dataset("mat")

        shape_ok = info.shape == [3, 2]
        dtype_ok = "<f4" in info.dtype
        vals = list(struct.unpack("<6f", data))
        expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        vals_ok = all(abs(a - b) < 1e-6 for a, b in zip(vals, expected))
        check(label, shape_ok and dtype_ok and vals_ok,
              f"shape={info.shape} dtype={info.dtype}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 4: consus writes f32 1-D → h5py reads
# ---------------------------------------------------------------------------
def test_consus_f32_1d_to_h5py() -> None:
    label = "consus writes f32 1-D → h5py reads"
    try:
        vals = [1.5, 2.5, 3.5, 4.5]
        raw_data = struct.pack("<4f", *vals)
        fb = consus.FileBuilder()
        fb.add_dataset("ds", "<f4", [4], raw_data)
        hdf5_bytes = fb.finish()

        buf = io.BytesIO(bytes(hdf5_bytes))
        with h5py.File(buf, "r") as f:
            arr = f["ds"][:]

        shape_ok = list(arr.shape) == [4]
        dtype_ok = arr.dtype == np.float32
        vals_ok = np.allclose(arr, vals)
        check(label, shape_ok and dtype_ok and vals_ok,
              f"shape={list(arr.shape)} dtype={arr.dtype}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 5: consus writes i32 2-D → h5py reads
# ---------------------------------------------------------------------------
def test_consus_i32_2d_to_h5py() -> None:
    label = "consus writes i32 2-D → h5py reads"
    try:
        vals = [1, 2, 3, 4, 5, 6]
        raw_data = struct.pack("<6i", *vals)
        fb = consus.FileBuilder()
        fb.add_dataset("m", "<i4", [2, 3], raw_data)
        hdf5_bytes = fb.finish()

        buf = io.BytesIO(bytes(hdf5_bytes))
        with h5py.File(buf, "r") as f:
            arr = f["m"][:]

        shape_ok = list(arr.shape) == [2, 3]
        dtype_ok = arr.dtype == np.int32
        vals_ok = arr.flatten().tolist() == vals
        check(label, shape_ok and dtype_ok and vals_ok,
              f"shape={list(arr.shape)} dtype={arr.dtype}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 6: consus writes chunked f64 → h5py reads
# ---------------------------------------------------------------------------
def test_consus_chunked_f64_to_h5py() -> None:
    label = "consus writes chunked f64 → h5py reads"
    try:
        n = 8
        vals = [float(i) * 0.5 for i in range(n)]
        raw_data = struct.pack(f"<{n}d", *vals)
        fb = consus.FileBuilder()
        fb.add_dataset("chunked", "<f8", [n], raw_data,
                       layout="chunked", chunk_dims=[4])
        hdf5_bytes = fb.finish()

        buf = io.BytesIO(bytes(hdf5_bytes))
        with h5py.File(buf, "r") as f:
            arr = f["chunked"][:]

        shape_ok = list(arr.shape) == [n]
        dtype_ok = arr.dtype == np.float64
        vals_ok = np.allclose(arr, vals)
        check(label, shape_ok and dtype_ok and vals_ok,
              f"shape={list(arr.shape)} chunks={arr.chunks if hasattr(arr, 'chunks') else 'n/a'}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 7: consus writes gzip-compressed f32 → h5py reads
# ---------------------------------------------------------------------------
def test_consus_gzip_f32_to_h5py() -> None:
    label = "consus writes gzip f32 → h5py reads"
    try:
        n = 16
        vals = [float(i) for i in range(n)]
        raw_data = struct.pack(f"<{n}f", *vals)
        fb = consus.FileBuilder()
        fb.add_dataset("compressed", "<f4", [n], raw_data,
                       layout="chunked", chunk_dims=[8], compression="gzip")
        hdf5_bytes = fb.finish()

        buf = io.BytesIO(bytes(hdf5_bytes))
        with h5py.File(buf, "r") as f:
            arr = f["compressed"][:]

        shape_ok = list(arr.shape) == [n]
        dtype_ok = arr.dtype == np.float32
        vals_ok = np.allclose(arr, vals)
        check(label, shape_ok and dtype_ok and vals_ok,
              f"shape={list(arr.shape)} dtype={arr.dtype}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 8: h5py writes chunked gzip i32 → consus reads
# ---------------------------------------------------------------------------
def test_h5py_chunked_gzip_to_consus() -> None:
    label = "h5py writes chunked gzip i32 → consus reads"
    try:
        arr = np.arange(20, dtype=np.int32)
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            f.create_dataset("data", data=arr, chunks=(10,), compression="gzip",
                             compression_opts=6)
        raw = buf.getvalue()

        hf = consus.Hdf5File.open_path(raw)
        info = hf.dataset_at("data")
        data = hf.read_dataset("data")

        shape_ok = info.shape == [20]
        dtype_ok = "<i4" in info.dtype
        vals = list(struct.unpack("<20i", data))
        vals_ok = vals == list(range(20))
        check(label, shape_ok and dtype_ok and vals_ok,
              f"shape={info.shape} dtype={info.dtype}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 9: h5py writes multiple datasets → consus lists root group
# ---------------------------------------------------------------------------
def test_h5py_multi_dataset_list() -> None:
    label = "h5py writes multiple datasets → consus lists all"
    try:
        buf = io.BytesIO()
        names = ["alpha", "beta", "gamma"]
        with h5py.File(buf, "w") as f:
            for i, name in enumerate(names):
                f.create_dataset(name, data=np.float32(i))
        raw = buf.getvalue()

        hf = consus.Hdf5File.open_path(raw)
        listed = sorted(hf.list_root_group())
        expected = sorted(names)
        check(label, listed == expected, f"listed={listed}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Test 10: consus roundtrip write → read, byte-exact values
# ---------------------------------------------------------------------------
def test_consus_roundtrip_byte_exact() -> None:
    label = "consus roundtrip write → read (byte-exact)"
    try:
        n = 12
        vals = [float(i) * 1.1 for i in range(n)]
        raw_data = struct.pack(f"<{n}d", *vals)
        fb = consus.FileBuilder()
        fb.add_dataset("rt", "<f8", [n], raw_data)
        hdf5_bytes = fb.finish()

        hf = consus.Hdf5File.open_path(bytes(hdf5_bytes))
        read_back = hf.read_dataset("rt")
        check(label, read_back == raw_data,
              f"n={n} bytes_match={read_back == raw_data}")
    except Exception as exc:
        check(label, False, str(exc))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("compare_hdf5.py \u2014 HDF5 compatibility tests\n")
    test_h5py_f64_scalar_to_consus()
    test_h5py_i32_1d_to_consus()
    test_h5py_f32_2d_to_consus()
    test_consus_f32_1d_to_h5py()
    test_consus_i32_2d_to_h5py()
    test_consus_chunked_f64_to_h5py()
    test_consus_gzip_f32_to_h5py()
    test_h5py_chunked_gzip_to_consus()
    test_h5py_multi_dataset_list()
    test_consus_roundtrip_byte_exact()

    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{passed}/{total} passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
