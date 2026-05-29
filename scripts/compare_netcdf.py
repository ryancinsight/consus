"""compare_netcdf.py — Cross-format compatibility test for netCDF-4.

Tests compatibility between consus netCDF-4 bindings and the reference
netCDF4-python library.

## Scope

Reading direction (netCDF4 → consus):
  All types, including 2-D layouts and schema introspection, are validated.

Writing direction (consus → external readers):
  The consus HDF5 writer encodes DIMENSION_LIST as a fixed array of object
  references.  Fully compliant netCDF4 DIMENSION_LIST requires a variable-
  length (VL) array of VL arrays per the HDF5 dimension-scale spec; adding
  VL-type support to the consus HDF5 writer is tracked as a future work item
  (see backlog.md).  The consus → consus internal roundtrip is verified here,
  and consus-written files are validated via h5py (which accepts the fixed-
  reference encoding).

Requirements:
    pip install netCDF4 h5py numpy  (netCDF4>=1.6, h5py>=3.0)
    maturin develop --release  (in crates/consus-python)
"""
from __future__ import annotations

import os
import sys
import tempfile

import h5py
import netCDF4
import numpy as np

import consus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\u2713"
FAIL = "\u2717"


def _run_test(name: str, fn) -> bool:
    try:
        fn()
        print(f"  {PASS} {name}")
        return True
    except AssertionError as exc:
        print(f"  {FAIL} {name}: {exc}")
        return False
    except Exception as exc:
        print(f"  {FAIL} {name}: {type(exc).__name__}: {exc}")
        return False


def _nc4_write_to_bytes(add_content) -> bytes:
    """Write a netCDF4 dataset via a temp file and return raw bytes."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tf:
        path = tf.name
    try:
        ds = netCDF4.Dataset(path, "w", format="NETCDF4")
        add_content(ds)
        ds.close()
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _close_enough(a: list, b: list, rel: float = 1e-5) -> bool:
    """Element-wise comparison with a relative tolerance."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == 0 and y == 0:
            continue
        denom = max(abs(x), abs(y), 1e-300)
        if abs(x - y) / denom > rel:
            return False
    return True


# ---------------------------------------------------------------------------
# Test 1 — netCDF4 writes f64, consus reads
# ---------------------------------------------------------------------------


def test_nc4_write_consus_read_f64() -> None:
    """Write a 1-D f64 variable via netCDF4, read back with consus."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0]

    def add_content(ds: netCDF4.Dataset) -> None:
        ds.createDimension("x", len(data))
        v = ds.createVariable("signal", "f8", ("x",))
        v[:] = np.array(data, dtype="f8")

    raw = _nc4_write_to_bytes(add_content)

    nf = consus.NetcdfFile(raw)
    vals = nf.read_variable("signal")
    assert _close_enough(vals, data, rel=1e-12), f"f64 readback mismatch: {vals} != {data}"


# ---------------------------------------------------------------------------
# Test 2 — netCDF4 writes i32, consus reads
# ---------------------------------------------------------------------------


def test_nc4_write_consus_read_i32() -> None:
    """Write a 1-D i32 variable via netCDF4, read back with consus."""
    data = [100, 200, 300, -100, 0]

    def add_content(ds: netCDF4.Dataset) -> None:
        ds.createDimension("x", len(data))
        v = ds.createVariable("index", "i4", ("x",))
        v[:] = np.array(data, dtype="i4")

    raw = _nc4_write_to_bytes(add_content)

    nf = consus.NetcdfFile(raw)
    vals = nf.read_variable("index")
    assert vals == data, f"i32 readback mismatch: {vals} != {data}"


# ---------------------------------------------------------------------------
# Test 3 — netCDF4 writes f32, consus reads
# ---------------------------------------------------------------------------


def test_nc4_write_consus_read_f32() -> None:
    """Write a 1-D f32 variable via netCDF4, read back with consus."""
    data = [1.0, 2.5, 3.75, -4.0, 0.0]

    def add_content(ds: netCDF4.Dataset) -> None:
        ds.createDimension("x", len(data))
        v = ds.createVariable("temperature", "f4", ("x",))
        v[:] = np.array(data, dtype="f4")

    raw = _nc4_write_to_bytes(add_content)

    nf = consus.NetcdfFile(raw)
    vals = nf.read_variable("temperature")
    assert _close_enough(vals, data), f"f32 readback mismatch: {vals} != {data}"


# ---------------------------------------------------------------------------
# Test 4 — netCDF4 writes 2-D f32, consus reads
# ---------------------------------------------------------------------------


def test_nc4_write_consus_read_2d() -> None:
    """Write a 2-D f32 variable via netCDF4, read back with consus."""
    rows, cols = 4, 5
    arr = np.arange(rows * cols, dtype="f4").reshape(rows, cols)
    expected = arr.flatten().tolist()

    def add_content(ds: netCDF4.Dataset) -> None:
        ds.createDimension("row", rows)
        ds.createDimension("col", cols)
        v = ds.createVariable("matrix", "f4", ("row", "col"))
        v[:] = arr

    raw = _nc4_write_to_bytes(add_content)

    nf = consus.NetcdfFile(raw)
    vals = nf.read_variable("matrix")
    assert _close_enough(vals, expected), f"2-D readback mismatch: {vals}"


# ---------------------------------------------------------------------------
# Test 5 — netCDF4 writes 2-D f64, consus reads
# ---------------------------------------------------------------------------


def test_nc4_write_consus_read_2d_f64() -> None:
    """Write a 2-D f64 variable via netCDF4, read back with consus."""
    rows, cols = 3, 4
    data = [float(r * cols + c) for r in range(rows) for c in range(cols)]

    def add_content(ds: netCDF4.Dataset) -> None:
        ds.createDimension("row", rows)
        ds.createDimension("col", cols)
        v = ds.createVariable("grid", "f8", ("row", "col"))
        v[:] = np.array(data, dtype="f8").reshape(rows, cols)

    raw = _nc4_write_to_bytes(add_content)

    nf = consus.NetcdfFile(raw)
    vals = nf.read_variable("grid")
    assert _close_enough(vals, data, rel=1e-12), f"2-D f64 readback mismatch"


# ---------------------------------------------------------------------------
# Test 6 — consus internal roundtrip: write then read (f32)
# ---------------------------------------------------------------------------


def test_consus_roundtrip_f32() -> None:
    """Write a 1-D f32 variable with consus, read back with consus."""
    data = [1.0, 2.5, 3.75, -4.0, 0.0]
    w = consus.NetcdfWriter()
    w.add_dimension("x", len(data))
    w.add_variable("temperature", "f32", ["x"], data)
    raw = w.write()

    nf = consus.NetcdfFile(raw)
    vals = nf.read_variable("temperature")
    assert _close_enough(vals, data), f"f32 roundtrip mismatch: {vals} != {data}"


# ---------------------------------------------------------------------------
# Test 7 — consus internal roundtrip: 2-D f64
# ---------------------------------------------------------------------------


def test_consus_roundtrip_2d_f64() -> None:
    """Write a 2-D f64 variable with consus, read back with consus."""
    rows, cols = 3, 4
    data = [float(r * cols + c) for r in range(rows) for c in range(cols)]
    w = consus.NetcdfWriter()
    w.add_dimension("row", rows)
    w.add_dimension("col", cols)
    w.add_variable("grid", "f64", ["row", "col"], data)
    raw = w.write()

    nf = consus.NetcdfFile(raw)
    vals = nf.read_variable("grid")
    assert _close_enough(vals, data, rel=1e-12), f"2-D f64 roundtrip mismatch"


# ---------------------------------------------------------------------------
# Test 8 — schema: variable_names / dimension_names / variable_info
# ---------------------------------------------------------------------------


def test_schema_report() -> None:
    """Verify variable_names, dimension_names, and variable_info via consus."""
    w = consus.NetcdfWriter()
    w.add_dimension("time", 10)
    w.add_dimension("depth", 5)
    w.add_variable("pressure", "f32", ["time", "depth"], [0.0] * 50)
    w.add_variable("temperature", "f64", ["time"], [0.0] * 10)
    raw = w.write()

    nf = consus.NetcdfFile(raw)
    assert set(nf.variable_names()) == {"pressure", "temperature"}, (
        f"variable_names: {nf.variable_names()}"
    )
    assert set(nf.dimension_names()) == {"time", "depth"}, (
        f"dimension_names: {nf.dimension_names()}"
    )

    dtype_p, dims_p = nf.variable_info("pressure")
    assert dtype_p == "f32", f"pressure dtype: {dtype_p}"
    assert dims_p == ["time", "depth"], f"pressure dims: {dims_p}"

    dtype_t, dims_t = nf.variable_info("temperature")
    assert dtype_t == "f64", f"temperature dtype: {dtype_t}"
    assert dims_t == ["time"], f"temperature dims: {dims_t}"


# ---------------------------------------------------------------------------
# Test 9 — consus writes, h5py validates structure
# ---------------------------------------------------------------------------


def test_consus_write_h5py_validates() -> None:
    """Write with consus, verify HDF5 structure is valid via h5py."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    w = consus.NetcdfWriter()
    w.add_dimension("x", len(data))
    w.add_variable("values", "f64", ["x"], data)
    raw = w.write()

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tf:
        tf.write(raw)
        path = tf.name
    try:
        with h5py.File(path, "r") as f:
            # Root-level _nc_properties marks the file as netCDF-4.
            assert "_nc_properties" in f.attrs, "missing _nc_properties root attr"
            assert "values" in f, "variable 'values' not found"
            assert "x" in f, "dimension scale 'x' not found"

            ds = f["values"]
            assert ds.dtype == np.float64, f"unexpected dtype: {ds.dtype}"
            assert ds.shape == (5,), f"unexpected shape: {ds.shape}"
            assert np.allclose(ds[:], data), f"data mismatch: {ds[:]}"

            dim_scale = f["x"]
            assert dim_scale.attrs.get("CLASS") == b"DIMENSION_SCALE", (
                "missing CLASS=DIMENSION_SCALE on 'x'"
            )

            # DIMENSION_LIST references the 'x' dimension scale.
            dl = ds.attrs["DIMENSION_LIST"]
            ref = dl[0] if dl.ndim == 1 else dl[0][0]
            target = f[ref]
            assert target.name == "/x", f"DIMENSION_LIST ref → {target.name}, expected /x"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test 10 — netCDF4 writes multi-variable file, consus reads both
# ---------------------------------------------------------------------------


def test_nc4_multi_variable() -> None:
    """Write multiple variables via netCDF4, read each with consus."""
    temps = [20.0, 21.5, 22.0, 19.5]
    pressures = [1013, 1014, 1012, 1015]

    def add_content(ds: netCDF4.Dataset) -> None:
        ds.createDimension("time", len(temps))
        t = ds.createVariable("temperature", "f8", ("time",))
        p = ds.createVariable("pressure", "i4", ("time",))
        t[:] = np.array(temps)
        p[:] = np.array(pressures, dtype="i4")

    raw = _nc4_write_to_bytes(add_content)
    nf = consus.NetcdfFile(raw)

    assert set(nf.variable_names()) == {"temperature", "pressure"}, (
        f"variable_names: {nf.variable_names()}"
    )
    t_vals = nf.read_variable("temperature")
    assert _close_enough(t_vals, temps, rel=1e-12), f"temperature mismatch: {t_vals}"
    p_vals = nf.read_variable("pressure")
    assert p_vals == pressures, f"pressure mismatch: {p_vals}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("netCDF4 writes f64 → consus reads", test_nc4_write_consus_read_f64),
    ("netCDF4 writes i32 → consus reads", test_nc4_write_consus_read_i32),
    ("netCDF4 writes f32 → consus reads", test_nc4_write_consus_read_f32),
    ("netCDF4 writes 2-D f32 → consus reads", test_nc4_write_consus_read_2d),
    ("netCDF4 writes 2-D f64 → consus reads", test_nc4_write_consus_read_2d_f64),
    ("consus roundtrip f32 (write → read)", test_consus_roundtrip_f32),
    ("consus roundtrip 2-D f64 (write → read)", test_consus_roundtrip_2d_f64),
    ("schema: variable_names / dimension_names / variable_info", test_schema_report),
    ("consus writes → h5py validates structure", test_consus_write_h5py_validates),
    ("netCDF4 writes multi-variable → consus reads all", test_nc4_multi_variable),
]

if __name__ == "__main__":
    print("compare_netcdf.py — netCDF-4 compatibility tests\n")
    passed = 0
    for name, fn in TESTS:
        if _run_test(name, fn):
            passed += 1
    total = len(TESTS)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
