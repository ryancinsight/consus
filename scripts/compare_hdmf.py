#!/usr/bin/env python3
"""compare_hdmf.py — Cross-validation between consus and hdmf Python 4.x.

Tests:
  1. Write with hdmf Python, read with consus (via child-group extraction)
  2. Write with consus, read with hdmf Python
  3. Roundtrip: consus write → consus read
  4. Multi-column table (f64, i64, u64, str)
  5. Empty table (no columns)
  6. Single-row table
  7. Ragged column (VectorIndex) — write with consus, verify structure
  8. Negative: open non-DynamicTable HDF5 raises ValueError
  9. Builder __repr__ and DynamicTable __repr__
 10. String column roundtrip (hdmf Python write → consus read)
"""

import io
import os
import sys
import tempfile
import traceback

import numpy as np
import h5py
import hdmf.common as hc
from hdmf.common import DynamicTable, VectorData
from hdmf.backends.hdf5 import HDF5IO

import consus

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _ok(name: str):
    print(f"  {PASS}  {name}")


def _fail(name: str, exc: Exception):
    print(f"  {FAIL}  {name}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# hdmf I/O helpers (hdmf 4.x requires a real file path)
# ---------------------------------------------------------------------------

def _hdmf_write(obj) -> bytes:
    """Serialise an hdmf object to a temp HDF5 file and return bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        with HDF5IO(tmp_path, manager=hc.get_manager(), mode="w") as io_obj:
            io_obj.write(obj)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def _hdmf_read(raw: bytes):
    """Read the top-level object from HDF5 bytes via a temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.write(raw)
    tmp_path = tmp.name
    tmp.close()
    try:
        with HDF5IO(tmp_path, manager=hc.get_manager(), mode="r") as io_obj:
            return io_obj.read()
    finally:
        os.unlink(tmp_path)


def _extract_group_bytes(raw: bytes, group_name: str) -> bytes:
    """Copy a named child group into a standalone HDF5 image for consus.

    Copies all attributes and direct children of `group_name` into the root
    of a fresh HDF5 image.
    """
    buf = io.BytesIO()
    with h5py.File(io.BytesIO(raw), "r") as src:
        grp = src[group_name]
        with h5py.File(buf, "w") as dst:
            # Copy group attributes to dst root
            for k, v in grp.attrs.items():
                dst.attrs[k] = v
            # Copy each child dataset/group to dst root
            for name in grp.keys():
                src.copy(grp[name], dst, name=name)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test 1 — hdmf Python write → consus read
# ---------------------------------------------------------------------------
def test_hdmf_write_consus_read():
    """Write a DynamicTable with hdmf Python, read back with consus."""
    table = DynamicTable(
        name="stimuli",
        description="hdmf-generated table",
        columns=[
            VectorData(name="freq", description="stimulus frequency",
                       data=np.array([100.0, 200.0, 300.0], dtype=np.float64)),
        ],
    )

    raw = _hdmf_write(table)

    # hdmf Python 4.x writes the DynamicTable at the HDF5 root group
    t = consus.read_dynamic_table_bytes(raw)

    assert t.description == "hdmf-generated table", f"description: {t.description!r}"
    assert "freq" in t.colnames, f"colnames={t.colnames}"
    freq = t.get_column_f64("freq")
    assert freq == [100.0, 200.0, 300.0], f"freq={freq}"
    _ok("hdmf Python write → consus read")


# ---------------------------------------------------------------------------
# Test 2 — consus write → hdmf Python read
# ---------------------------------------------------------------------------
def test_consus_write_hdmf_read():
    """Write a DynamicTable with consus, read back with hdmf Python.

    Data is accessed inside the HDF5IO context so that lazy-loaded h5py
    datasets remain accessible.
    """
    b = consus.HdmfFileBuilder("electrodes", "electrode metadata")
    b.add_column_f64("impedance", "impedance in Ohms", [1e6, 2e6, 3e6])
    b.add_column_str("location", "brain region", ["CA1", "CA3", "DG"])
    raw = bytes(b.finish())

    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.write(raw)
    tmp_path = tmp.name
    tmp.close()
    try:
        with HDF5IO(tmp_path, manager=hc.get_manager(), mode="r") as io_obj:
            table = io_obj.read()
            # Access data while the file is still open
            assert table.description == "electrode metadata", \
                f"description={table.description!r}"
            imp = list(table["impedance"].data[:])
            loc = list(table["location"].data[:])
    finally:
        os.unlink(tmp_path)

    assert imp == [1e6, 2e6, 3e6], f"impedance={imp}"
    # consus writes fixed-length HDF5 strings; hdmf returns them as bytes
    loc_str = [s.decode() if isinstance(s, bytes) else s for s in loc]
    assert loc_str == ["CA1", "CA3", "DG"], f"location={loc}"
    _ok("consus write → hdmf Python read")


# ---------------------------------------------------------------------------
# Test 3 — consus roundtrip (write → read)
# ---------------------------------------------------------------------------
def test_consus_roundtrip():
    """Write with consus and read back with consus; verify all fields."""
    b = consus.HdmfFileBuilder("units", "spike sorted units")
    b.add_column_f64("firing_rate", "mean firing rate (Hz)", [12.3, 4.5, 8.9])
    b.add_column_i64("spike_count", "total spikes", [1230, 450, 890])
    raw = bytes(b.finish())

    t = consus.read_dynamic_table_bytes(raw)
    assert t.description == "spike sorted units"
    assert t.colnames == ["firing_rate", "spike_count"], f"colnames={t.colnames}"
    assert t.id == [0, 1, 2], f"id={t.id}"
    fr = t.get_column_f64("firing_rate")
    for got, exp in zip(fr, [12.3, 4.5, 8.9]):
        assert abs(got - exp) < 1e-10, f"firing_rate mismatch: {got} vs {exp}"
    sc = t.get_column_i64("spike_count")
    assert sc == [1230, 450, 890], f"spike_count={sc}"
    _ok("consus roundtrip (write → read)")


# ---------------------------------------------------------------------------
# Test 4 — multi-column table
# ---------------------------------------------------------------------------
def test_multi_column():
    """Write all column types and read back."""
    b = consus.HdmfFileBuilder("mixed", "mixed types")
    b.add_column_f64("f", "floats", [1.1, 2.2])
    b.add_column_i64("i", "signed ints", [-10, 20])
    b.add_column_u64("u", "unsigned ints", [100, 200])
    b.add_column_str("s", "strings", ["hello", "world"])
    raw = bytes(b.finish())

    t = consus.read_dynamic_table_bytes(raw)
    assert t.colnames == ["f", "i", "u", "s"], f"colnames={t.colnames}"
    assert t.get_column_f64("f") == [1.1, 2.2]
    assert t.get_column_i64("i") == [-10, 20]
    # u64 columns also accessible via get_column_i64 (wrapping cast)
    assert t.get_column_i64("u") == [100, 200]
    assert t.get_column_str("s") == ["hello", "world"]
    _ok("multi-column table (f64, i64, u64, str)")


# ---------------------------------------------------------------------------
# Test 5 — empty table (no columns)
# ---------------------------------------------------------------------------
def test_empty_table():
    """Empty table: no columns, no rows."""
    b = consus.HdmfFileBuilder("empty_tbl", "no data here")
    raw = bytes(b.finish())

    t = consus.read_dynamic_table_bytes(raw)
    assert t.description == "no data here"
    assert t.colnames == [], f"colnames={t.colnames}"
    assert t.id == [], f"id={t.id}"
    assert t.column_names() == [], f"column_names={t.column_names()}"
    _ok("empty table")


# ---------------------------------------------------------------------------
# Test 6 — single-row table
# ---------------------------------------------------------------------------
def test_single_row():
    """Single-row table roundtrip."""
    b = consus.HdmfFileBuilder("one_row", "just one")
    b.add_column_f64("value", "a single value", [3.14159])
    raw = bytes(b.finish())

    t = consus.read_dynamic_table_bytes(raw)
    assert t.id == [0], f"id={t.id}"
    v = t.get_column_f64("value")
    assert abs(v[0] - 3.14159) < 1e-10, f"value={v}"
    _ok("single-row table")


# ---------------------------------------------------------------------------
# Test 7 — ragged column (VectorIndex): consus write, h5py + consus verify
# ---------------------------------------------------------------------------
def test_ragged_column_structure():
    """Write ragged column and verify VectorIndex dataset exists."""
    # 3 rows: [10, 20], [30], [40, 50, 60]
    flat = [10, 20, 30, 40, 50, 60]
    index = [2, 3, 6]  # cumulative end positions

    b = consus.HdmfFileBuilder("ragged_tbl", "table with ragged column")
    b.add_ragged_column_i64("events", "per-row event ids", flat, list(index))
    raw = bytes(b.finish())

    # Verify structure with h5py
    with h5py.File(io.BytesIO(raw), "r") as f:
        assert "events" in f, "events dataset missing"
        assert "events_index" in f, "events_index dataset missing"
        assert list(f["events"][:]) == flat, "events data mismatch"
        assert list(f["events_index"][:]) == index, "index mismatch"
        raw_dt = f["events_index"].attrs["data_type"]
        dt_str = raw_dt.decode() if isinstance(raw_dt, bytes) else str(raw_dt)
        assert dt_str == "VectorIndex", f"wrong data_type: {dt_str!r}"

    # Also read back via consus
    t = consus.read_dynamic_table_bytes(raw)
    assert t.get_column_i64("events") == flat
    assert t.get_column_index("events") == [2, 3, 6]
    _ok("ragged column structure (VectorIndex)")


# ---------------------------------------------------------------------------
# Test 8 — non-DynamicTable HDF5 raises ValueError
# ---------------------------------------------------------------------------
def test_non_dynamic_table_raises():
    """Opening a plain HDF5 file (no DynamicTable) raises ValueError."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("data", data=np.arange(10))
    raw = buf.getvalue()

    try:
        consus.read_dynamic_table_bytes(raw)
        assert False, "expected ValueError not raised"
    except ValueError:
        pass
    _ok("non-DynamicTable raises ValueError")


# ---------------------------------------------------------------------------
# Test 9 — repr
# ---------------------------------------------------------------------------
def test_repr():
    """__repr__ returns informative strings."""
    b = consus.HdmfFileBuilder("tbl", "test")
    b.add_column_f64("x", "x", [1.0])
    assert "HdmfFileBuilder" in repr(b), f"repr(b)={repr(b)!r}"

    raw = bytes(b.finish())
    t = consus.read_dynamic_table_bytes(raw)
    r = repr(t)
    assert "DynamicTable" in r, f"repr(t)={r!r}"
    assert "rows=1" in r, f"repr(t)={r!r}"
    _ok("__repr__ returns informative strings")


# ---------------------------------------------------------------------------
# Test 10 — hdmf Python string column write → consus read
# ---------------------------------------------------------------------------
def test_hdmf_str_column_consus_read():
    """String column written by hdmf Python is readable by consus."""
    table = DynamicTable(
        name="words",
        description="a word list",
        columns=[
            VectorData(name="word", description="a word",
                       data=["apple", "banana", "cherry"]),
        ],
    )
    raw = _hdmf_write(table)

    # hdmf Python 4.x writes the DynamicTable at the HDF5 root group
    t = consus.read_dynamic_table_bytes(raw)
    words = t.get_column_str("word")
    assert words == ["apple", "banana", "cherry"], f"words={words}"
    _ok("hdmf Python string VectorData → consus read")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
TESTS = [
    test_hdmf_write_consus_read,
    test_consus_write_hdmf_read,
    test_consus_roundtrip,
    test_multi_column,
    test_empty_table,
    test_single_row,
    test_ragged_column_structure,
    test_non_dynamic_table_raises,
    test_repr,
    test_hdmf_str_column_consus_read,
]


def main():
    print("consus vs hdmf Python comparison")
    print("=" * 50)
    failed = 0
    for fn in TESTS:
        try:
            fn()
        except Exception as exc:
            _fail(fn.__name__, exc)
            failed += 1
    print("=" * 50)
    total = len(TESTS)
    passed = total - failed
    print(f"{passed}/{total} tests passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
