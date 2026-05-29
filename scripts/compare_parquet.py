"""compare_parquet.py — Cross-format compatibility test for Apache Parquet.

Tests bidirectional compatibility between consus.ParquetBuilder/ParquetFile
and pyarrow.

Requirements:
    pip install pyarrow numpy  (pyarrow>=12, numpy>=1.24)
    maturin develop --release  (in crates/consus-python)
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import consus


# ---------------------------------------------------------------------------
# Test 1: consus writes, pyarrow reads
# ---------------------------------------------------------------------------


def test_consus_to_pyarrow() -> None:
    """Write a Parquet file with consus.ParquetBuilder, read with pyarrow."""
    n = 200
    ids = list(range(n))
    scores = [float(i) * 0.01 for i in range(n)]
    labels = [i % 2 == 0 for i in range(n)]

    b = consus.PyParquetBuilder()
    b.add_column("id", "INT32")
    b.add_column("score", "DOUBLE")
    b.add_column("flag", "BOOLEAN")
    data = b.write({"id": ids, "score": scores, "flag": labels})

    assert isinstance(data, bytes), f"expected bytes, got {type(data)}"

    table = pq.read_table(io.BytesIO(data))
    assert table.num_rows == n, f"row count mismatch: {table.num_rows} != {n}"
    assert list(table.schema.names) == ["id", "score", "flag"], \
        f"schema mismatch: {table.schema.names}"

    read_ids = table.column("id").to_pylist()
    read_scores = table.column("score").to_pylist()
    read_flags = table.column("flag").to_pylist()

    assert read_ids == ids, "id column mismatch"
    assert np.allclose(read_scores, scores), "score column mismatch"
    assert read_flags == labels, "flag column mismatch"

    print(
        f"[PASS] test_consus_to_pyarrow: {n} rows, "
        f"INT32+DOUBLE+BOOLEAN, Parquet len={len(data)} bytes"
    )


# ---------------------------------------------------------------------------
# Test 2: pyarrow writes, consus reads
# ---------------------------------------------------------------------------


def test_pyarrow_to_consus() -> None:
    """Write a Parquet file with pyarrow, read columns with consus."""
    rng = np.random.default_rng(99)
    n = 150
    ids = pa.array(range(n), type=pa.int32())
    weights = pa.array(rng.standard_normal(n).tolist(), type=pa.float64())
    flags = pa.array([bool(i % 3 == 0) for i in range(n)], type=pa.bool_())

    pa_table = pa.table({"id": ids, "weight": weights, "active": flags})
    buf = io.BytesIO()
    pq.write_table(pa_table, buf)
    raw = buf.getvalue()

    pf = consus.PyParquetFile(raw)

    assert pf.row_count() == n, f"row count mismatch: {pf.row_count()} != {n}"
    assert pf.num_row_groups() >= 1, "expected at least 1 row group"
    assert pf.column_names() == ["id", "weight", "active"], \
        f"column names mismatch: {pf.column_names()}"

    read_ids = pf.read_column(0)
    read_weights = pf.read_column(1)
    read_flags = pf.read_column(2)

    assert read_ids == list(range(n)), "id column mismatch"
    assert np.allclose(read_weights, weights.to_pylist()), "weight column mismatch"
    assert read_flags == [bool(i % 3 == 0) for i in range(n)], \
        "active column mismatch"

    print(
        f"[PASS] test_pyarrow_to_consus: {n} rows, "
        f"INT32+DOUBLE+BOOLEAN read back correctly"
    )


# ---------------------------------------------------------------------------
# Test 3: INT64 large values
# ---------------------------------------------------------------------------


def test_int64_large() -> None:
    """INT64 round-trip preserving values that exceed INT32 range."""
    n = 50
    vals = [2**40 + i for i in range(n)]  # values beyond INT32

    b = consus.PyParquetBuilder()
    b.add_column("big", "INT64")
    data = b.write({"big": vals})

    table = pq.read_table(io.BytesIO(data))
    result = table.column("big").to_pylist()
    assert result == vals, f"INT64 mismatch: {result[:5]} ..."
    print(f"[PASS] test_int64_large: {n} INT64 values, first={vals[0]}")


# ---------------------------------------------------------------------------
# Test 4: BYTE_ARRAY column
# ---------------------------------------------------------------------------


def test_byte_array_column() -> None:
    """BYTE_ARRAY (binary) column round-trip."""
    payloads = [bytes(range(i % 256, (i % 256) + 4)) for i in range(10)]

    b = consus.PyParquetBuilder()
    b.add_column("blob", "BYTE_ARRAY")
    data = b.write({"blob": payloads})

    pf = consus.PyParquetFile(data)
    result = pf.read_column(0)
    assert result == payloads, f"BYTE_ARRAY mismatch"
    print(f"[PASS] test_byte_array_column: {len(payloads)} binary values")


# ---------------------------------------------------------------------------
# Test 5: schema round-trip
# ---------------------------------------------------------------------------


def test_schema_report() -> None:
    """Schema reported by ParquetFile matches the builder's columns."""
    b = consus.PyParquetBuilder()
    for name, dtype in [("a", "INT32"), ("b", "DOUBLE"), ("c", "BOOLEAN")]:
        b.add_column(name, dtype)
    data = b.write({"a": [1], "b": [1.0], "c": [True]})

    pf = consus.PyParquetFile(data)
    schema = pf.schema()
    assert [s[0] for s in schema] == ["a", "b", "c"], f"names: {schema}"
    assert [s[1] for s in schema] == ["INT32", "DOUBLE", "BOOLEAN"], \
        f"types: {schema}"
    print(f"[PASS] test_schema_report: {schema}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("pyarrow version:", pa.__version__)
    print("consus module:", consus)

    tests = [
        test_consus_to_pyarrow,
        test_pyarrow_to_consus,
        test_int64_large,
        test_byte_array_column,
        test_schema_report,
    ]
    failures: list[str] = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"[FAIL] {fn.__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            failures.append(fn.__name__)

    if failures:
        print(f"\n{len(failures)} test(s) FAILED: {failures}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} Parquet compatibility tests PASSED.")


if __name__ == "__main__":
    main()
