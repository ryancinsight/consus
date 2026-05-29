"""compare_zarr.py — Cross-format compatibility test for Zarr v2.

Tests bidirectional compatibility between consus.ZarrArray and zarr-python 3.x.

Requirements:
    pip install zarr numpy  (zarr>=3.2, numpy>=1.24)
    maturin develop --release  (in crates/consus-python)
"""
from __future__ import annotations

import io
import sys

import numpy as np
import zarr
import zarr.storage

try:
    from zarr.core.buffer.cpu import Buffer as ZarrBuffer
except ImportError:
    # zarr-python 3.x alternative path
    from zarr.buffer import Buffer as ZarrBuffer  # type: ignore[no-redef]

import consus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_zarr_store(raw: dict[str, bytes]) -> dict[str, ZarrBuffer]:
    """Wrap raw bytes values in ZarrBuffer for MemoryStore consumption."""
    return {k: ZarrBuffer.from_bytes(v) for k, v in raw.items()}


def _from_zarr_store(store_dict: dict) -> dict[str, bytes]:
    """Extract raw bytes from a zarr MemoryStore's backing dict."""
    result: dict[str, bytes] = {}
    for k, v in store_dict.items():
        if isinstance(v, (bytes, bytearray)):
            result[k] = bytes(v)
        elif hasattr(v, "as_array_like"):
            result[k] = bytes(v.as_array_like())
        elif hasattr(v, "to_bytes"):
            result[k] = v.to_bytes()
        else:
            result[k] = bytes(v)
    return result


# ---------------------------------------------------------------------------
# Test 1: consus writes, zarr-python reads
# ---------------------------------------------------------------------------


def test_consus_to_zarr() -> None:
    """Write chunks with consus.ZarrArray, read with zarr-python."""
    rng = np.random.default_rng(42)
    shape = [50, 40]
    chunks = [10, 10]
    dtype = "<f8"
    original = rng.standard_normal(shape).astype(dtype)

    # Write all chunks with consus.
    arr = consus.PyZarrArray(shape, chunks, dtype)
    rows_per_chunk = chunks[0]
    cols_per_chunk = chunks[1]
    num_row_chunks = shape[0] // rows_per_chunk
    num_col_chunks = shape[1] // cols_per_chunk

    for ri in range(num_row_chunks):
        for ci in range(num_col_chunks):
            tile = original[
                ri * rows_per_chunk : (ri + 1) * rows_per_chunk,
                ci * cols_per_chunk : (ci + 1) * cols_per_chunk,
            ]
            arr.write_chunk([ri, ci], np.ascontiguousarray(tile).tobytes())

    # Export to raw store dict and hand to zarr-python.
    raw_store = arr.to_store()
    zarr_store_dict = _to_zarr_store(raw_store)
    ms = zarr.storage.MemoryStore(store_dict=zarr_store_dict)
    z = zarr.open_array(store=ms, mode="r", zarr_format=2)

    assert list(z.shape) == shape, f"shape mismatch: {z.shape} != {shape}"
    assert z.dtype == np.dtype(dtype), f"dtype mismatch: {z.dtype}"

    result = z[:]
    assert np.allclose(result, original), "value mismatch in consus→zarr direction"
    print("[PASS] test_consus_to_zarr: shape=%s, dtype=%s" % (shape, dtype))


# ---------------------------------------------------------------------------
# Test 2: zarr-python writes, consus reads
# ---------------------------------------------------------------------------


def test_zarr_to_consus() -> None:
    """Write an array with zarr-python, read chunks back with consus."""
    rng = np.random.default_rng(7)
    shape = [30, 20]
    chunks = [10, 10]
    dtype = "float64"
    original = rng.standard_normal(shape).astype(dtype)

    # Build zarr array with zarr-python.
    store_dict: dict = {}
    ms = zarr.storage.MemoryStore(store_dict=store_dict)
    z = zarr.open_array(
        store=ms,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        zarr_format=2,
    )
    z[:] = original

    # Import into consus.
    raw = _from_zarr_store(store_dict)
    c_arr = consus.PyZarrArray.from_store(raw)

    assert c_arr.shape == shape, f"shape mismatch: {c_arr.shape} != {shape}"

    # Read every chunk back and compare.
    rows_per_chunk = chunks[0]
    cols_per_chunk = chunks[1]
    num_row_chunks = shape[0] // rows_per_chunk
    num_col_chunks = shape[1] // cols_per_chunk

    for ri in range(num_row_chunks):
        for ci in range(num_col_chunks):
            chunk_bytes = c_arr.read_chunk([ri, ci])
            tile = np.frombuffer(chunk_bytes, dtype="<f8").reshape(
                rows_per_chunk, cols_per_chunk
            )
            expected = original[
                ri * rows_per_chunk : (ri + 1) * rows_per_chunk,
                ci * cols_per_chunk : (ci + 1) * cols_per_chunk,
            ]
            if not np.allclose(tile, expected):
                raise AssertionError(
                    f"value mismatch at chunk [{ri},{ci}]:\n{tile}\nvs\n{expected}"
                )

    print("[PASS] test_zarr_to_consus: shape=%s, dtype=%s" % (shape, dtype))


# ---------------------------------------------------------------------------
# Test 3: 1-D int32 array
# ---------------------------------------------------------------------------


def test_1d_int32() -> None:
    """1-D int32 round-trip through consus→zarr-python."""
    n = 100
    data = np.arange(n, dtype="<i4")
    arr = consus.PyZarrArray([n], [25], "<i4")
    for i in range(0, n, 25):
        arr.write_chunk([i // 25], data[i : i + 25].tobytes())

    raw_store = arr.to_store()
    zarr_store_dict = _to_zarr_store(raw_store)
    ms = zarr.storage.MemoryStore(store_dict=zarr_store_dict)
    z = zarr.open_array(store=ms, mode="r", zarr_format=2)

    result = z[:]
    assert np.array_equal(result, data), "value mismatch in 1-D int32 test"
    print("[PASS] test_1d_int32")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("zarr-python version:", zarr.__version__)
    print("consus module:", consus)

    failures: list[str] = []
    for fn in (test_consus_to_zarr, test_zarr_to_consus, test_1d_int32):
        try:
            fn()
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}", file=sys.stderr)
            failures.append(fn.__name__)

    if failures:
        print(f"\n{len(failures)} test(s) FAILED: {failures}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {3} zarr compatibility tests PASSED.")


if __name__ == "__main__":
    main()
