"""
Generate a deterministic HDF5 string reference fixture for consus integration tests.

Run from the workspace root:
    python data/gen_hdf5_string_ref.py

Output: data/hdf5_string_ref_sample.h5

HDF5 type encoding
------------------
Fixed-length strings  → numpy dtype 'SN' → HDF5 H5T_STRING(STRSIZE=N, STRPAD=H5T_STR_NULLPAD,
                         CSET=H5T_CSET_ASCII). This is the canonical representation produced by
                         h5py when the data source is numpy bytes_.  Passing
                         h5py.string_dtype(length=N) with numpy 'SN' data fails at HDF5 level
                         because HDF5 has no built-in ASCII→UTF-8 array conversion path.
Variable-length strings → h5py.string_dtype() → HDF5 H5T_STRING(H5T_VARIABLE, CSET=H5T_CSET_UTF8).

Dataset manifest
----------------
/fixed_str_scalar     fixed-length string (len=8), scalar, value b"hello\\0\\0\\0"
/vlen_str_scalar      variable-length UTF-8 string, scalar,   value "world"
/fixed_str_1d         fixed-length string (len=4), shape=(3,), values ["foo\\0","bar\\0","baz\\0"]
/vlen_str_1d          variable-length UTF-8 string, shape=(3,), values ["abc","de","fghij"]
/grp_with_str_attr    group, attr label="test_label" (variable-length UTF-8)
/grp_with_str_attr/inner_ds  int32 scalar dataset, value 99
"""

import os

import h5py
import numpy as np

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hdf5_string_ref_sample.h5"
)

# Variable-length UTF-8: HDF5 H5T_STRING H5T_VARIABLE / H5T_CSET_UTF8.
VLEN_STR = h5py.string_dtype()

with h5py.File(OUT, "w") as f:
    # ── 1. /fixed_str_scalar ──────────────────────────────────────────────
    # numpy dtype='S8' → HDF5 H5T_STRING(STRSIZE=8, STRPAD=NULLPAD, CSET=ASCII).
    # Value: 5 printable bytes + 3 explicit null-padding bytes.
    f.create_dataset(
        "fixed_str_scalar",
        data=np.bytes_(b"hello\x00\x00\x00"),  # numpy scalar, dtype inferred as S8
    )

    # ── 2. /vlen_str_scalar ───────────────────────────────────────────────
    # HDF5 H5T_STRING(H5T_VARIABLE, CSET=UTF8), scalar.
    f.create_dataset(
        "vlen_str_scalar",
        data="world",
        dtype=VLEN_STR,
    )

    # ── 3. /fixed_str_1d ─────────────────────────────────────────────────
    # numpy dtype='S4' → HDF5 H5T_STRING(STRSIZE=4, STRPAD=NULLPAD, CSET=ASCII), shape=(3,).
    # Each element is exactly 4 bytes: 3 printable + 1 null.
    f.create_dataset(
        "fixed_str_1d",
        data=np.array([b"foo\x00", b"bar\x00", b"baz\x00"], dtype="S4"),
    )

    # ── 4. /vlen_str_1d ──────────────────────────────────────────────────
    # HDF5 H5T_STRING(H5T_VARIABLE, CSET=UTF8), shape=(3,), strings of varying length.
    f.create_dataset(
        "vlen_str_1d",
        data=np.array(["abc", "de", "fghij"], dtype=object),
        dtype=VLEN_STR,
    )

    # ── 5. /grp_with_str_attr ────────────────────────────────────────────
    # Group carrying a variable-length UTF-8 string attribute.
    grp = f.create_group("grp_with_str_attr")
    grp.attrs.create("label", data="test_label", dtype=VLEN_STR)

    # ── 6. /grp_with_str_attr/inner_ds ───────────────────────────────────
    # Scalar int32 dataset nested inside the group.
    grp.create_dataset("inner_ds", data=np.int32(99))

print(f"Written: {OUT}")
print()


# ── Self-verification + manifest ─────────────────────────────────────────
def _decode_fixed(v):
    """Strip null padding and decode ASCII bytes to str."""
    if isinstance(v, (bytes, np.bytes_)):
        return v.rstrip(b"\x00").decode("ascii", errors="replace")
    if isinstance(v, np.ndarray):
        return [_decode_fixed(x) for x in v.flat]
    return v


with h5py.File(OUT, "r") as f:
    print("=== Manifest ===")

    # /fixed_str_scalar
    ds = f["fixed_str_scalar"]
    raw = ds[()]
    decoded = _decode_fixed(raw)
    print(
        f"  /fixed_str_scalar             dtype={ds.dtype}  shape={ds.shape}  raw={raw!r}  decoded={decoded!r}"
    )
    assert decoded == "hello", f"fixed_str_scalar mismatch: {decoded!r}"

    # /vlen_str_scalar
    ds = f["vlen_str_scalar"]
    raw = ds[()]
    # h5py 3.x returns str for vlen UTF-8; h5py 2.x returns bytes
    val = raw.decode() if isinstance(raw, bytes) else raw
    print(
        f"  /vlen_str_scalar              dtype={ds.dtype}  shape={ds.shape}  value={val!r}"
    )
    assert val == "world", f"vlen_str_scalar mismatch: {val!r}"

    # /fixed_str_1d
    ds = f["fixed_str_1d"]
    raw = ds[()]
    decoded_list = _decode_fixed(raw)
    print(
        f"  /fixed_str_1d                 dtype={ds.dtype}  shape={ds.shape}  decoded={decoded_list}"
    )
    assert decoded_list == ["foo", "bar", "baz"], (
        f"fixed_str_1d mismatch: {decoded_list}"
    )

    # /vlen_str_1d
    ds = f["vlen_str_1d"]
    raw = ds[()]
    str_list = [x.decode() if isinstance(x, bytes) else x for x in raw]
    print(
        f"  /vlen_str_1d                  dtype={ds.dtype}  shape={ds.shape}  values={str_list}"
    )
    assert str_list == ["abc", "de", "fghij"], f"vlen_str_1d mismatch: {str_list}"

    # /grp_with_str_attr
    grp = f["grp_with_str_attr"]
    label_raw = grp.attrs["label"]
    label_str = label_raw.decode() if isinstance(label_raw, bytes) else label_raw
    print(f"  /grp_with_str_attr            attrs={{label: {label_str!r}}}")
    assert label_str == "test_label", f"label attr mismatch: {label_str!r}"

    # /grp_with_str_attr/inner_ds
    inner = grp["inner_ds"]
    val_i = int(inner[()])
    print(
        f"  /grp_with_str_attr/inner_ds   dtype={inner.dtype}  shape={inner.shape}  value={val_i}"
    )
    assert val_i == 99, f"inner_ds mismatch: {val_i}"

    print()
    print("Self-verification passed.")

file_size = os.path.getsize(OUT)
print(f"File size: {file_size} bytes ({file_size / 1024:.2f} KiB)")
