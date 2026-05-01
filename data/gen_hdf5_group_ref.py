"""
Generate a deterministic HDF5 group-navigation reference fixture for consus integration tests.

Run from the workspace root:
    python data/gen_hdf5_group_ref.py

Output: data/hdf5_group_ref_sample.h5

Dataset manifest
----------------
/flat_ds                        int32 scalar,   value 7
/grp_a                          group,  attr tag="grp_a" (variable-length string)
/grp_a/ds_one                   int32 scalar,   value 11
/grp_a/ds_two                   float64 1-D, shape=(2,), values [1.5, 2.5]
/grp_b                          group,  no attributes
/grp_b/nested                   group
/grp_b/nested/deep_ds           int32 scalar,   value 42
/grp_b/nested/deep_ds_float     float64 scalar, value 3.14
"""

import os

import h5py
import numpy as np

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hdf5_group_ref_sample.h5"
)

VLEN_STR = h5py.string_dtype()  # variable-length UTF-8

with h5py.File(OUT, "w") as f:
    # ── 1. /flat_ds ───────────────────────────────────────────────────────
    # Top-level int32 scalar dataset.
    f.create_dataset("flat_ds", data=np.int32(7))

    # ── 2. /grp_a ─────────────────────────────────────────────────────────
    # Group with a variable-length string attribute.
    grp_a = f.create_group("grp_a")
    grp_a.attrs.create("tag", data="grp_a", dtype=VLEN_STR)

    # ── 3. /grp_a/ds_one ──────────────────────────────────────────────────
    grp_a.create_dataset("ds_one", data=np.int32(11))

    # ── 4. /grp_a/ds_two ──────────────────────────────────────────────────
    grp_a.create_dataset(
        "ds_two",
        data=np.array([1.5, 2.5], dtype=np.float64),
    )

    # ── 5. /grp_b ─────────────────────────────────────────────────────────
    # Group with no attributes.
    grp_b = f.create_group("grp_b")

    # ── 6. /grp_b/nested ──────────────────────────────────────────────────
    nested = grp_b.create_group("nested")

    # ── 7. /grp_b/nested/deep_ds ──────────────────────────────────────────
    nested.create_dataset("deep_ds", data=np.int32(42))

    # ── 8. /grp_b/nested/deep_ds_float ───────────────────────────────────
    nested.create_dataset("deep_ds_float", data=np.float64(3.14))

print(f"Written: {OUT}")
print()

# ── Self-verification + manifest ─────────────────────────────────────────
with h5py.File(OUT, "r") as f:
    print("=== Manifest ===")

    # flat_ds
    ds = f["flat_ds"]
    val = int(ds[()])
    print(
        f"  /flat_ds                     dtype={ds.dtype}  shape={ds.shape}  value={val}"
    )
    assert val == 7, f"flat_ds mismatch: {val}"

    # grp_a attributes
    grp_a = f["grp_a"]
    tag = grp_a.attrs["tag"]
    tag_str = tag.decode() if isinstance(tag, bytes) else tag
    print(f"  /grp_a                       attrs={{tag: {tag_str!r}}}")
    assert tag_str == "grp_a", f"grp_a tag mismatch: {tag_str!r}"

    # grp_a/ds_one
    ds = grp_a["ds_one"]
    val = int(ds[()])
    print(
        f"  /grp_a/ds_one                dtype={ds.dtype}  shape={ds.shape}  value={val}"
    )
    assert val == 11, f"ds_one mismatch: {val}"

    # grp_a/ds_two
    ds = grp_a["ds_two"]
    vals = list(ds[()])
    print(
        f"  /grp_a/ds_two                dtype={ds.dtype}  shape={ds.shape}  values={vals}"
    )
    assert vals == [1.5, 2.5], f"ds_two mismatch: {vals}"

    # grp_b (no attributes)
    grp_b = f["grp_b"]
    print(f"  /grp_b                       attrs={dict(grp_b.attrs)}")
    assert len(grp_b.attrs) == 0, f"grp_b unexpected attrs: {dict(grp_b.attrs)}"

    # grp_b/nested (group exists)
    nested = grp_b["nested"]
    print(f"  /grp_b/nested                keys={list(nested.keys())}")

    # grp_b/nested/deep_ds
    ds = nested["deep_ds"]
    val = int(ds[()])
    print(
        f"  /grp_b/nested/deep_ds        dtype={ds.dtype}  shape={ds.shape}  value={val}"
    )
    assert val == 42, f"deep_ds mismatch: {val}"

    # grp_b/nested/deep_ds_float
    ds = nested["deep_ds_float"]
    val_f = float(ds[()])
    print(
        f"  /grp_b/nested/deep_ds_float  dtype={ds.dtype}  shape={ds.shape}  value={val_f}"
    )
    assert abs(val_f - 3.14) < 1e-12, f"deep_ds_float mismatch: {val_f}"

    print()
    print("Self-verification passed.")

file_size = os.path.getsize(OUT)
print(f"File size: {file_size} bytes ({file_size / 1024:.2f} KiB)")
