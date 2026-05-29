"""
Generate HDF5 test fixtures using h5py for the h5py→consus roundtrip tests.

Usage::

    D:\\miniforge3\\python.exe data/gen_h5py_compressed.py --case <name> --file <path>

Each ``--case`` generates one HDF5 file at ``--file`` containing specific
datasets that the Rust test then reads with consus and verifies.

| Case                        | Contents                                          |
|-----------------------------|---------------------------------------------------|
| deflate_1d_i32              | deflate-compressed int32 (8,) chunks=(4,)         |
| deflate_2d_f64              | deflate-compressed float64 (4,6) chunks=(2,3)     |
| deflate_3d_i32              | deflate-compressed int32 (2,3,4) chunks=(2,3,4)   |
| contiguous_u8               | contiguous uint8 (5,) [10,20,30,40,50]            |
| contiguous_f32              | contiguous float32 (4,) [1.5,2.5,3.5,4.5]        |
| contiguous_i16              | contiguous int16 (4,) [-100,0,100,200]            |
| chunked_1d_f32_deflate      | deflate-compressed float32 (8,) chunks=(4,)       |
| big_endian_i32              | big-endian int32 (4,) [100,200,300,400]           |
| multi_group                 | /grp_a/ds_a int32 scalar=10, /grp_b/ds_b f64=3.14|
| shuffle_deflate_i32         | shuffle+deflate int32 (16,) arange(16) chunks=(8,)|
| deflate_level_1             | deflate level 1 int32 (8,) arange(8) chunks=(4,) |
| deflate_level_9             | deflate level 9 int32 (8,) arange(8) chunks=(4,) |
"""

import argparse
import sys

try:
    import h5py
    import numpy as np
except ImportError as e:
    print(f"Dependency unavailable: {e}", file=sys.stderr)
    sys.exit(2)


def gen_deflate_1d_i32(path: str) -> None:
    with h5py.File(path, "w") as f:
        data = np.arange(8, dtype=np.int32)
        f.create_dataset("deflate_1d_i32", data=data, chunks=(4,), compression="gzip",
                         compression_opts=6)


def gen_deflate_2d_f64(path: str) -> None:
    with h5py.File(path, "w") as f:
        data = np.arange(24, dtype=np.float64).reshape(4, 6)
        f.create_dataset("deflate_2d_f64", data=data, chunks=(2, 3), compression="gzip",
                         compression_opts=6)


def gen_deflate_3d_i32(path: str) -> None:
    with h5py.File(path, "w") as f:
        data = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
        f.create_dataset("deflate_3d_i32", data=data, chunks=(2, 3, 4), compression="gzip",
                         compression_opts=6)


def gen_contiguous_u8(path: str) -> None:
    with h5py.File(path, "w") as f:
        data = np.array([10, 20, 30, 40, 50], dtype=np.uint8)
        f.create_dataset("contiguous_u8", data=data)


def gen_contiguous_f32(path: str) -> None:
    with h5py.File(path, "w") as f:
        data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        f.create_dataset("contiguous_f32", data=data)


def gen_contiguous_i16(path: str) -> None:
    with h5py.File(path, "w") as f:
        data = np.array([-100, 0, 100, 200], dtype=np.int16)
        f.create_dataset("contiguous_i16", data=data)


def gen_chunked_1d_f32_deflate(path: str) -> None:
    with h5py.File(path, "w") as f:
        data = np.arange(8, dtype=np.float32)
        f.create_dataset("chunked_1d_f32_deflate", data=data, chunks=(4,),
                         compression="gzip", compression_opts=6)


def gen_big_endian_i32(path: str) -> None:
    """Contiguous big-endian int32 dataset (4,) with values [100,200,300,400]."""
    with h5py.File(path, "w") as f:
        data = np.array([100, 200, 300, 400], dtype=np.dtype(">i4"))
        f.create_dataset("big_endian_i32", data=data, dtype=np.dtype(">i4"))


def gen_multi_group(path: str) -> None:
    """Two groups: /grp_a/ds_a (int32 scalar=10) and /grp_b/ds_b (float64 scalar=3.14)."""
    with h5py.File(path, "w") as f:
        grp_a = f.create_group("grp_a")
        grp_a.create_dataset("ds_a", data=np.int32(10))
        grp_b = f.create_group("grp_b")
        grp_b.create_dataset("ds_b", data=np.float64(3.14))


def gen_shuffle_deflate_i32(path: str) -> None:
    """int32 (16,) with shuffle+deflate filters, chunks=(8,), values arange(16)."""
    with h5py.File(path, "w") as f:
        data = np.arange(16, dtype=np.int32)
        f.create_dataset(
            "shuffle_deflate_i32",
            data=data,
            chunks=(8,),
            shuffle=True,
            compression="gzip",
            compression_opts=6,
        )


def gen_deflate_level_1(path: str) -> None:
    """Deflate compression level 1 (minimal) — int32 (8,) arange(8), chunks=(4,)."""
    with h5py.File(path, "w") as f:
        data = np.arange(8, dtype=np.int32)
        f.create_dataset("deflate_level_1", data=data, chunks=(4,),
                         compression="gzip", compression_opts=1)


def gen_deflate_level_9(path: str) -> None:
    """Deflate compression level 9 (maximum) — int32 (8,) arange(8), chunks=(4,)."""
    with h5py.File(path, "w") as f:
        data = np.arange(8, dtype=np.int32)
        f.create_dataset("deflate_level_9", data=data, chunks=(4,),
                         compression="gzip", compression_opts=9)


GENERATORS = {
    "deflate_1d_i32":         gen_deflate_1d_i32,
    "deflate_2d_f64":         gen_deflate_2d_f64,
    "deflate_3d_i32":         gen_deflate_3d_i32,
    "contiguous_u8":          gen_contiguous_u8,
    "contiguous_f32":         gen_contiguous_f32,
    "contiguous_i16":         gen_contiguous_i16,
    "chunked_1d_f32_deflate": gen_chunked_1d_f32_deflate,
    "big_endian_i32":         gen_big_endian_i32,
    "multi_group":            gen_multi_group,
    "shuffle_deflate_i32":    gen_shuffle_deflate_i32,
    "deflate_level_1":        gen_deflate_level_1,
    "deflate_level_9":        gen_deflate_level_9,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate h5py HDF5 fixtures for consus roundtrip tests."
    )
    parser.add_argument("--case", required=True,
                        help=f"Case name. One of: {sorted(GENERATORS)}")
    parser.add_argument("--file", required=True, help="Output HDF5 file path.")
    args = parser.parse_args()

    if args.case not in GENERATORS:
        print(f"Unknown case: {args.case!r}. Known: {sorted(GENERATORS)}", file=sys.stderr)
        sys.exit(1)

    try:
        GENERATORS[args.case](args.file)
    except Exception as exc:
        print(f"ERROR [{args.case}]: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {args.case} → {args.file}")


if __name__ == "__main__":
    main()
