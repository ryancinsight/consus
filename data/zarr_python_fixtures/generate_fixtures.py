#!/usr/bin/env python3
"""
Generate deterministic Python Zarr interoperability fixtures for Consus.

This script materializes small Zarr v2 and v3 stores on the local filesystem
using zarr-python and NumPy. The generated fixtures are intended for Rust-side
integration tests that validate:

- metadata parsing
- chunk key layout
- codec interoperability
- full-array reads
- partial selection reads

Fixture design constraints:
- deterministic values
- small repository footprint
- explicit dtype/chunk/codec coverage
- no hidden randomness
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from numcodecs import GZip

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "generated"


def reset_output_root() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def write_manifest(entries: list[dict[str, object]]) -> None:
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest = {
        "generator": {
            "script": "generate_fixtures.py",
            "zarr_version": getattr(zarr, "__version__", "unknown"),
            "numpy_version": np.__version__,
        },
        "fixtures": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def int32_grid(shape: tuple[int, ...]) -> np.ndarray:
    total = int(np.prod(shape, dtype=np.int64))
    return np.arange(total, dtype=np.int32).reshape(shape)


def float64_grid(shape: tuple[int, ...]) -> np.ndarray:
    total = int(np.prod(shape, dtype=np.int64))
    base = np.arange(total, dtype=np.float64).reshape(shape)
    return (base * 0.5) - 7.0


def create_v2_uncompressed_i4() -> dict[str, object]:
    fixture_name = "v2_uncompressed_i4"
    store_path = OUTPUT_ROOT / fixture_name
    data = int32_grid((4, 6))

    arr = zarr.open(
        store=str(store_path),
        mode="w",
        zarr_format=2,
        shape=data.shape,
        chunks=(2, 3),
        dtype="<i4",
        compressor=None,
        fill_value=-1,
        order="C",
    )
    arr[:] = data

    return {
        "name": fixture_name,
        "zarr_format": 2,
        "dtype": "<i4",
        "shape": list(data.shape),
        "chunks": [2, 3],
        "fill_value": -1,
        "codec": "none",
        "order": "C",
        "expected": {
            "full": data.reshape(-1).tolist(),
            "selection_rows_1_4_cols_2_6": data[1:4, 2:6].reshape(-1).tolist(),
            "selection_strided": data[::2, 1::2].reshape(-1).tolist(),
        },
    }


def create_v2_gzip_f8() -> dict[str, object]:
    fixture_name = "v2_gzip_f8"
    store_path = OUTPUT_ROOT / fixture_name
    data = float64_grid((5, 4))

    arr = zarr.open(
        store=str(store_path),
        mode="w",
        zarr_format=2,
        shape=data.shape,
        chunks=(2, 2),
        dtype="<f8",
        compressor=GZip(level=1),
        fill_value=0.0,
        order="C",
    )
    arr[:] = data

    return {
        "name": fixture_name,
        "zarr_format": 2,
        "dtype": "<f8",
        "shape": list(data.shape),
        "chunks": [2, 2],
        "fill_value": 0.0,
        "codec": "gzip",
        "codec_configuration": {"level": 1},
        "order": "C",
        "expected": {
            "full": data.reshape(-1).tolist(),
            "selection_rows_1_5_cols_0_4_step_2_2": data[1:5:2, 0:4:2]
            .reshape(-1)
            .tolist(),
        },
    }


def create_v3_uncompressed_i4() -> dict[str, object]:
    fixture_name = "v3_uncompressed_i4"
    store_path = OUTPUT_ROOT / fixture_name
    data = int32_grid((3, 5))

    arr = zarr.open(
        store=str(store_path),
        mode="w",
        zarr_format=3,
        shape=data.shape,
        chunks=(2, 2),
        dtype="int32",
        fill_value=0,
        order="C",
    )
    arr[:] = data

    return {
        "name": fixture_name,
        "zarr_format": 3,
        "dtype": "int32",
        "shape": list(data.shape),
        "chunks": [2, 2],
        "fill_value": 0,
        "codec": "bytes",
        "order": "C",
        "expected": {
            "full": data.reshape(-1).tolist(),
            "selection_rows_0_3_cols_1_5": data[:, 1:5].reshape(-1).tolist(),
            "selection_strided": data[::2, ::2].reshape(-1).tolist(),
        },
    }


def create_v3_gzip_f8() -> dict[str, object]:
    fixture_name = "v3_gzip_f8"
    store_path = OUTPUT_ROOT / fixture_name
    data = float64_grid((4, 4))

    arr = zarr.open(
        store=str(store_path),
        mode="w",
        zarr_format=3,
        shape=data.shape,
        chunks=(2, 2),
        dtype="float64",
        codecs=(
            zarr.codecs.BytesCodec(endian="little"),
            zarr.codecs.GzipCodec(level=1),
        ),
        fill_value=0.0,
        order="C",
    )
    arr[:] = data

    return {
        "name": fixture_name,
        "zarr_format": 3,
        "dtype": "float64",
        "shape": list(data.shape),
        "chunks": [2, 2],
        "fill_value": 0.0,
        "codec": "bytes+gzip",
        "codec_configuration": {
            "bytes": {"endian": "little"},
            "gzip": {"level": 1},
        },
        "order": "C",
        "expected": {
            "full": data.reshape(-1).tolist(),
            "selection_rows_1_4_cols_1_4": data[1:4, 1:4].reshape(-1).tolist(),
            "selection_strided": data[0:4:2, 0:4:2].reshape(-1).tolist(),
        },
    }


def main() -> None:
    reset_output_root()

    fixtures = [
        create_v2_uncompressed_i4(),
        create_v2_gzip_f8(),
        create_v3_uncompressed_i4(),
        create_v3_gzip_f8(),
    ]

    write_manifest(fixtures)

    print(f"Generated {len(fixtures)} fixtures in {OUTPUT_ROOT}")
    for fixture in fixtures:
        print(f"- {fixture['name']}")


if __name__ == "__main__":
    main()
