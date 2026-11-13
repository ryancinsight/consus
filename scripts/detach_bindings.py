"""detach_bindings.py -- binding compute runs with the interpreter detached.

A helper thread waits on an event; the main thread sets it, makes one long
binding call and appends "main" after it, while the helper appends "helper"
when it runs. The GIL switch interval is set far beyond the test, so the
helper can run during the call only if the call released the GIL. A binding
that detaches yields ["helper", "main"]; one that holds the GIL yields
["main", "helper"], because the helper first runs when the main thread blocks
in `join`. The order is a fact about GIL ownership, not about call duration.
A free-threaded build has no GIL to release, so the test skips there."""

import struct
import sys
import threading

import pytest

import consus

N = 2_000_000
SWITCH_INTERVAL_SECONDS = 1000.0
GZIP_CHUNKED = ("chunked", [N // 8], "gzip")


def run_order(call):
    order = []
    go = threading.Event()

    def helper():
        go.wait()
        order.append("helper")

    thread = threading.Thread(target=helper)
    thread.start()
    previous = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL_SECONDS)
    try:
        go.set()
        call()
        order.append("main")
        thread.join()
    finally:
        sys.setswitchinterval(previous)
    return order


def calls():
    """One zero-argument call per detached binding path."""
    zeros = bytes(4 * N)
    hdf5 = consus.FileBuilder()
    hdf5.add_dataset("d", "<i4", [N], zeros, *GZIP_CHUNKED)
    h5 = consus.Hdf5File.open_path(hdf5.finish())
    netcdf = consus.NetcdfWriter()
    netcdf.add_dimension("x", N)
    netcdf.add_variable("v", "f64", ["x"], [0.0] * N)
    nc = consus.NetcdfFile(netcdf.write())
    hdmf = consus.HdmfFileBuilder("t", "table")
    hdmf.add_column_f64("c", "column", [0.0] * N)
    hdmf_image = bytes(hdmf.finish())
    parquet = consus.ParquetBuilder()
    parquet.add_column("c", "INT64")
    column = {"c": list(range(N))}
    pq = consus.ParquetFile(parquet.write(column))
    zarr = consus.ZarrArray([N], [N], "<i4", "gzip")
    zarr.write_chunk([0], zeros)
    mat_image = struct.pack("<5i", 0, N, 1, 0, 2) + b"a\0" + bytes(8 * N)
    return {
        "hdf5_read_dataset": lambda: h5.read_dataset("d"),
        "hdf5_add_dataset": lambda: consus.FileBuilder().add_dataset(
            "d", "<i4", [N], zeros, *GZIP_CHUNKED
        ),
        "mat_loadmat_bytes": lambda: consus.loadmat_bytes(mat_image),
        "netcdf_read_variable": lambda: nc.read_variable("v"),
        "hdmf_read_dynamic_table": lambda: consus.read_dynamic_table_bytes(hdmf_image),
        "parquet_write": lambda: parquet.write(column),
        "parquet_read_column": lambda: pq.read_column(0),
        "zarr_write_chunk": lambda: zarr.write_chunk([0], zeros),
        "zarr_read_chunk": lambda: zarr.read_chunk([0]),
    }


@pytest.mark.skipif(
    not getattr(sys, "_is_gil_enabled", lambda: True)(),
    reason="a free-threaded interpreter has no GIL to release",
)
def test_compute_releases_the_interpreter():
    attached = [
        name
        for name, call in calls().items()
        if run_order(call) != ["helper", "main"]
    ]
    assert attached == []
