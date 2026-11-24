"""detach_bindings.py -- binding compute runs with the interpreter detached.

A helper thread blocks on an event; the main thread sets it and then repeats
one binding call until the helper has appended to a list or a deadline
passes. The GIL switch interval is set to 1000 s before the helper starts, so
the helper cannot force a GIL handoff and runs only when the main thread
releases the GIL. "The helper ran" therefore witnesses that the call detached,
and a call that holds the GIL never produces the witness, whatever the
scheduler does. Sound under three conditions:

* The deadline is far below the interval, so a forced handoff cannot occur
  inside the loop (`DEADLINE_SECONDS` < `SWITCH_INTERVAL_SECONDS`).
* Nothing in the loop body releases the GIL except the probed call. The list
  truth test, `time.monotonic()` and the comparison are C calls that never
  allow other threads to run; `thread.start()` and `go.set()` precede the loop
  and cannot run the helper's append, since the event is set only after the
  helper has blocked on it.
* Between iterations the main thread executes only bytecode, and no bytecode
  boundary hands over the GIL while the interval exceeds the test duration.

The probe does not depend on the helper waking before a short detached region
ends: when it is slow, the next iteration offers another detached region. A
call that detaches always eventually produces the witness; a call that holds
the GIL fails at the deadline with the call count. A free-threaded build has
no GIL to release, so every test here skips there."""

import os
import struct
import sys
import threading
import time
from typing import NamedTuple

import pytest

import consus

N = 2_000_000
SWITCH_INTERVAL_SECONDS = 1000.0
GZIP_CHUNKED = ("chunked", [N // 8], "gzip")

# Bound on the hosted `test` job in .github/workflows/ci.yml: timeout-minutes
# 30. The job's other work (build, nextest, wheel build) took about 225 s on
# macOS in runs 45268922903 and 45326622304; four times that, 900 s, is
# reserved for it, leaving 900 s for the nine bindings. Each binding gets
# 900 s / 9 = 100 s, split into a loop deadline and a helper join bound of
# 50 s each, so nine failing bindings still end the job inside its bound.
JOB_TIMEOUT_SECONDS = 30 * 60
RESERVED_SECONDS = 900.0
BINDING_COUNT = 9
PER_BINDING_SECONDS = (JOB_TIMEOUT_SECONDS - RESERVED_SECONDS) / BINDING_COUNT
DEADLINE_SECONDS = PER_BINDING_SECONDS / 2
JOIN_SECONDS = PER_BINDING_SECONDS / 2
assert DEADLINE_SECONDS + JOIN_SECONDS < SWITCH_INTERVAL_SECONDS

requires_gil = pytest.mark.skipif(
    not getattr(sys, "_is_gil_enabled", lambda: True)(),
    reason="a free-threaded interpreter has no GIL to release",
)


class Probe(NamedTuple):
    released: bool
    calls: int


def releases_interpreter(call, deadline=DEADLINE_SECONDS, join=JOIN_SECONDS):
    """Repeat `call` until the helper thread has run or `deadline` passes.

    The verdict is read before the `finally` block: joining the helper
    releases the GIL, so a later read would always see it run.
    """
    ran = []
    go = threading.Event()

    def helper():
        go.wait()
        ran.append(True)

    calls_made = 0
    released = False
    previous = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL_SECONDS)
    thread = threading.Thread(target=helper)
    try:
        thread.start()
        go.set()
        end = time.monotonic() + deadline
        while not ran and time.monotonic() < end:
            call()
            calls_made += 1
        released = bool(ran)
    finally:
        go.set()
        thread.join(join)
        sys.setswitchinterval(previous)
    assert not thread.is_alive(), f"helper thread outlived its {join} s join bound"
    return Probe(released, calls_made)


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


@requires_gil
def test_compute_releases_the_interpreter():
    attached = []
    for name, call in calls().items():
        probe = releases_interpreter(call)
        if not probe.released:
            attached.append(
                f"{name}: held the interpreter for {probe.calls} calls "
                f"over {DEADLINE_SECONDS} s"
            )
    assert attached == []


@requires_gil
def test_probe_accepts_a_call_that_releases_the_interpreter():
    reader, writer = os.pipe()
    try:

        def pipe_round_trip():
            os.write(writer, b"x")
            os.read(reader, 1)

        probe = releases_interpreter(pipe_round_trip)
    finally:
        os.close(reader)
        os.close(writer)
    assert probe.released
    assert probe.calls >= 1


@requires_gil
def test_probe_rejects_a_call_that_holds_the_interpreter():
    def count():
        total = 0
        for value in range(100_000):
            total += value
        return total

    deadline = 1.0
    probe = releases_interpreter(count, deadline=deadline, join=deadline)
    assert not probe.released
    assert probe.calls >= 1
