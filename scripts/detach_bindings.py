"""detach_bindings.py -- binding compute runs with the interpreter detached.

A helper thread blocks on an event; the main thread sets it and then repeats
one binding call until the helper has appended to a list or a deadline
passes. The GIL switch interval is set to 1000 s before the helper starts, so
the helper cannot force a GIL handoff and runs only when the main thread
releases the GIL. "The helper ran" therefore witnesses that the call detached,
and a call that holds the GIL never produces the witness, whatever the
scheduler does. Sound under three conditions:

* The deadline plus the longest single call is below the interval
  (`DEADLINE_SECONDS + CALL_SECONDS < SWITCH_INTERVAL_SECONDS`). A call that
  holds the GIL never signals it, so the helper's one timed wait for the GIL
  starts at `go.set()` and is not cut short; it can request a drop only after
  the full interval passes without a switch. The loop checks the deadline
  between calls, so it ends at most one call after the deadline, and the
  verdict is read then, before the join.
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

# Time budget against the hosted `Test` job in .github/workflows/ci.yml
# (`timeout-minutes: 30`). Durations are job startedAt to completedAt.
# Reserve: the slowest `Test` job measured, Windows 450 s (job 138273720187
# of run 45343892223; the same runner took 384 s and 440 s in jobs
# 138023200694 and 138217660816). That job's duration already includes this
# file's passing run, so the reserve stands for all the job's other work. No
# margin factor is applied, so the bound holds only while that work stays at
# or below the measured figure.
# Call bound: the whole pytest invocation over eight scripts took 3.35 s in
# that job (`61 passed, 1 skipped in 3.35s`), which bounds any single call.
# Probes: the nine bindings plus the accepting stand-in run up to the full
# deadline and join bound when they fail; the rejecting stand-in runs its
# short deadline and join bound. The deadline is checked only between calls,
# so each probe overshoots by up to one call. The per-probe bound is the time
# left after the reserve, the rejecting stand-in and every overshoot, divided
# by the probe count and rounded down to whole seconds, split evenly into a
# loop deadline and a helper join bound.
JOB_TIMEOUT_SECONDS = 30 * 60
SLOWEST_TEST_JOB_SECONDS = 450.0
CALL_SECONDS = 3.35
BINDING_COUNT = 9
FULL_PROBES = BINDING_COUNT + 1
# The rejecting stand-in never produces the witness, so it always runs its
# whole deadline; any positive value is sound and one second keeps it short.
REJECT_DEADLINE_SECONDS = 1.0
PROBE_BOUND_SECONDS = (
    JOB_TIMEOUT_SECONDS
    - SLOWEST_TEST_JOB_SECONDS
    - 2 * REJECT_DEADLINE_SECONDS
    - (FULL_PROBES + 1) * CALL_SECONDS
) // FULL_PROBES
DEADLINE_SECONDS = PROBE_BOUND_SECONDS / 2
JOIN_SECONDS = PROBE_BOUND_SECONDS / 2
WORST_CASE_JOB_SECONDS = (
    SLOWEST_TEST_JOB_SECONDS
    + FULL_PROBES * (DEADLINE_SECONDS + CALL_SECONDS + JOIN_SECONDS)
    + 2 * REJECT_DEADLINE_SECONDS
    + CALL_SECONDS
)
assert WORST_CASE_JOB_SECONDS <= JOB_TIMEOUT_SECONDS
assert DEADLINE_SECONDS + CALL_SECONDS < SWITCH_INTERVAL_SECONDS

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
    try:
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
            if thread.ident is not None:
                thread.join(join)
    finally:
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
    bindings = calls()
    assert len(bindings) == BINDING_COUNT, "the time budget counts BINDING_COUNT probes"
    attached = []
    for name, call in bindings.items():
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

    probe = releases_interpreter(
        count, deadline=REJECT_DEADLINE_SECONDS, join=REJECT_DEADLINE_SECONDS
    )
    assert not probe.released
    assert probe.calls >= 1


class StartFailure(Exception):
    pass


def test_probe_restores_the_switch_interval_when_the_helper_cannot_start(monkeypatch):
    def refuse(self):
        raise StartFailure

    monkeypatch.setattr(threading.Thread, "start", refuse)
    previous = sys.getswitchinterval()
    # Exact in the whole microseconds the interpreter stores.
    expected = 0.0625
    sys.setswitchinterval(expected)
    try:
        with pytest.raises(StartFailure):
            releases_interpreter(lambda: None)
        restored = sys.getswitchinterval()
    finally:
        sys.setswitchinterval(previous)
    assert restored == expected
