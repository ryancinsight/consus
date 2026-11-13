"""concurrent_bindings.py -- one shared object of every consus binding class,
driven from a thread pool.

Each test builds its inputs with consus itself, computes the expected result
serially, then has THREADS threads (released together by a barrier) use the
same object. Results must equal the serial result and no call may raise a
borrow error. On a free-threaded interpreter the threads run in parallel; on a
GIL interpreter they interleave at call boundaries, so one suite serves both.

Run with `python -m pytest -q scripts/concurrent_bindings.py`.
"""

import os
import struct
import subprocess
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor

import consus
import pytest

THREADS = 8
ROUNDS = 20


def run_pooled(work):
    """Run `work(index)` on THREADS threads started together; return the
    results in index order and re-raise the first worker exception."""
    barrier = threading.Barrier(THREADS)

    def start(index):
        barrier.wait()
        return work(index)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = [pool.submit(start, i) for i in range(THREADS)]
        return [f.result() for f in futures]


def check_shared(read):
    """`read()` returns the serial result on every call from every thread."""
    expected = read()

    def repeat(_index):
        return [read() for _ in range(ROUNDS)]

    for per_thread in run_pooled(repeat):
        assert per_thread == [expected] * ROUNDS
    return expected


def payload(index):
    return struct.pack("<4i", index, index + 1, index + 2, index + 3)


def names(prefix):
    return sorted(f"{prefix}{i}" for i in range(THREADS))


# ---------------------------------------------------------------------------
# HDF5 builder
# ---------------------------------------------------------------------------


def test_file_builder_concurrent_adds():
    builder = consus.FileBuilder()
    run_pooled(lambda i: builder.add_dataset(f"d{i}", "<i4", [4], payload(i)))
    reader = consus.Hdf5File.open_path(builder.finish())
    assert sorted(reader.list_root_group()) == names("d")
    for i in range(THREADS):
        assert reader.read_dataset(f"d{i}") == payload(i)


def test_file_builder_finish_consumes_once():
    builder = consus.FileBuilder()
    builder.add_dataset("a", "<i4", [4], payload(0))

    def finish(_index):
        try:
            return builder.finish()
        except RuntimeError as err:
            return str(err)

    results = run_pooled(finish)
    images = [r for r in results if isinstance(r, bytes)]
    errors = [r for r in results if isinstance(r, str)]
    assert len(images) == 1
    assert errors == ["FileBuilder already consumed by finish()"] * (THREADS - 1)
    assert consus.Hdf5File.open_path(images[0]).read_dataset("a") == payload(0)


# ---------------------------------------------------------------------------
# Zarr
# ---------------------------------------------------------------------------


def test_zarr_array_concurrent_chunk_io():
    array = consus.ZarrArray([THREADS, 4], [1, 4], "<i4")
    assert array.num_chunks() == THREADS

    def write_read(i):
        for _ in range(ROUNDS):
            array.write_chunk([i, 0], payload(i))
            assert array.read_chunk([i, 0]) == payload(i)
            assert ".zarray" in array.to_store()
        return array.chunk_key([i, 0])

    keys = run_pooled(write_read)
    assert keys == [f"{i}.0" for i in range(THREADS)]
    store = array.to_store()
    assert sorted(store) == sorted([".zarray", *keys])
    assert all(store[f"{i}.0"] == payload(i) for i in range(THREADS))
    assert (array.shape, array.chunks, array.dtype) == ([THREADS, 4], [1, 4], "<i4")


# ---------------------------------------------------------------------------
# Parquet builder
# ---------------------------------------------------------------------------


def test_parquet_builder_concurrent_schema_and_write():
    builder = consus.ParquetBuilder()
    run_pooled(lambda i: builder.add_column(f"c{i}", "INT64"))
    columns = builder.column_names()
    assert sorted(columns) == names("c")
    assert builder.num_columns() == THREADS

    data = {name: [1, 2, 3] for name in columns}
    expected = check_shared(lambda: builder.write(data))
    assert consus.ParquetFile(expected).column_names() == columns


# ---------------------------------------------------------------------------
# netCDF-4 writer
# ---------------------------------------------------------------------------


def test_netcdf_writer_concurrent_build():
    writer = consus.NetcdfWriter()

    def build(i):
        writer.add_dimension(f"x{i}", 3)
        writer.add_variable(f"v{i}", "f64", [f"x{i}"], [float(i), i + 0.5, i + 1.0])

    run_pooled(build)
    reader = consus.NetcdfFile(check_shared(writer.write))
    assert sorted(reader.variable_names()) == names("v")
    for i in range(THREADS):
        assert reader.read_variable(f"v{i}") == [float(i), i + 0.5, i + 1.0]


def test_netcdf_writer_duplicate_name_races_to_one_winner():
    writer = consus.NetcdfWriter()

    def add(_index):
        try:
            writer.add_dimension("same", 3)
        except ValueError as err:
            return str(err)
        return None

    results = run_pooled(add)
    assert results.count(None) == 1
    assert set(results) - {None} == {"dimension 'same' already defined"}


# ---------------------------------------------------------------------------
# Module declaration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="needs a free-threaded interpreter",
)
def test_import_keeps_gil_disabled():
    """A module declaring itself GIL-dependent re-enables the GIL for the whole
    process on import. `PYTHON_GIL` is removed so the declaration, not the
    environment, decides."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    out = subprocess.run(
        [sys.executable, "-c", "import consus, sys; print(sys._is_gil_enabled())"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False", out.stderr


# ---------------------------------------------------------------------------
# HDMF
# ---------------------------------------------------------------------------


def test_hdmf_builder_concurrent_columns():
    builder = consus.HdmfFileBuilder("trials", "trial table")
    run_pooled(
        lambda i: builder.add_column_f64(f"c{i}", "col", [float(i), i + 0.5, i + 1.0])
    )
    table = consus.read_dynamic_table_bytes(check_shared(lambda: bytes(builder.finish())))
    assert sorted(table.column_names()) == names("c")
    for i in range(THREADS):
        assert table.get_column_f64(f"c{i}") == [float(i), i + 0.5, i + 1.0]
    assert repr(builder).endswith(f"columns={THREADS})")


def test_dynamic_table_shared_reads():
    builder = consus.HdmfFileBuilder("trials", "trial table")
    builder.add_column_f64("rt", "reaction time", [0.3, 0.5, 0.4])
    builder.add_column_i64("n", "count", [4, 5, 6])
    table = consus.read_dynamic_table_bytes(bytes(builder.finish()))
    fields = check_shared(
        lambda: (
            table.name,
            table.description,
            table.colnames,
            table.id,
            table.num_rows,
            table.column_names(),
            table.get_column_f64("rt"),
            table.get_column_i64("n"),
            repr(table),
        )
    )
    assert fields[4:8] == (3, ["rt", "n"], [0.3, 0.5, 0.4], [4, 5, 6])


# ---------------------------------------------------------------------------
# Read-only classes
# ---------------------------------------------------------------------------


def hdf5_image():
    builder = consus.FileBuilder()
    builder.add_dataset("a", "<i4", [4], payload(0))
    builder.add_dataset("b", "<f8", [2], struct.pack("<2d", 1.5, -2.5))
    return builder.finish()


def test_hdf5_file_and_dataset_info_shared_reads():
    reader = consus.Hdf5File.open_path(hdf5_image())
    info = reader.dataset_at("a")
    result = check_shared(
        lambda: (
            reader.list_root_group(),
            reader.read_dataset("a"),
            reader.read_dataset("b"),
            reader.attributes_at("a"),
            (info.dtype, info.shape, info.layout, info.chunk_shape, info.filters),
        )
    )
    assert sorted(result[0]) == ["a", "b"]
    assert result[1:3] == (payload(0), struct.pack("<2d", 1.5, -2.5))
    assert result[4] == ("<i4", [4], "contiguous", None, [])


def test_parquet_file_shared_reads():
    builder = consus.ParquetBuilder()
    builder.add_column("id", "INT64")
    builder.add_column("x", "DOUBLE")
    reader = consus.ParquetFile(builder.write({"id": [1, 2, 3], "x": [0.5, 1.5, 2.5]}))
    result = check_shared(
        lambda: (
            reader.row_count(),
            reader.num_row_groups(),
            reader.column_names(),
            reader.read_column(0),
            reader.read_column(1),
        )
    )
    assert result == (3, 1, ["id", "x"], [1, 2, 3], [0.5, 1.5, 2.5])
    assert [name for name, _ in check_shared(reader.schema)] == ["id", "x"]


def test_netcdf_file_shared_reads():
    writer = consus.NetcdfWriter()
    writer.add_dimension("x", 3)
    writer.add_variable("v", "f64", ["x"], [1.0, 2.0, 3.0])
    reader = consus.NetcdfFile(writer.write())
    result = check_shared(
        lambda: (
            reader.variable_names(),
            reader.dimension_sizes(),
            reader.variable_shape("v"),
            reader.read_variable("v"),
        )
    )
    assert result == (["v"], [("x", 3)], [3], [1.0, 2.0, 3.0])


def mat_v4_image():
    """MAT v4 file holding a 1x3 and a column-major 2x2 double matrix."""

    def variable(name, rows, cols, values):
        encoded = name.encode() + b"\0"
        header = struct.pack("<5i", 0, rows, cols, 0, len(encoded))
        return header + encoded + struct.pack(f"<{rows * cols}d", *values)

    return variable("a", 1, 3, [1.0, 2.0, 3.0]) + variable(
        "m", 2, 2, [1.0, 3.0, 2.0, 4.0]
    )


def variable_fields(var):
    return (var.name, var.array_class, var.shape, var.numeric_class, var.read_data())


def test_mat_file_and_variable_shared_reads():
    mat = consus.loadmat_bytes(mat_v4_image())
    var = mat.get_variable("m")
    result = check_shared(
        lambda: (
            mat.version,
            mat.variable_names(),
            [variable_fields(v) for v in mat.variables()],
            variable_fields(var),
            mat.get_variable("absent"),
        )
    )
    assert result[:2] == ("v4", ["a", "m"])
    assert result[2][0][4] == struct.pack("<3d", 1.0, 2.0, 3.0)
    assert result[3][2:] == ([2, 2], result[3][3], struct.pack("<4d", 1.0, 3.0, 2.0, 4.0))
    assert result[4] is None
