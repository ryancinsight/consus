"""concurrent_bindings.py -- one shared object per consus binding class, driven
from a thread pool released by a barrier. Results must equal the serial result
and no call may raise a borrow error. Runs on GIL and free-threaded builds."""

import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import consus

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
