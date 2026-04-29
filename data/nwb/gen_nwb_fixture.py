"""
Generate a deterministic NWB 2.7 integration-test fixture using h5py.

Run once from the workspace root:
    python data/nwb/gen_nwb_fixture.py

Output: data/nwb/nwb_fixture_v2_7.nwb

Expected content (for value-semantic Rust assertions):
  root attrs:
    neurodata_type_def = "NWBFile"
    nwb_version        = "2.7.0"
    identifier         = "test_session_001"
    session_description = "Integration test fixture"
    session_start_time  = "2023-01-15T10:00:00.000000+00:00"
  acquisition/test_timeseries:
    data       = [0.1, 0.2, 0.3, 0.4, 0.5]  (f64)
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]  (f64)
  acquisition/test_timeseries_rate:
    data         = [100, 200, 300]            (i16 -> promotes to f64)
    starting_time = 5.0                       (f64 scalar, rate attr = 1000.0 f32)
  Units:
    spike_times       = [1.0, 1.5, 2.0, 2.5, 3.0]  flat f64
    spike_times_index = [2, 5]                       cumulative u64 (unit0=[1.0,1.5], unit1=[2.0,2.5,3.0])
    id                = [0, 1]                       u64
  electrodes (variable-length strings):
    id         = [0, 1, 2]              u64
    location   = ["CA1", "CA2", "CA3"] VL string
    group_name = ["tetrode1","tetrode2","tetrode3"] VL string
  general/subject (attributes):
    subject_id  = "mouse_001"
    species     = "Mus musculus"
    sex         = "M"
    age         = "P60D"
    description = "Test mouse"
"""

import os

import h5py
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nwb_fixture_v2_7.nwb")

with h5py.File(OUT, "w") as f:
    # ── Root NWBFile attributes — fixed-length strings ────────────────────
    # Use h5py.string_dtype(length=N) so consus-hdf5 maps these to
    # Datatype::FixedString, which decode_attribute_value handles correctly.
    # (Default h5py VL strings map to VariableString, which the attribute
    # decoder cannot resolve without file-source access.)
    def fstr(n):
        return h5py.string_dtype(length=n)

    f.attrs.create("neurodata_type_def", data="NWBFile", dtype=fstr(16))
    f.attrs.create("nwb_version", data="2.7.0", dtype=fstr(16))
    f.attrs.create("identifier", data="test_session_001", dtype=fstr(32))
    f.attrs.create(
        "session_description", data="Integration test fixture", dtype=fstr(64)
    )
    f.attrs.create(
        "session_start_time", data="2023-01-15T10:00:00.000000+00:00", dtype=fstr(64)
    )

    # ── acquisition/test_timeseries — f64 data + f64 timestamps ───────────
    acq = f.create_group("acquisition")

    ts1 = acq.create_group("test_timeseries")
    ts1.attrs.create("neurodata_type_def", data="TimeSeries", dtype=fstr(16))
    ts1.attrs.create("neurodata_type_inc", data="TimeSeries", dtype=fstr(16))
    ts1.create_dataset(
        "data", data=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    )
    ts1.create_dataset(
        "timestamps", data=np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    )

    # ── acquisition/test_timeseries_rate — i16 data + starting_time/rate ──
    ts2 = acq.create_group("test_timeseries_rate")
    ts2.attrs.create("neurodata_type_def", data="TimeSeries", dtype=fstr(16))
    ts2.attrs.create("neurodata_type_inc", data="TimeSeries", dtype=fstr(16))
    ts2.create_dataset("data", data=np.array([100, 200, 300], dtype=np.int16))
    st_ds = ts2.create_dataset("starting_time", data=np.float64(5.0))
    st_ds.attrs["rate"] = np.float32(1000.0)

    # ── Units (HDMF VectorData / VectorIndex) ─────────────────────────────
    units = f.create_group("Units")
    units.attrs.create("neurodata_type_def", data="Units", dtype=fstr(16))

    st = units.create_dataset(
        "spike_times",
        data=np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64),
    )
    st.attrs.create("neurodata_type_def", data="VectorData", dtype=fstr(16))
    st.attrs.create("description", data="spike times for all units", dtype=fstr(64))

    sti = units.create_dataset(
        "spike_times_index",
        data=np.array([2, 5], dtype=np.uint64),
    )
    sti.attrs.create("neurodata_type_def", data="VectorIndex", dtype=fstr(16))

    units.create_dataset("id", data=np.array([0, 1], dtype=np.uint64))

    # ── electrodes — variable-length string columns ────────────────────────
    elec = f.create_group("electrodes")
    elec.attrs.create("neurodata_type_def", data="DynamicTable", dtype=fstr(16))
    elec.attrs.create("description", data="Electrode metadata", dtype=fstr(64))

    elec.create_dataset("id", data=np.array([0, 1, 2], dtype=np.uint64))

    vlen_str = h5py.string_dtype()  # variable-length UTF-8
    elec.create_dataset(
        "location",
        data=np.array(["CA1", "CA2", "CA3"], dtype=object),
        dtype=vlen_str,
    )
    elec.create_dataset(
        "group_name",
        data=np.array(["tetrode1", "tetrode2", "tetrode3"], dtype=object),
        dtype=vlen_str,
    )

    # ── general/subject — fixed-length string scalar attributes ─────────
    general = f.create_group("general")
    subj = general.create_group("subject")
    subj.attrs.create("subject_id", data="mouse_001", dtype=fstr(32))
    subj.attrs.create("species", data="Mus musculus", dtype=fstr(32))
    subj.attrs.create("sex", data="M", dtype=fstr(8))
    subj.attrs.create("age", data="P60D", dtype=fstr(16))
    subj.attrs.create("description", data="Test mouse", dtype=fstr(64))

print("Written:", OUT)

# Quick self-verification
with h5py.File(OUT, "r") as f:
    ident = f.attrs["identifier"]
    if hasattr(ident, "decode"):
        ident = ident.decode()
    assert ident == "test_session_001", f"identifier mismatch: {ident!r}"
    locs = [
        x.decode() if isinstance(x, bytes) else x for x in f["electrodes/location"][()]
    ]
    assert locs == ["CA1", "CA2", "CA3"], f"location mismatch: {locs}"
    assert list(f["Units/spike_times"][()]) == [1.0, 1.5, 2.0, 2.5, 3.0]
    print("Self-verification passed.")
