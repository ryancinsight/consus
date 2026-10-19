"""Generate the actual chunked variable-string regression fixture with h5py.

The committed fixture uses h5py 3.16.0 / HDF5 2.0.0 and the earliest format.
Each string dataset spans two complete chunks of two heap references. UTF-8,
empty strings, unequal payload lengths, compression and fixed-width dispatch
exercise distinct parts of the raw-read contract.
"""

from pathlib import Path

import h5py
import numpy as np


def main():
    path = Path(__file__).with_name("hdf5_chunked_vlen_sample.h5")
    values = np.array(
        ["α", "", "longer with spaces", "終"], dtype=h5py.string_dtype("utf-8")
    )
    with h5py.File(path, "w", libver="earliest", track_order=False) as file:
        file.create_dataset("plain", data=values, chunks=(2,))
        file.create_dataset(
            "deflate", data=values, chunks=(2,), compression="gzip", compression_opts=6
        )
        file.create_dataset("fixed", data=np.array([17, -23, 42, 99], dtype="<i4"), chunks=(2,))


if __name__ == "__main__":
    main()
