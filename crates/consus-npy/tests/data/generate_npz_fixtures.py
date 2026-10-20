"""Regenerate the NPZ interoperability fixtures read by `src/npz.rs`.

Run from this directory with NumPy installed:

    python generate_npz_fixtures.py

`savez` writes stored members and `savez_compressed` writes DEFLATE members,
so the two archives cover both ZIP methods NumPy produces.
"""

import numpy as np

a = np.arange(6, dtype="<i8").reshape(2, 3)
b = np.array([1.5, -2.0], dtype="<f8")
np.savez("numpy_savez.npz", a=a, b=b)
np.savez_compressed("numpy_savez_compressed.npz", a=a, b=b)
