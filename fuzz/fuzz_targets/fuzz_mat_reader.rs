//! Fuzz target: MATLAB `.mat` file reader (heap-buffer and logic).
//!
//! ## Strategy
//!
//! Drive `loadmat_bytes` with adversarial byte sequences to exercise:
//!
//! 1. Version detection (v4 / v5 / v7.3) via leading bytes.
//! 2. MAT v4 binary record parsing: type_code dispatch, mrows/ncols bounds,
//!    numeric payload reading (all machine/precision codes).
//! 3. MAT v5 data element tag and payload parsing: all `miType` tag codes,
//!    `miCOMPRESSED` branch (feature-gated), `miMATRIX` class dispatch,
//!    cell/struct recursion, and name/dimensions decoding.
//! 4. MAT v7.3 HDF5-backed parsing path (`v73` feature gate; returns
//!    `UnsupportedFeature` when the gate is absent).
//!
//! Expected outcomes for adversarial input:
//! - `MatError::InvalidFormat`    — structurally invalid payload.
//! - `MatError::UnsupportedFeature` — feature-gated or intentionally
//!   rejected class (mxOBJECT_CLASS, v4 sparse, v7.3 sparse).
//! - `MatError::Io`               — I/O-layer short read.
//!
//! Only panics constitute fuzzer failures.
#![no_main]

use consus_mat::loadmat_bytes;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // All Result variants are expected for adversarial input; panics are not.
    let _ = loadmat_bytes(data);
});
