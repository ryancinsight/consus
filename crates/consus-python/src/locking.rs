//! Attach-aware locking for state shared between Python threads.
//!
//! Blocking on a `std` lock while attached can deadlock against
//! interpreter-wide synchronization on a free-threaded build, so locks are
//! taken through PyO3's extension traits, which detach while waiting.

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use pyo3::exceptions::PyRuntimeError;
use pyo3::sync::{MutexExt, RwLockExt};
use pyo3::{PyErr, PyResult, Python};

/// Error for a lock whose holder panicked, leaving `class` state unspecified.
fn poisoned(class: &str) -> PyErr {
    PyRuntimeError::new_err(format!(
        "{class} state is unusable: a concurrent call panicked while updating it"
    ))
}

/// Lock `mutex`, which guards state of the binding class named `class`.
pub fn lock<'a, T>(
    mutex: &'a Mutex<T>,
    py: Python<'_>,
    class: &str,
) -> PyResult<MutexGuard<'a, T>> {
    mutex.lock_py_attached(py).map_err(|_| poisoned(class))
}

/// Lock `rwlock` for reading; `class` names the guarded binding class.
pub fn read<'a, T>(
    rwlock: &'a RwLock<T>,
    py: Python<'_>,
    class: &str,
) -> PyResult<RwLockReadGuard<'a, T>> {
    rwlock.read_py_attached(py).map_err(|_| poisoned(class))
}

/// Lock `rwlock` for writing; `class` names the guarded binding class.
pub fn write<'a, T>(
    rwlock: &'a RwLock<T>,
    py: Python<'_>,
    class: &str,
) -> PyResult<RwLockWriteGuard<'a, T>> {
    rwlock.write_py_attached(py).map_err(|_| poisoned(class))
}
