/// A checksum algorithm.
///
/// ## Contract
///
/// - **Incremental equivalence**: calling `update(a)` then `update(b)` produces
///   the same checksum as a single `update(a ++ b)` where `++` denotes
///   byte-sequence concatenation.
/// - **Determinism**: identical input sequences produce identical checksums.
/// - **Reuse**: after `reset`, the instance behaves identically to a freshly
///   constructed default instance.
///
/// ## Formal invariant
///
/// Let `H₀` denote the default (initial) state. For any byte sequences `a`, `b`:
///
/// ```text
/// let mut h = H₀;
/// h.update(a); h.update(b);
/// let r1 = h.finalize();
///
/// let mut h = H₀;
/// h.update(a ++ b);
/// let r2 = h.finalize();
///
/// assert!(r1 == r2);
/// ```
pub trait Checksum {
    /// The output type of the checksum (e.g., `u32`).
    type Output: Copy + Eq + core::fmt::Debug;

    /// Feed `data` into the running checksum computation.
    ///
    /// May be called zero or more times between construction (or `reset`)
    /// and `finalize`.
    fn update(&mut self, data: &[u8]);

    /// Return the checksum value accumulated so far.
    ///
    /// This method is non-destructive: calling `finalize` twice without
    /// an intervening `update` returns the same value.
    fn finalize(&self) -> Self::Output;

    /// Reset internal state to the initial (default-constructed) value.
    fn reset(&mut self);

    /// Convenience: compute the checksum of a single contiguous slice.
    ///
    /// Equivalent to:
    /// ```text
    /// let mut h = Self::default();
    /// h.update(data);
    /// h.finalize()
    /// ```
    fn compute(data: &[u8]) -> Self::Output
    where
        Self: Default,
    {
        let mut hasher = Self::default();
        hasher.update(data);
        hasher.finalize()
    }
}
