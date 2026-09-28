//! Synchronous zero-copy and parallel I/O helpers.
//!
//! This module contains only format-agnostic synchronous utilities used by the
//! public `consus` facade. It contains zero format-specific logic.

extern crate alloc;

use alloc::vec::Vec;

use consus_core::{Datatype, Error, Result, Selection, Shape};
use consus_io::{Length, ReadAt, WriteAt};

mod range_bytes;
pub use range_bytes::RangeBytes;

#[cfg(feature = "std")]
use moirai::ParallelSlice;

#[cfg(feature = "std")]
use std::sync::Arc;

/// Parallel I/O policy for synchronous facade operations.
///
/// This type is backend-neutral. It expresses only whether parallel execution
/// is enabled and how many partitions should be used for sufficiently large
/// reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parallelism {
    enabled: bool,
    min_len: usize,
    partitions: usize,
}

impl Default for Parallelism {
    fn default() -> Self {
        Self {
            enabled: true,
            min_len: 256 * 1024,
            partitions: default_partition_count(),
        }
    }
}

impl Parallelism {
    /// Creates the default policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables or disables parallel execution.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Sets the minimum byte length required before parallel partitioning is
    /// considered.
    pub fn min_len(mut self, min_len: usize) -> Self {
        self.min_len = min_len;
        self
    }

    /// Sets the preferred partition count.
    ///
    /// A value of zero is normalized to one.
    pub fn partitions(mut self, partitions: usize) -> Self {
        self.partitions = partitions.max(1);
        self
    }

    /// Returns whether parallel execution is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the minimum byte length threshold.
    pub fn min_parallel_len(&self) -> usize {
        self.min_len
    }

    /// Returns the configured preferred partition count.
    pub fn configured_partitions(&self) -> usize {
        self.partitions
    }

    /// Returns the effective partition count for a byte interval of `len`.
    pub fn partitions_for_len(&self, len: usize) -> usize {
        if !self.enabled || len < self.min_len {
            1
        } else {
            self.partitions.max(1)
        }
    }
}

#[cfg(feature = "std")]
fn default_partition_count() -> usize {
    #[cfg(feature = "atlas-themis")]
    {
        themis::CpuTopology::detect()
            .map(|topology| topology.logical_processors())
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1)
            .max(1)
    }

    #[cfg(not(feature = "atlas-themis"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1)
    }
}

#[cfg(not(feature = "std"))]
fn default_partition_count() -> usize {
    1
}

/// Borrowed-or-owned byte storage for zero-copy-capable synchronous reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ByteView<'a> {
    /// Borrowed bytes with zero-copy semantics.
    Borrowed(&'a [u8]),
    /// Owned bytes when materialization is required.
    Owned(Vec<u8>),
}

impl<'a> ByteView<'a> {
    /// Returns the underlying bytes.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes.as_slice(),
        }
    }

    /// Returns whether the view aliases existing storage.
    pub fn is_zero_copy(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Converts the view into owned bytes.
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            Self::Borrowed(bytes) => bytes.to_vec(),
            Self::Owned(bytes) => bytes,
        }
    }

    /// Returns the byte length.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns whether the view is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> AsRef<[u8]> for ByteView<'a> {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

/// Zero-copy-capable synchronous byte source.
///
/// Implementations may return borrowed storage when the requested region is
/// already resident in memory or otherwise directly exposable.
pub trait ZeroCopyRead {
    /// Reads `len` bytes starting at `offset`.
    fn read_zero_copy<'a>(&'a self, offset: u64, len: usize) -> Result<ByteView<'a>>;
}

impl<T> ZeroCopyRead for T
where
    T: ReadAt,
{
    fn read_zero_copy<'a>(&'a self, offset: u64, len: usize) -> Result<ByteView<'a>> {
        let mut buffer = vec![0u8; len];
        self.read_at(offset, &mut buffer)?;
        Ok(ByteView::Owned(buffer))
    }
}

/// Typed byte view validated against a canonical datatype.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedByteView<'a> {
    bytes: ByteView<'a>,
    datatype: Datatype,
    elements: usize,
}

impl<'a> TypedByteView<'a> {
    /// Creates a typed view after validating byte length against datatype size.
    pub fn new(bytes: ByteView<'a>, datatype: Datatype) -> Result<Self> {
        let element_size = datatype
            .element_size()
            .ok_or_else(variable_length_zero_copy_unsupported)?;

        if !bytes.len().is_multiple_of(element_size) {
            return Err(Error::DatatypeMismatch {
                #[cfg(feature = "alloc")]
                expected: alloc::format!("byte length multiple of element size {element_size}"),
                #[cfg(feature = "alloc")]
                found: alloc::format!("{} bytes", bytes.len()),
            });
        }

        Ok(Self {
            elements: bytes.len() / element_size,
            bytes,
            datatype,
        })
    }

    /// Returns the raw bytes.
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Returns the canonical datatype.
    pub fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    /// Returns the number of logical elements.
    pub fn elements(&self) -> usize {
        self.elements
    }

    /// Returns whether the underlying storage is borrowed.
    pub fn is_zero_copy(&self) -> bool {
        self.bytes.is_zero_copy()
    }

    /// Consumes the view and returns owned bytes.
    pub fn into_owned(self) -> Vec<u8> {
        self.bytes.into_owned()
    }
}

/// Absolute byte range used by parallel I/O planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IoRange {
    /// Absolute byte offset.
    pub offset: u64,
    /// Byte length.
    pub len: usize,
}

impl IoRange {
    /// Creates a new range.
    pub const fn new(offset: u64, len: usize) -> Self {
        Self { offset, len }
    }

    /// Returns the exclusive end offset.
    pub fn end(self) -> Result<u64> {
        self.offset
            .checked_add(self.len as u64)
            .ok_or(Error::Overflow)
    }
}

/// Computes the byte length for a selection over a dataset shape and datatype.
pub fn selection_byte_len(
    datatype: &Datatype,
    shape: &Shape,
    selection: &Selection,
) -> Result<usize> {
    let element_size = datatype
        .element_size()
        .ok_or_else(variable_length_selection_sizing_unsupported)?;

    if !selection.is_valid_for_shape(shape) {
        return Err(Error::SelectionOutOfBounds);
    }

    selection
        .num_elements(shape)
        .checked_mul(element_size)
        .ok_or(Error::Overflow)
}

/// Reads a typed byte region from a zero-copy-capable source.
pub fn read_typed<'a, R>(
    reader: &'a R,
    offset: u64,
    datatype: Datatype,
    byte_len: usize,
) -> Result<TypedByteView<'a>>
where
    R: ZeroCopyRead + ?Sized,
{
    let bytes = reader.read_zero_copy(offset, byte_len)?;
    TypedByteView::new(bytes, datatype)
}

/// Reads multiple disjoint byte ranges sequentially.
///
/// Returns one contiguous [`RangeBytes`] instead of a `Vec<Vec<u8>>`: `n`
/// ranges cost one allocation rather than `n`. This wins clearly (~4x in
/// `benches/range_reads.rs`) when the batch's total bytes stay small enough
/// that the jagged form's many small allocations dominate; on this host, a
/// single-call batch large enough to need a multi-MiB buffer instead loses
/// to the jagged form (measured up to ~1.75x slower), plausibly because a
/// buffer of that size is allocated and freed fresh on every call and falls
/// outside the allocator's reused small-object arena, paying OS-level
/// allocation and first-touch page-fault cost the jagged form's reused
/// small blocks avoid after warm-up (unconfirmed beyond this measurement;
/// no allocator-level profile was taken). Neither shape dominates
/// unconditionally, and this function has no in-tree caller yet to
/// establish which regime matters in practice.
pub fn read_ranges<R>(reader: &R, ranges: &[IoRange]) -> Result<RangeBytes>
where
    R: ReadAt + ?Sized,
{
    let mut out = RangeBytes::with_capacity(ranges);
    for range in ranges {
        let buf = out.push_zeroed(range.len);
        reader.read_at(range.offset, buf)?;
    }
    Ok(out)
}

/// Writes multiple disjoint byte ranges sequentially.
pub fn write_ranges<W>(writer: &mut W, writes: &[(IoRange, &[u8])]) -> Result<()>
where
    W: WriteAt + ?Sized,
{
    for (range, bytes) in writes {
        if bytes.len() != range.len {
            return Err(Error::BufferTooSmall {
                required: range.len,
                provided: bytes.len(),
            });
        }
        writer.write_at(range.offset, bytes)?;
    }

    writer.flush()
}

#[cfg(feature = "std")]
/// Reads multiple disjoint byte ranges in parallel.
///
/// This helper is format-agnostic. It parallelizes positioned reads over any
/// source implementing `ReadAt + Send + Sync`.
///
/// Each parallel task reads into its own owned buffer, and those per-task
/// buffers are concatenated into one contiguous [`RangeBytes`] afterward, so
/// no jagged container crosses this function's boundary. This costs one
/// extra full copy of the read bytes relative to returning the per-task
/// buffers directly: range lengths are known up front (a prefix sum over
/// `ranges`), so pre-splitting one flat buffer into disjoint `&mut [u8]`
/// views via repeated safe `split_at_mut` and reading directly into them
/// would avoid it, at the cost of building an intermediate `Vec` of those
/// views to pair with `ranges` for the parallel dispatch. Not done here:
/// this function has no in-tree caller to measure the tradeoff against.
pub fn par_read_ranges<R>(reader: Arc<R>, ranges: &[IoRange]) -> Result<RangeBytes>
where
    R: ReadAt + Send + Sync + 'static,
{
    // Order-preserving parallel map; `map_collect` keeps input order, so no
    // index/sort bookkeeping is needed.
    let chunks: Vec<Result<Vec<u8>>> = ranges.par().map_collect(|range| {
        let mut buffer = vec![0u8; range.len];
        reader.read_at(range.offset, &mut buffer)?;
        Ok::<Vec<u8>, Error>(buffer)
    });
    RangeBytes::try_from_chunks(chunks)
}

#[cfg(not(feature = "std"))]
/// Sequential fallback with the same signature shape as the parallel helper.
pub fn par_read_ranges<R>(reader: R, ranges: &[IoRange]) -> Result<RangeBytes>
where
    R: ReadAt,
{
    read_ranges(&reader, ranges)
}

#[cfg(feature = "std")]
/// Computes a balanced partition of a byte interval for parallel I/O.
pub fn partition_range(total_len: usize, partitions: usize) -> Result<Vec<IoRange>> {
    if partitions == 0 {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::string::String::from(
                "parallel partition count must be greater than zero",
            ),
        });
    }

    if total_len == 0 {
        return Ok(Vec::new());
    }

    let base = total_len / partitions;
    let remainder = total_len % partitions;

    let mut offset = 0usize;
    let mut ranges = Vec::with_capacity(partitions.min(total_len));

    for index in 0..partitions {
        let len = base + usize::from(index < remainder);
        if len == 0 {
            continue;
        }

        ranges.push(IoRange::new(offset as u64, len));
        offset = offset.checked_add(len).ok_or(Error::Overflow)?;
    }

    Ok(ranges)
}

#[cfg(not(feature = "std"))]
/// Computes a balanced partition of a byte interval.
pub fn partition_range(total_len: usize, partitions: usize) -> Result<Vec<IoRange>> {
    if partitions == 0 {
        return Err(Error::InvalidFormat {
            #[cfg(feature = "alloc")]
            message: alloc::string::String::from(
                "parallel partition count must be greater than zero",
            ),
        });
    }

    if total_len == 0 {
        return Ok(Vec::new());
    }

    let base = total_len / partitions;
    let remainder = total_len % partitions;

    let mut offset = 0usize;
    let mut ranges = Vec::with_capacity(partitions.min(total_len));

    for index in 0..partitions {
        let len = base + usize::from(index < remainder);
        if len == 0 {
            continue;
        }

        ranges.push(IoRange::new(offset as u64, len));
        offset = offset.checked_add(len).ok_or(Error::Overflow)?;
    }

    Ok(ranges)
}

/// Returns the byte length of a source.
pub fn source_len<S>(source: &S) -> Result<u64>
where
    S: Length + ?Sized,
{
    source.len()
}

fn variable_length_zero_copy_unsupported() -> Error {
    Error::UnsupportedFeature {
        #[cfg(feature = "alloc")]
        feature: alloc::string::String::from("variable-length zero-copy typed reads"),
    }
}

fn variable_length_selection_sizing_unsupported() -> Error {
    Error::UnsupportedFeature {
        #[cfg(feature = "alloc")]
        feature: alloc::string::String::from("variable-length selection byte sizing"),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
