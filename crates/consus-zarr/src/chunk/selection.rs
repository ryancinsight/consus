#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use super::error::ChunkError;

/// A step in a multi-dimensional selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionStep {
    /// Starting index of the selection.
    pub start: u64,
    /// Number of elements in this step.
    pub count: u64,
    /// Stride between elements (spacing).
    pub stride: u64,
}

impl SelectionStep {
    /// Returns true if this step represents a contiguous range.
    ///
    /// A step is contiguous when stride equals 1.
    pub fn contiguous(&self) -> bool {
        self.stride == 1
    }

    /// Returns the exclusive end index of this step.
    pub fn end(&self) -> u64 {
        self.start + (self.count.saturating_sub(1)) * self.stride + 1
    }

    /// Returns an iterator over the indices covered by this step.
    pub fn indices(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.count).map(move |i| self.start + i * self.stride)
    }
}

/// A multi-dimensional selection for array indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    /// The selection steps for each dimension.
    pub steps: Vec<SelectionStep>,
}

impl Selection {
    /// Creates a selection that covers the entire array.
    pub fn full(_dims: usize) -> Self {
        Self { steps: Vec::new() }
    }

    /// Creates a selection from explicit steps.
    pub fn from_steps(steps: Vec<SelectionStep>) -> Self {
        Self { steps }
    }

    /// Returns the total number of elements selected.
    pub fn num_elements(&self) -> u64 {
        if self.steps.is_empty() {
            return 0;
        }
        self.steps.iter().map(|s| s.count).product()
    }

    /// Returns true if this selection covers the full array.
    pub fn is_full(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns the selection step for each array dimension.
    ///
    /// For full-array selections, this materializes one contiguous step per
    /// dimension spanning the entire array extent.
    pub fn normalized_steps(&self, shape: &[usize]) -> Result<Vec<SelectionStep>, ChunkError> {
        if self.is_full() {
            return Ok(shape
                .iter()
                .map(|&extent| SelectionStep {
                    start: 0,
                    count: extent as u64,
                    stride: 1,
                })
                .collect());
        }

        if self.steps.len() != shape.len() {
            return Err(ChunkError::UnexpectedLength);
        }

        for (step, &extent) in self.steps.iter().zip(shape.iter()) {
            if step.stride == 0 {
                return Err(ChunkError::UnexpectedLength);
            }
            if step.count == 0 {
                return Err(ChunkError::UnexpectedLength);
            }
            if step.end() > extent as u64 {
                return Err(ChunkError::ChunkOutOfBounds);
            }
        }

        Ok(self.steps.clone())
    }
}
