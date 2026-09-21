/// Resource limits applied before decoding allocates image-dependent storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Maximum accepted encoded input length in bytes.
    pub max_encoded_bytes: usize,
    /// Maximum accepted width or height.
    pub max_dimension: u32,
    /// Maximum accepted number of encoded-grid pixels.
    pub max_pixels: usize,
    /// Maximum aggregate image-dependent working storage in bytes.
    ///
    /// This includes coefficients or lossless samples, reconstructed component
    /// planes, allocation descriptors, and the final output while they coexist.
    /// Use [`crate::jpeg::working_storage_bound`] when encoded-grid dimensions
    /// are available before decoding.
    pub max_working_bytes: usize,
}
