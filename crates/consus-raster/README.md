# consus-raster

`consus-raster` provides bounded, pure-Rust raster decoding for untrusted input.
It decodes sequential and progressive JPEG images to grayscale or RGB pixels and
lossless JPEG grayscale images at their encoded precision. Callers supply all
resource limits explicitly.

DCT images admit 8 or 12 bits per sample; single-component lossless images
admit 2 through 16 bits. Both Huffman and arithmetic entropy coding are supported.
`DecodedImage::sample_precision()` reports meaningful sample bits. Wide grayscale
and RGB formats store native-endian `u16` channels; the decoder preserves their
integer values. `DecodedImage::display_samples()` exposes an allocation-free
borrowed iterator that maps every encoded sample to the nearest packed eight-bit
value. It preserves encoded channel order and leaves orientation, alpha, gamma,
and clinical presentation policy to the consumer.
`DecodedImage::compression()` distinguishes quantized DCT from predictive coding;
a predictive point transform restores discarded low bits as zero. Hierarchical
and differential JPEG processes are rejected as unsupported.

```rust
use consus_raster::{DecodeLimits, PixelFormat, jpeg};

let encoded = jpeg::encode_gray(&[0; 64], 8, 8, 100)?;
let image = jpeg::decode(
    &encoded,
    DecodeLimits {
        max_encoded_bytes: 1 << 20,
        max_dimension: 4096,
        max_pixels: 16 << 20,
        max_working_bytes: 64 << 20,
    },
)?;
assert_eq!(image.format(), PixelFormat::Gray);
assert_eq!((image.width(), image.height()), (8, 8));
assert_eq!(image.pixels(), &[0; 64]);
assert!(image.display_samples().eq([0; 64]));
# Ok::<(), consus_raster::DecodeError>(())
```

JPEG decoding returns the encoded pixel grid. EXIF orientation is metadata and
is not applied to the returned pixels.

## Attribution

JPEG encoding uses `jpeg-encoder`. This software is based in part on the work
of the Independent JPEG Group.
