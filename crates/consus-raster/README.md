# consus-raster

`consus-raster` provides bounded, pure-Rust raster decoding for untrusted input.
It decodes sequential and progressive JPEG images to grayscale or RGB pixels and
lossless JPEG grayscale images at their encoded precision. Callers supply all
resource limits explicitly.

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
# Ok::<(), consus_raster::DecodeError>(())
```

JPEG decoding returns the encoded pixel grid. EXIF orientation is metadata and
is not applied to the returned pixels.
