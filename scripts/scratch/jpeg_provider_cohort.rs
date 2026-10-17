use consus_raster::{Compression, DecodeLimits, PixelFormat, jpeg};

fn main() {
    // A constant 8x8 block has only a DC coefficient. Its reconstruction is
    // exactly the input sample at quality 100, including both clipping limits.
    for value in [0, 64, 128, 192, 255] {
        let pixels = [value; 64];
        let encoded = jpeg::encode_gray(&pixels, 8, 8, 100).expect("encode constant block");
        let image = jpeg::decode(
            &encoded,
            DecodeLimits {
                max_encoded_bytes: 4096,
                max_dimension: 8,
                max_pixels: 64,
                max_working_bytes: 1 << 20,
            },
        )
        .expect("decode constant block through Apollo IDCT");
        assert_eq!((image.width(), image.height()), (8, 8));
        assert_eq!(image.format(), PixelFormat::Gray);
        assert_eq!(image.sample_precision(), 8);
        assert_eq!(image.compression(), Compression::Lossy);
        assert_eq!(image.pixels(), pixels, "constant sample {value}");
    }
    println!("five constant JPEG blocks preserve all 320 samples and metadata");
}
