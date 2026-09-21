# JPEG reference fixtures

The DCT pairs contain identical manufactured quantized coefficients. The source
streams use constant DC blocks, one horizontal AC coefficient, and sparse signed
coefficients. libjpeg-turbo 3.1.2 `jpegtran` produces the Huffman and arithmetic
variants without requantization, using sequential, six-scan refinement, and
one-MCU restart schedules. Tests compare decoded pairs and independent analytical
DC/AC pixels. Integer-IDCT rounding from another decoder is not a pixel oracle.

The lossless pairs encode prescribed 5-by-4 grayscale sample arrays at 2, 8, 12
and 16 bits using predictor 4 and restart intervals 0, 5 and 10. The unmodified
ISO reference implementation at revision
`c719010a26ce0c666e98b2acf924ad5fc24b4f5d` produces these fixtures. The command
form is `jpeg -p -c -a -z <interval> input.pgm output.jpg`; omitting `-a`
produces the Huffman peer. PGM maximum values specify `(1 << precision) - 1`.
The test stores the prescribed integer arrays and compares every decoded sample.

Reference programs are fixture-generation instruments only, not runtime
dependencies. The fixture bytes are small manufactured inputs. The decoder
implements ITU-T T.81 Annexes D, F, G and H; fixture agreement complements the
malformed-input, conditioning-boundary and working-storage tests.

## Twelve-bit precision fixtures

The `precision-reference-*` and `precision-turbo-*` DCT streams independently
encode a 24-by-8 analytical image whose three constant 8-by-8 tiles are 0,
2048 and 4095. The arithmetic streams come from the ISO reference encoder at
revision `c719010a26ce0c666e98b2acf924ad5fc24b4f5d` and libjpeg-turbo 3.1.2;
`jpegtran` produced their coefficient-preserving Huffman peers. Both external
decoders reconstruct every twelve-bit sample exactly.

The ISO reference encoder also emitted an empty Exif root image file directory
and two JPEG XT APP11 segments. TIFF Revision 6.0, Section 2, “Image File
Directory” (PDF page 14) requires every IFD to contain at least one entry:
<https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf>. Consus therefore
correctly rejects that Exif payload as malformed. The progressive stream also
contains three empty DAC segments, while T.81 B.2.4.3 defines a DAC segment as
containing one or more conditioning tables. The committed reference arithmetic
fixtures remove the complete APP1 and APP11 marker segments and, from the
progressive stream, those three empty DAC segments. The valid 12-byte Adobe
APP14 payload and every DQT, SOF, non-empty DAC, SOS, entropy, and EOI byte
remain unchanged. The original and normalized SHA-256 values are:

- sequential: `58956a81c2215308541b2136d43ed0123008e1ec48d04d49b90b62b1b172d15e`
  → `6e4a8d2428f0e6dfd708a82ae61d7369baafdae4061ef008211eb96cd5930f60`
- progressive: `ee618b0e16e6256a634df33a15074c57cbdc1bcbcd8253d966e0b07f062e37ef`
  → `361a2080a3ba7fb52ff5908d1255d9cdec5a3887b1b3199060264f20d2961a5d`

`precision-reference-empty-ifd-arithmetic.jpg` retains the original sequential
stream under its source hash as the malformed-metadata regression fixture. This
normalization removes marker segments with no coding payload and does not change
the analytical 0, 2048, and 4095 sample oracle.

The `precision-lossless-predictor-*` streams cover predictors 1 through 7 at
point transforms 0 and 3. libjpeg-turbo 3.1.2 encoded and decoded these files.
Their oracle is the prescribed source sample transformed by
`(sample >> point_transform) << point_transform`; this analytical law remains
authoritative where the ISO reference decoder disagrees for point transform 3.
The source `precision-manifest.json` SHA-256 is
`ba08cda88a0886334ea4a1f1d4a2a2a4dfd0e756c3c5388b0cc37fab5ac9ac91`.

## Color arithmetic fixtures

libjpeg-turbo 3.1.2 `jpegtran` transcoded Metis's 48-by-32 orientation image
without requantization into sequential, progressive, restart, and progressive
restart arithmetic streams. Coefficient-preserving Huffman transcodes of all
four streams are byte-identical, so one canonical peer represents them. Both
libjpeg-turbo and the ISO reference decoder independently confirm full RGB
image equality within each arithmetic/Huffman pair. The source image SHA-256
is `042782bd0f6c9488e84bfb7e229e3ab594e0d7a75d5f838290034d8199d7ea2e`;
the `color-manifest.json` SHA-256 is
`9cab30de4845e4a0dc1c27b07cffca723fbe86a44efb65665d2b1cd3e275b212`.
