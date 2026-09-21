# ADR 0004: Shared raster codecs

- Status: Accepted
- Date: 2026-09-20
- Class: [arch] [minor]
- Item: [CONSUS-RASTER-001](../../backlog.md#CONSUS-RASTER-001)

## Decision

Consus owns a separate `consus-raster` package for bounded raster-format bytes.
RITK's JPEG primitives and Metis's JPEG/EXIF admission converge here. The package
has no dependency on either consumer, medical image types, windows or rendering.
JPEG's lossy transform does not implement the exact-roundtrip compression trait.

The shared decoder preserves integer sample representation, including lossless
medical precision. Callers set resource limits before allocation. EXIF describes
encoded-grid orientation; display normalization is distinct from clinical geometry.
RITK retains DICOM layout, signedness and modality conversion. Metis retains
capability-controlled file reads, display limits and raster placement.

The admitted JPEG processes are eight-bit Huffman sequential/progressive DCT
and 8–16-bit single-component Huffman lossless coding. DCT reconstruction uses
a separable transform and nearest-neighbor chroma upsampling. The integer color
conversion and Adobe CMYK/YCCK conventions are tested independently of EXIF.
Arithmetic-coded JPEG and wider DCT samples remain unsupported.

[T.81](https://www.w3.org/Graphics/JPEG/itu-t81.pdf), Annexes A, F, G and H,
defines reconstruction, scan progression and lossless prediction. Lossless
addition wraps modulo 65,536, category 16 consumes no magnitude bits, and each
restart begins a new prediction row. Component quantization tables are captured
when the component's first scan begins; later definitions cannot change it.

## Alternatives

A Metis dependency on RITK creates a repository cycle because RITK's viewer
already depends on Metis. Separate desktop and clinical JPEG implementations
duplicate parsing and diverge on truncation and resource behavior. Iris explicitly
excludes file formats. Consus already owns format parsing and compression.

## Threat boundary and evidence

Encoded bytes are untrusted. Dimensions, table counts, entropy scans and metadata
offsets must validate before they control allocation or indexing. Caller-selected
encoded, pixel and working-memory bounds constrain resource use. Filesystem
capabilities remain outside the codec. Unknown presentation metadata is rejected
by display admission rather than silently interpreted as clinical orientation.

Acceptance includes exact lossless samples and modality conversion, independent
sequential/progressive fixtures, restart and refinement behavior, all eight EXIF
orientations, malformed prefixes, and resource-limit boundaries. Consumer native
presentation evidence remains in the Metis manual; decoder tests do not establish
Windows display behavior. Migration is incomplete until both consumer copies are
removed and their focused configured gates pass.

The complete decoder path has an analytical single-AC-coefficient pixel oracle.
Sequential and progressive encodings of one independently produced image must
also agree. A Pillow all-pixel comparison on that 4:2:0 fixture is not an equality
oracle: Pillow uses different chroma interpolation. It is not used to justify a
widened tolerance or a reconstruction correctness claim.
