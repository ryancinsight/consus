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

Revision 2026-09-21: Consus owns the shared encoded-sample to packed eight-bit
display mapping. For precision maximum `M = 2^precision - 1`, it returns
`floor((sample * 255 + floor(M / 2)) / M)` through an allocation-free borrowed
iterator in encoded channel order. The mapping applies no orientation, alpha,
luminance, gamma or clinical presentation policy. Those policies remain with
the consuming presentation or medical domain.

Revision 2026-09-20: the Atlas conformance gate identified implementation-bearing
module manifests and numeric helper names in the initial delivery. EXIF parsing,
TIFF traversal and fields, JPEG storage bounds, encoding and container decoding
now have separate operation modules. Module manifests expose the same public
surface; this organization does not alter decoding or resource limits.

The admitted JPEG processes are 8/12-bit sequential/progressive DCT
and 2–16-bit single-component lossless coding, with Huffman or arithmetic entropy.
DCT reconstruction uses
a separable transform and nearest-neighbor chroma upsampling. The integer color
conversion and Adobe CMYK/YCCK conventions are tested independently of EXIF.
Hierarchical and differential frames remain unsupported.

Revision 2026-09-21: the shared raster item includes 12-bit DCT and arithmetic
coding. Apollo owns the transform mathematics;
Consus owns quantization, sample precision, entropy contexts and color conversion.
The transform dependency must preserve precomputed basis coefficients and
caller-owned working storage without importing an FFT or GUI runtime into the
raster package. Existing Apollo callers must use the extracted implementation;
adding a second inverse-transform implementation does not satisfy this decision.
The `apollo-dctdst-core` extraction lands in
[Apollo PR 526](https://github.com/ryancinsight/apollo/pull/526). Consus uses its
fixed-capacity orthonormal DCT-III plan for both dimensions of each JPEG block.

Arithmetic termination follows T.81 D.2.6: a detected marker supplies implicit
zero input while the dimension-bounded scan finishes. Physical end of input
without a marker remains malformed. Consequently, the Huffman fixture property
that every entropy cut with a replacement EOI fails is not an arithmetic
property. Arithmetic evidence instead covers independently encoded decisions,
coefficient values, legal termination, restart resets and malformed syntax.
Arithmetic lossless coding uses the above/left difference contexts of H.1.2.3;
the DCT DC model cannot substitute for those contexts. Width-dependent context
storage must participate in the admission bound before allocation.

[T.81](https://www.w3.org/Graphics/JPEG/itu-t81.pdf), Annexes A, D, F, G and H,
defines reconstruction, scan progression and lossless prediction. Lossless
addition wraps modulo 65,536, Huffman category 16 consumes no magnitude bits, and each
restart begins a new prediction row. Component quantization tables are captured
when the component's first scan begins; later definitions cannot change it.
Baseline scans restrict Huffman table destinations to 0/1. Progressive scans
reject changed quantization values; separate sequential components may capture
different definitions of the same table destination.
The ten-block MCU limit applies only to components in an interleaved scan.
Noninterleaved scans admit the full frame sampling range; the public storage
bound accounts for four components with sampling factors up to 4×4 and their
padded coefficient and sample planes.

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
