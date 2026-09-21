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
