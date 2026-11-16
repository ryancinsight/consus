# ADR 0009: Parquet codec numbers and GZIP framing follow the specification

- Status: Accepted
- Date: 2026-11-15
- Class: [major] for `consus-parquet` (wire format)
- Item: CONSUS-PARQUET-CODEC-SPEC
- Revision 2026-11-15: GZIP reads every member (was first member only), because
  `Compression.md` says readers should support pages with several members; the
  decoded length of every codec must equal the declared size (consus#214,
  consus#213).

## Context

`consus-parquet` stored its own codec numbering in column chunk metadata
(3 = LZ4_RAW, 4 = ZSTD, 5 = LZ4, 6 = BROTLI, 7 = ZLIB) and wrote bare RFC 1951
DEFLATE under GZIP. apache/parquet-format
<https://github.com/apache/parquet-format/blob/ec90dd4661ff1c4d77f06b5b5232c5aedf68ffc5/src/main/thrift/parquet.thrift#L684-L693>
defines UNCOMPRESSED = 0, SNAPPY = 1, GZIP = 2, LZO = 3, BROTLI = 4,
LZ4 = 5 (deprecated), ZSTD = 6, LZ4_RAW = 7 and no ZLIB. At the same
revision, `Compression.md` lines 54-63 define GZIP as RFC 1952,
"not the closely-related 'zlib' or 'deflate' formats".

## Decision

1. `CompressionCodec` carries exactly the eight specified numbers. ZLIB is
   removed; every other number is rejected by `from_i32` and the reader
   reports `InvalidFormat`.
2. GZIP pages are written as one RFC 1952 member and read by
   `consus_compression::codec::gzip::GzipCodec`, which decodes every member
   (RFC 1952 section 2.2). `Compression.md` says readers should support
   several members. Bytes after the last member are rejected with
   `CompressionError`: section 2.2 allows no additional information after the
   members.
   `decompress_page_values` returns the page only when its decoded length
   equals the declared size, so a page is never returned truncated or padded.
   The declared size is the header `uncompressed_page_size`, less the level
   sections for DataPage v2 values. A declared size above the shared
   allocation budget is `ResourceLimit` for every codec, before any data is
   decoded. With a declared size within the budget, decoded output above the
   budget is `ResourceLimit` for UNCOMPRESSED, SNAPPY (the length its stream
   states) and GZIP, and `CompressionError` for ZSTD and LZ4_RAW, whose
   output limit is the declared size. Any other length that differs from the
   declared size is `CompressionError`. DataPage v2 values stored as is
   (UNCOMPRESSED, or `is_compressed = false`) do not pass through that
   function: they are only compared with the declared size, so a difference
   is `CompressionError` whatever its size. A DataPage v2 header smaller
   than its level sections is `InvalidFormat`.
3. LZO, BROTLI and the deprecated LZ4 return `UnsupportedFeature` in both
   directions. The former LZ4 (`lz4_flex` size-prepended blocks) matched no
   specified framing, so it is removed rather than renumbered. LZ4_RAW is the
   supported LZ4 codec.

## Rejected alternatives

- Keep ZLIB as an extension number: any value collides with a future codec.
- Read bare DEFLATE under GZIP as a fallback: a silent format guess.
- Implement Hadoop-framed LZ4: undocumented framing, no oracle here.
- Ignore bytes after the last GZIP member as zlib `gzread` does (`zlib.h`
  lines 1464-1466 in zlib 1.3.2), or tolerate zero padding as numcodecs and
  GNU gzip do: the stream is then not RFC 1952 section 2.2, and a decoder
  that accepts it hides truncation or concatenation errors. Overturned by a
  conforming Parquet or Zarr writer found to emit padded pages or chunks.

## Consequences

Files written by earlier consus versions with GZIP, ZSTD, LZ4_RAW, LZ4 or
ZLIB pages are not readable: the wire carries no writer version, and each old
number now names a different codec. They fail with a typed error
(`CompressionError` or `UnsupportedFeature`) except old ZLIB pages (7), which
are now read as LZ4_RAW and are expected to fail LZ4 block validation. Rewrite
them from source data. UNCOMPRESSED and SNAPPY files written by consus are
unaffected, since its writer declares the payload length.

Pages from other writers whose header `uncompressed_page_size` disagrees with
the payload now fail with a typed error; readers that ignore the field
accepted them. The declared size is the header `uncompressed_page_size`, less
the level sections for DataPage v2 values. The variant depends on the page:

- `ResourceLimit`: on a page read through `decompress_page_values`, a
  declared size above the shared allocation budget (for compressed DataPage
  v2 values behind `L` level bytes, a header above the budget plus `L`), or,
  with a declared size within the budget, decoded output above the budget
  under UNCOMPRESSED, SNAPPY or GZIP.
- `CompressionError`: a dictionary page, a DataPage v1 page (any codec,
  UNCOMPRESSED included) or DataPage v2 values compressed under any other
  codec, read through `decompress_page_values`, decodes to another length
  than the declared size and the previous bullet does not apply (ZSTD and
  LZ4_RAW fail inside the codec with its own message when the output exceeds
  the declared size); or DataPage v2 values stored as is (UNCOMPRESSED, or
  `is_compressed = false`) differ from the declared size, at any size.
- `InvalidFormat`: a DataPage v2 header smaller than its level sections.
- Index pages are skipped without comparing the declared size.

## Evidence

Unit tests assert each specified number, the `1f 8b 08` gzip header, rejection
of bare DEFLATE, a two-member GZIP page decoded in full, a declared size one
byte off the decoded length for every supported codec (DataPage v2 pages
with stored values and with level sections included), decoded output above
the budget under a small declared size for every codec, the budget boundary
behind four level bytes, and the codec number
stored in column chunk metadata for every supported codec. Two pyarrow 23.0.1
files (GZIP, ZSTD) generated by
`crates/consus-parquet/tests/data/gen_codec_fixtures.py` decode to the written
values with codec numbers 2 and 6.

## Overturning evidence

- A parquet-format revision that renumbers `CompressionCodec`.
- A conforming Parquet or Zarr writer found to emit trailing or padding bytes
  after the last GZIP member (Decision 2).
- A widely used Parquet writer found to declare an `uncompressed_page_size`
  that differs from the page payload (Decision 2).
