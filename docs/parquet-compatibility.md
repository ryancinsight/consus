# Parquet compatibility after the codec correction

The correction in PRs #209/#210 and the page-length work through #218 break
compatibility with some files written by earlier Consus revisions. ADR0009
is the normative decision record. This note is not a conversion tool.

## Published baseline and wire interpretation

The downloaded `consus-parquet 0.4.1` archive identifies source
`43ac0268304e1b04b1c1d21d35bfb22923385a19` in `.cargo_vcs_info.json`.
Its SHA256 is `64fa3157aaf1b89509e6985a124466a7e40bf342b5a6212029816ff950c23a93`.
Both that writer and the corrected writer emit `created_by = consus-parquet`
without a version. That field does not establish which format was written.

| Wire number | Historical Consus meaning | Corrected/spec meaning |
| --- | --- | --- |
| 0 | UNCOMPRESSED | UNCOMPRESSED |
| 1 | SNAPPY | SNAPPY |
| 2 | GZIP label, raw DEFLATE bytes | RFC 1952 GZIP |
| 3 | raw LZ4 block | LZO (unsupported) |
| 4 | ZSTD | BROTLI (unsupported) |
| 5 | size-prepended LZ4 block | deprecated LZ4 (unsupported) |
| 6 | BROTLI (unsupported) | ZSTD |
| 7 | ZLIB label, raw DEFLATE bytes | raw LZ4 block |

A matching codec number, successful decode or expected decoded length is not
proof of writer provenance. Never automatically try the old format after a
new-format failure, or remap every file carrying a Consus writer string.
Conforming files from other writers already use the corrected numbers.

## Non-destructive migration procedure

1. Prefer regenerating from authoritative source data with the corrected
   writer. Otherwise obtain the exact writer version/revision, codec options
   and a known-good schema/value oracle from the producing application.
2. Preserve the original bytes and SHA256 in immutable or separately backed-up
   storage. Work on a new file with a distinct name; do not overwrite input.
3. With established historical provenance, decode using that exact historical
   implementation in an isolated environment. Bound allocations and reject
   malformed/truncated input. Do not treat an obsolete decoder as a detector.
4. Write a new standard-format file using a corrected writer. GZIP framing and
   old size-prepended LZ4 require actual decoding/re-encoding, not just changing
   metadata numbers. Rewriting may change offsets, page sizes and checksums.
5. Independently reread the output and compare schema, row counts, nulls,
   logical types and every value against the oracle. Record tool revisions,
   options and input/output hashes. Retain the original after verification.

## Executed evidence and limits (2026-11-18)

An isolated PyArrow 23.0.1 fixture with values `[0, -1, 17, 2147483647]`
was written as ZSTD, then its footer codec changed from 6 to historical 4.
A conforming read failed with `Corrupt brotli compressed data.` Explicitly
restoring 6 on a copy with known ZSTD provenance recovered those exact values;
an uncompressed rewrite reread equal, and the synthetic original hash was
unchanged. A raw-DEFLATE payload was rejected as GZIP; explicit raw decoding
followed by RFC 1952 encoding preserved the payload.

These are synthetic codec witnesses, not execution of a historical Consus
binary, qualification of a general converter, or evidence of affected user
data. The four tracked Parquet fixtures inspected at
`0ad1040530ba854520d58e721fff0e801beb55fb` are third-party UNCOMPRESSED files
and conforming PyArrow GZIP/ZSTD fixtures. No actual legacy dataset has been
established by that inventory. Native Rust gates were not run for this note.

## Release requirements

Classify against the published baseline, advance the breaking release to at
least 0.5.0, and update affected downstream requirements together. A uniform
workspace bump also affects consumers of other Consus crates: for example,
Kwavers currently declares Consus core/io/hdf5/compression/npy `^0.4.0`.
Preserve all existing provider-coherence and registry-owner holds. Neither
this note nor a successful synthetic conversion establishes release readiness.
