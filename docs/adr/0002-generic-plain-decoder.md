# ADR 0002: Generic Parquet PLAIN scalar decoder

- Status: Accepted
- Date: 2026-08-15
- Board item: `ATLAS-CONSUS-PARQUET-058`

## Context

`consus-parquet::encoding::plain` has four public scalar decoders for Parquet
INT32, INT64, FLOAT, and DOUBLE. Each repeats the same checked `count * width`
validation, `ParseBudget` reservation, and fixed-width loop. The only semantic
variation is the scalar's width and little-endian bit interpretation. Keeping
one function per scalar duplicates the implementation and makes safety fixes
drift across physical types.

## Decision

Expose one `decode_plain<T: PlainValue>` function. `PlainValue` is sealed to
the four supported scalar types and provides an associated compile-time byte
width, a diagnostic value label, and fallible little-endian conversion. The
decoder iterates exact-width chunks after one checked byte-product validation;
it does not allocate heap buffers or widen numeric values.

The four type-named public entry points and their re-exports are removed in the
same change. In-repository callers select `T` at the Parquet physical-type
boundary, and external callers migrate to `decode_plain::<T>`.

## Alternatives rejected

- Retaining forwarding functions preserves the duplicate public surface and
  violates the provider replacement rule.
- A runtime enum or `dyn` decoder adds dispatch to a fixed-width hot loop and
  weakens the zero-cost generic contract.
- A macro-generated family reproduces the duplicated API and hides the shared
  invariants from the type system.

## Verification

Generic tests cover all four scalar implementations, empty input, truncated
input, and exact IEEE bit-pattern round trips. Focused Parquet Nextest,
strict Clippy, doctests, semver analysis, the provider conformance scan, and
the exact provider hosted matrix are required before Atlas advances the
gitlink.
