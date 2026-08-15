# ADR 0001: Generic endian scalar reader

- Status: Accepted
- Date: 2026-08-15
- Board item: `ATLAS-CONSUS-TYPES-057`

## Decision

Consus-core exposes one generic `read_integer<T>` operation for fixed-width
signed and unsigned scalar values. The scalar seam owns its byte width and
little-/big-endian conversion through an associated compile-time contract.
Each supported scalar implementation monomorphizes to the same native byte
conversion that the former type-specific functions used. Direct consumers
select `T` at the call site; no type-named compatibility functions remain.

## Constraints

The operation must preserve exact byte-order semantics, avoid allocation and
runtime dispatch, and remain usable by the no-`alloc` core. The public API
must not encode a scalar type in a function name when the operation is
generic. The change is allowed to break the obsolete in-repository names
because all authorized callers migrate in the same change and no forwarding
aliases are retained.

## Rejected alternative

Keeping six forwarding functions would preserve the prohibited type-suffixed
surface and duplicate the ownership boundary. A runtime enum would add
branching to every scalar read and would not provide the zero-cost
monomorphized contract.

## Verification

The generic conformance tests instantiate every supported signed and unsigned
width and assert both byte orders. Focused provider tests, strict Clippy,
formatting, doctests, and the hosted provider matrix gate integration.
