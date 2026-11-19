# ADR 0010: Select Level-5 variables in the Consus MAT reader

Status: Accepted

## Context

Kwavers had a second Level-5 parser to find one MRI array in a MAT file.
The shared reader previously decoded every array and treated the matrix
class as the storage type. The MathWorks Level-5 MAT-File Format guide
specifies separate class and numeric storage tags. It permits a double
matrix to store integral values using narrower integer elements.

## Decision

Use the same top-level traversal and matrix parser for full-file reads and
named selection. Named reads parse the common flags, dimensions, and name
and skip values for other variables. Decode packed double payloads by their
storage tag, normalize to little-endian double, and preserve column-major
element order. A source must contain exactly the product of its dimensions
in stored scalars. Check the expanded footprint against the parser budget
before allocation. Binary64 preserves the full range of integer types
through 32 bits. Conversion of 64-bit integers follows binary64 rounding.

Kwavers retains rank, shape and real-valuedness validation and the MRI
interpolation and calibration. Consus owns MATLAB framing, decompression,
byte order and scalar decoding.

## Verification

Use compressed packed doubles, consecutive unpadded elements, and an
unsupported unrelated variable. Assert decoded arrays and exact shape,
including a negative signed value and the largest unsigned 16-bit value.
Exercise the existing Kwavers phantom mapping through its public loader.
