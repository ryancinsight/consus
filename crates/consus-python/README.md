# atlas-consus

PyO3 bindings for the Consus scientific storage library. The PyPI distribution
is `atlas-consus`; the import name remains `consus`.

## Releases

GitHub Releases tagged `atlas-consus-v<version>` build locked Linux, Windows,
and universal macOS wheels for CPython 3.9 through 3.13. The release workflow
installs and imports each wheel, validates its distribution name and version,
generates SHA-256 checksums and build provenance, attaches those artifacts to
the GitHub Release, and publishes the same wheels to PyPI through OIDC Trusted
Publishing. The tag version must equal the `consus-python` Cargo package
version; Cargo is the sole version source.

## Architecture

This crate only translates Python values, maps Rust failures to Python
exceptions, and registers the extension module. Storage formats, validation,
and I/O behavior remain in their owning Consus Rust crates.
