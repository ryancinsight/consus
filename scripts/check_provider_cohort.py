#!/usr/bin/env python3
"""Exercise a fresh, overlay-free raster consumer against a Consus source tree.

The old 0.16 Apollo requirement cannot resolve current Apollo main (0.17).
Run this same consumer against the baseline and repaired roots to reproduce
the admission failure and verify real JPEG reconstruction after correction.
No lock, provider patch, credential, or registry configuration is supplied.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib


def compatible(version, requirement):
    """Whether `version` satisfies caret `requirement` under Cargo's 0.x rule.

    A pre-1.0 caret requirement admits only its own minor line at or above
    itself (https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html).
    A fresh consumer resolves the provider's current main, so it must land
    inside that line, not on the patch release the requirement names.
    """
    have = tuple(int(part) for part in version.split("."))
    want = tuple(int(part) for part in requirement.split("."))
    assert want[0] == 0, requirement
    return have[:2] == want[:2] and have >= want


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    raster = args.root.resolve() / "crates/consus-raster"
    manifest = tomllib.loads((args.root.resolve() / "Cargo.toml").read_text(encoding="utf-8"))
    requirement = manifest["workspace"]["dependencies"]["apollo-dctdst-core"]["version"]
    fixture = Path(__file__).resolve().parent / "scratch/jpeg_provider_cohort.rs"
    with tempfile.TemporaryDirectory(prefix="consus-provider-cohort-") as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "src/main.rs").write_bytes(fixture.read_bytes())
        (root / "Cargo.toml").write_text(
            '[package]\nname = "consus-provider-cohort"\nversion = "0.0.0"\n'
            'edition = "2024"\n[workspace]\n[dependencies]\n'
            'consus-raster = { path = ' + json.dumps(str(raster)) + ' }\n',
            encoding="utf-8",
        )
        subprocess.run(["cargo", "build"], cwd=root, check=True, timeout=300)
        metadata = json.loads(subprocess.check_output(
            ["cargo", "metadata", "--locked", "--format-version", "1"],
            cwd=root, timeout=60,
        ))
        provider = [p for p in metadata["packages"] if p["name"] == "apollo-dctdst-core"]
        assert len(provider) == 1 and compatible(provider[0]["version"], requirement), (
            requirement,
            provider,
        )
        assert provider[0]["source"].startswith("git+https://github.com/ryancinsight/apollo#"), provider
        print(json.dumps({k: provider[0][k] for k in ["name", "version", "source"]}), flush=True)
        target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
        executable = target / "debug" / ("consus-provider-cohort.exe" if os.name == "nt" else "consus-provider-cohort")
        subprocess.run([str(executable)], cwd=root, check=True, timeout=60)


if __name__ == "__main__":
    main()
