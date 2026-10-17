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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    raster = args.root.resolve() / "crates/consus-raster"
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
        assert len(provider) == 1 and provider[0]["version"] == "0.17.0", provider
        assert provider[0]["source"].startswith("git+https://github.com/ryancinsight/apollo#"), provider
        print(json.dumps({k: provider[0][k] for k in ["name", "version", "source"]}), flush=True)
        target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
        executable = target / "debug" / ("consus-provider-cohort.exe" if os.name == "nt" else "consus-provider-cohort")
        subprocess.run([str(executable)], cwd=root, check=True, timeout=60)


if __name__ == "__main__":
    main()
