#!/usr/bin/env python3
"""Execute the published-to-current public async reader compatibility check.

The identical generic consumer must build on published Consus HDF5 0.2.0 and
Moirai 0.7, fail with E0277 on the candidate's new public Moirai identity, and
build after the caller migrates to the candidate's Moirai 0.9 provider.
This fixture supplies no patch, old Git revision, or compatibility adapter.
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="consus-async-boundary-evidence-"))
    output.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).resolve().parent / "scratch/async_provider_boundary.rs"
    candidate = json.dumps(str(args.root.resolve() / "crates/consus-hdf5"))
    old = 'moirai-async = { version = "=0.7.0", default-features = false }\n'
    new = 'moirai-async = { version = "0.10.0", git = "https://github.com/ryancinsight/Moirai.git", default-features = false }\n'
    phases = [
        ("published", 'consus-hdf5 = { version = "=0.2.0", features = ["async"] }\n' + old, True),
        ("unchanged-caller", 'consus-hdf5 = { path = ' + candidate + ', features = ["async"] }\n' + old, False),
        ("migrated-caller", 'consus-hdf5 = { path = ' + candidate + ', features = ["async"] }\n' + new, True),
    ]
    for name, dependencies, succeeds in phases:
        with tempfile.TemporaryDirectory(prefix="consus-async-" + name + "-") as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/main.rs").write_bytes(fixture.read_bytes())
            (root / "Cargo.toml").write_text(
                '[package]\nname = "consus-async-boundary"\nversion = "0.0.0"\n'
                'edition = "2024"\n[workspace]\n[dependencies]\n' + dependencies,
                encoding="utf-8",
            )
            result = subprocess.run(["cargo", "build", "--message-format=json"], cwd=root,
                                    capture_output=True, text=True, timeout=300)
            (output / (name + ".stdout.jsonl")).write_text(result.stdout, encoding="utf-8")
            (output / (name + ".stderr.log")).write_text(result.stderr, encoding="utf-8")
            errors = [m["message"] for line in result.stdout.splitlines()
                      if (m := json.loads(line)).get("reason") == "compiler-message"
                      and m["message"]["level"] == "error"]
            codes = [m.get("code", {}).get("code") if m.get("code") else None for m in errors]
            if succeeds:
                assert result.returncode == 0, (name, result.stderr, errors)
                metadata = json.loads(subprocess.check_output(
                    ["cargo", "metadata", "--locked", "--format-version", "1"], cwd=root, timeout=60))
                executable = "consus-async-boundary.exe" if os.name == "nt" else "consus-async-boundary"
                subprocess.run([str(Path(metadata["target_directory"]) / "debug" / executable)],
                               cwd=root, check=True, timeout=60)
            else:
                assert result.returncode == 101 and codes and set(codes) == {"E0277"}, (name, result.stderr, errors)
                rendered = "\n".join(m["rendered"] for m in errors)
                assert "AsyncReadAt" in rendered and "AsyncLength" in rendered, rendered
            print(json.dumps({"phase": name, "exit": result.returncode, "error_codes": codes}), flush=True)
    print("evidence: " + str(output.resolve()))


if __name__ == "__main__":
    main()
