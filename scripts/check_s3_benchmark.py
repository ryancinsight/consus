#!/usr/bin/env python3
"""Enforce the ADR-0045 native S3 throughput qualification gate.

Criterion stores the median estimate as elapsed time in nanoseconds.  The
native and legacy cases read the same byte count, so native throughput divided
by legacy throughput is exactly::

    legacy_median_time / native_median_time

The parser compares every matching ``native_moirai``/``legacy_rusoto`` pair
under the supplied Criterion report directory.  It deliberately reads
``estimates.json`` rather than Criterion's HTML, which keeps the CI gate
machine-readable and independent of report formatting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

NATIVE_CASE = "native_moirai"
LEGACY_CASE = "legacy_rusoto"
DEFAULT_THRESHOLD = 0.90


def _cell_for(path: Path, report_dir: Path) -> str:
    """Return the benchmark-cell directory containing a backend report."""
    relative_backend = path.parent.relative_to(report_dir)
    cell = relative_backend.parent
    return "." if str(cell) == "." else cell.as_posix()


def _median_ns(path: Path) -> float:
    """Read and validate Criterion's median point estimate."""
    try:
        with path.open(encoding="utf-8") as report:
            document: Any = json.load(report)
        value = document["median"]["point_estimate"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid Criterion estimates file {path}: {error}") from error

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: median.point_estimate must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}: median.point_estimate must be finite and positive")
    return value


def _reports(report_dir: Path) -> dict[str, dict[str, Path]]:
    """Find backend reports grouped by benchmark cell."""
    if not report_dir.is_dir():
        raise ValueError(f"Criterion report directory does not exist: {report_dir}")

    reports: dict[str, dict[str, Path]] = {}
    for path in report_dir.rglob("estimates.json"):
        backend = path.parent.name
        if backend not in (NATIVE_CASE, LEGACY_CASE):
            continue
        cell = _cell_for(path, report_dir)
        reports.setdefault(cell, {})[backend] = path

    if not reports:
        raise ValueError(
            f"no {NATIVE_CASE}/{LEGACY_CASE} Criterion reports found under {report_dir}"
        )
    return reports


def compare_reports(report_dir: Path, threshold: float = DEFAULT_THRESHOLD) -> list[dict[str, float | str]]:
    """Return comparison rows, raising ``ValueError`` for an invalid gate."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("the native-throughput threshold must be finite and positive")

    rows: list[dict[str, float | str]] = []
    for cell, backends in sorted(_reports(report_dir).items()):
        missing = [name for name in (NATIVE_CASE, LEGACY_CASE) if name not in backends]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"benchmark cell {cell!r} is missing: {names}")

        native_ns = _median_ns(backends[NATIVE_CASE])
        legacy_ns = _median_ns(backends[LEGACY_CASE])
        ratio = legacy_ns / native_ns
        rows.append(
            {
                "cell": cell,
                "native_median_ns": native_ns,
                "legacy_median_ns": legacy_ns,
                "native_throughput_ratio": ratio,
                "threshold": threshold,
            }
        )

    failures = [row for row in rows if row["native_throughput_ratio"] < threshold]
    if failures:
        details = "; ".join(
            f"{row['cell']}: {row['native_throughput_ratio']:.4f} < {threshold:.4f}"
            for row in failures
        )
        raise ValueError(f"ADR-0045 native-throughput gate failed: {details}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report_dir",
        type=Path,
        help="Criterion report directory, usually target/criterion/s3_range_read",
    )
    parser.add_argument(
        "--min-native-throughput-ratio",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="minimum native/legacy throughput ratio (default: 0.90)",
    )
    args = parser.parse_args(argv)

    try:
        rows = compare_reports(args.report_dir, args.min_native_throughput_ratio)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for row in rows:
        print(
            f"PASS cell={row['cell']} "
            f"native_median_ns={row['native_median_ns']:.3f} "
            f"legacy_median_ns={row['legacy_median_ns']:.3f} "
            f"native_throughput_ratio={row['native_throughput_ratio']:.4f} "
            f"threshold={row['threshold']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
