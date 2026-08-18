import json
import tempfile
import unittest
from pathlib import Path

from check_s3_benchmark import compare_reports


class S3BenchmarkGateTests(unittest.TestCase):
    def write_report(self, root: Path, cell: str, backend: str, median_ns: float) -> None:
        report_dir = root / cell / backend
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "estimates.json").write_text(
            json.dumps({"median": {"point_estimate": median_ns}}),
            encoding="utf-8",
        )

    def test_accepts_exactly_ninety_percent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_report(root, ".", "native_moirai", 100.0)
            self.write_report(root, ".", "legacy_rusoto", 90.0)

            rows = compare_reports(root)

            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["native_throughput_ratio"], 0.90)

    def test_rejects_cell_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_report(root, ".", "native_moirai", 101.0)
            self.write_report(root, ".", "legacy_rusoto", 90.0)

            with self.assertRaisesRegex(ValueError, "gate failed"):
                compare_reports(root)

    def test_requires_both_backends_for_every_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_report(root, "small", "native_moirai", 100.0)
            self.write_report(root, "small", "legacy_rusoto", 100.0)
            self.write_report(root, "large", "native_moirai", 100.0)

            with self.assertRaisesRegex(ValueError, "legacy_rusoto"):
                compare_reports(root)

    def test_rejects_invalid_median(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_report(root, ".", "native_moirai", 0.0)
            self.write_report(root, ".", "legacy_rusoto", 100.0)

            with self.assertRaisesRegex(ValueError, "finite and positive"):
                compare_reports(root)


if __name__ == "__main__":
    unittest.main()
