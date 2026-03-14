from __future__ import annotations

from pathlib import Path

from flext_infra import c, m, p, u


class MockScanner:
    def __init__(self, *, detector_name: str) -> None:
        self.detector_name = detector_name
        self.scanned: list[Path] = []

    def scan_file(self, *, file_path: Path) -> m.Infra.Utilities.ScanResult:
        self.scanned.append(file_path)
        return m.Infra.Utilities.ScanResult(
            file_path=file_path,
            detector_name=self.detector_name,
            violations=[
                m.Infra.Utilities.ScanViolation(
                    line=1,
                    message=f"checked {file_path.name}",
                    severity="low",
                    rule_id="demo-rule",
                )
            ],
        )


class TestScanFileBatch:
    def test_scan_files_batch_runs_scanner_for_all_files(self, tmp_path: Path) -> None:
        scanner = MockScanner(detector_name="mock-scanner")
        assert isinstance(scanner, p.Infra.Scanner)

        file_one = tmp_path / "a.py"
        file_two = tmp_path / "b.py"
        file_one.write_text("print('a')\n", encoding="utf-8")
        file_two.write_text("print('b')\n", encoding="utf-8")

        results = u.Infra.scan_files_batch(scanner=scanner, files=[file_one, file_two])

        assert scanner.scanned == [file_one, file_two]
        assert len(results) == 2
        assert results[0].file_path == file_one
        assert results[1].file_path == file_two
        assert results[0].detector_name == "mock-scanner"

    def test_scan_files_batch_returns_empty_list_for_empty_input(self) -> None:
        scanner = MockScanner(detector_name="mock-scanner")

        results = u.Infra.scan_files_batch(scanner=scanner, files=[])

        assert results == []
        assert scanner.scanned == []


class TestScanModels:
    def test_scan_violation_model_fields(self) -> None:
        violation = m.Infra.Utilities.ScanViolation(
            line=42,
            message="forbidden import",
            severity="high",
            rule_id="FLEXT001",
        )

        assert violation.line == 42
        assert violation.message == "forbidden import"
        assert violation.severity == "high"
        assert violation.rule_id == "FLEXT001"

    def test_scan_result_model_fields_and_defaults(self, tmp_path: Path) -> None:
        result = m.Infra.Utilities.ScanResult(
            file_path=tmp_path / "sample.py",
            detector_name="scanner-x",
            violations=[],
        )
        payload = result.model_dump()

        assert result.file_path == tmp_path / "sample.py"
        assert result.detector_name == "scanner-x"
        assert payload["violations"] == []

    def test_scan_result_with_violations(self, tmp_path: Path) -> None:
        violation = m.Infra.Utilities.ScanViolation(
            line=7,
            message="rule hit",
            severity="medium",
            rule_id=None,
        )
        result = m.Infra.Utilities.ScanResult(
            file_path=tmp_path / "violating.py",
            detector_name="scanner-y",
            violations=[violation],
        )
        payload = result.model_dump()

        violations = payload["violations"]
        assert len(violations) == 1
        assert violations[0]["message"] == "rule hit"
        assert violations[0]["rule_id"] is None
        assert c.Infra.Cli.GIT == "git"
