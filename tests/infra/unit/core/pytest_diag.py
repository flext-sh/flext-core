"""Tests for FlextInfraPytestDiagExtractor.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra import m
from flext_infra.core import pytest_diag
from flext_infra.core.pytest_diag import FlextInfraPytestDiagExtractor
from flext_tests import tm


class TestDiagResult:
    """Test _DiagResult internal container."""

    def test_diag_result_init(self) -> None:
        """_DiagResult initializes with empty lists."""
        diag = pytest_diag._DiagResult()
        tm.that(diag.failed_cases, eq=[])
        tm.that(diag.error_traces, eq=[])
        tm.that(diag.skip_cases, eq=[])
        tm.that(diag.warning_lines, eq=[])
        tm.that(diag.slow_entries, eq=[])


class TestPytestDiagExtractorCore:
    """Core extraction tests for FlextInfraPytestDiagExtractor."""

    def test_extract_valid_junit_xml(self, tmp_path: Path) -> None:
        """Valid JUnit XML returns success with diagnostics model."""
        extractor = FlextInfraPytestDiagExtractor()
        junit = tmp_path / "junit.xml"
        junit.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t" tests="1"'
            ' failures="0" errors="0" skipped="0"></testsuite></testsuites>',
        )
        log = tmp_path / "log.txt"
        log.write_text("")
        report = tm.ok(extractor.extract(junit, log))
        tm.that(isinstance(report, m.Infra.Core.PytestDiagnostics), eq=True)

    def test_extract_fallback_to_log(self, tmp_path: Path) -> None:
        """Missing/invalid XML falls back to log parsing."""
        extractor = FlextInfraPytestDiagExtractor()
        log = tmp_path / "log.txt"
        log.write_text("FAILED test_case.py::test_foo")
        tm.ok(extractor.extract(tmp_path / "missing.xml", log))
        invalid_xml = tmp_path / "bad.xml"
        invalid_xml.write_text("invalid xml content")
        tm.ok(extractor.extract(invalid_xml, log))

    def test_extract_failed_and_error_tests(self, tmp_path: Path) -> None:
        """Failed and error tests are reported."""
        extractor = FlextInfraPytestDiagExtractor()
        log = tmp_path / "log.txt"
        log.write_text("")
        fail_xml = tmp_path / "fail.xml"
        fail_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t" tests="1"'
            ' failures="1" errors="0" skipped="0"><testcase name="test_fail"'
            ' classname="TC"><failure message="fail">Traceback</failure>'
            "</testcase></testsuite></testsuites>",
        )
        report = tm.ok(extractor.extract(fail_xml, log))
        tm.that(report.failed_count, eq=1)
        tm.that(report.error_traces, length_gt=0)
        err_xml = tmp_path / "err.xml"
        err_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t" tests="1"'
            ' failures="0" errors="1" skipped="0"><testcase name="test_err"'
            ' classname="TC"><error message="err">Trace</error>'
            "</testcase></testsuite></testsuites>",
        )
        report2 = tm.ok(extractor.extract(err_xml, log))
        tm.that(report2.error_count, eq=1)

    def test_extract_skipped_tests(self, tmp_path: Path) -> None:
        """Skipped tests are reported."""
        extractor = FlextInfraPytestDiagExtractor()
        junit = tmp_path / "junit.xml"
        junit.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t" tests="1"'
            ' failures="0" errors="0" skipped="1"><testcase name="test_skip"'
            ' classname="TC"><skipped message="skip"/>'
            "</testcase></testsuite></testsuites>",
        )
        log = tmp_path / "log.txt"
        log.write_text("")
        report = tm.ok(extractor.extract(junit, log))
        tm.that(report.skipped_count, eq=1)

    def test_extract_slow_tests(self, tmp_path: Path) -> None:
        """Slow test timings are extracted."""
        extractor = FlextInfraPytestDiagExtractor()
        junit = tmp_path / "junit.xml"
        junit.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t" tests="2">'
            '<testcase name="fast" time="0.1"/>'
            '<testcase name="slow" time="5.5"/>'
            "</testsuite></testsuites>",
        )
        log = tmp_path / "log.txt"
        log.write_text("")
        report = tm.ok(extractor.extract(junit, log))
        tm.that(report.slow_entries, length_gt=0)

    def test_extract_missing_log(self, tmp_path: Path) -> None:
        """Missing log file is handled gracefully."""
        extractor = FlextInfraPytestDiagExtractor()
        junit = tmp_path / "junit.xml"
        junit.write_text(
            '<?xml version="1.0"?><testsuites>'
            '<testsuite name="t" tests="0"/></testsuites>',
        )
        tm.ok(extractor.extract(junit, tmp_path / "missing.txt"))


class TestPytestDiagParseXml:
    """Tests for _parse_xml static method."""

    def test_parse_xml_missing_or_invalid(self, tmp_path: Path) -> None:
        """Missing/invalid file returns False."""
        diag = pytest_diag._DiagResult()
        tm.that(
            FlextInfraPytestDiagExtractor._parse_xml(
                tmp_path / "missing.xml",
                diag,
            ),
            eq=False,
        )
        (tmp_path / "bad.xml").write_text("not valid xml")
        tm.that(
            FlextInfraPytestDiagExtractor._parse_xml(
                tmp_path / "bad.xml",
                diag,
            ),
            eq=False,
        )

    def test_parse_xml_extracts_timing(self, tmp_path: Path) -> None:
        """Test timing data is extracted."""
        junit = tmp_path / "junit.xml"
        junit.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t">'
            '<testcase name="a" classname="TC" time="1.5"/>'
            '<testcase name="b" classname="TC" time="0.5"/>'
            "</testsuite></testsuites>",
        )
        diag = pytest_diag._DiagResult()
        tm.that(FlextInfraPytestDiagExtractor._parse_xml(junit, diag), eq=True)
        tm.that(diag.slow_entries, length=2)

    def test_parse_xml_invalid_time(self, tmp_path: Path) -> None:
        """Invalid time attribute is handled."""
        junit = tmp_path / "junit.xml"
        junit.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t">'
            '<testcase name="a" classname="TC" time="invalid"/>'
            "</testsuite></testsuites>",
        )
        diag = pytest_diag._DiagResult()
        tm.that(FlextInfraPytestDiagExtractor._parse_xml(junit, diag), eq=True)


class TestPytestDiagLogParsing:
    """Tests for log parsing methods."""

    def test_parse_log_failures_and_skips(self) -> None:
        """Failures and skips are extracted from log lines."""
        diag = pytest_diag._DiagResult()
        FlextInfraPytestDiagExtractor._parse_log_into_diag(
            ["FAILED test_case.py::test_foo", "SKIPPED test_case.py::test_skip"],
            diag,
        )
        tm.that(diag.failed_cases, length_gt=0)
        tm.that(diag.skip_cases, length_gt=0)

    def test_parse_log_error_block(self) -> None:
        """Error blocks are extracted from log lines."""
        lines = [
            "=== FAILURES ===",
            "test_case.py::test_foo",
            "AssertionError: expected True",
            "=== short test summary info ===",
        ]
        diag = pytest_diag._DiagResult()
        FlextInfraPytestDiagExtractor._parse_log_into_diag(lines, diag)
        tm.that(diag.error_traces, length_gt=0)

    def test_extract_warnings(self) -> None:
        """Warnings section and inline warnings are extracted."""
        diag = pytest_diag._DiagResult()
        FlextInfraPytestDiagExtractor._extract_warnings(
            [
                "=== warnings summary ===",
                "DeprecationWarning: test warning",
                "-- Docs: https://docs.pytest.org/",
            ],
            diag,
        )
        tm.that(diag.warning_lines, length_gt=0)
        diag2 = pytest_diag._DiagResult()
        FlextInfraPytestDiagExtractor._extract_warnings(
            [
                "test_case.py:10: DeprecationWarning: test",
            ],
            diag2,
        )
        tm.that(diag2.warning_lines, length_gt=0)

    def test_extract_slow_from_log(self) -> None:
        """Slow test durations are extracted from log."""
        lines = [
            "=== slowest durations ===",
            "5.50s call     test_case.py::test_slow",
            "0.50s call     test_case.py::test_fast",
            "=== 2 passed in 6.00s ===",
        ]
        diag = pytest_diag._DiagResult()
        FlextInfraPytestDiagExtractor._extract_slow_from_log(lines, diag)
        tm.that(diag.slow_entries, length_gt=0)


__all__: list[str] = []
