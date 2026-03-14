"""Tests for FlextInfraPytestDiagExtractor.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra.core import pytest_diag
from flext_infra.core.pytest_diag import FlextInfraPytestDiagExtractor
from flext_tests import tm
from tests.infra.models import m


class TestPytestDiagExtractorCore:
    """Core extraction and _DiagResult tests."""

    def test_diag_result_init(self) -> None:
        """_DiagResult initializes with empty lists."""
        diag = pytest_diag._DiagResult()
        for attr in (
            "failed_cases",
            "error_traces",
            "skip_cases",
            "warning_lines",
            "slow_entries",
        ):
            tm.that(getattr(diag, attr), eq=[])

    def test_extract_valid_junit_xml(self, tmp_path: Path) -> None:
        """Valid JUnit XML returns success with diagnostics model."""
        ext = FlextInfraPytestDiagExtractor()
        junit = tmp_path / "junit.xml"
        junit.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t" tests="1"'
            ' failures="0" errors="0" skipped="0"></testsuite></testsuites>',
        )
        log = tmp_path / "log.txt"
        log.write_text("")
        report = tm.ok(ext.extract(junit, log))
        tm.that(report, is_=m.Infra.Core.PytestDiagnostics)

    def test_extract_fallback_to_log(self, tmp_path: Path) -> None:
        """Missing/invalid XML falls back to log parsing."""
        ext = FlextInfraPytestDiagExtractor()
        log = tmp_path / "log.txt"
        log.write_text("FAILED test_case.py::test_foo")
        tm.ok(ext.extract(tmp_path / "missing.xml", log))
        (tmp_path / "bad.xml").write_text("invalid xml content")
        tm.ok(ext.extract(tmp_path / "bad.xml", log))

    def test_extract_failed_and_error_tests(self, tmp_path: Path) -> None:
        """Failed and error tests are reported."""
        ext = FlextInfraPytestDiagExtractor()
        log = tmp_path / "log.txt"
        log.write_text("")
        (tmp_path / "fail.xml").write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t" tests="1"'
            ' failures="1" errors="0" skipped="0"><testcase name="test_fail"'
            ' classname="TC"><failure message="fail">Traceback</failure>'
            "</testcase></testsuite></testsuites>",
        )
        report = tm.ok(ext.extract(tmp_path / "fail.xml", log))
        tm.that(report.failed_count, eq=1)
        tm.that(report.error_traces, length_gt=0)
        (tmp_path / "err.xml").write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t" tests="1"'
            ' failures="0" errors="1" skipped="0"><testcase name="test_err"'
            ' classname="TC"><error message="err">Trace</error>'
            "</testcase></testsuite></testsuites>",
        )
        tm.that(tm.ok(ext.extract(tmp_path / "err.xml", log)).error_count, eq=1)

    def test_extract_skipped_and_slow(self, tmp_path: Path) -> None:
        """Skipped tests and slow timings are extracted."""
        ext = FlextInfraPytestDiagExtractor()
        log = tmp_path / "log.txt"
        log.write_text("")
        skip_xml = tmp_path / "skip.xml"
        skip_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t" tests="1"'
            ' failures="0" errors="0" skipped="1"><testcase name="test_skip"'
            ' classname="TC"><skipped message="skip"/>'
            "</testcase></testsuite></testsuites>",
        )
        tm.that(tm.ok(ext.extract(skip_xml, log)).skipped_count, eq=1)
        slow_xml = tmp_path / "slow.xml"
        slow_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="t" tests="2">'
            '<testcase name="fast" time="0.1"/><testcase name="slow" time="5.5"/>'
            "</testsuite></testsuites>",
        )
        tm.that(tm.ok(ext.extract(slow_xml, log)).slow_entries, length_gt=0)

    def test_extract_missing_log(self, tmp_path: Path) -> None:
        """Missing log file is handled gracefully."""
        junit = tmp_path / "junit.xml"
        junit.write_text(
            '<?xml version="1.0"?><testsuites>'
            '<testsuite name="t" tests="0"/></testsuites>',
        )
        tm.ok(FlextInfraPytestDiagExtractor().extract(junit, tmp_path / "missing.txt"))


class TestPytestDiagParseXml:
    """Tests for _parse_xml static method."""

    def test_parse_xml_missing_or_invalid(self, tmp_path: Path) -> None:
        """Missing/invalid file returns False."""
        diag = pytest_diag._DiagResult()
        parse = FlextInfraPytestDiagExtractor._parse_xml
        tm.that(parse(tmp_path / "missing.xml", diag), eq=False)
        (tmp_path / "bad.xml").write_text("not valid xml")
        tm.that(parse(tmp_path / "bad.xml", diag), eq=False)

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
        warn_lines = [
            "=== warnings summary ===",
            "DeprecationWarning: test warning",
            "-- Docs: https://docs.pytest.org/",
        ]
        FlextInfraPytestDiagExtractor._extract_warnings(warn_lines, diag)
        tm.that(diag.warning_lines, length_gt=0)
        diag2 = pytest_diag._DiagResult()
        FlextInfraPytestDiagExtractor._extract_warnings(
            ["test_case.py:10: DeprecationWarning: test"],
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
