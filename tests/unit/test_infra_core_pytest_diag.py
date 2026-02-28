"""Tests for FlextInfraPytestDiagExtractor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from flext_infra.core.pytest_diag import (
    FlextInfraPytestDiagExtractor,
    _DiagResult,
)


class TestDiagResult:
    """Test _DiagResult internal container."""

    def test_diag_result_init(self) -> None:
        """Test _DiagResult initializes correctly."""
        diag = _DiagResult()
        assert diag.failed_cases == []
        assert diag.error_traces == []
        assert diag.skip_cases == []
        assert diag.warning_lines == []
        assert diag.slow_entries == []

    def test_diag_result_has_slots(self) -> None:
        """Test _DiagResult uses __slots__."""
        _DiagResult()
        assert hasattr(_DiagResult, "__slots__")
        assert len(_DiagResult.__slots__) == 5


class TestFlextInfraPytestDiagExtractor:
    """Test suite for FlextInfraPytestDiagExtractor."""

    def test_init_creates_service_instance(self) -> None:
        """Test that PytestDiagExtractor initializes correctly."""
        extractor = FlextInfraPytestDiagExtractor()
        assert extractor is not None

    def test_extract_with_valid_junit_xml_returns_success(self, tmp_path: Path) -> None:
        """Test that extract returns success for valid JUnit XML."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="0" errors="0" skipped="0"></testsuite></testsuites>'
        )
        log_path = tmp_path / "log.txt"
        log_path.write_text("")

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success
        assert isinstance(result.value, dict)

    def test_extract_with_missing_junit_xml_uses_log(self, tmp_path: Path) -> None:
        """Test extract falls back to log parsing when XML missing."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        log_path = tmp_path / "log.txt"
        log_path.write_text("FAILED test_case.py::test_foo")

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success

    def test_extract_with_failed_tests_reports_failures(self, tmp_path: Path) -> None:
        """Test that extract reports failed tests."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="1" errors="0" skipped="0"><testcase name="test_fail" classname="TestClass">'
            '<failure message="assertion failed">Traceback here</failure></testcase></testsuite></testsuites>'
        )
        log_path = tmp_path / "log.txt"
        log_path.write_text("")

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success
        assert result.value["failed_count"] == 1
        assert len(result.value["error_traces"]) > 0

    def test_extract_with_error_tests_reports_errors(self, tmp_path: Path) -> None:
        """Test that extract reports error tests."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="0" errors="1" skipped="0"><testcase name="test_error" classname="TestClass">'
            '<error message="runtime error">Error trace</error></testcase></testsuite></testsuites>'
        )
        log_path = tmp_path / "log.txt"
        log_path.write_text("")

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success
        assert result.value["error_count"] == 1

    def test_extract_with_skipped_tests_reports_skips(self, tmp_path: Path) -> None:
        """Test that extract reports skipped tests."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="0" errors="0" skipped="1"><testcase name="test_skip" classname="TestClass">'
            '<skipped message="skipped test"/></testcase></testsuite></testsuites>'
        )
        log_path = tmp_path / "log.txt"
        log_path.write_text("")

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success
        assert result.value["skipped_count"] == 1

    def test_extract_with_timing_data_extracts_slow_tests(self, tmp_path: Path) -> None:
        """Test that extract extracts slow test timings."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="2">'
            '<testcase name="test_fast" time="0.1"/>'
            '<testcase name="test_slow" time="5.5"/>'
            "</testsuite></testsuites>"
        )
        log_path = tmp_path / "log.txt"
        log_path.write_text("")

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success
        assert len(result.value["slow_entries"]) > 0

    def test_extract_with_invalid_junit_xml_uses_log_fallback(
        self, tmp_path: Path
    ) -> None:
        """Test extract falls back to log when XML is invalid."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text("invalid xml content")
        log_path = tmp_path / "log.txt"
        log_path.write_text("FAILED test_case.py::test_foo")

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success

    def test_extract_with_missing_log_file_handles_gracefully(
        self, tmp_path: Path
    ) -> None:
        """Test extract handles missing log file."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="0"/></testsuites>'
        )
        log_path = tmp_path / "missing.txt"

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success

    def test_extract_with_exception_returns_failure(self, tmp_path: Path) -> None:
        """Test extract handles exceptions."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        log_path = tmp_path / "log.txt"

        with patch.object(
            extractor, "_parse_xml", side_effect=ValueError("test error")
        ):
            result = extractor.extract(junit_xml, log_path)
            assert result.is_failure

    def test_parse_xml_with_missing_file_returns_false(self, tmp_path: Path) -> None:
        """Test _parse_xml returns False for missing file."""
        diag = _DiagResult()
        result = FlextInfraPytestDiagExtractor._parse_xml(
            tmp_path / "missing.xml", diag
        )
        assert result is False

    def test_parse_xml_with_invalid_xml_returns_false(self, tmp_path: Path) -> None:
        """Test _parse_xml returns False for invalid XML."""
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text("not valid xml")
        diag = _DiagResult()
        result = FlextInfraPytestDiagExtractor._parse_xml(junit_xml, diag)
        assert result is False

    def test_parse_xml_with_none_root_returns_false(self, tmp_path: Path) -> None:
        """Test _parse_xml returns False when root is None."""
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text('<?xml version="1.0"?>')
        diag = _DiagResult()
        with patch("flext_infra.core.pytest_diag.DefusedET.parse") as mock_parse:
            mock_parse.return_value.getroot.return_value = None
            result = FlextInfraPytestDiagExtractor._parse_xml(junit_xml, diag)
            assert result is False

    def test_parse_xml_extracts_test_timing(self, tmp_path: Path) -> None:
        """Test _parse_xml extracts test timing data."""
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test">'
            '<testcase name="test_a" classname="TestClass" time="1.5"/>'
            '<testcase name="test_b" classname="TestClass" time="0.5"/>'
            "</testsuite></testsuites>"
        )
        diag = _DiagResult()
        result = FlextInfraPytestDiagExtractor._parse_xml(junit_xml, diag)
        assert result is True
        assert len(diag.slow_entries) == 2

    def test_parse_xml_handles_invalid_time_attribute(self, tmp_path: Path) -> None:
        """Test _parse_xml handles invalid time attribute."""
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test">'
            '<testcase name="test_a" classname="TestClass" time="invalid"/>'
            "</testsuite></testsuites>"
        )
        diag = _DiagResult()
        result = FlextInfraPytestDiagExtractor._parse_xml(junit_xml, diag)
        assert result is True

    def test_parse_log_into_diag_extracts_failures(self, tmp_path: Path) -> None:
        """Test _parse_log_into_diag extracts failures."""
        lines = [
            "FAILED test_case.py::test_foo",
            "PASSED test_case.py::test_bar",
        ]
        diag = _DiagResult()
        FlextInfraPytestDiagExtractor._parse_log_into_diag(lines, diag)
        assert len(diag.failed_cases) > 0

    def test_parse_log_into_diag_extracts_skips(self, tmp_path: Path) -> None:
        """Test _parse_log_into_diag extracts skips."""
        lines = [
            "SKIPPED test_case.py::test_skip",
            "PASSED test_case.py::test_pass",
        ]
        diag = _DiagResult()
        FlextInfraPytestDiagExtractor._parse_log_into_diag(lines, diag)
        assert len(diag.skip_cases) > 0

    def test_parse_log_into_diag_extracts_error_block(self, tmp_path: Path) -> None:
        """Test _parse_log_into_diag extracts error block."""
        lines = [
            "=== FAILURES ===",
            "test_case.py::test_foo",
            "AssertionError: expected True",
            "=== short test summary info ===",
        ]
        diag = _DiagResult()
        FlextInfraPytestDiagExtractor._parse_log_into_diag(lines, diag)
        assert len(diag.error_traces) > 0

    def test_extract_warnings_with_warnings_section(self, tmp_path: Path) -> None:
        """Test _extract_warnings extracts warnings section."""
        lines = [
            "=== warnings summary ===",
            "DeprecationWarning: test warning",
            "-- Docs: https://docs.pytest.org/",
        ]
        diag = _DiagResult()
        FlextInfraPytestDiagExtractor._extract_warnings(lines, diag)
        assert len(diag.warning_lines) > 0

    def test_extract_warnings_with_inline_warnings(self, tmp_path: Path) -> None:
        """Test _extract_warnings extracts inline warnings."""
        lines = [
            "test_case.py:10: DeprecationWarning: test",
            "test_case.py:20: UserWarning: another",
        ]
        diag = _DiagResult()
        FlextInfraPytestDiagExtractor._extract_warnings(lines, diag)
        assert len(diag.warning_lines) > 0

    def test_extract_slow_from_log_extracts_durations(self, tmp_path: Path) -> None:
        """Test _extract_slow_from_log extracts slow test durations."""
        lines = [
            "=== slowest durations ===",
            "5.50s call     test_case.py::test_slow",
            "0.50s call     test_case.py::test_fast",
            "=== 2 passed in 6.00s ===",
        ]
        diag = _DiagResult()
        FlextInfraPytestDiagExtractor._extract_slow_from_log(lines, diag)
        assert len(diag.slow_entries) > 0

    def test_extract_returns_all_diagnostic_fields(self, tmp_path: Path) -> None:
        """Test extract returns all diagnostic fields."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="0"/></testsuites>'
        )
        log_path = tmp_path / "log.txt"
        log_path.write_text("")

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success
        value = result.value
        assert "failed_count" in value
        assert "error_count" in value
        assert "warning_count" in value
        assert "skipped_count" in value
        assert "failed_cases" in value
        assert "error_traces" in value
        assert "warning_lines" in value
        assert "skip_cases" in value
        assert "slow_entries" in value
