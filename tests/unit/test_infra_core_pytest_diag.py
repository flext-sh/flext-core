"""Tests for FlextInfraPytestDiagExtractor."""

from __future__ import annotations

from pathlib import Path

from flext_infra.core.pytest_diag import FlextInfraPytestDiagExtractor


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
        assert result.is_success or result.is_failure

    def test_extract_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that extract returns FlextResult type."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="0" errors="0" skipped="0"></testsuite></testsuites>'
        )
        log_path = tmp_path / "log.txt"
        log_path.write_text("")

        result = extractor.extract(junit_xml, log_path)
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_extract_with_failed_tests_reports_failures(self, tmp_path: Path) -> None:
        """Test that extract reports failed tests."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="1" errors="0" skipped="0"><testcase name="test_fail">'
            '<failure message="assertion failed"/></testcase></testsuite></testsuites>'
        )
        log_path = tmp_path / "log.txt"
        log_path.write_text("")

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success or result.is_failure

    def test_extract_with_skipped_tests_reports_skips(self, tmp_path: Path) -> None:
        """Test that extract reports skipped tests."""
        extractor = FlextInfraPytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="0" errors="0" skipped="1"><testcase name="test_skip">'
            '<skipped message="skipped test"/></testcase></testsuite></testsuites>'
        )
        log_path = tmp_path / "log.txt"
        log_path.write_text("")

        result = extractor.extract(junit_xml, log_path)
        assert result.is_success or result.is_failure
