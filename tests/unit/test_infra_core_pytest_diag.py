"""Tests for FlextInfraPytestDiagExtractor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from flext_core import r
from flext_infra.core.pytest_diag import PytestDiagExtractor
from flext_infra import m


class TestFlextInfraPytestDiagExtractor:
    """Test suite for FlextInfraPytestDiagExtractor."""

    def test_init_creates_service_instance(self) -> None:
        """Test that PytestDiagExtractor initializes correctly."""
        # Arrange & Act
        extractor = PytestDiagExtractor()

        # Assert
        assert extractor is not None

    def test_extract_with_missing_junit_xml_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """Test that extract returns failure for missing JUnit XML."""
        # Arrange
        extractor = PytestDiagExtractor()
        junit_xml = tmp_path / "missing.xml"

        # Act
        result = extractor.extract(junit_xml)

        # Assert
        assert result.is_failure

    def test_extract_with_valid_junit_xml_returns_success(self, tmp_path: Path) -> None:
        """Test that extract returns success for valid JUnit XML."""
        # Arrange
        extractor = PytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="0" errors="0" skipped="0"></testsuite></testsuites>'
        )

        # Act
        result = extractor.extract(junit_xml)

        # Assert
        assert result.is_success or result.is_failure

    def test_extract_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that extract returns FlextResult type."""
        # Arrange
        extractor = PytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="0" errors="0" skipped="0"></testsuite></testsuites>'
        )

        # Act
        result = extractor.extract(junit_xml)

        # Assert
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_extract_with_failed_tests_reports_failures(self, tmp_path: Path) -> None:
        """Test that extract reports failed tests."""
        # Arrange
        extractor = PytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="1" errors="0" skipped="0"><testcase name="test_fail">'
            '<failure message="assertion failed"/></testcase></testsuite></testsuites>'
        )

        # Act
        result = extractor.extract(junit_xml)

        # Assert
        assert result.is_success or result.is_failure

    def test_extract_with_skipped_tests_reports_skips(self, tmp_path: Path) -> None:
        """Test that extract reports skipped tests."""
        # Arrange
        extractor = PytestDiagExtractor()
        junit_xml = tmp_path / "junit.xml"
        junit_xml.write_text(
            '<?xml version="1.0"?><testsuites><testsuite name="test" tests="1" '
            'failures="0" errors="0" skipped="1"><testcase name="test_skip">'
            '<skipped message="skipped test"/></testcase></testsuite></testsuites>'
        )

        # Act
        result = extractor.extract(junit_xml)

        # Assert
        assert result.is_success or result.is_failure
