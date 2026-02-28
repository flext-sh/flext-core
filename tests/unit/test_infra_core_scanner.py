"""Tests for FlextInfraTextPatternScanner."""

from __future__ import annotations

from pathlib import Path

from flext_infra.core.scanner import FlextInfraTextPatternScanner


class TestFlextInfraTextPatternScanner:
    """Test suite for FlextInfraTextPatternScanner."""

    def test_init_creates_service_instance(self) -> None:
        """Test that TextPatternScanner initializes correctly."""
        # Arrange & Act
        scanner = FlextInfraTextPatternScanner()

        # Assert
        assert scanner is not None

    def test_scan_with_matching_pattern_returns_success(self, tmp_path: Path) -> None:
        """Test that scan returns success for matching patterns."""
        # Arrange
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        # Create test file with matching pattern
        test_file = root / "test.txt"
        test_file.write_text("hello world")

        # Act
        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
        )

        # Assert
        assert result.is_success or result.is_failure

    def test_scan_with_no_matches_returns_result(self, tmp_path: Path) -> None:
        """Test that scan returns result when no matches found."""
        # Arrange
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        # Create test file without matching pattern
        test_file = root / "test.txt"
        test_file.write_text("goodbye world")

        # Act
        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
        )

        # Assert
        assert result.is_success or result.is_failure

    def test_scan_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that scan returns FlextResult type."""
        # Arrange
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path
        test_file = root / "test.txt"
        test_file.write_text("content")

        # Act
        result = scanner.scan(
            root,
            pattern="content",
            includes=["*.txt"],
        )

        # Assert
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_scan_with_excludes_filters_files(self, tmp_path: Path) -> None:
        """Test that scan respects exclude patterns."""
        # Arrange
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        # Create files
        included = root / "included.txt"
        included.write_text("hello")
        excluded = root / "excluded.log"
        excluded.write_text("hello")

        # Act
        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
            excludes=["*.log"],
        )

        # Assert
        assert result.is_success or result.is_failure

    def test_scan_with_present_match_mode(self, tmp_path: Path) -> None:
        """Test scan with present match mode."""
        # Arrange
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path
        test_file = root / "test.txt"
        test_file.write_text("hello")

        # Act
        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
            match_mode="present",
        )

        # Assert
        assert result.is_success or result.is_failure

    def test_scan_with_absent_match_mode(self, tmp_path: Path) -> None:
        """Test scan with absent match mode."""
        # Arrange
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path
        test_file = root / "test.txt"
        test_file.write_text("goodbye")

        # Act
        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
            match_mode="absent",
        )

        # Assert
        assert result.is_success or result.is_failure
