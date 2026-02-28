"""Tests for FlextInfraTextPatternScanner."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

from flext_infra.core.scanner import FlextInfraTextPatternScanner


class TestFlextInfraTextPatternScanner:
    """Test suite for FlextInfraTextPatternScanner."""

    def test_init_creates_service_instance(self) -> None:
        """Test that TextPatternScanner initializes correctly."""
        scanner = FlextInfraTextPatternScanner()
        assert scanner is not None

    def test_scan_with_matching_pattern_returns_success(self, tmp_path: Path) -> None:
        """Test that scan returns success for matching patterns."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        test_file = root / "test.txt"
        test_file.write_text("hello world")

        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
        )

        assert result.is_success
        assert result.value["violation_count"] == 1

    def test_scan_with_no_matches_returns_result(self, tmp_path: Path) -> None:
        """Test that scan returns result when no matches found."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        test_file = root / "test.txt"
        test_file.write_text("goodbye world")

        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
        )

        assert result.is_success
        assert result.value["violation_count"] == 0

    def test_scan_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that scan returns FlextResult type."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path
        test_file = root / "test.txt"
        test_file.write_text("content")

        result = scanner.scan(
            root,
            pattern="content",
            includes=["*.txt"],
        )

        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_scan_with_excludes_filters_files(self, tmp_path: Path) -> None:
        """Test that scan respects exclude patterns."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        included = root / "included.txt"
        included.write_text("hello")
        excluded = root / "excluded.log"
        excluded.write_text("hello")

        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
            excludes=["*.log"],
        )

        assert result.is_success
        assert result.value["files_scanned"] == 1

    def test_scan_with_absent_match_mode_counts_missing(self, tmp_path: Path) -> None:
        """Test scan with absent match mode."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        test_file = root / "test.txt"
        test_file.write_text("goodbye")

        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
            match_mode="absent",
        )

        assert result.is_success
        assert result.value["violation_count"] == 1

    def test_scan_with_absent_match_mode_no_violation_if_found(
        self, tmp_path: Path
    ) -> None:
        """Test absent mode returns no violation if pattern found."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        test_file = root / "test.txt"
        test_file.write_text("hello world")

        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
            match_mode="absent",
        )

        assert result.is_success
        assert result.value["violation_count"] == 0

    def test_scan_with_nonexistent_root_returns_failure(self, tmp_path: Path) -> None:
        """Test scan with nonexistent root directory."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path / "nonexistent"

        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
        )

        assert result.is_failure

    def test_scan_with_empty_includes_returns_failure(self, tmp_path: Path) -> None:
        """Test scan with empty includes list."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        result = scanner.scan(
            root,
            pattern="hello",
            includes=[],
        )

        assert result.is_failure

    def test_scan_with_invalid_match_mode_returns_failure(self, tmp_path: Path) -> None:
        """Test scan with invalid match_mode."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
            match_mode="invalid",
        )

        assert result.is_failure

    def test_scan_with_invalid_regex_returns_failure(self, tmp_path: Path) -> None:
        """Test scan with invalid regex pattern."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        test_file = root / "test.txt"
        test_file.write_text("content")

        result = scanner.scan(
            root,
            pattern="[invalid",
            includes=["*.txt"],
        )

        assert result.is_failure

    def test_scan_with_multiple_files_counts_all_matches(self, tmp_path: Path) -> None:
        """Test scan counts matches across multiple files."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        (root / "file1.txt").write_text("hello world")
        (root / "file2.txt").write_text("hello again")
        (root / "file3.txt").write_text("goodbye")

        result = scanner.scan(
            root,
            pattern="hello",
            includes=["*.txt"],
        )

        assert result.is_success
        assert result.value["match_count"] == 2

    def test_scan_with_multiline_pattern(self, tmp_path: Path) -> None:
        """Test scan with multiline regex pattern."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        test_file = root / "test.txt"
        test_file.write_text("line1\nline2\nline3")

        result = scanner.scan(
            root,
            pattern="^line",
            includes=["*.txt"],
        )

        assert result.is_success
        assert result.value["match_count"] == 3

    def test_scan_with_nested_directories(self, tmp_path: Path) -> None:
        """Test scan finds files in nested directories."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        subdir = root / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("hello")

        result = scanner.scan(
            root,
            pattern="hello",
            includes=["**/*.txt"],
        )

        assert result.is_success
        assert result.value["files_scanned"] == 1

    def test_scan_with_unreadable_file_skips_gracefully(self, tmp_path: Path) -> None:
        """Test scan handles unreadable files gracefully."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        test_file = root / "test.txt"
        test_file.write_text("hello")

        with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
            result = scanner.scan(
                root,
                pattern="hello",
                includes=["*.txt"],
            )

            assert result.is_success

    def test_collect_files_with_glob_patterns(self, tmp_path: Path) -> None:
        """Test _collect_files with glob patterns."""
        root = tmp_path
        (root / "file1.py").write_text("")
        (root / "file2.txt").write_text("")
        (root / "file3.py").write_text("")

        files = FlextInfraTextPatternScanner._collect_files(root, ["*.py"], [])
        assert len(files) == 2

    def test_collect_files_with_exclude_patterns(self, tmp_path: Path) -> None:
        """Test _collect_files respects exclude patterns."""
        root = tmp_path
        (root / "file1.py").write_text("")
        (root / "file2.py").write_text("")
        (root / "test.py").write_text("")

        files = FlextInfraTextPatternScanner._collect_files(root, ["*.py"], ["test*"])
        assert len(files) == 2

    def test_collect_files_skips_directories(self, tmp_path: Path) -> None:
        """Test _collect_files skips directories."""
        root = tmp_path
        (root / "file.txt").write_text("")
        (root / "subdir").mkdir()

        files = FlextInfraTextPatternScanner._collect_files(root, ["*"], [])
        assert len(files) == 1

    def test_count_matches_with_multiple_matches_in_file(self, tmp_path: Path) -> None:
        """Test _count_matches counts all matches in a file."""
        root = tmp_path
        test_file = root / "test.txt"
        test_file.write_text("hello hello hello")

        regex = re.compile(r"hello")
        count = FlextInfraTextPatternScanner._count_matches([test_file], regex)
        assert count == 3

    def test_count_matches_with_empty_file(self, tmp_path: Path) -> None:
        """Test _count_matches with empty file."""
        root = tmp_path
        test_file = root / "test.txt"
        test_file.write_text("")

        regex = re.compile(r"hello")
        count = FlextInfraTextPatternScanner._count_matches([test_file], regex)
        assert count == 0

    def test_count_matches_with_unreadable_file(self, tmp_path: Path) -> None:
        """Test _count_matches skips unreadable files."""
        root = tmp_path
        test_file = root / "test.txt"
        test_file.write_text("hello")

        regex = re.compile(r"hello")
        with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
            count = FlextInfraTextPatternScanner._count_matches([test_file], regex)
            assert count == 0

    def test_scan_with_oserror_returns_failure(self, tmp_path: Path) -> None:
        """Test scan handles OSError exception (lines 84-85)."""
        scanner = FlextInfraTextPatternScanner()
        root = tmp_path

        test_file = root / "test.txt"
        test_file.write_text("hello")

        # Mock _collect_files to raise OSError
        with patch.object(
            FlextInfraTextPatternScanner,
            "_collect_files",
            side_effect=OSError("permission denied"),
        ):
            result = scanner.scan(
                root,
                pattern="hello",
                includes=["*.txt"],
            )
            assert result.is_failure
            assert "text pattern scan failed" in result.error
