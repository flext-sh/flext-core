"""Tests for FlextInfraTextPatternScanner.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from pathlib import Path

from flext_infra.core.scanner import FlextInfraTextPatternScanner
from flext_tests import tm


class TestScannerCore:
    """Core scanning and validation tests."""

    def test_scan_matching_pattern(self, tmp_path: Path) -> None:
        """Matching pattern returns violation count."""
        scanner = FlextInfraTextPatternScanner()
        (tmp_path / "test.txt").write_text("hello world")
        result = scanner.scan(tmp_path, pattern="hello", includes=["*.txt"])
        tm.ok(result)
        tm.that(result.value["violation_count"], eq=1)

    def test_scan_no_matches(self, tmp_path: Path) -> None:
        """No matches returns zero violations."""
        scanner = FlextInfraTextPatternScanner()
        (tmp_path / "test.txt").write_text("goodbye world")
        result = scanner.scan(tmp_path, pattern="hello", includes=["*.txt"])
        tm.ok(result)
        tm.that(result.value["violation_count"], eq=0)

    def test_scan_with_excludes(self, tmp_path: Path) -> None:
        """Exclude patterns filter files."""
        scanner = FlextInfraTextPatternScanner()
        (tmp_path / "included.txt").write_text("hello")
        (tmp_path / "excluded.log").write_text("hello")
        result = scanner.scan(
            tmp_path,
            pattern="hello",
            includes=["*.txt"],
            excludes=["*.log"],
        )
        tm.ok(result)
        tm.that(result.value["files_scanned"], eq=1)

    def test_scan_absent_mode(self, tmp_path: Path) -> None:
        """Absent mode counts files missing the pattern."""
        scanner = FlextInfraTextPatternScanner()
        (tmp_path / "missing.txt").write_text("goodbye")
        (tmp_path / "found.txt").write_text("hello world")
        missing = scanner.scan(
            tmp_path,
            pattern="hello",
            includes=["missing.txt"],
            match_mode="absent",
        )
        tm.ok(missing)
        tm.that(missing.value["violation_count"], eq=1)
        found = scanner.scan(
            tmp_path,
            pattern="hello",
            includes=["found.txt"],
            match_mode="absent",
        )
        tm.ok(found)
        tm.that(found.value["violation_count"], eq=0)

    def test_scan_nonexistent_root(self, tmp_path: Path) -> None:
        """Nonexistent root returns failure."""
        scanner = FlextInfraTextPatternScanner()
        tm.fail(scanner.scan(tmp_path / "nope", pattern="x", includes=["*.txt"]))

    def test_scan_invalid_inputs(self, tmp_path: Path) -> None:
        """Empty includes and invalid match_mode return failure."""
        tm.fail(FlextInfraTextPatternScanner().scan(tmp_path, pattern="x", includes=[]))
        tm.fail(
            FlextInfraTextPatternScanner().scan(
                tmp_path,
                pattern="x",
                includes=["*.txt"],
                match_mode="invalid",
            )
        )

    def test_scan_invalid_regex(self, tmp_path: Path) -> None:
        """Invalid regex pattern returns failure."""
        (tmp_path / "test.txt").write_text("content")
        tm.fail(
            FlextInfraTextPatternScanner().scan(
                tmp_path,
                pattern="[invalid",
                includes=["*.txt"],
            )
        )


class TestScannerMultiFile:
    """Multi-file and nested directory scanning tests."""

    def test_scan_multiple_files(self, tmp_path: Path) -> None:
        """Matches across multiple files are counted."""
        scanner = FlextInfraTextPatternScanner()
        (tmp_path / "file1.txt").write_text("hello world")
        (tmp_path / "file2.txt").write_text("hello again")
        (tmp_path / "file3.txt").write_text("goodbye")
        result = scanner.scan(tmp_path, pattern="hello", includes=["*.txt"])
        tm.ok(result)
        tm.that(result.value["match_count"], eq=2)

    def test_scan_multiline_pattern(self, tmp_path: Path) -> None:
        """Multiline regex pattern matches all lines."""
        scanner = FlextInfraTextPatternScanner()
        (tmp_path / "test.txt").write_text("line1\nline2\nline3")
        result = scanner.scan(tmp_path, pattern="^line", includes=["*.txt"])
        tm.ok(result)
        tm.that(result.value["match_count"], eq=3)

    def test_scan_nested_directories(self, tmp_path: Path) -> None:
        """Files in nested directories are found."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("hello")
        result = FlextInfraTextPatternScanner().scan(
            tmp_path,
            pattern="hello",
            includes=["**/*.txt"],
        )
        tm.ok(result)
        tm.that(result.value["files_scanned"], eq=1)

    def test_scan_unreadable_file_skips(self, tmp_path: Path) -> None:
        """Unreadable files are skipped gracefully."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        f.chmod(0o000)
        try:
            result = FlextInfraTextPatternScanner().scan(
                tmp_path,
                pattern="hello",
                includes=["*.txt"],
            )
            tm.that(result.is_success, eq=True)
        finally:
            f.chmod(0o644)


class TestScannerHelpers:
    """Tests for scanner helper methods."""

    def test_collect_files_glob_patterns(self, tmp_path: Path) -> None:
        """_collect_files respects include/exclude glob patterns."""
        (tmp_path / "file1.py").write_text("")
        (tmp_path / "file2.txt").write_text("")
        (tmp_path / "file3.py").write_text("")
        (tmp_path / "test.py").write_text("")
        included = FlextInfraTextPatternScanner._collect_files(tmp_path, ["*.py"], [])
        tm.that(len(included), eq=3)
        excluded = FlextInfraTextPatternScanner._collect_files(
            tmp_path,
            ["*.py"],
            ["test*"],
        )
        tm.that(len(excluded), eq=2)

    def test_collect_files_skips_directories(self, tmp_path: Path) -> None:
        """_collect_files skips directories."""
        (tmp_path / "file.txt").write_text("")
        (tmp_path / "subdir").mkdir()
        files = FlextInfraTextPatternScanner._collect_files(tmp_path, ["*"], [])
        tm.that(len(files), eq=1)

    def test_count_matches(self, tmp_path: Path) -> None:
        """_count_matches counts regex matches and handles edge cases."""
        f = tmp_path / "test.txt"
        f.write_text("hello hello hello")
        regex = re.compile(r"hello")
        tm.that(FlextInfraTextPatternScanner._count_matches([f], regex), eq=3)
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        tm.that(FlextInfraTextPatternScanner._count_matches([empty], regex), eq=0)

    def test_count_matches_unreadable_file(self, tmp_path: Path) -> None:
        """_count_matches skips unreadable files."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        f.chmod(0o000)
        try:
            tm.that(
                FlextInfraTextPatternScanner._count_matches([f], re.compile(r"hello")),
                eq=0,
            )
        finally:
            f.chmod(0o644)


__all__: list[str] = []
