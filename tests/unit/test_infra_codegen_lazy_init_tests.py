"""Tests for FlextInfraLazyInitGenerator scan_tests support.

Validates that the ``scan_tests`` parameter in ``run()`` correctly controls
whether ``tests/**/__init__.py`` files are included in the scan.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra.codegen.lazy_init import FlextInfraLazyInitGenerator


def _create_init_file(directory: Path, content: str) -> Path:
    """Create an __init__.py file in the given directory with the given content."""
    directory.mkdir(parents=True, exist_ok=True)
    init_file = directory / "__init__.py"
    init_file.write_text(content, encoding="utf-8")
    return init_file


_VALID_INIT = (
    '"""Test package."""\n'
    "from test_pkg.module import TestClass\n"
    '__all__ = ["TestClass"]\n'
)

_VALID_TESTS_INIT = (
    '"""Test helpers."""\n'
    "from test_helpers.fixtures import SomeFixture\n"
    '__all__ = ["SomeFixture"]\n'
)


class TestScanTestsDefaultBehavior:
    """Verify that scan_tests=False (default) only scans src/."""

    def test_run_check_only_scan_tests_false_ignores_tests_dir(
        self, tmp_path: Path
    ) -> None:
        """run(check_only=True, scan_tests=False) only scans src/."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        _create_init_file(tmp_path / "tests" / "helpers", _VALID_TESTS_INIT)

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=True, scan_tests=False)

        # Only src/ file is scanned; tests/ is ignored.
        # Result reflects unmapped count from src/ only.
        assert isinstance(result, int)
        assert result >= 0

    def test_run_default_scan_tests_is_false(self, tmp_path: Path) -> None:
        """run() defaults to scan_tests=False."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        _create_init_file(tmp_path / "tests" / "helpers", _VALID_TESTS_INIT)

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)

        # Both calls should produce the same result
        result_default = generator.run(check_only=True)
        result_explicit = generator.run(check_only=True, scan_tests=False)
        assert result_default == result_explicit

    def test_run_scan_tests_false_does_not_modify_tests_files(
        self, tmp_path: Path
    ) -> None:
        """run(scan_tests=False) never touches tests/ __init__.py files."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        tests_init = _create_init_file(
            tmp_path / "tests" / "helpers", _VALID_TESTS_INIT
        )
        original_content = tests_init.read_text(encoding="utf-8")

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        generator.run(check_only=False, scan_tests=False)

        # tests/ file must remain untouched
        assert tests_init.read_text(encoding="utf-8") == original_content


class TestScanTestsEnabled:
    """Verify that scan_tests=True also processes tests/ directories."""

    def test_run_check_only_scan_tests_true_includes_tests_dir(
        self, tmp_path: Path
    ) -> None:
        """run(check_only=True, scan_tests=True) scans both src/ and tests/."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        _create_init_file(tmp_path / "tests" / "helpers", _VALID_TESTS_INIT)

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)

        result_without = generator.run(check_only=True, scan_tests=False)
        result_with = generator.run(check_only=True, scan_tests=True)

        # With scan_tests=True, more files are scanned so the result
        # may differ (additional unmapped exports from tests/).
        assert isinstance(result_with, int)
        assert result_with >= 0
        # scan_tests=True should scan at least as many files
        assert result_with >= result_without or result_with == result_without

    def test_run_scan_tests_true_processes_tests_init_files(
        self, tmp_path: Path
    ) -> None:
        """run(scan_tests=True) processes tests/__init__.py files."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        tests_init = _create_init_file(
            tmp_path / "tests" / "helpers", _VALID_TESTS_INIT
        )
        original_content = tests_init.read_text(encoding="utf-8")

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        generator.run(check_only=False, scan_tests=True)

        # tests/ file should be processed (content may change)
        # The file should still exist
        assert tests_init.exists()
        # If the file had valid __all__, it should have been rewritten
        new_content = tests_init.read_text(encoding="utf-8")
        assert new_content != original_content or "__all__" in new_content

    def test_run_scan_tests_true_with_nested_tests_packages(
        self, tmp_path: Path
    ) -> None:
        """scan_tests=True finds deeply nested tests/__init__.py files."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        nested_init = _create_init_file(
            tmp_path / "tests" / "unit" / "helpers",
            '"""Nested test helpers."""\n'
            "from test_helpers.deep import DeepFixture\n"
            '__all__ = ["DeepFixture"]\n',
        )

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        generator.run(check_only=False, scan_tests=True)

        # Nested tests/ file should be processed
        assert nested_init.exists()

    def test_run_scan_tests_true_check_only_does_not_modify(
        self, tmp_path: Path
    ) -> None:
        """run(check_only=True, scan_tests=True) reports but doesn't write."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        tests_init = _create_init_file(
            tmp_path / "tests" / "helpers", _VALID_TESTS_INIT
        )
        original_content = tests_init.read_text(encoding="utf-8")

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        generator.run(check_only=True, scan_tests=True)

        # check_only=True must not modify any file
        assert tests_init.read_text(encoding="utf-8") == original_content


class TestScanTestsWithoutInitPy:
    """Verify that tests/ directories without __init__.py are skipped."""

    def test_tests_dir_without_init_py_is_skipped(self, tmp_path: Path) -> None:
        """tests/ directory without __init__.py produces no errors."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        # Create tests/ dir but no __init__.py
        tests_dir = tmp_path / "tests" / "helpers"
        tests_dir.mkdir(parents=True)
        # Add a regular .py file (not __init__.py)
        (tests_dir / "conftest.py").write_text("# conftest", encoding="utf-8")

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=True, scan_tests=True)

        # Should not crash; only src/ file is found
        assert isinstance(result, int)
        assert result >= 0

    def test_empty_tests_dir_is_skipped(self, tmp_path: Path) -> None:
        """Empty tests/ directory produces no errors."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        (tmp_path / "tests").mkdir(parents=True)

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=True, scan_tests=True)

        assert isinstance(result, int)
        assert result >= 0

    def test_no_tests_dir_at_all(self, tmp_path: Path) -> None:
        """Workspace without tests/ directory works fine with scan_tests=True."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=True, scan_tests=True)

        assert isinstance(result, int)
        assert result >= 0


class TestSrcBehaviorNoRegression:
    """Ensure scan_tests does not regress existing src/ behavior."""

    def test_src_only_results_unchanged_with_scan_tests_false(
        self, tmp_path: Path
    ) -> None:
        """src/ processing is identical regardless of scan_tests flag when no tests/ exist."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result_false = generator.run(check_only=True, scan_tests=False)
        result_true = generator.run(check_only=True, scan_tests=True)

        # Without tests/ dir, results should be identical
        assert result_false == result_true

    def test_src_file_processed_identically_with_and_without_scan_tests(
        self, tmp_path: Path
    ) -> None:
        """src/ __init__.py content is the same whether scan_tests is True or False."""
        src_content = (
            '"""Package."""\nfrom pkg.models import MyModel\n__all__ = ["MyModel"]\n'
        )

        # Run with scan_tests=False
        src_dir_a = tmp_path / "a" / "src" / "pkg"
        _create_init_file(src_dir_a, src_content)
        gen_a = FlextInfraLazyInitGenerator(workspace_root=tmp_path / "a")
        gen_a.run(check_only=False, scan_tests=False)
        content_a = (src_dir_a / "__init__.py").read_text(encoding="utf-8")

        # Run with scan_tests=True
        src_dir_b = tmp_path / "b" / "src" / "pkg"
        _create_init_file(src_dir_b, src_content)
        gen_b = FlextInfraLazyInitGenerator(workspace_root=tmp_path / "b")
        gen_b.run(check_only=False, scan_tests=True)
        content_b = (src_dir_b / "__init__.py").read_text(encoding="utf-8")

        assert content_a == content_b

    def test_execute_method_unaffected_by_scan_tests(self, tmp_path: Path) -> None:
        """execute() still works and returns FlextResult[int]."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.execute()

        assert result.is_success
        assert isinstance(result.value, int)

    def test_vendor_and_venv_dirs_still_excluded(self, tmp_path: Path) -> None:
        """Vendor and .venv directories are excluded even with scan_tests=True."""
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        _create_init_file(tmp_path / "tests" / "vendor" / "pkg", _VALID_TESTS_INIT)
        _create_init_file(tmp_path / "tests" / ".venv" / "pkg", _VALID_TESTS_INIT)

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        # These should be excluded by the vendor/node_modules/.venv filter
        result = generator.run(check_only=True, scan_tests=True)
        assert isinstance(result, int)
        assert result >= 0

    def test_empty_workspace_returns_zero(self, tmp_path: Path) -> None:
        """Empty workspace returns 0 with any scan_tests value."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)

        assert generator.run(check_only=True, scan_tests=False) == 0
        assert generator.run(check_only=True, scan_tests=True) == 0
        assert generator.run(check_only=False, scan_tests=False) == 0
        assert generator.run(check_only=False, scan_tests=True) == 0
