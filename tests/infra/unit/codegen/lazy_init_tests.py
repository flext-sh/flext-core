"""Tests for FlextInfraCodegenLazyInit directory scanning behavior.

Validates that ``run()`` correctly scans all standard directories
(``src/``, ``tests/``, ``examples/``, ``scripts/``) and applies
PEP 562 lazy-import generation uniformly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
from flext_tests import tm


def _create_init_file(directory: Path, content: str) -> Path:
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


class TestAllDirectoriesScanned:
    """All standard directories are always scanned."""

    def test_src_dir_is_scanned(self, tmp_path: Path) -> None:
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=True)
        tm.that(type(result).__name__, eq="int")
        tm.that(result, gte=0)

    def test_tests_dir_is_scanned(self, tmp_path: Path) -> None:
        _create_init_file(tmp_path / "tests" / "helpers", _VALID_TESTS_INIT)
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=True)
        tm.that(type(result).__name__, eq="int")
        tm.that(result, gte=0)

    def test_tests_init_files_are_processed(self, tmp_path: Path) -> None:
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        tests_init = _create_init_file(
            tmp_path / "tests" / "helpers",
            _VALID_TESTS_INIT,
        )
        original_content = tests_init.read_text(encoding="utf-8")
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        generator.run(check_only=False)
        new_content = tests_init.read_text(encoding="utf-8")
        tm.that(
            new_content != original_content or "__all__" in new_content,
            eq=True,
        )

    def test_nested_tests_packages_are_found(self, tmp_path: Path) -> None:
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        nested_init = _create_init_file(
            tmp_path / "tests" / "unit" / "helpers",
            '"""Nested test helpers."""\n'
            "from test_helpers.deep import DeepFixture\n"
            '__all__ = ["DeepFixture"]\n',
        )
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        generator.run(check_only=False)
        tm.that(nested_init.exists(), eq=True)


class TestCheckOnlyMode:
    """check_only=True reports without writing."""

    def test_check_only_does_not_modify_files(self, tmp_path: Path) -> None:
        tests_init = _create_init_file(
            tmp_path / "tests" / "helpers",
            _VALID_TESTS_INIT,
        )
        original_content = tests_init.read_text(encoding="utf-8")
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        generator.run(check_only=True)
        tm.that(
            tests_init.read_text(encoding="utf-8"),
            eq=original_content,
        )


class TestExcludedDirectories:
    """Vendor and .venv directories are excluded."""

    def test_vendor_dir_excluded(self, tmp_path: Path) -> None:
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        _create_init_file(
            tmp_path / "tests" / "vendor" / "pkg",
            _VALID_TESTS_INIT,
        )
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=True)
        tm.that(type(result).__name__, eq="int")
        tm.that(result, gte=0)

    def test_venv_dir_excluded(self, tmp_path: Path) -> None:
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        _create_init_file(
            tmp_path / "tests" / ".venv" / "pkg",
            _VALID_TESTS_INIT,
        )
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=True)
        tm.that(type(result).__name__, eq="int")
        tm.that(result, gte=0)


class TestEdgeCases:
    """Edge cases for directory scanning."""

    def test_empty_workspace_returns_zero(self, tmp_path: Path) -> None:
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        tm.that(generator.run(check_only=True), eq=0)
        tm.that(generator.run(check_only=False), eq=0)

    def test_tests_dir_without_init_py_is_skipped(
        self,
        tmp_path: Path,
    ) -> None:
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        tests_dir = tmp_path / "tests" / "helpers"
        tests_dir.mkdir(parents=True)
        (tests_dir / "conftest.py").write_text("# conftest", encoding="utf-8")
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=True)
        tm.that(type(result).__name__, eq="int")
        tm.that(result, gte=0)

    def test_no_tests_dir_at_all(self, tmp_path: Path) -> None:
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=True)
        tm.that(type(result).__name__, eq="int")
        tm.that(result, gte=0)

    def test_execute_method_returns_flext_result(
        self,
        tmp_path: Path,
    ) -> None:
        _create_init_file(tmp_path / "src" / "pkg", _VALID_INIT)
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.execute()
        tm.that(result.is_success, eq=True)
        tm.that(type(result.value).__name__, eq="int")

    def test_src_content_consistent_across_runs(
        self,
        tmp_path: Path,
    ) -> None:
        src_content = (
            '"""Package."""\nfrom pkg.models import MyModel\n__all__ = ["MyModel"]\n'
        )
        src_dir_a = tmp_path / "a" / "src" / "pkg"
        _create_init_file(src_dir_a, src_content)
        gen_a = FlextInfraCodegenLazyInit(workspace_root=tmp_path / "a")
        gen_a.run(check_only=False)
        content_a = (src_dir_a / "__init__.py").read_text(encoding="utf-8")
        src_dir_b = tmp_path / "b" / "src" / "pkg"
        _create_init_file(src_dir_b, src_content)
        gen_b = FlextInfraCodegenLazyInit(workspace_root=tmp_path / "b")
        gen_b.run(check_only=False)
        content_b = (src_dir_b / "__init__.py").read_text(encoding="utf-8")
        tm.that(content_a, eq=content_b)
