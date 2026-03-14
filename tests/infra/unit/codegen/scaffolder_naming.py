"""Tests for scaffolder generated code validity and naming conventions.

Validates AST parseability and {Prefix}{Suffix} class naming convention
for both src/ and tests/ generated modules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path

from flext_infra.codegen.scaffolder import FlextInfraCodegenScaffolder
from flext_tests import tm

_SRC_MODULE_FILES = (
    "constants.py",
    "typings.py",
    "protocols.py",
    "models.py",
    "utilities.py",
)


def _create_test_project(tmp_path: Path, *, with_all_modules: bool = True) -> Path:
    project = tmp_path / "test-project"
    project.mkdir()
    (project / "Makefile").touch()
    (project / "pyproject.toml").write_text("[project]\nname='test-project'\n")
    (project / ".git").mkdir()
    pkg = project / "src" / "test_project"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    if with_all_modules:
        for mod in _SRC_MODULE_FILES:
            (pkg / mod).write_text(
                f"class TestProject{mod.split('.')[0].title()}:\n    pass\n",
            )
    return project


class TestGeneratedFilesAreValidPython:
    def test_generated_src_modules_parse_successfully(
        self,
        tmp_path: Path,
    ) -> None:
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        scaffolder.scaffold_project(project)
        pkg = project / "src" / "test_project"
        for mod in _SRC_MODULE_FILES:
            source = (pkg / mod).read_text(encoding="utf-8")
            tree = ast.parse(source)
            tm.that(type(tree).__name__, eq="Module")

    def test_generated_tests_modules_parse_successfully(
        self,
        tmp_path: Path,
    ) -> None:
        project = _create_test_project(tmp_path, with_all_modules=True)
        tests_dir = project / "tests"
        tests_dir.mkdir()
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        scaffolder.scaffold_project(project)
        for mod in _SRC_MODULE_FILES:
            source = (tests_dir / mod).read_text(encoding="utf-8")
            tree = ast.parse(source)
            tm.that(type(tree).__name__, eq="Module")


class TestGeneratedClassNamingConvention:
    def test_src_class_names_use_prefix_suffix(self, tmp_path: Path) -> None:
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        scaffolder.scaffold_project(project)
        pkg = project / "src" / "test_project"
        expected_classes = {
            "constants.py": "TestProjectConstants",
            "typings.py": "TestProjectTypes",
            "protocols.py": "TestProjectProtocols",
            "models.py": "TestProjectModels",
            "utilities.py": "TestProjectUtilities",
        }
        for filename, expected_class in expected_classes.items():
            source = (pkg / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            class_names = [
                node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]
            tm.that(
                expected_class in class_names,
                eq=True,
                msg=f"{filename} should contain {expected_class}",
            )

    def test_tests_class_names_use_tests_prefix_suffix(
        self,
        tmp_path: Path,
    ) -> None:
        project = _create_test_project(tmp_path, with_all_modules=True)
        tests_dir = project / "tests"
        tests_dir.mkdir()
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        scaffolder.scaffold_project(project)
        expected_classes = {
            "constants.py": "TestsTestProjectConstants",
            "typings.py": "TestsTestProjectTypes",
            "protocols.py": "TestsTestProjectProtocols",
            "models.py": "TestsTestProjectModels",
            "utilities.py": "TestsTestProjectUtilities",
        }
        for filename, expected_class in expected_classes.items():
            source = (tests_dir / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            class_names = [
                node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]
            tm.that(
                expected_class in class_names,
                eq=True,
                msg=f"{filename} should contain {expected_class}",
            )

    def test_no_prefix_returns_empty_result(self, tmp_path: Path) -> None:
        project = tmp_path / "empty-project"
        project.mkdir()
        (project / "Makefile").touch()
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        result = scaffolder.scaffold_project(project)
        tm.that(result.files_created, eq=[])
        tm.that(result.files_skipped, eq=[])
        tm.that(result.project, eq="empty-project")


__all__: list[str] = []
