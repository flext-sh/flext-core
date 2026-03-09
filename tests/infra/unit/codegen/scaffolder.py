"""Tests for FlextInfraCodegenScaffolder service.

Validates module scaffolding for src/ and tests/ directories,
idempotency, generated code validity, and naming conventions.

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
    """Create a minimal test project directory structure."""
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


class TestScaffoldProjectNoop:
    """When all 5 modules exist, scaffold_project is a no-op."""

    def test_all_modules_present_creates_nothing(self, tmp_path: Path) -> None:
        """scaffold_project on a project with all 5 modules creates nothing."""
        project = _create_test_project(tmp_path, with_all_modules=True)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        result = scaffolder.scaffold_project(project)
        tm.that(result.files_created, eq=[])
        tm.that(len(result.files_skipped), eq=5)
        tm.that(result.project, eq="test-project")


class TestScaffoldProjectCreatesSrcModules:
    """scaffold_project creates missing src/ modules as skeletons."""

    def test_creates_missing_src_modules(self, tmp_path: Path) -> None:
        """Missing src/ modules are generated as skeleton files."""
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        result = scaffolder.scaffold_project(project)
        tm.that(len(result.files_created), eq=5)
        pkg = project / "src" / "test_project"
        for mod in _SRC_MODULE_FILES:
            tm.that((pkg / mod).exists(), eq=True)

    def test_creates_only_missing_modules(self, tmp_path: Path) -> None:
        """Only missing modules are created; existing ones are skipped."""
        project = _create_test_project(tmp_path, with_all_modules=False)
        pkg = project / "src" / "test_project"
        (pkg / "constants.py").write_text(
            "class TestProjectConstants:\n    pass\n",
        )
        (pkg / "models.py").write_text("class TestProjectModels:\n    pass\n")
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        result = scaffolder.scaffold_project(project)
        tm.that(len(result.files_created), eq=3)
        tm.that(len(result.files_skipped), eq=2)
        created_names = {Path(f).name for f in result.files_created}
        tm.that(created_names, eq={"typings.py", "protocols.py", "utilities.py"})


class TestScaffoldProjectCreatesTestsModules:
    """scaffold_project creates missing tests/ modules when tests/ exists."""

    def test_creates_tests_modules_when_tests_dir_exists(
        self,
        tmp_path: Path,
    ) -> None:
        """Tests modules are scaffolded when tests/ directory is present."""
        project = _create_test_project(tmp_path, with_all_modules=True)
        tests_dir = project / "tests"
        tests_dir.mkdir()
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        result = scaffolder.scaffold_project(project)
        tests_created = [f for f in result.files_created if "tests" in f]
        tm.that(len(tests_created), eq=5)
        for mod in _SRC_MODULE_FILES:
            tm.that((tests_dir / mod).exists(), eq=True)

    def test_skips_tests_modules_when_no_tests_dir(
        self,
        tmp_path: Path,
    ) -> None:
        """No tests/ modules are created when tests/ directory is absent."""
        project = _create_test_project(tmp_path, with_all_modules=True)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        result = scaffolder.scaffold_project(project)
        tests_created = [f for f in result.files_created if "tests" in f]
        tm.that(tests_created, eq=[])


class TestScaffoldProjectIdempotency:
    """Running scaffold_project twice creates nothing on second run."""

    def test_second_run_is_noop(self, tmp_path: Path) -> None:
        """Idempotency: second scaffold_project call creates zero files."""
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        first_result = scaffolder.scaffold_project(project)
        second_result = scaffolder.scaffold_project(project)
        tm.that(len(first_result.files_created), eq=5)
        tm.that(second_result.files_created, eq=[])
        tm.that(len(second_result.files_skipped), eq=5)


class TestGeneratedFilesAreValidPython:
    """Generated files must be parseable by ast.parse."""

    def test_generated_src_modules_parse_successfully(
        self,
        tmp_path: Path,
    ) -> None:
        """All generated src/ modules are valid Python."""
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        scaffolder.scaffold_project(project)
        pkg = project / "src" / "test_project"
        for mod in _SRC_MODULE_FILES:
            source = (pkg / mod).read_text(encoding="utf-8")
            tree = ast.parse(source)
            tm.that(isinstance(tree, ast.Module), eq=True)

    def test_generated_tests_modules_parse_successfully(
        self,
        tmp_path: Path,
    ) -> None:
        """All generated tests/ modules are valid Python."""
        project = _create_test_project(tmp_path, with_all_modules=True)
        tests_dir = project / "tests"
        tests_dir.mkdir()
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        scaffolder.scaffold_project(project)
        for mod in _SRC_MODULE_FILES:
            source = (tests_dir / mod).read_text(encoding="utf-8")
            tree = ast.parse(source)
            tm.that(isinstance(tree, ast.Module), eq=True)


class TestGeneratedClassNamingConvention:
    """Generated class names follow the {Prefix}{Suffix} convention."""

    def test_src_class_names_use_prefix_suffix(self, tmp_path: Path) -> None:
        """Src modules use {Prefix}{Suffix} naming convention."""
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
        """Tests modules use Tests{Prefix}{Suffix} naming convention."""
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
        """Project without src/ package returns empty ScaffoldResult."""
        project = tmp_path / "empty-project"
        project.mkdir()
        (project / "Makefile").touch()
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        result = scaffolder.scaffold_project(project)
        tm.that(result.files_created, eq=[])
        tm.that(result.files_skipped, eq=[])
        tm.that(result.project, eq="empty-project")


__all__: list[str] = []
