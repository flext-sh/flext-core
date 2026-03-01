"""Tests for FlextInfraModuleScaffolder service.

Validates module scaffolding for src/ and tests/ directories,
idempotency, generated code validity, and naming conventions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

from flext_infra.codegen.module_scaffolder import FlextInfraModuleScaffolder


_SRC_MODULE_FILES = (
    "constants.py",
    "typings.py",
    "protocols.py",
    "models.py",
    "utilities.py",
)

_PATCH_RUFF = "flext_infra.codegen.module_scaffolder.FlextInfraAstUtils.run_ruff_fix"
_PATCH_PREFIX = (
    "flext_infra.codegen.module_scaffolder.FlextInfraNamespaceValidator._derive_prefix"
)


def _derive_prefix_from_path(project_root: Path) -> str:
    """Replicate _derive_prefix logic for test isolation (no self needed)."""
    src_dir = project_root / "src"
    if not src_dir.is_dir():
        return ""
    for child in sorted(src_dir.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            return "".join(part.title() for part in child.name.split("_"))
    return ""


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
                f"class TestProject{mod.split('.')[0].title()}:\n    pass\n"
            )
    return project


class TestScaffoldProjectNoop:
    """When all 5 modules exist, scaffold_project is a no-op."""

    def test_all_modules_present_creates_nothing(self, tmp_path: Path) -> None:
        """scaffold_project on a project with all 5 modules → zero files created."""
        project = _create_test_project(tmp_path, with_all_modules=True)
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path):
            result = scaffolder.scaffold_project(project)

        assert result.files_created == []
        assert len(result.files_skipped) == 5
        assert result.project == "test-project"


class TestScaffoldProjectCreatesSrcModules:
    """scaffold_project creates missing src/ modules as skeletons."""

    def test_creates_missing_src_modules(self, tmp_path: Path) -> None:
        """Missing src/ modules are generated as skeleton files."""
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
            result = scaffolder.scaffold_project(project)

        assert len(result.files_created) == 5
        pkg = project / "src" / "test_project"
        for mod in _SRC_MODULE_FILES:
            assert (pkg / mod).exists()

    def test_creates_only_missing_modules(self, tmp_path: Path) -> None:
        """Only missing modules are created; existing ones are skipped."""
        project = _create_test_project(tmp_path, with_all_modules=False)
        pkg = project / "src" / "test_project"
        (pkg / "constants.py").write_text("class TestProjectConstants:\n    pass\n")
        (pkg / "models.py").write_text("class TestProjectModels:\n    pass\n")

        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
            result = scaffolder.scaffold_project(project)

        assert len(result.files_created) == 3
        assert len(result.files_skipped) == 2
        created_names = {Path(f).name for f in result.files_created}
        assert created_names == {"typings.py", "protocols.py", "utilities.py"}


class TestScaffoldProjectCreatesTestsModules:
    """scaffold_project creates missing tests/ modules when tests/ dir exists."""

    def test_creates_tests_modules_when_tests_dir_exists(self, tmp_path: Path) -> None:
        """Tests modules are scaffolded when tests/ directory is present."""
        project = _create_test_project(tmp_path, with_all_modules=True)
        tests_dir = project / "tests"
        tests_dir.mkdir()

        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
            result = scaffolder.scaffold_project(project)

        tests_created = [f for f in result.files_created if "tests" in f]
        assert len(tests_created) == 5
        for mod in _SRC_MODULE_FILES:
            assert (tests_dir / mod).exists()

    def test_skips_tests_modules_when_no_tests_dir(self, tmp_path: Path) -> None:
        """No tests/ modules are created when tests/ directory is absent."""
        project = _create_test_project(tmp_path, with_all_modules=True)
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path):
            result = scaffolder.scaffold_project(project)

        tests_created = [f for f in result.files_created if "tests" in f]
        assert tests_created == []


class TestScaffoldProjectIdempotency:
    """Running scaffold_project twice creates nothing on second run."""

    def test_second_run_is_noop(self, tmp_path: Path) -> None:
        """Idempotency: second scaffold_project call creates zero files."""
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
            first_result = scaffolder.scaffold_project(project)
            second_result = scaffolder.scaffold_project(project)

        assert len(first_result.files_created) == 5
        assert second_result.files_created == []
        assert len(second_result.files_skipped) == 5


class TestGeneratedFilesAreValidPython:
    """Generated files must be parseable by ast.parse."""

    def test_generated_src_modules_parse_successfully(self, tmp_path: Path) -> None:
        """All generated src/ modules are valid Python (ast.parse succeeds)."""
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
            scaffolder.scaffold_project(project)

        pkg = project / "src" / "test_project"
        for mod in _SRC_MODULE_FILES:
            source = (pkg / mod).read_text(encoding="utf-8")
            tree = ast.parse(source)
            assert isinstance(tree, ast.Module)

    def test_generated_tests_modules_parse_successfully(self, tmp_path: Path) -> None:
        """All generated tests/ modules are valid Python (ast.parse succeeds)."""
        project = _create_test_project(tmp_path, with_all_modules=True)
        tests_dir = project / "tests"
        tests_dir.mkdir()
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
            scaffolder.scaffold_project(project)

        for mod in _SRC_MODULE_FILES:
            source = (tests_dir / mod).read_text(encoding="utf-8")
            tree = ast.parse(source)
            assert isinstance(tree, ast.Module)


class TestGeneratedClassNamingConvention:
    """Generated class names follow the {Prefix}{Suffix} convention."""

    def test_src_class_names_use_prefix_suffix(self, tmp_path: Path) -> None:
        """Src modules use {Prefix}{Suffix}: TestProject + Constants/Types/etc."""
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
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
            assert expected_class in class_names, (
                f"{filename} should contain class {expected_class}, found {class_names}"
            )

    def test_tests_class_names_use_tests_prefix_suffix(self, tmp_path: Path) -> None:
        """Tests modules use Tests{Prefix}{Suffix}: TestsTestProject + Constants/etc."""
        project = _create_test_project(tmp_path, with_all_modules=True)
        tests_dir = project / "tests"
        tests_dir.mkdir()
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
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
            assert expected_class in class_names, (
                f"{filename} should contain class {expected_class}, found {class_names}"
            )

    def test_no_prefix_returns_empty_result(self, tmp_path: Path) -> None:
        """Project without src/ package returns empty ScaffoldResult."""
        project = tmp_path / "empty-project"
        project.mkdir()
        (project / "Makefile").touch()
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path):
            result = scaffolder.scaffold_project(project)

        assert result.files_created == []
        assert result.files_skipped == []
        assert result.project == "empty-project"


__all__: list[str] = []
