"""Tests for FlextInfraCodegenScaffolder service.

Validates module scaffolding for src/ and tests/ directories,
idempotency, and basic scaffold operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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


class TestScaffoldProjectNoop:
    def test_all_modules_present_creates_nothing(self, tmp_path: Path) -> None:
        project = _create_test_project(tmp_path, with_all_modules=True)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        result = scaffolder.scaffold_project(project)
        tm.that(result.files_created, eq=[])
        tm.that(len(result.files_skipped), eq=5)
        tm.that(result.project, eq="test-project")


class TestScaffoldProjectCreatesSrcModules:
    def test_creates_missing_src_modules(self, tmp_path: Path) -> None:
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        result = scaffolder.scaffold_project(project)
        tm.that(len(result.files_created), eq=5)
        pkg = project / "src" / "test_project"
        for mod in _SRC_MODULE_FILES:
            tm.that((pkg / mod).exists(), eq=True)

    def test_creates_only_missing_modules(self, tmp_path: Path) -> None:
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
        created_names = sorted(Path(f).name for f in result.files_created)
        tm.that(created_names, eq=["protocols.py", "typings.py", "utilities.py"])


class TestScaffoldProjectCreatesTestsModules:
    def test_creates_tests_modules_when_tests_dir_exists(
        self,
        tmp_path: Path,
    ) -> None:
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
        project = _create_test_project(tmp_path, with_all_modules=True)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        result = scaffolder.scaffold_project(project)
        tests_created = [f for f in result.files_created if "tests" in f]
        tm.that(tests_created, eq=[])


class TestScaffoldProjectIdempotency:
    def test_second_run_is_noop(self, tmp_path: Path) -> None:
        project = _create_test_project(tmp_path, with_all_modules=False)
        scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
        first_result = scaffolder.scaffold_project(project)
        second_result = scaffolder.scaffold_project(project)
        tm.that(len(first_result.files_created), eq=5)
        tm.that(second_result.files_created, eq=[])
        tm.that(len(second_result.files_skipped), eq=5)


__all__: list[str] = []
