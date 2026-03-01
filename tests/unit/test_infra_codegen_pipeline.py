"""Integration tests for the codegen pipeline (census, scaffold, auto-fix).

Validates end-to-end workflows combining discovery, scaffolding, census,
and auto-fix services on temporary workspaces.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

from flext_infra.codegen.auto_fix import FlextInfraAutoFixer
from flext_infra.codegen.census import FlextInfraCodegenCensus
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


def _make_project(
    tmp_path: Path,
    name: str,
    *,
    with_all_modules: bool = True,
) -> Path:
    """Create a minimal test project directory structure.

    Args:
        tmp_path: Temporary directory root.
        name: Project name (e.g., "test-project").
        with_all_modules: If True, create all 5 base modules.

    Returns:
        Path to the created project directory.

    """
    project = tmp_path / name
    project.mkdir()
    (project / "Makefile").touch()
    (project / "pyproject.toml").write_text(f"[project]\nname='{name}'\n")
    (project / ".git").mkdir()

    pkg_name = name.replace("-", "_")
    pkg = project / "src" / pkg_name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").touch()

    if with_all_modules:
        for mod in _SRC_MODULE_FILES:
            class_name = "".join(word.title() for word in mod.split(".")[0].split("_"))
            (pkg / mod).write_text(f"class {pkg_name.title()}{class_name}:\n    pass\n")

    return project


class TestScaffoldNoopForCompleteProject:
    """When all 5 modules exist, scaffold is a no-op."""

    def test_scaffold_noop_for_complete_project(self, tmp_path: Path) -> None:
        """Project with all 5 modules → files_created == []."""
        project = _make_project(tmp_path, "complete-project", with_all_modules=True)
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path):
            result = scaffolder.scaffold_project(project)

        assert result.files_created == []
        assert len(result.files_skipped) == 5
        assert result.project == "complete-project"


class TestScaffoldCreatesMissingModules:
    """Scaffold creates missing modules."""

    def test_scaffold_creates_missing_modules(self, tmp_path: Path) -> None:
        """Project missing models.py → files_created contains models.py."""
        project = _make_project(tmp_path, "partial-project", with_all_modules=False)
        pkg_name = "partial_project"
        pkg = project / "src" / pkg_name

        # Create only 4 modules, omit models.py
        for mod in ("constants.py", "typings.py", "protocols.py", "utilities.py"):
            class_name = "".join(word.title() for word in mod.split(".")[0].split("_"))
            (pkg / mod).write_text(f"class {pkg_name.title()}{class_name}:\n    pass\n")

        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
            result = scaffolder.scaffold_project(project)

        assert len(result.files_created) == 1
        created_names = {Path(f).name for f in result.files_created}
        assert "models.py" in created_names


class TestScaffoldIdempotency:
    """Scaffold is idempotent."""

    def test_scaffold_idempotency(self, tmp_path: Path) -> None:
        """Run scaffold twice → second run files_created == []."""
        project = _make_project(tmp_path, "idempotent-project", with_all_modules=False)
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        # First run: create missing modules
        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
            result1 = scaffolder.scaffold_project(project)

        assert len(result1.files_created) == 5

        # Second run: all modules exist, should be no-op
        with patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path):
            result2 = scaffolder.scaffold_project(project)

        assert result2.files_created == []
        assert len(result2.files_skipped) == 5


class TestCensusRunsOnWorkspace:
    """Census service runs on workspace."""

    def test_census_runs_on_workspace(self, tmp_path: Path) -> None:
        """Census on tmp workspace → returns CensusReport with total_violations >= 0."""
        _make_project(tmp_path, "census-project", with_all_modules=True)

        census = FlextInfraCodegenCensus(workspace_root=tmp_path)
        reports = census.run()

        assert isinstance(reports, list)
        assert len(reports) >= 1
        report = reports[0]
        assert report.project == "census-project"
        assert report.total >= 0
        assert report.fixable >= 0
        assert isinstance(report.violations, list)


class TestAutofixRunsOnWorkspace:
    """Auto-fix service runs on workspace."""

    def test_autofix_runs_on_workspace(self, tmp_path: Path) -> None:
        """Auto-fix on tmp workspace → returns AutoFixResult with violations_fixed list."""
        _make_project(tmp_path, "autofix-project", with_all_modules=True)

        fixer = FlextInfraAutoFixer(workspace_root=tmp_path)
        results = fixer.run()

        assert isinstance(results, list)
        assert len(results) >= 1
        result = results[0]
        assert result.project == "autofix-project"
        assert isinstance(result.violations_fixed, list)
        assert isinstance(result.violations_skipped, list)
        assert isinstance(result.files_modified, list)


class TestGeneratedFilesAreValidPython:
    """Generated files are valid Python."""

    def test_generated_files_are_valid_python(self, tmp_path: Path) -> None:
        """Scaffold creates files → ast.parse() succeeds on each."""
        project = _make_project(
            tmp_path, "valid-python-project", with_all_modules=False
        )
        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
            result = scaffolder.scaffold_project(project)

        pkg_name = "valid_python_project"
        pkg = project / "src" / pkg_name

        for file_path in result.files_created:
            full_path = pkg / Path(file_path).name
            assert full_path.exists()
            content = full_path.read_text()
            # Should not raise SyntaxError
            ast.parse(content)


class TestFlexcoreExcluded:
    """Flexcore projects are excluded from all services."""

    def test_flexcore_excluded(self, tmp_path: Path) -> None:
        """Add a flexcore project → scaffold result has no entry for it."""
        # Create a normal project
        _make_project(tmp_path, "normal-project", with_all_modules=False)

        # Create a flexcore project
        _make_project(tmp_path, "flexcore", with_all_modules=False)

        scaffolder = FlextInfraModuleScaffolder(workspace_root=tmp_path)

        with (
            patch(_PATCH_RUFF),
            patch(_PATCH_PREFIX, side_effect=_derive_prefix_from_path),
        ):
            results = scaffolder.run()

        project_names = {r.project for r in results}
        assert "normal-project" in project_names
        assert "flexcore" not in project_names
