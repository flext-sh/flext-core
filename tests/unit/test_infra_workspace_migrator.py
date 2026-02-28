"""Tests for FlextWorkspaceMigrator to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from flext_core import FlextResult as r
from flext_infra import m as im
from flext_infra.workspace.migrator import FlextInfraProjectMigrator


def _write_project(project_root: Path) -> None:
    (project_root / ".git").mkdir(parents=True, exist_ok=True)
    _ = (project_root / "base.mk").write_text("OLD_BASE\n", encoding="utf-8")
    _ = (project_root / "Makefile").write_text(
        'python "$(WORKSPACE_ROOT)/scripts/check/fix_pyrefly_config.py" "$(PROJECT_NAME)"\npython "$(WORKSPACE_ROOT)/scripts/check/workspace_check.py" --gates lint "$(PROJECT_NAME)"\n',
        encoding="utf-8",
    )
    _ = (project_root / "pyproject.toml").write_text(
        '[project]\nname = "project-a"\nversion = "0.1.0"\ndependencies = ["requests>=2"]\n',
        encoding="utf-8",
    )
    _ = (project_root / ".gitignore").write_text(
        "!scripts/\n!scripts/**\n*.pyc\n",
        encoding="utf-8",
    )


def _build_migrator(project: im.ProjectInfo, base_mk: str) -> FlextInfraProjectMigrator:
    migrator = FlextInfraProjectMigrator()

    def _discover_projects(workspace_root: Path) -> r[list[im.ProjectInfo]]:
        del workspace_root
        return r[list[im.ProjectInfo]].ok([project])

    migrator._discovery = cast(
        "Any",
        SimpleNamespace(discover_projects=_discover_projects),
    )
    migrator._generator = cast(
        "Any",
        SimpleNamespace(generate=lambda: r[str].ok(base_mk)),
    )
    return migrator


def test_migrator_dry_run_reports_changes_without_writes(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    _write_project(project_root)

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "NEW_BASE\n")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)

    assert result.is_success
    migration = result.value[0]
    assert any(change.startswith("[DRY-RUN]") for change in migration.changes)
    assert (project_root / "base.mk").read_text(encoding="utf-8") == "OLD_BASE\n"
    assert "scripts/check/workspace_check.py" in (project_root / "Makefile").read_text(
        encoding="utf-8",
    )


def test_migrator_apply_updates_project_files(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    _write_project(project_root)

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "NEW_BASE\n")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)

    assert result.is_success
    assert result.value[0].errors == []
    assert (project_root / "base.mk").read_text(encoding="utf-8") == "NEW_BASE\n"

    makefile_text = (project_root / "Makefile").read_text(encoding="utf-8")
    assert "scripts/check/workspace_check.py" not in makefile_text
    assert "python -m flext_infra check run" in makefile_text

    pyproject_text = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "flext-core @ ../flext-core" in pyproject_text

    gitignore_text = (project_root / ".gitignore").read_text(encoding="utf-8")
    assert "!scripts/" not in gitignore_text
    assert ".reports/" in gitignore_text


def test_migrator_handles_missing_pyproject_gracefully(tmp_path: Path) -> None:
    """Test that migrator handles missing pyproject.toml gracefully."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir(parents=True, exist_ok=True)
    _ = (project_root / "base.mk").write_text("OLD_BASE\n", encoding="utf-8")
    _ = (project_root / "Makefile").write_text("", encoding="utf-8")
    # No pyproject.toml

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "NEW_BASE\n")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)

    assert result.is_success
    assert (project_root / "base.mk").read_text(encoding="utf-8") == "NEW_BASE\n"


def test_migrator_preserves_custom_makefile_content(tmp_path: Path) -> None:
    """Test that migrator preserves custom Makefile content outside replacements."""
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    _write_project(project_root)
    custom_content = "# Custom rule\ncustom-target:\n\t@echo 'custom'\n"
    makefile_path = project_root / "Makefile"
    makefile_path.write_text(custom_content, encoding="utf-8")

    project = im.ProjectInfo.model_validate({
        "name": "project-a",
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })
    migrator = _build_migrator(project, "NEW_BASE\n")

    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)

    assert result.is_success
    makefile_text = makefile_path.read_text(encoding="utf-8")
    assert "custom-target" in makefile_text
    assert "@echo 'custom'" in makefile_text
