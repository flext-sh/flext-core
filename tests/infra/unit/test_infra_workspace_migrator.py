"""Tests for FlextWorkspaceMigrator — core migration scenarios.

Uses real service instances with typed stubs and monkeypatch.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import m as im
from flext_infra.workspace.migrator import FlextInfraProjectMigrator
from flext_tests import tm


class _StubDiscovery:
    """Typed stub for discovery service."""

    def __init__(self, projects: list[im.Infra.Workspace.ProjectInfo]) -> None:
        self._projects = projects

    def discover_projects(
        self, workspace_root: Path
    ) -> r[list[im.Infra.Workspace.ProjectInfo]]:
        return r[list[im.Infra.Workspace.ProjectInfo]].ok(self._projects)


class _StubDiscoveryFail:
    """Typed stub for discovery service that fails."""

    def __init__(self, error: str) -> None:
        self._error = error

    def discover_projects(
        self, workspace_root: Path
    ) -> r[list[im.Infra.Workspace.ProjectInfo]]:
        return r[list[im.Infra.Workspace.ProjectInfo]].fail(self._error)


class _StubGenerator:
    """Typed stub for base.mk generator."""

    def __init__(self, content: str = "", *, fail: str = "") -> None:
        self._content = content
        self._fail = fail

    def generate(self, *args: object, **kwargs: object) -> r[str]:
        if self._fail:
            return r[str].fail(self._fail)
        return r[str].ok(self._content)


def _project(
    project_root: Path,
    name: str = "project-a",
) -> im.Infra.Workspace.ProjectInfo:
    return im.Infra.Workspace.ProjectInfo.model_validate({
        "name": name,
        "path": project_root,
        "stack": "python/external",
        "has_tests": False,
        "has_src": True,
    })


def _write_project(project_root: Path) -> None:
    (project_root / ".git").mkdir(parents=True, exist_ok=True)
    _ = (project_root / "base.mk").write_text("OLD_BASE\n", encoding="utf-8")
    _ = (project_root / "Makefile").write_text(
        'python "$(WORKSPACE_ROOT)/scripts/check/fix_pyrefly_config.py" "$(PROJECT_NAME)"\n'
        'python "$(WORKSPACE_ROOT)/scripts/check/workspace_check.py" --gates lint "$(PROJECT_NAME)"\n',
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


def _build_migrator(
    project: im.Infra.Workspace.ProjectInfo,
    base_mk: str,
) -> FlextInfraProjectMigrator:
    migrator = FlextInfraProjectMigrator()
    migrator._discovery = _StubDiscovery([project])
    migrator._generator = _StubGenerator(base_mk)
    return migrator


def test_migrator_dry_run_reports_changes_without_writes(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    _write_project(project_root)
    migrator = _build_migrator(_project(project_root), "NEW_BASE\n")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(any(c.startswith("[DRY-RUN]") for c in migrations[0].changes), eq=True)
    tm.that((project_root / "base.mk").read_text(encoding="utf-8"), eq="OLD_BASE\n")


def test_migrator_apply_updates_project_files(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    _write_project(project_root)
    migrator = _build_migrator(_project(project_root), "NEW_BASE\n")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    migrations = tm.ok(result)
    tm.that(migrations[0].errors, eq=[])
    tm.that((project_root / "base.mk").read_text(encoding="utf-8"), eq="NEW_BASE\n")
    makefile_text = (project_root / "Makefile").read_text(encoding="utf-8")
    tm.that("scripts/check/workspace_check.py" not in makefile_text, eq=True)
    tm.that("python -m flext_infra check run" in makefile_text, eq=True)


def test_migrator_handles_missing_pyproject_gracefully(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir(parents=True, exist_ok=True)
    _ = (project_root / "base.mk").write_text("OLD_BASE\n", encoding="utf-8")
    _ = (project_root / "Makefile").write_text("", encoding="utf-8")
    migrator = _build_migrator(_project(project_root), "NEW_BASE\n")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    tm.ok(result)
    tm.that((project_root / "base.mk").read_text(encoding="utf-8"), eq="NEW_BASE\n")


def test_migrator_preserves_custom_makefile_content(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    _write_project(project_root)
    custom = "# Custom rule\ncustom-target:\n\t@echo 'custom'\n"
    (project_root / "Makefile").write_text(custom, encoding="utf-8")
    migrator = _build_migrator(_project(project_root), "NEW_BASE\n")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    tm.ok(result)
    text = (project_root / "Makefile").read_text(encoding="utf-8")
    tm.that(text, has="custom-target")
    tm.that(text, has="@echo 'custom'")


def test_migrator_execute_returns_failure() -> None:
    tm.fail(FlextInfraProjectMigrator().execute())


def test_migrator_workspace_root_not_exists(tmp_path: Path) -> None:
    migrator = FlextInfraProjectMigrator()
    result = migrator.migrate(workspace_root=tmp_path / "nonexistent", dry_run=False)
    tm.fail(result, has="does not exist")


def test_migrator_discovery_failure(tmp_path: Path) -> None:
    migrator = FlextInfraProjectMigrator()
    migrator._discovery = _StubDiscoveryFail("Discovery failed")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    tm.fail(result, has="Discovery failed")


def test_migrator_workspace_root_project_detection(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "Makefile").touch()
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    migrator = FlextInfraProjectMigrator()
    migrator._discovery = _StubDiscovery([])
    migrator._generator = _StubGenerator("base.mk")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(len(migrations), gte=1)


def test_migrator_no_changes_needed(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base.mk", encoding="utf-8")
    (project_root / "Makefile").write_text("migrated", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        '[project]\ndependencies = ["flext-core @ ../flext-core"]\n',
        encoding="utf-8",
    )
    (project_root / ".gitignore").write_text(
        ".reports/\n.venv/\n__pycache__/\n",
        encoding="utf-8",
    )
    migrator = _build_migrator(_project(project_root), "base.mk")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    migrations = tm.ok(result)
    tm.that("no changes needed" in migrations[0].changes, eq=True)
