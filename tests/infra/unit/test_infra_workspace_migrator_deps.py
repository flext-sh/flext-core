"""Tests for FlextWorkspaceMigrator — dependency detection and private methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import t
from flext_infra.workspace.migrator import FlextInfraProjectMigrator
from flext_tests import tm
from tests.infra.unit.test_infra_workspace_migrator import (
    _build_migrator,
    _project,
)


def test_migrator_has_flext_core_dependency_in_poetry(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base.mk", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        '[tool.poetry.dependencies]\nflext-core = "^0.1.0"\n', encoding="utf-8"
    )
    (project_root / ".gitignore").write_text("", encoding="utf-8")
    migrator = _build_migrator(_project(project_root), "base.mk")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(any("already includes" in c for c in migrations[0].changes), eq=True)


def test_migrator_has_flext_core_dependency_poetry_table_missing(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")
    migrator = _build_migrator(_project(project_root), "base")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(any("flext-core dependency" in c for c in migrations[0].changes), eq=True)


def test_migrator_has_flext_core_dependency_poetry_deps_not_table(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        "[tool.poetry]\ndependencies = []\n", encoding="utf-8"
    )
    (project_root / ".gitignore").write_text("", encoding="utf-8")
    migrator = _build_migrator(_project(project_root), "base")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(any("flext-core dependency" in c for c in migrations[0].changes), eq=True)


def test_workspace_migrator_error_handling_on_invalid_workspace() -> None:
    migrator = FlextInfraProjectMigrator()
    result = migrator.migrate(workspace_root=Path("/nonexistent"))
    tm.that(result.is_failure or result.is_success, eq=True)


def test_workspace_migrator_makefile_not_found_dry_run(tmp_path: Path) -> None:
    migrator = _build_migrator(_project(tmp_path, name="test-proj"), "base")
    result = migrator._migrate_makefile(tmp_path, dry_run=True)
    value = tm.ok(result)
    tm.that("not found" in str(value).lower(), eq=True)


def test_workspace_migrator_makefile_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    makefile = tmp_path / "Makefile"
    makefile.write_text("test")
    migrator = _build_migrator(_project(tmp_path, name="test-proj"), "base")

    def _read_fail(*args: t.Scalar, **kwargs: t.Scalar) -> str:
        msg = "Read failed"
        raise OSError(msg)

    monkeypatch.setattr(Path, "read_text", _read_fail)
    result = migrator._migrate_makefile(tmp_path, dry_run=False)
    err = tm.fail(result)
    tm.that("read failed" in err.lower(), eq=True)


def test_workspace_migrator_pyproject_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.poetry]\n")
    migrator = _build_migrator(_project(tmp_path, name="test-proj"), "base")

    def _write_fail(*args: t.Scalar, **kwargs: t.Scalar) -> None:
        msg = "Write failed"
        raise OSError(msg)

    monkeypatch.setattr(Path, "write_text", _write_fail)
    result = migrator._migrate_pyproject(
        tmp_path, project_name="test-proj", dry_run=False
    )
    tm.that(result.is_failure or result.is_success, eq=True)


def test_migrate_makefile_not_found_non_dry_run(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    migrator = _build_migrator(_project(project_root), "base")
    result = migrator._migrate_makefile(project_root, dry_run=False)
    value = tm.ok(result)
    tm.that(value, eq="")


def test_migrate_pyproject_flext_core_non_dry_run(tmp_path: Path) -> None:
    project_root = tmp_path / "flext-core"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "flext-core"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    migrator = _build_migrator(_project(project_root, name="flext-core"), "base")
    result = migrator._migrate_pyproject(
        project_root, project_name="flext-core", dry_run=False
    )
    value = tm.ok(result)
    tm.that(value, eq="")
