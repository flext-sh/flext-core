"""Tests for FlextWorkspaceMigrator — dry-run and skip scenarios.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_tests import tm
from tests.infra.unit.test_infra_workspace_migrator import (
    _build_migrator,
    _project,
)


def test_migrator_flext_core_project_skipped(tmp_path: Path) -> None:
    project_root = tmp_path / "flext-core"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base.mk", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")
    migrator = _build_migrator(_project(project_root, name="flext-core"), "base.mk")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(
        any("unchanged for flext-core" in c for c in migrations[0].changes), eq=True
    )


def test_migrator_makefile_not_found_dry_run(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")
    migrator = _build_migrator(_project(project_root), "base")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(
        any(
            "[DRY-RUN]" in c and "Makefile not found" in c
            for c in migrations[0].changes
        ),
        eq=True,
    )


def test_migrator_pyproject_not_found_dry_run(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")
    migrator = _build_migrator(_project(project_root), "base")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(
        any(
            "[DRY-RUN]" in c and "pyproject.toml not found" in c
            for c in migrations[0].changes
        ),
        eq=True,
    )


def test_migrator_flext_core_dry_run(tmp_path: Path) -> None:
    project_root = tmp_path / "flext-core"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")
    migrator = _build_migrator(_project(project_root, name="flext-core"), "base")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(
        any(
            "[DRY-RUN]" in c and "unchanged for flext-core" in c
            for c in migrations[0].changes
        ),
        eq=True,
    )


def test_migrator_gitignore_already_normalized_dry_run(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text(
        ".reports/\n.venv/\n__pycache__/\n", encoding="utf-8"
    )
    migrator = _build_migrator(_project(project_root), "base")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(
        any(
            "[DRY-RUN]" in c and ".gitignore already normalized" in c
            for c in migrations[0].changes
        ),
        eq=True,
    )


def test_migrator_makefile_read_failure(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    (project_root / ".git").mkdir()
    (project_root / "base.mk").write_text("base.mk", encoding="utf-8")
    (project_root / "Makefile").write_text("content", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project_root / ".gitignore").write_text("", encoding="utf-8")
    migrator = _build_migrator(_project(project_root), "base.mk")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    tm.ok(result)
