"""Tests for FlextInfraProjectMigrator — pyproject and dry-run scenarios.

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


class TestMigratorFlextCore:
    def test_flext_core_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "flext-core"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base.mk", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (root / ".gitignore").write_text("", encoding="utf-8")
        migrator = _build_migrator(_project(root, "flext-core"), "base.mk")
        result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
        migration = tm.ok(result)
        tm.that(
            any("unchanged for flext-core" in c for c in migration[0].changes),
            eq=True,
        )

    def test_flext_core_dry_run(self, tmp_path: Path) -> None:
        root = tmp_path / "flext-core"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (root / ".gitignore").write_text("", encoding="utf-8")
        migrator = _build_migrator(_project(root, "flext-core"), "base")
        result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
        migration = tm.ok(result)
        tm.that(
            any(
                "[DRY-RUN]" in c and "unchanged for flext-core" in c
                for c in migration[0].changes
            ),
            eq=True,
        )


class TestMigratorPoetryDeps:
    def test_has_flext_core_in_poetry(self, tmp_path: Path) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base.mk", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\nflext-core = "^0.1.0"\n',
            encoding="utf-8",
        )
        (root / ".gitignore").write_text("", encoding="utf-8")
        migrator = _build_migrator(_project(root), "base.mk")
        result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
        migration = tm.ok(result)
        tm.that(
            any("already includes" in c for c in migration[0].changes),
            eq=True,
        )

    def test_poetry_table_missing(self, tmp_path: Path) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / "pyproject.toml").write_text("[tool]\n", encoding="utf-8")
        (root / ".gitignore").write_text("", encoding="utf-8")
        migrator = _build_migrator(_project(root), "base")
        result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
        migration = tm.ok(result)
        tm.that(
            any("flext-core dependency" in c for c in migration[0].changes),
            eq=True,
        )

    def test_poetry_deps_not_table(self, tmp_path: Path) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / "pyproject.toml").write_text(
            "[tool.poetry]\ndependencies = []\n",
            encoding="utf-8",
        )
        (root / ".gitignore").write_text("", encoding="utf-8")
        migrator = _build_migrator(_project(root), "base")
        result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
        migration = tm.ok(result)
        tm.that(
            any("flext-core dependency" in c for c in migration[0].changes),
            eq=True,
        )


class TestMigratorDryRun:
    def test_makefile_not_found(self, tmp_path: Path) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (root / ".gitignore").write_text("", encoding="utf-8")
        migrator = _build_migrator(_project(root), "base")
        result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
        migration = tm.ok(result)
        tm.that(
            any(
                "[DRY-RUN]" in c and "Makefile not found" in c
                for c in migration[0].changes
            ),
            eq=True,
        )

    def test_pyproject_not_found(self, tmp_path: Path) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / ".gitignore").write_text("", encoding="utf-8")
        migrator = _build_migrator(_project(root), "base")
        result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
        migration = tm.ok(result)
        tm.that(
            any(
                "[DRY-RUN]" in c and "pyproject.toml not found" in c
                for c in migration[0].changes
            ),
            eq=True,
        )

    def test_gitignore_already_normalized(self, tmp_path: Path) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (root / ".gitignore").write_text(
            ".reports/\n.venv/\n__pycache__/\n",
            encoding="utf-8",
        )
        migrator = _build_migrator(_project(root), "base")
        result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
        migration = tm.ok(result)
        tm.that(
            any(
                "[DRY-RUN]" in c and ".gitignore already normalized" in c
                for c in migration[0].changes
            ),
            eq=True,
        )

    def test_makefile_read_failure(self, tmp_path: Path) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base.mk", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (root / ".gitignore").write_text("", encoding="utf-8")
        migrator = _build_migrator(_project(root), "base.mk")
        tm.ok(migrator.migrate(workspace_root=tmp_path, dry_run=False))


__all__: list[str] = []
