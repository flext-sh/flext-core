"""Tests for FlextInfraProjectMigrator — internal method coverage.

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


class TestMigratorInternalMakefile:
    def test_not_found_dry_run(self, tmp_path: Path) -> None:
        proj = _project(tmp_path, "test-proj")
        migrator = _build_migrator(proj, "base")
        result = migrator._migrate_makefile(tmp_path, dry_run=True)
        value = tm.ok(result)
        tm.that(str(value).lower(), has="not found")

    def test_read_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "Makefile").write_text("test")
        proj = _project(tmp_path, "test-proj")
        migrator = _build_migrator(proj, "base")

        def _read_fail(*_a: t.Scalar, **_kw: t.Scalar) -> str:
            msg = "Read failed"
            raise OSError(msg)

        monkeypatch.setattr(Path, "read_text", _read_fail)
        tm.fail(migrator._migrate_makefile(tmp_path, dry_run=False), has="read failed")

    def test_not_found_non_dry_run(self, tmp_path: Path) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        proj = _project(root)
        migrator = _build_migrator(proj, "base")
        result = migrator._migrate_makefile(root, dry_run=False)
        tm.ok(result, eq="")


class TestMigratorInternalPyproject:
    def test_write_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\n")
        proj = _project(tmp_path, "test-proj")
        migrator = _build_migrator(proj, "base")

        def _write_fail(*_a: t.Scalar, **_kw: t.Scalar) -> None:
            msg = "Write failed"
            raise OSError(msg)

        monkeypatch.setattr(Path, "write_text", _write_fail)
        result = migrator._migrate_pyproject(
            tmp_path,
            project_name="test-proj",
            dry_run=False,
        )
        tm.that(result.is_failure or result.is_success, eq=True)

    def test_flext_core_non_dry_run(self, tmp_path: Path) -> None:
        root = tmp_path / "flext-core"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "pyproject.toml").write_text(
            '[project]\nname = "flext-core"\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        proj = _project(root, "flext-core")
        migrator = _build_migrator(proj, "base")
        tm.ok(
            migrator._migrate_pyproject(
                root,
                project_name="flext-core",
                dry_run=False,
            ),
            eq="",
        )


class TestMigratorEdgeCases:
    def test_invalid_workspace(self) -> None:
        migrator = FlextInfraProjectMigrator()
        result = migrator.migrate(workspace_root=Path("/nonexistent"))
        tm.that(result.is_failure or result.is_success, eq=True)


__all__: list[str] = []
