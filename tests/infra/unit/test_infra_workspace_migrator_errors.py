"""Tests for FlextInfraProjectMigrator — write/read failure scenarios.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import t
from flext_infra.workspace.migrator import FlextInfraProjectMigrator
from flext_tests import tm
from tests.infra import h
from tests.infra.models import m as im
from tests.infra.unit.test_infra_workspace_migrator import (
    _build_migrator,
    _project,
    _StubDiscovery,
    _StubGenerator,
)


def _setup_basic(tmp_path: Path) -> tuple[Path, im.Infra.Workspace.ProjectInfo]:
    root = tmp_path / "project-a"
    root.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "base.mk").write_text("base", encoding="utf-8")
    (root / "Makefile").write_text("content", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (root / ".gitignore").write_text("", encoding="utf-8")
    return root, _project(root)


class TestMigratorWriteFailures:
    def test_gitignore_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _root, proj = _setup_basic(tmp_path)
        migrator = _build_migrator(proj, "base")

        def _write_fail(_self: Path, _data: str, **_kw: t.Scalar) -> int:
            msg = "Write failed"
            raise OSError(msg)

        monkeypatch.setattr(Path, "write_text", _write_fail)
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        migration = tm.ok(result)
        tm.that(
            any("Write failed" in err for err in migration[0].errors),
            eq=True,
        )

    def test_basemk_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("old", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (root / ".gitignore").write_text("", encoding="utf-8")
        migrator = _build_migrator(_project(root), "new content")

        def _write_fail(_self: Path, _data: str, **_kw: t.Scalar) -> int:
            msg = "Write failed"
            raise OSError(msg)

        monkeypatch.setattr(Path, "write_text", _write_fail)
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        migration = tm.ok(result)
        tm.that(
            any("Write failed" in err for err in migration[0].errors),
            eq=True,
        )

    def test_makefile_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        h.write_project(root)
        migrator = _build_migrator(_project(root), "base")
        original_write = Path.write_text

        def _selective_write(self: Path, data: str, **kwargs: str | None) -> int:
            if "Makefile" in str(self):
                msg = "Makefile write failed"
                raise OSError(msg)
            return original_write(self, data, **kwargs)

        monkeypatch.setattr(Path, "write_text", _selective_write)
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        migration = tm.ok(result)
        tm.that(
            any("Makefile write failed" in err for err in migration[0].errors),
            eq=True,
        )

    def test_pyproject_write_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _root, proj = _setup_basic(tmp_path)
        migrator = _build_migrator(proj, "base")
        original_write = Path.write_text

        def _selective_write(self: Path, data: str, **kwargs: str | None) -> int:
            if "pyproject.toml" in str(self):
                msg = "pyproject write failed"
                raise OSError(msg)
            return original_write(self, data, **kwargs)

        monkeypatch.setattr(Path, "write_text", _selective_write)
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        migration = tm.ok(result)
        tm.that(
            any("pyproject write failed" in err for err in migration[0].errors),
            eq=True,
        )


class TestMigratorReadFailures:
    def test_gitignore_read_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (root / ".gitignore").write_text("existing", encoding="utf-8")
        migrator = _build_migrator(_project(root), "base")
        original_read = Path.read_text

        def _selective_read(self: Path, **kwargs: str | None) -> str:
            if ".gitignore" in str(self):
                msg = ".gitignore read failed"
                raise OSError(msg)
            return original_read(self, **kwargs)

        monkeypatch.setattr(Path, "read_text", _selective_read)
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        migration = tm.ok(result)
        tm.that(
            any(".gitignore read failed" in err for err in migration[0].errors),
            eq=True,
        )

    def test_basemk_generation_failure(self, tmp_path: Path) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        h.write_project(root)
        proj = _project(root)
        migrator = FlextInfraProjectMigrator()
        migrator._discovery = _StubDiscovery([proj])
        migrator._generator = _StubGenerator(fail="Generation failed")
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        migration = tm.ok(result)
        tm.that(
            any("Generation failed" in err for err in migration[0].errors),
            eq=True,
        )

    def test_pyproject_parse_failure(self, tmp_path: Path) -> None:
        root = tmp_path / "project-a"
        root.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "base.mk").write_text("base.mk", encoding="utf-8")
        (root / "Makefile").write_text("content", encoding="utf-8")
        (root / "pyproject.toml").write_text("invalid toml {", encoding="utf-8")
        (root / ".gitignore").write_text("", encoding="utf-8")
        migrator = _build_migrator(_project(root), "base.mk")
        result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
        migration = tm.ok(result)
        tm.that(
            any("parse failed" in err for err in migration[0].errors),
            eq=True,
        )


__all__: list[str] = []
