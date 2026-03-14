from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import override

from flext_core import r
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator
from flext_infra.workspace.migrator import FlextInfraProjectMigrator
from flext_tests import tm
from tests.infra import h
from tests.infra.models import m as im
from tests.infra.typings import t


class _StubDiscovery:
    """Typed stub for discovery service."""

    def __init__(
        self,
        projects: list[im.Infra.Workspace.ProjectInfo] | None = None,
        *,
        error: str = "",
    ) -> None:
        self._projects = projects or []
        self._error = error

    def discover_projects(
        self, workspace_root: Path
    ) -> r[list[im.Infra.Workspace.ProjectInfo]]:
        _ = workspace_root
        if self._error:
            return r[list[im.Infra.Workspace.ProjectInfo]].fail(self._error)
        return r[list[im.Infra.Workspace.ProjectInfo]].ok(self._projects)


class _StubGenerator(FlextInfraBaseMkGenerator):
    """Typed stub for base.mk generator."""

    def __init__(self, content: str = "", *, fail: str = "") -> None:
        super().__init__()
        self._content = content
        self._fail = fail

    @override
    def generate(
        self,
        config: im.Infra.Basemk.BaseMkConfig | Mapping[str, t.Scalar] | None = None,
    ) -> r[str]:
        if self._fail:
            return r[str].fail(self._fail)
        return r[str].ok(self._content)


def _project(
    project_root: Path,
    name: str = "project-a",
) -> im.Infra.Workspace.ProjectInfo:
    return im.Infra.Workspace.ProjectInfo.model_validate(
        obj={
            "name": name,
            "path": project_root,
            "stack": "python/external",
            "has_tests": False,
            "has_src": True,
        }
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
    h.write_project(project_root)
    migrator = _build_migrator(_project(project_root), "NEW_BASE\n")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=True)
    migrations = tm.ok(result)
    tm.that(any(c.startswith("[DRY-RUN]") for c in migrations[0].changes), eq=True)
    tm.that((project_root / "base.mk").read_text(encoding="utf-8"), eq="OLD_BASE\n")


def test_migrator_apply_updates_project_files(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    h.write_project(project_root)
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
    (project_root / "base.mk").write_text("OLD_BASE\n", encoding="utf-8")
    (project_root / "Makefile").write_text("", encoding="utf-8")
    migrator = _build_migrator(_project(project_root), "NEW_BASE\n")
    result = migrator.migrate(workspace_root=tmp_path, dry_run=False)
    tm.ok(result)
    tm.that((project_root / "base.mk").read_text(encoding="utf-8"), eq="NEW_BASE\n")


def test_migrator_preserves_custom_makefile_content(tmp_path: Path) -> None:
    project_root = tmp_path / "project-a"
    project_root.mkdir(parents=True)
    h.write_project(project_root)
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
    migrator._discovery = _StubDiscovery(error="Discovery failed")
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
