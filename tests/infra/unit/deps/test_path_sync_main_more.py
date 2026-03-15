from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraDependencyPathSync, FlextInfraUtilitiesDiscovery, m
from flext_infra.deps import path_sync as path_sync_module
from flext_tests import tm


def _workspace_root() -> Path:
    return FlextInfraDependencyPathSync.ROOT


def _project(path: Path, name: str = "flext-core") -> m.Infra.Workspace.ProjectInfo:
    return m.Infra.Workspace.ProjectInfo(
        path=path,
        name=name,
        stack="python",
        has_tests=False,
        has_src=False,
    )


class _OutputRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def info(self, message: str) -> None:
        self.calls.append(message)


def test_main_project_invalid_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "flext-workspace"\n')
    project_dir = tmp_path / "flext-core"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("invalid toml [[[")

    def _discover_project(
        _self: FlextInfraUtilitiesDiscovery,
        _root: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        return r[list[m.Infra.Workspace.ProjectInfo]].ok([_project(project_dir)])

    monkeypatch.setattr(FlextInfraDependencyPathSync, "ROOT", tmp_path)
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        _discover_project,
    )
    monkeypatch.setattr(sys, "argv", ["prog"])
    tm.that(path_sync_module.main(), eq=1)


def test_main_project_no_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "flext-workspace"\n')
    project_dir = tmp_path / "flext-core"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\n")

    def _discover_project(
        _self: FlextInfraUtilitiesDiscovery,
        _root: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        return r[list[m.Infra.Workspace.ProjectInfo]].ok([_project(project_dir)])

    monkeypatch.setattr(FlextInfraDependencyPathSync, "ROOT", tmp_path)
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        _discover_project,
    )
    monkeypatch.setattr(sys, "argv", ["prog"])
    tm.that(path_sync_module.main(), eq=0)


def test_main_project_non_string_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "flext-workspace"\n')
    project_dir = tmp_path / "flext-core"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname = 123\n")

    def _discover_project(
        _self: FlextInfraUtilitiesDiscovery,
        _root: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        return r[list[m.Infra.Workspace.ProjectInfo]].ok([_project(project_dir)])

    monkeypatch.setattr(FlextInfraDependencyPathSync, "ROOT", tmp_path)
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        _discover_project,
    )
    monkeypatch.setattr(sys, "argv", ["prog"])
    tm.that(path_sync_module.main(), eq=0)


def test_main_discovery_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _discover_fail(
        _self: FlextInfraUtilitiesDiscovery,
        _root: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        return r[list[m.Infra.Workspace.ProjectInfo]].fail("discovery failed")

    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        _discover_fail,
    )
    monkeypatch.setattr(sys, "argv", ["sync-paths"])
    tm.that(path_sync_module.main(), eq=1)


def test_main_no_changes_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _discover_none(
        _self: FlextInfraUtilitiesDiscovery,
        _root: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        return r[list[m.Infra.Workspace.ProjectInfo]].ok([])

    def _rewrite_ok(
        _self: FlextInfraDependencyPathSync,
        _pyproject_path: Path,
        *,
        mode: str,
        internal_names: set[str],
        is_root: bool = False,
        dry_run: bool = False,
    ) -> r[list[str]]:
        _ = _self, _pyproject_path, mode, internal_names, is_root, dry_run
        return r[list[str]].ok([])

    monkeypatch.setattr(sys, "argv", ["sync-paths"])
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        _discover_none,
    )
    monkeypatch.setattr(FlextInfraDependencyPathSync, "rewrite_dep_paths", _rewrite_ok)
    tm.that(path_sync_module.main(), eq=0)


def test_workspace_root_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "f"
    deep.mkdir(parents=True)

    def _resolve(_self: Path) -> Path:
        return deep / "test.py"

    monkeypatch.setattr(Path, "resolve", _resolve)
    tm.that(_workspace_root().is_absolute(), eq=True)


def test_main_with_changes_and_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _OutputRecorder()

    def _discover_none(
        _self: FlextInfraUtilitiesDiscovery,
        _root: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        return r[list[m.Infra.Workspace.ProjectInfo]].ok([])

    def _rewrite_changes(
        _self: FlextInfraDependencyPathSync,
        _pyproject_path: Path,
        *,
        mode: str,
        internal_names: set[str],
        is_root: bool = False,
        dry_run: bool = False,
    ) -> r[list[str]]:
        _ = _self, _pyproject_path, mode, internal_names, is_root, dry_run
        return r[list[str]].ok(["  PEP621: old -> new"])

    monkeypatch.setattr(sys, "argv", ["sync-paths", "--dry-run"])
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        _discover_none,
    )
    monkeypatch.setattr(
        FlextInfraDependencyPathSync, "rewrite_dep_paths", _rewrite_changes
    )
    monkeypatch.setattr(path_sync_module, "output", recorder)
    tm.that(path_sync_module.main(), eq=0)
    tm.that(any("[DRY-RUN]" in call for call in recorder.calls), eq=True)


def test_main_with_changes_no_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _OutputRecorder()

    def _discover_none(
        _self: FlextInfraUtilitiesDiscovery,
        _root: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        return r[list[m.Infra.Workspace.ProjectInfo]].ok([])

    def _rewrite_changes(
        _self: FlextInfraDependencyPathSync,
        _pyproject_path: Path,
        *,
        mode: str,
        internal_names: set[str],
        is_root: bool = False,
        dry_run: bool = False,
    ) -> r[list[str]]:
        _ = _self, _pyproject_path, mode, internal_names, is_root, dry_run
        return r[list[str]].ok(["  PEP621: old -> new"])

    monkeypatch.setattr(sys, "argv", ["sync-paths"])
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        _discover_none,
    )
    monkeypatch.setattr(
        FlextInfraDependencyPathSync, "rewrite_dep_paths", _rewrite_changes
    )
    monkeypatch.setattr(path_sync_module, "output", recorder)
    tm.that(path_sync_module.main(), eq=0)
    tm.that(len(recorder.calls) > 0, eq=True)
