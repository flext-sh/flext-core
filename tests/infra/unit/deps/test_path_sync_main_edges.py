from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraDependencyPathSync
from flext_infra.deps import path_sync as path_sync_module
from flext_tests import tm
from tests.infra import m


def _project(path: Path, name: str = "flext-core") -> m.Infra.Workspace.ProjectInfo:
    return m.Infra.Workspace.ProjectInfo(
        path=path,
        name=name,
        stack="python",
        has_tests=False,
        has_src=False,
    )


class TestMainEdgeCases:
    def test_main_no_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )

        def _discover_none(
            _root: Path,
        ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
            return r[list[m.Infra.Workspace.ProjectInfo]].ok([])

        monkeypatch.setattr(FlextInfraDependencyPathSync, "ROOT", tmp_path)
        monkeypatch.setattr(
            "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
            _discover_none,
        )
        monkeypatch.setattr(sys, "argv", ["prog", "--workspace", str(tmp_path)])
        tm.that(path_sync_module.main(), eq=0)

    def test_main_with_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )
        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text('[project]\nname = "flext-core"\n')

        def _discover_project(
            _root: Path,
        ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
            return r[list[m.Infra.Workspace.ProjectInfo]].ok([_project(project_dir)])

        monkeypatch.setattr(FlextInfraDependencyPathSync, "ROOT", tmp_path)
        monkeypatch.setattr(
            "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
            _discover_project,
        )
        calls = {"n": 0}

        def rewrite_stub(
            _self: FlextInfraDependencyPathSync,
            _pyproject_path: Path,
            *,
            mode: str,
            internal_names: set[str],
            is_root: bool = False,
            dry_run: bool = False,
        ) -> r[list[str]]:
            _ = _self, _pyproject_path, mode, internal_names, is_root, dry_run
            calls["n"] += 1
            if calls["n"] == 1:
                return r[list[str]].ok([])
            return r[list[str]].ok(["change1"])

        monkeypatch.setattr(
            FlextInfraDependencyPathSync, "rewrite_dep_paths", rewrite_stub
        )
        monkeypatch.setattr(sys, "argv", ["prog", "--workspace", str(tmp_path)])
        tm.that(path_sync_module.main(), eq=0)

    def test_main_root_project_name_extraction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )

        def _discover_none(
            _root: Path,
        ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
            return r[list[m.Infra.Workspace.ProjectInfo]].ok([])

        monkeypatch.setattr(FlextInfraDependencyPathSync, "ROOT", tmp_path)
        monkeypatch.setattr(
            "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
            _discover_none,
        )
        monkeypatch.setattr(sys, "argv", ["prog", "--workspace", str(tmp_path)])
        tm.that(path_sync_module.main(), eq=0)

    def test_main_project_name_extraction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )
        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text('[project]\nname = "flext-core"\n')

        def _discover_project(
            _root: Path,
        ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
            return r[list[m.Infra.Workspace.ProjectInfo]].ok([_project(project_dir)])

        monkeypatch.setattr(FlextInfraDependencyPathSync, "ROOT", tmp_path)
        monkeypatch.setattr(
            "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
            _discover_project,
        )

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

        monkeypatch.setattr(
            FlextInfraDependencyPathSync, "rewrite_dep_paths", _rewrite_ok
        )
        monkeypatch.setattr(sys, "argv", ["prog", "--workspace", str(tmp_path)])
        tm.that(path_sync_module.main(), eq=0)

    def test_main_invalid_project_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "pyproject.toml").write_text("invalid toml [[[")
        monkeypatch.setattr(FlextInfraDependencyPathSync, "ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", "--workspace", str(tmp_path)])
        tm.that(path_sync_module.main(), eq=1)

    def test_main_missing_root_pyproject(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(FlextInfraDependencyPathSync, "ROOT", tmp_path)

        def _discover_none(
            _root: Path,
        ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
            return r[list[m.Infra.Workspace.ProjectInfo]].ok([])

        monkeypatch.setattr(
            "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
            _discover_none,
        )
        monkeypatch.setattr(sys, "argv", ["prog", "--workspace", str(tmp_path)])
        tm.that(path_sync_module.main(), eq=0)

    def test_main_project_without_pyproject(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )
        project_dir = tmp_path / "flext-core"
        project_dir.mkdir()
        monkeypatch.setattr(FlextInfraDependencyPathSync, "ROOT", tmp_path)

        def _discover_project(
            _root: Path,
        ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
            return r[list[m.Infra.Workspace.ProjectInfo]].ok([_project(project_dir)])

        monkeypatch.setattr(
            "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
            _discover_project,
        )
        monkeypatch.setattr(sys, "argv", ["prog", "--workspace", str(tmp_path)])
        tm.that(path_sync_module.main(), eq=0)
