from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flext_core import r
from flext_infra import m
from flext_infra.deps import path_sync as path_sync_module
from flext_tests import tm


def _project(path: Path, name: str = "flext-core") -> m.Infra.Workspace.ProjectInfo:
    return m.Infra.Workspace.ProjectInfo(
        path=path,
        name=name,
        stack="python",
        has_tests=False,
        has_src=False,
    )


class TestMain:
    def test_main_auto_detect_workspace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / ".gitmodules").touch()
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )
        monkeypatch.setattr(path_sync_module, "ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", "--mode", "auto"])
        tm.that(path_sync_module.main(), eq=0)

    def test_main_explicit_workspace_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )
        monkeypatch.setattr(path_sync_module, "ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", "--mode", "workspace"])
        tm.that(path_sync_module.main(), eq=0)

    def test_main_explicit_standalone_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )
        monkeypatch.setattr(path_sync_module, "ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", "--mode", "standalone"])
        tm.that(path_sync_module.main(), eq=0)

    def test_main_dry_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )
        monkeypatch.setattr(path_sync_module, "ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", "--dry-run"])
        tm.that(path_sync_module.main(), eq=0)

    def test_main_specific_projects(
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
        monkeypatch.setattr(path_sync_module, "ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["prog", "--project", "flext-core"])
        tm.that(path_sync_module.main(), eq=0)

    def test_main_discovery_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )
        monkeypatch.setattr(path_sync_module, "ROOT", tmp_path)
        monkeypatch.setattr(
            "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
            lambda _self, _root: r[list[m.Infra.Workspace.ProjectInfo]].fail(
                "discovery failed"
            ),
        )
        monkeypatch.setattr(sys, "argv", ["prog"])
        tm.that(path_sync_module.main(), eq=1)

    def test_main_root_rewrite_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-workspace"\n'
        )
        monkeypatch.setattr(path_sync_module, "ROOT", tmp_path)
        monkeypatch.setattr(
            path_sync_module,
            "rewrite_dep_paths",
            lambda *args, **kwargs: r[list[str]].fail("rewrite failed"),
        )
        monkeypatch.setattr(sys, "argv", ["prog"])
        tm.that(path_sync_module.main(), eq=1)

    def test_main_project_rewrite_failure(
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
        monkeypatch.setattr(path_sync_module, "ROOT", tmp_path)
        monkeypatch.setattr(
            "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
            lambda _self, _root: r[list[m.Infra.Workspace.ProjectInfo]].ok([
                _project(project_dir)
            ]),
        )
        calls = {"n": 0}

        def rewrite_stub(*args: object, **kwargs: object) -> r[list[str]]:
            calls["n"] += 1
            if calls["n"] == 1:
                return r[list[str]].ok([])
            return r[list[str]].fail("project rewrite failed")

        monkeypatch.setattr(path_sync_module, "rewrite_dep_paths", rewrite_stub)
        monkeypatch.setattr(sys, "argv", ["prog"])
        tm.that(path_sync_module.main(), eq=1)


def test_helpers_alias_is_reachable_main() -> None:
    tm.that(hasattr(h, "assert_toml_valid"), eq=True)
