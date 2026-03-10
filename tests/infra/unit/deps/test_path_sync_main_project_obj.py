from __future__ import annotations

import sys
from pathlib import Path

import pytest
import tomlkit
from tomlkit.toml_document import TOMLDocument

from flext_core import r
from flext_infra import m
from flext_infra.deps import path_sync as path_sync_module
from flext_tests import tm
from tests.infra import h


class _OutputNoop:
    def info(self, _message: str) -> None:
        return None


def _project(path: Path) -> m.Infra.Workspace.ProjectInfo:
    return m.Infra.Workspace.ProjectInfo(
        path=path,
        name="test",
        stack="test-stack",
        has_tests=False,
        has_src=False,
    )


def test_main_project_obj_not_dict_first_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "test-project"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").touch()
    monkeypatch.setattr(sys, "argv", ["sync-paths"])
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        lambda _self, _root: r[list[m.Infra.Workspace.ProjectInfo]].ok([
            _project(project_dir)
        ]),
    )
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesToml.read_document",
        lambda _self, _path: r[TOMLDocument].ok(
            tomlkit.parse('[project]\nvalue = "not-a-dict"\n')
        ),
    )
    monkeypatch.setattr(path_sync_module, "output", _OutputNoop())
    tm.that(path_sync_module.main(), eq=0)


def test_main_project_obj_not_dict_second_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["sync-paths"])
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        lambda _self, _root: r[list[m.Infra.Workspace.ProjectInfo]].ok([
            _project(tmp_path / "test-project")
        ]),
    )
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesToml.read_document",
        lambda _self, _path: r[TOMLDocument].ok(
            tomlkit.parse('[project]\nvalue = "not-a-dict"\n')
        ),
    )
    monkeypatch.setattr(path_sync_module, "output", _OutputNoop())
    tm.that(path_sync_module.main(), eq=0)


def test_helpers_alias_is_reachable_project_obj() -> None:
    tm.that(hasattr(h, "assert_ok"), eq=True)
