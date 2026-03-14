from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import tomlkit
from tomlkit.toml_document import TOMLDocument

from flext_core import r, t
from flext_infra import m
from flext_infra.deps import path_sync as path_sync_module
from flext_tests import tm


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

    def _discover_projects(
        _self: t.Scalar,
        _root: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        return r[list[m.Infra.Workspace.ProjectInfo]].ok([_project(project_dir)])

    def _read_document(_self: t.Scalar, _path: Path) -> r[TOMLDocument]:
        return r[TOMLDocument].ok(tomlkit.parse('[project]\nvalue = "not-a-dict"\n'))

    monkeypatch.setattr(sys, "argv", ["sync-paths"])
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        _discover_projects,
    )
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesToml.read_document",
        _read_document,
    )
    monkeypatch.setattr(path_sync_module, "output", _OutputNoop())
    tm.that(path_sync_module.main(), eq=0)


def test_main_project_obj_not_dict_second_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _discover_projects(
        _self: t.Scalar,
        _root: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        return r[list[m.Infra.Workspace.ProjectInfo]].ok([
            _project(tmp_path / "test-project")
        ])

    def _read_document(_self: t.Scalar, _path: Path) -> r[TOMLDocument]:
        return r[TOMLDocument].ok(tomlkit.parse('[project]\nvalue = "not-a-dict"\n'))

    monkeypatch.setattr(sys, "argv", ["sync-paths"])
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesDiscovery.discover_projects",
        _discover_projects,
    )
    monkeypatch.setattr(
        "flext_infra.FlextInfraUtilitiesToml.read_document",
        _read_document,
    )
    monkeypatch.setattr(path_sync_module, "output", _OutputNoop())
    tm.that(path_sync_module.main(), eq=0)


def test_helpers_alias_is_reachable_project_obj() -> None:
    infra_helpers_module = importlib.import_module("tests.infra.helpers")
    helper_alias = getattr(infra_helpers_module, "h", None)
    tm.that(hasattr(helper_alias, "assert_ok"), eq=True)
