from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import (
    build_project_report,
    classify_issues,
    dm,
    load_dependency_limits,
    module_to_types_package,
)
from flext_infra.deps import detection
from flext_tests import tm


class _StubService:
    def __init__(self) -> None:
        self.called: dict[str, tuple[object, ...]] = {}

    def discover_project_paths(
        self, workspace_root: Path, projects_filter: list[str] | None = None
    ) -> r[list[Path]]:
        self.called["discover_project_paths"] = (workspace_root, projects_filter)
        return r[list[Path]].ok([workspace_root])

    def run_deptry(
        self,
        project_path: Path,
        venv_bin: Path,
        *,
        config_path: Path | None = None,
        json_output_path: Path | None = None,
        extend_exclude: list[str] | None = None,
    ) -> r[tuple[list[dict[str, str]], int]]:
        self.called["run_deptry"] = (
            project_path,
            venv_bin,
            config_path,
            json_output_path,
            extend_exclude,
        )
        return r[tuple[list[dict[str, str]], int]].ok(([], 0))

    def run_pip_check(
        self, workspace_root: Path, venv_bin: Path
    ) -> r[tuple[list[str], int]]:
        self.called["run_pip_check"] = (workspace_root, venv_bin)
        return r[tuple[list[str], int]].ok(([], 0))

    def run_mypy_stub_hints(
        self, project_path: Path, venv_bin: Path, *, timeout: int = 300
    ) -> r[tuple[list[str], list[str]]]:
        self.called["run_mypy_stub_hints"] = (project_path, venv_bin, timeout)
        return r[tuple[list[str], list[str]]].ok(([], []))

    def get_current_typings_from_pyproject(self, project_path: Path) -> list[str]:
        self.called["get_current_typings_from_pyproject"] = (project_path,)
        return ["types-requests"]

    def get_required_typings(
        self,
        project_path: Path,
        venv_bin: Path,
        limits_path: Path | None = None,
        *,
        include_mypy: bool = True,
    ) -> r[dm.TypingsReport]:
        self.called["get_required_typings"] = (
            project_path,
            venv_bin,
            limits_path,
            include_mypy,
        )
        return r[dm.TypingsReport].ok(
            dm.TypingsReport(
                required_packages=[],
                hinted=[],
                missing_modules=[],
                current=[],
                to_add=[],
                to_remove=[],
            )
        )

    def load_dependency_limits(self, limits_path: Path | None = None) -> dict[str, str]:
        self.called["load_dependency_limits"] = (limits_path,)
        return {}


class TestModuleLevelWrappers:
    def test_classify_issues_wrapper(self) -> None:
        assert classify_issues([]).dep001 == []

    def test_build_project_report_wrapper(self) -> None:
        tm.that(build_project_report("proj", []).project, eq="proj")

    def test_module_to_types_package_wrapper(self) -> None:
        tm.that(module_to_types_package("yaml", {}), eq="types-pyyaml")

    def test_load_dependency_limits_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(detection, "_service", _StubService())
        assert load_dependency_limits() == {}


def test_discover_projects_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    result = detection.discover_project_paths(tmp_path)
    tm.that(result.is_success, eq=True)
    assert stub.called["discover_project_paths"][1] is None


def test_run_deptry_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    tm.that(detection.run_deptry(tmp_path, venv_bin).is_success, eq=True)
    tm.that(str(stub.called["run_deptry"][0]), eq=str(tmp_path))


def test_run_pip_check_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    tm.that(detection.run_pip_check(tmp_path, venv_bin).is_success, eq=True)
    tm.that(str(stub.called["run_pip_check"][0]), eq=str(tmp_path))


def test_run_mypy_stub_hints_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    tm.that(detection.run_mypy_stub_hints(tmp_path, venv_bin).is_success, eq=True)
    tm.that(str(stub.called["run_mypy_stub_hints"][0]), eq=str(tmp_path))


def test_get_current_typings_from_pyproject_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    tm.that(
        detection.get_current_typings_from_pyproject(tmp_path), eq=["types-requests"]
    )
    tm.that(str(stub.called["get_current_typings_from_pyproject"][0]), eq=str(tmp_path))


def test_get_required_typings_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    tm.that(detection.get_required_typings(tmp_path, venv_bin).is_success, eq=True)
    tm.that(str(stub.called["get_required_typings"][0]), eq=str(tmp_path))
