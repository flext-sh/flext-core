from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from flext_core import r
from flext_infra.deps import detection
from flext_infra.deps.detection import (
    FlextInfraDependencyDetectionService,
    build_project_report,
    classify_issues,
    dm,
    load_dependency_limits,
    module_to_types_package,
)
from flext_tests import tm
from tests.infra import h


class _StubRunner:
    def __init__(self, result) -> None:
        self._result = result

    def run_raw(self, *args, **kwargs):
        _ = args
        _ = kwargs
        return self._result


class _StubService:
    def __init__(self) -> None:
        self.called: dict[str, tuple] = {}

    def discover_projects(self, workspace_root: Path, projects_filter=None):
        self.called["discover_projects"] = (workspace_root, projects_filter)
        return r[list[Path]].ok([workspace_root])

    def run_deptry(self, project_path: Path, venv_bin: Path, **kwargs):
        self.called["run_deptry"] = (project_path, venv_bin, kwargs)
        return r[tuple[list[dict[str, str]], int]].ok(([], 0))

    def run_pip_check(self, workspace_root: Path, venv_bin: Path):
        self.called["run_pip_check"] = (workspace_root, venv_bin)
        return r[tuple[list[str], int]].ok(([], 0))

    def run_mypy_stub_hints(
        self, project_path: Path, venv_bin: Path, *, timeout: int = 300
    ):
        self.called["run_mypy_stub_hints"] = (project_path, venv_bin, timeout)
        return r[tuple[list[str], list[str]]].ok(([], []))

    def get_current_typings_from_pyproject(self, project_path: Path):
        self.called["get_current_typings_from_pyproject"] = (project_path,)
        return ["types-requests"]

    def get_required_typings(
        self,
        project_path: Path,
        venv_bin: Path,
        limits_path=None,
        *,
        include_mypy: bool = True,
    ):
        self.called["get_required_typings"] = (
            project_path,
            venv_bin,
            limits_path,
            include_mypy,
        )
        report = dm.TypingsReport(
            required_packages=[],
            hinted=[],
            missing_modules=[],
            current=[],
            to_add=[],
            to_remove=[],
            limits_applied=False,
            python_version=None,
        )
        return r[dm.TypingsReport].ok(report)

    def load_dependency_limits(self, limits_path=None):
        self.called["load_dependency_limits"] = (limits_path,)
        return {}


class TestModuleLevelWrappers:
    def test_classify_issues_wrapper(self) -> None:
        tm.that(classify_issues([]).dep001, eq=[])

    def test_build_project_report_wrapper(self) -> None:
        tm.that(build_project_report("proj", []).project, eq="proj")

    def test_module_to_types_package_wrapper(self) -> None:
        tm.that(module_to_types_package("yaml", {}), eq="types-pyyaml")

    def test_load_dependency_limits_wrapper(self, monkeypatch) -> None:
        stub = _StubService()
        monkeypatch.setattr(detection, "_service", stub)
        tm.that(load_dependency_limits(), eq={})


class TestDetectionUncoveredLines:
    def test_run_deptry_with_non_dict_issue(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        out_file = project / ".deptry-report.json"
        out_file.write_text(json.dumps(["not_a_dict", {"error": {"code": "DEP001"}}]))
        out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        monkeypatch.setattr(service, "runner", _StubRunner(r[type(out)].ok(out)))
        issues, _ = h.assert_ok(
            service.run_deptry(project, venv_bin, json_output_path=out_file)
        )
        tm.that(len(issues), eq=1)

    def test_run_pip_check_with_empty_output(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "pip").write_text("")
        out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        monkeypatch.setattr(service, "runner", _StubRunner(r[type(out)].ok(out)))
        lines, exit_code = h.assert_ok(service.run_pip_check(tmp_path, venv_bin))
        tm.that(lines, eq=[])
        tm.that(exit_code, eq=0)

    def test_get_required_typings_with_limits_applied(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")
        out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        monkeypatch.setattr(service, "runner", _StubRunner(r[type(out)].ok(out)))

        class _Toml:
            def __init__(self) -> None:
                self._i = 0

            def read_plain(self, path: Path):
                _ = path
                self._i += 1
                if self._i == 1:
                    return r[dict[str, dict[str, str]]].ok({
                        "python": {"version": "3.13"}
                    })
                return r[dict[str, str]].ok({})

        monkeypatch.setattr(service, "toml", _Toml())
        report = h.assert_ok(service.get_required_typings(tmp_path, venv_bin))
        tm.that(report.limits_applied, eq=True)


def test_discover_projects_wrapper(tmp_path: Path, monkeypatch) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    result = detection.discover_projects(tmp_path)
    tm.that(h.assert_ok(result), eq=[tmp_path])
    tm.that(stub.called["discover_projects"], eq=(tmp_path, None))


def test_run_deptry_wrapper(tmp_path: Path, monkeypatch) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    h.assert_ok(detection.run_deptry(tmp_path, venv_bin))
    tm.that(stub.called["run_deptry"][0], eq=tmp_path)


def test_run_pip_check_wrapper(tmp_path: Path, monkeypatch) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    h.assert_ok(detection.run_pip_check(tmp_path, venv_bin))
    tm.that(stub.called["run_pip_check"], eq=(tmp_path, venv_bin))


def test_run_mypy_stub_hints_wrapper(tmp_path: Path, monkeypatch) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    h.assert_ok(detection.run_mypy_stub_hints(tmp_path, venv_bin))
    tm.that(stub.called["run_mypy_stub_hints"][0], eq=tmp_path)


def test_get_current_typings_from_pyproject_wrapper(
    tmp_path: Path, monkeypatch
) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    tm.that(
        detection.get_current_typings_from_pyproject(tmp_path), eq=["types-requests"]
    )
    tm.that(stub.called["get_current_typings_from_pyproject"], eq=(tmp_path,))


def test_get_required_typings_wrapper(tmp_path: Path, monkeypatch) -> None:
    stub = _StubService()
    monkeypatch.setattr(detection, "_service", stub)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    h.assert_ok(detection.get_required_typings(tmp_path, venv_bin))
    tm.that(stub.called["get_required_typings"][0], eq=tmp_path)
