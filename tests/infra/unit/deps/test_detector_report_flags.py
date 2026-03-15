from __future__ import annotations

import types
from pathlib import Path

import pytest

import flext_infra as detector_module
from flext_core import r
from flext_tests import tm


class _ReportStub:
    def __init__(self, raw_count: int) -> None:
        self._raw_count = raw_count

    def model_dump(self) -> dict[str, dict[str, int]]:
        return {"deptry": {"raw_count": self._raw_count}}


class _DepsStub:
    def __init__(self, project: Path, raw_count: int, pip_exit: int) -> None:
        self._project = project
        self._raw_count = raw_count
        self._pip_exit = pip_exit

    def discover_project_paths(
        self,
        root: Path,
        *,
        projects_filter: list[str] | None = None,
    ) -> r[list[Path]]:
        del root
        del projects_filter
        return r[list[Path]].ok([self._project])

    def run_deptry(
        self, project_path: Path, venv_bin: Path
    ) -> r[tuple[list[dict[str, str]], int]]:
        del project_path
        del venv_bin
        return r[tuple[list[dict[str, str]], int]].ok(([], 0))

    def build_project_report(
        self, project_name: str, issues: list[dict[str, str]]
    ) -> _ReportStub:
        del project_name
        del issues
        return _ReportStub(self._raw_count)

    def run_pip_check(self, root: Path, venv_bin: Path) -> r[tuple[list[str], int]]:
        del root
        del venv_bin
        return r[tuple[list[str], int]].ok(([], self._pip_exit))


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    deps: _DepsStub,
) -> detector_module.FlextInfraRuntimeDevDependencyDetector:
    def _workspace_root_from_file(path: str) -> r[Path]:
        del path
        return r[Path].ok(tmp_path)

    def _exists(path: Path) -> bool:
        del path
        return True

    paths = types.SimpleNamespace(workspace_root_from_file=_workspace_root_from_file)

    def _paths_factory() -> types.SimpleNamespace:
        return paths

    def _deps_factory() -> _DepsStub:
        return deps

    monkeypatch.setattr(detector_module, "FlextInfraUtilitiesPaths", _paths_factory)
    monkeypatch.setattr(
        detector_module, "FlextInfraDependencyDetectionService", _deps_factory
    )
    monkeypatch.setattr(Path, "exists", _exists)
    return detector_module.FlextInfraRuntimeDevDependencyDetector()


class TestDetectorReportFlags:
    def test_run_with_issues_and_pip_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        detector = _setup(monkeypatch, tmp_path, _DepsStub(tmp_path / "proj-a", 5, 1))
        tm.that(tm.ok(detector.run(["--dry-run"])), eq=1)

    def test_run_with_no_fail_flag_with_issues(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        detector = _setup(monkeypatch, tmp_path, _DepsStub(tmp_path / "proj-a", 5, 1))
        tm.that(tm.ok(detector.run(["--no-fail", "--dry-run"])), eq=0)

    def test_run_with_json_stdout_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        detector = _setup(monkeypatch, tmp_path, _DepsStub(tmp_path / "proj-a", 0, 0))
        tm.that(tm.ok(detector.run(["--format", "json", "--no-pip-check"])), eq=0)
