from __future__ import annotations

import types
from pathlib import Path

import pytest

import flext_infra.deps as detector_module
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
        _ = root
        _ = projects_filter
        return r[list[Path]].ok([self._project])

    def run_deptry(
        self, project_path: Path, venv_bin: Path
    ) -> r[tuple[list[dict[str, str]], int]]:
        _ = project_path
        _ = venv_bin
        return r[tuple[list[dict[str, str]], int]].ok(([], 0))

    def build_project_report(
        self, project_name: str, issues: list[dict[str, str]]
    ) -> _ReportStub:
        _ = project_name
        _ = issues
        return _ReportStub(self._raw_count)

    def run_pip_check(self, root: Path, venv_bin: Path) -> r[tuple[list[str], int]]:
        _ = root
        _ = venv_bin
        return r[tuple[list[str], int]].ok(([], self._pip_exit))


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    deps: _DepsStub,
    *,
    json_service: types.SimpleNamespace | None = None,
    reporting_service: types.SimpleNamespace | None = None,
) -> detector_module.FlextInfraRuntimeDevDependencyDetector:
    def _workspace_root_from_file(path: str) -> r[Path]:
        del path
        return r[Path].ok(tmp_path)

    def _exists(path: Path) -> bool:
        del path
        return True

    def _paths_factory() -> types.SimpleNamespace:
        return paths

    def _deps_factory() -> _DepsStub:
        return deps

    def _json_factory() -> types.SimpleNamespace | None:
        return json_service

    def _reporting_factory() -> types.SimpleNamespace | None:
        return reporting_service

    paths = types.SimpleNamespace(workspace_root_from_file=_workspace_root_from_file)
    monkeypatch.setattr(detector_module, "FlextInfraUtilitiesPaths", _paths_factory)
    monkeypatch.setattr(
        detector_module, "FlextInfraDependencyDetectionService", _deps_factory
    )
    monkeypatch.setattr(Path, "exists", _exists)
    if json_service is not None:
        monkeypatch.setattr(detector_module, "FlextInfraUtilitiesIo", _json_factory)
    if reporting_service is not None:
        monkeypatch.setattr(
            detector_module, "FlextInfraUtilitiesReporting", _reporting_factory
        )
    return detector_module.FlextInfraRuntimeDevDependencyDetector()


class TestFlextInfraRuntimeDevDependencyDetectorRunReport:
    def test_run_with_output_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        call_paths: list[str] = []

        def _write_json(
            path: Path, payload: dict[str, dict[str, dict[str, int]]]
        ) -> r[bool]:
            _ = payload
            call_paths.append(str(path))
            return r[bool].ok(True)

        custom_output = tmp_path / "custom_report.json"
        detector = _setup(
            monkeypatch,
            tmp_path,
            _DepsStub(tmp_path / "proj-a", 0, 0),
            json_service=types.SimpleNamespace(write_json=_write_json),
        )
        tm.ok(
            detector.run([
                "--output",
                str(custom_output),
                "--no-pip-check",
                "--apply",
            ])
        )
        tm.that(len(call_paths), eq=1)
        tm.that(call_paths[0], eq=str(custom_output))

    def test_run_with_report_directory_creation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        def _report_dir(root: Path, category: str, name: str) -> Path:
            del root
            del category
            del name
            return tmp_path / "readonly"

        reporting = types.SimpleNamespace(get_report_dir=_report_dir)

        def _mkdir_fail(
            path: Path, *, parents: bool = False, exist_ok: bool = False
        ) -> None:
            del path
            del parents
            del exist_ok
            msg = "Permission denied"
            raise OSError(msg)

        monkeypatch.setattr(Path, "mkdir", _mkdir_fail)
        detector = _setup(
            monkeypatch,
            tmp_path,
            _DepsStub(tmp_path / "proj-a", 0, 0),
            reporting_service=reporting,
        )
        tm.that(
            "failed to create report directory"
            in tm.fail(detector.run(["--no-pip-check", "--apply"])),
            eq=True,
        )

    def test_run_with_json_write_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        def _write_json_fail(
            path: Path,
            payload: dict[str, dict[str, dict[str, int]]],
        ) -> r[bool]:
            del path
            del payload
            return r[bool].fail("write failed")

        def _report_dir(root: Path, category: str, name: str) -> Path:
            del root
            del category
            del name
            return tmp_path / "reports"

        def _mkdir_ok(
            path: Path, *, parents: bool = False, exist_ok: bool = False
        ) -> None:
            del path
            del parents
            del exist_ok

        json_service = types.SimpleNamespace(write_json=_write_json_fail)
        reporting = types.SimpleNamespace(get_report_dir=_report_dir)
        monkeypatch.setattr(Path, "mkdir", _mkdir_ok)
        detector = _setup(
            monkeypatch,
            tmp_path,
            _DepsStub(tmp_path / "proj-a", 0, 0),
            json_service=json_service,
            reporting_service=reporting,
        )
        error = tm.fail(detector.run(["--no-pip-check", "--apply"]))
        tm.that("write failed" in error or "failed to write report" in error, eq=True)
