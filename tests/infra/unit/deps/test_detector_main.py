from __future__ import annotations

import types
from pathlib import Path

import pytest

import flext_infra.deps as detector_module
from flext_core import r
from flext_tests import tm


class _ReportStub:
    def model_dump(self) -> dict[str, dict[str, int]]:
        return {"deptry": {"raw_count": 0}}


class _TypingsStub:
    def __init__(self, to_add: list[str | int | None]) -> None:
        self.to_add = to_add

    def model_dump(self) -> dict[str, list[str | int | None]]:
        return {"to_add": self.to_add}


class _DepsStub:
    def __init__(self, project: Path, to_add: list[str | int | None]) -> None:
        self._project = project
        self._to_add = to_add

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
        return _ReportStub()

    def get_required_typings(
        self,
        project_path: Path,
        venv_bin: Path,
        *,
        limits_path: Path,
    ) -> r[_TypingsStub]:
        _ = project_path
        _ = venv_bin
        _ = limits_path
        return r[_TypingsStub].ok(_TypingsStub(self._to_add))

    def load_dependency_limits(self, limits_path: Path) -> dict[str, dict[str, str]]:
        del limits_path
        return {}

    def run_pip_check(self, root: Path, venv_bin: Path) -> r[tuple[list[str], int]]:
        _ = root
        _ = venv_bin
        return r[tuple[list[str], int]].ok(([], 0))


def _setup_typings_detector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    to_add: list[str | int | None],
    run_raw_result: r[types.SimpleNamespace],
) -> tuple[
    detector_module.FlextInfraRuntimeDevDependencyDetector, list[list[str | int | None]]
]:
    project_path = tmp_path / "proj-a"
    (project_path / "src").mkdir(parents=True)
    captured_commands: list[list[str | int | None]] = []

    def _run_raw(
        cmd: list[str | int | None],
        *,
        cwd: Path,
        timeout: int,
        env: dict[str, str],
    ) -> r[types.SimpleNamespace]:
        _ = cwd
        _ = timeout
        _ = env
        captured_commands.append(cmd)
        return run_raw_result

    def _workspace_root_from_file(path: str) -> r[Path]:
        _ = path
        return r[Path].ok(tmp_path)

    def _exists(path: Path) -> bool:
        _ = path
        return True

    paths = types.SimpleNamespace(workspace_root_from_file=_workspace_root_from_file)
    deps = _DepsStub(project_path, to_add)
    runner = types.SimpleNamespace(run_raw=_run_raw)
    monkeypatch.setattr(detector_module, "FlextInfraUtilitiesPaths", lambda: paths)
    monkeypatch.setattr(
        detector_module, "FlextInfraDependencyDetectionService", lambda: deps
    )
    monkeypatch.setattr(
        detector_module, "FlextInfraUtilitiesSubprocess", lambda: runner
    )
    monkeypatch.setattr(Path, "exists", _exists)
    return detector_module.FlextInfraRuntimeDevDependencyDetector(), captured_commands


class TestFlextInfraRuntimeDevDependencyDetectorRunTypings:
    def test_run_with_apply_typings_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_result = r[types.SimpleNamespace].ok(types.SimpleNamespace(exit_code=0))
        detector, calls = _setup_typings_detector(
            monkeypatch, tmp_path, ["types-requests"], run_result
        )
        tm.ok(
            detector.run([
                "--workspace",
                str(tmp_path),
                "--typings",
                "--apply-typings",
                "--apply",
                "--no-pip-check",
            ])
        )
        tm.that(len(calls), eq=1)

    def test_run_with_apply_typings_non_string_package(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_result = r[types.SimpleNamespace].ok(types.SimpleNamespace(exit_code=0))
        detector, calls = _setup_typings_detector(
            monkeypatch,
            tmp_path,
            ["types-requests", 123, None],
            run_result,
        )
        tm.ok(
            detector.run([
                "--workspace",
                str(tmp_path),
                "--typings",
                "--apply-typings",
                "--apply",
                "--no-pip-check",
            ])
        )
        tm.that(len(calls), eq=3)

    def test_run_with_apply_typings_poetry_add_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_result = r[types.SimpleNamespace].ok(types.SimpleNamespace(exit_code=1))
        detector, _ = _setup_typings_detector(
            monkeypatch, tmp_path, ["types-requests"], run_result
        )
        tm.ok(detector.run(["--typings", "--apply-typings", "--no-pip-check"]))

    def test_run_with_apply_typings_poetry_add_failure_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        detector, _ = _setup_typings_detector(
            monkeypatch,
            tmp_path,
            ["types-requests"],
            r[types.SimpleNamespace].fail("poetry add failed"),
        )
        tm.ok(detector.run(["--typings", "--apply-typings", "--no-pip-check"]))


class TestMainFunction:
    def test_main_returns_failure_code_on_run_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _DetectorStub:
            def run(self, argv: list[str] | None = None) -> r[int]:
                _ = argv
                return r[int].fail("boom")

        monkeypatch.setattr(
            detector_module, "FlextInfraRuntimeDevDependencyDetector", _DetectorStub
        )
        tm.that(detector_module.main(), eq=1)
