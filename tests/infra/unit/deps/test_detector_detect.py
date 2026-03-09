from __future__ import annotations

import types
from pathlib import Path

import pytest

import flext_infra.deps.detector as detector_module
from flext_core import r
from flext_tests import tm


class _ReportStub:
    def __init__(self, raw_count: int) -> None:
        self._raw_count = raw_count

    def model_dump(self) -> dict[str, dict[str, int]]:
        return {"deptry": {"raw_count": self._raw_count}}


class _DepsStub:
    def __init__(self, project_paths: list[Path]) -> None:
        self.project_paths = project_paths
        self.discovery_failure: str | None = None
        self.deptry_failure: str | None = None
        self.typings_failure: str | None = None

    def discover_projects(
        self,
        root: Path,
        *,
        projects_filter: list[str] | None = None,
    ) -> r[list[Path]]:
        _ = root
        _ = projects_filter
        if self.discovery_failure is not None:
            return r[list[Path]].fail(self.discovery_failure)
        return r[list[Path]].ok(self.project_paths)

    def run_deptry(
        self, project_path: Path, venv_bin: Path
    ) -> r[tuple[list[dict[str, str]], int]]:
        _ = project_path
        _ = venv_bin
        if self.deptry_failure is not None:
            return r[tuple[list[dict[str, str]], int]].fail(self.deptry_failure)
        return r[tuple[list[dict[str, str]], int]].ok(([], 0))

    def build_project_report(
        self, project_name: str, issues: list[dict[str, str]]
    ) -> _ReportStub:
        _ = project_name
        _ = issues
        return _ReportStub(0)

    def get_required_typings(
        self,
        project_path: Path,
        venv_bin: Path,
        *,
        limits_path: Path,
    ) -> r[types.SimpleNamespace]:
        _ = project_path
        _ = venv_bin
        _ = limits_path
        if self.typings_failure is not None:
            return r[types.SimpleNamespace].fail(self.typings_failure)
        typings = types.SimpleNamespace(to_add=[])

        def _model_dump() -> dict[str, list[str]]:
            return {"to_add": []}

        setattr(typings, "model_dump", _model_dump)
        return r[types.SimpleNamespace].ok(typings)


def _patch_deptry_exists(monkeypatch: pytest.MonkeyPatch, exists: bool) -> None:
    def _exists(_: Path) -> bool:
        return exists

    monkeypatch.setattr(Path, "exists", _exists)


def _setup_detector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    deps: _DepsStub,
    *,
    deptry_exists: bool = True,
) -> detector_module.FlextInfraRuntimeDevDependencyDetector:
    def _workspace_root_from_file(path: str) -> r[Path]:
        _ = path
        return r[Path].ok(tmp_path)

    paths = types.SimpleNamespace(workspace_root_from_file=_workspace_root_from_file)
    monkeypatch.setattr(detector_module, "FlextInfraUtilitiesPaths", lambda: paths)
    monkeypatch.setattr(
        detector_module, "FlextInfraDependencyDetectionService", lambda: deps
    )
    _patch_deptry_exists(monkeypatch, deptry_exists)
    return detector_module.FlextInfraRuntimeDevDependencyDetector()


class TestFlextInfraRuntimeDevDependencyDetectorRunDetect:
    def test_run_with_no_projects(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        result = _setup_detector(monkeypatch, tmp_path, _DepsStub([])).run([
            "--no-pip-check"
        ])
        tm.that(tm.ok(result), eq=2)

    def test_run_with_deptry_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        result = _setup_detector(
            monkeypatch,
            tmp_path,
            _DepsStub([tmp_path / "proj-a"]),
            deptry_exists=False,
        ).run(["--no-pip-check"])
        tm.that(tm.ok(result), eq=3)

    def test_run_with_projects_and_deptry(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        result = _setup_detector(
            monkeypatch, tmp_path, _DepsStub([tmp_path / "proj-a"])
        ).run(
            ["--no-pip-check", "--dry-run"],
        )
        tm.that(tm.ok(result), eq=0)

    def test_run_with_workspace_root_resolution_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _workspace_root_from_file(path: str) -> r[Path]:
            _ = path
            return r[Path].fail("root not found")

        paths = types.SimpleNamespace(
            workspace_root_from_file=_workspace_root_from_file
        )
        monkeypatch.setattr(detector_module, "FlextInfraUtilitiesPaths", lambda: paths)
        error = tm.fail(
            detector_module.FlextInfraRuntimeDevDependencyDetector().run([])
        )
        tm.that(
            "root not found" in error or "workspace root resolution failed" in error,
            eq=True,
        )

    def test_run_with_project_discovery_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        deps = _DepsStub([tmp_path / "proj-a"])
        deps.discovery_failure = "discovery failed"
        error = tm.fail(_setup_detector(monkeypatch, tmp_path, deps).run([]))
        tm.that(
            "discovery failed" in error or "project discovery failed" in error, eq=True
        )

    def test_run_with_deptry_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        deps = _DepsStub([tmp_path / "proj-a"])
        deps.deptry_failure = "deptry failed"
        error = tm.fail(
            _setup_detector(monkeypatch, tmp_path, deps).run(["--no-pip-check"])
        )
        tm.that("deptry failed" in error or "deptry run failed" in error, eq=True)

    def test_run_with_typings_detection_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        (tmp_path / "proj-a" / "src").mkdir(parents=True)
        deps = _DepsStub([tmp_path / "proj-a"])
        deps.typings_failure = "typing detection failed"
        error = tm.fail(
            _setup_detector(monkeypatch, tmp_path, deps).run([
                "--typings",
                "--no-pip-check",
            ])
        )
        tm.that(
            "typing detection failed" in error
            or "typing dependency detection failed" in error,
            eq=True,
        )
