from __future__ import annotations

import types
from pathlib import Path

import flext_infra.deps.detector as detector_module
from flext_core import r
from flext_tests import tm
from tests.infra import h


class _ReportStub:
    def __init__(self, payload: dict[str, dict[str, int]]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, dict[str, int]]:
        return self._payload


def _setup_detector(
    monkeypatch,
    tmp_path: Path,
    deps: types.SimpleNamespace,
    *,
    deptry_exists: bool = True,
):
    paths = types.SimpleNamespace(
        workspace_root_from_file=lambda _: r[Path].ok(tmp_path),
    )
    monkeypatch.setattr(detector_module, "FlextInfraUtilitiesPaths", lambda: paths)
    monkeypatch.setattr(
        detector_module,
        "FlextInfraDependencyDetectionService",
        lambda: deps,
    )
    monkeypatch.setattr(Path, "exists", lambda _: deptry_exists)
    return detector_module.FlextInfraRuntimeDevDependencyDetector()


class TestFlextInfraRuntimeDevDependencyDetectorRunDetect:
    def test_run_with_no_projects(self, monkeypatch, tmp_path: Path) -> None:
        deps = types.SimpleNamespace(
            discover_projects=lambda *_a, **_k: r[list[Path]].ok([])
        )
        result = _setup_detector(monkeypatch, tmp_path, deps).run(["--no-pip-check"])
        tm.that(h.assert_ok(result), eq=2)

    def test_run_with_deptry_missing(self, monkeypatch, tmp_path: Path) -> None:
        deps = types.SimpleNamespace(
            discover_projects=lambda *_a, **_k: r[list[Path]].ok([tmp_path / "proj-a"]),
        )
        result = _setup_detector(
            monkeypatch,
            tmp_path,
            deps,
            deptry_exists=False,
        ).run(["--no-pip-check"])
        tm.that(h.assert_ok(result), eq=3)

    def test_run_with_projects_and_deptry(self, monkeypatch, tmp_path: Path) -> None:
        deps = types.SimpleNamespace(
            discover_projects=lambda *_a, **_k: r[list[Path]].ok([tmp_path / "proj-a"]),
            run_deptry=lambda *_a, **_k: r[tuple[list[dict[str, str]], int]].ok((
                [],
                0,
            )),
            build_project_report=lambda *_a, **_k: _ReportStub({
                "deptry": {"raw_count": 0}
            }),
        )
        result = _setup_detector(monkeypatch, tmp_path, deps).run([
            "--no-pip-check",
            "--dry-run",
        ])
        tm.that(h.assert_ok(result), eq=0)

    def test_run_with_workspace_root_resolution_failure(self, monkeypatch) -> None:
        paths = types.SimpleNamespace(
            workspace_root_from_file=lambda *_: r[Path].fail("root not found"),
        )
        monkeypatch.setattr(detector_module, "FlextInfraUtilitiesPaths", lambda: paths)
        result = detector_module.FlextInfraRuntimeDevDependencyDetector().run([])
        error = h.assert_fail(result)
        tm.that(
            "root not found" in error or "workspace root resolution failed" in error,
            eq=True,
        )

    def test_run_with_project_discovery_failure(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        deps = types.SimpleNamespace(
            discover_projects=lambda *_a, **_k: r[list[Path]].fail("discovery failed"),
        )
        result = _setup_detector(monkeypatch, tmp_path, deps).run([])
        error = h.assert_fail(result)
        tm.that(
            "discovery failed" in error or "project discovery failed" in error, eq=True
        )

    def test_run_with_deptry_failure(self, monkeypatch, tmp_path: Path) -> None:
        deps = types.SimpleNamespace(
            discover_projects=lambda *_a, **_k: r[list[Path]].ok([tmp_path / "proj-a"]),
            run_deptry=lambda *_a, **_k: r[tuple[list[dict[str, str]], int]].fail(
                "deptry failed"
            ),
        )
        result = _setup_detector(monkeypatch, tmp_path, deps).run(["--no-pip-check"])
        error = h.assert_fail(result)
        tm.that("deptry failed" in error or "deptry run failed" in error, eq=True)

    def test_run_with_typings_detection_failure(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        (tmp_path / "proj-a" / "src").mkdir(parents=True)
        deps = types.SimpleNamespace(
            discover_projects=lambda *_a, **_k: r[list[Path]].ok([tmp_path / "proj-a"]),
            run_deptry=lambda *_a, **_k: r[tuple[list[dict[str, str]], int]].ok((
                [],
                0,
            )),
            build_project_report=lambda *_a, **_k: _ReportStub({
                "deptry": {"raw_count": 0}
            }),
            get_required_typings=lambda *_a, **_k: r[types.SimpleNamespace].fail(
                "typing detection failed",
            ),
        )
        result = _setup_detector(monkeypatch, tmp_path, deps).run([
            "--typings",
            "--no-pip-check",
        ])
        error = h.assert_fail(result)
        tm.that(
            "typing detection failed" in error
            or "typing dependency detection failed" in error,
            eq=True,
        )
