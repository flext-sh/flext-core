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


def _base_deps(tmp_path: Path, raw_count: int = 0, pip_exit: int = 0):
    return types.SimpleNamespace(
        discover_projects=lambda *_a, **_k: r[list[Path]].ok([tmp_path / "proj-a"]),
        run_deptry=lambda *_a, **_k: r[tuple[list[dict[str, str]], int]].ok(([], 0)),
        build_project_report=lambda *_a, **_k: _ReportStub({
            "deptry": {"raw_count": raw_count}
        }),
        run_pip_check=lambda *_a, **_k: r[tuple[list[str], int]].ok(([], pip_exit)),
    )


def _setup(monkeypatch, tmp_path: Path, deps, json_svc=None, reporting_svc=None):
    paths = types.SimpleNamespace(
        workspace_root_from_file=lambda *_: r[Path].ok(tmp_path)
    )
    monkeypatch.setattr(detector_module, "FlextInfraUtilitiesPaths", lambda: paths)
    monkeypatch.setattr(
        detector_module, "FlextInfraDependencyDetectionService", lambda: deps
    )
    monkeypatch.setattr(Path, "exists", lambda _: True)
    if json_svc is not None:
        monkeypatch.setattr(detector_module, "FlextInfraUtilitiesIo", lambda: json_svc)
    if reporting_svc is not None:
        monkeypatch.setattr(
            detector_module,
            "FlextInfraUtilitiesReporting",
            lambda: reporting_svc,
        )
    return detector_module.FlextInfraRuntimeDevDependencyDetector()


class TestFlextInfraRuntimeDevDependencyDetectorRunReport:
    def test_run_with_output_flag(self, monkeypatch, tmp_path: Path) -> None:
        calls: list[tuple[Path, dict[str, object]]] = []

        def _write_json(path: Path, payload: dict[str, object]):
            calls.append((path, payload))
            return r[bool].ok(True)

        json_svc = types.SimpleNamespace(write_json=_write_json)
        custom_output = tmp_path / "custom_report.json"
        result = _setup(
            monkeypatch, tmp_path, _base_deps(tmp_path), json_svc=json_svc
        ).run([
            "--output",
            str(custom_output),
            "--no-pip-check",
        ])
        tm.ok(result)
        tm.that(len(calls), eq=1)
        tm.that(calls[0][0], eq=custom_output)

    def test_run_with_issues_and_pip_failure(self, monkeypatch, tmp_path: Path) -> None:
        result = _setup(
            monkeypatch,
            tmp_path,
            _base_deps(tmp_path, raw_count=5, pip_exit=1),
        ).run(["--dry-run"])
        tm.that(h.assert_ok(result), eq=1)

    def test_run_with_report_directory_creation_failure(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        reporting = types.SimpleNamespace(
            get_report_dir=lambda *_: tmp_path / "readonly"
        )
        monkeypatch.setattr(
            Path,
            "mkdir",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("Permission denied")),
        )
        result = _setup(
            monkeypatch,
            tmp_path,
            _base_deps(tmp_path),
            reporting_svc=reporting,
        ).run(["--no-pip-check"])
        tm.that("failed to create report directory" in h.assert_fail(result), eq=True)

    def test_run_with_json_write_failure(self, monkeypatch, tmp_path: Path) -> None:
        json_svc = types.SimpleNamespace(
            write_json=lambda *_a, **_k: r[bool].fail("write failed")
        )
        reporting = types.SimpleNamespace(
            get_report_dir=lambda *_: tmp_path / "reports"
        )
        monkeypatch.setattr(Path, "mkdir", lambda *_a, **_k: None)
        result = _setup(
            monkeypatch,
            tmp_path,
            _base_deps(tmp_path),
            json_svc=json_svc,
            reporting_svc=reporting,
        ).run(["--no-pip-check"])
        error = h.assert_fail(result)
        tm.that("write failed" in error or "failed to write report" in error, eq=True)

    def test_run_with_no_fail_flag_with_issues(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        result = _setup(
            monkeypatch,
            tmp_path,
            _base_deps(tmp_path, raw_count=5, pip_exit=1),
        ).run(["--no-fail", "--dry-run"])
        tm.that(h.assert_ok(result), eq=0)

    def test_run_with_json_stdout_flag(self, monkeypatch, tmp_path: Path) -> None:
        result = _setup(monkeypatch, tmp_path, _base_deps(tmp_path)).run([
            "--json",
            "--no-pip-check",
        ])
        tm.that(h.assert_ok(result), eq=0)
