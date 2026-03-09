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


class _TypingsStub:
    def __init__(self, to_add: list[object]) -> None:
        self.to_add = to_add

    def model_dump(self) -> dict[str, list[object]]:
        return {"to_add": self.to_add}


def _setup_typings_detector(
    monkeypatch, tmp_path: Path, to_add: list[object], run_raw_result
):
    (tmp_path / "proj-a" / "src").mkdir(parents=True)
    runner_calls: list[tuple[list[object], dict[str, object]]] = []

    def _run_raw(cmd: list[object], **kwargs):
        runner_calls.append((cmd, kwargs))
        return run_raw_result

    deps = types.SimpleNamespace(
        discover_projects=lambda *_a, **_k: r[list[Path]].ok([tmp_path / "proj-a"]),
        run_deptry=lambda *_a, **_k: r[tuple[list[dict[str, str]], int]].ok(([], 0)),
        build_project_report=lambda *_a, **_k: _ReportStub({
            "deptry": {"raw_count": 0}
        }),
        get_required_typings=lambda *_a, **_k: r[_TypingsStub].ok(_TypingsStub(to_add)),
        run_pip_check=lambda *_a, **_k: r[tuple[list[str], int]].ok(([], 0)),
    )
    paths = types.SimpleNamespace(
        workspace_root_from_file=lambda *_: r[Path].ok(tmp_path)
    )
    runner = types.SimpleNamespace(run_raw=_run_raw)
    monkeypatch.setattr(detector_module, "FlextInfraUtilitiesPaths", lambda: paths)
    monkeypatch.setattr(
        detector_module, "FlextInfraDependencyDetectionService", lambda: deps
    )
    monkeypatch.setattr(
        detector_module, "FlextInfraUtilitiesSubprocess", lambda: runner
    )
    monkeypatch.setattr(Path, "exists", lambda _: True)
    return detector_module.FlextInfraRuntimeDevDependencyDetector(), runner_calls


class TestFlextInfraRuntimeDevDependencyDetectorRunTypings:
    def test_run_with_apply_typings_success(self, monkeypatch, tmp_path: Path) -> None:
        run_result = r[types.SimpleNamespace].ok(types.SimpleNamespace(exit_code=0))
        detector, calls = _setup_typings_detector(
            monkeypatch,
            tmp_path,
            ["types-requests"],
            run_result,
        )
        tm.ok(detector.run(["--typings", "--apply-typings", "--no-pip-check"]))
        tm.that(len(calls), eq=1)

    def test_run_with_apply_typings_non_string_package(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        run_result = r[types.SimpleNamespace].ok(types.SimpleNamespace(exit_code=0))
        detector, calls = _setup_typings_detector(
            monkeypatch,
            tmp_path,
            ["types-requests", 123, None],
            run_result,
        )
        tm.ok(detector.run(["--typings", "--apply-typings", "--no-pip-check"]))
        tm.that(len(calls), eq=3)

    def test_run_with_apply_typings_poetry_add_failure(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        run_result = r[types.SimpleNamespace].ok(types.SimpleNamespace(exit_code=1))
        detector, _calls = _setup_typings_detector(
            monkeypatch,
            tmp_path,
            ["types-requests"],
            run_result,
        )
        tm.ok(detector.run(["--typings", "--apply-typings", "--no-pip-check"]))

    def test_run_with_apply_typings_poetry_add_failure_result(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        detector, _calls = _setup_typings_detector(
            monkeypatch,
            tmp_path,
            ["types-requests"],
            r[types.SimpleNamespace].fail("poetry add failed"),
        )
        tm.ok(detector.run(["--typings", "--apply-typings", "--no-pip-check"]))


class TestMainFunction:
    def test_main_returns_failure_code_on_run_failure(self, monkeypatch) -> None:
        class _DetectorStub:
            def run(self, argv: list[str] | None = None):
                _ = argv
                return r[int].fail("boom")

        monkeypatch.setattr(
            detector_module,
            "FlextInfraRuntimeDevDependencyDetector",
            _DetectorStub,
        )
        tm.that(detector_module.main(), eq=1)

    def test_helpers_alias_still_available(self) -> None:
        tm.that(h.__name__, eq="FlextInfraTestHelpers")
