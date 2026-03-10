from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flext_core import r
from flext_infra.deps.detection import FlextInfraDependencyDetectionService
from flext_tests import tm
from tests.infra.typings import t


class _StubRunner:
    def __init__(self, result: t.ContainerValue) -> None:
        self._result = result

    def run_raw(
        self, *args: t.ContainerValue, **kwargs: t.ContainerValue
    ) -> t.ContainerValue:
        _ = args
        _ = kwargs
        return self._result


class TestDetectionUncoveredLines:
    def test_run_deptry_with_non_dict_issue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        issues, _ = tm.ok(
            service.run_deptry(project, venv_bin, json_output_path=out_file)
        )
        tm.that(len(issues), eq=1)

    def test_run_pip_check_with_empty_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "pip").write_text("")
        out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        monkeypatch.setattr(service, "runner", _StubRunner(r[type(out)].ok(out)))
        lines, exit_code = tm.ok(service.run_pip_check(tmp_path, venv_bin))
        tm.that(lines, eq=[])
        tm.that(exit_code, eq=0)

    def test_get_required_typings_with_limits_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

            def read_plain(self, path: Path) -> t.ContainerValue:
                _ = path
                self._i += 1
                if self._i == 1:
                    return r[dict[str, dict[str, str]]].ok({
                        "python": {"version": "3.13"}
                    })
                return r[dict[str, str]].ok({})

        monkeypatch.setattr(service, "toml", _Toml())
        report = tm.ok(service.get_required_typings(tmp_path, venv_bin))
        tm.that(report.limits_applied, eq=True)
