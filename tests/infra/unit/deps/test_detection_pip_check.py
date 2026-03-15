from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraDependencyDetectionService
from flext_tests import tm
from tests.infra import m


class _StubRunner:
    def __init__(self, result: r[m.Infra.Core.CommandOutput]) -> None:
        self._result = result

    def run_raw(
        self, *args: list[str], **kwargs: Path | int | dict[str, str]
    ) -> r[m.Infra.Core.CommandOutput]:
        _ = args
        _ = kwargs
        return self._result


class TestRunPipCheck:
    def test_pip_not_found(self, tmp_path: Path) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        tm.that(tm.ok(service.run_pip_check(tmp_path, venv_bin)), eq=([], 0))

    def test_success_with_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "pip").write_text("")
        out = m.Infra.Core.CommandOutput(
            exit_code=1, stdout="pkg1 has requirement\npkg2 conflict\n", stderr=""
        )
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[m.Infra.Core.CommandOutput].ok(out))
        )
        lines, exit_code = tm.ok(service.run_pip_check(tmp_path, venv_bin))
        tm.that(len(lines), eq=2)
        tm.that(exit_code, eq=1)

    def test_runner_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "pip").write_text("")
        monkeypatch.setattr(
            service,
            "runner",
            _StubRunner(r[m.Infra.Core.CommandOutput].fail("pip failed")),
        )
        tm.fail(service.run_pip_check(tmp_path, venv_bin))

    def test_success_no_issues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "pip").write_text("")
        out = m.Infra.Core.CommandOutput(exit_code=0, stdout="", stderr="")
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[m.Infra.Core.CommandOutput].ok(out))
        )
        lines, exit_code = tm.ok(service.run_pip_check(tmp_path, venv_bin))
        tm.that(lines, eq=[])
        tm.that(exit_code, eq=0)
