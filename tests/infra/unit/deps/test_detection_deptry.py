from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from flext_core import r
from flext_infra.deps.detection import FlextInfraDependencyDetectionService
from flext_tests import tm
from tests.infra import h


class _StubRunner:
    def __init__(self, result) -> None:
        self._result = result
        self.last_kwargs: dict[str, str | int | Path | dict[str, str]] = {}

    def run_raw(self, *args, **kwargs):
        _ = args
        self.last_kwargs = kwargs
        return self._result


class _StubSelector:
    def __init__(self, result) -> None:
        self._result = result

    def resolve_projects(self, workspace_root: Path, names: list[str]):
        _ = workspace_root
        _ = names
        return self._result


class TestDiscoverProjects:
    def test_success(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        proj = SimpleNamespace(path=tmp_path / "proj")
        proj.path.mkdir()
        (proj.path / "pyproject.toml").write_text("")
        monkeypatch.setattr(
            service,
            "selector",
            _StubSelector(r[list[SimpleNamespace]].ok([proj])),
        )
        result = service.discover_projects(tmp_path)
        paths = h.assert_ok(result)
        tm.that(len(paths), eq=1)

    def test_failure(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service,
            "selector",
            _StubSelector(r[list[Path]].fail("failed")),
        )
        h.assert_fail(service.discover_projects(tmp_path))

    def test_filters_without_pyproject(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        proj = SimpleNamespace(path=tmp_path / "no-pyproject")
        proj.path.mkdir()
        monkeypatch.setattr(
            service,
            "selector",
            _StubSelector(r[list[SimpleNamespace]].ok([proj])),
        )
        result = service.discover_projects(tmp_path)
        tm.that(h.assert_ok(result), eq=[])


class TestRunDeptry:
    def test_success_with_issues(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        out_file = project / ".deptry-report.json"
        out_file.write_text(
            json.dumps([{"error": {"code": "DEP001"}, "module": "foo"}])
        )
        cmd_out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[type(cmd_out)].ok(cmd_out))
        )
        issues, exit_code = h.assert_ok(
            service.run_deptry(project, venv_bin, json_output_path=out_file)
        )
        tm.that(exit_code, eq=0)
        tm.that(len(issues), eq=1)

    def test_no_config_file(self, tmp_path: Path) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        tm.that(h.assert_ok(service.run_deptry(project, venv_bin)), eq=([], 0))

    def test_runner_failure(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[SimpleNamespace].fail("deptry crash"))
        )
        h.assert_fail(service.run_deptry(project, venv_bin))

    def test_invalid_json_output(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        out_file = project / ".deptry-report.json"
        out_file.write_text("not valid json")
        cmd_out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[type(cmd_out)].ok(cmd_out))
        )
        tm.that(
            h.assert_ok(
                service.run_deptry(project, venv_bin, json_output_path=out_file)
            ),
            eq=([], 0),
        )

    def test_empty_json_output(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        out_file = project / ".deptry-report.json"
        out_file.write_text("")
        cmd_out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[type(cmd_out)].ok(cmd_out))
        )
        tm.that(
            h.assert_ok(
                service.run_deptry(project, venv_bin, json_output_path=out_file)
            ),
            eq=([], 0),
        )

    def test_with_extend_exclude(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        cmd_out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[type(cmd_out)].ok(cmd_out))
        )
        h.assert_ok(
            service.run_deptry(project, venv_bin, extend_exclude=["tests", "docs"])
        )

    def test_cleanup_temp_file(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        default_out = project / ".deptry-report.json"
        default_out.write_text("[]")
        cmd_out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[type(cmd_out)].ok(cmd_out))
        )
        h.assert_ok(service.run_deptry(project, venv_bin))
        tm.that(default_out.exists(), eq=False)
