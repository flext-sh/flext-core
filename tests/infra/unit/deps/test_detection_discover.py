from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraDependencyDetectionService
from flext_tests import tm
from tests.infra import m


class _StubSelector:
    def __init__(self, result: r[list[m.Infra.Workspace.ProjectInfo]]) -> None:
        self._result = result

    def resolve_projects(
        self, workspace_root: Path, names: list[str]
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        _ = workspace_root
        _ = names
        return self._result


class TestDiscoverProjects:
    def test_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        service = FlextInfraDependencyDetectionService()
        proj = m.Infra.Workspace.ProjectInfo(
            name="proj", path=tmp_path / "proj", stack="py"
        )
        proj.path.mkdir()
        (proj.path / "pyproject.toml").write_text("")
        monkeypatch.setattr(
            service,
            "selector",
            _StubSelector(r[list[m.Infra.Workspace.ProjectInfo]].ok([proj])),
        )
        result = service.discover_project_paths(tmp_path)
        tm.that(result.is_success, eq=True)
        if result.is_success:
            tm.that(len(result.value), eq=1)

    def test_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service,
            "selector",
            _StubSelector(r[list[m.Infra.Workspace.ProjectInfo]].fail("failed")),
        )
        tm.fail(service.discover_project_paths(tmp_path))

    def test_filters_without_pyproject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        proj = m.Infra.Workspace.ProjectInfo(
            name="no-pyproject", path=tmp_path / "no-pyproject", stack="py"
        )
        proj.path.mkdir()
        monkeypatch.setattr(
            service,
            "selector",
            _StubSelector(r[list[m.Infra.Workspace.ProjectInfo]].ok([proj])),
        )
        result = service.discover_project_paths(tmp_path)
        tm.that(result.is_success, eq=True)
        if result.is_success:
            tm.that(result.value, eq=[])
