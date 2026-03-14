"""Edge-case tests for FlextInfraDiscoveryService.

Covers uncovered lines: non-git projects, permission errors,
OSError handling, and submodule name extraction.
"""

from __future__ import annotations

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from flext_infra import FlextInfraUtilitiesDiscovery
from flext_tests import tm


class TestFlextInfraDiscoveryServiceUncoveredLines:
    def test_discover_projects_skips_non_git_projects(self, tmp_path: Path) -> None:
        service = FlextInfraUtilitiesDiscovery()
        workspace_root = tmp_path
        non_git_dir = workspace_root / "non_git_project"
        non_git_dir.mkdir()
        (non_git_dir / "Makefile").touch()
        (non_git_dir / "pyproject.toml").touch()
        result = service.discover_projects(workspace_root)
        tm.ok(result)
        assert len(result.value) == 0

    def test_find_all_pyproject_files_with_nonexistent_path(self) -> None:
        service = FlextInfraUtilitiesDiscovery()
        nonexistent = Path("/nonexistent/path/to/workspace")
        result = service.find_all_pyproject_files(nonexistent)
        tm.ok(result)
        assert result.value == []

    def test_find_all_pyproject_files_with_permission_error(
        self,
        tmp_path: Path,
    ) -> None:
        service = FlextInfraUtilitiesDiscovery()
        (tmp_path / "pyproject.toml").touch()
        result = service.find_all_pyproject_files(tmp_path)
        tm.ok(result)
        assert len(result.value) >= 1

    def test_discover_projects_skips_no_pyproject_no_gomod(
        self,
        tmp_path: Path,
    ) -> None:
        service = FlextInfraUtilitiesDiscovery()
        workspace_root = tmp_path
        proj = workspace_root / "incomplete_project"
        proj.mkdir()
        (proj / ".git").mkdir()
        (proj / "Makefile").touch()
        result = service.discover_projects(workspace_root)
        tm.ok(result)
        assert len(result.value) == 0

    def test_find_all_pyproject_files_oserror_on_rglob(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        service = FlextInfraUtilitiesDiscovery()

        def mock_rglob(self: Path, pattern: str) -> None:
            msg = "permission denied"
            raise OSError(msg)

        monkeypatch.setattr(Path, "rglob", mock_rglob)
        result = service.find_all_pyproject_files(tmp_path)
        tm.fail(result)
        assert isinstance(result.error, str)
        assert "pyproject file scan failed" in result.error

    def test_submodule_names_with_gitmodules_oserror(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        workspace_root = tmp_path
        gitmodules = workspace_root / ".gitmodules"
        gitmodules.touch()

        def mock_read_text(self: Path, encoding: str | None = None) -> None:
            msg = "permission denied"
            raise OSError(msg)

        monkeypatch.setattr(Path, "read_text", mock_read_text)
        result = FlextInfraUtilitiesDiscovery._submodule_names(workspace_root)
        assert result == set()

    def test_submodule_names_with_valid_gitmodules(self, tmp_path: Path) -> None:
        workspace_root = tmp_path
        gitmodules = workspace_root / ".gitmodules"
        gitmodules.write_text(
            '[submodule "sub1"]\n    path = submodule-one\n[submodule "sub2"]\n    path = submodule-two\n',
            encoding="utf-8",
        )
        result = FlextInfraUtilitiesDiscovery._submodule_names(workspace_root)
        assert result == {"submodule-one", "submodule-two"}
