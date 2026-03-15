from __future__ import annotations

from pathlib import Path

import pytest

import flext_infra.deps.internal_sync as _internal_sync_mod
from flext_core import r
from flext_infra import FlextInfraInternalDependencySyncService
from flext_tests import tm
from tests.infra import h, t


class TestWorkspaceRootFromEnv:
    def test_env_not_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FLEXT_WORKSPACE_ROOT", raising=False)
        tm.that(
            FlextInfraInternalDependencySyncService().workspace_root_from_env(tmp_path),
            eq=None,
        )

    def test_env_set_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", str(tmp_path))
        result = FlextInfraInternalDependencySyncService().workspace_root_from_env(
            project
        )
        tm.that(str(result), eq=str(tmp_path))

    def test_env_set_nonexistent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", "/nonexistent/path")
        tm.that(
            FlextInfraInternalDependencySyncService().workspace_root_from_env(tmp_path),
            eq=None,
        )

    def test_env_set_not_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        project = tmp_path / "other" / "project"
        project.mkdir(parents=True)
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", str(workspace))
        tm.that(
            FlextInfraInternalDependencySyncService().workspace_root_from_env(project),
            eq=None,
        )


class TestWorkspaceRootFromParents:
    def test_found_in_parent(self, tmp_path: Path) -> None:
        (tmp_path / ".gitmodules").touch()
        project = tmp_path / "sub" / "project"
        project.mkdir(parents=True)
        result = FlextInfraInternalDependencySyncService.workspace_root_from_parents(
            project
        )
        tm.that(str(result), eq=str(tmp_path))

    def test_not_found(self, tmp_path: Path) -> None:
        project = tmp_path / "isolated"
        project.mkdir()
        result = FlextInfraInternalDependencySyncService.workspace_root_from_parents(
            project
        )
        tm.that(result, eq=None)

    def test_found_in_self(self, tmp_path: Path) -> None:
        (tmp_path / ".gitmodules").touch()
        result = FlextInfraInternalDependencySyncService.workspace_root_from_parents(
            tmp_path
        )
        tm.that(str(result), eq=str(tmp_path))


class TestIsWorkspaceMode:
    def test_standalone_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLEXT_STANDALONE", "1")
        is_ws, root = FlextInfraInternalDependencySyncService().is_workspace_mode(
            tmp_path
        )
        tm.that(is_ws, eq=False)
        tm.that(root, eq=None)

    def test_env_workspace_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", str(tmp_path))
        monkeypatch.setenv("FLEXT_STANDALONE", "")
        is_ws, root = FlextInfraInternalDependencySyncService().is_workspace_mode(
            project
        )
        tm.that(is_ws, eq=True)
        tm.that(str(root), eq=str(tmp_path))

    def test_git_superproject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FLEXT_STANDALONE", "")
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", "")

        def _git_run(
            *_args: t.Infra.TomlValue,
            **_kwargs: t.Infra.TomlValue,
        ) -> r[str]:
            return r[str].ok(str(tmp_path))

        monkeypatch.setattr(_internal_sync_mod.u.Infra, "git_run", _git_run)
        is_ws, root = FlextInfraInternalDependencySyncService().is_workspace_mode(
            tmp_path / "sub"
        )
        tm.that(is_ws, eq=True)
        tm.that(str(root), eq=str(tmp_path))

    def test_heuristic_gitmodules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".gitmodules").touch()
        project = tmp_path / "sub"
        project.mkdir()
        monkeypatch.setenv("FLEXT_STANDALONE", "")
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", "")

        def _git_run(
            *_args: t.Infra.TomlValue,
            **_kwargs: t.Infra.TomlValue,
        ) -> r[str]:
            return r[str].ok("")

        monkeypatch.setattr(_internal_sync_mod.u.Infra, "git_run", _git_run)
        is_ws, root = FlextInfraInternalDependencySyncService().is_workspace_mode(
            project
        )
        tm.that(is_ws, eq=True)
        tm.that(str(root), eq=str(tmp_path))

    def test_no_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "isolated"
        project.mkdir()
        monkeypatch.setenv("FLEXT_STANDALONE", "")
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", "")

        def _git_run(
            *_args: t.Infra.TomlValue,
            **_kwargs: t.Infra.TomlValue,
        ) -> r[str]:
            return r[str].ok("")

        monkeypatch.setattr(_internal_sync_mod.u.Infra, "git_run", _git_run)
        is_ws, root = FlextInfraInternalDependencySyncService().is_workspace_mode(
            project
        )
        tm.that(is_ws, eq=False)
        tm.that(root, eq=None)
        tm.that(hasattr(h, "assert_ok"), eq=True)
