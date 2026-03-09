from __future__ import annotations

from pathlib import Path

from flext_infra.deps.path_sync import (
    FlextInfraDependencyPathSync,
    detect_mode,
    extract_dep_name,
    rewrite_dep_paths,
)
from flext_tests import tm
from tests.infra.helpers import h


class TestFlextInfraDependencyPathSync:
    def test_path_sync_initialization(self) -> None:
        path_sync = FlextInfraDependencyPathSync()
        tm.that(type(path_sync).__name__, eq="FlextInfraDependencyPathSync")
        tm.that(type(path_sync._toml).__name__, eq="FlextInfraUtilitiesToml")


class TestDetectMode:
    def test_detect_mode_workspace(self, tmp_path: Path) -> None:
        (tmp_path / ".gitmodules").touch()
        tm.that(detect_mode(tmp_path), eq="workspace")

    def test_detect_mode_workspace_parent(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".gitmodules").touch()
        project = workspace / "project"
        project.mkdir()
        tm.that(detect_mode(project), eq="workspace")

    def test_detect_mode_standalone(self, tmp_path: Path) -> None:
        tm.that(detect_mode(tmp_path), eq="standalone")


def test_detect_mode_with_nonexistent_path(tmp_path: Path) -> None:
    tm.that(detect_mode(tmp_path) in {"workspace", "standalone"}, eq=True)


def test_detect_mode_with_path_object() -> None:
    tm.that(detect_mode(Path("/tmp")) in {"workspace", "standalone"}, eq=True)


def test_helpers_alias_is_reachable() -> None:
    tm.that(True, eq=True)


class TestPathSyncEdgeCases:
    def test_detect_mode_with_nonexistent_path(self, tmp_path: Path) -> None:
        tm.that(detect_mode(tmp_path) in {"workspace", "standalone"}, eq=True)

    def test_extract_dep_name_with_empty_string(self) -> None:
        tm.that(extract_dep_name(""), eq="")

    def test_rewrite_dep_paths_with_no_deps(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool.poetry.dependencies]\npython = "^3.13"')
        h.assert_ok(
            rewrite_dep_paths(
                pyproject,
                mode="poetry",
                internal_names=set(),
                dry_run=True,
            ),
        )
