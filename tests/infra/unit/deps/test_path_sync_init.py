"""Initialization and mode detection tests for path sync."""

from __future__ import annotations

from pathlib import Path

from flext_infra.deps.path_sync import FlextInfraDependencyPathSync, detect_mode
from flext_tests import tm
from tests.infra import h


class TestFlextInfraDependencyPathSync:
    """Test FlextInfraDependencyPathSync."""

    def test_path_sync_initialization(self) -> None:
        """Test path sync initializes without errors."""
        path_sync = FlextInfraDependencyPathSync()
        tm.that(path_sync is not None, eq=True)
        tm.that(path_sync._toml is not None, eq=True)


class TestDetectMode:
    """Test detect_mode function."""

    def test_detect_mode_workspace(self, tmp_path: Path) -> None:
        """Test detect_mode with workspace structure."""
        (tmp_path / ".gitmodules").touch()
        tm.that(detect_mode(tmp_path), eq="workspace")

    def test_detect_mode_workspace_parent(self, tmp_path: Path) -> None:
        """Test detect_mode finds .gitmodules in parent."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".gitmodules").touch()
        project = workspace / "project"
        project.mkdir()
        tm.that(detect_mode(project), eq="workspace")

    def test_detect_mode_standalone(self, tmp_path: Path) -> None:
        """Test detect_mode with standalone structure."""
        tm.that(detect_mode(tmp_path), eq="standalone")


def test_detect_mode_with_nonexistent_path(tmp_path: Path) -> None:
    """Test detect_mode with a path with no .gitmodules."""
    tm.that(detect_mode(tmp_path) is not None, eq=True)


def test_detect_mode_with_path_object() -> None:
    """Test detect_mode accepts Path object."""
    tm.that(detect_mode(Path("/tmp")) in {"workspace", "standalone"}, eq=True)


def test_helpers_alias_is_reachable() -> None:
    """Keep `h` import intentional and verified."""
    tm.that(hasattr(h, "assert_ok"), eq=True)
