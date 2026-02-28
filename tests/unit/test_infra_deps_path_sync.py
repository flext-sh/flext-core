"""Tests for FlextInfraDependencyPathSync."""

from __future__ import annotations

from pathlib import Path

from flext_infra.deps.path_sync import (
    FlextInfraDependencyPathSync,
    detect_mode,
    extract_dep_name,
)


class TestFlextInfraDependencyPathSync:
    """Test FlextInfraDependencyPathSync."""

    def test_path_sync_initialization(self) -> None:
        """Test path sync initializes without errors."""
        path_sync = FlextInfraDependencyPathSync()
        assert path_sync is not None


class TestDetectMode:
    """Test detect_mode function."""

    def test_detect_mode_workspace(self, tmp_path: Path) -> None:
        """Test detect_mode with workspace structure."""
        gitmodules = tmp_path / ".gitmodules"
        gitmodules.touch()
        mode = detect_mode(tmp_path)
        assert mode == "workspace"

    def test_detect_mode_standalone(self, tmp_path: Path) -> None:
        """Test detect_mode with standalone structure."""
        mode = detect_mode(tmp_path)
        assert mode == "standalone"


class TestExtractDepName:
    """Test extract_dep_name function."""

    def test_extract_dep_name_simple(self) -> None:
        """Test extract_dep_name with simple path."""
        name = extract_dep_name("flext-core")
        assert name == "flext-core"

    def test_extract_dep_name_with_prefix(self) -> None:
        """Test extract_dep_name with .flext-deps prefix."""
        name = extract_dep_name(".flext-deps/flext-core")
        assert name == "flext-core"

    def test_extract_dep_name_with_parent_ref(self) -> None:
        """Test extract_dep_name with parent directory reference."""
        name = extract_dep_name("../flext-core")
        assert name == "flext-core"

    def test_extract_dep_name_with_slash(self) -> None:
        """Test extract_dep_name with leading slash."""
        name = extract_dep_name("/flext-core")
        assert name == "flext-core"
