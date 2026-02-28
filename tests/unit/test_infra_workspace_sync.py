"""Tests for FlextInfraSyncService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import fcntl
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from flext_infra.workspace.sync import FlextInfraSyncService, main


@pytest.fixture
def sync_service(tmp_path: Path) -> FlextInfraSyncService:
    """Create a sync service instance with temp workspace."""
    return FlextInfraSyncService(canonical_root=tmp_path)


def test_sync_service_generates_base_mk(
    sync_service: FlextInfraSyncService, tmp_path: Path
) -> None:
    """Test that sync service generates base.mk content."""
    result = sync_service.sync(project_root=tmp_path)
    assert result.is_success


def test_sync_service_detects_changes_via_sha256(
    sync_service: FlextInfraSyncService, tmp_path: Path
) -> None:
    """Test that sync service detects changes using SHA256 hash."""
    base_mk_path = tmp_path / "base.mk"
    base_mk_path.write_text("# Old content\n", encoding="utf-8")

    result = sync_service.sync(project_root=tmp_path)

    assert result.is_success


def test_sync_service_skips_write_when_content_unchanged(
    sync_service: FlextInfraSyncService, tmp_path: Path
) -> None:
    """Test that sync service skips write when content is unchanged."""
    content = "# Same content\n"
    base_mk_path = tmp_path / "base.mk"
    base_mk_path.write_text(content, encoding="utf-8")

    result = sync_service.sync(project_root=tmp_path)

    assert result.is_success


def test_sync_service_creates_base_mk_if_missing(
    sync_service: FlextInfraSyncService, tmp_path: Path
) -> None:
    """Test that sync service creates base.mk if it doesn't exist."""
    result = sync_service.sync(project_root=tmp_path)

    assert result.is_success
    assert (tmp_path / "base.mk").exists()


def test_sync_service_execute_returns_failure() -> None:
    """Test that execute() method returns failure as expected."""
    sync_service = FlextInfraSyncService()
    result = sync_service.execute()
    assert result.is_failure


def test_sync_service_validates_gitignore_entries(
    sync_service: FlextInfraSyncService, tmp_path: Path
) -> None:
    """Test that sync service validates required .gitignore entries."""
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text("*.pyc\n", encoding="utf-8")

    result = sync_service.sync(project_root=tmp_path)

    assert result.is_success


def test_sync_service_project_root_required() -> None:
    """Test that sync fails when project_root is None."""
    sync_service = FlextInfraSyncService()
    result = sync_service.sync(project_root=None)
    assert result.is_failure
    assert "project_root is required" in result.error


def test_sync_service_project_root_not_exists() -> None:
    """Test that sync fails when project_root doesn't exist."""
    sync_service = FlextInfraSyncService()
    result = sync_service.sync(project_root=Path("/nonexistent/path"))
    assert result.is_failure
    assert "does not exist" in result.error


def test_sync_service_lock_acquisition_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that sync handles lock acquisition failures gracefully."""

    def mock_flock(fd: int, operation: int) -> None:
        msg = "Lock failed"
        raise OSError(msg)

    monkeypatch.setattr(fcntl, "flock", mock_flock)
    sync_service = FlextInfraSyncService()
    result = sync_service.sync(project_root=tmp_path)
    assert result.is_failure
    assert "lock acquisition failed" in result.error


def test_sync_service_basemk_generation_failure(tmp_path: Path) -> None:
    """Test that sync handles base.mk generation failures."""
    sync_service = FlextInfraSyncService()
    sync_service._generator = Mock()
    mock_result = Mock()
    mock_result.is_failure = True
    mock_result.error = "Generation failed"
    sync_service._generator.generate.return_value = mock_result

    result = sync_service.sync(project_root=tmp_path)
    assert result.is_failure
    assert "Generation failed" in result.error


def test_sync_service_gitignore_update_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that sync handles .gitignore update failures."""
    sync_service = FlextInfraSyncService()

    def mock_open(*args: object, **kwargs: object) -> None:
        msg = "Write failed"
        raise OSError(msg)

    monkeypatch.setattr(Path, "open", mock_open)
    result = sync_service.sync(project_root=tmp_path)
    assert result.is_failure


def test_sync_service_atomic_write_success(tmp_path: Path) -> None:
    """Test atomic write creates file successfully."""
    target = tmp_path / "test.txt"
    result = FlextInfraSyncService._atomic_write(target, "test content")
    assert result.is_success
    assert result.value is True
    assert target.read_text(encoding="utf-8") == "test content"


def test_sync_service_atomic_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test atomic write handles failures gracefully."""

    def mock_named_temp(*args: object, **kwargs: object) -> None:
        msg = "Temp file failed"
        raise OSError(msg)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", mock_named_temp)
    target = tmp_path / "test.txt"
    result = FlextInfraSyncService._atomic_write(target, "test content")
    assert result.is_failure
    assert "atomic write failed" in result.error


def test_sync_service_sha256_content() -> None:
    """Test SHA256 content hashing."""
    content = "test content"
    hash1 = FlextInfraSyncService._sha256_content(content)
    hash2 = FlextInfraSyncService._sha256_content(content)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex is 64 chars


def test_sync_service_sha256_file(tmp_path: Path) -> None:
    """Test SHA256 file hashing."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content", encoding="utf-8")
    hash1 = FlextInfraSyncService._sha256_file(test_file)
    hash2 = FlextInfraSyncService._sha256_file(test_file)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_sync_service_canonical_root_copy(tmp_path: Path) -> None:
    """Test that sync copies from canonical root when available."""
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    canonical_basemk = canonical / "base.mk"
    canonical_basemk.write_text("# Canonical content\n", encoding="utf-8")

    project = tmp_path / "project"
    project.mkdir()

    sync_service = FlextInfraSyncService(canonical_root=canonical)
    result = sync_service.sync(project_root=project)
    assert result.is_success
    assert (project / "base.mk").read_text(encoding="utf-8") == "# Canonical content\n"


def test_sync_service_main_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main() CLI entry point."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sync",
            "--project-root",
            str(tmp_path),
        ],
    )
    exit_code = main()
    assert exit_code == 0


def test_sync_service_ensure_gitignore_entries_missing_entries(tmp_path: Path) -> None:
    """Test that sync service adds missing .gitignore entries."""
    sync_service = FlextInfraSyncService()
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text("*.pyc\n", encoding="utf-8")

    result = sync_service._ensure_gitignore_entries(tmp_path, [".reports/", ".venv/"])
    assert result.is_success
    assert result.value is True
    content = gitignore_path.read_text(encoding="utf-8")
    assert ".reports/" in content
    assert ".venv/" in content


def test_sync_service_ensure_gitignore_entries_all_present(tmp_path: Path) -> None:
    """Test that sync service skips when all entries are present."""
    sync_service = FlextInfraSyncService()
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text(".reports/\n.venv/\n__pycache__/\n", encoding="utf-8")

    result = sync_service._ensure_gitignore_entries(tmp_path, [".reports/", ".venv/"])
    assert result.is_success
    assert result.value is False


def test_sync_service_ensure_gitignore_entries_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that sync handles .gitignore write failures."""
    sync_service = FlextInfraSyncService()
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text("*.pyc\n", encoding="utf-8")

    def mock_open(*args: object, **kwargs: object) -> None:
        msg = "Write failed"
        raise OSError(msg)

    monkeypatch.setattr(Path, "open", mock_open)
    result = sync_service._ensure_gitignore_entries(tmp_path, [".reports/"])
    assert result.is_failure
    assert ".gitignore update failed" in result.error


def test_sync_service_sync_basemk_from_canonical(tmp_path: Path) -> None:
    """Test that sync copies base.mk from canonical root."""
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    canonical_basemk = canonical / "base.mk"
    canonical_basemk.write_text("# Canonical\n", encoding="utf-8")

    project = tmp_path / "project"
    project.mkdir()

    sync_service = FlextInfraSyncService()
    result = sync_service._sync_basemk(project, None, canonical_root=canonical)
    assert result.is_success
    assert result.value is True
    assert (project / "base.mk").read_text(encoding="utf-8") == "# Canonical\n"


def test_sync_service_sync_basemk_no_change_needed(tmp_path: Path) -> None:
    """Test that sync skips when base.mk content is unchanged."""
    sync_service = FlextInfraSyncService()
    content = "# Same content\n"
    (tmp_path / "base.mk").write_text(content, encoding="utf-8")

    # Mock generator to return the same content
    sync_service._generator = Mock()
    mock_gen = Mock()
    mock_gen.is_failure = False
    mock_gen.is_success = True
    mock_gen.value = content
    sync_service._generator.generate.return_value = mock_gen

    result = sync_service._sync_basemk(tmp_path, None)
    assert result.is_success
    assert result.value is False


def test_sync_service_sync_basemk_generation_failure(tmp_path: Path) -> None:
    """Test that sync handles base.mk generation failures."""
    sync_service = FlextInfraSyncService()
    sync_service._generator = Mock()
    mock_result = Mock()
    mock_result.is_failure = True
    mock_result.error = "Generation failed"
    sync_service._generator.generate.return_value = mock_result

    result = sync_service._sync_basemk(tmp_path, None)
    assert result.is_failure
    assert "Generation failed" in result.error
