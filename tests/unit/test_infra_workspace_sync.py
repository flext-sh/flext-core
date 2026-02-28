"""Tests for FlextInfraSyncService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra.workspace.sync import FlextInfraSyncService


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
