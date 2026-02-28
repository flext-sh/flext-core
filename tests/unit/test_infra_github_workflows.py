"""Tests for FlextInfraWorkflowSyncer and SyncOperation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from flext_infra.github.workflows import FlextInfraWorkflowSyncer, SyncOperation


class TestSyncOperation:
    """Test suite for SyncOperation data class."""

    def test_sync_operation_creation(self) -> None:
        """Test creating a SyncOperation instance."""
        op = SyncOperation(
            project="flext-core",
            path=".github/workflows/ci.yml",
            action="create",
            reason="New workflow file",
        )

        assert op.project == "flext-core"
        assert op.path == ".github/workflows/ci.yml"
        assert op.action == "create"
        assert op.reason == "New workflow file"

    def test_sync_operation_frozen(self) -> None:
        """Test that SyncOperation is immutable."""
        assert SyncOperation.model_config.get("frozen") is True


class TestFlextInfraWorkflowSyncer:
    """Test suite for FlextInfraWorkflowSyncer."""

    def test_resolve_source_workflow_absolute_path(self, tmp_path: Path) -> None:
        """Test resolving absolute source workflow path."""
        mock_selector = Mock()
        mock_json = Mock()
        mock_templates = Mock()

        workflow_file = tmp_path / "source.yml"
        workflow_file.write_text("name: CI")

        syncer = FlextInfraWorkflowSyncer(
            selector=mock_selector, json_io=mock_json, templates=mock_templates
        )
        result = syncer.resolve_source_workflow(tmp_path, workflow_file)

        assert result.is_success
        assert result.value == workflow_file

    def test_resolve_source_workflow_relative_path(self, tmp_path: Path) -> None:
        """Test resolving relative source workflow path."""
        mock_selector = Mock()
        mock_json = Mock()
        mock_templates = Mock()

        workflow_file = tmp_path / "source.yml"
        workflow_file.write_text("name: CI")

        syncer = FlextInfraWorkflowSyncer(
            selector=mock_selector, json_io=mock_json, templates=mock_templates
        )
        result = syncer.resolve_source_workflow(tmp_path, Path("source.yml"))

        assert result.is_success
        assert result.value == workflow_file

    def test_resolve_source_workflow_not_found(self, tmp_path: Path) -> None:
        """Test handling of missing source workflow file."""
        mock_selector = Mock()
        mock_json = Mock()
        mock_templates = Mock()

        syncer = FlextInfraWorkflowSyncer(
            selector=mock_selector, json_io=mock_json, templates=mock_templates
        )
        result = syncer.resolve_source_workflow(tmp_path, Path("nonexistent.yml"))

        assert result.is_failure

    def test_default_initialization(self) -> None:
        """Test syncer initializes with default dependencies."""
        syncer = FlextInfraWorkflowSyncer()
        assert syncer._selector is not None
        assert syncer._json is not None
        assert syncer._templates is not None
