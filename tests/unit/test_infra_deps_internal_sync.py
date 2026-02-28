"""Tests for FlextInfraInternalDependencySyncService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from flext_infra.deps.internal_sync import (
    FlextInfraInternalDependencySyncService,
)


class TestFlextInfraInternalDependencySyncService:
    """Test suite for FlextInfraInternalDependencySyncService."""

    def test_service_initialization(self) -> None:
        """Test that service initializes without errors."""
        service = FlextInfraInternalDependencySyncService()
        assert service is not None

    def test_validate_git_ref_valid(self) -> None:
        """Test git ref validation with valid reference."""
        result = FlextInfraInternalDependencySyncService._validate_git_ref("main")
        assert result.is_success
        assert result.value == "main"

    def test_validate_git_ref_invalid(self) -> None:
        """Test git ref validation with invalid reference."""
        result = FlextInfraInternalDependencySyncService._validate_git_ref(
            "invalid@ref!",
        )
        assert result.is_failure

    def test_validate_repo_url_https(self) -> None:
        """Test repository URL validation with HTTPS URL."""
        url = "https://github.com/flext-sh/flext.git"
        result = FlextInfraInternalDependencySyncService._validate_repo_url(url)
        assert result.is_success

    def test_validate_repo_url_ssh(self) -> None:
        """Test repository URL validation with SSH URL."""
        url = "git@github.com:flext-sh/flext.git"
        result = FlextInfraInternalDependencySyncService._validate_repo_url(url)
        assert result.is_success

    def test_validate_repo_url_invalid(self) -> None:
        """Test repository URL validation with invalid URL."""
        result = FlextInfraInternalDependencySyncService._validate_repo_url(
            "not-a-url",
        )
        assert result.is_failure

    def test_ssh_to_https_conversion(self) -> None:
        """Test SSH to HTTPS URL conversion."""
        ssh_url = "git@github.com:flext-sh/flext.git"
        https_url = FlextInfraInternalDependencySyncService._ssh_to_https(ssh_url)
        assert https_url.startswith("https://")
        assert "flext-sh/flext" in https_url

    @patch("flext_infra.deps.internal_sync.CommandRunner")
    def test_service_with_mocked_runner(self, mock_runner: MagicMock) -> None:
        """Test service with mocked command runner."""
        service = FlextInfraInternalDependencySyncService()
        assert service is not None
