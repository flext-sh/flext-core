"""Lifecycle service integration cases kept below module LOC cap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.utilities import u

from .service_fixtures import TestsFlextFlextServiceFixtures

if TYPE_CHECKING:
    from pathlib import Path

    from tests.protocols import p


class TestsFlextFlextServiceLifecycleCases(TestsFlextFlextServiceFixtures):
    @pytest.mark.integration
    def test_lifecycle_service_execution(self, clean_container: p.Container) -> None:
        """Test lifecycle service execution."""
        lifecycle_service = self.LifecycleService()
        result = lifecycle_service.execute()
        _ = u.Tests.assert_success(result)
        assert result.value == "ready"
        assert lifecycle_service.initialized is False

    @pytest.mark.integration
    def test_lifecycle_service_initialization(
        self,
        clean_container: p.Container,
        temp_dir: Path,
    ) -> None:
        """Test lifecycle service initialization with settings."""
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_dir),
        )
        result = lifecycle_service.initialize(service_config)
        _ = u.Tests.assert_success(result)
        assert result.value == "initialized"
        assert lifecycle_service.initialized is True
        assert lifecycle_service.service_config is not None
        assert lifecycle_service.service_config.name == "test_service"

    @pytest.mark.integration
    def test_lifecycle_service_health_check(
        self,
        clean_container: p.Container,
        temp_dir: Path,
    ) -> None:
        """Test lifecycle service health check."""
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_dir),
        )
        health_before = lifecycle_service.health_check()
        _ = lifecycle_service.initialize(service_config)
        health_after = lifecycle_service.health_check()
        assert health_before is False
        assert health_after is True

    @pytest.mark.integration
    def test_lifecycle_service_shutdown(
        self,
        clean_container: p.Container,
        temp_dir: Path,
    ) -> None:
        """Test lifecycle service shutdown."""
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_dir),
        )
        _ = lifecycle_service.initialize(service_config)
        result = lifecycle_service.shutdown()
        _ = u.Tests.assert_success(result)
        assert result.value == "shutdown"
        assert lifecycle_service.shutdown_called is True
        assert lifecycle_service.health_check() is False

    @pytest.mark.integration
    def test_lifecycle_service_failure_modes(
        self,
        clean_container: p.Container,
        temp_dir: Path,
    ) -> None:
        """Test lifecycle service failure modes."""
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_dir),
        )
        lifecycle_service.configure_failure_mode(fail_init=True)
        init_result = lifecycle_service.initialize(service_config)
        assert init_result.success is False
        assert init_result.error == "Initialization failed"
        lifecycle_service.configure_failure_mode(fail_init=False, fail_shutdown=True)
        _ = lifecycle_service.initialize(service_config)
        shutdown_result = lifecycle_service.shutdown()
        assert shutdown_result.success is False
        assert shutdown_result.error == "Shutdown failed"
