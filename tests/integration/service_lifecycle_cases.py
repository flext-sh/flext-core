"""Lifecycle service integration cases kept below module LOC cap.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import tm, u
from tests.integration.service_fixtures import TestsFlextFlextServiceFixtures

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.usefixtures("clean_container")
class TestsFlextFlextServiceLifecycleCases(TestsFlextFlextServiceFixtures):
    """Exercise lifecycle service behavior with an isolated container."""

    @pytest.mark.integration
    def test_lifecycle_service_execution(self) -> None:
        """Test lifecycle service execution."""
        lifecycle_service = self.LifecycleService()
        result = lifecycle_service.execute()
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq="ready")
        tm.fail(lifecycle_service.initialized)

    @pytest.mark.integration
    def test_lifecycle_service_initialization(self, temp_dir: Path) -> None:
        """Test lifecycle service initialization with settings."""
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_dir),
        )
        result = lifecycle_service.initialize(service_config)
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq="initialized")
        tm.ok(lifecycle_service.initialized)
        effective_config = tm.not_none(lifecycle_service.service_config)
        tm.that(effective_config.name, eq="test_service")

    @pytest.mark.integration
    def test_lifecycle_service_health_check(self, temp_dir: Path) -> None:
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
        tm.fail(health_before)
        tm.ok(health_after)

    @pytest.mark.integration
    def test_lifecycle_service_shutdown(self, temp_dir: Path) -> None:
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
        tm.that(result.value, eq="shutdown")
        tm.ok(lifecycle_service.shutdown_called)
        tm.fail(lifecycle_service.health_check())

    @pytest.mark.integration
    def test_lifecycle_service_failure_modes(self, temp_dir: Path) -> None:
        """Test lifecycle service failure modes."""
        lifecycle_service = self.LifecycleService()
        service_config = self._build_service_config(
            name="test_service",
            version="1.0.0",
            temp_dir=str(temp_dir),
        )
        lifecycle_service.configure_failure_mode(fail_init=True)
        init_result = lifecycle_service.initialize(service_config)
        tm.fail(init_result.success)
        tm.that(init_result.error, eq="Initialization failed")
        lifecycle_service.configure_failure_mode(fail_init=False, fail_shutdown=True)
        _ = lifecycle_service.initialize(service_config)
        shutdown_result = lifecycle_service.shutdown()
        tm.fail(shutdown_result.success)
        tm.that(shutdown_result.error, eq="Shutdown failed")
