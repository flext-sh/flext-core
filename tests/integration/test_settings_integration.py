"""Behavioral integration tests for FlextSettings public contract.

Asserts observable public behavior only — singleton identity, container
wiring, environment/constructor precedence, computed fields, cloning, and
thread-safe global access. No private attributes, no internal-collaborator
spying, no patching of the unit under test.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from flext_core import FlextContainer, FlextSettings
from tests.constants import c
from tests.models import m
from tests.utilities import u

from .settings_integration_precedence import TestsFlextFlextSettingsPrecedenceCase

if TYPE_CHECKING:
    from collections.abc import MutableSequence


class TestsFlextSettingsIntegration(TestsFlextFlextSettingsPrecedenceCase):
    """Behavioral tests for the FlextSettings public configuration contract."""

    def setup_method(self) -> None:
        """Reset the global singleton and container before each test."""
        FlextSettings.reset_for_testing()
        FlextContainer().clear()

    def teardown_method(self) -> None:
        """Reset the global singleton and container after each test."""
        FlextSettings.reset_for_testing()
        FlextContainer().clear()

    def test_fetch_global_returns_same_singleton_instance(self) -> None:
        """fetch_global() always yields the identical global instance."""
        # Arrange / Act

        # Assert — identity is the observable singleton contract
        assert first is second is third
        assert first.model_dump()["app_name"] == first.app_name

    def test_reset_for_testing_replaces_the_global_instance(self) -> None:
        """reset_for_testing() causes the next fetch_global() to be a new object."""
        # Arrange

        # Act
        FlextSettings.reset_for_testing()

        # Assert — a fresh instance is served after reset
        assert after is not before
        assert after.model_dump()["app_name"] == after.app_name

    def test_container_resolves_settings_to_the_global_singleton(self) -> None:
        """The container's "settings" binding is the same object as fetch_global()."""
        # Arrange

        # Act
        resolved = FlextContainer().resolve("settings")

        # Assert — public r[T] success and identity with the global
        assert resolved.success
        assert resolved.unwrap() is global_settings

    def test_default_settings_expose_documented_public_defaults(self) -> None:
        """A defaults-only instance exposes the documented public field values."""
        # Arrange / Act
        with FlextSettings.singleton_disabled():
            settings = FlextSettings()

        # Assert — public field contract
        assert settings.app_name == "flext"
        assert settings.debug is False
        assert settings.trace is False
        assert settings.timeout_seconds == 30
        assert settings.max_retry_attempts == 3
        assert settings.cache_ttl == 300

    def test_environment_variables_override_settings(self) -> None:
        """Environment variables are reflected in the resolved global settings."""
        # Arrange
        with u.Tests.env_vars_context({
            "FLEXT_APP_NAME": "test-app-from-env",
            "FLEXT_LOG_LEVEL": "DEBUG",
            "FLEXT_MAX_WORKERS": "8",
            "FLEXT_TIMEOUT_SECONDS": "90",
            "FLEXT_DEBUG": "true",
        }):
            FlextSettings.reset_for_testing()
            # Act

            # Assert — public fields carry the environment values
            assert settings.app_name == "test-app-from-env"
            assert settings.log_level == "DEBUG"
            assert settings.max_workers == 8
            assert settings.timeout_seconds == 90
            assert settings.debug is True

    def test_explicit_constructor_arguments_win_over_defaults(self) -> None:
        """Constructor keyword arguments set the corresponding public fields."""
        # Arrange / Act
        with FlextSettings.singleton_disabled():
            settings = FlextSettings(
                app_name="from-init",
                log_level=c.LogLevel.ERROR,
                debug=True,
                timeout_seconds=90,
            )

        # Assert — fields and their public serialization agree
        assert settings.app_name == "from-init"
        assert settings.log_level == "ERROR"
        assert settings.debug is True
        assert settings.timeout_seconds == 90

        dumped = settings.model_dump()
        assert dumped["app_name"] == "from-init"
        assert dumped["debug"] is True
        assert dumped["timeout_seconds"] == 90

    def test_model_validate_builds_settings_from_a_mapping(self) -> None:
        """model_validate() maps a public payload onto typed public fields."""
        # Arrange
        payload = {
            "app_name": "from-mapping",
            "log_level": "WARNING",
            "timeout_seconds": 60,
            "debug": True,
        }

        # Act
        with FlextSettings.singleton_disabled():
            settings = FlextSettings.model_validate(payload)

        # Assert — validated values are observable on the public surface
        assert settings.app_name == "from-mapping"
        assert settings.log_level == "WARNING"
        assert settings.timeout_seconds == 60
        assert settings.debug is True

    @pytest.mark.parametrize(
        ("trace", "debug", "log_level", "expected"),
        [
            (True, True, c.LogLevel.ERROR, c.LogLevel.DEBUG),
            (True, True, c.LogLevel.WARNING, c.LogLevel.DEBUG),
            (False, True, c.LogLevel.WARNING, c.LogLevel.INFO),
            (False, False, c.LogLevel.WARNING, c.LogLevel.WARNING),
            (False, False, c.LogLevel.ERROR, c.LogLevel.ERROR),
        ],
    )
    def test_effective_log_level_resolves_from_trace_and_debug(
        self,
        *,
        trace: bool,
        debug: bool,
        log_level: c.LogLevel,
        expected: c.LogLevel,
    ) -> None:
        """effective_log_level: DEBUG if trace, else INFO if debug, else log_level."""
        # Arrange / Act
        with FlextSettings.singleton_disabled():
            settings = FlextSettings(trace=trace, debug=debug, log_level=log_level)

        # Assert — computed_field contract
        assert settings.effective_log_level == expected

    def test_trace_without_debug_is_rejected(self) -> None:
        """Enabling trace without debug violates a public model invariant."""
        # Act / Assert — the invariant surfaces as a validation error
        with (
            FlextSettings.singleton_disabled(),
            pytest.raises(m.ValidationError, match="Trace mode requires debug mode"),
        ):
            FlextSettings(trace=True, debug=False)

    def test_clone_produces_independent_copy_with_overrides(self) -> None:
        """clone() returns a new object; overrides apply without mutating the source."""
        # Arrange
        with FlextSettings.singleton_disabled():
            original = FlextSettings(app_name="base", timeout_seconds=30)

        # Act
        clone = original.clone(app_name="cloned")

        # Assert — independence + override are the observable behavior
        assert clone is not original
        assert clone.app_name == "cloned"
        assert clone.timeout_seconds == 30
        assert original.app_name == "base"

    def test_fetch_global_is_thread_safe(self) -> None:
        """Concurrent fetch_global() calls all observe the same singleton."""
        # Arrange
        collected: MutableSequence[FlextSettings] = []

        def collect() -> None:
            collected.append(FlextSettings.fetch_global())

        threads = [threading.Thread(target=collect) for _ in range(10)]

        # Act
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Assert — every thread saw the identical instance
        assert len(collected) == 10
        first = collected[0]
        assert all(settings is first for settings in collected[1:])
