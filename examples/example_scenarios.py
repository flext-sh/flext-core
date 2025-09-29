"""Shared scenario helpers for the FLEXT examples.

This module centralizes the access to the official ``flext_tests`` factories
so that every numbered example can focus on the behaviour it demonstrates
instead of re-creating mock data. All helpers return plain dictionaries or
``FlextResult`` instances so that the example scripts can stay lightweight
and easy to read.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Final

from flext_core import FlextLogger, FlextResult
from flext_tests import (
    FlextTestsFactories,
    FlextTestsFixtures,
)


class ExampleScenarios:
    """Provide canonical scenario data for the tutorial examples."""

    _DEFAULT_USER_COUNT: Final[int] = 5

    @classmethod
    @lru_cache(maxsize=1)
    def dataset(cls) -> dict[str, Any]:
        """Return a reusable dataset with users, configs, and fields."""
        builder = FlextTestsFactories.TestDataBuilder()
        return (
            builder.with_users(count=cls._DEFAULT_USER_COUNT)
            .with_configs(production=False)
            .with_validation_fields(count=cls._DEFAULT_USER_COUNT)
            .build()
        )

    @staticmethod
    def validation_data() -> dict[str, Any]:
        """Return shared validation data used by multiple examples."""
        return FlextTestsFactories.create_validation_test_data()

    @staticmethod
    def realistic_data() -> dict[str, Any]:
        """Return realistic integration-style data for advanced flows."""
        return FlextTestsFactories.create_realistic_test_data()

    @staticmethod
    def user(**overrides: Any) -> dict[str, Any]:
        """Return a single user dictionary."""
        return FlextTestsFactories.UserFactory.create(**overrides)

    @staticmethod
    def users(count: int = _DEFAULT_USER_COUNT) -> list[dict[str, Any]]:
        """Return multiple users."""
        return FlextTestsFactories.UserFactory.create_batch(count)

    @staticmethod
    def config(*, production: bool = False, **overrides: Any) -> dict[str, Any]:
        """Return environment configuration data."""
        if production:
            return FlextTestsFactories.ConfigFactory.production_config(**overrides)
        return FlextTestsFactories.ConfigFactory.create(**overrides)

    @staticmethod
    def service_batch(logger_name: str = "example_batch") -> dict[str, Any]:
        """Return services ready for ``FlextContainer.batch_register``."""
        return {
            "logger": FlextLogger(logger_name),
            "config": ExampleScenarios.config(),
            "metrics": {"requests": 0, "errors": 0},
        }

    @staticmethod
    def payload(**overrides: Any) -> dict[str, Any]:
        """Return a messaging payload."""
        return ExampleScenarios.realistic_data()["api_response"] | overrides

    @staticmethod
    def result_success(data: Any | None = None) -> FlextResult[Any]:
        """Return a successful ``FlextResult`` instance."""
        return FlextTestsFactories.ResultFactory.success_result(data)

    @staticmethod
    def result_failure(message: str = "Scenario error") -> FlextResult[Any]:
        """Return a failed ``FlextResult`` instance."""
        return FlextTestsFactories.ResultFactory.failure_result(message)

    @staticmethod
    def user_result(success: bool = True) -> FlextResult[dict[str, Any]]:
        """Return a user-specific ``FlextResult``."""
        return FlextTestsFactories.ResultFactory.user_result(success=success)

    @staticmethod
    def error_scenario(error_type: str = "ValidationError") -> dict[str, Any]:
        """Return a structured error scenario from fixtures."""
        return FlextTestsFixtures.ErrorSimulationFactory.create_error_scenario(
            error_type
        )

    @staticmethod
    def metadata(source: str = "examples", **extra: Any) -> dict[str, Any]:
        """Return standard metadata for logging and utilities examples."""
        tags = extra.pop("tags", ["scenario", "demo"])
        return {
            "source": source,
            "component": "flext_core",
            "tags": tags,
            **extra,
        }
