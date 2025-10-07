"""Shared scenario helpers for the FLEXT examples.

This module centralizes the access to the official ``flext_tests`` factories
so that every numbered example can focus on the behaviour it demonstrates
instead of re-creating mock data. All helpers return plain dictionaries or
``FlextCore.Result`` instances so that the example scripts can stay lightweight
and easy to read.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Final, TypedDict, cast

from flext_core import FlextCore
from flext_tests import FlextTestsMatchers


# TypedDict definitions for type-safe scenario data
class OrderItemDict(TypedDict):
    """Typed dictionary for order items."""

    product_id: str
    name: str
    price: Decimal | float
    quantity: int


class RealisticOrderDict(TypedDict):
    """Typed dictionary for realistic order data."""

    customer_id: str
    items: list[OrderItemDict]
    order_id: str  # Order identifier used by examples
    total: float  # Order total amount used by examples


class RealisticDataDict(TypedDict):
    """Typed dictionary for complete realistic data."""

    order: RealisticOrderDict
    api_response: FlextCore.Types.Dict
    user_registration: FlextCore.Types.Dict  # User registration data used by examples


class ExampleScenarios:
    """Provide canonical scenario data for the tutorial examples."""

    _DEFAULT_USER_COUNT: Final[int] = 5

    @classmethod
    @lru_cache(maxsize=1)
    def dataset(cls) -> FlextCore.Types.Dict:
        """Return a reusable dataset with users, configs, and fields."""
        builder = FlextTestsMatchers.TestDataBuilder()
        return (
            builder.with_users(count=cls._DEFAULT_USER_COUNT)
            .with_configs(production=False)
            .with_validation_fields(count=cls._DEFAULT_USER_COUNT)
            .build()
        )

    @staticmethod
    def validation_data() -> FlextCore.Types.Dict:
        """Return shared validation data used by multiple examples."""
        return FlextTestsMatchers.create_validation_test_data()

    @staticmethod
    def realistic_data() -> RealisticDataDict:
        """Return realistic integration-style data for advanced flows."""
        # Type narrowing through casting for type safety
        return cast(
            "RealisticDataDict", FlextTestsMatchers.create_realistic_test_data()
        )

    @staticmethod
    def user(**overrides: object) -> FlextCore.Types.Dict:
        """Return a single user dictionary."""
        return FlextTestsMatchers.UserFactory.create(**overrides)

    @staticmethod
    def users(count: int = _DEFAULT_USER_COUNT) -> list[FlextCore.Types.Dict]:
        """Return multiple users."""
        return FlextTestsMatchers.UserFactory.create_batch(count)

    @staticmethod
    def config(
        *, production: bool = False, **overrides: object
    ) -> FlextCore.Types.Dict:
        """Return environment configuration data."""
        if production:
            return FlextTestsMatchers.ConfigFactory.production_config(**overrides)
        return FlextTestsMatchers.ConfigFactory.create(**overrides)

    @staticmethod
    def service_batch(logger_name: str = "example_batch") -> FlextCore.Types.Dict:
        """Return services ready for ``FlextCore.Container.batch_register``."""
        return {
            "logger": FlextCore.Logger(logger_name),
            "config": ExampleScenarios.config(),
            "metrics": {"requests": 0, "errors": 0},
        }

    @staticmethod
    def payload(**overrides: object) -> FlextCore.Types.Dict:
        """Return a messaging payload."""
        return ExampleScenarios.realistic_data()["api_response"] | overrides

    @staticmethod
    def result_success(data: object | None = None) -> FlextCore.Result[object]:
        """Return a successful ``FlextCore.Result`` instance."""
        return FlextTestsMatchers.ResultFactory.success_result(data)

    @staticmethod
    def result_failure(message: str = "Scenario error") -> FlextCore.Result[object]:
        """Return a failed ``FlextCore.Result`` instance."""
        return FlextTestsMatchers.ResultFactory.failure_result(message)

    @staticmethod
    def user_result(success: bool = True) -> FlextCore.Result[dict[str, object]]:
        """Return a user-specific ``FlextCore.Result``."""
        return FlextTestsMatchers.ResultFactory.user_result(success=success)

    @staticmethod
    def error_scenario(error_type: str = "ValidationError") -> FlextCore.Types.Dict:
        """Return a structured error scenario for examples."""
        return {
            "error_type": error_type,
            "error_code": f"{error_type.upper()}_001",
            "message": f"Example {error_type} scenario",
            "timestamp": "2025-01-01T00:00:00Z",
            "severity": "error",
        }

    @staticmethod
    def metadata(source: str = "examples", **extra: object) -> FlextCore.Types.Dict:
        """Return standard metadata for logging and utilities examples."""
        tags = extra.pop("tags", ["scenario", "demo"])
        return {
            "source": source,
            "component": "flext_core",
            "tags": tags,
            **extra,
        }
