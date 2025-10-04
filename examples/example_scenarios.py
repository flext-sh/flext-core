"""Shared scenario helpers for the FLEXT examples.

This module centralizes the access to the official ``flext_tests`` factories
so that every numbered example can focus on the behaviour it demonstrates
instead of re-creating mock data. All helpers return plain dictionaries or
``FlextResult`` instances so that the example scripts can stay lightweight
and easy to read.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Final, TypedDict, cast

from flext_core import FlextLogger, FlextResult, FlextTypes
from flext_tests import FlextTestsFactories


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
    api_response: FlextTypes.Dict
    user_registration: FlextTypes.Dict  # User registration data used by examples


class ExampleScenarios:
    """Provide canonical scenario data for the tutorial examples."""

    _DEFAULT_USER_COUNT: Final[int] = 5

    @classmethod
    @lru_cache(maxsize=1)
    def dataset(cls) -> FlextTypes.Dict:
        """Return a reusable dataset with users, configs, and fields."""
        builder = FlextTestsFactories.TestDataBuilder()
        return (
            builder.with_users(count=cls._DEFAULT_USER_COUNT)
            .with_configs(production=False)
            .with_validation_fields(count=cls._DEFAULT_USER_COUNT)
            .build()
        )

    @staticmethod
    def validation_data() -> FlextTypes.Dict:
        """Return shared validation data used by multiple examples."""
        return FlextTestsFactories.create_validation_test_data()

    @staticmethod
    def realistic_data() -> RealisticDataDict:
        """Return realistic integration-style data for advanced flows."""
        # Type narrowing through casting for type safety
        return cast(
            "RealisticDataDict", FlextTestsFactories.create_realistic_test_data()
        )

    @staticmethod
    def user(**overrides: object) -> FlextTypes.Dict:
        """Return a single user dictionary."""
        return FlextTestsFactories.UserFactory.create(**overrides)

    @staticmethod
    def users(count: int = _DEFAULT_USER_COUNT) -> list[FlextTypes.Dict]:
        """Return multiple users."""
        return FlextTestsFactories.UserFactory.create_batch(count)

    @staticmethod
    def config(*, production: bool = False, **overrides: object) -> FlextTypes.Dict:
        """Return environment configuration data."""
        if production:
            return FlextTestsFactories.ConfigFactory.production_config(**overrides)
        return FlextTestsFactories.ConfigFactory.create(**overrides)

    @staticmethod
    def service_batch(logger_name: str = "example_batch") -> FlextTypes.Dict:
        """Return services ready for ``FlextContainer.batch_register``."""
        return {
            "logger": FlextLogger(logger_name),
            "config": ExampleScenarios.config(),
            "metrics": {"requests": 0, "errors": 0},
        }

    @staticmethod
    def payload(**overrides: object) -> FlextTypes.Dict:
        """Return a messaging payload."""
        return ExampleScenarios.realistic_data()["api_response"] | overrides

    @staticmethod
    def result_success(data: object | None = None) -> FlextResult[object]:
        """Return a successful ``FlextResult`` instance."""
        return FlextTestsFactories.ResultFactory.success_result(data)

    @staticmethod
    def result_failure(message: str = "Scenario error") -> FlextResult[object]:
        """Return a failed ``FlextResult`` instance."""
        return FlextTestsFactories.ResultFactory.failure_result(message)

    @staticmethod
    def user_result(success: bool = True) -> FlextResult[FlextTypes.Dict]:
        """Return a user-specific ``FlextResult``."""
        return FlextTestsFactories.ResultFactory.user_result(success=success)

    @staticmethod
    def error_scenario(error_type: str = "ValidationError") -> FlextTypes.Dict:
        """Return a structured error scenario for examples."""
        return {
            "error_type": error_type,
            "error_code": f"{error_type.upper()}_001",
            "message": f"Example {error_type} scenario",
            "timestamp": "2025-01-01T00:00:00Z",
            "severity": "error",
        }

    @staticmethod
    def metadata(source: str = "examples", **extra: object) -> FlextTypes.Dict:
        """Return standard metadata for logging and utilities examples."""
        tags = extra.pop("tags", ["scenario", "demo"])
        return {
            "source": source,
            "component": "flext_core",
            "tags": tags,
            **extra,
        }
