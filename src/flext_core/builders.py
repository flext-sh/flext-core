"""FlextBuilder - Builder and Factory Patterns for Reduced Boilerplate.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Enterprise builder patterns and factory utilities that dramatically reduce
boilerplate code for common object construction scenarios. These patterns
provide fluent APIs for building complex objects with validation and
type safety.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Self
from typing import TypeVar
from typing import final

from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
U = TypeVar("U")


class FlextBuilder[T](ABC):
    """Abstract base class for fluent builder pattern implementation.

    Provides type-safe fluent interface for building complex objects
    with validation and error handling built-in.
    """

    def __init__(self) -> None:
        """Initialize builder with empty state."""
        self._errors: list[str] = []
        self._built = False

    def _add_error(self, error: str) -> Self:
        """Add validation error to builder state."""
        self._errors.append(error)
        return self

    def _validate_not_built(self) -> None:
        """Ensure builder hasn't been used yet."""
        if self._built:
            msg = "Builder has already been used"
            raise ValueError(msg)

    @abstractmethod
    def build(self) -> FlextResult[T]:
        """Build the final object with validation."""

    def try_build(self) -> T:
        """Build and unwrap result, raising exception on error."""
        return self.build().unwrap()

    def can_build(self) -> bool:
        """Check if builder can successfully build object."""
        return len(self._errors) == 0 and not self._built

    def get_errors(self) -> list[str]:
        """Get accumulated validation errors."""
        return self._errors.copy()


@final
class FlextServiceBuilder(FlextBuilder[dict[str, Any]]):
    """Builder for service configuration with validation."""

    def __init__(self) -> None:
        """Initialize service builder."""
        super().__init__()
        self._services: dict[str, Any] = {}
        self._factories: dict[str, Any] = {}
        self._singletons: list[str] = []

    def add_service(self, name: str, service: object) -> FlextServiceBuilder:
        """Add a service instance."""
        self._validate_not_built()

        if not name or not name.strip():
            return self._add_error("Service name cannot be empty")

        if name in self._services:
            return self._add_error(f"Service '{name}' already registered")

        self._services[name] = service
        return self

    def add_factory(
        self,
        name: str,
        factory: Callable[..., object],
        *,
        singleton: bool = True,
    ) -> FlextServiceBuilder:
        """Add a service factory."""
        self._validate_not_built()

        if not name or not name.strip():
            return self._add_error("Factory name cannot be empty")

        if name in self._factories:
            return self._add_error(f"Factory '{name}' already registered")

        self._factories[name] = factory
        if singleton:
            self._singletons.append(name)

        return self

    def add_services(self, **services: object) -> FlextServiceBuilder:
        """Add multiple services at once."""
        for name, service in services.items():
            self.add_service(name, service)
        return self

    def build(self) -> FlextResult[dict[str, Any]]:
        """Build service configuration dictionary."""
        self._validate_not_built()

        if self._errors:
            errors_str = "; ".join(self._errors)
            return FlextResult.fail(f"Builder errors: {errors_str}")

        self._built = True

        return FlextResult.ok(
            {
                "services": self._services.copy(),
                "factories": self._factories.copy(),
                "singletons": self._singletons.copy(),
            },
        )


@final
class FlextConfigBuilder(FlextBuilder[dict[str, Any]]):
    """Builder for configuration objects with type validation."""

    def __init__(self) -> None:
        """Initialize config builder."""
        super().__init__()
        self._config: dict[str, Any] = {}
        self._required_keys: set[str] = set()

    def set(self, key: str, value: object) -> FlextConfigBuilder:
        """Set a configuration value."""
        self._validate_not_built()

        if not key or not key.strip():
            return self._add_error("Configuration key cannot be empty")

        self._config[key] = value
        return self

    def set_required(self, key: str, value: object) -> FlextConfigBuilder:
        """Set a required configuration value."""
        self._required_keys.add(key)
        return self.set(key, value)

    def set_if_not_none(self, key: str, value: object) -> FlextConfigBuilder:
        """Set configuration value only if not None."""
        if value is not None:
            return self.set(key, value)
        return self

    def set_default(
        self,
        key: str,
        default_value: object,
    ) -> FlextConfigBuilder:
        """Set default value if key not already set."""
        if key not in self._config:
            return self.set(key, default_value)
        return self

    def require(self, *keys: str) -> FlextConfigBuilder:
        """Mark keys as required."""
        self._required_keys.update(keys)
        return self

    def merge(self, other_config: dict[str, Any]) -> FlextConfigBuilder:
        """Merge another configuration dictionary."""
        for key, value in other_config.items():
            self.set(key, value)
        return self

    def build(self) -> FlextResult[dict[str, Any]]:
        """Build configuration with validation."""
        self._validate_not_built()

        # Check for builder errors
        if self._errors:
            errors_str = "; ".join(self._errors)
            return FlextResult.fail(f"Builder errors: {errors_str}")

        # Check required keys
        missing_keys = self._required_keys - set(self._config.keys())
        if missing_keys:
            missing_str = ", ".join(missing_keys)
            return FlextResult.fail(f"Missing required keys: {missing_str}")

        self._built = True
        return FlextResult.ok(self._config.copy())


class FlextFactory[T]:
    """Generic factory class for creating objects with error handling."""

    def __init__(self, factory_func: Callable[..., T]) -> None:
        """Initialize factory with creation function."""
        self._factory_func = factory_func
        self._created_count = 0

    def create(self, *args: object, **kwargs: object) -> FlextResult[T]:
        """Create object using factory function."""
        try:
            result = self._factory_func(*args, **kwargs)
            self._created_count += 1
            return FlextResult.ok(result)
        except Exception as e:  # noqa: BLE001
            return FlextResult.fail(f"Factory creation failed: {e}")

    def try_create(self, *args: object, **kwargs: object) -> T:
        """Create object, raising exception on failure."""
        return self.create(*args, **kwargs).unwrap()

    def create_many(
        self,
        count: int,
        *args: object,
        **kwargs: object,
    ) -> FlextResult[list[T]]:
        """Create multiple objects."""
        if count < 0:
            return FlextResult.fail("Count cannot be negative")

        results: list[T] = []
        for _ in range(count):
            result = self.create(*args, **kwargs)
            if not result.success:
                item_num = len(results) + 1
                return FlextResult.fail(
                    f"Failed to create item {item_num}: {result.error}",
                )
            if result.data is not None:
                results.append(result.data)

        return FlextResult.ok(results)

    @property
    def created_count(self) -> int:
        """Get number of objects created by this factory."""
        return self._created_count


class FlextSingletonFactory[T]:
    """Factory that creates only one instance (singleton pattern)."""

    def __init__(self, factory_func: object) -> None:
        """Initialize singleton factory."""
        if not callable(factory_func):
            msg = "Factory function must be callable"
            raise TypeError(msg)
        self._factory_func: Callable[..., T] = factory_func
        self._instance: T | None = None
        self._created = False

    def get_instance(self, *args: object, **kwargs: object) -> FlextResult[T]:
        """Get singleton instance, creating if necessary."""
        if self._created and self._instance is not None:
            return FlextResult.ok(self._instance)

        try:
            self._instance = self._factory_func(*args, **kwargs)
            self._created = True
            return FlextResult.ok(self._instance)
        except Exception as e:  # noqa: BLE001
            return FlextResult.fail(f"Singleton creation failed: {e}")

    def reset(self) -> None:
        """Reset singleton to allow recreation."""
        self._instance = None
        self._created = False

    @property
    def is_created(self) -> bool:
        """Check if singleton instance has been created."""
        return self._created


# Utility functions for common builder patterns
def build_service_config(**services: object) -> FlextResult[dict[str, Any]]:
    """Quick utility to build service configuration."""
    return FlextServiceBuilder().add_services(**services).build()


def build_config(**config: object) -> FlextResult[dict[str, Any]]:
    """Quick utility to build configuration dictionary."""
    builder = FlextConfigBuilder()
    for key, value in config.items():
        builder.set(key, value)
    return builder.build()


def create_factory[T](factory_func: Callable[..., T]) -> FlextFactory[Any]:
    """Create a factory instance from a function."""
    return FlextFactory(factory_func)


def create_singleton_factory[T](
    factory_func: Callable[..., T],
) -> FlextSingletonFactory[Any]:
    """Create a singleton factory instance from a function."""
    return FlextSingletonFactory(factory_func)


__all__ = [
    "FlextBuilder",
    "FlextConfigBuilder",
    "FlextFactory",
    "FlextServiceBuilder",
    "FlextSingletonFactory",
    "build_config",
    "build_service_config",
    "create_factory",
    "create_singleton_factory",
]
