"""FLEXT Core Helpers - Reduce Boilerplate for Applications.

This module provides helper functions, mixins, and decorators to dramatically
reduce boilerplate code in applications using FLEXT Core.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

from flext_core.container import FlextContainer
from flext_core.container import get_flext_container
from flext_core.domain.entity import FlextEntity
from flext_core.domain.value_object import FlextValueObject
from flext_core.exceptions import FlextError
from flext_core.patterns.logging import get_logger
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable

# Type variables for generic helpers
T = TypeVar("T")
R = TypeVar("R")

# =============================================================================
# QUICK RESULT HELPERS
# =============================================================================


def ok[T](data: T) -> FlextResult[T]:
    """Quick success result creation."""
    return FlextResult.ok(data)


def fail(error: str) -> FlextResult[Any]:
    """Quick failure result creation."""
    return FlextResult.fail(error)


def safe[T](func: Callable[..., T]) -> Callable[..., FlextResult[T]]:
    """Decorator to wrap function calls in FlextResult.

    Automatically catches exceptions and converts to failure results.

    Example:
        @safe
        def divide(a: int, b: int) -> float:
            return a / b

        result = divide(10, 2)  # Returns FlextResult[float]
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> FlextResult[T]:
        try:
            result = func(*args, **kwargs)
            return FlextResult.ok(result)
        except Exception as e:
            return FlextResult.fail(str(e))
    return wrapper


def chain(*results: FlextResult[Any]) -> FlextResult[list[Any]]:
    """Chain multiple results into a single result.

    Returns success only if all results are successful.
    """
    data = []
    for result in results:
        if result.is_failure:
            return FlextResult.fail(result.error or "Chain operation failed")
        data.append(result.data)
    return FlextResult.ok(data)


# =============================================================================
# CONTAINER HELPERS
# =============================================================================


def register(name: str, service: object) -> FlextResult[None]:
    """Quick service registration in global container."""
    container = get_flext_container()
    return container.register(name, service)


def get_service(name: str) -> FlextResult[object]:
    """Quick service retrieval from global container."""
    container = get_flext_container()
    return container.get(name)


def inject(service_name: str) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator for automatic dependency injection.

    Example:
        @inject("database")
        def process_data(data: dict, database) -> dict:
            # database is automatically injected
            return database.save(data)
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            service_result = get_service(service_name)
            if service_result.is_failure:
                msg = f"Service '{service_name}' not available"
                raise FlextError(msg)

            # Add service as last argument
            return func(*args, service_result.data, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# LOGGING HELPERS
# =============================================================================


def log_calls(
    level: str = "INFO",
    include_args: bool = True,
    include_result: bool = True,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorator to automatically log function calls.

    Example:
        @log_calls(level="DEBUG", include_args=False)
        def process_user(user_id: str) -> dict:
            return {"processed": True}
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            func_name = func.__name__

            # Log entry
            if include_args:
                logger.info(f"Calling {func_name}", args=args, kwargs=kwargs)
            else:
                logger.info(f"Calling {func_name}")

            try:
                result = func(*args, **kwargs)

                # Log success
                if include_result:
                    logger.info(f"Completed {func_name}", result=result)
                else:
                    logger.info(f"Completed {func_name}")

                return result
            except Exception as e:
                logger.exception(f"Failed {func_name}", error=str(e))
                raise

        return wrapper
    return decorator


# =============================================================================
# DOMAIN MODEL HELPERS
# =============================================================================


class QuickEntity(FlextEntity):
    """Simplified entity base class with common patterns."""

    def update_timestamp(self) -> None:
        """Update the entity timestamp (requires updated_at field)."""
        if hasattr(self, "updated_at"):
            object.__setattr__(self, "updated_at", datetime.now(UTC))


class QuickValueObject(FlextValueObject):
    """Simplified value object base class."""

    def validate_domain_rules(self) -> None:
        """Default implementation - override if needed."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlextResult[QuickValueObject]:
        """Create value object from dictionary with validation."""
        try:
            return FlextResult.ok(cls(**data))
        except Exception as e:
            return FlextResult.fail(f"Invalid data for {cls.__name__}: {e}")


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_required(*fields: str) -> Callable[[dict[str, Any]], FlextResult[dict[str, Any]]]:
    """Create a validator that checks required fields.

    Example:
        validator = validate_required("name", "email")
        result = validator(user_data)
    """
    def validator(data: dict[str, Any]) -> FlextResult[dict[str, Any]]:
        missing = [field for field in fields if field not in data or not data[field]]
        if missing:
            return FlextResult.fail(f"Missing required fields: {', '.join(missing)}")
        return FlextResult.ok(data)
    return validator


def validate_email(email: str) -> FlextResult[str]:
    """Quick email validation."""
    if "@" not in email or "." not in email.split("@")[1]:
        return FlextResult.fail("Invalid email format")
    return FlextResult.ok(email)


def validate_non_empty(value: str) -> FlextResult[str]:
    """Quick non-empty string validation."""
    if not value or not value.strip():
        return FlextResult.fail("Value cannot be empty")
    return FlextResult.ok(value.strip())


# =============================================================================
# DATA TRANSFORMATION HELPERS
# =============================================================================


def to_dict(obj: Any) -> dict[str, Any]:
    """Convert FlextEntity or FlextValueObject to dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {"value": obj}


def from_dict[T](cls: type[T], data: dict[str, Any]) -> FlextResult[T]:
    """Create object from dictionary with error handling."""
    try:
        if hasattr(cls, "model_validate"):
            return FlextResult.ok(cls.model_validate(data))
        return FlextResult.ok(cls(**data))
    except Exception as e:
        return FlextResult.fail(f"Failed to create {cls.__name__}: {e}")


# =============================================================================
# PIPELINE HELPERS
# =============================================================================


class Pipeline:
    """Fluent pipeline for chaining operations with FlextResult."""

    def __init__(self, initial_value: Any) -> None:
        self._result: FlextResult[Any] = FlextResult.ok(initial_value)

    def then(self, func: Callable[[Any], FlextResult[Any]]) -> Pipeline:
        """Chain another operation."""
        if self._result.is_failure:
            return self

        try:
            self._result = func(self._result.data)
        except Exception as e:
            self._result = FlextResult.fail(str(e))

        return self

    def map(self, func: Callable[[Any], Any]) -> Pipeline:
        """Transform the value."""
        if self._result.is_failure:
            return self

        try:
            new_value = func(self._result.data)
            self._result = FlextResult.ok(new_value)
        except Exception as e:
            self._result = FlextResult.fail(str(e))

        return self

    def validate(self, func: Callable[[Any], FlextResult[Any]]) -> Pipeline:
        """Add validation step."""
        return self.then(func)

    def result(self) -> FlextResult[Any]:
        """Get the final result."""
        return self._result


def pipeline(initial_value: Any) -> Pipeline:
    """Create a new pipeline with initial value."""
    return Pipeline(initial_value)


# =============================================================================
# MIXIN HELPERS
# =============================================================================


class LoggerMixin:
    """Mixin to add logging capabilities to any class."""

    @property
    def logger(self):
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


class ContainerMixin:
    """Mixin to add container access to any class."""

    @property
    def container(self) -> FlextContainer:
        """Get global container."""
        return get_flext_container()

    def get_service(self, name: str) -> FlextResult[object]:
        """Get service from container."""
        return self.container.get(name)


class ValidatorMixin:
    """Mixin to add validation helpers to any class."""

    def validate_data(self, data: dict[str, Any], *required_fields: str) -> FlextResult[dict[str, Any]]:
        """Validate data with required fields."""
        validator = validate_required(*required_fields)
        return validator(data)


# =============================================================================
# UTILITY DECORATORS
# =============================================================================


def retry(max_attempts: int = 3) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Retry decorator for functions that might fail.

    Example:
        @retry(max_attempts=3)
        def unstable_operation() -> str:
            # might fail
            return "success"
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise

            # This shouldn't be reached, but for type safety
            if last_exception:
                raise last_exception
            return func(*args, **kwargs)

        return wrapper
    return decorator


def cache_result[R](func: Callable[..., R]) -> Callable[..., R]:
    """Simple result caching decorator."""
    cache: dict[str, R] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        # Simple cache key (not perfect but good enough for helpers)
        key = str(args) + str(sorted(kwargs.items()))

        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    return wrapper
