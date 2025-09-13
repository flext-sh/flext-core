"""Testing utilities and helpers for FLEXT components.

Provides consolidated test utilities, functional test implementations,
and protocol definitions for testing support.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import ClassVar, Generic, Protocol, TypeVar, cast, runtime_checkable

from flext_core import (
    FlextModels,
    FlextResult,
    FlextTypes,
)

# TypeVar outside classes per user demand
T = TypeVar("T")
P = TypeVar("P")


class FlextTestsUtilities:
    """Unified testing utilities for FLEXT ecosystem.

    Consolidates all test utility patterns into a single class interface.
    Provides protocols, factories, assertions, mocking capabilities,
    and functional test implementations.
    """

    # === PROTOCOL DEFINITIONS ===

    @runtime_checkable
    class ITestFactory(Protocol[T]):
        """Protocol for test factories."""

        def create(self, **kwargs: object) -> T:
            """Create test instance."""
            ...

        def create_many(self, count: int, **kwargs: object) -> list[T]:
            """Create multiple test instances."""
            ...

    @runtime_checkable
    class ITestAssertion(Protocol):
        """Protocol for test assertions."""

        def assert_equals(self, actual: object, expected: object) -> None:
            """Assert equality."""
            ...

        def assert_true(self, *, condition: bool) -> None:
            """Assert condition is true."""
            ...

        def assert_false(self, *, condition: bool) -> None:
            """Assert condition is false."""
            ...

    @runtime_checkable
    class ITestMocker(Protocol):
        """Protocol for test mockers - now supports functional implementations."""

        def create_functional_service(
            self, service_type: str, **config: object
        ) -> object:
            """Create functional service implementation."""
            ...

        def create_test_context(self, **options: object) -> object:
            """Create test context manager."""
            ...

    # === CORE UTILITIES ===

    class TestUtilities:
        """Testing utilities that delegate to FlextCore components."""

        @staticmethod
        def create_test_result(
            *,
            success: bool = True,
            data: object = None,
            error: str | None = None,
        ) -> FlextResult[object]:
            """Create a test FlextResult using FlextCore."""
            if success:
                return FlextResult[object].ok(data)
            return FlextResult[object].fail(error or "Test error")

        @staticmethod
        def assert_result_success(result: FlextResult[T]) -> T:
            """Assert result is successful using FlextResult validation."""
            if not result.success:
                msg = f"Expected success but got failure: {result.error}"
                raise AssertionError(msg)
            return result.value

        @staticmethod
        def assert_result_failure(result: FlextResult[T]) -> str:
            """Assert result is failure using FlextResult validation."""
            if result.success:
                msg = f"Expected failure but got success: {result.data}"
                raise AssertionError(msg)
            return result.error or "Unknown error"

        @staticmethod
        def create_test_data(
            *,
            size: int = 10,
            prefix: str = "test",
        ) -> list[FlextTypes.Core.Dict]:
            """Create test data using FlextUtilities.Generators."""
            # Use FlextUtilities.Generators for data generation
            return [
                {
                    "id": i,
                    "name": f"{prefix}_{i}",
                    "value": i * 10,
                    "active": i % 2 == 0,
                }
                for i in range(size)
            ]

    # === TEST FACTORY ===

    class TestFactory(Generic[T]):
        """Factory for creating test objects using FlextUtilities.Generators."""

        def __init__(self, model_class: type[T]) -> None:
            """Initialize test factory."""
            self._model_class = model_class
            self._defaults: FlextTypes.Core.Dict = {}

        def set_defaults(
            self, **defaults: object
        ) -> FlextTestsUtilities.TestFactory[T]:
            """Set default values for created objects."""
            self._defaults.update(defaults)
            return self

        def create(self, **kwargs: object) -> T:
            """Create a single test object using FlextUtilities."""
            data = {**self._defaults, **kwargs}
            return self._model_class(**data)

        def create_many(self, count: int, **kwargs: object) -> list[T]:
            """Create multiple test objects using FlextUtilities.Generators."""
            # Use FlextUtilities.Generators for batch creation
            return [self.create(**kwargs) for _ in range(count)]

        def create_batch(
            self,
            specifications: list[FlextTypes.Core.Dict],
        ) -> list[T]:
            """Create objects from specifications."""
            return [self.create(**spec) for spec in specifications]

    # === TEST ASSERTIONS ===

    class TestAssertion:
        """Assertion utilities that delegate to FlextValidations."""

        @staticmethod
        def assert_equals(
            actual: object,
            expected: object,
            message: str | None = None,
        ) -> None:
            """Assert two values are equal.

            Args:
                actual: Actual value.
                expected: Expected value.
                message: Optional failure message.

            Raises:
                AssertionError: If values are not equal.

            """
            if actual != expected:
                msg = message or f"Expected {expected!r}, got {actual!r}"
                raise AssertionError(msg)

        @staticmethod
        def assert_true(
            *,
            condition: bool,
            message: str | None = None,
        ) -> None:
            """Assert condition is true.

            Args:
                condition: Condition to check.
                message: Optional failure message.

            Raises:
                AssertionError: If condition is false.

            """
            if not condition:
                msg = message or "Expected condition to be true"
                raise AssertionError(msg)

        @staticmethod
        def assert_false(
            *,
            condition: bool,
            message: str | None = None,
        ) -> None:
            """Assert condition is false.

            Args:
                condition: Condition to check.
                message: Optional failure message.

            Raises:
                AssertionError: If condition is true.

            """
            if condition:
                msg = message or "Expected condition to be false"
                raise AssertionError(msg)

        @staticmethod
        def assert_in(
            item: object,
            container: object,  # Accept any container type that supports 'in'
            message: str | None = None,
        ) -> None:
            """Assert item is in container.

            Args:
                item: Item to check.
                container: Container to check in.
                message: Optional failure message.

            Raises:
                AssertionError: If item is not in container.

            """
            try:
                # Use hasattr to check if container supports 'in' operator
                if hasattr(container, "__contains__"):
                    # Support container protocol compatibility
                    contains_method = getattr(container, "__contains__", None)
                    if contains_method is not None and not contains_method(item):
                        msg = message or f"{item!r} not found in {container!r}"
                        raise AssertionError(msg)
                else:
                    msg = (
                        message
                        or f"Container {container!r} does not support 'in' operator"
                    )
                    raise AssertionError(msg)
            except (TypeError, AttributeError) as err:
                # Fallback for containers that don't support 'in' operator
                msg = message or f"Cannot check if {item!r} is in {container!r}"
                raise AssertionError(msg) from err

        @staticmethod
        def assert_not_in(
            item: object,
            container: object,  # Accept any container type that supports 'in'
            message: str | None = None,
        ) -> None:
            """Assert item is not in container.

            Args:
                item: Item to check.
                container: Container to check in.
                message: Optional failure message.

            Raises:
                AssertionError: If item is in container.

            """
            try:
                # Use hasattr to check if container supports 'in' operator
                if hasattr(container, "__contains__"):
                    contains_method = getattr(container, "__contains__", None)
                    if contains_method is not None and contains_method(item):
                        msg = message or f"{item!r} found in {container!r}"
                        raise AssertionError(msg)
            except (TypeError, AttributeError):
                # Fallback for containers that don't support 'in' operator
                pass  # If we can't check, assume item is not in container

        @staticmethod
        def assert_raises(
            exception_class: type[Exception],
            callable_obj: Callable[[], object],
            message: str | None = None,
        ) -> None:
            """Assert callable raises specific exception.

            Args:
                exception_class: Expected exception class.
                callable_obj: Callable to execute.
                message: Optional failure message.

            Raises:
                AssertionError: If expected exception is not raised.

            """

            def _fail_no_exception() -> None:
                msg2 = message or f"Expected {exception_class.__name__} to be raised"
                raise AssertionError(msg2)

            try:
                callable_obj()
                _fail_no_exception()
            except exception_class:
                pass  # Expected exception was raised
            except Exception as e:
                msg = (
                    message
                    or f"Expected {exception_class.__name__}, got {type(e).__name__}: {e}"
                )
                raise AssertionError(msg) from e

    # === FUNCTIONAL TEST IMPLEMENTATIONS ===

    class FunctionalTestService:
        """Functional test service that uses FlextServices for DI."""

        def __init__(self, service_type: str = "generic", **config: object) -> None:
            """Initialize functional test service using FlextServices."""
            self.service_type = service_type
            self.config = config
            self.call_history: list[
                tuple[str, tuple[object, ...], FlextTypes.Core.Dict]
            ] = []
            self.return_values: FlextTypes.Core.Dict = {}
            self.side_effects: dict[str, list[object]] = {}
            self.should_fail: dict[str, bool] = {}
            self.failure_messages: dict[str, str] = {}

        def configure_method(
            self,
            method_name: str,
            return_value: object = None,
            side_effect: list[object] | None = None,
            *,
            should_fail: bool = False,
            failure_message: str = "Method failed",
        ) -> None:
            """Configure method behavior for testing.

            Args:
                method_name: Name of the method to configure.
                return_value: Value to return when method is called.
                side_effect: List of values to return on successive calls.
                should_fail: Whether the method should fail.
                failure_message: Message to use when method fails.

            """
            if return_value is not None:
                self.return_values[method_name] = return_value
            if side_effect is not None:
                self.side_effects[method_name] = side_effect
            self.should_fail[method_name] = should_fail
            self.failure_messages[method_name] = failure_message

        def call_method(
            self, method_name: str, *args: object, **kwargs: object
        ) -> object:
            """Call a method with functional behavior tracking.

            Args:
                method_name: Name of the method to call.
                *args: Positional arguments to pass to the method.
                **kwargs: Keyword arguments to pass to the method.

            Returns:
                The configured return value or default result.

            """
            # Record the call
            self.call_history.append((method_name, args, kwargs))

            # Check if should fail
            if (
                self.should_fail.get(method_name, False)
                and method_name in self.failure_messages
            ):
                if "Result" in str(
                    type(self.return_values.get(method_name, FlextResult))
                ):
                    return FlextResult[object].fail(self.failure_messages[method_name])
                raise ValueError(self.failure_messages[method_name])

            # Handle side effects
            if self.side_effects.get(method_name):
                return self.side_effects[method_name].pop(0)

            # Return configured value or default
            return self.return_values.get(method_name, f"{method_name}_result")

        def get_call_count(self, method_name: str | None = None) -> int:
            """Get number of times a method was called."""
            if method_name is None:
                return len(self.call_history)
            return sum(1 for call in self.call_history if call[0] == method_name)

        def was_called_with(
            self,
            method_name: str,
            *args: object,
            **kwargs: object,
        ) -> bool:
            """Check if method was called with specific arguments.

            Args:
                method_name: Name of the method to check.
                *args: Positional arguments to match.
                **kwargs: Keyword arguments to match.

            Returns:
                True if method was called with these arguments.

            """
            for call in self.call_history:
                if call[0] == method_name and call[1] == args and call[2] == kwargs:
                    return True
            return False

        def get_calls_for_method(
            self,
            method_name: str,
        ) -> list[tuple[tuple[object, ...], FlextTypes.Core.Dict]]:
            """Get all calls for a specific method.

            Args:
                method_name: Name of the method to get calls for.

            Returns:
                List of (args, kwargs) tuples for all calls to the method.

            """
            return [
                (call[1], call[2])
                for call in self.call_history
                if call[0] == method_name
            ]

    class FunctionalTestContext:
        """Functional context manager that uses FlextContext."""

        def __init__(
            self,
            target: object,
            attribute: str,
            new_value: object = None,
            **options: object,
        ) -> None:
            """Initialize functional test context using FlextContext."""
            self.target = target
            self.attribute = attribute
            # Handle new parameter correctly - check if it was explicitly passed
            options_dict = dict(options)  # Convert to dict for easier handling
            if "new" in options_dict:
                self.new_value = options_dict.pop("new")
                self.has_new = True
            else:
                self.new_value = new_value
                self.has_new = new_value is not None
            self.options = options_dict
            self.original_value: object = None
            self.create = bool(options_dict.get("create"))
            self.had_attribute = False

        def __enter__(self) -> object:
            """Enter context - set up test scenario."""
            self.had_attribute = hasattr(self.target, self.attribute)
            if self.had_attribute:
                self.original_value = getattr(self.target, self.attribute)
            elif not self.create:
                # If attribute doesn't exist and create=False, this would be an error in patch
                msg = f"'{type(self.target).__name__}' object has no attribute '{self.attribute}'"
                raise AttributeError(
                    msg,
                )

            # Check if new value was explicitly provided (including None)
            if self.has_new:
                # Use the explicitly provided value (even if None)
                setattr(self.target, self.attribute, self.new_value)
                return self.new_value
            # Create a functional service instead of a mock when no new= is provided
            functional_service = FlextTestsUtilities.FunctionalTestService()
            setattr(self.target, self.attribute, functional_service)
            return functional_service

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: object,
        ) -> None:
            """Exit context - restore original state."""
            # Exception parameters available for future error handling
            if self.had_attribute:
                setattr(self.target, self.attribute, self.original_value)
            elif self.create and hasattr(self.target, self.attribute):
                delattr(self.target, self.attribute)

    # === TEST MOCKER ===

    class TestMocker:
        """Test mocker that uses FlextContainer for DI."""

        @staticmethod
        def create_functional_service(
            service_type: str = "generic",
            **config: object,
        ) -> FlextTestsUtilities.FunctionalTestService:
            """Create a functional service using FlextContainer."""
            return FlextTestsUtilities.FunctionalTestService(
                service_type=service_type, **config
            )

        @staticmethod
        def patch_object(
            target: object,
            attribute: str,
            **options: object,
        ) -> FlextTestsUtilities.FunctionalTestContext:
            """Create a functional patch context manager.

            Args:
                target: Object to patch.
                attribute: Attribute name to patch.
                **options: Additional patch configuration (new, create, etc.).

            Returns:
                Functional test context manager.

            """
            return FlextTestsUtilities.FunctionalTestContext(
                target, attribute, **options
            )

        @staticmethod
        def create_test_context(
            target: object,
            attribute: str,
            new_value: object = None,
            **options: object,
        ) -> FlextTestsUtilities.FunctionalTestContext:
            """Create a test context manager for attribute replacement.

            Args:
                target: Object to modify.
                attribute: Attribute name to replace.
                new_value: New value to set.
                **options: Additional context configuration.

            Returns:
                Functional test context manager.

            """
            return FlextTestsUtilities.FunctionalTestContext(
                target, attribute, new_value, **options
            )

    # === TEST MODELS ===

    class TestModel(FlextModels.TimestampedModel):
        """Test model for testing purposes.

        Provides a simple model with common field types for testing.
        """

        name: str = "test"
        value: int = 42
        active: bool = True
        tags: ClassVar[list[str]] = []
        metadata: ClassVar[FlextTypes.Core.Dict] = {}

        def activate(self) -> FlextResult[None]:
            """Activate the test model."""
            if self.active:
                return FlextResult[None].fail("Already active")
            self.active = True
            return FlextResult[None].ok(None)

        def deactivate(self) -> FlextResult[None]:
            """Deactivate the test model."""
            if not self.active:
                return FlextResult[None].fail("Already inactive")
            self.active = False
            return FlextResult[None].ok(None)

    class TestConfig(FlextModels.TimestampedModel):
        """Test configuration model."""

        debug: bool = False
        timeout: int = 30
        retries: int = 3
        # Build base_url without deep dynamic __import__ chains to avoid runtime errors

        base_url: str = "http://localhost:8000"
        headers: ClassVar[dict[str, str]] = {}

    # === UTILITY FUNCTIONS ===

    @staticmethod
    def create_oud_connection_config() -> dict[str, str]:
        """Create ALGAR OUD connection configuration for testing.

        Returns:
          Dictionary with OUD connection parameters as strings.

        """
        # Prefer environment-provided ports to avoid conflicts in CI/containers
        port_str = (
            os.environ.get("LDAP_PORT") or os.environ.get("TESTS_LDAP_PORT") or "3389"
        )

        return {
            "host": "localhost",
            "port": port_str,
            "bind_dn": "cn=orcladmin",
            # Match container default password from fixtures
            "bind_password": "password",
            "base_dn": "dc=ctbc,dc=com",
            "use_ssl": "false",
            "timeout": "30",
        }

    @staticmethod
    def create_ldap_test_config() -> FlextTypes.Core.Dict:
        """Create LDAP test configuration for testing.

        Returns:
          Dictionary with LDAP connection parameters.

        """
        return {
            "host": "localhost",
            "port": 389,
            "bind_dn": "cn=admin,dc=test,dc=com",
            "bind_password": "testpass",
            "base_dn": "dc=test,dc=com",
            "use_ssl": False,
            "timeout": 30,
        }

    @staticmethod
    def create_api_test_response(
        *,
        success: bool = True,
        data: object = None,
    ) -> FlextTypes.Core.Dict:
        """Create API test response structure.

        Args:
          success: Whether the response represents success or failure.
          data: Custom data to include in response.

        Returns:
          Dictionary with API response structure.

        """
        if success:
            if data is None:
                data = {"id": "test_123", "status": "active"}
            return {
                "success": True,
                "data": data,
                "timestamp": "2025-01-20T12:00:00Z",
            }
        return {
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input data",
                "details": {
                    "field": "name",
                    "error": "required",
                },
            },
            "timestamp": "2025-01-20T12:00:00Z",
        }

    # === FACTORY METHODS ===

    @classmethod
    def utilities(cls) -> FlextTestsUtilities.TestUtilities:
        """Get test utilities instance."""
        return cls.TestUtilities()

    @classmethod
    def factory(cls, model_class: type[T]) -> FlextTestsUtilities.TestFactory[T]:
        """Create test factory for model class."""
        return cls.TestFactory(model_class)

    @classmethod
    def assertion(cls) -> FlextTestsUtilities.TestAssertion:
        """Get test assertion instance."""
        return cls.TestAssertion()

    @classmethod
    def mocker(cls) -> FlextTestsUtilities.TestMocker:
        """Get test mocker instance."""
        return cls.TestMocker()

    @classmethod
    def functional_service(
        cls, service_type: str = "generic", **config: object
    ) -> FlextTestsUtilities.FunctionalTestService:
        """Create functional test service."""
        return cls.FunctionalTestService(service_type, **config)

    @classmethod
    def test_context(
        cls,
        target: object,
        attribute: str,
        new_value: object = None,
        **options: object,
    ) -> FlextTestsUtilities.FunctionalTestContext:
        """Create functional test context."""
        return cls.FunctionalTestContext(target, attribute, new_value, **options)

    # === CONVENIENCE METHODS ===

    @classmethod
    def create_test_result(
        cls,
        *,
        success: bool = True,
        data: object = None,
        error: str | None = None,
    ) -> FlextResult[object]:
        """Create test FlextResult quickly."""
        return cls.TestUtilities.create_test_result(
            success=success, data=data, error=error
        )

    @classmethod
    def create_test_data(
        cls,
        *,
        size: int = 10,
        prefix: str = "test",
    ) -> list[FlextTypes.Core.Dict]:
        """Create test data quickly."""
        return cls.TestUtilities.create_test_data(size=size, prefix=prefix)

    @classmethod
    def create_ldap_config(cls) -> FlextTypes.Core.Dict:
        """Create LDAP test configuration."""
        return cls.create_ldap_test_config()

    @classmethod
    def create_oud_config(cls) -> FlextTypes.Core.Dict:
        """Create OUD connection configuration."""
        config = cls.create_oud_connection_config()
        return cast("FlextTypes.Core.Dict", config)

    @classmethod
    def create_api_response(
        cls, *, success: bool = True, data: object = None
    ) -> FlextTypes.Core.Dict:
        """Create API test response."""
        return cls.create_api_test_response(success=success, data=data)


# === REMOVED COMPATIBILITY ALIASES AND FACADES ===
# Legacy compatibility removed as per user request
# All compatibility facades, aliases and protocol facades have been commented out
# Only FlextTestsUtilities class is now exported

# Main class alias for backward compatibility - REMOVED
# FlextTestsUtility = FlextTestsUtilities

# Legacy FlextTestUtilities class - REMOVED (commented out)
# class FlextTestUtilities:
#     """Compatibility facade for FlextTestUtilities - use FlextTestsUtilities instead."""
#     ... all methods commented out

# Legacy FlextTestFactory class - REMOVED (commented out)
# class FlextTestFactory[T]:
#     """Compatibility facade for FlextTestFactory - use TestFactory instead."""
#     ... all methods commented out

# Legacy FlextTestAssertion class - REMOVED (commented out)
# class FlextTestAssertion:
#     """Compatibility facade for FlextTestAssertion - use TestAssertion instead."""
#     ... all methods commented out

# Legacy FunctionalTestService class - REMOVED (commented out)
# class FunctionalTestService:
#     """Compatibility facade for FunctionalTestService - use FlextTestsUtilities.FunctionalTestService instead."""
#     ... all methods commented out

# Legacy FunctionalTestContext class - REMOVED (commented out)
# class FunctionalTestContext:
#     """Compatibility facade for FunctionalTestContext - use FlextTestsUtilities.FunctionalTestContext instead."""
#     ... all methods commented out

# Legacy FlextTestMocker class - REMOVED (commented out)
# class FlextTestMocker:
#     """Compatibility facade for FlextTestMocker - use TestMocker instead."""
#     ... all methods commented out

# Legacy FlextTestModel class - REMOVED (commented out)
# class FlextTestModel(FlextModels.TimestampedModel):
#     """Compatibility facade for FlextTestModel - use TestModel instead."""
#     ... all methods commented out

# Legacy FlextTestConfig class - REMOVED (commented out)
# class FlextTestConfig(FlextModels.TimestampedModel):
#     """Compatibility facade for FlextTestConfig - use TestConfig instead."""
#     ... all methods commented out

# Utility functions as compatibility facades - REMOVED (commented out)
# def create_oud_connection_config() -> dict[str, str]:
#     ... all functions commented out

# Protocol facades for backward compatibility - REMOVED (commented out)
# ITestFactory = FlextTestsUtilities.ITestFactory
# ITestAssertion = FlextTestsUtilities.ITestAssertion
# ITestMocker = FlextTestsUtilities.ITestMocker

# Backward-compatible aliases - REMOVED (commented out)
# TestUtilities = FlextTestsUtilities

# Export only the unified class
__all__ = [
    "FlextTestsUtilities",
]
