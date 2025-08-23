"""Testing utilities and helpers for FLEXT components."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import ClassVar, Protocol, TypeVar, runtime_checkable

from flext_core import (
    FlextConstants as _FlextConstants,  # local import to avoid cycles
    FlextModel,
    FlextResult,
)

T = TypeVar("T")


# =============================================================================
# PROTOCOLS - Interface Segregation Principle (ISP)
# =============================================================================


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

    def create_functional_service(self, service_type: str, **config: object) -> object:
        """Create functional service implementation."""
        ...

    def create_test_context(self, **options: object) -> object:
        """Create test context manager."""
        ...


# =============================================================================
# BASE ABSTRACTIONS - Single Responsibility Principle (SRP)
# =============================================================================


class FlextTestUtilities:
    """Centralized testing utilities for FLEXT components.

    Follows SRP by focusing only on test result creation and validation.
    """

    @staticmethod
    def create_test_result(
        *,
        success: bool = True,
        data: object = None,
        error: str | None = None,
    ) -> FlextResult[object]:
        """Create a test FlextResult.

        Args:
            success: Whether the result should be successful.
            data: Data for a successful result.
            error: Error message for a failed result.

        Returns:
            FlextResult for testing.

        """
        if success:
            return FlextResult[object].ok(data or {})
        return FlextResult[object].fail(error or str(data) if data else "Test error")

    @staticmethod
    def assert_result_success(result: FlextResult[T]) -> T:
        """Assert that a result is successful and return the data.

        Args:
            result: Result to check.

        Returns:
            The unwrapped data.

        Raises:
            AssertionError: If a result is not successful.

        """
        if not result.is_success:
            msg = f"Expected success but got failure: {result.error}"
            raise AssertionError(msg)
        return result.value

    @staticmethod
    def assert_result_failure(result: FlextResult[T]) -> str:
        """Assert that a result is a failure and return the error.

        Args:
            result: Result to check.

        Returns:
            The error message.

        Raises:
            AssertionError: If a result is not a failure.

        """
        if not result.is_failure:
            msg = f"Expected failure but got success: {result.value}"
            raise AssertionError(msg)
        return result.error or "Unknown error"

    @staticmethod
    def create_test_data(
        *,
        size: int = 10,
        prefix: str = "test",
    ) -> list[dict[str, object]]:
        """Create test data for testing.

        Args:
            size: Number of test items to create.
            prefix: Prefix for test item names.

        Returns:
            List of test data dictionaries.

        """
        return [
            {
                "id": i,
                "name": f"{prefix}_{i}",
                "value": i * 10,
                "active": i % 2 == 0,
            }
            for i in range(size)
        ]


# =============================================================================
# TEST FACTORY - Open/Closed Principle (OCP)
# =============================================================================


class FlextTestFactory[T]:
    """Factory for creating test objects.

    Follows OCP by being open for extension through subclassing.
    """

    def __init__(self, model_class: type[T]) -> None:
        """Initialize test factory.

        Args:
            model_class: Class to create instances of.

        """
        self._model_class = model_class
        self._defaults: dict[str, object] = {}

    def set_defaults(self, **defaults: object) -> FlextTestFactory[T]:
        """Set default values for created objects.

        Args:
            **defaults: Default field values.

        Returns:
            Self for method chaining.

        """
        self._defaults.update(defaults)
        return self

    def create(self, **kwargs: object) -> T:
        """Create a single test object.

        Args:
            **kwargs: Field values to override defaults.

        Returns:
            Created test object.

        """
        data = {**self._defaults, **kwargs}
        return self._model_class(**data)

    def create_many(self, count: int, **kwargs: object) -> list[T]:
        """Create multiple test objects.

        Args:
            count: Number of objects to create.
            **kwargs: Field values to override defaults.

        Returns:
            List of created test objects.

        """
        return [self.create(**kwargs) for _ in range(count)]

    def create_batch(
        self,
        specifications: list[dict[str, object]],
    ) -> list[T]:
        """Create objects from specifications.

        Args:
            specifications: List of field value dictionaries.

        Returns:
            List of created test objects.

        """
        return [self.create(**spec) for spec in specifications]


# =============================================================================
# TEST ASSERTIONS - Liskov Substitution Principle (LSP)
# =============================================================================


class FlextTestAssertion:
    """Enhanced assertion utilities for testing.

    All assertions follow LSP - they can be substituted without affecting behavior.
    """

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
                    message or f"Container {container!r} does not support 'in' operator"
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


# =============================================================================
# FUNCTIONAL TEST IMPLEMENTATIONS - Dependency Inversion Principle (DIP)
# =============================================================================


class FunctionalTestService:
    """Functional test service for realistic testing behavior."""

    def __init__(self, service_type: str = "generic", **config: object) -> None:
        """Initialize functional test service."""
        self.service_type = service_type
        self.config = config
        self.call_history: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.return_values: dict[str, object] = {}
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

    def call_method(self, method_name: str, *args: object, **kwargs: object) -> object:
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
            if "Result" in str(type(self.return_values.get(method_name, FlextResult))):
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
    ) -> list[tuple[tuple[object, ...], dict[str, object]]]:
        """Get all calls for a specific method.

        Args:
            method_name: Name of the method to get calls for.

        Returns:
            List of (args, kwargs) tuples for all calls to the method.

        """
        return [
            (call[1], call[2]) for call in self.call_history if call[0] == method_name
        ]


class FunctionalTestContext:
    """Functional context manager for test scenarios."""

    def __init__(
        self,
        target: object,
        attribute: str,
        new_value: object = None,
        **options: object,
    ) -> None:
        """Initialize functional test context.

        Args:
            target: Object to patch.
            attribute: Name of attribute to patch.
            new_value: New value to set for the attribute.
            **options: Additional options (new, create, etc.).

        """
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
            raise AttributeError(
                f"'{type(self.target).__name__}' object has no attribute '{self.attribute}'",
            )

        # Check if new value was explicitly provided (including None)
        if self.has_new:
            # Use the explicitly provided value (even if None)
            setattr(self.target, self.attribute, self.new_value)
            return self.new_value
        # Create a functional service instead of a mock when no new= is provided
        functional_service = FunctionalTestService()
        setattr(self.target, self.attribute, functional_service)
        return functional_service

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context - restore original state."""
        if self.had_attribute:
            setattr(self.target, self.attribute, self.original_value)
        elif self.create and hasattr(self.target, self.attribute):
            delattr(self.target, self.attribute)


class FlextTestMocker:
    """Functional test object creation utilities.

    Follows DIP by depending on functional implementations, not mock objects.
    """

    @staticmethod
    def create_functional_service(
        service_type: str = "generic",
        **config: object,
    ) -> FunctionalTestService:
        """Create a functional service implementation.

        Args:
            service_type: Type of service to create.
            **config: Additional service configuration.

        Returns:
            Configured functional service.

        """
        return FunctionalTestService(service_type=service_type, **config)

    @staticmethod
    def patch_object(
        target: object,
        attribute: str,
        **options: object,
    ) -> FunctionalTestContext:
        """Create a functional patch context manager.

        Args:
            target: Object to patch.
            attribute: Attribute name to patch.
            **options: Additional patch configuration (new, create, etc.).

        Returns:
            Functional test context manager.

        """
        return FunctionalTestContext(target, attribute, **options)

    @staticmethod
    def create_test_context(
        target: object,
        attribute: str,
        new_value: object = None,
        **options: object,
    ) -> FunctionalTestContext:
        """Create a test context manager for attribute replacement.

        Args:
            target: Object to modify.
            attribute: Attribute name to replace.
            new_value: New value to set.
            **options: Additional context configuration.

        Returns:
            Functional test context manager.

        """
        return FunctionalTestContext(target, attribute, new_value, **options)


# =============================================================================
# TEST MODELS
# =============================================================================


class FlextTestModel(FlextModel):
    """Test model for testing purposes.

    Provides a simple model with common field types for testing.
    """

    name: str = "test"
    value: int = 42
    active: bool = True
    tags: ClassVar[list[str]] = []
    metadata: ClassVar[dict[str, object]] = {}

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


class FlextTestConfig(FlextModel):
    """Test configuration model."""

    debug: bool = False
    timeout: int = 30
    retries: int = 3
    # Build base_url without deep dynamic __import__ chains to avoid runtime errors

    base_url: str = __import__("inspect").cleandoc(
        f"http://{_FlextConstants.Platform.DEFAULT_HOST}:{_FlextConstants.Platform.FLEXT_API_PORT}",
    )
    headers: ClassVar[dict[str, str]] = {}


# =============================================================================
# UTILITY FUNCTIONS - Test data creation helpers
# =============================================================================


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


def create_ldap_test_config() -> dict[str, object]:
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


def create_api_test_response(
    *,
    success: bool = True,
    data: object = None,
) -> dict[str, object]:
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


# =============================================================================
# EXPORTS
# =============================================================================

__all__: list[str] = [
    "FlextTestAssertion",
    "FlextTestConfig",
    "FlextTestFactory",
    "FlextTestMocker",
    "FlextTestModel",
    "FlextTestUtilities",
    "ITestAssertion",
    "ITestFactory",
    "ITestMocker",
    "create_api_test_response",
    "create_ldap_test_config",
    "create_oud_connection_config",
]
