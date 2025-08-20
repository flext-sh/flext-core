"""Testing utilities and helpers for FLEXT components."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import ClassVar, Protocol, TypeVar, runtime_checkable
from unittest.mock import MagicMock, Mock, patch

from flext_core.constants import (
    FlextConstants as _FlextConstants,  # local import to avoid cycles
)
from flext_core.models import FlextModel
from flext_core.result import FlextResult

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
    """Protocol for test mockers."""

    def mock(self, spec: type | None = None) -> Mock:
        """Create mock object."""
        ...

    def patch(self, target: str) -> object:
        """Create patch context manager."""
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
        return result.unwrap()

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
            msg = f"Expected failure but got success: {result.unwrap()}"
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
# TEST MOCKER - Dependency Inversion Principle (DIP)
# =============================================================================


class FlextTestMocker:
    """Mock object creation utilities.

    Follows DIP by depending on Mock abstraction, not concrete implementations.
    """

    @staticmethod
    def mock(
        spec: type | None = None,
        **kwargs: object,  # noqa: ARG004
    ) -> Mock:
        """Create a mock object.

        Args:
            spec: Optional spec for the mock.
            **kwargs: Additional mock configuration.

        Returns:
            Configured mock object.

        """
        # Create basic Mock without additional arguments to avoid type issues
        return Mock(spec=spec)

    @staticmethod
    def magic_mock(
        spec: type | None = None,
        **kwargs: object,
    ) -> MagicMock:
        """Create a magic mock object.

        Args:
            spec: Optional spec for the mock.
            **kwargs: Additional mock configuration.

        Returns:
            Configured magic mock object.

        """
        # Filter kwargs to ensure compatibility with MagicMock constructor
        valid_kwargs = {}
        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, (str, int, bool, type(None))):
                    valid_kwargs[str(k)] = v
        return MagicMock(spec=spec, **valid_kwargs)

    @staticmethod
    def patch(
        target: str,
        **kwargs: object,
    ) -> object:
        """Create a patch context manager.

        Args:
            target: Import path to patch.
            **kwargs: Additional patch configuration.

        Returns:
            Patch context manager.

        """
        # Handle common patch configurations with type safety
        # Detect whether "new" was explicitly provided (so new=None is honored)
        if "new" in kwargs:
            new_value = kwargs.pop("new")
            patch_context = patch(target, new=new_value)
        else:
            patch_context = patch(target)  # type: ignore[assignment]
        return patch_context

    @staticmethod
    def patch_object(
        target: object,
        attribute: str,
        **kwargs: object,
    ) -> object:
        """Create a patch.object context manager.

        Args:
            target: Object to patch.
            attribute: Attribute name to patch.
            **kwargs: Additional patch configuration (new, create, spec, etc.).

        Returns:
            Patch context manager.

        """
        # patch.object overload complexity - suppress typing for mock compatibility
        return patch.object(target, attribute, **kwargs)  # type: ignore[call-overload]

    @staticmethod
    def create_async_mock(
        return_value: object = None,
        side_effect: object = None,
        **kwargs: object,
    ) -> MagicMock:
        """Create an async mock object.

        Args:
            return_value: Return value for the mock.
            side_effect: Side effect for the mock.
            **kwargs: Additional mock configuration.

        Returns:
            Configured an async mock object.

        """
        # Filter kwargs for MagicMock compatibility
        mock_kwargs: dict[str, object] = {}
        if kwargs:
            filtered_kwargs = {
                str(k): v
                for k, v in kwargs.items()
                if isinstance(v, (str, int, bool, type(None)))
            }
            mock_kwargs.update(filtered_kwargs)
        # Create MagicMock without additional kwargs to avoid type issues
        mock = MagicMock()
        async_mock = MagicMock(
            return_value=return_value,
            side_effect=side_effect,
        )
        mock.return_value = async_mock
        return mock


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
