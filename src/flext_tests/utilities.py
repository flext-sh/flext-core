"""Test utilities for FLEXT ecosystem tests.

Provides essential test utilities extending FlextUtilities with test-specific
helpers for result validation, context management, and test data creation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import csv
import hashlib
import os
import re
from collections.abc import Callable, Generator, Mapping, Sequence, Sized
from contextlib import contextmanager
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING

from flext_core import (
    FlextContext,
    FlextDispatcher,
    FlextRegistry,
    FlextSettings,
    FlextTypes as t,
    FlextUtilities,
    T,
    p,
    r,
)
from flext_core.utilities import u as u_core
from flext_tests.constants import c
from flext_tests.models import m

if TYPE_CHECKING:
    from pydantic import BaseModel

    from flext_core._models.base import FlextModelsBase

# Type alias for model factory methods
ModelFactory = Callable[..., T]


class FlextTestsUtilities(FlextUtilities):
    """Test utilities for FLEXT ecosystem - extends FlextUtilities.

    Provides essential test helpers that complement FlextUtilities.
    All FlextUtilities functionality is available via inheritance.
    """

    class Tests:
        """Test-specific utilities namespace.

        All test utilities organized under u.Tests.* pattern.
        """

        class Result:
            """Result helpers for test assertions."""

            @staticmethod
            def assert_success[TResult](
                result: r[TResult] | p.Result[TResult],
                error_msg: str | None = None,
            ) -> TResult:
                """Assert result is success and return unwrapped value.

                Args:
                    result: FlextResult or Result protocol to check
                    error_msg: Optional custom error message

                Returns:
                    Unwrapped value from result

                Raises:
                    AssertionError: If result is failure

                """
                if not result.is_success:
                    msg = (
                        error_msg or f"Expected success but got failure: {result.error}"
                    )
                    raise AssertionError(msg)
                # Protocol guarantees value property exists
                value: TResult = result.value
                return value

            @staticmethod
            def assert_failure[TResult](
                result: r[TResult] | p.Result[TResult],
                expected_error: str | None = None,
            ) -> str:
                """Assert result is failure and return error message.

                Args:
                    result: FlextResult or Result protocol to check
                    expected_error: Optional expected error substring

                Returns:
                    Error message from result

                Raises:
                    AssertionError: If result is success

                """
                if result.is_success:
                    msg = f"Expected failure but got success: {result.value}"
                    raise AssertionError(msg)
                error = result.error
                if error is None:
                    msg = "Expected error but got None"
                    raise AssertionError(msg)
                if expected_error and expected_error not in error:
                    msg = (
                        f"Expected error containing '{expected_error}' but got: {error}"
                    )
                    raise AssertionError(msg)
                return error

            @staticmethod
            def assert_success_with_value[T](
                result: r[T] | p.Result[T],
                expected_value: T,
            ) -> None:
                """Assert result is success and has expected value.

                Args:
                    result: FlextResult or Result protocol to check
                    expected_value: Expected value

                Raises:
                    AssertionError: If result is failure or value doesn't match

                """
                if not result.is_success:
                    msg = f"Expected success, got failure: {result.error}"
                    raise AssertionError(msg)
                assert result.value == expected_value

            @staticmethod
            def assert_failure_with_error[T](
                result: r[T] | p.Result[T],
                expected_error: str | None = None,
            ) -> None:
                """Assert result is failure and has expected error.

                Args:
                    result: FlextResult or Result protocol to check
                    expected_error: Optional expected error substring

                Raises:
                    AssertionError: If result is success or error doesn't match

                """
                if result.is_success:
                    msg = f"Expected failure, got success: {result.value}"
                    raise AssertionError(msg)
                if expected_error:
                    assert result.error is not None
                    assert expected_error in result.error

            # Backward compatibility aliases (old API names)
            assert_result_success = assert_success
            assert_result_failure = assert_failure
            assert_result_failure_with_error = assert_failure_with_error

            @staticmethod
            def create_success_result[T](value: T) -> r[T]:
                """Create a success result with the given value.

                Args:
                    value: Value for the success result

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult with success and value

                """
                return r[T].ok(value)

            @staticmethod
            def create_failure_result(error: str) -> r[object]:
                """Create a failure result with the given error.

                Args:
                    error: Error message for the failure result

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult with failure and error message

                """
                return r[object].fail(error)

        class TestContext:
            """Context managers for tests."""

            @staticmethod
            @contextmanager
            def temporary_attribute(
                target: object,
                attribute: str,
                value: t.GeneralValueType,
            ) -> Generator[None]:
                """Temporarily set attribute on target object.

                Args:
                    target: Object to modify
                    attribute: Attribute name
                    value: Temporary value

                Yields:
                    None

                """
                attribute_existed = hasattr(target, attribute)
                original_value = (
                    getattr(target, attribute, None) if attribute_existed else None
                )
                setattr(target, attribute, value)
                try:
                    yield
                finally:
                    if attribute_existed:
                        setattr(target, attribute, original_value)
                    else:
                        delattr(target, attribute)

        class Factory:
            """Factory helpers for test data creation."""

            @staticmethod
            def create_result[T](
                value: T | None = None,
                *,
                error: str | None = None,
            ) -> r[T]:
                """Create FlextResult for tests.

                Args:
                    value: Value for success result
                    error: Error message for failure result

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult with value or error

                """
                if error is not None:
                    return r[T].fail(error)
                if value is not None:
                    return r[T].ok(value)
                return r[T].fail("No value or error provided")

            @staticmethod
            def create_test_data(
                **kwargs: t.GeneralValueType,
            ) -> dict[str, t.GeneralValueType]:
                """Create test data dictionary.

                Args:
                    **kwargs: Key-value pairs for test data

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Configuration dictionary

                """
                return dict(kwargs)

            # =====================================================================
            # Operations - Reusable operation factories for testing
            # =====================================================================

            @staticmethod
            def simple_operation() -> t.GeneralValueType:
                """Execute simple operation returning success message.

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Success message string from constants.

                """
                return c.Tests.Factory.SUCCESS_MESSAGE

            @staticmethod
            def add_operation(
                a: t.GeneralValueType,
                b: t.GeneralValueType,
            ) -> t.GeneralValueType:
                """Execute add operation for numeric or string values.

                Args:
                    a: First operand (numeric or string)
                    b: Second operand (numeric or string)

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Sum if both numeric, concatenation otherwise.

                """
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return a + b
                return str(a) + str(b)

            @staticmethod
            def format_operation(name: str, value: int = 10) -> str:
                """Execute format operation returning formatted string.

                Args:
                    name: Name part of format
                    value: Value part of format (default: 10)

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Formatted string "name: value".

                """
                return f"{name}: {value}"

            @staticmethod
            def create_error_operation(
                error_message: str,
            ) -> Callable[[], t.GeneralValueType]:
                """Create callable that raises ValueError.

                Args:
                    error_message: Error message for ValueError

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Callable that raises ValueError when called.

                """

                def error_op() -> t.GeneralValueType:
                    raise ValueError(error_message)

                return error_op

            @staticmethod
            def create_type_error_operation(
                error_message: str,
            ) -> Callable[[], t.GeneralValueType]:
                """Create callable that raises TypeError.

                Args:
                    error_message: Error message for TypeError

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Callable that raises TypeError when called.

                """

                def type_error_op() -> t.GeneralValueType:
                    raise TypeError(error_message)

                return type_error_op

            # =====================================================================
            # Service execution - Reusable service execution helpers
            # =====================================================================

            @staticmethod
            def execute_user_service(
                overrides: dict[str, t.GeneralValueType],
            ) -> r[t.GeneralValueType]:
                """Execute user service operation.

                Args:
                    overrides: Service configuration overrides

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult with user data.

                """
                user_id = "default_123" if overrides.get("default") else "test_123"
                user_data: t.GeneralValueType = {
                    "user_id": user_id,
                    "email": "test@example.com",
                }
                return r[t.GeneralValueType].ok(user_data)

            @staticmethod
            def execute_complex_service(
                validation_result: r[bool],
            ) -> r[t.GeneralValueType]:
                """Execute complex service operation.

                Args:
                    validation_result: Result of business rules validation

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult with service data or error.

                """
                if validation_result.is_failure:
                    return r[t.GeneralValueType].fail(
                        validation_result.error or "Validation failed",
                    )
                result_data: t.GeneralValueType = {"result": "success"}
                return r[t.GeneralValueType].ok(result_data)

            @staticmethod
            def execute_default_service(
                service_type: str,
            ) -> r[t.GeneralValueType]:
                """Execute default service operation.

                Args:
                    service_type: Type of service

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult with service type data.

                """
                service_data: t.GeneralValueType = {"service_type": service_type}
                return r[t.GeneralValueType].ok(service_data)

            @staticmethod
            def generate_id() -> str:
                """Generate unique ID using u.generate().

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Generated UUID string.

                """
                return u_core.generate()

            @staticmethod
            def generate_short_id(length: int = 8) -> str:
                """Generate short unique ID using u.generate('ulid', length=...).

                Args:
                    length: Length of ID (default: 8)

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Generated short ID string.

                """
                return u_core.generate("ulid", length=length)

        # Compatibility aliases for existing test code
        class TestUtilities:
            """Compatibility alias - use Result instead."""

            @staticmethod
            def assert_result_success[TResult](
                result: r[TResult] | p.Result[TResult],
            ) -> None:
                """Assert result is success - compatibility method."""
                _ = FlextTestsUtilities.Tests.Result.assert_success(result)

            @staticmethod
            def assert_result_failure[TResult](
                result: r[TResult] | p.Result[TResult],
            ) -> None:
                """Assert result is failure - compatibility method."""
                _ = FlextTestsUtilities.Tests.Result.assert_failure(result)

        class ResultHelpers:
            """Result helpers for test creation and assertions."""

            @staticmethod
            def create_success_result[T](value: T) -> r[T]:
                """Create a success result with the given value.

                Args:
                    value: Value for the success result

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult with success and value

                """
                return r[T].ok(value)

            @staticmethod
            def create_failure_result(error: str) -> r[object]:
                """Create a failure result with the given error.

                Args:
                    error: Error message for the failure result

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult with failure and error

                """
                return r[object].fail(error)

            @staticmethod
            def assert_success_with_value[T](
                result: r[T],
                expected_value: T,
            ) -> None:
                """Assert result is success with expected value (compat)."""
                FlextTestsUtilities.Tests.Result.assert_success_with_value(
                    result,
                    expected_value,
                )

            @staticmethod
            def assert_failure_with_error[T](
                result: r[T],
                expected_error: str | None = None,
            ) -> None:
                """Assert result is failure with expected error (compat)."""
                FlextTestsUtilities.Tests.Result.assert_failure_with_error(
                    result,
                    expected_error,
                )

            @staticmethod
            def assert_result_success_and_unwrap[T](
                result: r[T],
            ) -> T:
                """Assert result is success and return unwrapped value.

                Args:
                    result: FlextResult to check and unwrap

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    The unwrapped value if result is success

                Raises:
                    AssertionError: If result is not success

                """
                _ = FlextTestsUtilities.Tests.Result.assert_success(result)
                return result.value

            @staticmethod
            def assert_result_failure_with_error[T](
                result: r[T],
                expected_error: str,
            ) -> None:
                """Assert result failure with error (compat alias).

                Args:
                    result: FlextResult to check
                    expected_error: Expected error substring

                Raises:
                    AssertionError: If result is not failure or error mismatch

                """
                FlextTestsUtilities.Tests.Result.assert_failure_with_error(
                    result,
                    expected_error,
                )

            @staticmethod
            def assert_result_success_and_type[T](
                result: r[T],
                expected_type: str | type[object] | None = None,
            ) -> T:
                """Assert result is success and return unwrapped value (type-safe).

                Args:
                    result: FlextResult to check
                    expected_type: Optional type hint (for documentation)

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Unwrapped value from result

                Raises:
                    AssertionError: If result is failure

                """
                return FlextTestsUtilities.Tests.Result.assert_success(result)

            @staticmethod
            def assert_result_success_and_unwrap_string(
                result: p.Result[str],
            ) -> str:
                """Assert result is success and return unwrapped string.

                Args:
                    result: FlextResult[str] to check

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Unwrapped string value

                Raises:
                    AssertionError: If result is failure

                """
                return FlextTestsUtilities.Tests.Result.assert_success(result)

            @staticmethod
            def assert_result_success_and_unwrap_list[T](
                result: p.Result[list[T]],
            ) -> list[T]:
                """Assert result is success and return unwrapped list.

                Args:
                    result: FlextResult[list[T]] to check

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Unwrapped list value

                Raises:
                    AssertionError: If result is failure

                """
                return FlextTestsUtilities.Tests.Result.assert_success(result)

        class GenericHelpers:
            """Generic helpers for test data creation."""

            @staticmethod
            def create_result_from_value[T](
                value: T | None,
                error_on_none: str = "Value cannot be None",
                default_on_none: T | None = None,
            ) -> r[T]:
                """Create result from value, failing if None (unless default).

                Args:
                    value: Value to wrap in result
                    error_on_none: Error message if value is None
                    default_on_none: Default value to use if value is None

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult with success or failure

                """
                if value is None:
                    if default_on_none is not None:
                        return r[T].ok(default_on_none)
                    return r[T].fail(error_on_none)
                return r[T].ok(value)

            @staticmethod
            def validate_model_attributes(
                model: object,
                required_attrs: list[str],
                optional_attrs: list[str] | None = None,
            ) -> r[bool]:
                """Validate model has required attributes.

                Args:
                    model: Model object to validate
                    required_attrs: List of required attribute names
                    optional_attrs: Optional list of optional attribute names

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult with True if all required attrs exist

                """
                missing = [attr for attr in required_attrs if not hasattr(model, attr)]
                if missing:
                    return r[bool].fail(f"Missing required attributes: {missing}")
                return r[bool].ok(True)

            @staticmethod
            def create_parametrized_cases(
                success_values: list[t.GeneralValueType],
                failure_errors: list[str] | None = None,
                *,
                error_codes: list[str | None] | None = None,
            ) -> list[
                tuple[
                    r[t.GeneralValueType],
                    bool,
                    t.GeneralValueType | None,
                    str | None,
                ]
            ]:
                """Create parametrized test cases from values and errors.

                Args:
                    success_values: List of values for success results
                    failure_errors: Optional list of error messages for failure results
                    error_codes: Optional list of error codes for failure results

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    List of tuples (result, is_success, value, error)

                """
                cases: list[
                    tuple[
                        r[t.GeneralValueType],
                        bool,
                        t.GeneralValueType | None,
                        str | None,
                    ]
                ] = []

                # Create success cases
                for value in success_values:
                    result = r[t.GeneralValueType].ok(value)
                    cases.append((result, True, value, None))

                # Create failure cases
                if failure_errors:
                    codes = error_codes or [None] * len(failure_errors)
                    for i, error in enumerate(failure_errors):
                        error_code = codes[i] if i < len(codes) else None
                        result = r[t.GeneralValueType].fail(
                            error,
                            error_code=error_code,
                        )
                        cases.append((result, False, None, error))

                return cases

            @staticmethod
            def assert_result_chain[T](
                results: list[r[T]],
                expected_successes: int | None = None,
                expected_failures: int | None = None,
                expected_success_count: int | None = None,
                expected_failure_count: int | None = None,
                first_failure_index: int | None = None,
            ) -> None:
                """Assert result chain has expected success/failure counts.

                Args:
                    results: List of results to check
                    expected_successes: Expected number of successes
                    expected_failures: Expected number of failures
                    expected_success_count: Alias for expected_successes
                    expected_failure_count: Alias for expected_failures
                    first_failure_index: Expected index of first failure (if any)

                Raises:
                    AssertionError: If counts don't match

                """
                # Use alias if main param not provided
                successes_expected = expected_successes or expected_success_count
                failures_expected = expected_failures or expected_failure_count

                successes = sum(1 for res in results if res.is_success)
                failures = sum(1 for res in results if res.is_failure)

                if successes_expected is not None:
                    assert successes == successes_expected, (
                        f"Expected {successes_expected} successes, got {successes}"
                    )
                if failures_expected is not None:
                    assert failures == failures_expected, (
                        f"Expected {failures_expected} failures, got {failures}"
                    )

                # Check first failure index
                if first_failure_index is not None:
                    actual_first_failure = next(
                        (i for i, res in enumerate(results) if res.is_failure),
                        None,
                    )
                    assert actual_first_failure == first_failure_index, (
                        f"Expected first failure at index {first_failure_index}, "
                        f"got {actual_first_failure}"
                    )
                elif failures == 0:
                    # Verify no failures when first_failure_index is None
                    actual_first_failure = next(
                        (i for i, res in enumerate(results) if res.is_failure),
                        None,
                    )
                    assert actual_first_failure is None, (
                        f"Expected no failures but found first failure at index "
                        f"{actual_first_failure}"
                    )

            @staticmethod
            def normalize_dict_values_to_lists(
                data: dict[str, str | list[str] | tuple[str, ...] | set[str] | None]
                | None,
            ) -> dict[str, list[str]]:
                """Normalize dictionary values to lists.

                Converts single values to single-element lists for LDAP attribute
                compatibility where all values must be list[str].

                Args:
                    data: Dictionary with mixed value types (strings, lists, ec.)

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Dictionary with all values as lists of strings

                """
                if data is None:
                    return {}

                result: dict[str, list[str]] = {}
                for key, value in data.items():
                    # Type narrowing: value is from data.items(), which is dict[str, str | None]
                    # But runtime checks may reveal other types (set, list, etc.), so check first
                    if value is None:
                        result[key] = []
                    elif isinstance(value, (list, tuple)):
                        # Type narrowing: value is list or tuple after isinstance check
                        value_seq: Sequence[str] = value
                        result[key] = [str(v) for v in value_seq]
                    elif isinstance(value, set):
                        # Type narrowing: value is set after isinstance check
                        value_set: set[str] = value
                        result[key] = [str(v) for v in value_set]
                    else:
                        # value is str (after None and iterable checks above)
                        value_str: str = value
                        result[key] = [value_str]

                return result

        class ModelTestHelpers:
            """Model testing helpers."""

            @staticmethod
            def assert_model_creation_success[TResult](
                factory_method: ModelFactory[TResult],
                expected_attrs: t.ConfigurationMapping,
                **factory_kwargs: t.GeneralValueType,
            ) -> TResult:
                """Assert successful model creation and validate attributes.

                Args:
                    factory_method: Factory method to call
                    expected_attrs: Expected attributes to validate
                    **factory_kwargs: Factory method arguments

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Created model instance

                Raises:
                    AssertionError: If creation fails or attributes don't match

                """
                instance = factory_method(**factory_kwargs)
                for key, expected_value in expected_attrs.items():
                    actual_value = getattr(instance, key, None)
                    msg = f"Attr {key}: expected {expected_value}, got {actual_value}"
                    assert actual_value == expected_value, msg
                return instance

        class RegistryHelpers:
            """Registry testing helpers - use FlextRegistry directly when possible."""

            @staticmethod
            def create_test_registry() -> FlextRegistry:
                """Create a test registry instance.

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    New FlextRegistry instance

                """
                return FlextRegistry()

        class ConfigHelpers:
            """Config testing helpers - use FlextSettings directly when possible."""

            @staticmethod
            def create_test_config(
                **kwargs: t.FlexibleValue,
            ) -> FlextSettings:
                """Create a test config instance.

                Args:
                    **kwargs: Config field values

                Returns:
                    New FlextSettings instance

                """
                return FlextSettings.materialize(config_overrides=kwargs)

            @staticmethod
            def assert_config_fields(
                config: FlextSettings,
                expected_fields: t.ConfigurationMapping,
            ) -> None:
                """Assert config has expected field values.

                Args:
                    config: Config instance to check
                    expected_fields: Expected field values

                Raises:
                    AssertionError: If fields don't match

                """
                for key, expected_value in expected_fields.items():
                    actual_value = getattr(config, key, None)
                    msg = f"Config {key}: expected {expected_value}, got {actual_value}"
                    assert actual_value == expected_value, msg

            @staticmethod
            @contextmanager
            def env_vars_context(
                env_vars: dict[str, t.GeneralValueType],
                vars_to_clear: list[str] | None = None,
            ) -> Generator[None]:
                """Context manager for temporary environment variable changes.

                Args:
                    env_vars: Environment variables to set
                    vars_to_clear: Variables to clear on entry

                Yields:
                    None

                """
                original_values: dict[str, str | None] = {}

                # Save and clear specified vars
                if vars_to_clear:
                    for var in vars_to_clear:
                        original_values[var] = os.environ.get(var)
                        if var in os.environ:
                            del os.environ[var]

                # Save original values and set new ones
                for key, value in env_vars.items():
                    if key not in original_values:
                        original_values[key] = os.environ.get(key)
                    os.environ[key] = str(value)

                try:
                    yield
                finally:
                    # Restore original values
                    for key, original in original_values.items():
                        if original is None:
                            if key in os.environ:
                                del os.environ[key]
                        else:
                            os.environ[key] = original

        class ContextHelpers:
            """Helpers for context testing."""

            @staticmethod
            def create_test_context() -> FlextContext:
                """Create a test context instance.

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    New FlextContext instance

                """
                return FlextContext.create()

            @staticmethod
            def assert_context_get_success(
                context: FlextContext,
                key: str,
                expected_value: t.GeneralValueType,
            ) -> None:
                """Assert context get returns expected value.

                Args:
                    context: FlextContext instance
                    key: Key to get
                    expected_value: Expected value

                Raises:
                    AssertionError: If value doesn't match

                """
                result = context.get(key)
                assert result.is_success, (
                    f"Expected success for key '{key}', got: {result.error}"
                )
                assert result.value == expected_value, (
                    f"Expected {expected_value} for key '{key}', got {result.value}"
                )

            @staticmethod
            def clear_context() -> None:
                """Clear the global context."""
                FlextContext.Utilities.clear_context()

        class ContainerHelpers:
            """Helpers for container testing."""

            @staticmethod
            def create_factory[TFactory](
                return_value: TFactory,
            ) -> Callable[[], TFactory]:
                """Create a factory function that returns a fixed value.

                Args:
                    return_value: Value to return from factory

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Factory function

                """

                def factory() -> TFactory:
                    return return_value

                return factory

            @staticmethod
            def create_counting_factory[TFactory](
                return_value: TFactory,
            ) -> tuple[Callable[[], TFactory], Callable[[], int]]:
                """Create a factory that counts invocations.

                Args:
                    return_value: Value to return from factory

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Tuple of (factory function, count getter)

                """
                count = [0]

                def factory() -> TFactory:
                    count[0] += 1
                    return return_value

                def get_count() -> int:
                    return count[0]

                return factory, get_count

        class HandlerHelpers:
            """Helpers for handler testing."""

            @staticmethod
            def create_handler_config(
                handler_id: str,
                handler_name: str,
                handler_type: c.Cqrs.HandlerType | None = None,
                handler_mode: c.Cqrs.HandlerType | None = None,
                command_timeout: int | None = None,
                max_command_retries: int | None = None,
                metadata: FlextModelsBase.Metadata | None = None,
            ) -> m.Handler:
                """Create a handler configuration model.

                Args:
                    handler_id: Handler identifier
                    handler_name: Handler name
                    handler_type: Optional handler type (default: COMMAND)
                    handler_mode: Optional handler mode (default: type or COMMAND)
                    command_timeout: Optional command timeout in seconds
                    max_command_retries: Optional max retry count
                    metadata: Optional handler metadata

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Handler configuration model

                """
                # Default values
                h_type = handler_type or c.Cqrs.HandlerType.COMMAND
                h_mode = handler_mode or h_type

                return m.Handler(
                    handler_id=handler_id,
                    handler_name=handler_name,
                    handler_type=h_type,
                    handler_mode=h_mode,
                    command_timeout=command_timeout or c.Cqrs.DEFAULT_COMMAND_TIMEOUT,
                    max_command_retries=max_command_retries
                    or c.Cqrs.DEFAULT_MAX_COMMAND_RETRIES,
                    metadata=metadata,
                )

        class DispatcherHelpers:
            """Helpers for dispatcher testing."""

            @staticmethod
            def create_test_dispatcher(
                handlers: dict[str, t.HandlerType] | None = None,
            ) -> FlextDispatcher:
                """Create a test dispatcher instance with optional handlers.

                Args:
                    handlers: Optional dict of handler name -> handler instance

                Returns:
                    New FlextDispatcher instance with registered handlers

                """
                dispatcher = FlextDispatcher()
                if handlers:
                    for name, handler in handlers.items():
                        _ = dispatcher.register_handler(name, handler)
                return dispatcher

            @staticmethod
            def assert_handler_result(
                result: p.Result[t.GeneralValueType],
                *,
                expected_success: bool = True,
                expected_value: t.GeneralValueType | None = None,
                expected_error: str | None = None,
            ) -> None:
                """Assert handler result matches expectations.

                Args:
                    result: FlextResult from handler
                    expected_success: Whether result should be success
                    expected_value: Expected value if success
                    expected_error: Expected error substring if failure

                Raises:
                    AssertionError: If result doesn't match expectations

                """
                if expected_success:
                    assert result.is_success, (
                        f"Expected success, got failure: {result.error}"
                    )
                    if expected_value is not None:
                        assert result.value == expected_value, (
                            f"Expected {expected_value}, got {result.value}"
                        )
                else:
                    assert result.is_failure, (
                        f"Expected failure, got success: {result.value}"
                    )
                    if expected_error is not None:
                        assert expected_error in str(result.error), (
                            f"Expected error '{expected_error}' in '{result.error}'"
                        )

        class ParserHelpers:
            """Helpers for parser testing."""

            @staticmethod
            def execute_and_assert_parser_result(
                operation: Callable[[], r[t.GeneralValueType]],
                expected_value: t.GeneralValueType | None = None,
                expected_error: str | None = None,
                description: str = "",
            ) -> None:
                """Execute parser operation and assert result.

                Args:
                    operation: Callable that returns a FlextResult
                    expected_value: Expected value on success
                    expected_error: Expected error substring on failure
                    description: Test case description for error messages

                """
                result = operation()

                if expected_error is not None:
                    assert result.is_failure, (
                        f"Expected failure for: {description}, got success"
                    )
                    m = f"'{expected_error}' not in '{result.error}': {description}"
                    assert expected_error in str(result.error), m
                else:
                    assert result.is_success, (
                        f"Expected success for: {description}, got: {result.error}"
                    )
                    if expected_value is not None:
                        m = f"Want {expected_value}, got {result.value}: {description}"
                        assert result.value == expected_value, m

        class TestCaseHelpers:
            """Helpers for creating test cases."""

            @staticmethod
            def create_operation_test_case(
                operation: str,
                description: str,
                input_data: dict[str, t.GeneralValueType],
                expected_result: t.GeneralValueType,
                **kwargs: t.GeneralValueType,
            ) -> dict[str, t.GeneralValueType]:
                """Create a test case dict for operation testing.

                Args:
                    operation: Operation name
                    description: Test case description
                    input_data: Input data for the operation
                    expected_result: Expected result or type
                    **kwargs: Additional test case parameters

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Test case dictionary

                """
                result: dict[str, t.GeneralValueType] = {
                    "operation": operation,
                    "description": description,
                    "input_data": input_data,
                    "expected_result": expected_result,
                }
                result.update(kwargs)
                return result

            @staticmethod
            def create_batch_operation_test_cases(
                operation: str,
                descriptions: list[str],
                input_data_list: list[dict[str, t.GeneralValueType]],
                expected_results: list[t.GeneralValueType],
                **common_kwargs: t.GeneralValueType,
            ) -> list[dict[str, t.GeneralValueType]]:
                """Create batch test cases for operation testing.

                Args:
                    operation: Operation name
                    descriptions: List of descriptions
                    input_data_list: List of input data dicts
                    expected_results: List of expected results
                    **common_kwargs: Common parameters for all cases

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    List of test case dictionaries

                """
                cases: list[dict[str, t.GeneralValueType]] = []
                for desc, data, expected in zip(
                    descriptions,
                    input_data_list,
                    expected_results,
                    strict=True,
                ):
                    th = FlextTestsUtilities.Tests.TestCaseHelpers
                    case = th.create_operation_test_case(
                        operation=operation,
                        description=desc,
                        input_data=data,
                        expected_result=expected,
                        **common_kwargs,
                    )
                    cases.append(case)
                return cases

            @staticmethod
            def execute_and_assert_operation_result(
                operation: Callable[[], t.GeneralValueType],
                test_case: dict[str, t.GeneralValueType],
            ) -> None:
                """Execute operation and assert result.

                Args:
                    operation: Callable that returns the result
                    test_case: Test case dict with expected_result

                Raises:
                    AssertionError: If result doesn't match expectation

                """
                result = operation()
                expected = test_case.get("expected_result")

                # Handle type expectations (e.g., int, bool, str)
                if isinstance(expected, type):
                    m = f"Want type {expected.__name__}, got {type(result).__name__}"
                    assert isinstance(result, expected), m
                else:
                    assert result == expected, f"Expected {expected}, got {result}"

        class DomainHelpers:
            """Helpers for domain model testing."""

            @staticmethod
            def create_test_entity_instance[TEntity](
                name: str,
                value: t.GeneralValueType,
                entity_class: Callable[..., TEntity],
                *,
                remove_id: bool = False,
            ) -> TEntity:
                """Create a test entity instance.

                Args:
                    name: Entity name
                    value: Entity value
                    entity_class: Entity class or factory callable
                    remove_id: If True, remove unique_id attribute

                Returns:
                    TEntity: Created entity instance

                """
                entity = entity_class(name=name, value=value)
                if remove_id and hasattr(entity, "unique_id"):
                    delattr(entity, "unique_id")
                return entity

            @staticmethod
            def create_test_entities_batch[TEntity](
                names: list[str],
                values: list[t.GeneralValueType],
                entity_class: Callable[..., TEntity],
                remove_ids: list[bool] | None = None,
            ) -> r[list[TEntity]]:
                """Create batch of test entities.

                Args:
                    names: List of entity names
                    values: List of entity values
                    entity_class: Entity class to instantiate
                    remove_ids: List of booleans for ID removal

                Returns:
                    FlextResult[list[TEntity]]: Result containing list of entities or error

                """
                ids_removal = remove_ids or [False] * len(names)
                entities: list[TEntity] = []
                dh = FlextTestsUtilities.Tests.DomainHelpers
                for name, value, remove_id in zip(
                    names,
                    values,
                    ids_removal,
                    strict=True,
                ):
                    try:
                        entity = dh.create_test_entity_instance(
                            name=name,
                            value=value,
                            entity_class=entity_class,
                            remove_id=remove_id,
                        )
                        entities.append(entity)
                    except Exception as e:
                        return r[list[TEntity]].fail(
                            f"Failed to create entity {name}: {e}",
                        )
                return r[list[TEntity]].ok(entities)

            @staticmethod
            def create_test_value_object_instance[TValue](
                data: str,
                count: int,
                value_class: Callable[..., TValue],
            ) -> TValue:
                """Create a test value object instance.

                Args:
                    data: Data field value
                    count: Count field value
                    value_class: Value object class or factory callable

                Returns:
                    TValue: Created value object instance

                """
                return value_class(data=data, count=count)

            @staticmethod
            def create_test_value_objects_batch[TValue](
                data_list: list[str],
                count_list: list[int],
                value_class: Callable[..., TValue],
            ) -> list[TValue]:
                """Create batch of test value objects.

                Args:
                    data_list: List of data values
                    count_list: List of count values
                    value_class: Value object class to instantiate

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    List of created value objects

                """
                return [
                    FlextTestsUtilities.Tests.DomainHelpers.create_test_value_object_instance(
                        data=data,
                        count=count,
                        value_class=value_class,
                    )
                    for data, count in zip(data_list, count_list, strict=True)
                ]

            @staticmethod
            def execute_domain_operation(
                operation: str,
                input_data: dict[str, t.GeneralValueType],
                **kwargs: t.GeneralValueType,
            ) -> object:
                """Execute a domain utility operation.

                Args:
                    operation: Operation name from FlextUtilities.Domain
                    input_data: Input data dictionary
                    **kwargs: Additional arguments

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Operation result (type depends on operation)

                """
                op_method = getattr(FlextUtilities.Domain, operation, None)
                if op_method is None:
                    msg = f"Unknown operation: {operation}"
                    raise ValueError(msg)

                # Merge input_data with kwargs
                all_args = {**input_data, **kwargs}
                return op_method(**all_args)

        class ExceptionHelpers:
            """Helpers for exception testing."""

            @staticmethod
            def create_metadata_object(
                attributes: dict[str, t.GeneralValueType],
            ) -> dict[str, t.GeneralValueType]:
                """Create a metadata object for exceptions.

                Args:
                    attributes: Metadata attributes

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Metadata object with attributes as dict

                """
                return {"attributes": attributes, **attributes}

        class MapperHelpers:
            """Helpers for data mapper testing."""

            @staticmethod
            def execute_mapper_operation(
                operation: str,
                input_data: dict[str, t.GeneralValueType],
                **kwargs: t.GeneralValueType,
            ) -> r[object]:
                """Execute a mapper utility operation.

                Args:
                    operation: Operation name from FlextUtilities.Mapper
                    input_data: Input data dictionary
                    **kwargs: Additional arguments

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    FlextResult from mapper operation

                """
                op_method = getattr(FlextUtilities.Mapper, operation, None)
                if op_method is None:
                    msg = f"Unknown operation: {operation}"
                    raise ValueError(msg)

                all_args = {**input_data, **kwargs}
                result = op_method(**all_args)
                # Type narrowing: op_method returns FlextResult
                if isinstance(result, r):
                    return result
                # If operation returns a value directly, wrap it
                return r[object].ok(result)

        class BadObjects:
            """Factory for objects that cause errors during testing."""

            class BadModelDump:
                """Object with model_dump that raises."""

                def model_dump(self) -> dict[str, t.GeneralValueType]:
                    """Raise error on model_dump."""
                    msg = "Bad model_dump"
                    raise RuntimeError(msg)

            class BadConfig:
                """Config object that raises on attribute access."""

                def __getattribute__(self, name: str) -> t.GeneralValueType:
                    """Raise error on attribute access - test helper for error testing."""
                    # Skip __class__ and other special attributes
                    if name.startswith("__") and name.endswith("__"):
                        result: t.GeneralValueType = super().__getattribute__(name)
                        return result
                    msg = f"Bad config: {name}"
                    raise AttributeError(msg)

            class BadConfigTypeError:
                """Config object that raises TypeError on attribute access."""

                def __getattribute__(self, name: str) -> t.GeneralValueType:
                    """Raise TypeError on attribute access - test helper for error testing."""
                    # Skip __class__ and other special attributes
                    if name.startswith("__") and name.endswith("__"):
                        result: t.GeneralValueType = super().__getattribute__(name)
                        return result
                    msg = f"Bad config type: {name}"
                    raise TypeError(msg)

        class ConstantsHelpers:
            """Helpers for testing FlextConstants."""

            @staticmethod
            def get_constant_by_path(path: str) -> object:
                """Get a constant value by dot-separated path.

                Args:
                    path: Dot-separated path like "Utilities.MAX_TIMEOUT_SECONDS"

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    The constant value at the given path

                """
                parts = path.split(".")
                current: object = c
                for part in parts:
                    current = getattr(current, part)
                return current

            @staticmethod
            def compile_pattern(pattern_attr: str) -> Pattern[str]:
                """Compile a regex pattern from FlextConstants.

                Args:
                    pattern_attr: Attribute name like "Patterns.EMAIL_REGEX"

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Compiled regex pattern

                """
                parts = pattern_attr.split(".")
                current: object = c
                for part in parts:
                    current = getattr(current, part)
                pattern_str = str(current)
                return re.compile(pattern_str, re.IGNORECASE)

        class Assertions:
            """Common assertion helpers for tests."""

            @staticmethod
            def assert_result_success(result: r[T]) -> T:
                """Assert result is success and return value.

                Args:
                    result: FlextResult to check

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    The success value

                Raises:
                    AssertionError: If result is not successful

                """
                assert result.is_success, (
                    f"Expected success, got failure: {result.error}"
                )
                return result.value

            @staticmethod
            def assert_result_failure(
                result: r[T],
                expected_error: str | None = None,
            ) -> str:
                """Assert result is failure and optionally check error message.

                Args:
                    result: FlextResult to check
                    expected_error: Optional expected error substring

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    The error message

                Raises:
                    AssertionError: If result is not failure or error doesn't match

                """
                assert result.is_failure, (
                    f"Expected failure, got success: {result.value}"
                )
                error = result.error or ""
                if expected_error is not None:
                    assert expected_error in error, (
                        f"Expected error '{expected_error}' not in '{error}'"
                    )
                return error

            @staticmethod
            def assert_result_failure_with_error(
                result: r[T],
                expected_error: str,
            ) -> None:
                """Assert result is failure with specific error message.

                Args:
                    result: FlextResult to check
                    expected_error: Expected error substring

                Raises:
                    AssertionError: If result is not failure or error doesn't match

                """
                _ = FlextTestsUtilities.Tests.Assertions.assert_result_failure(
                    result,
                    expected_error,
                )

            @staticmethod
            def assert_result_matches_expected(
                result: t.GeneralValueType,
                expected_type: type,
                description: str = "",
            ) -> None:
                """Assert result is instance of expected type.

                Args:
                    result: Value to check
                    expected_type: Expected type
                    description: Optional test description for error messages

                Raises:
                    AssertionError: If result is not instance of expected_type

                """
                assert isinstance(result, expected_type), (
                    f"Expected {expected_type.__name__}, got {type(result).__name__}"
                    f"{f' for {description}' if description else ''}"
                )

        class Files:
            """File utilities for test file operations.

            Provides reusable helper functions for file operations that can be
            used by FlextTestsFiles and other test utilities.
            """

            @staticmethod
            def compute_hash(path: Path, chunk_size: int | None = None) -> str:
                """Compute SHA256 hash of file.

                Args:
                    path: Path to file
                    chunk_size: Size of chunks to read (default: from constants)

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    SHA256 hash as hex string

                """
                size = chunk_size or c.Tests.Files.HASH_CHUNK_SIZE
                sha256 = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(size), b""):
                        sha256.update(chunk)
                return sha256.hexdigest()

            @staticmethod
            def detect_format(
                content: str
                | bytes
                | Mapping[str, t.GeneralValueType]
                | list[list[str]],
                name: str,
                fmt: str,
            ) -> str:
                """Detect file format from content type or filename.

                Args:
                    content: File content (type determines format)
                    name: Filename (extension hints format)
                    fmt: Explicit format override ("auto" for detection)

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Detected format string

                """
                if fmt != c.Tests.Files.Format.AUTO:
                    return fmt

                # Detect from content type
                if isinstance(content, bytes):
                    return c.Tests.Files.Format.BIN
                if u.is_type(content, "mapping"):
                    # Check extension for yaml vs json
                    ext = Path(name).suffix.lower()
                    if ext in {".yaml", ".yml"}:
                        return c.Tests.Files.Format.YAML
                    return c.Tests.Files.Format.JSON
                # Runtime check needed to distinguish nested sequences from flat sequences
                if u.is_type(content, "list") and all(
                    u.is_type(row, "list") for row in content
                ):
                    return c.Tests.Files.Format.CSV

                # Detect from extension
                return c.Tests.Files.get_format(Path(name).suffix)

            @staticmethod
            def detect_format_from_path(path: Path, fmt: str) -> str:
                """Detect format from file path.

                Args:
                    path: File path
                    fmt: Explicit format override ("auto" for detection)

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Detected format string

                """
                if fmt != c.Tests.Files.Format.AUTO:
                    return fmt
                return c.Tests.Files.get_format(path.suffix)

            @staticmethod
            def write_csv(
                path: Path,
                content: str
                | bytes
                | Mapping[str, t.GeneralValueType]
                | list[list[str]],
                headers: list[str] | None,
                delimiter: str | None = None,
                encoding: str | None = None,
            ) -> None:
                """Write CSV file.

                Args:
                    path: File path
                    content: Content to write (list of rows)
                    headers: Optional header row
                    delimiter: CSV delimiter (default: from constants)
                    encoding: File encoding (default: from constants)

                """
                delim = delimiter or c.Tests.Files.DEFAULT_CSV_DELIMITER
                enc = encoding or c.Tests.Files.DEFAULT_ENCODING
                with path.open("w", newline="", encoding=enc) as f:
                    writer = csv.writer(f, delimiter=delim)
                    if headers:
                        writer.writerow(headers)
                    if isinstance(content, list):
                        for row in content:
                            if isinstance(row, list):
                                writer.writerow(row)

            @staticmethod
            def read_csv(
                path: Path,
                delimiter: str | None = None,
                encoding: str | None = None,
                *,
                has_headers: bool = True,
            ) -> list[list[str]]:
                """Read CSV file.

                Args:
                    path: File path
                    delimiter: CSV delimiter (default: from constants)
                    encoding: File encoding (default: from constants)
                    has_headers: If True, skip first row (headers)

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    List of rows (each row is list of strings)

                """
                delim = delimiter or c.Tests.Files.DEFAULT_CSV_DELIMITER
                enc = encoding or c.Tests.Files.DEFAULT_ENCODING
                with path.open(newline="", encoding=enc) as f:
                    reader = csv.reader(f, delimiter=delim)
                    rows = list(reader)
                    if has_headers and rows:
                        return rows[1:]  # Skip header row
                    return rows

            @staticmethod
            def format_size(size: int) -> str:
                """Format size in human-readable format.

                Delegates to constants.Files.format_size for consistency.

                Args:
                    size: Size in bytes

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Human-readable size string like "1.2 KB"

                """
                return c.Tests.Files.format_size(size)

            @staticmethod
            def get_format_from_extension(extension: str) -> str:
                """Get format from file extension.

                Delegates to constants.Files.get_format for consistency.

                Args:
                    extension: File extension (e.g., ".json")

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Format string or "text" as default

                """
                return c.Tests.Files.get_format(extension)

        class Validator:
            """Validator utilities for architecture validation (tv.* methods).

            Provides reusable helper functions for validators. All validators
            should use these instead of implementing their own versions.
            """

            @staticmethod
            def is_approved(
                rule_id: str,
                file_path: Path,
                approved: dict[str, list[str]],
            ) -> bool:
                """Check if file is approved for this rule.

                Args:
                    rule_id: Rule identifier (e.g., "IMPORT-001")
                    file_path: Path to file being checked
                    approved: Dict mapping rule IDs to list of approved file patterns

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    True if file matches any approved pattern for this rule

                """
                patterns = approved.get(rule_id, [])
                file_str = str(file_path)
                return any(re.search(pattern, file_str) for pattern in patterns)

            @staticmethod
            def create_violation(
                file_path: Path,
                line_number: int,
                rule_id: str,
                lines: list[str],
                extra_desc: str = "",
            ) -> m.Tests.Validator.Violation:
                """Create a violation model using c.Tests.Validator.Rules.

                Args:
                    file_path: Path to file with violation
                    line_number: Line number of violation (1-indexed)
                    rule_id: Rule identifier (e.g., "IMPORT-001")
                    lines: File content as list of lines
                    extra_desc: Optional extra description

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Violation model instance

                """
                severity, desc = c.Tests.Validator.Rules.get(rule_id)
                description = f"{desc}: {extra_desc}" if extra_desc else desc
                line = lines[line_number - 1] if line_number <= len(lines) else ""
                return m.Tests.Validator.Violation(
                    file_path=file_path,
                    line_number=line_number,
                    rule_id=rule_id,
                    severity=severity,
                    description=description,
                    code_snippet=line.strip(),
                )

            @staticmethod
            def get_parent(tree: ast.AST, node: ast.AST) -> ast.AST | None:
                """Get parent node of an AST node.

                Args:
                    tree: AST tree root
                    node: Node to find parent of

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Parent node or None if not found

                """
                for parent in ast.walk(tree):
                    for child in ast.iter_child_nodes(parent):
                        if child is node:
                            return parent
                return None

            @staticmethod
            def get_exception_names(exc_type: ast.expr) -> set[str]:
                """Extract exception names from exception type AST node.

                Args:
                    exc_type: Exception type AST node

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Set of exception names found

                """
                names: set[str] = set()
                if isinstance(exc_type, ast.Name):
                    names.add(exc_type.id)
                elif isinstance(exc_type, ast.Tuple):
                    for elt in exc_type.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
                return names

            @staticmethod
            def is_any_type(node: ast.expr) -> bool:
                """Check if an annotation node represents the typing.Any type.

                Args:
                    node: AST annotation node

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    True if node represents typing.Any type annotation

                """
                return (
                    (isinstance(node, ast.Name) and node.id == "Any")
                    or (isinstance(node, ast.Attribute) and node.attr == "Any")
                    or (isinstance(node, ast.Constant) and node.value == "Any")
                )

            @staticmethod
            def find_line_number(lines: list[str], pattern: str) -> int:
                """Find line number containing pattern.

                Args:
                    lines: File content as list of lines
                    pattern: Pattern to search for

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    Line number (1-indexed) or 1 if not found

                """
                for i, line in enumerate(lines, start=1):
                    if pattern in line:
                        return i
                return 1

            @staticmethod
            def is_only_pass(body: list[ast.stmt]) -> bool:
                """Check if exception handler body contains only pass or ellipsis.

                Used by BYPASS-003 to detect exception swallowing patterns.

                Args:
                    body: AST statement list (exception handler body)

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    True if body contains only pass or ellipsis (...)

                """
                if len(body) == 1:
                    stmt = body[0]
                    if isinstance(stmt, ast.Pass):
                        return True
                    # Also check for ellipsis (...)
                    if (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and stmt.value.value is ...
                    ):
                        return True
                return False

            @staticmethod
            def is_real_comment(line: str, pattern: re.Pattern[str]) -> bool:
                """Check if pattern match is in a real comment, not inside a string.

                Used by validators to avoid false positives from patterns appearing
                in docstrings or string literals.

                Args:
                    line: Source code line
                    pattern: Compiled regex pattern to search

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    True if pattern appears in real code comment (after #),
                    not inside a string literal (single/double/triple quoted)

                """
                match = pattern.search(line)
                if not match:
                    return False

                pos = match.start()

                # Track quote state to determine if position is inside string
                in_single = False
                in_double = False
                in_triple_single = False
                in_triple_double = False
                i = 0
                while i < pos:
                    # Check for triple quotes first
                    if (
                        line[i : i + 3] == '"""'
                        and not in_single
                        and not in_triple_single
                    ):
                        in_triple_double = not in_triple_double
                        i += 3
                        continue
                    if (
                        line[i : i + 3] == "'''"
                        and not in_double
                        and not in_triple_double
                    ):
                        in_triple_single = not in_triple_single
                        i += 3
                        continue
                    # Check for single quotes
                    if (
                        line[i] == '"'
                        and not in_single
                        and not in_triple_single
                        and not in_triple_double
                    ):
                        in_double = not in_double
                    elif (
                        line[i] == "'"
                        and not in_double
                        and not in_triple_single
                        and not in_triple_double
                    ):
                        in_single = not in_single
                    i += 1

                # If inside any string, it's not a real comment
                return not (
                    in_single or in_double or in_triple_single or in_triple_double
                )

        class DeepMatch:
            """Deep structural matching utilities - delegates to u.Mapper.extract().

            Follows FLEXT patterns:
            - Zero code duplication - delegates to flext-core utilities
            - Uses t.Tests.Matcher.DeepSpec for type safety
            - Returns m.Tests.Matcher.DeepMatchResult for structured results
            - Supports unlimited nesting depth via dot notation

            All operations delegate to FlextUtilities.Mapper.extract() for
            path extraction, ensuring consistency with flext-core patterns.
            """

            @staticmethod
            def match(
                obj: BaseModel | Mapping[str, t.GeneralValueType],
                spec: Mapping[str, object | Callable[[object], bool]],
                *,
                path_sep: str = ".",
            ) -> m.Tests.Matcher.DeepMatchResult:
                """Match object against deep specification.

                Uses u.Mapper.extract() for path extraction - NO code duplication.
                Supports unlimited nesting depth via dot notation paths.

                Args:
                    obj: Object to match against (dict or Pydantic model)
                    spec: DeepSpec mapping of path -> expected value or predicate
                    path_sep: Path separator (default: ".")

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    DeepMatchResult with match status and details

                Examples:
                    result = u.Tests.DeepMatch.match(
                        data,
                        {
                            "user.name": "John",
                            "user.email": lambda e: "@" in e,
                            "user.profile.age": 25,
                        }
                    )
                    if not result.matched:
                        raise AssertionError(f"Failed at {result.path}: {result.reason}")

                """
                for path, expected in spec.items():
                    result = FlextUtilities.Mapper.extract(
                        obj,
                        path,
                        separator=path_sep,
                    )
                    if result.is_failure:
                        return m.Tests.Matcher.DeepMatchResult(
                            path=path,
                            expected=expected,
                            actual=None,
                            matched=False,
                            reason=f"Path not found: {path}",
                        )

                    actual = result.value
                    if callable(expected):
                        if not expected(actual):
                            return m.Tests.Matcher.DeepMatchResult(
                                path=path,
                                expected="<predicate>",
                                actual=actual,
                                matched=False,
                                reason="Predicate failed",
                            )
                    elif actual != expected:
                        return m.Tests.Matcher.DeepMatchResult(
                            path=path,
                            expected=expected,
                            actual=actual,
                            matched=False,
                            reason="Value mismatch",
                        )

                return m.Tests.Matcher.DeepMatchResult(
                    path="",
                    expected=spec,
                    actual=obj,
                    matched=True,
                )

        class Length:
            """Length validation utilities - delegates to u.chk().

            Follows FLEXT patterns:
            - Zero code duplication - delegates to flext-core utilities
            - Uses t.Tests.Matcher.LengthSpec for type safety
            - Supports exact length or range validation
            - Works with any object that has __len__

            All operations delegate to FlextUtilities.chk() for validation,
            ensuring consistency with flext-core patterns.
            """

            @staticmethod
            def validate(
                value: object,
                spec: int | tuple[int, int],
            ) -> bool:
                """Validate length against spec.

                Uses u.chk() for validation - NO code duplication.
                Supports exact length (int) or range (tuple[int, int]).

                Args:
                    value: Value to check length of (must have __len__)
                    spec: LengthSpec - exact int or (min, max) tuple

                Returns:
                    FlextResult[TEntity]: Result containing created entity or error
                    True if length matches spec, False otherwise

                Examples:
                    u.Tests.Length.validate("hello", 5)           # Exact: True
                    u.Tests.Length.validate([1, 2, 3], (1, 10))  # Range: True
                    u.Tests.Length.validate("hi", 5)              # Exact: False

                """
                if not hasattr(value, "__len__"):
                    return False

                # Type guard: value has __len__ so it's Sized
                # isinstance check with Sized narrows type for len()
                if not isinstance(value, Sized):
                    return False
                actual_len = len(value)

                if isinstance(spec, int):
                    # Delegate to flext-core chk() - zero duplication
                    return FlextUtilities.chk(actual_len, eq=spec)
                min_len, max_len = spec
                # Delegate to flext-core chk() - zero duplication
                return FlextUtilities.chk(actual_len, gte=min_len, lte=max_len)


u = FlextTestsUtilities

__all__ = ["FlextTestsUtilities", "ModelFactory", "u"]
