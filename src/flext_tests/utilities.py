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
from collections.abc import (
    Callable,
    Generator,
    Mapping,
    MutableMapping,
    Sequence,
    Sized,
)
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from re import Pattern
from typing import Protocol, TypeGuard, cast, override

from pydantic import BaseModel, RootModel, TypeAdapter, ValidationError

from flext_core import (
    FlextContext,
    FlextRegistry,
    FlextSettings,
    FlextUtilities,
    r,
)
from flext_tests import c, m, p, t

_PAYLOAD_MAPPING_ADAPTER = TypeAdapter(dict[str, t.Tests.object])
_PAYLOAD_SEQUENCE_ADAPTER: TypeAdapter[list[t.Tests.object]] = TypeAdapter(
    list[t.Tests.object]
)


def _to_scalar(
    value: BaseModel
    | Mapping[str, t.Tests.object]
    | Path
    | Sequence[t.Tests.object]
    | bool
    | bytes
    | datetime
    | float
    | str
    | None,
) -> t.Scalar:
    """Convert a value to ScalarValue for config overrides.

    Args:
        value value to convert

    Returns:
        ScalarValue (str | int | float | bool | datetime | None)

    """
    if isinstance(value, (str, int, float, bool)):
        scalar_value: t.Scalar = value
        return scalar_value
    if isinstance(value, datetime):
        return value
    return str(value)


def _to_payload(
    value: t.Tests.object
    | RootModel[t.Tests.object]
    | set[t.Tests.object]
    | type
    | None,
) -> t.Tests.object:
    """Convert a value to tesobject.

    Args:
        value value to convert

    Returns:
        object suitable for test assertions

    """
    if isinstance(value, RootModel):
        # isinstance erases generic type parameter; cast restores it
        root_model = cast("RootModel[t.Tests.object]", value)
        return _to_payload(root_model.root)
    if value is None or isinstance(
        value, (str, int, float, bool, bytes, datetime, Path, BaseModel)
    ):
        payload_value: t.Tests.object = value
        return payload_value
    if isinstance(value, Mapping):
        try:
            mapping_value = _PAYLOAD_MAPPING_ADAPTER.validate_python(value)
        except ValidationError:
            return {}
        payload_map: dict[str, t.Tests.object] = {}
        for key_raw, item_obj in mapping_value.items():
            payload_map[str(key_raw)] = _to_payload(item_obj)
        return payload_map
    if isinstance(value, (list, tuple, set)):
        try:
            iterable_items: list[t.Tests.object] = (
                _PAYLOAD_SEQUENCE_ADAPTER.validate_python(value)
            )
        except ValidationError:
            return []
        payload_items: list[t.Tests.object] = [
            _to_payload(item_obj) for item_obj in iterable_items
        ]
        return payload_items
    return str(value)


def _to_normalized_value(value: t.Tests.object) -> t.NormalizedValue:
    """Convert _Testobject to pure NormalizedValue (no BaseModel in output)."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value
    if isinstance(value, Path):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    if isinstance(value, BaseModel):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_normalized_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_normalized_value(item) for item in value]
    return str(value)


def _to_config_map_value(value: t.Tests.object) -> t.NormalizedValue | BaseModel:
    """Convert value to NormalizedValue or BaseModel for AccessibleData compatibility."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    if isinstance(value, datetime):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_normalized_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_normalized_value(item) for item in value]
    return str(value)


def _coerce_to_test_object(raw: object) -> t.Tests.object:
    """Coerce an arbitrary object to t.Tests.object."""
    if isinstance(raw, (str, int, float, bool, bytes, BaseModel)):
        return raw
    if raw is None:
        return None
    return str(raw)


def _is_flext_result(
    val: object,
) -> TypeGuard[r[BaseModel]]:
    """TypeGuard: narrow any object to r[BaseModel]."""
    return isinstance(val, r)


def _do_extract_model[T: BaseModel](
    result: (
        BaseModel
        | list[BaseModel]
        | Mapping[str, BaseModel]
        | r[BaseModel]
        | r[list[BaseModel]]
        | r[Mapping[str, BaseModel]]
    ),
    expected: type[T],
) -> T:
    """Extract a BaseModel from various tt.model() return shapes."""
    if isinstance(result, expected):
        return result
    if _is_flext_result(result) and result.is_success:
        payload: BaseModel = result.value
        if isinstance(payload, expected):
            return payload
    if isinstance(result, list):
        for item in result:
            if isinstance(item, expected):
                return item
    if isinstance(result, Mapping) and not isinstance(result, BaseModel):
        for val in result.values():
            if isinstance(val, expected):
                return val
    raise TypeError(f"Expected {expected.__name__}, got {type(result)}")


def _merge_test_dicts(
    base: Mapping[str, t.Tests.object],
    override: Mapping[str, t.Tests.object],
    *,
    strategy: str = "deep",
) -> dict[str, t.Tests.object]:
    """Merge two test dicts via u.merge(), handling payload conversion.

    DRY helper: consolidates the repeated pattern of
    ``u.merge(to_normalized_dict(...), to_normalized_dict(...))``.
    """
    from flext_tests import u as _u  # noqa: PLC0415

    mr = _u.merge(
        {k: _to_normalized_value(v) for k, v in base.items()},
        {k: _to_normalized_value(v) for k, v in override.items()},
        strategy=strategy,
    )
    if mr.is_success:
        return {str(k): _to_payload(v) for k, v in mr.value.items()}
    return dict(base.items())


def _entity_factory_for[T: BaseModel](
    cls: type[T],
) -> _EntityFactory[T]:
    """Build an _EntityFactory lambda for *cls* (DRY helper)."""

    def _factory(*, name: str, value: t.Tests.object, **_kw: t.Tests.object) -> T:
        return cls(name=name, value=value)  # type: ignore[call-arg]

    return _factory  # type: ignore[return-value]


def _value_factory_for[T: BaseModel](
    cls: type[T],
) -> _ValueFactory[T]:
    """Build a _ValueFactory lambda for *cls* (DRY helper)."""

    def _factory(*, data: str, count: int) -> T:
        return cls(data=data, count=count)  # type: ignore[call-arg]

    return _factory  # type: ignore[return-value]


class _EntityFactory[TEntity](Protocol):
    def __call__(self, *, name: str, value: t.Tests.object) -> TEntity: ...


class _ValueFactory[TValue](Protocol):
    def __call__(self, *, data: str, count: int) -> TValue: ...


class FlextTestsUtilities(FlextUtilities):
    """Test utilities for FLEXT ecosystem - extends FlextUtilities.

    Provides essential test helpers that complement FlextUtilities.
    All FlextUtilities functionality is available via inheritance.
    """

    class Validation:
        """Validation helpers for tests - extends FlextUtilities.Validation."""

        @staticmethod
        def validate_pipeline(
            value: str, validators: list[Callable[[str], r[bool]]]
        ) -> r[bool]:
            """Execute validation pipeline with multiple validators.

            Runs validators sequentially. If any validator fails, returns that failure.
            If all validators succeed, returns success.

            Args:
                value: Value to validate
                validators: List of validator functions, each returns r[bool]

            Returns:
                r[bool]: Success if all validators pass, failure from first failure

            """
            for validator in validators:
                try:
                    result = validator(value)
                    if result.is_failure:
                        return result
                    if result.value is False:
                        return r[bool].fail("Validator must return r[bool].ok(True)")
                except TypeError as e:
                    return r[bool].fail(f"Validator failed: {e}")
                except (ValueError, AttributeError, RuntimeError) as e:
                    return r[bool].fail(str(e))
            return r[bool].ok(value=True)

    class Tests:
        """Test-specific utilities namespace.

        All test utilities organized under u.Tests.* pattern.
        """

        @staticmethod
        def to_payload(value: t.Tests.object) -> t.Tests.object:
            """Convert a value to test payload object.

            Delegates to module-level _to_payload for recursive conversion.
            """
            return _to_payload(value)

        @staticmethod
        def to_normalized_value(value: t.Tests.object) -> t.NormalizedValue:
            """Convert test payload value to NormalizedValue.

            Delegates to module-level _to_normalized_value.
            """
            return _to_normalized_value(value)

        @staticmethod
        def to_normalized_dict(
            data: dict[str, t.Tests.object],
        ) -> dict[str, t.NormalizedValue]:
            """Convert test payload dict to NormalizedValue dict for u.merge().

            Uses _to_normalized_value for proper runtime type narrowing.
            """
            return {k: _to_normalized_value(v) for k, v in data.items()}

        @staticmethod
        def coerce_to_test_object(raw: object) -> t.Tests.object:
            """Coerce an arbitrary object to t.Tests.object.

            Delegates to module-level _coerce_to_test_object.
            """
            return _coerce_to_test_object(raw)

        @staticmethod
        def is_flext_result(
            val: object,
        ) -> TypeGuard[r[BaseModel]]:
            """TypeGuard: narrow any object to r[BaseModel].

            Delegates to module-level _is_flext_result.
            """
            return _is_flext_result(val)

        @staticmethod
        def extract_model[T: BaseModel](
            result: (
                BaseModel
                | list[BaseModel]
                | Mapping[str, BaseModel]
                | r[BaseModel]
                | r[list[BaseModel]]
                | r[Mapping[str, BaseModel]]
            ),
            expected: type[T],
        ) -> T:
            """Extract a BaseModel from various tt.model() return shapes.

            Delegates to module-level _do_extract_model.
            """
            return _do_extract_model(result, expected)

        @staticmethod
        def merge_test_dicts(
            base: Mapping[str, t.Tests.object],
            override: Mapping[str, t.Tests.object],
            *,
            strategy: str = "deep",
        ) -> dict[str, t.Tests.object]:
            """Merge two test dicts via u.merge() (DRY helper).

            Delegates to module-level _merge_test_dicts.
            """
            return _merge_test_dicts(base, override, strategy=strategy)

        @staticmethod
        def entity_factory_for[T: BaseModel](
            model_cls: type[T],
        ) -> _EntityFactory[T]:
            """Build an _EntityFactory for the given class (DRY helper).

            Delegates to module-level _entity_factory_for.
            """
            return _entity_factory_for(model_cls)

        @staticmethod
        def value_factory_for[T: BaseModel](
            model_cls: type[T],
        ) -> _ValueFactory[T]:
            """Build a _ValueFactory for the given class (DRY helper).

            Delegates to module-level _value_factory_for.
            """
            return _value_factory_for(model_cls)

        class Result:
            """Result helpers for test assertions."""

            @staticmethod
            def assert_failure[TResult](
                result: r[TResult] | p.Result[TResult],
                expected_error: str | None = None,
            ) -> str:
                """Assert result is failure and return error message.

                Args:
                    result: r or Result protocol to check
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
            def assert_failure_with_error[T](
                result: r[T] | p.Result[T], expected_error: str | None = None
            ) -> None:
                """Assert result is failure and has expected error.

                Args:
                    result: r or Result protocol to check
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

            @staticmethod
            def assert_result_failure_with_error[T](
                result: r[T] | p.Result[T], expected_error: str
            ) -> None:
                """Assert result failure with error (compat alias).

                Args:
                    result: r or Result protocol to check
                    expected_error: Expected error substring

                Raises:
                    AssertionError: If result is not failure or error mismatch

                """
                FlextTestsUtilities.Tests.Result.assert_failure_with_error(
                    result, expected_error
                )

            @staticmethod
            def assert_success[TResult](
                result: r[TResult] | p.Result[TResult], error_msg: str | None = None
            ) -> TResult:
                """Assert result is success and return unwrapped value.

                Args:
                    result: r or Result protocol to check
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
                return result.value

            @staticmethod
            def assert_success_with_value[T](
                result: r[T] | p.Result[T], expected_value: T
            ) -> None:
                """Assert result is success and has expected value.

                Args:
                    result: r or Result protocol to check
                    expected_value: Expected value

                Raises:
                    AssertionError: If result is failure or value doesn't match

                """
                if not result.is_success:
                    msg = f"Expected success, got failure: {result.error}"
                    raise AssertionError(msg)
                assert result.value == expected_value

            @staticmethod
            def create_failure_result(error: str) -> r[str]:
                """Create a failure result with the given error.

                Args:
                    error: Error message for the failure result

                Returns:
                    r[TEntity]: Result containing created entity or error
                    r with failure and error message

                """
                return r[str].fail(error)

            @staticmethod
            def create_success_result[T](value: T) -> r[T]:
                """Create a success result with the given value.

                Args:
                    value: Value for the success result

                Returns:
                    r[TEntity]: Result containing created entity or error
                    r with success and value

                """
                return r[T].ok(value)

        class TestContext:
            """Context managers for tests."""

            @staticmethod
            @contextmanager
            def temporary_attribute(
                target: BaseModel, attribute: str, value: t.Tests.object
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
                original_value: t.Tests.object | None = None
                if attribute_existed:
                    original_value = target.__getattribute__(attribute)
                object.__setattr__(target, attribute, value)
                try:
                    yield
                finally:
                    if attribute_existed:
                        object.__setattr__(target, attribute, original_value)
                    else:
                        object.__delattr__(target, attribute)

        class Factory:
            """Factory helpers for test data creation."""

            @staticmethod
            def add_operation(a: t.Tests.object, b: t.Tests.object) -> t.Tests.object:
                """Execute add operation for numeric or string values.

                Args:
                    a: First operand (numeric or string)
                    b: Second operand (numeric or string)

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Sum if both numeric, concatenation otherwise.

                """
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return a + b
                return str(a) + str(b)

            @staticmethod
            def create_error_operation(
                error_message: str,
            ) -> Callable[[], t.Tests.object]:
                """Create callable that raises ValueError.

                Args:
                    error_message: Error message for ValueError

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Callable that raises ValueError when called.

                """

                def error_op() -> t.Tests.object:
                    raise ValueError(error_message)

                return error_op

            @staticmethod
            def create_result[T](
                value: T | None = None, *, error: str | None = None
            ) -> r[T]:
                """Create r for tests.

                Args:
                    value: Value for success result
                    error: Error message for failure result

                Returns:
                    r[TEntity]: Result containing created entity or error
                    r with value or error

                """
                if error is not None:
                    return r[T].fail(error)
                if value is not None:
                    return r[T].ok(value)
                return r[T].fail("No value or error provided")

            @staticmethod
            def create_test_data(
                **kwargs: t.Tests.object,
            ) -> MutableMapping[str, t.Tests.object]:
                """Create test data dictionary.

                Args:
                    **kwargs: Key-value pairs for test data

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Configuration dictionary

                """
                return dict(kwargs)

            @staticmethod
            def execute_complex_service(
                validation_result: r[bool],
            ) -> r[t.Tests.object]:
                """Execute complex service operation.

                Args:
                    validation_result: Result of business rules validation

                Returns:
                    r[TEntity]: Result containing created entity or error
                    r with service data or error.

                """
                if validation_result.is_failure:
                    return r[t.Tests.object].fail(
                        validation_result.error or "Validation failed"
                    )
                result_data: t.Tests.object = {"result": "success"}
                return r[t.Tests.object].ok(result_data)

            @staticmethod
            def execute_default_service(service_type: str) -> r[t.Tests.object]:
                """Execute default service operation.

                Args:
                    service_type: Type of service

                Returns:
                    r[TEntity]: Result containing created entity or error
                    r with service type data.

                """
                service_data: t.Tests.object = {"service_type": service_type}
                return r[t.Tests.object].ok(service_data)

            @staticmethod
            def execute_user_service(
                overrides: Mapping[str, t.Tests.object],
            ) -> r[t.Tests.object]:
                """Execute user service operation.

                Args:
                    overrides: Service configuration overrides

                Returns:
                    r[TEntity]: Result containing created entity or error
                    r with user data.

                """
                user_id = "default_123" if overrides.get("default") else "test_123"
                user_data: t.Tests.object = {
                    "user_id": user_id,
                    "email": "test@example.com",
                }
                return r[t.Tests.object].ok(user_data)

            @staticmethod
            def format_operation(name: str, value: int = 10) -> str:
                """Execute format operation returning formatted string.

                Args:
                    name: Name part of format
                    value: Value part of format (default: 10)

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Formatted string "name: value".

                """
                return f"{name}: {value}"

            @staticmethod
            def generate_id() -> str:
                """Generate unique ID using u.generate().

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Generated UUID string.

                """
                return FlextUtilities.generate()

            @staticmethod
            def generate_short_id(length: int = 8) -> str:
                """Generate short unique ID using u.generate('ulid', length=...).

                Args:
                    length: Length of ID (default: 8)

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Generated short ID string.

                """
                return FlextUtilities.generate("ulid", length=length)

            @staticmethod
            def simple_operation() -> t.Tests.object:
                """Execute simple operation returning success message.

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Success message string from constants.

                """
                return c.Tests.Factory.SUCCESS_MESSAGE

        class ResultHelpers:
            """Result helpers — delegates to Tests.Result (compat).

            .. deprecated::
                Use ``u.Tests.Result`` directly. This namespace exists only
                for backward-compatibility.
            """

            @staticmethod
            def assert_failure_with_error[T](
                result: r[T] | p.Result[T], expected_error: str | None = None
            ) -> None:
                """Delegate to Result.assert_failure_with_error."""
                FlextTestsUtilities.Tests.Result.assert_failure_with_error(
                    result, expected_error
                )

            @staticmethod
            def assert_result_failure_with_error[T](
                result: r[T] | p.Result[T], expected_error: str
            ) -> None:
                """Delegate to Result.assert_result_failure_with_error."""
                FlextTestsUtilities.Tests.Result.assert_result_failure_with_error(
                    result, expected_error
                )

            @staticmethod
            def assert_success_with_value[T](
                result: r[T] | p.Result[T], expected_value: T
            ) -> None:
                """Delegate to Result.assert_success_with_value."""
                FlextTestsUtilities.Tests.Result.assert_success_with_value(
                    result, expected_value
                )

            @staticmethod
            def create_failure_result(error: str) -> r[str]:
                """Delegate to Result.create_failure_result."""
                return FlextTestsUtilities.Tests.Result.create_failure_result(error)

            @staticmethod
            def create_success_result[T](value: T) -> r[T]:
                """Delegate to Result.create_success_result."""
                return FlextTestsUtilities.Tests.Result.create_success_result(value)

        class GenericHelpers:
            """Generic helpers for test data creation."""

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
                if first_failure_index is not None:
                    actual_first_failure = next(
                        (i for i, res in enumerate(results) if res.is_failure), None
                    )
                    assert actual_first_failure == first_failure_index, (
                        f"Expected first failure at index {first_failure_index}, got {actual_first_failure}"
                    )
                elif failures == 0:
                    actual_first_failure = next(
                        (i for i, res in enumerate(results) if res.is_failure), None
                    )
                    assert actual_first_failure is None, (
                        f"Expected no failures but found first failure at index {actual_first_failure}"
                    )

            @staticmethod
            def create_parametrized_cases(
                success_values: list[t.Tests.object],
                failure_errors: list[str] | None = None,
                *,
                error_codes: list[str | None] | None = None,
            ) -> list[
                tuple[
                    r[t.Tests.object],
                    bool,
                    t.Tests.object | None,
                    str | None,
                ]
            ]:
                """Create parametrized test cases from values and errors.

                Args:
                    success_values: List of values for success results
                    failure_errors: Optional list of error messages for failure results
                    error_codes: Optional list of error codes for failure results

                Returns:
                    r[TEntity]: Result containing created entity or error
                    List of tuples (result, is_success, value, error)

                """
                cases: list[
                    tuple[
                        r[t.Tests.object],
                        bool,
                        t.Tests.object | None,
                        str | None,
                    ]
                ] = []
                for value in success_values:
                    result = r[t.Tests.object].ok(value)
                    cases.append((result, True, value, None))
                if failure_errors:
                    codes = error_codes or [None] * len(failure_errors)
                    for i, error in enumerate(failure_errors):
                        error_code = codes[i] if i < len(codes) else None
                        result = r[t.Tests.object].fail(error, error_code=error_code)
                        cases.append((result, False, None, error))
                return cases

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
                    r[TEntity]: Result containing created entity or error
                    r with success or failure

                """
                if value is None:
                    if default_on_none is not None:
                        return r[T].ok(default_on_none)
                    return r[T].fail(error_on_none)
                return r[T].ok(value)

            @staticmethod
            def validate_model_attributes(
                model: BaseModel,
                required_attrs: list[str],
                optional_attrs: list[str] | None = None,
            ) -> r[bool]:
                """Validate model has required attributes.

                Args:
                    model: Model object to validate
                    required_attrs: List of required attribute names
                    optional_attrs: Optional list of optional attribute names

                Returns:
                    r[TEntity]: Result containing created entity or error
                    r with True if all required attrs exist

                """
                missing = [attr for attr in required_attrs if not hasattr(model, attr)]
                if missing:
                    return r[bool].fail(f"Missing required attributes: {missing}")
                return r[bool].ok(value=True)

        class RegistryHelpers:
            """Registry testing helpers - use FlextRegistry directly when possible."""

            @staticmethod
            def create_test_registry() -> FlextRegistry:
                """Create a test registry instance.

                Returns:
                    r[TEntity]: Result containing created entity or error
                    New FlextRegistry instance

                """
                return FlextRegistry()

        class ConfigHelpers:
            """Config testing helpers - use FlextSettings directly when possible."""

            @staticmethod
            def assert_config_fields(
                config: FlextSettings, expected_fields: m.ConfigMap
            ) -> None:
                """Assert config has expected field values.

                Args:
                    config: Config instance to check
                    expected_fields: Expected field values

                Raises:
                    AssertionError: If fields don't match

                """
                for key, expected_value in expected_fields.items():
                    actual_value = (
                        getattr(config, key) if hasattr(config, key) else None
                    )
                    msg = f"Config {key}: expected {expected_value}, got {actual_value}"
                    assert actual_value == expected_value, msg

            @staticmethod
            def create_test_config(**kwargs: t.Tests.object) -> FlextSettings:
                """Create a test config instance.

                Args:
                    **kwargs: Config field values

                Returns:
                    New FlextSettings instance

                """
                scalar_overrides: dict[str, t.Scalar] = {
                    str(key): _to_scalar(value) for key, value in kwargs.items()
                }
                return FlextSettings.get_global(overrides=scalar_overrides)

            @staticmethod
            @contextmanager
            def env_vars_context(
                env_vars: Mapping[str, t.Tests.object],
                vars_to_clear: list[str] | None = None,
            ) -> Generator[None]:
                """Context manager for temporary environment variable changes.

                Args:
                    env_vars: Environment variables to set
                    vars_to_clear: Variables to clear on entry

                Yields:
                    None

                """
                original_values: MutableMapping[str, str | None] = {}
                if vars_to_clear:
                    for var in vars_to_clear:
                        original_values[var] = os.environ.get(var)
                        if var in os.environ:
                            del os.environ[var]
                for key, value in env_vars.items():
                    if key not in original_values:
                        original_values[key] = os.environ.get(key)
                    os.environ[key] = str(value)
                try:
                    yield
                finally:
                    for key, original in original_values.items():
                        if original is None:
                            if key in os.environ:
                                del os.environ[key]
                        else:
                            os.environ[key] = original

        class ContextHelpers:
            """Helpers for context testing."""

            @staticmethod
            def assert_context_get_success(
                context: FlextContext, key: str, expected_value: t.Tests.object
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
                    f"Expected success for key '{key}', got: {result.error!r}"
                )
                actual = _to_payload(result.value)
                assert actual == expected_value, (
                    f"Expected {expected_value!r} for key '{key}', got {result.value!r}"
                )

            @staticmethod
            def clear_context() -> None:
                """Clear the global context."""
                FlextContext.Utilities.clear_context()

            @staticmethod
            def create_test_context() -> FlextContext:
                """Create a test context instance.

                Returns:
                    r[TEntity]: Result containing created entity or error
                    New FlextContext instance

                """
                return FlextContext.create()

        class ContainerHelpers:
            """Helpers for container testing."""

            @staticmethod
            def create_counting_factory[TFactory](
                return_value: TFactory,
            ) -> tuple[Callable[[], TFactory], Callable[[], int]]:
                """Create a factory that counts invocations.

                Args:
                    return_value: Value to return from factory

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Tuple of (factory function, count getter)

                """
                count = [0]

                def factory() -> TFactory:
                    count[0] += 1
                    return return_value

                def get_count() -> int:
                    return count[0]

                return (factory, get_count)

            @staticmethod
            def create_factory[TFactory](
                return_value: TFactory,
            ) -> Callable[[], TFactory]:
                """Create a factory function that returns a fixed value.

                Args:
                    return_value: Value to return from factory

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Factory function

                """

                def factory() -> TFactory:
                    return return_value

                return factory

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
                metadata: m.Metadata | None = None,
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
                    r[TEntity]: Result containing created entity or error
                    Handler configuration model

                """
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

        class ParserHelpers:
            """Helpers for parser testing."""

            @staticmethod
            def execute_and_assert_parser_result(
                operation: Callable[[], r[t.Tests.object]],
                expected_value: t.Tests.object | None = None,
                expected_error: str | None = None,
                description: str = "",
            ) -> None:
                """Execute parser operation and assert result.

                Args:
                    operation: Callable that returns a r
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
            def create_batch_operation_test_cases(
                operation: str,
                descriptions: list[str],
                input_data_list: list[Mapping[str, t.Tests.object]],
                expected_results: list[t.Tests.object],
                **common_kwargs: t.Tests.object,
            ) -> list[MutableMapping[str, t.Tests.object]]:
                """Create batch test cases for operation testing.

                Args:
                    operation: Operation name
                    descriptions: List of descriptions
                    input_data_list: List of input data dicts
                    expected_results: List of expected results
                    **common_kwargs: Common parameters for all cases

                Returns:
                    r[TEntity]: Result containing created entity or error
                    List of test case dictionaries

                """
                cases: list[MutableMapping[str, t.Tests.object]] = []
                for desc, data, expected in zip(
                    descriptions, input_data_list, expected_results, strict=True
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
            def create_operation_test_case(
                operation: str,
                description: str,
                input_data: Mapping[str, t.Tests.object],
                expected_result: t.Tests.object,
                **kwargs: t.Tests.object,
            ) -> MutableMapping[str, t.Tests.object]:
                """Create a test case dict for operation testing.

                Args:
                    operation: Operation name
                    description: Test case description
                    input_data: Input data for the operation
                    expected_result: Expected result or type
                    **kwargs: Additional test case parameters

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Test case dictionary

                """
                result: MutableMapping[str, t.Tests.object] = {
                    "operation": operation,
                    "description": description,
                    "input_data": input_data,
                    "expected_result": expected_result,
                }
                result.update(kwargs)
                return result

            @staticmethod
            def execute_and_assert_operation_result(
                operation: Callable[[], t.Tests.object],
                test_case: Mapping[str, t.Tests.object],
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
                assert result == expected, f"Expected {expected}, got {result}"

        class DomainHelpers:
            """Helpers for domain model testing."""

            @staticmethod
            def create_test_entities_batch[TEntity](
                names: list[str],
                values: list[t.Tests.object],
                entity_class: _EntityFactory[TEntity],
                remove_ids: list[bool] | None = None,
            ) -> r[list[TEntity]]:
                """Create batch of test entities.

                Args:
                    names: List of entity names
                    values: List of entity values
                    entity_class: Entity class to instantiate
                    remove_ids: List of booleans for ID removal

                Returns:
                    r[list[TEntity]]: Result containing list of entities or error

                """
                ids_removal = remove_ids or [False] * len(names)
                entities: list[TEntity] = []
                dh = FlextTestsUtilities.Tests.DomainHelpers
                for name, value, remove_id in zip(
                    names, values, ids_removal, strict=True
                ):
                    try:
                        entity = dh.create_test_entity_instance(
                            name=name,
                            value=value,
                            entity_class=entity_class,
                            remove_id=remove_id,
                        )
                        entities.append(entity)
                    except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                        return r[list[TEntity]].fail(
                            f"Failed to create entity {name}: {e}"
                        )
                return r[list[TEntity]].ok(entities)

            @staticmethod
            def create_test_entity_instance[TEntity](
                name: str,
                value: t.Tests.object,
                entity_class: _EntityFactory[TEntity],
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
                    attr_name = "unique_id"
                    delattr(entity, attr_name)
                return entity

            @staticmethod
            def create_test_value_object_instance[TValue](
                data: str, count: int, value_class: _ValueFactory[TValue]
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
                value_class: _ValueFactory[TValue],
            ) -> list[TValue]:
                """Create batch of test value objects.

                Args:
                    data_list: List of data values
                    count_list: List of count values
                    value_class: Value object class to instantiate

                Returns:
                    r[TEntity]: Result containing created entity or error
                    List of created value objects

                """
                return [
                    FlextTestsUtilities.Tests.DomainHelpers.create_test_value_object_instance(
                        data=data, count=count, value_class=value_class
                    )
                    for data, count in zip(data_list, count_list, strict=True)
                ]

            @staticmethod
            def execute_domain_operation(
                operation: str,
                input_data: Mapping[str, t.Tests.object],
                **kwargs: t.Tests.object,
            ) -> t.Tests.object:
                """Execute a domain utility operation.

                Args:
                    operation: Operation name from FlextUtilities
                    input_data: Input data dictionary
                    **kwargs: Additional arguments

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Operation result (type depends on operation)

                """
                if not hasattr(FlextUtilities, operation):
                    msg = f"Unknown operation: {operation}"
                    raise ValueError(msg)
                op_method = getattr(FlextUtilities, operation)
                if not callable(op_method):
                    msg = f"Unknown operation: {operation}"
                    raise ValueError(msg)
                all_args = {**input_data, **kwargs}
                result = op_method(**all_args)
                if isinstance(result, RootModel):
                    root_result = cast("RootModel[t.Tests.object]", result)
                    return _to_payload(root_result)
                if isinstance(result, (BaseModel, Path)):
                    return _to_payload(result)
                if isinstance(result, (str, int, float, bool, bytes, datetime)):
                    payload_scalar: t.Tests.object = result
                    return _to_payload(payload_scalar)
                if isinstance(result, type):
                    return _to_payload(result)
                if result is None:
                    return _to_payload(result)
                return _to_payload(str(result))

        class ExceptionHelpers:
            """Helpers for exception testing."""

            @staticmethod
            def create_metadata_object(
                attributes: Mapping[str, t.Tests.object],
            ) -> MutableMapping[str, t.Tests.object]:
                """Create a metadata object for exceptions.

                Args:
                    attributes: Metadata attributes

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Metadata object with attributes as dict

                """
                return {"attributes": attributes, **attributes}

        class BadObjects:
            """Factory for objects that cause errors during testing."""

            class BadModelDump:
                """Object with model_dump that raises."""

                model_dump: Callable[[], MutableMapping[str, t.Tests.object]] = (
                    staticmethod(
                        lambda: (_ for _ in ()).throw(RuntimeError("Bad model_dump"))
                    )
                )

            class BadConfig:
                """Config object that raises on attribute access."""

                @override
                def __getattribute__(self, name: str) -> t.Tests.object:
                    """Raise error on attribute access - test helper for error testing."""
                    if name.startswith("__") and name.endswith("__"):
                        result: t.Tests.object = super().__getattribute__(name)
                        return result
                    msg = f"Bad config: {name}"
                    raise AttributeError(msg)

            class BadConfigTypeError:
                """Config object that raises TypeError on attribute access."""

                @override
                def __getattribute__(self, name: str) -> t.Tests.object:
                    """Raise TypeError on attribute access - test helper for error testing."""
                    if name.startswith("__") and name.endswith("__"):
                        result: t.Tests.object = super().__getattribute__(name)
                        return result
                    msg = f"Bad config type: {name}"
                    raise TypeError(msg)

        class ConstantsHelpers:
            """Helpers for testing FlextConstants."""

            @staticmethod
            def compile_pattern(pattern_attr: str) -> Pattern[str]:
                """Compile a regex pattern from FlextConstants.

                Args:
                    pattern_attr: Attribute name like "Patterns.EMAIL_REGEX"

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Compiled regex pattern

                """
                parts = pattern_attr.split(".")
                current = c
                for part in parts:
                    current = getattr(current, part)
                pattern_str = str(current)
                return re.compile(pattern_str, re.IGNORECASE)

            @staticmethod
            def get_constant_by_path(path: str) -> t.Tests.object:
                """Get a constant value by dot-separated path.

                Args:
                    path: Dot-separated path like "Utilities.MAX_TIMEOUT_SECONDS"

                Returns:
                    r[TEntity]: Result containing created entity or error
                    The constant value at the given path

                """
                parts = path.split(".")
                current = c
                for part in parts:
                    current = getattr(current, part)
                return _to_payload(current)

        class Assertions:
            """Common assertion helpers — delegates to Tests.Result (compat).

            .. deprecated::
                Use ``u.Tests.Result`` directly for assert_result_failure /
                assert_result_success. Only ``assert_result_matches_expected``
                is unique to this namespace.
            """

            @staticmethod
            def assert_result_failure[TResult](
                result: r[TResult], expected_error: str | None = None
            ) -> str:
                """Delegate to Result.assert_failure."""
                return FlextTestsUtilities.Tests.Result.assert_failure(
                    result, expected_error
                )

            @staticmethod
            def assert_result_matches_expected(
                result: t.Tests.object,
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
                    f"Expected {expected_type.__name__}, got {type(result).__name__}{(f' for {description}' if description else '')}"
                )

            @staticmethod
            def assert_result_success[TResult](result: r[TResult]) -> TResult:
                """Delegate to Result.assert_success."""
                return FlextTestsUtilities.Tests.Result.assert_success(result)

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
                    r[TEntity]: Result containing created entity or error
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
                content: str | bytes | Mapping[str, t.Tests.object] | list[list[str]],
                name: str,
                fmt: str,
            ) -> str:
                """Detect file format from content type or filename.

                Args:
                    content: File content (type determines format)
                    name: Filename (extension hints format)
                    fmt: Explicit format override ("auto" for detection)

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Detected format string

                """
                if fmt != c.Tests.Files.Format.AUTO:
                    return fmt
                if isinstance(content, bytes):
                    return c.Tests.Files.Format.BIN
                if isinstance(content, Mapping):
                    ext = Path(name).suffix.lower()
                    if ext in {".yaml", ".yml"}:
                        return c.Tests.Files.Format.YAML
                    return c.Tests.Files.Format.JSON
                if isinstance(content, list):
                    return c.Tests.Files.Format.CSV
                return c.Tests.Files.get_format(Path(name).suffix)

            @staticmethod
            def detect_format_from_path(path: Path, fmt: str) -> str:
                """Detect format from file path.

                Args:
                    path: File path
                    fmt: Explicit format override ("auto" for detection)

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Detected format string

                """
                if fmt != c.Tests.Files.Format.AUTO:
                    return fmt
                return c.Tests.Files.get_format(path.suffix)

            @staticmethod
            def format_size(size: int) -> str:
                """Format size in human-readable format.

                Delegates to constants.Files.format_size for consistency.

                Args:
                    size: Size in bytes

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Human-readable size string like "1.2 KB"

                """
                return c.Tests.Files.format_size(size)

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
                    r[TEntity]: Result containing created entity or error
                    List of rows (each row is list of strings)

                """
                delim = delimiter or c.Tests.Files.DEFAULT_CSV_DELIMITER
                enc = encoding or c.Tests.Files.DEFAULT_ENCODING
                with path.open(newline="", encoding=enc) as f:
                    reader = csv.reader(f, delimiter=delim)
                    rows = list(reader)
                    if has_headers and rows:
                        return rows[1:]
                    return rows

            @staticmethod
            def write_csv(
                path: Path,
                content: str | bytes | Mapping[str, t.Tests.object] | list[list[str]],
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
                            writer.writerow(row)

        class Validator:
            """Validator utilities for architecture validation (tv.* methods).

            Provides reusable helper functions for validators. All validators
            should use these instead of implementing their own versions.
            """

            @staticmethod
            def create_violation(
                file_path: Path,
                line_number: int,
                rule_id: str,
                lines: list[str],
                extra_desc: str = "",
            ) -> m.Tests.Violation:
                """Create a violation model using c.Tests.Validator.Rules.

                Args:
                    file_path: Path to file with violation
                    line_number: Line number of violation (1-indexed)
                    rule_id: Rule identifier (e.g., "IMPORT-001")
                    lines: File content as list of lines
                    extra_desc: Optional extra description

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Violation model instance

                """
                severity, desc = c.Tests.Validator.Rules.get(rule_id)
                description = f"{desc}: {extra_desc}" if extra_desc else desc
                line = lines[line_number - 1] if line_number <= len(lines) else ""
                return m.Tests.Violation(
                    file_path=file_path,
                    line_number=line_number,
                    rule_id=rule_id,
                    severity=severity,
                    description=description,
                    code_snippet=line.strip(),
                )

            @staticmethod
            def find_line_number(lines: list[str], pattern: str) -> int:
                """Find line number containing pattern.

                Args:
                    lines: File content as list of lines
                    pattern: Pattern to search for

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Line number (1-indexed) or 1 if not found

                """
                for i, line in enumerate(lines, start=1):
                    if pattern in line:
                        return i
                return 1

            @staticmethod
            def get_exception_names(exc_type: ast.expr) -> set[str]:
                """Extract exception names from exception type AST node.

                Args:
                    exc_type: Exception type AST node

                Returns:
                    r[TEntity]: Result containing created entity or error
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
            def get_parent(tree: ast.AST, node: ast.AST) -> ast.AST | None:
                """Get parent node of an AST node.

                Args:
                    tree: AST tree root
                    node: Node to find parent of

                Returns:
                    r[TEntity]: Result containing created entity or error
                    Parent node or None if not found

                """
                for parent in ast.walk(tree):
                    for child in ast.iter_child_nodes(parent):
                        if child is node:
                            return parent
                return None

            @staticmethod
            def is_any_type(node: ast.expr) -> bool:
                """Check if an annotation node represents the typing.Any type.

                Args:
                    node: AST annotation node

                Returns:
                    r[TEntity]: Result containing created entity or error
                    True if node represents typing.Any type annotation

                """
                return (
                    (isinstance(node, ast.Name) and node.id == "Any")
                    or (isinstance(node, ast.Attribute) and node.attr == "Any")
                    or (isinstance(node, ast.Constant) and node.value == "Any")
                )

            @staticmethod
            def is_approved(
                rule_id: str, file_path: Path, approved: Mapping[str, list[str]]
            ) -> bool:
                """Check if file is approved for this rule.

                Args:
                    rule_id: Rule identifier (e.g., "IMPORT-001")
                    file_path: Path to file being checked
                    approved: Dict mapping rule IDs to list of approved file patterns

                Returns:
                    r[TEntity]: Result containing created entity or error
                    True if file matches any approved pattern for this rule

                """
                patterns = approved.get(rule_id, [])
                file_str = str(file_path)
                return any(re.search(pattern, file_str) for pattern in patterns)

            @staticmethod
            def is_only_pass(body: list[ast.stmt]) -> bool:
                """Check if exception handler body contains only pass or ellipsis.

                Used by BYPASS-003 to detect exception swallowing patterns.

                Args:
                    body: AST statement list (exception handler body)

                Returns:
                    r[TEntity]: Result containing created entity or error
                    True if body contains only pass or ellipsis (...)

                """
                if len(body) == 1:
                    stmt = body[0]
                    if isinstance(stmt, ast.Pass):
                        return True
                    if (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and (stmt.value.value is ...)
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
                    r[TEntity]: Result containing created entity or error
                    True if pattern appears in real code comment (after #),
                    not inside a string literal (single/double/triple quoted)

                """
                match = pattern.search(line)
                if not match:
                    return False
                pos = match.start()
                in_single = False
                in_double = False
                in_triple_single = False
                in_triple_double = False
                i = 0
                while i < pos:
                    if (
                        line[i : i + 3] == '"""'
                        and (not in_single)
                        and (not in_triple_single)
                    ):
                        in_triple_double = not in_triple_double
                        i += 3
                        continue
                    if (
                        line[i : i + 3] == "'''"
                        and (not in_double)
                        and (not in_triple_double)
                    ):
                        in_triple_single = not in_triple_single
                        i += 3
                        continue
                    if (
                        line[i] == '"'
                        and (not in_single)
                        and (not in_triple_single)
                        and (not in_triple_double)
                    ):
                        in_double = not in_double
                    elif (
                        line[i] == "'"
                        and (not in_double)
                        and (not in_triple_single)
                        and (not in_triple_double)
                    ):
                        in_single = not in_single
                    i += 1
                return not (
                    in_single or in_double or in_triple_single or in_triple_double
                )

        class DeepMatch:
            """Deep structural matching utilities - delegates to u.extract().

            Follows FLEXT patterns:
            - Zero code duplication - delegates to flext-core utilities
            - Uses t.Tests.Matcher.DeepSpec for type safety
            - Returns m.Tests.DeepMatchResult for structured results
            - Supports unlimited nesting depth via dot notation

            All operations delegate to FlextUtilities.extract() for
            path extraction, ensuring consistency with flext-core patterns.
            """

            @staticmethod
            def match(
                obj: BaseModel | Mapping[str, t.Tests.object],
                spec: Mapping[
                    str,
                    t.Tests.object | Callable[[t.Tests.object], bool],
                ],
                *,
                path_sep: str = ".",
            ) -> m.Tests.DeepMatchResult:
                """Match object against deep specification.

                Uses u.extract() for path extraction - NO code duplication.
                Supports unlimited nesting depth via dot notation paths.

                Args:
                    obj: Object to match against (dict or Pydantic model)
                    spec: DeepSpec mapping of path -> expected value or predicate
                    path_sep: Path separator (default: ".")

                Returns:
                    r[TEntity]: Result containing created entity or error
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
                source_obj: dict[str, t.NormalizedValue | BaseModel]
                if isinstance(obj, BaseModel):
                    dumped = obj.model_dump(mode="python")
                    source_obj = {
                        str(key): _to_config_map_value(_to_payload(value))
                        for key, value in dumped.items()
                    }
                else:
                    source_obj = {
                        str(key): _to_config_map_value(value)
                        for key, value in obj.items()
                    }
                for path, expected in spec.items():
                    result = FlextUtilities.extract(
                        source_obj, path, separator=path_sep
                    )
                    if result.is_failure:
                        return m.Tests.DeepMatchResult(
                            path=path,
                            expected=expected,
                            actual=None,
                            matched=False,
                            reason=f"Path not found: {path}",
                        )
                    actual = result.value
                    if callable(expected):
                        actual_payload = _to_payload(actual)
                        if not expected(actual_payload):
                            return m.Tests.DeepMatchResult(
                                path=path,
                                expected="<predicate>",
                                actual=actual_payload,
                                matched=False,
                                reason="Predicate failed",
                            )
                    elif actual != expected:
                        actual_payload = _to_payload(actual)
                        return m.Tests.DeepMatchResult(
                            path=path,
                            expected=expected,
                            actual=actual_payload,
                            matched=False,
                            reason="Value mismatch",
                        )
                return m.Tests.DeepMatchResult(
                    path="",
                    expected=_to_payload(obj),
                    actual=_to_payload(obj),
                    matched=True,
                    reason="",
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
            def validate(value: t.Tests.object, spec: int | tuple[int, int]) -> bool:
                """Validate length against spec.

                Uses u.chk() for validation - NO code duplication.
                Supports exact length (int) or range (tuple[int, int]).

                Args:
                    value: Value to check length of (must have __len__)
                    spec: LengthSpec - exact int or (min, max) tuple

                Returns:
                    r[TEntity]: Result containing created entity or error
                    True if length matches spec, False otherwise

                Examples:
                    u.Tests.Length.validate("hello", 5)           # Exact: True
                    u.Tests.Length.validate([1, 2, 3], (1, 10))  # Range: True
                    u.Tests.Length.validate("hi", 5)              # Exact: False

                """
                try:
                    if not isinstance(value, Sized):
                        return False
                    actual_len = len(value)
                except TypeError:
                    return False
                if isinstance(spec, int):
                    return FlextUtilities.chk(actual_len, eq=spec)
                min_len, max_len = spec
                return FlextUtilities.chk(actual_len, gte=min_len, lte=max_len)


u = FlextTestsUtilities
__all__ = ["FlextTestsUtilities", "u"]
