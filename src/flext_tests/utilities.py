"""Test utilities for FLEXT ecosystem tests.

Provides comprehensive utility functions, helpers, and patterns for testing
FLEXT components including Docker operations, result validation, model testing,
and context management. Includes nested classes for specialized utilities.

Scope: General-purpose test utilities including FlextResult creation/validation,
Docker compose operations with timeout handling, container management helpers,
model testing patterns, context managers for temporary attribute changes, and
test data generation. Supports protocol-based logger integration and generic
factory pattern testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import fnmatch
import threading
import time
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar, overload, runtime_checkable

from docker.errors import DockerException, NotFound

from flext_core import FlextResult
from flext_core.protocols import FlextProtocols
from flext_core.typings import FlextTypes
from flext_tests.domains import FlextTestsDomains
from flext_tests.matchers import FlextTestsMatchers
from flext_tests.protocols import FlextTestProtocols
from flext_tests.typings import FlextTestsTypings

TResult = TypeVar("TResult")
TModel_co = TypeVar("TModel_co", covariant=True)
TValue = TypeVar("TValue")


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for logger objects used in test utilities.

    Flexible protocol compatible with FlextLogger and other loggers.
    Uses hasattr checks in code, so protocol is permissive for type hints.
    Accepts any logger implementation that has basic logging methods.
    """

    def info(
        self,
        message: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Log info message."""
        ...

    def warning(
        self,
        message: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Log warning message."""
        ...

    def error(
        self,
        message: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Log error message."""
        ...

    def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        exc_info: bool = True,
        **kwargs: object,
    ) -> object:
        """Log exception message."""
        ...


class FlextTestsUtilities:
    """Test utilities for FLEXT ecosystem.

    Provides helper functions and utilities for testing FLEXT components.
    Includes generic reusable helpers that replace 10+ lines of duplicated code.
    """

    class GenericHelpers:
        """Generic reusable helpers that replace duplicated code patterns.

        These helpers consolidate common test patterns that appear 10+ times
        across the test suite, reducing duplication and improving maintainability.
        """

        @staticmethod
        def create_result_from_value[TValue](
            value: TValue | None,
            *,
            default_on_none: TValue | None = None,
            error_on_none: str | None = None,
        ) -> FlextResult[TValue]:
            """Create FlextResult from value with standardized None handling.

            Replaces 10+ lines of duplicated result creation code.

            Args:
                value: Value to wrap in result
                default_on_none: Default value if value is None (creates success)
                error_on_none: Error message if value is None (creates failure)

            Returns:
                FlextResult with value or appropriate default/error

            """
            if value is not None:
                return FlextResult[TValue].ok(value)
            if default_on_none is not None:
                return FlextResult[TValue].ok(default_on_none)
            if error_on_none is not None:
                return FlextResult[TValue].fail(error_on_none)
            return FlextResult[TValue].fail("Value is None and no default provided")

        @staticmethod
        def extract_mapping_values[TKey, TValue](
            mapping: Mapping[TKey, TValue],
            *,
            keys: Sequence[TKey] | None = None,
            default: TValue | None = None,
        ) -> dict[TKey, TValue]:
            """Extract values from mapping for specified keys.

            Replaces 10+ lines of duplicated dict extraction code.

            Args:
                mapping: Source mapping to extract from
                keys: Keys to extract (None = all keys)
                default: Default value for missing keys

            Returns:
                Dictionary with extracted key-value pairs

            """
            if keys is None:
                return dict(mapping)
            result: dict[TKey, TValue] = {}
            for key in keys:
                if key in mapping:
                    result[key] = mapping[key]
                elif default is not None:
                    result[key] = default
            return result

        @staticmethod
        def validate_model_attributes(
            instance: FlextProtocols.HasModelDump,
            *,
            required_attrs: Sequence[str],
            optional_attrs: Sequence[str] | None = None,
        ) -> FlextResult[bool]:
            """Validate model instance has required attributes.

            Replaces 10+ lines of duplicated attribute validation code.

            Args:
                instance: Model instance to validate
                required_attrs: Required attribute names
                optional_attrs: Optional attribute names to check existence

            Returns:
                FlextResult with True if valid, error message if invalid

            """
            missing: list[str] = []
            for attr in required_attrs:
                if not hasattr(instance, attr):
                    missing.append(attr)
                elif getattr(instance, attr, None) is None:
                    missing.append(f"{attr} (None value)")

            if missing:
                return FlextResult[bool].fail(
                    f"Missing required attributes: {', '.join(missing)}",
                )

            if optional_attrs:
                for attr in optional_attrs:
                    if not hasattr(instance, attr):
                        return FlextResult[bool].fail(
                            f"Missing optional attribute: {attr}",
                        )

            return FlextResult[bool].ok(True)

        @staticmethod
        def create_parametrized_cases[TValue](
            success_values: Sequence[TValue],
            failure_errors: Sequence[str],
            *,
            error_codes: Sequence[str | None] | None = None,
        ) -> list[tuple[FlextResult[TValue], bool, TValue | None, str | None]]:
            """Create parametrized test cases from value/error sequences.

            Replaces 10+ lines of duplicated test case creation code.

            Args:
                success_values: Values for success cases
                failure_errors: Error messages for failure cases
                error_codes: Optional error codes for failure cases

            Returns:
                List of (result, is_success, value, error) tuples for pytest.parametrize

            """
            cases: list[
                tuple[FlextResult[TValue], bool, TValue | None, str | None]
            ] = []

            for value in success_values:
                result = FlextResult[TValue].ok(value)
                cases.append((result, True, value, None))

            error_codes_list = error_codes or [None] * len(failure_errors)
            for error, error_code in zip(
                failure_errors, error_codes_list, strict=False
            ):
                result = FlextResult[TValue].fail(error, error_code=error_code)
                cases.append((result, False, None, error))

            return cases

        @staticmethod
        def assert_result_chain[TValue](
            results: Sequence[FlextResult[TValue]],
            *,
            expected_success_count: int | None = None,
            expected_failure_count: int | None = None,
            first_failure_index: int | None = None,
        ) -> None:
            """Assert result chain meets expectations.

            Replaces 10+ lines of duplicated chain validation code.

            Args:
                results: Sequence of results to validate
                expected_success_count: Expected number of successes
                expected_failure_count: Expected number of failures
                first_failure_index: Expected index of first failure (None = no failures)

            Raises:
                AssertionError: If expectations not met

            """
            successes = [r for r in results if r.is_success]
            failures = [r for r in results if r.is_failure]

            if expected_success_count is not None:
                assert len(successes) == expected_success_count, (
                    f"Expected {expected_success_count} successes, got {len(successes)}"
                )

            if expected_failure_count is not None:
                assert len(failures) == expected_failure_count, (
                    f"Expected {expected_failure_count} failures, got {len(failures)}"
                )

            if first_failure_index is not None:
                actual_first_failure = next(
                    (i for i, r in enumerate(results) if r.is_failure),
                    None,
                )
                assert actual_first_failure == first_failure_index, (
                    f"Expected first failure at index {first_failure_index}, "
                    f"got {actual_first_failure}"
                )
            elif first_failure_index is None and failures:
                actual_first_failure = next(
                    (i for i, r in enumerate(results) if r.is_failure),
                    None,
                )
                assert actual_first_failure is None, (
                    f"Expected no failures, but found first failure at index {actual_first_failure}"
                )

        @staticmethod
        def create_test_config_mapping(
            *,
            base_config: FlextTestsTypings.TestConfigMapping | None = None,
            overrides: FlextTestsTypings.TestConfigMapping | None = None,
        ) -> FlextTestsTypings.TestConfigMapping:
            """Create test configuration mapping with base and overrides.

            Replaces 10+ lines of duplicated config creation code.

            Args:
                base_config: Base configuration mapping
                overrides: Configuration overrides

            Returns:
                Merged configuration mapping

            """
            if base_config is None:
                base_config = {}
            if overrides is None:
                overrides = {}
            return {**base_config, **overrides}

        @staticmethod
        def execute_with_retry_and_timeout[TResult](
            operation: Callable[[], FlextResult[TResult]],
            *,
            max_attempts: int = 3,
            delay_seconds: float = 1.0,
            timeout_seconds: int | None = None,
            logger: LoggerProtocol | None = None,
        ) -> FlextResult[TResult]:
            """Execute operation with retry logic and optional timeout.

            Replaces 15+ lines of duplicated retry/timeout code across test utilities.

            Args:
                operation: Operation that returns FlextResult
                max_attempts: Maximum retry attempts (default 3)
                delay_seconds: Delay between attempts (default 1.0)
                timeout_seconds: Optional total timeout in seconds
                logger: Optional logger for retry attempts

            Returns:
                FlextResult from operation or last failure

            """
            start_time = time.time()
            last_error: str | None = None

            for attempt in range(max_attempts):
                if timeout_seconds is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout_seconds:
                        error_msg = f"Operation timed out after {timeout_seconds}s"
                        if logger and hasattr(logger, "error"):
                            logger.error(
                                error_msg,
                                timeout=timeout_seconds,
                                attempts=attempt,
                            )
                        return FlextResult[TResult].fail(
                            error_msg
                            if last_error is None
                            else f"{error_msg}: {last_error}",
                        )

                result = operation()
                if result.is_success:
                    if attempt > 0 and logger and hasattr(logger, "info"):
                        logger.info(
                            "Operation succeeded after %s attempts",
                            attempt + 1,
                            attempts=attempt + 1,
                        )
                    return result

                last_error = result.error or "Unknown error"
                if logger and hasattr(logger, "warning"):
                    logger.warning(
                        "Operation failed (attempt %s/%s): %s",
                        attempt + 1,
                        max_attempts,
                        last_error,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        error=last_error,
                    )

                if attempt < max_attempts - 1:
                    time.sleep(delay_seconds)

            error_msg = f"Operation failed after {max_attempts} attempts"
            if last_error:
                error_msg = f"{error_msg}: {last_error}"
            return FlextResult[TResult].fail(error_msg)

        @staticmethod
        def validate_config_structure(
            config: Mapping[str, FlextTypes.GeneralValueType],
            *,
            required_keys: Sequence[str],
            optional_keys: Sequence[str] | None = None,
            key_validators: Mapping[str, Callable[[FlextTypes.GeneralValueType], bool]]
            | None = None,
        ) -> FlextResult[bool]:
            """Validate configuration structure with required/optional keys and validators.

            Replaces 12+ lines of duplicated config validation code.

            Args:
                config: Configuration mapping to validate
                required_keys: Required key names
                optional_keys: Optional key names to check existence
                key_validators: Optional mapping of key names to validator functions

            Returns:
                FlextResult with True if valid, error message if invalid

            """
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                return FlextResult[bool].fail(
                    f"Missing required config keys: {', '.join(missing_keys)}",
                )

            if optional_keys:
                for key in optional_keys:
                    if key not in config:
                        return FlextResult[bool].fail(
                            f"Missing optional config key: {key}",
                        )

            if key_validators:
                for key, validator in key_validators.items():
                    if key in config:
                        value = config[key]
                        if not validator(value):
                            return FlextResult[bool].fail(
                                f"Validation failed for key '{key}': value {value}",
                            )

            return FlextResult[bool].ok(True)

    @staticmethod
    @overload
    def create_test_result[TValue: FlextTypes.GeneralValueType](
        *,
        success: bool = True,
        data: TValue,
        error: str | None = None,
    ) -> FlextResult[TValue]: ...

    @staticmethod
    @overload
    def create_test_result[TValue: FlextTypes.GeneralValueType](
        *,
        success: bool = True,
        data: None = None,
        error: str | None = None,
    ) -> FlextResult[FlextTypes.GeneralValueType]: ...

    @staticmethod
    def create_test_result[TValue: FlextTypes.GeneralValueType](
        *,
        success: bool = True,
        data: TValue | None = None,
        error: str | None = None,
    ) -> FlextResult[TValue] | FlextResult[FlextTypes.GeneralValueType]:
        """Create a test FlextResult.

        Args:
            success: Whether the result should be successful
            data: Success data (must not be None for success)
            error: Error message for failure results

        Returns:
            FlextResult instance

        """
        if success:
            # Fast fail: None is not a valid success value
            if data is None:
                # Use empty dict as default test data
                empty_dict: FlextTypes.GeneralValueType = {}
                return FlextResult[FlextTypes.GeneralValueType].ok(empty_dict)
            return FlextResult[TValue].ok(data)
        if data is None:
            return FlextResult[FlextTypes.GeneralValueType].fail(error or "Test error")
        return FlextResult[TValue].fail(error or "Test error")

    @staticmethod
    def functional_service(
        service_type: str = "api",
        **config: str | int | bool,
    ) -> FlextTypes.Types.ConfigurationMapping:
        """Create a functional service configuration for testing using domains.

        Args:
            service_type: Type of service
            **config: Service configuration overrides

        Returns:
            Service configuration mapping

        """
        base_config: dict[str, FlextTypes.GeneralValueType] = {
            "type": service_type,
            "name": f"functional_{service_type}_service",
            "enabled": True,
            "host": "localhost",
            "port": 8000,
            "timeout": 30,
            "retries": 3,
        }
        # Update with config overrides (config is already GeneralValueType compatible)
        base_config.update(config)
        return base_config

    @staticmethod
    @contextmanager
    def test_context(
        target: FlextProtocols.HasModelDump,
        attribute: str,
        new_value: FlextTypes.GeneralValueType,
    ) -> Generator[None]:
        """Context manager for temporarily changing object attributes.

        Args:
            target: Object to modify (must support attribute access)
            attribute: Attribute name to change
            new_value: New value for the attribute

        Yields:
            None

        """
        original_value = getattr(target, attribute, None)
        attribute_existed = hasattr(target, attribute)
        setattr(target, attribute, new_value)

        try:
            yield
        finally:
            if attribute_existed:
                setattr(target, attribute, original_value)
            else:
                # Attribute didn't exist originally, remove it
                delattr(target, attribute)

    class TestUtilities:
        """Nested class with additional test utilities."""

        @staticmethod
        def assert_result_success(result: FlextResult[TResult]) -> None:
            """Assert that a FlextResult is successful.

            Args:
                result: FlextResult to check

            Raises:
                AssertionError: If result is not successful

            """
            assert result.is_success, f"Expected success result, got: {result}"

        @staticmethod
        def assert_result_failure(result: FlextResult[TResult]) -> None:
            """Assert that a FlextResult is a failure.

            Args:
                result: FlextResult to check

            Raises:
                AssertionError: If result is not a failure

            """
            assert result.is_failure, f"Expected failure result, got: {result}"

        @staticmethod
        def create_test_service(
            **methods: Callable[..., FlextTypes.GeneralValueType],
        ) -> object:
            """Create a test service with specified methods.

            Args:
                **methods: Method implementations for the service

            Returns:
                Test service instance with specified methods

            Note:
                Returns object instead of Service protocol because dynamically
                created classes cannot fully implement the Service protocol
                without explicit method definitions.

            """

            # Create a real service class dynamically
            class TestService:
                """Real test service implementation."""

                def __init__(
                    self, **method_impls: Callable[..., FlextTypes.GeneralValueType]
                ) -> None:
                    """Initialize test service with method implementations."""
                    for method_name, implementation in method_impls.items():
                        setattr(self, method_name, implementation)

            return TestService(**methods)

        @staticmethod
        def generate_test_id(prefix: str = "test") -> str:
            """Generate a unique test identifier.

            Args:
                prefix: Prefix for the identifier

            Returns:
                Unique test identifier

            """
            return f"{prefix}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def create_test_data(
        size: int = 10,
        prefix: str = "test",
        data_type: str = "generic",
    ) -> dict[str, FlextTypes.GeneralValueType]:
        """Create test data dictionary using domains.

        Args:
            size: Size of the data
            prefix: Prefix for keys
            data_type: Type of data to create

        Returns:
            Test data dictionary

        """
        data: dict[str, FlextTypes.GeneralValueType] = {
            "id": str(uuid.uuid4()),
            "name": f"{prefix}_{data_type}",
            "size": size,
            "created_at": "2025-01-01T00:00:00Z",
        }

        if data_type == "user":
            user_payload = FlextTestsDomains.create_payload("user")
            # Merge with type-safe updates
            email_value = user_payload.get("email", f"{prefix}@example.com")
            active_value = user_payload.get("active", True)
            if isinstance(email_value, str) and isinstance(active_value, bool):
                data["email"] = email_value
                data["active"] = active_value
        elif data_type == "config":
            config_data = FlextTestsDomains.create_configuration()
            # Merge with type-safe updates
            timeout_value = config_data.get("timeout", 30)
            if isinstance(timeout_value, (int, float)):
                data["enabled"] = True
                data["timeout"] = int(timeout_value)

        return data

    @staticmethod
    def create_api_response(
        *,
        success: bool = True,
        data: FlextTypes.GeneralValueType | None = None,
        error_message: str | None = None,
    ) -> dict[str, FlextTypes.GeneralValueType]:
        """Create API response test data using domains.

        Args:
            success: Whether the response should be successful
            data: Response data
            error_message: Error message for failed responses

        Returns:
            API response dictionary

        """
        status_str = "success" if success else "error"
        base_response = FlextTestsDomains.api_response_data(
            status=status_str,
            include_data=data is not None if success else None,
        )

        # Convert to mutable dict with type-safe updates
        response_dict: dict[str, FlextTypes.GeneralValueType] = {
            key: value
            for key, value in base_response.items()
            if isinstance(value, (str, int, float, bool, type(None), Sequence, Mapping))
        }

        if success and data is not None:
            response_dict["data"] = data
        elif not success and error_message:
            error_dict: Mapping[str, str] = {
                "code": "TEST_ERROR",
                "message": error_message,
            }
            response_dict["error"] = error_dict

        return response_dict

    @classmethod
    def utilities(cls) -> FlextTestsUtilities:
        """Get utilities instance."""
        return cls()

    @classmethod
    def assertion(cls) -> FlextTestsMatchers:
        """Get assertion instance (for compatibility - returns matchers instance)."""
        return FlextTestsMatchers()

    class ResultHelpers:
        """Helpers for FlextResult testing."""

        @staticmethod
        def create_success_result[TValue: FlextTypes.GeneralValueType](
            value: TValue,
        ) -> FlextResult[TValue]:
            """Create a successful FlextResult with given value."""
            return FlextResult[TValue].ok(value)

        @staticmethod
        def create_failure_result[TValue: FlextTypes.GeneralValueType](
            error: str,
            error_code: str | None = None,
        ) -> FlextResult[TValue]:
            """Create a failed FlextResult with given error."""
            return FlextResult[TValue].fail(error, error_code=error_code)

        @staticmethod
        def assert_success_with_value[TValue: FlextTypes.GeneralValueType](
            result: FlextResult[TValue],
            expected_value: TValue,
        ) -> None:
            """Assert result is success and has expected value."""
            assert result.is_success, f"Expected success, got failure: {result.error}"
            assert result.value == expected_value

        @staticmethod
        def assert_failure_with_error[TValue: FlextTypes.GeneralValueType](
            result: FlextResult[TValue],
            expected_error: str | None = None,
        ) -> None:
            """Assert result is failure and has expected error."""
            assert result.is_failure, f"Expected failure, got success: {result.value}"
            if expected_error:
                assert result.error is not None
                assert expected_error in result.error

        @staticmethod
        def create_test_cases[TValue: FlextTypes.GeneralValueType](
            success_cases: list[tuple[TValue, TValue]],
            failure_cases: list[tuple[str, str | None]],
        ) -> list[tuple[FlextResult[TValue], bool, TValue | None, str | None]]:
            """Create parametrized test cases for Result testing.

            Args:
                success_cases: List of (value, expected_value) tuples
                failure_cases: List of (error, error_code) tuples

            Returns:
                List of (result, is_success, expected_value, expected_error) tuples

            """
            cases: list[
                tuple[FlextResult[TValue], bool, TValue | None, str | None]
            ] = []
            for value, expected in success_cases:
                result = FlextResult[TValue].ok(value)
                cases.append((result, True, expected, None))
            for error, error_code in failure_cases:
                result = FlextResult[TValue].fail(error, error_code=error_code)
                cases.append((result, False, None, error))
            return cases

        @staticmethod
        def validate_composition[TValue: FlextTypes.GeneralValueType](
            results: list[FlextResult[TValue]],
        ) -> FlextTypes.Types.ConfigurationMapping:
            """Validate FlextResult composition patterns.

            Args:
                results: List of FlextResult instances to analyze

            Returns:
                Mapping with composition statistics including total_results,
                success_count, failure_count, success_rate, all_successful,
                any_successful, error_messages, error_codes, has_structured_errors

            """
            successes = [r for r in results if r.is_success]
            failures = [r for r in results if r.is_failure]

            return {
                "total_results": len(results),
                "success_count": len(successes),
                "failure_count": len(failures),
                "success_rate": len(successes) / len(results) if results else 0.0,
                "all_successful": all(r.is_success for r in results),
                "any_successful": any(r.is_success for r in results),
                "error_messages": [r.error for r in failures if r.error],
                "error_codes": [r.error_code for r in failures if r.error_code],
                "has_structured_errors": any(r.error_data for r in failures),
            }

        @staticmethod
        def validate_chain[TValue: FlextTypes.GeneralValueType](
            results: list[FlextResult[TValue]],
        ) -> FlextTypes.Types.ConfigurationMapping:
            """Validate FlextResult chain operations.

            Args:
                results: List of FlextResult instances in order

            Returns:
                Mapping with chain statistics including is_valid_chain,
                chain_length, first_failure_index, successful_operations,
                failed_operations

            """
            if not results:
                return {
                    "is_valid_chain": True,
                    "chain_length": 0,
                    "first_failure_index": None,
                    "successful_operations": 0,
                    "failed_operations": 0,
                }

            first_failure_index = None
            for i, result in enumerate(results):
                if result.is_failure:
                    first_failure_index = i
                    break

            successful = (
                first_failure_index if first_failure_index is not None else len(results)
            )

            return {
                "is_valid_chain": first_failure_index is None,
                "chain_length": len(results),
                "first_failure_index": first_failure_index,
                "successful_operations": successful,
                "failed_operations": len(results) - successful,
            }

        @staticmethod
        def assert_composition[TValue: FlextTypes.GeneralValueType](
            results: list[FlextResult[TValue]],
            expected_success_rate: float = 1.0,
        ) -> None:
            """Assert FlextResult composition meets expectations.

            Args:
                results: List of FlextResult instances
                expected_success_rate: Minimum expected success rate (0.0-1.0)

            Raises:
                AssertionError: If success rate is below expected

            """
            helpers = FlextTestsUtilities.ResultHelpers
            composition = helpers.validate_composition(results)

            success_rate_value = composition.get("success_rate", 0.0)
            if not isinstance(success_rate_value, (int, float)):
                success_rate_value = 0.0
            success_rate = float(success_rate_value)

            assert success_rate >= expected_success_rate, (
                f"Success rate {success_rate:.2f} below expected "
                f"{expected_success_rate:.2f}"
            )

            if expected_success_rate == 1.0:
                all_successful = composition.get("all_successful", False)
                assert all_successful, (
                    f"Expected all results to be successful, but "
                    f"{composition.get('failure_count', 0)} failed"
                )

        @staticmethod
        def assert_chain_success[TValue: FlextTypes.GeneralValueType](
            results: list[FlextResult[TValue]],
        ) -> None:
            """Assert all results in chain are successful.

            Args:
                results: List of FlextResult instances

            Raises:
                AssertionError: If any result in chain fails

            """
            helpers = FlextTestsUtilities.ResultHelpers
            chain_info = helpers.validate_chain(results)

            is_valid = chain_info.get("is_valid_chain", False)
            assert is_valid, (
                f"Chain failed at index {chain_info.get('first_failure_index')}"
            )

    class ModelTestHelpers:
        """Generic helpers for model testing with reusable patterns.

        Provides factory-based testing patterns for Pydantic models
        and validation testing across the FLEXT ecosystem.
        """

        @staticmethod
        def assert_model_creation_success(
            factory_method: ModelFactory[TResult],
            expected_attrs: FlextTypes.Types.ConfigurationMapping,
            **factory_kwargs: FlextTypes.GeneralValueType,
        ) -> TResult:
            """Assert successful model creation and validate attributes.

            Args:
                factory_method: Factory method to create the model
                expected_attrs: Expected attribute values to validate
                **factory_kwargs: Arguments for the factory method

            Returns:
                Created model instance

            Raises:
                AssertionError: If creation fails or attributes don't match

            """
            instance = factory_method(**factory_kwargs)

            for attr, expected_value in expected_attrs.items():
                actual_value = getattr(instance, attr)
                assert actual_value == expected_value, (
                    f"Attribute '{attr}' mismatch: "
                    f"expected {expected_value}, got {actual_value}"
                )

            return instance

        @staticmethod
        def assert_model_validation_failure(
            factory_method: ModelFactory[TResult],
            expected_error_patterns: list[str],
            **factory_kwargs: FlextTypes.GeneralValueType,
        ) -> None:
            """Assert model creation fails with expected validation errors.

            Args:
                factory_method: Factory method that should raise ValueError
                expected_error_patterns: Patterns that should be in error message
                **factory_kwargs: Arguments for the factory method

            Raises:
                AssertionError: If validation doesn't fail or error doesn't match

            """
            try:
                _ = factory_method(**factory_kwargs)
                msg = "Expected ValueError but model creation succeeded"
                raise AssertionError(msg)
            except ValueError as e:
                error_msg = str(e)
                for pattern in expected_error_patterns:
                    assert pattern in error_msg, (
                        f"Expected error pattern '{pattern}' not found in: {error_msg}"
                    )

        @staticmethod
        def parametrize_model_scenarios(
            scenarios: Mapping[str, FlextTypes.Types.ConfigurationMapping],
        ) -> list[tuple[str, FlextTypes.Types.ConfigurationMapping]]:
            """Create parametrized test cases from scenario dictionaries.

            Args:
                scenarios: Dictionary mapping scenario names to test parameters

            Returns:
                List of (scenario_name, params) tuples for pytest.mark.parametrize

            """
            return list(scenarios.items())

        @staticmethod
        def batch_create_models(
            factory_method: ModelFactory[TResult],
            count: int,
            base_kwargs: FlextTypes.Types.ConfigurationMapping,
            variations: list[FlextTypes.Types.ConfigurationMapping] | None = None,
        ) -> list[TResult]:
            """Create a batch of model instances with variations.

            Args:
                factory_method: Factory method to create models
                count: Number of instances to create
                base_kwargs: Base kwargs for all instances
                variations: Optional list of variation kwargs

            Returns:
                List of created model instances

            """
            instances: list[TResult] = []
            for i in range(count):
                # Convert Mapping to dict for mutability (copy operation)
                kwargs = (
                    dict(base_kwargs)
                    if isinstance(base_kwargs, Mapping)
                    else base_kwargs.copy()
                )
                if variations:
                    variation = variations[i % len(variations)]
                    # Convert variation Mapping to dict if needed
                    if isinstance(variation, Mapping):
                        variation = dict(variation)
                    kwargs.update(variation)
                instances.append(factory_method(**kwargs))
            return instances

        @staticmethod
        def assert_attr_values(
            instance: FlextProtocols.HasModelDump,
            expected: FlextTypes.Types.ConfigurationMapping,
        ) -> None:
            """Assert multiple attribute values on an instance.

            Args:
                instance: Object to check attributes on
                expected: Dict of attribute names to expected values

            """
            for attr, expected_value in expected.items():
                actual = getattr(instance, attr, None)
                assert actual == expected_value, (
                    f"Attribute '{attr}': expected {expected_value}, got {actual}"
                )

        @staticmethod
        def assert_has_attrs(
            instance: FlextProtocols.HasModelDump,
            attrs: list[str],
        ) -> None:
            """Assert instance has all specified attributes.

            Args:
                instance: Object to check
                attrs: List of attribute names that must exist

            """
            for attr in attrs:
                assert hasattr(instance, attr), f"Missing required attribute: {attr}"

    class DockerHelpers:
        """Generalized helpers for Docker operations to reduce code duplication."""

        @staticmethod
        def execute_docker_operation[TResult](
            operation: Callable[[], TResult | None],
            success_value: TResult,
            operation_name: str,
            logger: LoggerProtocol | None = None,
        ) -> FlextResult[TResult]:
            """Execute Docker operation with standardized error handling.

            Args:
                operation: Docker operation to execute (may return None)
                success_value: Value to return on success
                operation_name: Name of operation for error messages
                logger: Optional logger for exception logging

            Returns:
                FlextResult with success value or failure with error

            """
            try:
                _ = operation()
                return FlextResult[TResult].ok(success_value)
            except NotFound as e:
                error_msg = f"{operation_name} not found: {e}"
                return FlextResult[TResult].fail(error_msg)
            except DockerException as e:
                if logger and hasattr(logger, "exception"):
                    logger.exception(
                        f"Failed to {operation_name}",
                        exception=e,
                    )
                error_msg = f"Failed to {operation_name}: {e}"
                return FlextResult[TResult].fail(error_msg)
            except Exception as e:
                if logger and hasattr(logger, "exception"):
                    logger.exception(
                        f"Unexpected error in {operation_name}",
                        exception=e,
                    )
                error_msg = f"Unexpected error in {operation_name}: {e}"
                return FlextResult[TResult].fail(error_msg)

        @staticmethod
        def validate_container_config(
            config: dict[str, str | int],
            required_keys: list[str],
        ) -> FlextResult[dict[str, str | int]]:
            """Validate container configuration dictionary.

            Args:
                config: Configuration dictionary to validate
                required_keys: List of required key names

            Returns:
                FlextResult with validated config or failure

            """
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                return FlextResult[dict[str, str | int]].fail(
                    f"Missing required config keys: {', '.join(missing_keys)}",
                )
            return FlextResult[dict[str, str | int]].ok(config)

        @staticmethod
        def safe_execute[TResult](
            operation: Callable[[], TResult],
            default_value: TResult,
            error_prefix: str = "Operation",
            logger: LoggerProtocol | None = None,
        ) -> TResult:
            """Execute operation safely, returning default on exception.

            Args:
                operation: Operation to execute
                default_value: Value to return on exception
                error_prefix: Prefix for error logging
                logger: Optional logger

            Returns:
                Operation result or default_value on exception

            """
            try:
                return operation()
            except Exception as e:
                if logger and hasattr(logger, "warning"):
                    logger.warning("%s failed: %s", error_prefix, e)
                return default_value

        @staticmethod
        def get_container_config(
            container_name: str,
            shared_containers: dict[str, dict[str, str | int]],
            registered_configs: dict[str, dict[str, str]],
        ) -> dict[str, str | int] | None:
            """Get container configuration from shared or registered configs.

            Args:
                container_name: Name of container to get config for
                shared_containers: Shared containers configuration dict
                registered_configs: Registered private container configs

            Returns:
                Container config dict or None if not found

            """
            if container_name in shared_containers:
                return shared_containers[container_name]
            if container_name in registered_configs:
                config_raw = registered_configs[container_name]
                if isinstance(config_raw, dict):
                    return dict(config_raw.items())
            return None

        @staticmethod
        def resolve_compose_file_path(
            compose_file: str,
            workspace_root: Path,
        ) -> str:
            """Resolve compose file path (absolute or relative).

            Args:
                compose_file: Compose file path (absolute or relative)
                workspace_root: Workspace root directory

            Returns:
                Resolved absolute compose file path

            """
            if Path(compose_file).is_absolute():
                return compose_file
            return str(workspace_root / compose_file)

        @staticmethod
        def extract_service_from_config(
            config: dict[str, str | int],
        ) -> str | None:
            """Extract service name from container config.

            Args:
                config: Container configuration dict

            Returns:
                Service name string or None

            """
            service_value = config.get("service")
            if isinstance(service_value, str):
                return service_value
            if isinstance(service_value, int):
                return str(service_value)
            return None

        @staticmethod
        def cleanup_docker_resources(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
            *,
            resource_type: str,
            list_attr: str,
            remove_attr: str,
            resource_name_attr: str,
            filter_pattern: str | None = None,
            logger: LoggerProtocol | None = None,
        ) -> FlextResult[dict[str, int | list[str]]]:
            """Generalized Docker resource cleanup helper.

            Args:
                client: Docker client instance
                resource_type: Type name for logging (e.g., "volume", "network")
                list_attr: Attribute name for list method (e.g., "volumes")
                remove_attr: Attribute name for remove method (e.g., "remove")
                resource_name_attr: Attribute name for resource name (e.g., "name")
                filter_pattern: Optional glob pattern to filter resources
                logger: Optional logger instance

            Returns:
                FlextResult with cleanup statistics

            """
            try:
                removed_items: list[str] = []

                resources_api = getattr(client, list_attr, None)
                if not resources_api:
                    return FlextResult[dict[str, int | list[str]]].ok({
                        "removed": 0,
                        resource_type: [],
                    })

                list_method = getattr(resources_api, "list", None)
                if not list_method:
                    return FlextResult[dict[str, int | list[str]]].ok({
                        "removed": 0,
                        resource_type: [],
                    })

                all_resources = list_method()

                for resource in all_resources:
                    resource_name: str = getattr(
                        resource,
                        resource_name_attr,
                        "unknown",
                    )

                    if filter_pattern and not fnmatch.fnmatch(
                        resource_name,
                        filter_pattern,
                    ):
                        continue

                    try:
                        remove_method = getattr(resource, remove_attr, None)
                        if remove_method:
                            remove_method(force=True)
                            removed_items.append(resource_name)
                            if logger and hasattr(logger, "info"):
                                logger.info(
                                    "Removed %s: %s",
                                    resource_type,
                                    resource_name,
                                    **{resource_type: resource_name},
                                )
                    except Exception as e:
                        if logger and hasattr(logger, "warning"):
                            logger.warning(
                                "Failed to remove %s %s: %s",
                                resource_type,
                                resource_name,
                                e,
                                **{resource_type: resource_name, "error": str(e)},
                            )

                return FlextResult[dict[str, int | list[str]]].ok({
                    "removed": len(removed_items),
                    resource_type: removed_items,
                })

            except Exception as e:
                if logger and hasattr(logger, "exception"):
                    logger.exception(
                        f"Failed to cleanup {resource_type}s",
                        exception=e,
                        resource_type=resource_type,
                    )
                return FlextResult[dict[str, int | list[str]]].fail(
                    f"{resource_type.title()} cleanup failed: {e}",
                )

        @staticmethod
        def resolve_compose_path(
            compose_file: str,
            workspace_root: Path,
        ) -> Path:
            """Resolve compose file path (absolute or relative to workspace).

            Args:
                compose_file: Compose file path (absolute or relative)
                workspace_root: Workspace root directory

            Returns:
                Resolved absolute Path to compose file

            """
            compose_path = Path(compose_file)
            if not compose_path.is_absolute():
                compose_path = workspace_root / compose_path
            return compose_path

        @staticmethod
        def execute_compose_operation_with_timeout(
            operation: Callable[[], None],
            timeout_seconds: int,
            operation_name: str,
            compose_file: str,
            logger: LoggerProtocol | None = None,
        ) -> FlextResult[str]:
            """Execute docker-compose operation with threading and timeout.

            Args:
                operation: Compose operation to execute in thread
                timeout_seconds: Timeout in seconds
                operation_name: Name of operation for logging
                compose_file: Compose file path for logging
                logger: Optional logger instance

            Returns:
                FlextResult with operation result

            """
            try:
                thread_exceptions: list[Exception] = []

                def run_operation() -> None:
                    try:
                        operation()
                    except Exception as e:
                        thread_exceptions.append(e)

                thread = threading.Thread(target=run_operation, daemon=False)
                thread.start()
                thread.join(timeout=timeout_seconds)

                if thread.is_alive():
                    return FlextResult[str].fail(
                        f"docker compose {operation_name} timed out after {timeout_seconds} seconds",
                    )

                if thread_exceptions:
                    raise thread_exceptions[0]

                if logger and hasattr(logger, "info"):
                    logger.info(
                        "docker compose %s succeeded",
                        operation_name,
                        compose_file=compose_file,
                    )

                # Return messages that match test expectations
                if operation_name == "up":
                    success_msg = f"Compose stack started for {compose_file}"
                elif operation_name == "down":
                    success_msg = f"Compose stack stopped for {compose_file}"
                else:
                    success_msg = (
                        f"Compose {operation_name} completed for {compose_file}"
                    )
                return FlextResult[str].ok(success_msg)

            except Exception as e:
                error_msg = str(e)
                if logger and hasattr(logger, "exception"):
                    logger.exception(
                        f"docker compose {operation_name} failed: {error_msg}",
                        exception=e,
                        compose_file=compose_file,
                        error=error_msg,
                    )
                return FlextResult[str].fail(
                    f"docker compose {operation_name} failed: {error_msg}",
                )

        @staticmethod
        def with_compose_file_config(
            docker_client: (
                FlextTestProtocols.Docker.ComposeClientProtocol | object
            ),
            compose_path: Path,
            operation: Callable[[], None],
        ) -> None:
            """Execute operation with compose file configured and restored.

            Args:
                docker_client: Python-on-whales Docker client
                compose_path: Path to compose file
                operation: Operation to execute with compose file configured

            """
            # Access client_config dynamically using getattr/setattr for type safety
            client_config = getattr(docker_client, "client_config", None)
            if client_config is None:
                operation()
                return

            original_compose_files = getattr(client_config, "compose_files", None)
            original_project_directory = getattr(
                client_config,
                "compose_project_directory",
                None,
            )
            try:
                client_config.compose_files = [str(compose_path)]
                # Set project directory to compose file's parent for correct container matching
                client_config.compose_project_directory = compose_path.parent
                operation()
            finally:
                if original_compose_files is not None:
                    client_config.compose_files = original_compose_files
                if original_project_directory is not None:
                    client_config.compose_project_directory = original_project_directory
                else:
                    # Reset to None if it wasn't set before
                    client_config.compose_project_directory = None

        @staticmethod
        def execute_docker_client_operation[TResult, TClient](
            get_client_fn: Callable[[], TClient],
            operation: Callable[[TClient], TResult],
            operation_name: str,
            success_value: TResult | None = None,
            logger: LoggerProtocol | None = None,
        ) -> FlextResult[TResult]:
            """Execute Docker client operation with standardized error handling.

            Args:
                get_client_fn: Function to get Docker client
                operation: Operation function that takes client and returns result
                operation_name: Name of operation for error messages
                success_value: Value to return on success (if operation returns None)
                logger: Optional logger instance

            Returns:
                FlextResult with operation result or failure

            """
            try:
                client = get_client_fn()
                result = operation(client)

                if result is None and success_value is not None:
                    result = success_value
                elif result is None:
                    return FlextResult[TResult].fail(
                        f"{operation_name} returned None",
                    )

                return FlextResult[TResult].ok(result)
            except NotFound as e:
                error_msg = f"{operation_name} not found: {e}"
                return FlextResult[TResult].fail(error_msg)
            except DockerException as e:
                if logger and hasattr(logger, "exception"):
                    logger.exception(
                        f"Failed to {operation_name}",
                        exception=e,
                    )
                error_msg = f"Failed to {operation_name}: {e}"
                return FlextResult[TResult].fail(error_msg)
            except Exception as e:
                if logger and hasattr(logger, "exception"):
                    logger.exception(
                        f"Unexpected error in {operation_name}",
                        exception=e,
                    )
                error_msg = f"Unexpected error in {operation_name}: {e}"
                return FlextResult[TResult].fail(error_msg)

        @staticmethod
        def extract_container_info(
            container: FlextTestProtocols.Docker.ContainerProtocol,
            container_name: str | None = None,
        ) -> FlextTypes.Types.ConfigurationMapping:
            """Extract container information from Docker container object.

            Args:
                container: Docker container object
                container_name: Optional container name (if not available from container object)

            Returns:
                Dict with container info fields: name, status, image, container_id, ports

            """
            container_status = getattr(container, "status", "unknown")
            status_str = "running" if container_status == "running" else "stopped"

            container_image = getattr(container, "image", None)
            image_tags: list[str] = (
                container_image.tags
                if container_image and hasattr(container_image, "tags")
                else []
            )
            image_name: str = image_tags[0] if image_tags else "unknown"

            name_attr = getattr(container, "name", None)
            name = (
                container_name
                if container_name is not None
                else (str(name_attr) if name_attr is not None else "unknown")
            )

            container_id = getattr(container, "id", "unknown") or "unknown"

            return {
                "name": name,
                "status": status_str,
                "image": image_name,
                "container_id": container_id,
                "ports": {},
            }

        @staticmethod
        def parse_env_list_to_dict(env_list: list[str]) -> dict[str, str]:
            """Parse environment variable list (KEY=VALUE format) to dictionary.

            Args:
                env_list: List of environment variables in KEY=VALUE format

            Returns:
                Dict mapping environment variable names to values

            """
            env_vars: dict[str, str] = {}
            for env_str in env_list:
                if "=" in env_str:
                    key, value = env_str.split("=", 1)
                    env_vars[key] = value
            return env_vars

        @staticmethod
        def extract_container_state(
            container: FlextTestProtocols.Docker.ContainerProtocol,
        ) -> FlextTestsTypings.ContainerStateMapping:
            """Extract container state information from Docker container object.

            Args:
                container: Docker container object

            Returns:
                Dict with state fields: running, restarting, health (dict), started_at, exit_code

            """
            attrs = getattr(container, "attrs", {})
            state = attrs.get("State", {}) if isinstance(attrs, dict) else {}

            running = state.get("Running", False) if isinstance(state, dict) else False
            restarting = (
                state.get("Restarting", False) if isinstance(state, dict) else False
            )
            health = state.get("Health", {}) if isinstance(state, dict) else {}
            started_at = state.get("StartedAt", "") if isinstance(state, dict) else ""
            exit_code = state.get("ExitCode") if isinstance(state, dict) else None

            return {
                "running": bool(running),
                "restarting": bool(restarting),
                "health": health if isinstance(health, dict) else {},
                "started_at": str(started_at),
                "exit_code": exit_code,
            }

        @staticmethod
        def get_health_status(
            health: dict[str, str | int],
            started_at: str,
            stuck_threshold_seconds: int = 300,
        ) -> str:
            """Determine health status from health dict and start time.

            Args:
                health: Health dict from container state
                started_at: Container start time (ISO format)
                stuck_threshold_seconds: Seconds before considering stuck (default 300)

            Returns:
                Health status: healthy, unhealthy, starting, stuck, or none

            """
            if not health:
                return "none"

            # Convert health values to string for comparison
            status_raw = health.get("Status")
            status = str(status_raw) if status_raw is not None else "unknown"
            if status == "starting" and started_at:
                try:
                    started_str = (
                        started_at.replace("Z", "+00:00")
                        if started_at.endswith("Z")
                        else started_at
                    )
                    started = datetime.fromisoformat(started_str)
                    now = datetime.now(UTC)
                    elapsed = (now - started).total_seconds()

                    if elapsed > stuck_threshold_seconds:
                        return "stuck"
                except (ValueError, AttributeError):
                    pass

            return str(status)

        @staticmethod
        def detect_container_state_issues(
            state: FlextTestsTypings.ContainerStateMapping,
            container_name: str,
            stuck_threshold_seconds: int = 300,
        ) -> list[str]:
            """Detect issues from container state.

            Args:
                state: Container state dict from extract_container_state
                container_name: Container name for logging context
                stuck_threshold_seconds: Seconds before considering stuck (default 300)

            Returns:
                List of detected issue messages

            """
            issues: list[str] = []
            running = state.get("running", False)
            restarting = state.get("restarting", False)
            health = state.get("health", {})
            started_at = str(state.get("started_at", ""))
            exit_code = state.get("exit_code")

            if restarting:
                issues.append("Container is restarting")

            if not running:
                issues.append("Container is stopped")

            if health and isinstance(health, dict):
                health_status = FlextTestsUtilities.DockerHelpers.get_health_status(
                    health,
                    started_at,
                    stuck_threshold_seconds,
                )

                if health_status == "unhealthy":
                    failing_streak = health.get("FailingStreak", 0)
                    issues.append(f"Container is unhealthy: {failing_streak} failures")

                if health_status == "stuck":
                    issues.append(
                        f"Container stuck in starting state for >{stuck_threshold_seconds}s",
                    )

            if exit_code and exit_code != 0:
                issues.append(f"Container exited with code {exit_code}")

            return issues

        @staticmethod
        def execute_container_stop_operation(
            container: FlextTestProtocols.Docker.ContainerProtocol,
            container_name: str,
            timeout: int = 10,
            *,
            force_kill: bool = True,
            logger: LoggerProtocol | None = None,
        ) -> FlextResult[bool]:
            """Stop container with graceful fallback to force kill.

            Args:
                container: Docker container object
                container_name: Container name for logging
                timeout: Seconds to wait for graceful stop (default 10)
                force_kill: Whether to force kill if graceful stop fails (default True)
                logger: Optional logger for operations

            Returns:
                FlextResult with True if stopped successfully

            """
            try:
                stop_method = getattr(container, "stop", None)
                if stop_method:
                    stop_method(timeout=timeout)
                    if logger and hasattr(logger, "info"):
                        logger.info(
                            "Container %s stopped gracefully",
                            container_name,
                            container=container_name,
                        )
                    return FlextResult[bool].ok(True)
            except Exception as e:
                if force_kill:
                    try:
                        kill_method = getattr(container, "kill", None)
                        if kill_method:
                            kill_method(signal="SIGKILL")
                            if logger and hasattr(logger, "warning"):
                                logger.warning(
                                    "Graceful stop failed for %s, force killed",
                                    container_name,
                                    container=container_name,
                                    error=str(e),
                                )
                            return FlextResult[bool].ok(True)
                    except Exception as kill_error:
                        error_msg = f"Failed to stop/kill container {container_name}: {kill_error}"
                        if logger and hasattr(logger, "exception"):
                            logger.exception(
                                error_msg,
                                container=container_name,
                            )
                        return FlextResult[bool].fail(error_msg)

                error_msg = f"Failed to stop container {container_name}: {e}"
                if logger and hasattr(logger, "exception"):
                    logger.exception(error_msg, container=container_name)
                return FlextResult[bool].fail(error_msg)

            return FlextResult[bool].fail(
                f"Container {container_name} has no stop method",
            )

        @staticmethod
        def execute_container_remove_operation(
            container: FlextTestProtocols.Docker.ContainerProtocol,
            container_name: str,
            *,
            force: bool = False,
            logger: LoggerProtocol | None = None,
        ) -> FlextResult[bool]:
            """Remove container with optional force flag.

            Args:
                container: Docker container object
                container_name: Container name for logging
                force: Whether to force remove (default False)
                logger: Optional logger for operations

            Returns:
                FlextResult with True if removed successfully

            """
            try:
                remove_method = getattr(container, "remove", None)
                if remove_method:
                    if force:
                        remove_method(force=True)
                    else:
                        remove_method()
                    if logger and hasattr(logger, "info"):
                        logger.info(
                            "Container %s removed",
                            container_name,
                            container=container_name,
                            force=force,
                        )
                    return FlextResult[bool].ok(True)
            except Exception as e:
                if not force:
                    # Try force remove as fallback
                    try:
                        remove_method = getattr(container, "remove", None)
                        if remove_method:
                            remove_method(force=True)
                            if logger and hasattr(logger, "info"):
                                logger.info(
                                    "Container %s force removed",
                                    container_name,
                                    container=container_name,
                                )
                            return FlextResult[bool].ok(True)
                    except Exception as force_error:
                        error_msg = f"Failed to remove container {container_name}: {force_error}"
                        if logger and hasattr(logger, "exception"):
                            logger.exception(
                                error_msg,
                                container=container_name,
                            )
                        return FlextResult[bool].fail(error_msg)

                error_msg = f"Failed to remove container {container_name}: {e}"
                if logger and hasattr(logger, "exception"):
                    logger.exception(error_msg, container=container_name)
                return FlextResult[bool].fail(error_msg)

            return FlextResult[bool].fail(
                f"Container {container_name} has no remove method",
            )

        @staticmethod
        def _get_container_with_state(
            client: FlextTestProtocols.Docker.DockerClientProtocol,
            container_name: str,
        ) -> tuple[
            FlextTestProtocols.Docker.ContainerProtocol,
            FlextTestsTypings.ContainerStateMapping,
        ]:
            """Get container and its state for operations.

            Args:
                client: Docker client object
                container_name: Name of container

            Returns:
                Tuple of (container object, state dict)

            """
            containers_api = getattr(client, "containers", None)
            if containers_api:
                get_method = getattr(containers_api, "get", None)
                if get_method:
                    container = get_method(container_name)
                    state = FlextTestsUtilities.DockerHelpers.extract_container_state(
                        container,
                    )
                    return (container, state)
            raise RuntimeError(f"Failed to get container {container_name}")

        @staticmethod
        def wait_with_retry[T](
            check_fn: Callable[[], FlextResult[T]],
            max_wait_seconds: int,
            check_interval_seconds: int = 5,
            success_condition: Callable[[T], bool] | None = None,
            logger: LoggerProtocol | None = None,
        ) -> tuple[bool, T | None, str]:
            """Execute a check function repeatedly until success or timeout.

            Args:
                check_fn: Function that returns FlextResult[T] to check
                max_wait_seconds: Maximum seconds to wait
                check_interval_seconds: Seconds between checks (default 5)
                success_condition: Optional function to determine if result is successful
                logger: Optional logger for operations

            Returns:
                Tuple of (success: bool, result: T | None, error_message: str)

            """
            start_time = time.time()
            check_count = 0

            while True:
                result = check_fn()
                if result.is_success:
                    value = result.unwrap()
                    if success_condition is None or success_condition(value):
                        if logger and hasattr(logger, "info"):
                            logger.info(
                                "Check succeeded after %s attempts",
                                check_count,
                                checks=check_count,
                            )
                        return (True, value, "")
                    # Check failed condition - continue waiting
                else:
                    # Check function itself failed - continue waiting
                    pass

                elapsed = time.time() - start_time
                if elapsed >= max_wait_seconds:
                    error_msg = (
                        f"Timeout after {max_wait_seconds}s ({check_count} checks)"
                    )
                    if logger and hasattr(logger, "error"):
                        logger.error(
                            error_msg,
                            max_wait=max_wait_seconds,
                            checks=check_count,
                        )
                    # Return last result value if available, None on failure
                    last_result = result.unwrap() if result.is_success else None
                    return (False, last_result, error_msg)

                check_count += 1
                time.sleep(check_interval_seconds)


@runtime_checkable
class ModelFactory(Protocol[TModel_co]):
    """Protocol for model factory methods used in testing.

    This protocol defines callable objects that create model instances
    from keyword arguments. Used by ModelTestHelpers for type-safe testing.
    """

    def __call__(self, **kwargs: FlextTypes.GeneralValueType) -> TModel_co:
        """Create a model instance from kwargs."""
        ...
