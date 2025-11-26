"""Generic test helpers for string parser and other test domains.

Provides reusable test utilities, factories, and test case data structures
using Python 3.13 advanced features. Designed to be generic and reusable
across multiple test modules with advanced patterns for code reduction.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TypeVar

from flext_core import FlextResult
from flext_core._utilities.string_parser import ParseOptions

TResult = TypeVar("TResult")
TService = TypeVar("TService")
TModel = TypeVar("TModel")


@dataclass(frozen=True, slots=True)
class ParseDelimitedCase:
    """Test case for parse_delimited method."""

    text: str
    delimiter: str
    expected: list[str] | None = None
    expected_error: str | None = None
    options: ParseOptions | None = None
    strip: bool = True
    remove_empty: bool = True
    validator: Callable[[str], bool] | None = None
    use_legacy: bool = False
    description: str = field(default="", compare=False)


@dataclass(frozen=True, slots=True)
class SplitEscapeCase:
    """Test case for split_on_char_with_escape method."""

    text: str
    split_char: str
    escape_char: str = "\\"
    expected: list[str] | None = None
    expected_error: str | None = None
    description: str = field(default="", compare=False)


@dataclass(frozen=True, slots=True)
class NormalizeWhitespaceCase:
    """Test case for normalize_whitespace method."""

    text: str
    pattern: str = r"\s+"
    replacement: str = " "
    expected: str | None = None
    expected_error: str | None = None
    description: str = field(default="", compare=False)


@dataclass(frozen=True, slots=True)
class RegexPipelineCase:
    """Test case for apply_regex_pipeline method."""

    text: str
    patterns: list[tuple[str, str] | tuple[str, str, int]]
    expected: str | None = None
    expected_error: str | None = None
    description: str = field(default="", compare=False)


@dataclass(frozen=True, slots=True)
class ObjectKeyCase:
    """Test case for get_object_key method."""

    obj: object
    expected_contains: list[str] | None = None
    expected_exact: str | None = None
    description: str = field(default="", compare=False)


class TestCaseFactory(ABC):
    """Abstract base class for test case factories."""

    @abstractmethod
    def generate_cases(self) -> Iterator[object]:
        """Generate test cases."""
        ...

    @abstractmethod
    def get_case_count(self) -> int:
        """Get total number of cases."""
        ...


@dataclass(frozen=True, slots=True)
class ServiceTestCase[TService]:
    """Generic service test case factory."""

    service_class: type[TService]
    input_data: dict[str, object]
    expected_success: bool = True
    expected_result: object = None
    expected_error: str | None = None
    description: str = field(default="", compare=False)

    def create_service(self) -> TService:
        """Create service instance with input data."""
        instance: TService = self.service_class(**self.input_data)
        return instance

    def assert_result(self, result: object) -> None:
        """Assert result matches expectations."""
        if self.expected_success:
            assert result == self.expected_result, (
                f"Expected {self.expected_result}, got {result}"
            )
        else:
            if isinstance(result, Exception):
                error_msg = str(result)
            elif isinstance(result, FlextResult):
                assert result.is_failure, f"Expected failure, got {result}"
                error_msg = result.error or ""
            else:
                msg = f"Expected Exception or FlextResult, got {type(result)}"
                raise TypeError(msg)
            if self.expected_error:
                assert self.expected_error in error_msg, (
                    f"Expected error containing '{self.expected_error}', got '{error_msg}'"
                )


class GenericTestFactories:
    """Generic factories for common test patterns."""

    @staticmethod
    def success_cases(
        service_class: type[TService],
        input_variations: list[dict[str, object]],
        expected_results: list[object],
    ) -> list[ServiceTestCase[TService]]:
        """Generate success test cases for a service."""
        return [
            ServiceTestCase[TService](
                service_class=service_class,
                input_data=inputs,
                expected_result=result,
                description=f"Success case {i + 1}",
            )
            for i, (inputs, result) in enumerate(
                zip(input_variations, expected_results, strict=True)
            )
        ]

    @staticmethod
    def failure_cases(
        service_class: type[TService],
        input_variations: list[dict[str, object]],
        error_messages: list[str],
    ) -> list[ServiceTestCase[TService]]:
        """Generate failure test cases for a service."""
        return [
            ServiceTestCase[TService](
                service_class=service_class,
                input_data=inputs,
                expected_success=False,
                expected_error=error,
                description=f"Failure case {i + 1}",
            )
            for i, (inputs, error) in enumerate(
                zip(input_variations, error_messages, strict=True)
            )
        ]


class TestHelpers:
    """Generic test helpers reusable across test modules using advanced patterns."""

    # Expose common utilities at class level for convenience
    @staticmethod
    def cast_to_str(obj: object) -> str:
        """Cast object to str for type checking."""
        if isinstance(obj, str):
            return obj
        msg = f"Expected str, got {type(obj)}"
        raise TypeError(msg)

    class BadObjects:
        """Factory for objects that fail operations (for exception testing)."""

        @staticmethod
        def create_for_split() -> object:
            """Create object that fails on split operation."""

            class BadString:
                def split(self, delimiter: str) -> list[str]:
                    msg = "Split failed"
                    raise RuntimeError(msg)

            return BadString()

        @staticmethod
        def create_for_index() -> object:
            """Create object that fails on index operation."""

            class BadString:
                def __len__(self) -> int:
                    return 5

                def __getitem__(self, key: int) -> str:
                    msg = "Index failed"
                    raise KeyError(msg)

            return BadString()

        @staticmethod
        def create_for_str() -> object:
            """Create object that fails on str conversion."""

            class BadString:
                def __str__(self) -> str:
                    msg = "String conversion failed"
                    raise RuntimeError(msg)

            return BadString()

        @staticmethod
        def create_for_attribute_access() -> object:
            """Create object that fails on attribute access."""

            class BadObject:
                def __getattr__(self, name: str) -> object:
                    msg = f"Attribute '{name}' access failed"
                    raise AttributeError(msg)

            return BadObject()

    class Assertions:
        """Generic assertion helpers for test results with advanced patterns."""

        @staticmethod
        def assert_success(
            result: FlextResult[TResult],
            expected: TResult,
            description: str = "",
        ) -> None:
            """Assert result is success with expected value."""
            assert result.is_success, f"Expected success for: {description}"
            assert result.unwrap() == expected

        @staticmethod
        def assert_failure(
            result: FlextResult[TResult],
            expected_error: str,
            description: str = "",
        ) -> None:
            """Assert result is failure with expected error."""
            assert result.is_failure, f"Expected failure for: {description}"
            assert result.error is not None
            assert expected_error in result.error

        @staticmethod
        def assert_service_result(
            service_result: object,
            *,
            expected_success: bool = True,
            expected_value: object = None,
            expected_error: str | None = None,
            description: str = "",
        ) -> None:
            """Assert service result with flexible checking."""
            if isinstance(service_result, FlextResult):
                result_obj: FlextResult[object] = service_result
                if expected_success:
                    assert result_obj.is_success, f"Expected success for: {description}"
                    if expected_value is not None:
                        unwrapped = result_obj.unwrap()
                        assert unwrapped == expected_value
                else:
                    assert result_obj.is_failure, f"Expected failure for: {description}"
                    if expected_error:
                        error_attr = result_obj.error
                        assert error_attr is not None
                        assert expected_error in error_attr
            elif isinstance(service_result, Exception):
                # Should be an exception
                if expected_error:
                    assert expected_error in str(service_result)
            # Direct result (auto_execute=True)
            elif expected_success:
                assert service_result == expected_value, (
                    f"Expected {expected_value}, got {service_result}"
                )
            else:
                msg = f"Expected FlextResult or Exception, got {type(service_result)}"
                raise TypeError(msg)

        @staticmethod
        def assert_type_and_value(
            obj: object,
            expected_type: type,
            expected_value: object = None,
            description: str = "",
        ) -> None:
            """Assert object type and optionally value."""
            assert isinstance(obj, expected_type), (
                f"Expected {expected_type.__name__}, got {type(obj).__name__} for: {description}"
            )
            if expected_value is not None:
                assert obj == expected_value, (
                    f"Expected {expected_value}, got {obj} for: {description}"
                )

    class Utilities:
        """Generic utility functions for tests."""

        @staticmethod
        def cast_to_str(obj: object) -> str:
            """Cast object to str for type checking."""
            if isinstance(obj, str):
                return obj
            msg = f"Expected str, got {type(obj)}"
            raise TypeError(msg)

        @staticmethod
        def create_test_data_variations(
            base_data: dict[str, object],
            variations: dict[str, list[object]],
        ) -> list[dict[str, object]]:
            """Create variations of test data."""
            results = [base_data.copy()]

            for key, values in variations.items():
                new_results = []
                for result in results:
                    for value in values:
                        new_result = result.copy()
                        new_result[key] = value
                        new_results.append(new_result)
                results = new_results

            return results

        @staticmethod
        def parametrize_test_cases(test_cases: list[object]) -> Iterator[object]:
            """Convert test cases to pytest parametrize format."""
            yield from test_cases


# Type aliases for better readability
TestCase = ServiceTestCase
Factory = GenericTestFactories
