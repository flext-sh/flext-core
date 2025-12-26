"""Utilities for flext-core tests.

Provides TestsFlextUtilities, extending FlextTestsUtilities with flext-core-specific
utilities. All generic test utilities come from flext_tests.

Architecture:
- FlextTestsUtilities (flext_tests) = Generic utilities for all FLEXT projects
- TestsFlextUtilities (tests/) = flext-core-specific utilities extending FlextTestsUtilities

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Callable, Iterator

from flext_core import FlextResult, t as core_t
from flext_core.utilities import u as core_u
from flext_tests.utilities import FlextTestsUtilities
from tests.test_utils import assertion_helpers


class TestsFlextUtilities(FlextTestsUtilities):
    """Utilities for flext-core tests - extends FlextTestsUtilities.

    Architecture: Extends FlextTestsUtilities with flext-core-specific utility
    definitions. All generic utilities from FlextTestsUtilities are available
    through inheritance.

    Rules:
    - NEVER redeclare utilities from FlextTestsUtilities
    - Only flext-core-specific utilities allowed
    - All generic utilities come from FlextTestsUtilities
    """

    # NOTE: FlextTestsUtilities extends FlextUtilities and provides:
    # - Result: assert_success, assert_failure, assert_success_with_value, etc.
    # - TestContext: temporary_attribute context manager
    # - Factory: create_result, create_test_data
    # - ModelTestHelpers, RegistryHelpers, ConfigHelpers
    # - All FlextUtilities classes via inheritance
    #
    # These are available through inheritance.

    # Expose FlextUtilities classes through real inheritance
    class Args(core_u.Args):
        """Args utility class for tests - real inheritance."""

    class Cache(core_u.Cache):
        """Cache utility class for tests - real inheritance."""

    class Checker(core_u.Checker):
        """Checker utility class for tests - real inheritance."""

    class Collection(core_u.Collection):
        """Collection utility class for tests - real inheritance."""

    class Configuration(core_u.Configuration):
        """Configuration utility class for tests - real inheritance."""

    class Context(core_u.Context):
        """Context utility class for tests - real inheritance."""

    class Domain(core_u.Domain):
        """Domain utility class for tests - real inheritance."""

    class Enum(core_u.Enum):
        """Enum utility class for tests - real inheritance."""

    class Generators(core_u.Generators):
        """Generators utility class for tests - real inheritance."""

    class Guards(core_u.Guards):
        """Guards utility class for tests - real inheritance."""

    class Mapper(core_u.Mapper):
        """Mapper utility class for tests - real inheritance."""

    class Model(core_u.Model):
        """Model utility class for tests - real inheritance."""

    class Pagination(core_u.Pagination):
        """Pagination utility class for tests - real inheritance."""

    class Parser(core_u.Parser):
        """Parser utility class for tests - real inheritance."""

    class Reliability(core_u.Reliability):
        """Reliability utility class for tests - real inheritance."""

    class Text(core_u.Text):
        """Text utility class for tests - real inheritance."""

    class Validation(core_u.Validation):
        """Validation utility class for tests - real inheritance."""

    class CoreParserHelpers:
        """Helper methods for parser testing - flext-core specific."""

        @staticmethod
        def execute_and_assert_parser_result(
            operation: Callable[[], FlextResult[core_t.GeneralValueType]],
            expected_value: core_t.GeneralValueType | None = None,
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
                (
                    assertion_helpers.assert_flext_result_failure(result),
                    (f"Expected failure for: {description}, got success"),
                )
                assert expected_error in str(result.error), (
                    f"Expected error '{expected_error}' in '{result.error}' for: {description}"
                )
            else:
                (
                    assertion_helpers.assert_flext_result_success(result),
                    (f"Expected success for: {description}, got: {result.error}"),
                )
                if expected_value is not None:
                    assert result.value == expected_value, (
                        f"Expected {expected_value}, got {result.value} for: {description}"
                    )

    class CoreBadObjects:
        """Factory for objects that cause errors during testing - flext-core specific."""

        class BadSplitString:
            """String-like object that raises on split()."""

            def split(
                self,
                *_args: core_t.GeneralValueType,
                **_kwargs: core_t.GeneralValueType,
            ) -> list[str]:
                """Raise error on split attempt."""
                msg = "Bad split"
                raise RuntimeError(msg)

            def __str__(self) -> str:
                """Return string representation."""
                return "bad_split_string"

        @staticmethod
        def create_for_split() -> TestsFlextUtilities.CoreBadObjects.BadSplitString:
            """Create object that fails on split()."""
            return TestsFlextUtilities.CoreBadObjects.BadSplitString()

        class BadIndexString:
            """String-like object that raises on indexing."""

            def __getitem__(self, key: int) -> str:
                """Raise error on index attempt."""
                msg = "Bad index"
                raise RuntimeError(msg)

            def __str__(self) -> str:
                """Return string representation."""
                return "bad_index_string"

        @staticmethod
        def create_for_index() -> TestsFlextUtilities.CoreBadObjects.BadIndexString:
            """Create object that fails on indexing."""
            return TestsFlextUtilities.CoreBadObjects.BadIndexString()

        class BadStrObject:
            """Object that raises on str() conversion."""

            def __str__(self) -> str:
                """Raise error on str() attempt."""
                msg = "Bad str"
                raise RuntimeError(msg)

        @staticmethod
        def create_for_str() -> TestsFlextUtilities.CoreBadObjects.BadStrObject:
            """Create object that fails on str()."""
            return TestsFlextUtilities.CoreBadObjects.BadStrObject()

        class BadDict(UserDict[str, core_t.GeneralValueType]):
            """Dict that raises on get()."""

            def __getitem__(self, key: str) -> core_t.GeneralValueType:
                """Raise error on get attempt."""
                msg = "Bad dict get"
                raise RuntimeError(msg)

        class BadList(UserList[core_t.GeneralValueType]):
            """List that raises on iteration."""

            def __iter__(self) -> Iterator[core_t.GeneralValueType]:
                """Raise error on iteration."""
                msg = "Bad list iteration"
                raise RuntimeError(msg)

        class BadModelDump:
            """Object with model_dump that raises."""

            def model_dump(self) -> dict[str, core_t.GeneralValueType]:
                """Raise error on model_dump."""
                msg = "Bad model_dump"
                raise RuntimeError(msg)

        class BadConfig:
            """Config object that raises on attribute access."""

            def get_attribute(self, name: str) -> core_t.GeneralValueType:
                """Raise error on attribute access."""
                msg = f"Bad config: {name}"
                raise AttributeError(msg)

    class CoreAssertions:
        """Assertion helpers for test validation - flext-core specific."""

        @staticmethod
        def assert_failure(
            result: FlextResult[core_t.GeneralValueType],
            expected_error: str,
            description: str = "",
        ) -> None:
            """Assert that result is a failure with expected error.

            Args:
                result: FlextResult to check
                expected_error: Expected error substring
                description: Test case description for error messages

            """
            (
                assertion_helpers.assert_flext_result_failure(result),
                (f"Expected failure for: {description}, got success"),
            )
            assert expected_error in str(result.error), (
                f"Expected error '{expected_error}' in '{result.error}' for: {description}"
            )

        @staticmethod
        def assert_success(
            result: FlextResult[core_t.GeneralValueType],
            description: str = "",
        ) -> None:
            """Assert that result is a success.

            Args:
                result: FlextResult to check
                description: Test case description for error messages

            """
            (
                assertion_helpers.assert_flext_result_success(result),
                (f"Expected success for: {description}, got: {result.error}"),
            )

        @staticmethod
        def assert_success_with_value(
            result: FlextResult[core_t.GeneralValueType],
            expected_value: core_t.GeneralValueType,
            description: str = "",
        ) -> None:
            """Assert result is success with specific value.

            Args:
                result: FlextResult to check
                expected_value: Expected value
                description: Test case description for error messages

            """
            (
                assertion_helpers.assert_flext_result_success(result),
                (f"Expected success for: {description}, got: {result.error}"),
            )
            assert result.value == expected_value, (
                f"Expected {expected_value}, got {result.value} for: {description}"
            )


__all__ = [
    "TestsFlextUtilities",
]
