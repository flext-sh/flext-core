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
from typing import Never, override

from flext_core import r
from flext_infra import FlextInfraUtilities
from flext_tests import FlextTestsUtilities, t

from .test_utils import assertion_helpers


class TestsFlextUtilities(FlextTestsUtilities, FlextInfraUtilities):
    """Utilities for flext-core tests - extends FlextTestsUtilities.

    Architecture: Extends FlextTestsUtilities with flext-core-specific utility
    definitions. All generic utilities from FlextTestsUtilities are available
    through inheritance.

    Rules:
    - NEVER redeclare utilities from FlextTestsUtilities
    - Only flext-core-specific utilities allowed
    - All generic utilities come from FlextTestsUtilities
    """

    class Tests(FlextTestsUtilities.Tests):
        """flext-core-specific test utilities namespace."""

        class CoreParserHelpers:
            """Helper methods for parser testing - flext-core specific."""

            @staticmethod
            def execute_and_assert_parser_result(
                operation: Callable[[], r[t.Container]],
                expected_value: t.Container | None = None,
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
                    _ = assertion_helpers.assert_flext_result_failure(
                        result,
                        f"Expected failure for: {description}, got success",
                        error_contains=expected_error,
                    )
                    assert expected_error in str(result.error), (
                        f"Expected error '{expected_error}' in '{result.error}' for: {description}"
                    )
                else:
                    _ = assertion_helpers.assert_flext_result_success(
                        result,
                        f"Expected success for: {description}, got: {result.error}",
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
                    *_args: str,
                    **_kwargs: t.Scalar,
                ) -> list[str]:
                    """Raise error on split attempt."""
                    msg = "Bad split"
                    raise RuntimeError(msg)

                @override
                def __str__(self) -> str:
                    """Return string representation."""
                    return "bad_split_string"

            @staticmethod
            def create_for_split() -> (
                TestsFlextUtilities.Tests.CoreBadObjects.BadSplitString
            ):
                """Create object that fails on split()."""
                return TestsFlextUtilities.Tests.CoreBadObjects.BadSplitString()

            class BadIndexString:
                """String-like object that raises on indexing."""

                def __getitem__(self, key: int) -> str:
                    """Raise error on index attempt."""
                    msg = "Bad index"
                    raise RuntimeError(msg)

                @override
                def __str__(self) -> str:
                    """Return string representation."""
                    return "bad_index_string"

            @staticmethod
            def create_for_index() -> (
                TestsFlextUtilities.Tests.CoreBadObjects.BadIndexString
            ):
                """Create object that fails on indexing."""
                return TestsFlextUtilities.Tests.CoreBadObjects.BadIndexString()

            class BadStrObject:
                """Object that raises on str() conversion."""

                @override
                def __str__(self) -> str:
                    """Raise error on str() attempt."""
                    msg = "Bad str"
                    raise RuntimeError(msg)

            @staticmethod
            def create_for_str() -> (
                TestsFlextUtilities.Tests.CoreBadObjects.BadStrObject
            ):
                """Create object that fails on str()."""
                return TestsFlextUtilities.Tests.CoreBadObjects.BadStrObject()

            class BadDict(UserDict[str, t.Tests.object]):
                """Dict that raises on get()."""

                @override
                def __getitem__(self, key: str) -> Never:
                    """Raise error on get attempt."""
                    msg = "Bad dict get"
                    raise RuntimeError(msg)

            class BadList(UserList[t.Tests.object]):
                """List that raises on iteration."""

                @override
                def __iter__(self) -> Iterator[t.Tests.object]:
                    """Raise error on iteration."""
                    msg = "Bad list iteration"
                    raise RuntimeError(msg)

            class BadModelDump:
                """Object with model_dump that raises."""

                model_dump: Callable[[], dict[str, t.Tests.object]] = staticmethod(
                    lambda: (_ for _ in ()).throw(RuntimeError("Bad model_dump")),
                )

            class BadConfig:
                """Config object that raises on attribute access."""

                def get_attribute(self, name: str) -> Never:
                    """Raise error on attribute access."""
                    msg = f"Bad config: {name}"
                    raise AttributeError(msg)

        class CoreAssertions:
            """Assertion helpers for test validation - flext-core specific."""

            @staticmethod
            def assert_failure(
                result: r[t.Container],
                expected_error: str,
                description: str = "",
            ) -> None:
                """Assert that result is a failure with expected error.

                Args:
                    result: r to check
                    expected_error: Expected error substring
                    description: Test case description for error messages

                """
                _ = assertion_helpers.assert_flext_result_failure(
                    result,
                    description,
                    error_contains=expected_error,
                )

            @staticmethod
            def assert_success(
                result: r[t.Container],
                description: str = "",
            ) -> None:
                """Assert that result is a success.

                Args:
                    result: r to check
                    description: Test case description for error messages

                """
                _ = assertion_helpers.assert_flext_result_success(
                    result,
                    f"Expected success for: {description}, got: {result.error}",
                )

            @staticmethod
            def assert_success_with_value(
                result: r[t.Container],
                expected_value: t.Container,
                description: str = "",
            ) -> None:
                """Assert result is success with specific value.

                Args:
                    result: r to check
                    expected_value: Expected value
                    description: Test case description for error messages

                """
                _ = assertion_helpers.assert_flext_result_success(
                    result,
                    f"Expected success for: {description}, got: {result.error}",
                )
                assert result.value == expected_value, (
                    f"Expected {expected_value}, got {result.value} for: {description}"
                )


u = TestsFlextUtilities

__all__ = ["TestsFlextUtilities", "u"]
