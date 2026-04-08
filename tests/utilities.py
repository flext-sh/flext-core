"""Utilities for flext-core tests.

Provides TestsFlextCoreUtilities, extending TestsFlextUtilities with flext-core-specific
utilities. All generic test utilities come from flext_tests.

Architecture:
- TestsFlextUtilities (flext_tests) = Generic utilities for all FLEXT projects
- TestsFlextCoreUtilities (tests/) = flext-core-specific utilities extending TestsFlextUtilities

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Callable, Iterator, Mapping
from typing import Never, override

from flext_cli import u as _cli_u
from flext_tests import FlextTestsUtilities, tm
from tests import r, t


class TestsFlextCoreUtilities(FlextTestsUtilities, _cli_u):
    """Utilities for flext-core tests - extends TestsFlextUtilities.

    Architecture: Extends TestsFlextUtilities with flext-core-specific utility
    definitions. All generic utilities from TestsFlextUtilities are available
    through inheritance.

    Rules:
    - NEVER redeclare utilities from TestsFlextUtilities
    - Only flext-core-specific utilities allowed
    - All generic utilities come from TestsFlextUtilities
    """

    class Core:
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
                    tm.fail(
                        result,
                        has=expected_error,
                        msg=f"Expected failure for: {description}, got success",
                    )
                else:
                    actual_value = tm.ok(
                        result,
                        msg=f"Expected success for: {description}, got: {result.error}",
                    )
                    if expected_value is not None:
                        tm.that(
                            actual_value,
                            eq=expected_value,
                            msg=f"Expected {expected_value}, got {actual_value} for: {description}",
                        )

        class CoreBadObjects:
            """Factory for objects that cause errors during testing - flext-core specific."""

            class BadSplitString:
                """String-like t.NormalizedValue that raises on split()."""

                def split(
                    self,
                    *_args: str,
                    **_kwargs: t.Scalar,
                ) -> t.StrSequence:
                    """Raise error on split attempt."""
                    msg = "Bad split"
                    raise RuntimeError(msg)

                @override
                def __str__(self) -> str:
                    """Return string representation."""
                    return "bad_split_string"

            @staticmethod
            def create_for_split() -> (
                TestsFlextCoreUtilities.Core.CoreBadObjects.BadSplitString
            ):
                """Create t.NormalizedValue that fails on split()."""
                return TestsFlextCoreUtilities.Core.CoreBadObjects.BadSplitString()

            class BadIndexString:
                """String-like t.NormalizedValue that raises on indexing."""

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
                TestsFlextCoreUtilities.Core.CoreBadObjects.BadIndexString
            ):
                """Create t.NormalizedValue that fails on indexing."""
                return TestsFlextCoreUtilities.Core.CoreBadObjects.BadIndexString()

            class BadStrObject:
                """Object that raises on str() conversion."""

                @override
                def __str__(self) -> str:
                    """Raise error on str() attempt."""
                    msg = "Bad str"
                    raise RuntimeError(msg)

            @staticmethod
            def create_for_str() -> (
                TestsFlextCoreUtilities.Core.CoreBadObjects.BadStrObject
            ):
                """Create t.NormalizedValue that fails on str()."""
                return TestsFlextCoreUtilities.Core.CoreBadObjects.BadStrObject()

            class BadDict(UserDict[str, t.Tests.TestobjectSerializable]):
                """Dict that raises on get()."""

                @override
                def __getitem__(self, key: str) -> Never:
                    """Raise error on get attempt."""
                    _ = key
                    msg = "Bad dict get"
                    raise RuntimeError(msg)

            class BadList(UserList[t.Tests.TestobjectSerializable]):
                """List that raises on iteration."""

                @override
                def __iter__(self) -> Iterator[t.Tests.TestobjectSerializable]:
                    """Raise error on iteration."""
                    msg = "Bad list iteration"
                    raise RuntimeError(msg)

            class BadModelDump:
                """Object with model_dump that raises."""

                model_dump: Callable[
                    [], Mapping[str, t.Tests.TestobjectSerializable]
                ] = staticmethod(
                    lambda: (_ for _ in ()).throw(RuntimeError("Bad model_dump")),
                )

            class BadConfig:
                """Config t.NormalizedValue that raises on attribute access."""

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
                tm.fail(
                    result,
                    has=expected_error,
                    msg=description,
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
                tm.ok(
                    result,
                    msg=f"Expected success for: {description}, got: {result.error}",
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
                actual_value = tm.ok(
                    result,
                    msg=f"Expected success for: {description}, got: {result.error}",
                )
                tm.that(
                    actual_value,
                    eq=expected_value,
                    msg=f"Expected {expected_value}, got {actual_value} for: {description}",
                )


u = TestsFlextCoreUtilities

__all__ = ["TestsFlextCoreUtilities", "u"]
