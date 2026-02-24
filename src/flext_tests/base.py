"""Service base for FLEXT tests.

Provides two base classes:
1. FlextTestsServiceBase - Simple base for test classes (pytest-friendly)
2. FlextTestsUtilityBase - Extends FlextService for utility classes

IMPORTANT: Test classes should use FlextTestsServiceBase (alias: s) which
does NOT extend FlextService (Pydantic model) because pytest cannot collect
Pydantic models as test classes.

Utility classes (factories, builders, validators) should use
FlextTestsUtilityBase (alias: s) which extends FlextService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.result import r
from flext_core.service import FlextService
from flext_core.typings import T


class FlextTestsServiceBase[T]:
    """Base class for FLEXT test classes.

    Architecture: Simple base class providing test helper methods.
    Does NOT extend FlextService to ensure pytest can collect test classes.

    Type parameter T is used for inheritance pattern compatibility -
    test subclasses can use FlextTestsServiceBase[T] for open generic inheritance.

    Test classes inheriting from this base can use:
    - assert_success(result) -> unwrap successful results
    - assert_failure(result) -> verify failure and get error
    """

    def assert_success[TResult](self, result: r[TResult]) -> TResult:
        """Assert result is success and return unwrapped value.

        Args:
            result: FlextResult to check

        Returns:
            TResult: Unwrapped value from successful result

        Raises:
            AssertionError: If result is not successful

        """
        if not result.is_success:
            msg = f"Expected success but got failure: {result.error}"
            raise AssertionError(msg)
        return result.value

    def assert_failure[TResult](self, result: r[TResult]) -> str:
        """Assert result is failure and return error message.

        Args:
            result: FlextResult to check

        Returns:
            str: Error message from failed result

        Raises:
            AssertionError: If result is not a failure

        """
        if result.is_success:
            msg = "Expected failure but got success"
            raise AssertionError(msg)
        return result.error or ""


class FlextTestsUtilityBase(FlextService[T]):
    """Base class for FLEXT test utility classes (factories, builders, validators).

    Architecture: Extends FlextService for service functionality.
    This is NOT for test classes - use FlextTestsServiceBase for tests.

    Utility classes inheriting from this base get:
    - Full FlextService functionality
    - Generic type parameter support
    """


s = FlextTestsServiceBase


__all__ = ["FlextTestsServiceBase", "FlextTestsUtilityBase", "s"]
