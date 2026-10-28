"""Behavioral contract tests for the public ``FlextDecoratorsRuntime`` facade (``d``).

Railway, retry, timeout, combined, inject and logging behavior is asserted in
the dedicated decorator modules. This module keeps only the contracts owned by
no other module: deprecation, factory identity, and preserved callable metadata.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import d, tm


class TestsFlextCoreDecorators:
    """Public behavioral contract for FLEXT infrastructure decorators."""

    @staticmethod
    def test_deprecated_emits_warning_and_returns_value() -> None:
        """Deprecated warns with ``DeprecationWarning`` yet still runs the call."""

        @d.deprecated("use the replacement")
        def legacy() -> str:
            return "still works"

        with pytest.warns(DeprecationWarning, match="use the replacement"):
            tm.that(legacy(), eq="still works")

    @staticmethod
    def test_factory_returns_same_callable_that_still_runs() -> None:
        """Factory marks without wrapping: the callable identity and result hold."""

        def build() -> int:
            return 7

        marked = d.factory("builder")(build)

        tm.that(marked is build, truth=True)
        tm.that(marked(), eq=7)

    @staticmethod
    def test_decorators_preserve_callable_name() -> None:
        """Wrapping decorators keep ``__name__`` via ``functools.wraps``."""

        @d.railway()
        def named_operation() -> int:
            return 1

        tm.that(named_operation.__name__, eq="named_operation")
