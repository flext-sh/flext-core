"""Behavioral tests for the callable-handler public contract.

Every test asserts observable public behavior of ``h.create_from_callable``
and the handler it returns (``handler_name``, ``mode``, ``handle``,
``execute`` and their ``r[T]`` outcomes) -- never internal state.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

import pytest
from flext_tests import h, r, tm
from hypothesis import given, strategies as st

import tests.utilities
from tests import c, t

_TOKENS: st.SearchStrategy[str] = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
)


class TestsFlextCoreHandlersProperties(
    tests.utilities.TestsFlextUtilities.TestsFlextFlextHandlers,
):
    """Public-contract behavior of the callable handler factory."""

    @staticmethod
    @given(_TOKENS)
    def test_explicit_name_is_exposed_verbatim(handler_name: str) -> None:
        """Any non-empty explicit name is exposed by ``handler_name``."""
        handler = h.create_from_callable(str, handler_name=handler_name)

        tm.that(handler.handler_name, eq=handler_name)

    @staticmethod
    @given(_TOKENS)
    def test_handle_wraps_plain_return_value_as_success(message: str) -> None:
        """``handle`` wraps a callable's plain return in a success result."""
        handler = h.create_from_callable(str, handler_name="echo")

        tm.ok(handler.handle(message), eq=message)

    @staticmethod
    @given(_TOKENS)
    def test_execute_runs_pipeline_and_returns_success(message: str) -> None:
        """``execute`` drives the full pipeline to a success outcome."""
        handler = h.create_from_callable(str, handler_name="echo")

        tm.ok(handler.execute(message), eq=message)

    @staticmethod
    def test_default_name_falls_back_to_callable_dunder_name() -> None:
        """When no name is given, ``handler_name`` uses the callable name."""

        def my_named_handler(message: t.Scalar) -> t.Scalar:
            return message

        handler = h.create_from_callable(my_named_handler)

        tm.that(handler.handler_name, eq="my_named_handler")

    @staticmethod
    def test_callable_returning_result_is_passed_through() -> None:
        """A callable already returning ``r[T]`` is not double-wrapped."""

        def result_handler(message: t.Scalar) -> t.Scalar:
            payload = message.decode() if isinstance(message, bytes) else message
            return (
                r[t.Scalar]
                .ok(
                    f"pre_{payload}",
                )
                .value
            )

        handler = h.create_from_callable(result_handler, handler_name="pre")

        tm.ok(handler.handle("x"), eq="pre_x")

    @staticmethod
    def test_raising_callable_yields_failure_not_exception() -> None:
        """A raising callable surfaces as a failure result, never a raise."""

        def boom(message: t.Scalar) -> t.Scalar:
            _ = message
            raise ValueError(c.Tests.VALIDATION_FAILED_FOR_TEST)

        handler = h.create_from_callable(boom, handler_name="boom")

        result = handler.handle("x")

        tm.fail(result, contains=c.Tests.VALIDATION_FAILED_FOR_TEST)

    @pytest.mark.parametrize(
        "handler_type",
        [
            c.HandlerType.COMMAND,
            c.HandlerType.QUERY,
            c.HandlerType.EVENT,
            c.HandlerType.SAGA,
        ],
    )
    @staticmethod
    def test_handler_type_is_reflected_in_mode(
        handler_type: c.HandlerType,
    ) -> None:
        """The requested handler type becomes the handler's public mode."""
        handler = h.create_from_callable(
            str,
            handler_name="typed",
            handler_type=handler_type,
        )

        tm.that(handler.mode, eq=handler_type)

    @staticmethod
    def test_invalid_handler_type_is_rejected() -> None:
        """A handler type outside the HandlerType enum is rejected."""
        with pytest.raises(c.ValidationError):
            h.create_from_callable(
                str,
                handler_name="bad",
                handler_type=cast("c.HandlerType", "unknown-mode"),
            )
