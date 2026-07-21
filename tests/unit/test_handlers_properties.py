"""Behavioral tests for the callable-handler public contract.

Every test asserts observable public behavior of ``h.create_from_callable``
and the handler it returns (``handler_name``, ``mode``, ``handle``,
``execute`` and their ``r[T]`` outcomes) -- never internal state.
"""

from __future__ import annotations

from typing import cast

import pytest
from hypothesis import given, strategies as st

from flext_tests import h, r, tm
from tests import c, t
from tests.unit._handlers_support import TestsFlextFlextHandlers

_TOKENS: st.SearchStrategy[str] = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1
)


class TestsFlextCoreHandlersProperties(TestsFlextFlextHandlers):
    """Public-contract behavior of the callable handler factory."""

    @given(_TOKENS)
    def test_explicit_name_is_exposed_verbatim(self, handler_name: str) -> None:
        """Any non-empty explicit name is exposed by ``handler_name``."""
        handler = h.create_from_callable(str, handler_name=handler_name)

        tm.that(handler.handler_name, eq=handler_name)

    @given(_TOKENS)
    def test_handle_wraps_plain_return_value_as_success(self, message: str) -> None:
        """``handle`` wraps a callable's plain return in a success result."""
        handler = h.create_from_callable(str, handler_name="echo")

        tm.ok(handler.handle(message), eq=message)

    @given(_TOKENS)
    def test_execute_runs_pipeline_and_returns_success(self, message: str) -> None:
        """``execute`` drives the full pipeline to a success outcome."""
        handler = h.create_from_callable(str, handler_name="echo")

        tm.ok(handler.execute(message), eq=message)

    def test_default_name_falls_back_to_callable_dunder_name(self) -> None:
        """When no name is given, ``handler_name`` uses the callable name."""

        def my_named_handler(message: t.Scalar) -> t.Scalar:
            return message

        handler = h.create_from_callable(my_named_handler)

        tm.that(handler.handler_name, eq="my_named_handler")

    def test_callable_returning_result_is_passed_through(self) -> None:
        """A callable already returning ``r[T]`` is not double-wrapped."""

        def result_handler(message: t.Scalar) -> t.Scalar:
            return r[t.Scalar].ok(f"pre_{message}").value

        handler = h.create_from_callable(result_handler, handler_name="pre")

        tm.ok(handler.handle("x"), eq="pre_x")

    def test_raising_callable_yields_failure_not_exception(self) -> None:
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
    def test_handler_type_is_reflected_in_mode(
        self, handler_type: c.HandlerType
    ) -> None:
        """The requested handler type becomes the handler's public mode."""
        handler = h.create_from_callable(
            str, handler_name="typed", handler_type=handler_type
        )

        tm.that(handler.mode, eq=handler_type)

    def test_invalid_handler_type_is_rejected(self) -> None:
        """A handler type outside the HandlerType enum is rejected."""
        with pytest.raises(c.ValidationError):
            h.create_from_callable(
                str,
                handler_name="bad",
                handler_type=cast("c.HandlerType", "unknown-mode"),
            )
