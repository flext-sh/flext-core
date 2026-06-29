"""Handler property-based tests."""

from __future__ import annotations

from flext_tests import h, tm
from hypothesis import given, strategies as st

from tests.unit._handlers_support import TestsFlextFlextHandlers

HANDLER_TYPES = TestsFlextFlextHandlers.HANDLER_TYPES
HandlerTypeScenario = TestsFlextFlextHandlers.HandlerTypeScenario
VALIDATION_TYPES = TestsFlextFlextHandlers.VALIDATION_TYPES


class TestsFlextHandlersProperties(TestsFlextFlextHandlers):
    @given(st.text(min_size=1))
    def test_create_from_callable_hypothesis(self, handler_name: str) -> None:
        """Property: create_from_callable works with any non-empty name."""
        handler = h.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name=handler_name,
        )
        tm.that(handler.handler_name, eq=handler_name)
        tm.ok(handler.execute("x"), eq="x")
