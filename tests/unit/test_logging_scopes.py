"""Behavior contract for logging context scopes — public API only.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import c, m, p, t, tm, u


class TestsFlextLoggingScopes:
    """Binding and clearing logging context scopes through ``u``."""

    @staticmethod
    def test_nested_scope_restores_outer_value_after_failure() -> None:
        """An inner operation cannot erase the outer operation's correlation."""
        events: list[p.LoggingRecord] = []
        u.configure_structlog(
            settings=m.StructlogOptions(observers=(events.append,)),
        )
        tm.ok(u.clear_global_context())
        logger = u.create_module_logger(__name__)

        def inner_operation() -> None:
            """Emit and fail inside a nested correlation scope.

            Raises:
                ValueError: To exercise exception-safe scope restoration.

            """
            with u.logging_scope("operation", run_id="inner"):
                logger.info("inner")
                msg = "inner failed"
                raise ValueError(msg)

        try:
            with u.logging_scope("operation", run_id="outer"):
                logger.info("outer")
                with u.Tests.raises(ValueError, match="inner failed"):
                    inner_operation()
                logger.info("restored")
            logger.info("outside")
            tm.that(
                [event.context.get("run_id") for event in events],
                eq=["outer", "inner", "outer", None],
            )
        finally:
            tm.ok(u.clear_global_context())
            u.configure_structlog(settings=m.StructlogOptions())

    @staticmethod
    @pytest.mark.parametrize(
        ("scope", "context"),
        [
            (c.ContextScope.APPLICATION, {"app_name": "test-app"}),
            (c.ContextScope.REQUEST, {c.ContextKey.CORRELATION_ID: "flext-123"}),
            (c.ContextScope.OPERATION, {c.ContextScope.OPERATION: "sync-users"}),
        ],
    )
    def test_clear_scope_returns_success_for_bound_scopes(
        scope: str, context: t.JsonMapping
    ) -> None:
        """Clearing a bound scope succeeds."""
        bind_result = u.bind_context(scope=scope, **context)
        tm.ok(bind_result.success)
        tm.ok(bind_result.value)

        result = u.clear_scope(scope)

        tm.ok(result.success)
        tm.ok(result.value)

    @staticmethod
    def test_clear_scope_ignores_unknown_scope() -> None:
        """Clearing a scope that was never bound succeeds."""
        result = u.clear_scope("nonexistent")

        tm.ok(result.success)
        tm.ok(result.value)
