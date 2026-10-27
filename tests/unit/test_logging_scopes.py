"""Behavior contract for logging context scopes — public API only.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import c, t, tm, u


class TestsFlextLoggingScopes:
    """Binding and clearing logging context scopes through ``u``."""

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
        scope: str,
        context: t.MappingKV[str, str],
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
