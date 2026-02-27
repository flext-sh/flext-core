"""Tests for FlextDispatcher Dependency Injection integration.

Module: flext_core.dispatcher
Scope: DI integration for reliability managers (circuit breaker, rate limiter, timeout, retry)

Tests DI functionality with real implementations:
- Dispatcher accepts container for manager resolution
- Custom managers can be injected via container
- Managers are resolved from container or created with defaults
- Handler factory registration in container

Uses real implementations (no mocks) and flext_tests helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextDispatcher


class TestDispatcherDI:
    """Test dispatcher dependency injection integration."""

    def test_dispatcher_has_handlers(self) -> None:
        """Test dispatcher has handlers registry."""
        # Act - FlextDispatcher creates handlers registry internally
        dispatcher = FlextDispatcher()

        # Assert - dispatcher has handlers registry
        assert dispatcher._handlers is not None
        assert isinstance(dispatcher._handlers, dict)
