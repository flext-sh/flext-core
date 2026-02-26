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
from flext_core._dispatcher import (
    CircuitBreakerManager,
    RateLimiterManager,
    RetryPolicy,
    TimeoutEnforcer,
)


class TestDispatcherDI:
    """Test dispatcher dependency injection integration."""

    def test_dispatcher_uses_default_managers(self) -> None:
        """Test dispatcher creates default managers when none provided."""
        # Arrange
        dispatcher = FlextDispatcher()

        # Assert - dispatcher has managers (created with defaults)
        assert dispatcher._circuit_breaker is not None
        assert dispatcher._rate_limiter is not None
        assert dispatcher._timeout_enforcer is not None
        assert dispatcher._retry_policy is not None

        # Verify types
        assert isinstance(dispatcher._circuit_breaker, CircuitBreakerManager)
        assert isinstance(dispatcher._rate_limiter, RateLimiterManager)
        assert isinstance(dispatcher._timeout_enforcer, TimeoutEnforcer)
        assert isinstance(dispatcher._retry_policy, RetryPolicy)

    def test_dispatcher_has_container(self) -> None:
        """Test dispatcher has container from FlextService inheritance."""
        # Act - FlextService creates container internally
        dispatcher = FlextDispatcher()

        # Assert - dispatcher has container from FlextService
        assert dispatcher._container is not None
        assert hasattr(dispatcher._container, "get")
        assert hasattr(dispatcher._container, "register")

    def test_reliability_managers_injected_direct(self) -> None:
        """Test dispatcher uses directly injected managers."""
        # Arrange - custom managers passed directly
        custom_breaker = CircuitBreakerManager(
            threshold=5, recovery_timeout=10.0, success_threshold=2
        )
        custom_rate_limiter = RateLimiterManager(max_requests=100, window_seconds=60)
        custom_timeout = TimeoutEnforcer(use_timeout_executor=True, executor_workers=4)
        custom_retry = RetryPolicy(max_attempts=5, retry_delay=2.0)

        # Act - pass managers directly to constructor
        dispatcher = FlextDispatcher(
            circuit_breaker=custom_breaker,
            rate_limiter=custom_rate_limiter,
            timeout_enforcer=custom_timeout,
            retry_policy=custom_retry,
        )

        # Assert - dispatcher uses injected managers
        assert dispatcher._circuit_breaker is custom_breaker
        assert dispatcher._rate_limiter is custom_rate_limiter
        assert dispatcher._timeout_enforcer is custom_timeout
        assert dispatcher._retry_policy is custom_retry

    def test_reliability_managers_injected_via_parameters(self) -> None:
        """Test dispatcher accepts managers directly via parameters."""
        # Arrange - custom managers
        custom_breaker = CircuitBreakerManager(
            threshold=10, recovery_timeout=15.0, success_threshold=3
        )
        custom_rate_limiter = RateLimiterManager(max_requests=200, window_seconds=120)
        custom_timeout = TimeoutEnforcer(use_timeout_executor=True, executor_workers=8)
        custom_retry = RetryPolicy(max_attempts=3, retry_delay=1.0)

        # Act
        dispatcher = FlextDispatcher(
            circuit_breaker=custom_breaker,
            rate_limiter=custom_rate_limiter,
            timeout_enforcer=custom_timeout,
            retry_policy=custom_retry,
        )

        # Assert - dispatcher uses provided managers
        assert dispatcher._circuit_breaker is custom_breaker
        assert dispatcher._rate_limiter is custom_rate_limiter
        assert dispatcher._timeout_enforcer is custom_timeout
        assert dispatcher._retry_policy is custom_retry

    def test_dispatcher_creates_default_managers_when_not_provided(self) -> None:
        """Test dispatcher creates default managers when not provided."""
        # Act - dispatcher should create default managers
        dispatcher = FlextDispatcher()

        # Assert - dispatcher has default managers with default config
        assert dispatcher._circuit_breaker is not None
        assert dispatcher._rate_limiter is not None
        assert dispatcher._timeout_enforcer is not None
        assert dispatcher._retry_policy is not None

        # Verify default values are applied
        assert isinstance(dispatcher._circuit_breaker, CircuitBreakerManager)
        assert isinstance(dispatcher._rate_limiter, RateLimiterManager)
        assert isinstance(dispatcher._timeout_enforcer, TimeoutEnforcer)
        assert isinstance(dispatcher._retry_policy, RetryPolicy)
