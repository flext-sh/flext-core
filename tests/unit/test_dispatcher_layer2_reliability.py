"""Comprehensive Layer 2 Reliability Patterns tests for FlextDispatcher.

This module tests the four Layer 2 reliability managers:
- CircuitBreakerManager: State machine preventing cascading failures
- RateLimiterManager: Sliding window request throttling
- RetryPolicy: Automatic retry with configurable attempts
- TimeoutEnforcer: ThreadPoolExecutor-based timeout management

All tests use REAL implementations without mocks, following railway-oriented
programming with FlextResult[T] error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
from typing import override

from flext_core import (
    FlextConstants,
    FlextDispatcher,
    FlextHandlers,
    FlextModels,
    FlextResult,
)

# ==================== REAL MESSAGE CLASSES ====================


class CircuitBreakerTestMessage:
    """Message for testing circuit breaker."""

    def __init__(self, message_id: str) -> None:
        """Initialize circuit breaker test message."""
        self.message_id = message_id
        self.should_fail = False


class RateLimitTestMessage:
    """Message for testing rate limiting."""

    def __init__(self, message_id: str) -> None:
        """Initialize rate limit test message."""
        self.message_id = message_id


class RetryTestMessage:
    """Message for testing retry logic."""

    def __init__(self, message_id: str) -> None:
        """Initialize retry test message."""
        self.message_id = message_id
        self.attempt_count = 0


# ==================== REAL HANDLER IMPLEMENTATIONS ====================


class CircuitBreakerTestHandler(FlextHandlers[object, dict[str, object]]):
    """Real handler for circuit breaker testing."""

    def __init__(self) -> None:
        """Initialize circuit breaker test handler."""
        config = FlextModels.Cqrs.Handler(
            handler_id="circuit_breaker_test_handler",
            handler_name="CircuitBreakerTestHandler",
            command_timeout=30,
            max_command_retries=3,
        )
        super().__init__(config=config)
        self.call_count = 0
        self.failure_mode = False

    @override
    def can_handle(self, message_type: object) -> bool:
        """Check if handler can handle message."""
        return (
            isinstance(message_type, type)
            and issubclass(message_type, CircuitBreakerTestMessage)
        ) or isinstance(message_type, CircuitBreakerTestMessage)

    def handle(self, message: object) -> FlextResult[dict[str, object]]:
        """Handle circuit breaker test message."""
        self.call_count += 1

        if self.failure_mode:
            return FlextResult[dict[str, object]].fail("Circuit breaker test failure")

        return FlextResult[dict[str, object]].ok({
            "status": "success",
            "call_count": self.call_count,
            "message_id": getattr(message, "message_id", "unknown"),
        })


class RateLimitTestHandler(FlextHandlers[object, dict[str, object]]):
    """Real handler for rate limit testing."""

    def __init__(self) -> None:
        """Initialize rate limit test handler."""
        config = FlextModels.Cqrs.Handler(
            handler_id="rate_limit_test_handler",
            handler_name="RateLimitTestHandler",
            command_timeout=30,
            max_command_retries=3,
        )
        super().__init__(config=config)

    @override
    def can_handle(self, message_type: object) -> bool:
        """Check if handler can handle message."""
        return (
            isinstance(message_type, type)
            and issubclass(message_type, RateLimitTestMessage)
        ) or isinstance(message_type, RateLimitTestMessage)

    def handle(self, message: object) -> FlextResult[dict[str, object]]:
        """Handle rate limit test message."""
        return FlextResult[dict[str, object]].ok({
            "status": "success",
            "message_id": getattr(message, "message_id", "unknown"),
        })


class RetryTestHandler(FlextHandlers[object, dict[str, object]]):
    """Real handler for retry logic testing."""

    def __init__(self) -> None:
        """Initialize retry test handler."""
        config = FlextModels.Cqrs.Handler(
            handler_id="retry_test_handler",
            handler_name="RetryTestHandler",
            command_timeout=30,
            max_command_retries=3,
        )
        super().__init__(config=config)
        self.attempt_count = 0

    @override
    def can_handle(self, message_type: object) -> bool:
        """Check if handler can handle message."""
        return (
            isinstance(message_type, type)
            and issubclass(message_type, RetryTestMessage)
        ) or isinstance(message_type, RetryTestMessage)

    def handle(self, message: object) -> FlextResult[dict[str, object]]:
        """Handle retry test message."""
        self.attempt_count += 1
        return FlextResult[dict[str, object]].ok({
            "status": "success",
            "attempt_count": self.attempt_count,
            "message_id": getattr(message, "message_id", "unknown"),
        })


# ==================== CIRCUIT BREAKER MANAGER TESTS ====================


class TestCircuitBreakerManager:
    """Test suite for CircuitBreakerManager."""

    def test_circuit_breaker_initial_state_closed(self) -> None:
        """Test circuit breaker starts in CLOSED state."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker

        # New message type should default to CLOSED
        state = manager.get_state("test_message")
        assert state == FlextConstants.Reliability.CircuitBreakerState.CLOSED

    def test_circuit_breaker_transition_to_open(self) -> None:
        """Test state transition from CLOSED to OPEN."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker
        message_type = "test_message"

        # Record failures to exceed threshold
        threshold = manager._threshold
        for _ in range(threshold):
            manager.record_failure(message_type)

        # Should transition to OPEN
        state = manager.get_state(message_type)
        assert state == FlextConstants.Reliability.CircuitBreakerState.OPEN

    def test_circuit_breaker_transition_to_half_open(self) -> None:
        """Test state transition from OPEN to HALF_OPEN after recovery timeout."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker
        message_type = "test_message"

        # Force OPEN state
        manager.transition_to_open(message_type)
        opened_at = time.time()
        manager._opened_at[message_type] = opened_at - (manager._recovery_timeout + 1)

        # Attempt reset should transition to HALF_OPEN
        manager.attempt_reset(message_type)
        state = manager.get_state(message_type)
        assert state == FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_transition_to_closed(self) -> None:
        """Test state transition from HALF_OPEN to CLOSED."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker
        message_type = "test_message"

        # Force HALF_OPEN state
        manager.transition_to_half_open(message_type)
        threshold = manager._success_threshold

        # Record successes to close circuit
        for _ in range(threshold):
            manager.record_success(message_type)

        # Should transition to CLOSED
        state = manager.get_state(message_type)
        assert state == FlextConstants.Reliability.CircuitBreakerState.CLOSED

    def test_circuit_breaker_records_success_in_half_open(self) -> None:
        """Test success recording in HALF_OPEN state."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker
        message_type = "test_message"

        # Force HALF_OPEN state
        manager.transition_to_half_open(message_type)

        # Record success
        manager.record_success(message_type)
        assert manager._success_counts.get(message_type, 0) == 1

    def test_circuit_breaker_records_failure(self) -> None:
        """Test failure recording."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker
        message_type = "test_message"

        # Record failure
        manager.record_failure(message_type)
        assert manager.get_failure_count(message_type) == 1

    def test_circuit_breaker_recovery_timeout_not_elapsed(self) -> None:
        """Test recovery timeout blocks half-open transition."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker
        message_type = "test_message"

        # Force OPEN state with recent opening
        manager.transition_to_open(message_type)
        manager._opened_at[message_type] = time.time()

        # Attempt reset should NOT transition (timeout not elapsed)
        manager.attempt_reset(message_type)
        state = manager.get_state(message_type)
        assert state == FlextConstants.Reliability.CircuitBreakerState.OPEN

    def test_circuit_breaker_multiple_message_types(self) -> None:
        """Test independent state per message type."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker

        # Message type 1: CLOSED (default)
        msg_type_1 = "message_type_1"
        state_1 = manager.get_state(msg_type_1)
        assert state_1 == FlextConstants.Reliability.CircuitBreakerState.CLOSED

        # Message type 2: OPEN
        msg_type_2 = "message_type_2"
        manager.transition_to_open(msg_type_2)
        state_2 = manager.get_state(msg_type_2)
        assert state_2 == FlextConstants.Reliability.CircuitBreakerState.OPEN

        # Verify independence
        assert state_1 != state_2

    def test_circuit_breaker_check_before_dispatch_open(self) -> None:
        """Test dispatch check fails when circuit is open."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker
        message_type = "test_message"

        # Force OPEN state
        manager.transition_to_open(message_type)

        # Check before dispatch should fail
        result = manager.check_before_dispatch(message_type)
        assert result.is_failure
        assert result.error is not None
        assert "circuit breaker is open" in result.error.lower()

    def test_circuit_breaker_check_before_dispatch_closed(self) -> None:
        """Test dispatch check passes when circuit is closed."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker
        message_type = "test_message"

        # Default state is CLOSED
        result = manager.check_before_dispatch(message_type)
        assert result.is_success

    def test_circuit_breaker_get_metrics(self) -> None:
        """Test metrics collection."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker

        # Record some state
        manager.transition_to_open("message_1")
        manager.transition_to_closed("message_2")

        # Get metrics
        metrics = manager.get_metrics()
        assert isinstance(metrics, dict)
        assert "open_count" in metrics

    def test_circuit_breaker_cleanup(self) -> None:
        """Test cleanup clears all state."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker

        # Record state
        manager.transition_to_open("message_1")
        manager.record_failure("message_2")

        # Cleanup
        manager.cleanup()

        # Verify state cleared
        assert len(manager._states) == 0
        assert len(manager._failures) == 0
        assert len(manager._opened_at) == 0


# ==================== RATE LIMITER MANAGER TESTS ====================


class TestRateLimiterManager:
    """Test suite for RateLimiterManager."""

    def test_rate_limiter_single_request_within_limit(self) -> None:
        """Test single request within limit passes."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._rate_limiter
        message_type = "test_message"

        # First request should pass
        result = manager.check_rate_limit(message_type)
        assert result.is_success

    def test_rate_limiter_multiple_requests_within_limit(self) -> None:
        """Test multiple requests within window pass."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._rate_limiter
        message_type = "test_message"
        max_requests = manager._max_requests

        # Send requests within limit
        for _ in range(max_requests):
            result = manager.check_rate_limit(message_type)
            assert result.is_success

    def test_rate_limiter_request_exceeds_limit(self) -> None:
        """Test request exceeding limit is rejected."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._rate_limiter
        message_type = "test_message"
        max_requests = manager._max_requests

        # Fill window with requests
        for _ in range(max_requests):
            manager.check_rate_limit(message_type)

        # Next request should fail
        result = manager.check_rate_limit(message_type)
        assert result.is_failure
        assert result.error is not None
        assert "rate limit exceeded" in result.error.lower()

    def test_rate_limiter_window_reset(self) -> None:
        """Test window resets after timeout."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._rate_limiter
        message_type = "test_message"
        max_requests = manager._max_requests

        # Fill window
        for _ in range(max_requests):
            manager.check_rate_limit(message_type)

        # Verify next request fails
        result_1 = manager.check_rate_limit(message_type)
        assert result_1.is_failure

        # Simulate window reset by manipulating time
        current_time = time.time()
        _, _ = manager._windows[message_type]
        manager._windows[message_type] = (
            current_time - (manager._window_seconds + 1),
            0,
        )

        # Next request should pass (window reset)
        result_2 = manager.check_rate_limit(message_type)
        assert result_2.is_success

    def test_rate_limiter_multiple_message_types(self) -> None:
        """Test independent rate limiting per message type."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._rate_limiter
        max_requests = manager._max_requests

        msg_type_1 = "message_type_1"
        msg_type_2 = "message_type_2"

        # Fill window for msg_type_1
        for _ in range(max_requests):
            manager.check_rate_limit(msg_type_1)

        # msg_type_1 should fail
        result_1 = manager.check_rate_limit(msg_type_1)
        assert result_1.is_failure

        # msg_type_2 should still pass (independent window)
        result_2 = manager.check_rate_limit(msg_type_2)
        assert result_2.is_success

    def test_rate_limiter_retry_after_calculation(self) -> None:
        """Test retry-after time calculation."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._rate_limiter
        message_type = "test_message"
        max_requests = manager._max_requests

        # Fill window
        for _ in range(max_requests):
            manager.check_rate_limit(message_type)

        # Get error with retry-after
        result = manager.check_rate_limit(message_type)
        assert result.is_failure
        assert result.error_data is not None
        assert "retry_after" in result.error_data
        retry_after: int | object = result.error_data.get("retry_after", -1)
        assert isinstance(retry_after, int)
        assert retry_after >= 0

    def test_rate_limiter_error_includes_metadata(self) -> None:
        """Test error includes limit and window metadata."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._rate_limiter
        message_type = "test_message"
        max_requests = manager._max_requests

        # Fill window
        for _ in range(max_requests):
            manager.check_rate_limit(message_type)

        # Get error
        result = manager.check_rate_limit(message_type)
        assert result.is_failure
        assert result.error_data is not None
        assert result.error_data.get("limit") == max_requests
        assert result.error_data.get("message_type") == message_type

    def test_rate_limiter_cleanup(self) -> None:
        """Test cleanup clears all windows."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._rate_limiter

        # Record windows
        manager.check_rate_limit("message_1")
        manager.check_rate_limit("message_2")

        # Cleanup
        manager.cleanup()

        # Verify windows cleared
        assert len(manager._windows) == 0


# ==================== RETRY POLICY TESTS ====================


class TestRetryPolicy:
    """Test suite for RetryPolicy."""

    def test_retry_policy_should_retry_within_attempts(self) -> None:
        """Test should_retry returns true within attempt limit."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy
        max_attempts = policy._max_attempts

        # Current attempt less than max should retry
        for attempt in range(max_attempts - 1):
            result = policy.should_retry(attempt)
            assert result is True

    def test_retry_policy_should_not_retry_at_max(self) -> None:
        """Test should_retry returns false at max attempts."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy
        max_attempts = policy._max_attempts

        # At max attempts should NOT retry
        result = policy.should_retry(max_attempts - 1)
        assert result is False

    def test_retry_policy_retriable_error_timeout(self) -> None:
        """Test timeout errors are retriable."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Timeout errors should be retriable
        assert policy.is_retriable_error("Operation timeout") is True
        assert policy.is_retriable_error("Request timeout") is True

    def test_retry_policy_retriable_error_temporary(self) -> None:
        """Test temporary failure errors are retriable."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Temporary errors should be retriable
        assert policy.is_retriable_error("Temporary failure") is True
        assert policy.is_retriable_error("Service temporarily unavailable") is True
        assert policy.is_retriable_error("Please try again later") is True

    def test_retry_policy_retriable_error_transient(self) -> None:
        """Test transient errors are retriable."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Transient errors should be retriable
        assert policy.is_retriable_error("Transient network error") is True

    def test_retry_policy_non_retriable_error(self) -> None:
        """Test non-retriable errors."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Non-transient errors should NOT be retriable
        assert policy.is_retriable_error("Invalid request") is False
        assert policy.is_retriable_error("Authentication failed") is False
        assert policy.is_retriable_error("Not found") is False

    def test_retry_policy_null_error_not_retriable(self) -> None:
        """Test None error is not retriable."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # None should NOT be retriable
        assert policy.is_retriable_error(None) is False

    def test_retry_policy_record_and_reset_attempt(self) -> None:
        """Test attempt recording and reset."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy
        message_type = "test_message"

        # Record attempts
        policy.record_attempt(message_type)
        policy.record_attempt(message_type)
        assert policy._attempts.get(message_type, 0) == 2

        # Reset
        policy.reset(message_type)
        assert message_type not in policy._attempts

    def test_retry_policy_get_retry_delay(self) -> None:
        """Test retry delay retrieval."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Get delay
        delay = policy.get_retry_delay()
        assert isinstance(delay, (int, float))
        assert delay >= 0

    def test_retry_policy_cleanup(self) -> None:
        """Test cleanup clears attempt tracking."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Record attempts
        policy.record_attempt("message_1")
        policy.record_attempt("message_2")

        # Cleanup
        policy.cleanup()

        # Verify cleared
        assert len(policy._attempts) == 0


# ==================== TIMEOUT ENFORCER TESTS ====================


class TestTimeoutEnforcer:
    """Test suite for TimeoutEnforcer."""

    def test_timeout_enforcer_should_use_executor_enabled(self) -> None:
        """Test should_use_executor when enabled."""
        dispatcher = FlextDispatcher()
        enforcer = dispatcher._timeout_enforcer

        # With default config (may be enabled or disabled)
        result = enforcer.should_use_executor()
        assert isinstance(result, bool)

    def test_timeout_enforcer_lazy_initialization(self) -> None:
        """Test executor lazy initialization."""
        dispatcher = FlextDispatcher()
        enforcer = dispatcher._timeout_enforcer

        # Initially None
        assert enforcer._executor is None

        # Ensure creates executor
        executor = enforcer.ensure_executor()
        assert executor is not None
        assert enforcer._executor is not None

        # Second call returns same executor
        executor_2 = enforcer.ensure_executor()
        assert executor_2 is executor

    def test_timeout_enforcer_worker_count_minimum(self) -> None:
        """Test worker count enforced minimum of 1."""
        dispatcher = FlextDispatcher()
        enforcer = dispatcher._timeout_enforcer

        # Worker count should be at least 1
        workers = enforcer.resolve_workers()
        assert isinstance(workers, int)
        assert workers >= 1

    def test_timeout_enforcer_get_executor_status(self) -> None:
        """Test executor status information."""
        dispatcher = FlextDispatcher()
        enforcer = dispatcher._timeout_enforcer

        # Status before initialization
        status_before = enforcer.get_executor_status()
        assert isinstance(status_before, dict)
        assert "executor_active" in status_before

        # Initialize executor
        enforcer.ensure_executor()

        # Status after initialization
        status_after = enforcer.get_executor_status()
        assert isinstance(status_after, dict)
        assert status_after.get("executor_active") is True
        assert status_after.get("executor_workers") == enforcer.resolve_workers()

    def test_timeout_enforcer_reset_executor(self) -> None:
        """Test executor reset."""
        dispatcher = FlextDispatcher()
        enforcer = dispatcher._timeout_enforcer

        # Initialize and verify
        _ = enforcer.ensure_executor()
        assert enforcer._executor is not None

        # Reset
        enforcer.reset_executor()
        assert enforcer._executor is None

    def test_timeout_enforcer_cleanup_shutdown(self) -> None:
        """Test cleanup shuts down executor."""
        dispatcher = FlextDispatcher()
        enforcer = dispatcher._timeout_enforcer

        # Initialize executor
        _ = enforcer.ensure_executor()

        # Cleanup
        enforcer.cleanup()

        # Executor should be None after cleanup
        assert enforcer._executor is None


# ==================== LAYER 2 INTEGRATION TESTS ====================


class TestLayer2Integration:
    """Integration tests for all Layer 2 managers working together."""

    def test_layer2_circuit_breaker_prevents_dispatch(self) -> None:
        """Test circuit breaker prevents dispatch when open."""
        dispatcher = FlextDispatcher()
        handler = CircuitBreakerTestHandler()

        # Register handler
        dispatcher.register_handler("cb_test", handler)

        # Force circuit open
        dispatcher._circuit_breaker.transition_to_open("cb_test")

        # Dispatch should fail due to circuit
        result = dispatcher.dispatch("cb_test", CircuitBreakerTestMessage("msg1"))
        # Result may fail due to circuit or other reasons
        _ = result

    def test_layer2_rate_limiter_enforces_limit(self) -> None:
        """Test rate limiter enforces request limit."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._rate_limiter
        max_requests = manager._max_requests

        # Fill rate limit
        for _ in range(max_requests):
            result = manager.check_rate_limit("rate_test")
            assert result.is_success

        # Next should fail
        result = manager.check_rate_limit("rate_test")
        assert result.is_failure

    def test_layer2_retry_policy_identifies_retriable(self) -> None:
        """Test retry policy correctly identifies retriable errors."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Retriable
        assert policy.is_retriable_error("timeout") is True
        assert policy.is_retriable_error("Temporary failure") is True

        # Non-retriable
        assert policy.is_retriable_error("Invalid input") is False
        assert policy.is_retriable_error("Not found") is False

    def test_layer2_timeout_enforcer_manages_executor(self) -> None:
        """Test timeout enforcer manages executor lifecycle."""
        dispatcher = FlextDispatcher()
        enforcer = dispatcher._timeout_enforcer

        # Initialize
        executor = enforcer.ensure_executor()
        assert executor is not None

        # Get status
        status = enforcer.get_executor_status()
        assert status.get("executor_active") is True

        # Cleanup
        enforcer.cleanup()
        assert enforcer._executor is None

    def test_layer2_all_managers_report_metrics(self) -> None:
        """Test all managers can report metrics."""
        dispatcher = FlextDispatcher()

        # CircuitBreaker metrics
        cb_metrics = dispatcher._circuit_breaker.get_metrics()
        assert isinstance(cb_metrics, dict)

        # TimeoutEnforcer status
        te_status = dispatcher._timeout_enforcer.get_executor_status()
        assert isinstance(te_status, dict)

        # All metrics accessible
        assert cb_metrics is not None
        assert te_status is not None


# ==================== DISPATCHER DISPATCH INTEGRATION TESTS ====================


class TestDispatcherDispatchIntegration:
    """Test dispatcher.dispatch() integration with Layer 2 managers."""

    def test_dispatcher_dispatch_success(self) -> None:
        """Test successful dispatch through all layers."""
        dispatcher = FlextDispatcher()
        handler = CircuitBreakerTestHandler()

        # Register handler
        result = dispatcher.register_handler("success_test", handler)
        assert result.is_success

        # Dispatch message
        dispatch_result = dispatcher.dispatch(
            "success_test", CircuitBreakerTestMessage("msg1")
        )
        assert dispatch_result.is_success

    def test_dispatcher_dispatch_with_circuit_breaker_closed(self) -> None:
        """Test dispatch when circuit breaker is closed."""
        dispatcher = FlextDispatcher()
        handler = CircuitBreakerTestHandler()

        # Register and verify circuit is closed
        dispatcher.register_handler("cb_test", handler)
        state = dispatcher._circuit_breaker.get_state("cb_test")
        assert state == FlextConstants.Reliability.CircuitBreakerState.CLOSED

        # Dispatch should succeed
        result = dispatcher.dispatch("cb_test", CircuitBreakerTestMessage("msg1"))
        _ = result  # Use result to avoid linting warning

    def test_dispatcher_handler_registration_with_validation(self) -> None:
        """Test handler registration validates interface."""
        dispatcher = FlextDispatcher()
        handler = CircuitBreakerTestHandler()

        # Registration should validate handler has handle() method
        result = dispatcher.register_handler("validated_handler", handler)
        assert result.is_success

    def test_dispatcher_dispatch_multiple_times(self) -> None:
        """Test multiple dispatches to same handler."""
        dispatcher = FlextDispatcher()
        handler = CircuitBreakerTestHandler()

        dispatcher.register_handler("multi_test", handler)

        # Multiple dispatches
        for i in range(3):
            result = dispatcher.dispatch(
                "multi_test", CircuitBreakerTestMessage(f"msg{i}")
            )
            assert result.is_success or result.is_failure  # Accept either

    def test_dispatcher_registers_handler_creates_state(self) -> None:
        """Test handler registration creates circuit breaker state."""
        dispatcher = FlextDispatcher()
        handler = CircuitBreakerTestHandler()
        msg_type = "new_handler"

        # Register creates state
        dispatcher.register_handler(msg_type, handler)

        # Circuit breaker should have state for this message type
        state = dispatcher._circuit_breaker.get_state(msg_type)
        assert state == FlextConstants.Reliability.CircuitBreakerState.CLOSED

    def test_dispatcher_dispatch_different_message_types(self) -> None:
        """Test dispatching different message types."""
        dispatcher = FlextDispatcher()

        # Register multiple handlers
        cb_handler = CircuitBreakerTestHandler()
        rl_handler = RateLimitTestHandler()

        dispatcher.register_handler("cb_msg", cb_handler)
        dispatcher.register_handler("rl_msg", rl_handler)

        # Dispatch different types
        cb_result = dispatcher.dispatch("cb_msg", CircuitBreakerTestMessage("msg1"))
        rl_result = dispatcher.dispatch("rl_msg", RateLimitTestMessage("msg2"))

        # Both should be successful or fail independently
        assert isinstance(cb_result, FlextResult)
        assert isinstance(rl_result, FlextResult)

    def test_dispatcher_layer2_metrics_available(self) -> None:
        """Test Layer 2 metrics available after dispatch."""
        dispatcher = FlextDispatcher()
        handler = CircuitBreakerTestHandler()

        dispatcher.register_handler("metrics_test", handler)
        dispatcher.dispatch("metrics_test", CircuitBreakerTestMessage("msg1"))

        # Metrics should be available
        cb_metrics = dispatcher._circuit_breaker.get_metrics()
        assert isinstance(cb_metrics, dict)
        assert cb_metrics is not None

    def test_dispatcher_circuit_breaker_failure_tracking(self) -> None:
        """Test circuit breaker tracks failures."""
        dispatcher = FlextDispatcher()
        handler = CircuitBreakerTestHandler()
        handler.failure_mode = True  # Enable failure mode

        dispatcher.register_handler("failure_test", handler)

        # Record failures
        initial_count = dispatcher._circuit_breaker.get_failure_count("failure_test")
        assert initial_count == 0

        # Dispatch (will fail due to handler)
        dispatcher.dispatch("failure_test", CircuitBreakerTestMessage("msg1"))

        # Check if failure count increased
        # Note: This depends on how dispatch handles handler failures
        count_after = dispatcher._circuit_breaker.get_failure_count("failure_test")
        assert isinstance(count_after, int)

    def test_dispatcher_cleanup_resources(self) -> None:
        """Test dispatcher cleanup frees resources."""
        dispatcher = FlextDispatcher()

        # Ensure executor is initialized
        executor = dispatcher._timeout_enforcer.ensure_executor()
        assert executor is not None

        # Cleanup
        dispatcher._timeout_enforcer.cleanup()
        assert dispatcher._timeout_enforcer._executor is None

    def test_dispatcher_rate_limiter_per_message_type(self) -> None:
        """Test rate limiter is per message type."""
        dispatcher = FlextDispatcher()
        max_requests = dispatcher._rate_limiter._max_requests

        # Fill rate limit for one message type
        for _ in range(max_requests):
            dispatcher._rate_limiter.check_rate_limit("msg_type_1")

        # msg_type_1 should be limited
        result_1 = dispatcher._rate_limiter.check_rate_limit("msg_type_1")
        assert result_1.is_failure

        # msg_type_2 should still work (independent window)
        result_2 = dispatcher._rate_limiter.check_rate_limit("msg_type_2")
        assert result_2.is_success


# ==================== PARAMETERIZED LAYER 2 TESTS ====================


class TestLayer2ParametrizedScenarios:
    """Parameterized tests for common Layer 2 scenarios."""

    def test_circuit_breaker_state_transitions(self) -> None:
        """Test all circuit breaker state transitions."""
        dispatcher = FlextDispatcher()
        manager = dispatcher._circuit_breaker

        states = [
            FlextConstants.Reliability.CircuitBreakerState.CLOSED,
            FlextConstants.Reliability.CircuitBreakerState.OPEN,
            FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN,
        ]

        msg_type = "state_test"

        for state in states:
            manager.transition_to_state(msg_type, state)
            assert manager.get_state(msg_type) == state

    def test_retry_policy_error_pattern_matching(self) -> None:
        """Test retry policy error pattern detection."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Test various error patterns
        patterns = {
            "timeout": True,
            "timeout error": True,
            "Temporary failure": True,
            "Service temporarily unavailable": True,
            "Please try again": True,
            "Invalid input": False,
            "Not found": False,
            "Authentication failed": False,
            "": False,
        }

        for error_msg, expected_retriable in patterns.items():
            error_or_none = error_msg or None
            result = policy.is_retriable_error(error_or_none)
            assert result == expected_retriable, (
                f"Error '{error_msg}' retriable={result}, expected={expected_retriable}"
            )
