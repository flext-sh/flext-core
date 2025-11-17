"""Real tests to achieve 100% dispatcher coverage - no mocks.

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in dispatcher.py.

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


class TestMessage100:
    """Simple message for 100% coverage tests."""

    def __init__(self, value: object) -> None:
        """Initialize test message."""
        self.value = value
        self.data = value


class TimeoutTestMessage:
    """Message for timeout testing."""

    def __init__(self, delay: float = 0.1) -> None:
        """Initialize timeout test message."""
        self.delay = delay


class MetadataTestMessage:
    """Message for metadata testing."""

    def __init__(self, value: str) -> None:
        """Initialize metadata test message."""
        self.value = value


# ==================== REAL HANDLER IMPLEMENTATIONS ====================


class SlowHandler(FlextHandlers[object, dict[str, object]]):
    """Handler that takes time to execute."""

    def __init__(self, delay: float = 0.1) -> None:
        """Initialize slow handler."""
        config = FlextModels.Cqrs.Handler(
            handler_id="slow_handler",
            handler_name="SlowHandler",
            command_timeout=30,
            max_command_retries=3,
        )
        super().__init__(config=config)
        self.delay = delay

    @override
    def can_handle(self, message_type: object) -> bool:
        """Check if handler can handle message."""
        return (
            isinstance(message_type, type)
            and issubclass(message_type, TimeoutTestMessage)
        ) or isinstance(message_type, TimeoutTestMessage)

    def handle(self, message: object) -> FlextResult[dict[str, object]]:
        """Handle message with delay."""
        time.sleep(self.delay)
        return FlextResult[dict[str, object]].ok({
            "status": "completed",
            "delay": getattr(message, "delay", 0),
        })


class MetadataAwareHandler(FlextHandlers[object, dict[str, object]]):
    """Handler that uses metadata."""

    def __init__(self) -> None:
        """Initialize metadata aware handler."""
        config = FlextModels.Cqrs.Handler(
            handler_id="metadata_handler",
            handler_name="MetadataAwareHandler",
            command_timeout=30,
            max_command_retries=3,
        )
        super().__init__(config=config)

    @override
    def can_handle(self, message_type: object) -> bool:
        """Check if handler can handle message."""
        return (
            isinstance(message_type, type)
            and issubclass(message_type, MetadataTestMessage)
        ) or isinstance(message_type, MetadataTestMessage)

    def handle(self, message: object) -> FlextResult[dict[str, object]]:
        """Handle message."""
        return FlextResult[dict[str, object]].ok({
            "status": "completed",
            "value": getattr(message, "value", "unknown"),
        })


# ==================== COVERAGE TESTS ====================


class TestDispatcher100Coverage:
    """Real tests to achieve 100% dispatcher coverage."""

    def test_dispatch_with_none_message_fails(self) -> None:
        """Test dispatch with None message returns failure."""
        dispatcher = FlextDispatcher()
        result = dispatcher.dispatch(None)
        assert result.is_failure
        assert "Message cannot be None" in result.error

    def test_dispatch_with_string_message_fails(self) -> None:
        """Test dispatch with string message - should fail in normalize."""
        dispatcher = FlextDispatcher()
        # String messages are caught in _normalize_dispatch_message which raises
        # But dispatch() catches and returns FlextResult, so test the internal method
        try:
            dispatcher._normalize_dispatch_message("string", None)
            msg = "Should have raised TypeError"
            raise AssertionError(msg)
        except TypeError as e:
            assert "String message_type not supported" in str(e)

    def test_dispatch_with_invalid_metadata_type(self) -> None:
        """Test dispatch with invalid metadata type."""
        dispatcher = FlextDispatcher()
        handler = MetadataAwareHandler()
        dispatcher.register_function(MetadataTestMessage, handler.handle)

        message = MetadataTestMessage("test")
        # Pass invalid metadata type (not dict)
        result = dispatcher.dispatch(message, metadata="invalid")
        assert result.is_failure
        assert "Invalid metadata type" in result.error

    def test_dispatch_with_config_object(self) -> None:
        """Test dispatch with config object."""
        dispatcher = FlextDispatcher()
        handler = MetadataAwareHandler()
        dispatcher.register_function(MetadataTestMessage, handler.handle)

        # Create config object
        class DispatchConfig:
            def __init__(self) -> None:
                self.metadata = {"key": "value"}
                self.correlation_id = "test-correlation"
                self.timeout_override = 30

        message = MetadataTestMessage("test")
        config = DispatchConfig()
        result = dispatcher.dispatch(message, config=config)
        assert result.is_success

    def test_dispatch_timeout_handling(self) -> None:
        """Test dispatch with timeout."""
        dispatcher = FlextDispatcher()
        handler = SlowHandler(delay=2.0)  # 2 second delay
        dispatcher.register_function(TimeoutTestMessage, handler.handle)

        message = TimeoutTestMessage(delay=2.0)
        # Set very short timeout
        result = dispatcher.dispatch(message, timeout_override=1)
        # Should fail due to timeout
        assert result.is_failure
        assert "timeout" in result.error.lower() or "Timeout" in result.error

    def test_circuit_breaker_recovery_half_open_to_closed(self) -> None:
        """Test circuit breaker recovery from HALF_OPEN to CLOSED."""
        dispatcher = FlextDispatcher()
        handler = MetadataAwareHandler()
        dispatcher.register_function(MetadataTestMessage, handler.handle)

        message = MetadataTestMessage("test")
        message_type = type(message).__name__

        # Force circuit to HALF_OPEN
        dispatcher._circuit_breaker.transition_to_half_open(message_type)

        # Record enough successes to close circuit
        success_threshold = dispatcher._circuit_breaker._success_threshold
        for _ in range(success_threshold):
            dispatcher._circuit_breaker.record_success(message_type)

        # Circuit should be CLOSED now
        state = dispatcher._circuit_breaker.get_state(message_type)
        assert state == FlextConstants.Reliability.CircuitBreakerState.CLOSED

    def test_circuit_breaker_recovery_half_open_to_open(self) -> None:
        """Test circuit breaker recovery failure from HALF_OPEN to OPEN."""
        dispatcher = FlextDispatcher()
        handler = MetadataAwareHandler()
        dispatcher.register_function(MetadataTestMessage, handler.handle)

        message = MetadataTestMessage("test")
        message_type = type(message).__name__

        # Force circuit to HALF_OPEN
        dispatcher._circuit_breaker.transition_to_half_open(message_type)

        # Record failure in HALF_OPEN state
        dispatcher._circuit_breaker.record_failure(message_type)

        # Circuit should be OPEN now
        state = dispatcher._circuit_breaker.get_state(message_type)
        assert state == FlextConstants.Reliability.CircuitBreakerState.OPEN

    def test_timeout_enforcer_executor_shutdown_handling(self) -> None:
        """Test timeout enforcer handles executor shutdown."""
        dispatcher = FlextDispatcher()
        enforcer = dispatcher._timeout_enforcer

        # Ensure executor exists
        executor = enforcer.ensure_executor()
        assert executor is not None

        # Shutdown executor
        executor.shutdown(wait=False)

        # Reset should recreate executor
        enforcer.reset_executor()
        new_executor = enforcer.ensure_executor()
        assert new_executor is not None

    def test_retry_policy_identifies_retriable_errors(self) -> None:
        """Test retry policy identifies retriable errors."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Test retriable errors
        assert policy.is_retriable_error("timeout") is True
        assert policy.is_retriable_error("Temporary failure") is True
        # Note: "Executor was shutdown" is handled separately in _should_retry_on_error

        # Test non-retriable errors
        assert policy.is_retriable_error("Invalid input") is False
        assert policy.is_retriable_error("Not found") is False

    def test_dispatch_config_extraction_exception_handling(self) -> None:
        """Test dispatch config extraction handles exceptions."""
        dispatcher = FlextDispatcher()

        # Create config object that raises exception on attribute access
        class BadConfig:
            @property
            def metadata(self) -> object:
                msg = "Config error"
                raise RuntimeError(msg)

        message = TestMessage100("test")
        config = BadConfig()
        result = dispatcher.dispatch(message, config=config)
        # Should handle exception gracefully
        assert result.is_failure
        assert "Configuration extraction failed" in result.error

    def test_normalize_dispatch_message_none_raises(self) -> None:
        """Test _normalize_dispatch_message raises on None."""
        dispatcher = FlextDispatcher()
        try:
            dispatcher._normalize_dispatch_message(None, None)
            msg = "Should have raised TypeError"
            raise AssertionError(msg)
        except TypeError as e:
            assert "Message cannot be None" in str(e)

    def test_normalize_dispatch_message_string_raises(self) -> None:
        """Test _normalize_dispatch_message raises on string."""
        dispatcher = FlextDispatcher()
        try:
            dispatcher._normalize_dispatch_message("string", None)
            msg = "Should have raised TypeError"
            raise AssertionError(msg)
        except TypeError as e:
            assert "String message_type not supported" in str(e)

    def test_timeout_deadline_tracking(self) -> None:
        """Test timeout deadline tracking."""
        dispatcher = FlextDispatcher()
        operation_id = "test-op-123"

        # Track timeout
        dispatcher._track_timeout_context(operation_id, 1.0)

        # Check if deadline exists
        deadline = dispatcher._timeout_deadlines.get(operation_id)
        assert deadline is not None
        assert deadline > time.time()

        # Cleanup
        dispatcher._cleanup_timeout_context(operation_id)
        deadline_after = dispatcher._timeout_deadlines.get(operation_id)
        assert deadline_after is None

    def test_should_retry_on_error_logic(self) -> None:
        """Test retry decision logic."""
        dispatcher = FlextDispatcher()

        # Test retriable error
        should_retry = dispatcher._should_retry_on_error(
            1, error_message="timeout error"
        )
        assert should_retry is True

        # Test non-retriable error
        should_retry = dispatcher._should_retry_on_error(
            1, error_message="Invalid input"
        )
        assert should_retry is False

        # Test max attempts reached
        max_attempts = dispatcher._retry_policy.get_max_attempts()
        should_retry = dispatcher._should_retry_on_error(
            max_attempts + 1, error_message="timeout"
        )
        assert should_retry is False
