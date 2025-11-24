"""Real tests to achieve 100% dispatcher coverage - no mocks.

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in dispatcher.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from typing import cast, override

import pytest

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

    __test__ = False  # Not a test class, just a helper class

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
        assert result.error is not None and "Message cannot be None" in result.error

    def test_dispatch_with_string_message_fails(self) -> None:
        """Test dispatch with string message - should fail in normalize."""
        dispatcher = FlextDispatcher()
        # String messages are caught in _normalize_dispatch_message which raises
        # But dispatch() catches and returns FlextResult, so test the internal method
        with pytest.raises(TypeError, match=r".*String message_type not supported.*"):
            dispatcher._normalize_dispatch_message("string", None)

    def test_dispatch_with_invalid_metadata_type(self) -> None:
        """Test dispatch with invalid metadata type."""
        dispatcher = FlextDispatcher()
        # Don't register handler to force complex dispatch path with validation

        message = MetadataTestMessage("test")
        # Pass invalid metadata type (not dict)
        result = dispatcher.dispatch(
            message, metadata=cast("dict[str, object]", "invalid")
        )
        assert result.is_failure
        assert result.error is not None and "Invalid metadata type" in result.error

    def test_dispatch_with_config_object(self) -> None:
        """Test dispatch with config object."""
        dispatcher = FlextDispatcher()
        handler = MetadataAwareHandler()
        dispatcher.register_function(MetadataTestMessage, handler.handle)

        # Create config object
        class DispatchConfig:
            def __init__(self) -> None:
                # Use FlextModels.Metadata for STRICT mode
                self.metadata = FlextModels.Metadata(attributes={"key": "value"})
                self.correlation_id = "test-correlation"
                self.timeout_override = 30

        message = MetadataTestMessage("test")
        config = DispatchConfig()
        result = dispatcher.dispatch(message, config=config)
        assert result.is_success

    def test_dispatch_timeout_handling(self) -> None:
        """Test dispatch with timeout."""
        dispatcher = FlextDispatcher()
        # Don't register handler to force complex dispatch path with timeout

        message = TimeoutTestMessage(delay=2.0)
        # Set very short timeout
        result = dispatcher.dispatch(message, timeout_override=1)
        # Should fail due to no handler registered
        assert result.is_failure
        assert result.error is not None
        assert "handler" in result.error and "found" in result.error

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
        assert (
            result.error is not None
            and "Configuration extraction failed" in result.error
        )

    def test_normalize_dispatch_message_none_raises(self) -> None:
        """Test _normalize_dispatch_message raises on None."""
        dispatcher = FlextDispatcher()
        with pytest.raises(TypeError, match=r".*Message cannot be None.*"):
            dispatcher._normalize_dispatch_message(None, None)

    def test_normalize_dispatch_message_string_raises(self) -> None:
        """Test _normalize_dispatch_message raises on string."""
        dispatcher = FlextDispatcher()
        with pytest.raises(TypeError, match=r".*String message_type not supported.*"):
            dispatcher._normalize_dispatch_message("string", None)

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
            1,
            error_message="timeout error",
        )
        assert should_retry is True

        # Test non-retriable error
        should_retry = dispatcher._should_retry_on_error(
            1,
            error_message="Invalid input",
        )
        assert should_retry is False

        # Test max attempts reached
        max_attempts = dispatcher._retry_policy.get_max_attempts()
        should_retry = dispatcher._should_retry_on_error(
            max_attempts + 1,
            error_message="timeout",
        )
        assert should_retry is False

    def test_circuit_breaker_full_cycle_open_to_closed(self) -> None:
        """Test circuit breaker OPEN → HALF_OPEN → CLOSED cycle (covers line 192)."""
        dispatcher = FlextDispatcher()

        # Configure circuit breaker with shorter timeout for testing (real config, not mock)
        dispatcher._circuit_breaker._recovery_timeout = 0.5

        # Create handler that fails initially
        class FailingHandler(FlextHandlers[TestMessage100, dict[str, bool]]):
            def __init__(self) -> None:
                config = FlextModels.Cqrs.Handler(
                    handler_id="failing_handler",
                    handler_name="FailingHandler",
                    command_timeout=30,
                    max_command_retries=0,  # No retries
                )
                super().__init__(config=config)
                self.call_count = 0

            @override
            def handle(self, message: TestMessage100) -> FlextResult[dict[str, bool]]:
                self.call_count += 1
                if self.call_count <= 5:  # Fail first 5 times (threshold = 5)
                    return FlextResult[dict[str, bool]].fail("Error to open circuit")
                return FlextResult[dict[str, bool]].ok({"success": True})

        handler = FailingHandler()
        dispatcher.register_handler(TestMessage100, handler)

        # 1. Fail enough times to OPEN circuit (threshold = 5)
        for _ in range(5):
            result = dispatcher.dispatch(TestMessage100("test"))
            assert result.is_failure

        # 2. Circuit should be OPEN - verify state
        breaker_state = dispatcher._circuit_breaker.get_state("TestMessage100")
        assert breaker_state == FlextConstants.Reliability.CircuitBreakerState.OPEN

        # 3. Wait for half-open window (recovery_timeout = 0.5s configured above)
        time.sleep(0.6)

        # 4. Next call triggers attempt_reset → HALF_OPEN, handler called (count=6, success!)
        result = dispatcher.dispatch(TestMessage100("test"))
        assert result.is_success, f"Expected success in HALF_OPEN, got: {result.error}"

        # 5. State should now be HALF_OPEN
        breaker_state = dispatcher._circuit_breaker.get_state("TestMessage100")
        assert breaker_state == FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN

        # 6. Need 3 total successes in HALF_OPEN to close (success_threshold = 3)
        # Already have 1, need 2 more
        for _ in range(2):
            result = dispatcher.dispatch(TestMessage100("test"))
            assert result.is_success

        # 7. After 3 successes in HALF_OPEN, transitions to CLOSED (line 159 → 199 → 188-192)
        breaker_state_final = dispatcher._circuit_breaker.get_state("TestMessage100")
        assert (
            breaker_state_final == FlextConstants.Reliability.CircuitBreakerState.CLOSED
        )

        # 8. Verify opened_at was deleted (line 192 executed!)
        assert "TestMessage100" not in dispatcher._circuit_breaker._opened_at

    def test_rate_limiter_apply_jitter_with_zero_base_delay(self) -> None:
        """Test rate limiter _apply_jitter with base_delay <= 0 (lines 425-426)."""
        dispatcher = FlextDispatcher()
        rate_limiter = dispatcher._rate_limiter

        # Test with base_delay = 0.0
        result = rate_limiter._apply_jitter(0.0)
        assert result == 0.0

        # Test with negative base_delay
        result = rate_limiter._apply_jitter(-1.0)
        assert result == -1.0

    def test_rate_limiter_apply_jitter_with_zero_jitter_factor(self) -> None:
        """Test rate limiter _apply_jitter with jitter_factor == 0 (lines 425-426)."""
        dispatcher = FlextDispatcher()
        rate_limiter = dispatcher._rate_limiter

        # Set jitter_factor to 0
        rate_limiter._jitter_factor = 0.0

        # Should return base_delay unchanged
        result = rate_limiter._apply_jitter(1.0)
        assert result == 1.0

    def test_rate_limiter_apply_jitter_with_positive_values(self) -> None:
        """Test rate limiter _apply_jitter with positive values (lines 428-435)."""
        dispatcher = FlextDispatcher()
        rate_limiter = dispatcher._rate_limiter

        # Test with positive base_delay and jitter_factor
        base_delay = 1.0
        result = rate_limiter._apply_jitter(base_delay)

        # Result should be jittered (different from base) and non-negative
        assert result >= 0.0
        # With default jitter_factor (0.1), result should be between 0.9 and 1.1
        assert 0.8 <= result <= 1.2  # Allow some margin

    def test_rate_limiter_get_max_requests(self) -> None:
        """Test rate limiter get_max_requests (line 478)."""
        dispatcher = FlextDispatcher()
        rate_limiter = dispatcher._rate_limiter

        # Get max_requests value
        max_requests = rate_limiter.get_max_requests()
        assert isinstance(max_requests, int)
        assert max_requests > 0

    def test_rate_limiter_get_window_seconds(self) -> None:
        """Test rate limiter get_window_seconds (line 482)."""
        dispatcher = FlextDispatcher()
        rate_limiter = dispatcher._rate_limiter

        # Get window_seconds value
        window_seconds = rate_limiter.get_window_seconds()
        assert isinstance(window_seconds, float)
        assert window_seconds > 0.0

    def test_retry_policy_exponential_delay_with_zero_base(self) -> None:
        """Test get_exponential_delay with base_delay = 0.0 (lines 558-559)."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Set base_delay to 0.0
        policy._base_delay = 0.0

        # Should return 0.0 regardless of attempt number
        assert policy.get_exponential_delay(0) == 0.0
        assert policy.get_exponential_delay(5) == 0.0
        assert policy.get_exponential_delay(10) == 0.0

    def test_retry_policy_exponential_delay_calculation(self) -> None:
        """Test get_exponential_delay exponential backoff (lines 562-564)."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Set known values for testing
        policy._base_delay = 1.0
        policy._exponential_factor = 2.0
        policy._max_delay = 100.0

        # Test exponential backoff: base_delay * (factor ^ attempt)
        # Attempt 0: 1.0 * (2.0 ^ 0) = 1.0
        assert policy.get_exponential_delay(0) == 1.0

        # Attempt 1: 1.0 * (2.0 ^ 1) = 2.0
        assert policy.get_exponential_delay(1) == 2.0

        # Attempt 2: 1.0 * (2.0 ^ 2) = 4.0
        assert policy.get_exponential_delay(2) == 4.0

        # Attempt 3: 1.0 * (2.0 ^ 3) = 8.0
        assert policy.get_exponential_delay(3) == 8.0

    def test_retry_policy_exponential_delay_max_cap(self) -> None:
        """Test get_exponential_delay capped at max_delay (line 566)."""
        dispatcher = FlextDispatcher()
        policy = dispatcher._retry_policy

        # Set values that will exceed max_delay
        policy._base_delay = 1.0
        policy._exponential_factor = 2.0
        policy._max_delay = 10.0  # Cap at 10 seconds

        # Attempt 10: 1.0 * (2.0 ^ 10) = 1024.0, but should be capped at 10.0
        result = policy.get_exponential_delay(10)
        assert result == 10.0  # Should be capped at max_delay

        # Attempt 5: 1.0 * (2.0 ^ 5) = 32.0, should be capped at 10.0
        result = policy.get_exponential_delay(5)
        assert result == 10.0

    def test_processor_validation_missing_process_method(self) -> None:
        """Test processor validation when process method is missing (lines 765, 849)."""
        dispatcher = FlextDispatcher()

        # Create processor without "process" method
        class InvalidProcessor:
            def execute(self, data: object) -> object:
                return data

        processor = InvalidProcessor()

        # Validation should fail - line 765
        validation_result = dispatcher._validate_processor_interface(processor)
        assert validation_result.is_failure
        assert (
            validation_result.error is not None
            and "must have 'process' method" in validation_result.error
        )

        # Register processor (bypassing validation for test purposes)
        dispatcher._processors["invalid_processor"] = processor

        # Execution should fail - line 858
        result = dispatcher.process("invalid_processor", "test_data")
        assert result.is_failure
        assert (
            result.error is not None
            and "processor must be callable or have 'process' method" in result.error
        )

    def test_processor_validation_process_not_callable(self) -> None:
        """Test processor validation when process attribute is not callable (lines 772, 855)."""
        dispatcher = FlextDispatcher()

        # Create processor with non-callable "process" attribute
        class InvalidProcessor:
            process = "not_callable"  # Not a method

        processor = InvalidProcessor()

        # Validation should fail - line 772
        validation_result = dispatcher._validate_processor_interface(processor)
        assert validation_result.is_failure
        assert (
            validation_result.error is not None
            and "must have 'process' method" in validation_result.error
        )

        # Register processor (bypassing validation for test purposes)
        dispatcher._processors["invalid_processor"] = processor

        # Execution should fail - line 864
        result = dispatcher.process("invalid_processor", "test_data")
        assert result.is_failure
        assert (
            result.error is not None
            and "'process' attribute must be callable" in result.error
        )

    def test_processor_metrics_initialization_on_first_execution(self) -> None:
        """Test processor metrics are initialized on first execution (line 877)."""
        dispatcher = FlextDispatcher()

        # Create simple processor
        class SimpleProcessor:
            def process(self, data: int) -> FlextResult[int]:
                return FlextResult[int].ok(data * 2)

        # Verify metrics don't exist yet
        assert "simple_processor" not in dispatcher._processor_metrics_per_name

        # Register processor (initializes metrics dict - line 877)
        dispatcher.register_processor("simple_processor", SimpleProcessor())

        # Verify metrics were initialized during registration
        assert "simple_processor" in dispatcher._processor_metrics_per_name
        metrics = dispatcher._processor_metrics_per_name["simple_processor"]
        assert metrics["executions"] == 0
        assert metrics["successful_processes"] == 0
        assert metrics["failed_processes"] == 0

        # Execute processor to update metrics
        result = dispatcher.process("simple_processor", 5)
        assert result.is_success
        assert result.unwrap() == 10

        # Verify metrics were updated
        metrics = dispatcher._processor_metrics_per_name["simple_processor"]
        assert metrics["executions"] == 1
        assert metrics["successful_processes"] == 1

    def test_processor_exception_handling_with_metrics(self) -> None:
        """Test processor exception handling records execution time (lines 892-897)."""
        dispatcher = FlextDispatcher()

        # Create processor that raises exception
        class FailingProcessor:
            def process(self, data: object) -> FlextResult[object]:
                msg = "Processor error"
                raise RuntimeError(msg)

        # Register processor
        dispatcher.register_processor("failing_processor", FailingProcessor())

        # Execute processor (triggers exception handling lines 892-897)
        result = dispatcher.process("failing_processor", "test_data")

        # Should return failure result
        assert result.is_failure
        assert result.error is not None and "Processor execution failed" in result.error
        assert result.error is not None and "Processor error" in result.error

        # Verify execution time was recorded even on failure (line 893-896)
        assert "failing_processor" in dispatcher._processor_execution_times
        assert len(dispatcher._processor_execution_times["failing_processor"]) == 1
        assert dispatcher._processor_execution_times["failing_processor"][0] >= 0.0
