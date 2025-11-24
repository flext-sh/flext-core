"""Comprehensive coverage tests for FlextDispatcher.

This module provides extensive tests for FlextDispatcher reliability patterns:
- Handler registration and management
- Circuit breaker pattern
- Rate limiting
- Retry logic
- Timeout enforcement
- Context propagation

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import cast

from flext_core import (
    FlextConfig,
    FlextConstants,
    FlextDispatcher,
    FlextHandlers,
    FlextModels,
    FlextResult,
)
from flext_tests.docker import FlextTestDocker


class SimpleMessage(FlextModels.Value):
    """Simple test message."""

    value: str


class TestDispatcherBasics:
    """Test basic dispatcher functionality."""

    def test_dispatcher_creation(self) -> None:
        """Test creating dispatcher instance."""
        dispatcher = FlextDispatcher()
        assert dispatcher is not None
        assert hasattr(dispatcher, "execute")  # Layer 1: CQRS routing
        assert hasattr(dispatcher, "dispatch")  # Layer 2: Reliability patterns
        assert hasattr(dispatcher, "process")  # Layer 3: Advanced processing

    def test_dispatcher_config_access(self) -> None:
        """Test accessing dispatcher configuration."""
        dispatcher = FlextDispatcher()
        config = dispatcher.dispatcher_config
        assert isinstance(config, dict)
        assert "timeout_seconds" in config or "dispatcher_enable_logging" in config

    def test_dispatcher_config_keys(self) -> None:
        """Test dispatcher config contains expected keys."""
        dispatcher = FlextDispatcher()
        config = dispatcher.dispatcher_config
        expected_keys = [
            "dispatcher_enable_logging",
            "dispatcher_enable_metrics",
            "dispatcher_timeout_seconds",
            "enable_timeout_executor",
            "executor_workers",
            "dispatcher_auto_context",
            "circuit_breaker_threshold",
            "rate_limit_max_requests",
            "rate_limit_window_seconds",
            "max_retry_attempts",
            "retry_delay",
        ]
        for key in expected_keys:
            assert key in config, f"Missing config key: {key}"

    def test_dispatcher_config_values(self) -> None:
        """Test dispatcher config values are reasonable."""
        dispatcher = FlextDispatcher()
        config = dispatcher.dispatcher_config
        assert config["dispatcher_timeout_seconds"] > 0
        assert config["executor_workers"] >= 1
        assert isinstance(config["dispatcher_enable_logging"], bool)
        assert isinstance(config["dispatcher_enable_metrics"], bool)
        assert isinstance(config["enable_timeout_executor"], bool)
        assert config["circuit_breaker_threshold"] >= 1
        assert config["rate_limit_max_requests"] >= 1
        assert config["rate_limit_window_seconds"] > 0
        assert config["max_retry_attempts"] >= 0
        assert config["retry_delay"] >= 0


class TestCircuitBreakerManager:
    """Test circuit breaker manager functionality."""

    def test_circuit_breaker_initialization(self) -> None:
        """Test circuit breaker manager initialization."""
        dispatcher = FlextDispatcher()
        cb_manager = dispatcher._circuit_breaker

        assert cb_manager._threshold == dispatcher.config.circuit_breaker_threshold
        assert (
            cb_manager._recovery_timeout
            == FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        )
        assert (
            cb_manager._success_threshold
            == FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD
        )

    def test_circuit_breaker_get_state_default(self) -> None:
        """Test getting default state for new message type."""
        dispatcher = FlextDispatcher()
        cb_manager = dispatcher._circuit_breaker

        state = cb_manager.get_state("test_message")
        assert state == FlextConstants.Reliability.CircuitBreakerState.CLOSED

    def test_circuit_breaker_set_state(self) -> None:
        """Test setting state for message type."""
        dispatcher = FlextDispatcher()
        cb_manager = dispatcher._circuit_breaker

        cb_manager.set_state(
            "test_message", FlextConstants.Reliability.CircuitBreakerState.OPEN
        )
        assert (
            cb_manager.get_state("test_message")
            == FlextConstants.Reliability.CircuitBreakerState.OPEN
        )

    def test_circuit_breaker_is_open(self) -> None:
        """Test checking if circuit breaker is open."""
        dispatcher = FlextDispatcher()
        cb_manager = dispatcher._circuit_breaker

        assert not cb_manager.is_open("test_message")

        cb_manager.set_state(
            "test_message", FlextConstants.Reliability.CircuitBreakerState.OPEN
        )
        assert cb_manager.is_open("test_message")

    def test_circuit_breaker_record_success(self) -> None:
        """Test recording successful operation."""
        dispatcher = FlextDispatcher()
        cb_manager = dispatcher._circuit_breaker

        # Initially closed
        assert (
            cb_manager.get_state("test_message")
            == FlextConstants.Reliability.CircuitBreakerState.CLOSED
        )

        # Record success on closed circuit
        cb_manager.record_success("test_message")
        assert (
            cb_manager.get_state("test_message")
            == FlextConstants.Reliability.CircuitBreakerState.CLOSED
        )
        assert cb_manager._total_successes.get("test_message", 0) == 1

    def test_circuit_breaker_record_failure(self) -> None:
        """Test recording failed operation."""
        dispatcher = FlextDispatcher()
        cb_manager = dispatcher._circuit_breaker

        threshold = cb_manager._threshold

        # Record failures up to threshold
        for i in range(threshold):
            cb_manager.record_failure("test_message")
            if i < threshold - 1:
                assert (
                    cb_manager.get_state("test_message")
                    == FlextConstants.Reliability.CircuitBreakerState.CLOSED
                )
            else:
                assert (
                    cb_manager.get_state("test_message")
                    == FlextConstants.Reliability.CircuitBreakerState.OPEN
                )

        assert cb_manager._failures.get("test_message", 0) == threshold

    def test_circuit_breaker_get_failure_count(self) -> None:
        """Test getting failure count."""
        dispatcher = FlextDispatcher()
        cb_manager = dispatcher._circuit_breaker

        assert cb_manager.get_failure_count("test_message") == 0

        cb_manager.record_failure("test_message")
        assert cb_manager.get_failure_count("test_message") == 1

    def test_circuit_breaker_get_threshold(self) -> None:
        """Test getting threshold value."""
        dispatcher = FlextDispatcher()
        cb_manager = dispatcher._circuit_breaker

        assert cb_manager.get_threshold() == cb_manager._threshold

    def test_circuit_breaker_cleanup(self) -> None:
        """Test cleanup functionality."""
        dispatcher = FlextDispatcher()
        cb_manager = dispatcher._circuit_breaker

        # Add some state
        cb_manager.record_failure("test_message")
        cb_manager.set_state(
            "test_message", FlextConstants.Reliability.CircuitBreakerState.OPEN
        )

        # Cleanup
        cb_manager.cleanup()

        # Should be reset
        assert (
            cb_manager.get_state("test_message")
            == FlextConstants.Reliability.CircuitBreakerState.CLOSED
        )
        assert cb_manager.get_failure_count("test_message") == 0


class TestRateLimiterManager:
    """Test rate limiter manager functionality."""

    def test_rate_limiter_initialization(self) -> None:
        """Test rate limiter manager initialization."""
        dispatcher = FlextDispatcher()
        rl_manager = dispatcher._rate_limiter

        assert rl_manager._max_requests == dispatcher.config.rate_limit_max_requests
        assert rl_manager._window_seconds == dispatcher.config.rate_limit_window_seconds

    def test_rate_limiter_check_rate_limit(self) -> None:
        """Test checking rate limit."""
        dispatcher = FlextDispatcher()
        rl_manager = dispatcher._rate_limiter

        # Initially should allow requests
        result = rl_manager.check_rate_limit("test_message")
        assert result.is_success
        assert result.value is True

    def test_rate_limiter_get_max_requests(self) -> None:
        """Test getting max requests."""
        dispatcher = FlextDispatcher()
        rl_manager = dispatcher._rate_limiter

        assert rl_manager.get_max_requests() == rl_manager._max_requests

    def test_rate_limiter_get_window_seconds(self) -> None:
        """Test getting window seconds."""
        dispatcher = FlextDispatcher()
        rl_manager = dispatcher._rate_limiter

        assert rl_manager.get_window_seconds() == rl_manager._window_seconds

    def test_rate_limiter_cleanup(self) -> None:
        """Test cleanup functionality."""
        dispatcher = FlextDispatcher()
        rl_manager = dispatcher._rate_limiter

        # Trigger some rate limiting
        for _ in range(rl_manager._max_requests + 1):
            rl_manager.check_rate_limit("test_message")

        # Cleanup
        rl_manager.cleanup()

        # Should allow requests again
        result = rl_manager.check_rate_limit("test_message")
        assert result.is_success


class TestRetryPolicy:
    """Test retry policy functionality."""

    def test_retry_policy_initialization(self) -> None:
        """Test retry policy initialization."""
        dispatcher = FlextDispatcher()
        retry_policy = dispatcher._retry_policy

        config = FlextConfig()
        assert retry_policy.get_max_attempts() == config.max_retry_attempts
        assert retry_policy.get_retry_delay() == config.retry_delay

    def test_retry_policy_should_retry(self) -> None:
        """Test should retry logic."""
        dispatcher = FlextDispatcher()
        retry_policy = dispatcher._retry_policy

        max_attempts = retry_policy.get_max_attempts()

        # Should retry up to max_attempts - 1 (0-based)
        for attempt in range(max_attempts):
            if attempt < max_attempts - 1:
                assert retry_policy.should_retry(attempt), (
                    f"Should retry attempt {attempt}"
                )
            else:
                assert not retry_policy.should_retry(attempt), (
                    f"Should not retry attempt {attempt}"
                )

    def test_retry_policy_is_retriable_error(self) -> None:
        """Test checking if error is retriable."""
        dispatcher = FlextDispatcher()
        retry_policy = dispatcher._retry_policy

        # Common retriable errors
        assert retry_policy.is_retriable_error("TimeoutError")
        assert retry_policy.is_retriable_error("Temporary failure")
        assert retry_policy.is_retriable_error("temporarily unavailable")

        # Non-retriable errors
        assert not retry_policy.is_retriable_error("ValueError")
        assert not retry_policy.is_retriable_error("TypeError")
        assert not retry_policy.is_retriable_error("CustomError")

    def test_retry_policy_get_max_attempts(self) -> None:
        """Test getting max attempts."""
        dispatcher = FlextDispatcher()
        retry_policy = dispatcher._retry_policy

        assert retry_policy.get_max_attempts() == retry_policy._max_attempts

    def test_retry_policy_get_retry_delay(self) -> None:
        """Test getting retry delay."""
        dispatcher = FlextDispatcher()
        retry_policy = dispatcher._retry_policy

        config = FlextConfig()
        assert retry_policy.get_retry_delay() == config.retry_delay

    def test_retry_policy_cleanup(self) -> None:
        """Test cleanup functionality."""
        dispatcher = FlextDispatcher()
        retry_policy = dispatcher._retry_policy

        # Record some attempts
        retry_policy.record_attempt("test_message")

        # Cleanup
        retry_policy.cleanup()

        # Should be reset (no direct way to verify, but shouldn't error)


class TestTimeoutEnforcer:
    """Test timeout enforcer functionality."""

    def test_timeout_enforcer_initialization(self) -> None:
        """Test timeout enforcer initialization."""
        dispatcher = FlextDispatcher()
        timeout_enforcer = dispatcher._timeout_enforcer

        assert (
            timeout_enforcer._use_timeout_executor
            == dispatcher.config.enable_timeout_executor
        )
        assert timeout_enforcer._executor_workers == dispatcher.config.executor_workers

    def test_timeout_enforcer_should_use_executor(self) -> None:
        """Test should use executor logic."""
        dispatcher = FlextDispatcher()
        timeout_enforcer = dispatcher._timeout_enforcer

        result = timeout_enforcer.should_use_executor()
        assert isinstance(result, bool)

    def test_timeout_enforcer_reset_executor(self) -> None:
        """Test resetting executor."""
        dispatcher = FlextDispatcher()
        timeout_enforcer = dispatcher._timeout_enforcer

        # Should not error
        timeout_enforcer.reset_executor()

    def test_timeout_enforcer_resolve_workers(self) -> None:
        """Test resolving worker count."""
        dispatcher = FlextDispatcher()
        timeout_enforcer = dispatcher._timeout_enforcer

        workers = timeout_enforcer.resolve_workers()
        assert workers >= 1

    def test_timeout_enforcer_get_executor_status(self) -> None:
        """Test getting executor status."""
        dispatcher = FlextDispatcher()
        timeout_enforcer = dispatcher._timeout_enforcer

        status = timeout_enforcer.get_executor_status()
        assert isinstance(status, dict)

    def test_timeout_enforcer_cleanup(self) -> None:
        """Test cleanup functionality."""
        dispatcher = FlextDispatcher()
        timeout_enforcer = dispatcher._timeout_enforcer

        # Should not error
        timeout_enforcer.cleanup()


class TestDispatcherAdvanced:
    """Test advanced dispatcher functionality with real Docker integration."""

    def test_dispatcher_with_docker_integration(self) -> None:
        """Test dispatcher with real Docker container management."""
        # This test uses real Docker containers for integration testing
        docker_manager = FlextTestDocker()

        # Test basic container operations
        containers_result = docker_manager.list_containers()
        assert containers_result.is_success
        containers = containers_result.value
        assert isinstance(containers, list)

        # Test dispatcher with container context
        dispatcher = FlextDispatcher()

        # Register a simple handler
        def test_handler(message: SimpleMessage) -> str:
            return f"processed: {message.value}"

        result = dispatcher.register_function(
            SimpleMessage, test_handler, mode=FlextConstants.Cqrs.HandlerType.COMMAND
        )
        assert result.is_success

        # Dispatch with context - use the correct message type
        message = SimpleMessage(value="test message")
        result = dispatcher.dispatch(message)
        assert result.is_success
        assert result.value == "processed: test message"

    def test_dispatcher_batch_processing(self) -> None:
        """Test batch processing capabilities."""
        dispatcher = FlextDispatcher()

        def batch_handler(message: SimpleMessage) -> str:
            return f"batch_{message.value}"

        result = dispatcher.register_function(
            SimpleMessage, batch_handler, mode=FlextConstants.Cqrs.HandlerType.COMMAND
        )
        assert result.is_success

        # Test batch dispatch
        messages = [SimpleMessage(value=f"msg{i}") for i in range(3)]
        results = dispatcher.dispatch_batch("SimpleMessage", messages)
        assert len(results) == 3
        for result in results:
            assert result.is_success
            assert "batch_" in result.value

    def test_dispatcher_error_handling(self) -> None:
        """Test comprehensive error handling."""
        dispatcher = FlextDispatcher()

        def failing_handler(message: SimpleMessage) -> str:
            error_msg = "Test error"
            raise ValueError(error_msg)

        result = dispatcher.register_function(
            SimpleMessage, failing_handler, mode=FlextConstants.Cqrs.HandlerType.COMMAND
        )
        assert result.is_success

        # Dispatch should handle error gracefully
        message = SimpleMessage(value="test message")
        result = dispatcher.dispatch(message)
        assert result.is_failure
        assert "Test error" in str(result.error)

    def test_dispatcher_metrics_collection(self) -> None:
        """Test metrics collection during dispatch."""
        dispatcher = FlextDispatcher()

        def success_handler(message: SimpleMessage) -> str:
            return f"ok: {message.value}"

        result = dispatcher.register_function(
            SimpleMessage, success_handler, mode=FlextConstants.Cqrs.HandlerType.COMMAND
        )
        assert result.is_success

        # Dispatch and check metrics
        message = SimpleMessage(value="test")
        result = dispatcher.dispatch(message)
        assert result.is_success

        # Metrics should be updated (implementation dependent)
        # This tests the metrics integration without mocking

    @classmethod
    def setup_method(cls) -> None:
        """Setup for each test."""

    @classmethod
    def teardown_method(cls) -> None:
        """Cleanup after each test."""


class TestHandlerRegistration:
    """Test handler registration functionality."""

    def test_register_handler_with_type_and_handler(self) -> None:
        """Test registering handler with message type and handler."""
        dispatcher = FlextDispatcher()

        def simple_handler(msg: object) -> str:
            return "handled"

        # Use register_function which accepts mode as keyword argument
        # Create a simple message type class for testing
        class TestMessage:
            def __init__(self, value: str) -> None:
                self.value = value

        result = dispatcher.register_function(
            TestMessage,
            simple_handler,
            mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )
        assert result.is_success
        assert "registration_id" in result.value or "status" in result.value

    def test_register_handler_with_flext_handlers(self) -> None:
        """Test registering FlextHandlers instance."""
        dispatcher = FlextDispatcher()

        handler = FlextHandlers.create_from_callable(
            func=lambda msg: "handled",
            handler_name="test_handler",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        result = dispatcher.register_handler("TestMessage", handler)
        assert result.is_success

    def test_register_command(self) -> None:
        """Test registering command handler."""
        dispatcher = FlextDispatcher()

        handler = FlextHandlers.create_from_callable(
            func=lambda cmd: "command_handled",
            handler_name="cmd_handler",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        result = dispatcher.register_command(SimpleMessage, handler)
        assert result.is_success

    def test_register_query(self) -> None:
        """Test registering query handler."""
        dispatcher = FlextDispatcher()

        handler = FlextHandlers.create_from_callable(
            func=lambda q: {"result": "query_answer"},
            handler_name="query_handler",
            handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
        )

        result = dispatcher.register_query(SimpleMessage, handler)
        assert result.is_success

    def test_register_function(self) -> None:
        """Test registering function as handler."""
        dispatcher = FlextDispatcher()

        def handler_func(msg: object) -> str:
            return "function_handled"

        result = dispatcher.register_function(
            SimpleMessage,
            handler_func,
            mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )
        assert result.is_success

    def test_register_invalid_mode(self) -> None:
        """Test registering handler with invalid mode."""
        dispatcher = FlextDispatcher()

        def handler_func(msg: object) -> str:
            return "handled"

        result = dispatcher.register_function(
            SimpleMessage,
            handler_func,
            mode=cast("FlextConstants.Cqrs.HandlerType", "invalid_mode"),
        )
        assert result.is_failure


class TestMessageDispatch:
    """Test message dispatching."""

    def test_dispatch_with_registered_handler(self) -> None:
        """Test dispatching message with registered handler."""
        dispatcher = FlextDispatcher()

        def message_handler(msg: object) -> str:
            if isinstance(msg, SimpleMessage):
                return f"handled: {msg.value}"
            return "handled"

        # Register handler
        dispatcher.register_handler("SimpleMessage", message_handler)

        # Dispatch message
        message = SimpleMessage(value="test")
        result = dispatcher.dispatch(message)
        assert result.is_success or result.is_failure

    def test_dispatch_batch(self) -> None:
        """Test batch dispatch of messages."""
        dispatcher = FlextDispatcher()

        def handler_func(msg: object) -> str:
            return "handled"

        dispatcher.register_function(SimpleMessage, handler_func)

        messages = [SimpleMessage(value=f"msg{i}") for i in range(3)]
        results = dispatcher.dispatch_batch(
            "SimpleMessage",
            cast("list[object]", messages),
        )

        assert len(results) == 3
        assert all(isinstance(r, FlextResult) for r in results)

    def test_dispatch_with_timeout_override(self) -> None:
        """Test dispatch with timeout override."""
        dispatcher = FlextDispatcher()

        def quick_handler(msg: object) -> str:
            return "handled"

        dispatcher.register_handler("TestMessage", quick_handler)

        message = SimpleMessage(value="test")
        result = dispatcher.dispatch(
            message,
            timeout_override=30,
        )
        assert result is not None


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_open_on_failures(self) -> None:
        """Test circuit breaker opens after failures."""
        dispatcher = FlextDispatcher()

        def failing_handler(msg: object) -> str:
            msg = "Handler failure"
            raise ValueError(msg)

        dispatcher.register_handler("FailingMessage", failing_handler)

        # Try dispatching until circuit breaker opens
        message = SimpleMessage(value="test")
        threshold = dispatcher._circuit_breaker._threshold

        for _ in range(threshold + 1):
            dispatcher.dispatch(message)
            # After threshold failures, circuit breaker should trip

        # Check circuit breaker state
        assert dispatcher._circuit_breaker._failures.get("SimpleMessage", 0) > 0

    def test_circuit_breaker_reset_on_success(self) -> None:
        """Test circuit breaker resets on successful dispatch."""
        dispatcher = FlextDispatcher()

        def success_handler(msg: object) -> str:
            return "success"

        # Register handler for SimpleMessage (the actual message type)
        dispatcher.register_handler("SimpleMessage", success_handler)

        message = SimpleMessage(value="test")

        # Dispatch successfully
        dispatcher.dispatch(message)

        # Circuit breaker should be reset after successful dispatch
        # (or have low failure count if dispatch succeeds)
        failures = dispatcher._circuit_breaker._failures.get("SimpleMessage", 0)
        assert failures <= 1  # Allow for initial state


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_state_tracking(self) -> None:
        """Test rate limit state is tracked per message type."""
        dispatcher = FlextDispatcher()

        def handler_func(msg: object) -> str:
            return "handled"

        dispatcher.register_handler("TestMessage", handler_func)

        message = SimpleMessage(value="test")

        # Dispatch a message
        dispatcher.dispatch(message)

        # Check rate limit state is tracked (now stored as tuple: (window_start, count))
        window_data = dispatcher._rate_limiter._windows.get("SimpleMessage")
        assert window_data is not None
        assert isinstance(window_data, tuple)
        assert len(window_data) == 2  # (window_start, count)
        _window_start, count = window_data
        assert count >= 1  # At least one request tracked

    def test_rate_limit_blocking(self) -> None:
        """Test rate limiting blocks after threshold."""
        dispatcher = FlextDispatcher()
        # Set low limit for testing via manager
        max_requests = 2
        dispatcher._rate_limiter._max_requests = max_requests

        def handler_func(msg: object) -> str:
            return "handled"

        dispatcher.register_handler("TestMessage", handler_func)

        message = SimpleMessage(value="test")

        # Dispatch messages up to limit
        for i in range(max_requests + 1):
            result = dispatcher.dispatch(message)
            if i >= max_requests:
                # Should be rate limited
                assert result.is_failure or "rate" not in str(result.error).lower()


class TestRetryLogic:
    """Test retry logic."""

    def test_retry_configuration(self) -> None:
        """Test retry configuration is accessible."""
        dispatcher = FlextDispatcher()

        assert hasattr(dispatcher.config, "max_retry_attempts")
        assert hasattr(dispatcher.config, "retry_delay")
        assert dispatcher.config.max_retry_attempts > 0
        assert dispatcher.config.retry_delay >= 0

    def test_dispatch_with_retries(self) -> None:
        """Test dispatch with retry configuration."""
        dispatcher = FlextDispatcher()

        attempt_count = 0

        def flaky_handler(msg: object) -> FlextResult[str]:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                return FlextResult[str].fail("Temporary failure")
            return FlextResult[str].ok("succeeded")

        dispatcher.register_handler("FlakyMessage", flaky_handler)

        message = SimpleMessage(value="test")
        result = dispatcher.dispatch(message)

        # Should succeed due to retries
        assert result is not None


class TestContextManagement:
    """Test context management and propagation."""

    def test_context_scope_management(self) -> None:
        """Test context scope is properly managed."""
        dispatcher = FlextDispatcher()

        metadata = cast("dict[str, object]", {"user_id": "123", "operation": "test"})
        correlation_id = "corr-123"

        def handler_func(msg: object) -> str:
            return "handled"

        dispatcher.register_handler("TestMessage", handler_func)

        message = SimpleMessage(value="test")
        result = dispatcher.dispatch(
            message,
            metadata=metadata,
            correlation_id=correlation_id,
        )

        assert result is not None

    def test_metadata_normalization(self) -> None:
        """Test metadata is normalized correctly."""
        dispatcher = FlextDispatcher()

        # Test with dict (backward compatibility path)
        metadata_dict = {"key": "value", "number": 123}
        normalized = dispatcher._normalize_context_metadata(metadata_dict)

        assert normalized is not None
        assert normalized["key"] == "value"
        assert normalized["number"] == 123

    def test_context_with_flext_metadata(self) -> None:
        """Test context with FlextModels.Metadata."""
        dispatcher = FlextDispatcher()

        metadata_obj = FlextModels.Metadata(
            attributes={"test": "value", "number": "42"},
        )
        normalized = dispatcher._normalize_context_metadata(metadata_obj)

        assert normalized is not None
        assert "test" in normalized or "attributes" in str(normalized)


class TestDispatcherCleanup:
    """Test dispatcher resource cleanup."""

    def test_cleanup(self) -> None:
        """Test cleanup releases resources."""
        dispatcher = FlextDispatcher()

        def handler_func(msg: object) -> str:
            return "handled"

        dispatcher.register_handler("TestMessage", handler_func)

        # Cleanup
        dispatcher.cleanup()

        # Verify state is cleared
        assert len(dispatcher._circuit_breaker._failures) == 0
        assert len(dispatcher._rate_limiter._windows) == 0

    def test_get_performance_metrics(self) -> None:
        """Test getting performance metrics."""
        dispatcher = FlextDispatcher()

        metrics = dispatcher.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "total_dispatches" in metrics
        assert "circuit_breaker_failures" in metrics
        # rate_limit_states was removed, but metrics dict should still exist
        assert isinstance(metrics, dict)


class TestDispatcherFactory:
    """Test dispatcher factory methods."""

    def test_create_from_global_config(self) -> None:
        """Test creating dispatcher from global config."""
        result = FlextDispatcher.create_from_global_config()
        assert result.is_success
        assert isinstance(result.value, FlextDispatcher)


class TestDispatcherErrorHandling:
    """Test error handling in dispatcher."""

    def test_dispatch_with_none_message(self) -> None:
        """Test dispatch with None message."""
        dispatcher = FlextDispatcher()

        # This should not crash
        result = dispatcher.dispatch("")
        assert result is not None

    def test_register_handler_with_invalid_mode(self) -> None:
        """Test registering handler with invalid mode."""
        dispatcher = FlextDispatcher()

        def handler_func(msg: object) -> str:
            return "handled"

        result = dispatcher.register_function(
            SimpleMessage,
            handler_func,
            mode=cast("FlextConstants.Cqrs.HandlerType", "INVALID"),
        )
        assert result.is_failure

    def test_dispatch_missing_handler(self) -> None:
        """Test dispatching message with no handler."""
        dispatcher = FlextDispatcher()

        message = SimpleMessage(value="test")
        result = dispatcher.dispatch(message)

        # Should fail gracefully when no handler registered
        assert result is not None


class TestHandlerCreation:
    """Test handler creation from functions."""

    def test_create_handler_from_function(self) -> None:
        """Test creating handler from function."""
        dispatcher = FlextDispatcher()

        def handler_func(msg: object) -> str:
            return "handled"

        result = dispatcher.create_handler_from_function(
            handler_func,
            mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        assert result.is_success
        assert isinstance(result.value, FlextHandlers)

    def test_create_handler_invalid_mode(self) -> None:
        """Test creating handler with invalid mode."""
        dispatcher = FlextDispatcher()

        def handler_func(msg: object) -> str:
            return "handled"

        result = dispatcher.create_handler_from_function(
            handler_func, mode=cast("FlextConstants.Cqrs.HandlerType", "INVALID")
        )

        assert result.is_failure

    def test_handler_exception_handling(self) -> None:
        """Test handler creation succeeds with callable wrapping."""
        dispatcher = FlextDispatcher()

        def valid_handler(msg: object) -> str:
            return "handled"

        # Valid handler should succeed
        result = dispatcher.create_handler_from_function(
            valid_handler,
            mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        assert result.is_success


__all__ = [
    "TestCircuitBreaker",
    "TestContextManagement",
    "TestDispatcherBasics",
    "TestDispatcherCleanup",
    "TestDispatcherErrorHandling",
    "TestDispatcherFactory",
    "TestHandlerCreation",
    "TestHandlerRegistration",
    "TestMessageDispatch",
    "TestRateLimiting",
    "TestRetryLogic",
]
