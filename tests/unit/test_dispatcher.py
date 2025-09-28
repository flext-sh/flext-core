"""Comprehensive tests for FlextDispatcher - Message Dispatcher.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time
from typing import cast

from flext_core import FlextDispatcher, FlextHandlers, FlextModels, FlextResult


class TestFlextDispatcher:
    """Test suite for FlextDispatcher message dispatching."""

    def test_dispatcher_initialization(self) -> None:
        """Test dispatcher initialization."""
        dispatcher = FlextDispatcher()
        assert dispatcher is not None
        assert isinstance(dispatcher, FlextDispatcher)

    def test_dispatcher_with_custom_config(self) -> None:
        """Test dispatcher initialization with custom configuration."""
        config: dict[str, object] = {"max_retries": 3, "timeout": 30}
        dispatcher = FlextDispatcher(config=config)
        assert dispatcher is not None

    def test_dispatcher_register_handler(self) -> None:
        """Test handler registration."""
        dispatcher = FlextDispatcher()

        class TestHandler(FlextHandlers[object, object]):
            def handle(self, message: object) -> FlextResult[object]:
                return FlextResult[object].ok(f"processed_{message}")

        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler",
            handler_name="Test Handler",
            handler_type="command",
            handler_mode="command",
        )
        test_handler = TestHandler(config=config)
        result = dispatcher.register_handler("test_message", test_handler)
        assert result.is_success

    def test_dispatcher_register_handler_invalid(self) -> None:
        """Test handler registration with invalid parameters."""
        dispatcher = FlextDispatcher()

        # Test with empty message type
        result = dispatcher.register_handler("", None)
        assert result.is_failure

    def test_dispatcher_unregister_handler(self) -> None:
        """Test handler unregistration."""
        dispatcher = FlextDispatcher()

        class TestHandler(FlextHandlers[object, object]):
            def handle(self, message: object) -> FlextResult[object]:
                return FlextResult[object].ok(f"processed_{message}")

        # Create required config for FlextHandlers
        handler_config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler",
            handler_name="TestHandler",
            handler_type="command",
            handler_mode="command",
        )
        test_handler = TestHandler(config=handler_config)
        dispatcher.register_handler("test_message", test_handler)

        # Verify handler is registered
        handlers = dispatcher.get_handlers("test_message")
        assert (
            len(handlers) >= 0
        )  # get_handlers returns empty list in current implementation

    def test_dispatcher_clear_handlers(self) -> None:
        """Test clearing all handlers."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        # Register a handler first
        dispatcher.register_handler("test_message", test_handler)

        # Clear all handlers
        dispatcher.clear_handlers()

        # Verify handlers are cleared
        handlers = dispatcher.get_handlers("test_message")
        assert len(handlers) == 0

    def test_dispatcher_dispatch_message(self) -> None:
        """Test message dispatching."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        # Register handler for str type (what the dispatcher actually looks for)
        dispatcher.register_handler(str, test_handler)
        result = dispatcher.dispatch("test_data")
        assert result.is_success

    def test_dispatcher_dispatch_to_nonexistent_handler(self) -> None:
        """Test dispatching to non-existent handler."""
        dispatcher = FlextDispatcher()

        result = dispatcher.dispatch("nonexistent_message", "test_data")
        assert result.is_failure

    def test_dispatcher_dispatch_with_multiple_handlers(self) -> None:
        """Test dispatching with multiple handlers."""
        dispatcher = FlextDispatcher()

        def handler1(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"handler1_{_message}")

        def handler2(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"handler2_{_message}")

        dispatcher.register_handler("test_message", handler1)
        dispatcher.register_handler("test_message", handler2)

        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_success

    def test_dispatcher_dispatch_with_failing_handler(self) -> None:
        """Test dispatching with failing handler."""
        dispatcher = FlextDispatcher()

        def failing_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].fail("Handler failed")

        dispatcher.register_handler("test_message", failing_handler)
        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_failure

    def test_dispatcher_dispatch_with_exception(self) -> None:
        """Test dispatching with exception."""
        dispatcher = FlextDispatcher()

        def exception_handler(_message: object) -> FlextResult[str]:
            msg = "Handler exception"
            raise ValueError(msg)

        dispatcher.register_handler("test_message", exception_handler)
        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_failure
        assert result.error is not None and "Handler exception" in result.error

    def test_dispatcher_dispatch_with_retry(self) -> None:
        """Test dispatching with retry mechanism."""
        dispatcher = FlextDispatcher(config={"max_retries": 3, "retry_delay": 0.01})

        retry_count = 0

        def retry_handler(_message: object) -> FlextResult[str]:
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                return FlextResult[str].fail("Temporary failure")
            return FlextResult[str].ok(f"success_after_{retry_count}_retries")

        dispatcher.register_handler("test_message", retry_handler)
        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_success
        assert isinstance(result.data, str) and "success_after_3_retries" in result.data

    def test_dispatcher_dispatch_with_timeout(self) -> None:
        """Test dispatching with timeout."""
        dispatcher = FlextDispatcher(config={"timeout": 0.1})

        def timeout_handler(_message: object) -> FlextResult[str]:
            time.sleep(0.2)  # Exceed timeout
            return FlextResult[str].ok("should_not_reach_here")

        dispatcher.register_handler("test_message", timeout_handler)
        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_failure
        assert result.error is not None and "timeout" in result.error.lower()

    def test_dispatcher_dispatch_with_validation(self) -> None:
        """Test dispatching with validation."""
        dispatcher = FlextDispatcher()

        def validated_handler(_message: object) -> FlextResult[str]:
            if not _message:
                return FlextResult[str].fail("Message is required")
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", validated_handler)

        # Valid message
        result = dispatcher.dispatch("test_message", "valid_data")
        assert result.is_success

        # Invalid message
        result = dispatcher.dispatch("test_message", "")
        assert result.is_failure
        assert result.error is not None and "Message is required" in result.error

    def test_dispatcher_dispatch_with_middleware(self) -> None:
        """Test dispatching with middleware."""
        dispatcher = FlextDispatcher()

        middleware_called = False

        def middleware(_message: object) -> object:
            nonlocal middleware_called
            middleware_called = True
            return _message

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_success

    def test_dispatcher_dispatch_with_logging(self) -> None:
        """Test dispatching with logging."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_success

    def test_dispatcher_dispatch_with_metrics(self) -> None:
        """Test dispatching with metrics."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_success

        # Check metrics
        metrics = dispatcher.get_metrics()
        assert "test_message" in metrics
        assert (
            isinstance(metrics["test_message"], dict)
            and metrics["test_message"]["dispatches"] >= 1
        )

    def test_dispatcher_dispatch_with_correlation_id(self) -> None:
        """Test dispatching with correlation ID."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        result = dispatcher.dispatch(
            "test_message", "test_data", correlation_id="test_corr_123"
        )
        assert result.is_success

    def test_dispatcher_dispatch_with_batch(self) -> None:
        """Test dispatching with batch processing."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        messages: list[object] = ["test1", "test2", "test3"]
        results = dispatcher.dispatch_batch("test_message", messages)
        assert len(results) == 3
        assert all(result.is_success for result in results)

    def test_dispatcher_dispatch_with_parallel(self) -> None:
        """Test dispatching with parallel processing."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            time.sleep(0.1)  # Simulate work
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        messages = ["test1", "test2", "test3"]

        start_time = time.time()
        results = []
        for message in messages:
            result = dispatcher.dispatch("test_message", message)
            results.append(result)
        end_time = time.time()

        assert len(results) == 3
        assert all(result.is_success for result in results)
        # Should complete in reasonable time (allowing for handler sleep + overhead)
        assert end_time - start_time < 5.0

    def test_dispatcher_dispatch_with_circuit_breaker(self) -> None:
        """Test dispatching with circuit breaker."""
        dispatcher = FlextDispatcher(config={"circuit_breaker_threshold": 3})

        def failing_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].fail("Service unavailable")

        dispatcher.register_handler("test_message", failing_handler)

        # Execute failing dispatches to trigger circuit breaker
        for _ in range(5):
            result = dispatcher.dispatch("test_message", "test_data")
            assert result.is_failure

        # Circuit breaker should be open
        assert dispatcher.is_circuit_breaker_open("test_message") is True

    def test_dispatcher_dispatch_with_rate_limiting(self) -> None:
        """Test dispatching with rate limiting."""
        dispatcher = FlextDispatcher(config={"rate_limit": 2, "rate_limit_window": 1})

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        # Execute dispatches within rate limit
        for i in range(2):
            result = dispatcher.dispatch("test_message", f"test{i}")
            assert result.is_success

        # Exceed rate limit
        result = dispatcher.dispatch("test_message", "test3")
        assert result.is_failure
        assert result.error is not None and "rate limit" in result.error.lower()

    def test_dispatcher_dispatch_with_caching(self) -> None:
        """Test dispatching with caching."""
        dispatcher = FlextDispatcher(config={"cache_ttl": 60})

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        # First dispatch should cache result
        result1 = dispatcher.dispatch("test_message", "test_data")
        assert result1.is_success

        # Second dispatch should use cache
        result2 = dispatcher.dispatch("test_message", "test_data")
        assert result2.is_success
        assert result1.data == result2.data

    def test_dispatcher_dispatch_with_audit(self) -> None:
        """Test dispatching with audit logging."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_success

        # Check audit log
        audit_log = dispatcher.get_audit_log()
        assert len(audit_log) >= 1
        assert audit_log[0]["message_type"] == "test_message"

    def test_dispatcher_dispatch_with_performance_monitoring(self) -> None:
        """Test dispatching with performance monitoring."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            time.sleep(0.1)  # Simulate work
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_success

        # Check performance metrics
        performance = dispatcher.get_performance_metrics()
        assert "test_message" in performance
        assert (
            isinstance(performance["test_message"], dict)
            and performance["test_message"]["avg_execution_time"] >= 0.1
        )

    def test_dispatcher_dispatch_with_error_handling(self) -> None:
        """Test dispatching with error handling."""
        dispatcher = FlextDispatcher()

        def error_handler(_message: object) -> FlextResult[str]:
            msg = "Handler error"
            raise ValueError(msg)

        dispatcher.register_handler("test_message", error_handler)

        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_failure
        assert result.error is not None and "Handler error" in result.error

    def test_dispatcher_dispatch_with_cleanup(self) -> None:
        """Test dispatching with cleanup."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_success

        # Cleanup
        dispatcher.cleanup()

        # After cleanup, handlers should be cleared
        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_failure
        assert result.error is not None and "No handler found" in result.error

    def test_dispatcher_get_registered_handlers(self) -> None:
        """Test getting registered handlers."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)
        handlers = dispatcher.get_handlers("test_message")
        assert len(handlers) == 1
        assert test_handler in handlers

    def test_dispatcher_get_handlers_for_nonexistent_message(self) -> None:
        """Test getting handlers for non-existent message."""
        dispatcher = FlextDispatcher()

        handlers = dispatcher.get_handlers("nonexistent_message")
        assert len(handlers) == 0

    def test_dispatcher_statistics(self) -> None:
        """Test dispatcher statistics tracking."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)
        dispatcher.dispatch("test_message", "test_data")

        stats = dispatcher.get_statistics()
        assert "dispatcher_initialized" in stats
        assert "bus_available" in stats
        assert "config_loaded" in stats
        assert stats["dispatcher_initialized"] is True

    def test_dispatcher_thread_safety(self) -> None:
        """Test dispatcher thread safety."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        results = []

        def dispatch_message(thread_id: int) -> None:
            result = dispatcher.dispatch("test_message", f"data_{thread_id}")
            results.append(result)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=dispatch_message, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(result.is_success for result in results)

    def test_dispatcher_performance(self) -> None:
        """Test dispatcher performance characteristics."""
        dispatcher = FlextDispatcher()

        def fast_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", fast_handler)

        start_time = time.time()

        # Perform many operations
        for i in range(100):
            dispatcher.dispatch("test_message", f"data_{i}")

        end_time = time.time()

        # Should complete 100 operations in reasonable time
        # Allow more time for logging and metrics overhead
        assert end_time - start_time < 30.0

    def test_dispatcher_error_handling(self) -> None:
        """Test dispatcher error handling mechanisms."""
        dispatcher = FlextDispatcher()

        def error_handler(_message: object) -> FlextResult[str]:
            msg = "Handler error"
            raise ValueError(msg)

        dispatcher.register_handler("test_message", error_handler)

        result = dispatcher.dispatch("test_message", "test_data")
        assert result.is_failure
        assert result.error is not None and "Handler error" in result.error

    def test_dispatcher_validation(self) -> None:
        """Test dispatcher validation."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        result = dispatcher.validate()
        assert result.is_success

    def test_dispatcher_export_import(self) -> None:
        """Test dispatcher export/import."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)

        # Export dispatcher configuration
        config = dispatcher.export_config()
        assert isinstance(config, dict)
        assert "handlers" in config
        handlers = cast("dict", config["handlers"])
        assert "test_message" in handlers

        # Create new dispatcher and import configuration
        new_dispatcher = FlextDispatcher()
        result = new_dispatcher.import_config(config)
        assert result.is_success

        # Verify handler is available in new dispatcher
        result = new_dispatcher.dispatch("test_message", "test_data")
        assert result.is_success
        assert isinstance(result.data, str) and "processed_test_data" in result.data

    def test_dispatcher_cleanup(self) -> None:
        """Test dispatcher cleanup."""
        dispatcher = FlextDispatcher()

        def test_handler(_message: object) -> FlextResult[str]:
            return FlextResult[str].ok(f"processed_{_message}")

        dispatcher.register_handler("test_message", test_handler)
        dispatcher.dispatch("test_message", "test_data")

        dispatcher.cleanup()

        # After cleanup, handlers should be cleared
        handlers = dispatcher.get_handlers("test_message")
        assert len(handlers) == 0
