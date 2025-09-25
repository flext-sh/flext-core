"""Comprehensive tests for FlextExceptions - Exception Handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time

from flext_core import FlextExceptions, FlextResult


class TestFlextExceptions:
    """Test suite for FlextExceptions exception handling."""

    def test_exceptions_initialization(self) -> None:
        """Test exceptions initialization."""
        exceptions = FlextExceptions()
        assert exceptions is not None
        assert isinstance(exceptions, FlextExceptions)

    def test_exceptions_with_custom_config(self) -> None:
        """Test exceptions initialization with custom configuration."""
        config = {"max_retries": 3, "timeout": 30}
        exceptions = FlextExceptions(config=config)
        assert exceptions is not None

    def test_exceptions_register_handler(self) -> None:
        """Test exception handler registration."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        result = exceptions.register_handler(ValueError, test_handler)
        assert result.is_success

    def test_exceptions_register_handler_invalid(self) -> None:
        """Test exception handler registration with invalid parameters."""
        exceptions = FlextExceptions()

        result = exceptions.register_handler("", object())
        assert result.is_failure

    def test_exceptions_unregister_handler(self) -> None:
        """Test exception handler unregistration."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)
        result = exceptions.unregister_handler(ValueError, test_handler)
        assert result.is_success

    def test_exceptions_unregister_nonexistent_handler(self) -> None:
        """Test unregistering non-existent handler."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        result = exceptions.unregister_handler(ValueError, test_handler)
        assert result.is_failure

    def test_exceptions_handle_exception(self) -> None:
        """Test exception handling."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_success
        assert "handled_ValueError" in result.data

    def test_exceptions_handle_unhandled_exception(self) -> None:
        """Test handling unhandled exception."""
        exceptions = FlextExceptions()

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_failure
        assert "No handler found" in result.error

    def test_exceptions_handle_exception_with_failing_handler(self) -> None:
        """Test exception handling with failing handler."""
        exceptions = FlextExceptions()

        def failing_handler(_exc: Exception) -> FlextResult[str]:
            return FlextResult[str].fail("Handler failed")

        exceptions.register_handler(ValueError, failing_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_failure
        assert "Handler failed" in result.error

    def test_exceptions_handle_exception_with_exception(self) -> None:
        """Test exception handling with exception in handler."""
        exceptions = FlextExceptions()

        def exception_handler(_exc: Exception) -> FlextResult[str]:
            msg = "Handler exception"
            raise RuntimeError(msg)

        exceptions.register_handler(ValueError, exception_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_failure
        assert "Handler exception" in result.error

    def test_exceptions_handle_exception_with_retry(self) -> None:
        """Test exception handling with retry mechanism."""
        exceptions = FlextExceptions(config={"max_retries": 3, "retry_delay": 0.01})

        retry_count = 0

        def retry_handler(_exc: Exception) -> FlextResult[str]:
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                return FlextResult[str].fail("Temporary failure")
            return FlextResult[str].ok(f"success_after_{retry_count}_retries")

        exceptions.register_handler(ValueError, retry_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_success
        assert "success_after_3_retries" in result.data

    def test_exceptions_handle_exception_with_timeout(self) -> None:
        """Test exception handling with timeout."""
        exceptions = FlextExceptions(config={"timeout": 0.1})

        def timeout_handler(_exc: Exception) -> FlextResult[str]:
            time.sleep(0.2)  # Exceed timeout
            return FlextResult[str].ok("should_not_reach_here")

        exceptions.register_handler(ValueError, timeout_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_failure
        assert "timeout" in result.error.lower()

    def test_exceptions_handle_exception_with_validation(self) -> None:
        """Test exception handling with validation."""
        exceptions = FlextExceptions()

        def validated_handler(exc: Exception) -> FlextResult[str]:
            if not exc:
                return FlextResult[str].fail("Exception is required")
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, validated_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_success

    def test_exceptions_handle_exception_with_middleware(self) -> None:
        """Test exception handling with middleware."""
        exceptions = FlextExceptions()

        middleware_called = False

        def middleware(exc: Exception) -> Exception:
            nonlocal middleware_called
            middleware_called = True
            return exc

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.add_middleware(middleware)
        exceptions.register_handler(ValueError, test_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_success
        assert middleware_called is True

    def test_exceptions_handle_exception_with_logging(self) -> None:
        """Test exception handling with logging."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_success

    def test_exceptions_handle_exception_with_metrics(self) -> None:
        """Test exception handling with metrics."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_success

        # Check metrics
        metrics = exceptions.get_metrics()
        assert "ValueError" in metrics
        assert metrics["ValueError"]["handled"] >= 1

    def test_exceptions_handle_exception_with_correlation_id(self) -> None:
        """Test exception handling with correlation ID."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc, correlation_id="test_corr_123")
        assert result.is_success

    def test_exceptions_handle_exception_with_batch(self) -> None:
        """Test exception handling with batch processing."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exceptions_list = [
            ValueError("Test error 1"),
            ValueError("Test error 2"),
            ValueError("Test error 3"),
        ]

        results = exceptions.handle_batch(exceptions_list)
        assert len(results) == 3
        assert all(result.is_success for result in results)

    def test_exceptions_handle_exception_with_parallel(self) -> None:
        """Test exception handling with parallel processing."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            time.sleep(0.1)  # Simulate work
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exceptions_list = [
            ValueError("Test error 1"),
            ValueError("Test error 2"),
            ValueError("Test error 3"),
        ]

        start_time = time.time()
        results = exceptions.handle_parallel(exceptions_list)
        end_time = time.time()

        assert len(results) == 3
        assert all(result.is_success for result in results)
        # Should complete faster than sequential execution
        assert end_time - start_time < 0.3

    def test_exceptions_handle_exception_with_circuit_breaker(self) -> None:
        """Test exception handling with circuit breaker."""
        exceptions = FlextExceptions(config={"circuit_breaker_threshold": 3})

        def failing_handler(_exc: Exception) -> FlextResult[str]:
            return FlextResult[str].fail("Service unavailable")

        exceptions.register_handler(ValueError, failing_handler)

        # Execute failing handlers to trigger circuit breaker
        for _ in range(5):
            exc = ValueError("Test error")
            result = exceptions.handle_exception(exc)
            assert result.is_failure

        # Circuit breaker should be open
        assert exceptions.is_circuit_breaker_open("ValueError") is True

    def test_exceptions_handle_exception_with_rate_limiting(self) -> None:
        """Test exception handling with rate limiting."""
        exceptions = FlextExceptions(config={"rate_limit": 2, "rate_limit_window": 1})

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        # Execute handlers within rate limit
        for i in range(2):
            exc = ValueError(f"Test error {i}")
            result = exceptions.handle_exception(exc)
            assert result.is_success

        # Exceed rate limit
        exc = ValueError("Test error 3")
        result = exceptions.handle_exception(exc)
        assert result.is_failure
        assert "rate limit" in result.error.lower()

    def test_exceptions_handle_exception_with_caching(self) -> None:
        """Test exception handling with caching."""
        exceptions = FlextExceptions(config={"cache_ttl": 60})

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        # First handling should cache result
        exc = ValueError("Test error")
        result1 = exceptions.handle_exception(exc)
        assert result1.is_success

        # Second handling should use cache
        result2 = exceptions.handle_exception(exc)
        assert result2.is_success
        assert result1.data == result2.data

    def test_exceptions_handle_exception_with_audit(self) -> None:
        """Test exception handling with audit logging."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_success

        # Check audit log
        audit_log = exceptions.get_audit_log()
        assert len(audit_log) >= 1
        assert audit_log[0]["exception_type"] == "ValueError"

    def test_exceptions_handle_exception_with_performance_monitoring(self) -> None:
        """Test exception handling with performance monitoring."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            time.sleep(0.1)  # Simulate work
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_success

        # Check performance metrics
        performance = exceptions.get_performance_metrics()
        assert "ValueError" in performance
        assert performance["ValueError"]["avg_execution_time"] >= 0.1

    def test_exceptions_handle_exception_with_error_handling(self) -> None:
        """Test exception handling with error handling."""
        exceptions = FlextExceptions()

        def error_handler(_exc: Exception) -> FlextResult[str]:
            msg = "Handler error"
            raise RuntimeError(msg)

        exceptions.register_handler(ValueError, error_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_failure
        assert "Handler error" in result.error

    def test_exceptions_handle_exception_with_cleanup(self) -> None:
        """Test exception handling with cleanup."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_success

        # Cleanup
        exceptions.cleanup()

        # After cleanup, handlers should be cleared
        result = exceptions.handle_exception(exc)
        assert result.is_failure
        assert "No handler found" in result.error

    def test_exceptions_get_registered_handlers(self) -> None:
        """Test getting registered handlers."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)
        handlers = exceptions.get_handlers(ValueError)
        assert len(handlers) == 1
        assert test_handler in handlers

    def test_exceptions_get_handlers_for_nonexistent_exception(self) -> None:
        """Test getting handlers for non-existent exception."""
        exceptions = FlextExceptions()

        handlers = exceptions.get_handlers(ValueError)
        assert len(handlers) == 0

    def test_exceptions_clear_handlers(self) -> None:
        """Test clearing all handlers."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)
        exceptions.clear_handlers()

        handlers = exceptions.get_handlers(ValueError)
        assert len(handlers) == 0

    def test_exceptions_statistics(self) -> None:
        """Test exceptions statistics tracking."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exc = ValueError("Test error")
        exceptions.handle_exception(exc)

        stats = exceptions.get_statistics()
        assert "ValueError" in stats
        assert stats["ValueError"]["handled"] >= 1

    def test_exceptions_thread_safety(self) -> None:
        """Test exceptions thread safety."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        results = []

        def handle_exception(thread_id: int) -> None:
            exc = ValueError(f"Test error {thread_id}")
            result = exceptions.handle_exception(exc)
            results.append(result)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=handle_exception, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(result.is_success for result in results)

    def test_exceptions_performance(self) -> None:
        """Test exceptions performance characteristics."""
        exceptions = FlextExceptions()

        def fast_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, fast_handler)

        start_time = time.time()

        # Perform many operations
        for i in range(100):
            exc = ValueError(f"Test error {i}")
            exceptions.handle_exception(exc)

        end_time = time.time()

        # Should complete 100 operations in reasonable time
        assert end_time - start_time < 1.0

    def test_exceptions_error_handling(self) -> None:
        """Test exceptions error handling mechanisms."""
        exceptions = FlextExceptions()

        def error_handler(_exc: Exception) -> FlextResult[str]:
            msg = "Handler error"
            raise RuntimeError(msg)

        exceptions.register_handler(ValueError, error_handler)

        exc = ValueError("Test error")
        result = exceptions.handle_exception(exc)
        assert result.is_failure
        assert "Handler error" in result.error

    def test_exceptions_validation(self) -> None:
        """Test exceptions validation."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        result = exceptions.validate()
        assert result.is_success

    def test_exceptions_export_import(self) -> None:
        """Test exceptions export/import."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        # Export exceptions configuration
        config = exceptions.export_config()
        assert isinstance(config, dict)
        assert "ValueError" in config

        # Create new exceptions and import configuration
        new_exceptions = FlextExceptions()
        result = new_exceptions.import_config(config)
        assert result.is_success

        # Verify handler is available in new exceptions
        exc = ValueError("Test error")
        result = new_exceptions.handle_exception(exc)
        assert result.is_success
        assert "handled_ValueError" in result.data

    def test_exceptions_cleanup(self) -> None:
        """Test exceptions cleanup."""
        exceptions = FlextExceptions()

        def test_handler(exc: Exception) -> FlextResult[str]:
            return FlextResult[str].ok(f"handled_{type(exc).__name__}")

        exceptions.register_handler(ValueError, test_handler)

        exc = ValueError("Test error")
        exceptions.handle_exception(exc)

        exceptions.cleanup()

        # After cleanup, handlers should be cleared
        handlers = exceptions.get_handlers(ValueError)
        assert len(handlers) == 0
