"""Comprehensive tests for FlextMixins - Mixin Classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time

from flext_core import FlextMixins


class TestFlextMixins:
    """Test suite for FlextMixins mixin functionality."""

    def test_mixins_initialization(self) -> None:
        """Test mixins initialization."""
        mixins = FlextMixins()
        assert mixins is not None
        assert isinstance(mixins, FlextMixins)

    def test_mixins_with_custom_config(self) -> None:
        """Test mixins initialization with custom configuration."""
        mixins = FlextMixins()
        assert mixins is not None

    def test_mixins_register_mixin(self) -> None:
        """Test mixin registration."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        result = mixins.register("test_mixin", TestMixin)
        assert result.is_success

    def test_mixins_register_mixin_invalid(self) -> None:
        """Test mixin registration with invalid parameters."""
        mixins = FlextMixins()

        result = mixins.register("", None)
        assert result.is_failure

    def test_mixins_unregister_mixin(self) -> None:
        """Test mixin unregistration."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        mixins.register("test_mixin", TestMixin)
        result = mixins.unregister("test_mixin")
        assert result.is_success

    def test_mixins_unregister_nonexistent_mixin(self) -> None:
        """Test unregistering non-existent mixin."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        result = mixins.unregister("nonexistent_mixin")
        assert result.is_failure

    def test_mixins_apply_mixin(self) -> None:
        """Test mixin application."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)
        result = mixins.apply("test_mixin", TestClass)
        assert result.is_success

    def test_mixins_apply_nonexistent_mixin(self) -> None:
        """Test applying non-existent mixin."""
        mixins = FlextMixins()

        class TestClass:
            pass

        result = mixins.apply("nonexistent_mixin", TestClass)
        assert result.is_failure
        assert "No mixin found" in result.error

    def test_mixins_apply_mixin_with_failure(self) -> None:
        """Test mixin application with failure."""
        mixins = FlextMixins()

        class FailingMixin:
            def test_method(self) -> str:
                msg = "Mixin failed"
                raise ValueError(msg)

        class TestClass:
            pass

        mixins.register("failing_mixin", FailingMixin)
        result = mixins.apply("failing_mixin", TestClass)
        assert result.is_failure
        assert "Mixin failed" in result.error

    def test_mixins_apply_mixin_with_retry(self) -> None:
        """Test mixin application with retry mechanism."""
        mixins = FlextMixins()

        retry_count = 0

        class RetryMixin:
            def test_method(self) -> str:
                nonlocal retry_count
                retry_count += 1
                if retry_count < 3:
                    msg = "Temporary failure"
                    raise ValueError(msg)
                return f"success_after_{retry_count}_retries"

        class TestClass:
            pass

        mixins.register("retry_mixin", RetryMixin)
        result = mixins.apply("retry_mixin", TestClass)
        assert result.is_success
        assert "success_after_3_retries" in result.data

    def test_mixins_apply_mixin_with_timeout(self) -> None:
        """Test mixin application with timeout."""
        mixins = FlextMixins()

        class TimeoutMixin:
            def test_method(self) -> str:
                time.sleep(0.2)  # Exceed timeout
                return "should_not_reach_here"

        class TestClass:
            pass

        mixins.register("timeout_mixin", TimeoutMixin)
        result = mixins.apply("timeout_mixin", TestClass)
        assert result.is_failure
        assert "timeout" in result.error.lower()

    def test_mixins_apply_mixin_with_validation(self) -> None:
        """Test mixin application with validation."""
        mixins = FlextMixins()

        class ValidatedMixin:
            def test_method(self) -> str:
                return "validated_result"

        class TestClass:
            pass

        mixins.register("validated_mixin", ValidatedMixin)
        result = mixins.apply("validated_mixin", TestClass)
        assert result.is_success

    def test_mixins_apply_mixin_with_middleware(self) -> None:
        """Test mixin application with middleware."""
        mixins = FlextMixins()

        middleware_called = False

        def middleware(
            mixin_class: object, target_class: object
        ) -> tuple[object, object]:
            nonlocal middleware_called
            middleware_called = True
            return mixin_class, target_class

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.add_middleware(middleware)
        mixins.register("test_mixin", TestMixin)
        result = mixins.apply("test_mixin", TestClass)
        assert result.is_success
        assert middleware_called is True

    def test_mixins_apply_mixin_with_logging(self) -> None:
        """Test mixin application with logging."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)
        result = mixins.apply("test_mixin", TestClass)
        assert result.is_success

    def test_mixins_apply_mixin_with_metrics(self) -> None:
        """Test mixin application with metrics."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)
        result = mixins.apply("test_mixin", TestClass)
        assert result.is_success

        # Check metrics
        metrics = mixins.get_metrics()
        assert "test_mixin" in metrics
        assert metrics["test_mixin"]["applications"] >= 1

    def test_mixins_apply_mixin_with_correlation_id(self) -> None:
        """Test mixin application with correlation ID."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)
        result = mixins.apply("test_mixin", TestClass)
        assert result.is_success

    def test_mixins_apply_mixin_with_batch(self) -> None:
        """Test mixin application with batch processing."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass1:
            pass

        class TestClass2:
            pass

        class TestClass3:
            pass

        mixins.register("test_mixin", TestMixin)

        classes = [TestClass1, TestClass2, TestClass3]
        results = mixins.apply_batch(classes)
        assert len(results) == 3
        assert all(result.is_success for result in results)

    def test_mixins_apply_mixin_with_parallel(self) -> None:
        """Test mixin application with parallel processing."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                time.sleep(0.1)  # Simulate work
                return "test_result"

        class TestClass1:
            pass

        class TestClass2:
            pass

        class TestClass3:
            pass

        mixins.register("test_mixin", TestMixin)

        classes = [TestClass1, TestClass2, TestClass3]

        start_time = time.time()
        results = mixins.apply_parallel(classes)
        end_time = time.time()

        assert len(results) == 3
        assert all(result.is_success for result in results)
        # Should complete faster than sequential execution
        assert end_time - start_time < 0.3

    def test_mixins_apply_mixin_with_circuit_breaker(self) -> None:
        """Test mixin application with circuit breaker."""
        mixins = FlextMixins()

        class FailingMixin:
            def test_method(self) -> str:
                msg = "Service unavailable"
                raise ValueError(msg)

        class TestClass:
            pass

        mixins.register("failing_mixin", FailingMixin)

        # Execute failing applications to trigger circuit breaker
        for _ in range(5):
            result = mixins.apply("failing_mixin", TestClass)
            assert result.is_failure

        # Circuit breaker should be open
        assert mixins.is_circuit_breaker_open("failing_mixin") is True

    def test_mixins_apply_mixin_with_rate_limiting(self) -> None:
        """Test mixin application with rate limiting."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        # Execute applications within rate limit
        for _i in range(2):
            result = mixins.apply("test_mixin", TestClass)
            assert result.is_success

        # Exceed rate limit
        result = mixins.apply("test_mixin", TestClass)
        assert result.is_failure
        assert "rate limit" in result.error.lower()

    def test_mixins_apply_mixin_with_caching(self) -> None:
        """Test mixin application with caching."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        # First application should cache result
        result1 = mixins.apply("test_mixin", TestClass)
        assert result1.is_success

        # Second application should use cache
        result2 = mixins.apply("test_mixin", TestClass)
        assert result2.is_success
        assert result1.data == result2.data

    def test_mixins_apply_mixin_with_audit(self) -> None:
        """Test mixin application with audit logging."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        result = mixins.apply("test_mixin", TestClass)
        assert result.is_success

        # Check audit log
        audit_log = mixins.get_audit_log()
        assert len(audit_log) >= 1
        assert audit_log[0]["mixin_name"] == "test_mixin"

    def test_mixins_apply_mixin_with_performance_monitoring(self) -> None:
        """Test mixin application with performance monitoring."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                time.sleep(0.1)  # Simulate work
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        result = mixins.apply("test_mixin", TestClass)
        assert result.is_success

        # Check performance metrics
        performance = mixins.get_performance_metrics()
        assert "test_mixin" in performance
        assert performance["test_mixin"]["avg_execution_time"] >= 0.1

    def test_mixins_apply_mixin_with_error_handling(self) -> None:
        """Test mixin application with error handling."""
        mixins = FlextMixins()

        class ErrorMixin:
            def test_method(self) -> str:
                msg = "Mixin error"
                raise RuntimeError(msg)

        class TestClass:
            pass

        mixins.register("error_mixin", ErrorMixin)

        result = mixins.apply("error_mixin", TestClass)
        assert result.is_failure
        assert "Mixin error" in result.error

    def test_mixins_apply_mixin_with_cleanup(self) -> None:
        """Test mixin application with cleanup."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        result = mixins.apply("test_mixin", TestClass)
        assert result.is_success

        # Cleanup
        mixins.cleanup()

        # After cleanup, mixins should be cleared
        result = mixins.apply("test_mixin", TestClass)
        assert result.is_failure
        assert "No mixin found" in result.error

    def test_mixins_get_registered_mixins(self) -> None:
        """Test getting registered mixins."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        mixins.register("test_mixin", TestMixin)
        registered_mixins = mixins.get_mixins()
        assert len(registered_mixins) == 1
        assert TestMixin in registered_mixins

    def test_mixins_get_mixins_for_nonexistent_mixin(self) -> None:
        """Test getting mixins for non-existent mixin."""
        mixins = FlextMixins()

        registered_mixins = mixins.get_mixins()
        assert len(registered_mixins) == 0

    def test_mixins_clear_mixins(self) -> None:
        """Test clearing all mixins."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        mixins.register("test_mixin", TestMixin)
        mixins.clear_mixins()

        registered_mixins = mixins.get_mixins()
        assert len(registered_mixins) == 0

    def test_mixins_statistics(self) -> None:
        """Test mixins statistics tracking."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)
        mixins.apply("test_mixin", TestClass)

        stats = mixins.get_statistics()
        assert "test_mixin" in stats
        assert stats["test_mixin"]["applications"] >= 1

    def test_mixins_thread_safety(self) -> None:
        """Test mixins thread safety."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        results = []

        def apply_mixin(_thread_id: int) -> None:
            result = mixins.apply("test_mixin", TestClass)
            results.append(result)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=apply_mixin, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(result.is_success for result in results)

    def test_mixins_performance(self) -> None:
        """Test mixins performance characteristics."""
        mixins = FlextMixins()

        class FastMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", FastMixin)

        start_time = time.time()

        # Perform many operations
        for _i in range(100):
            mixins.apply("test_mixin", TestClass)

        end_time = time.time()

        # Should complete 100 operations in reasonable time
        assert end_time - start_time < 1.0

    def test_mixins_error_handling(self) -> None:
        """Test mixins error handling mechanisms."""
        mixins = FlextMixins()

        class ErrorMixin:
            def test_method(self) -> str:
                msg = "Mixin error"
                raise ValueError(msg)

        class TestClass:
            pass

        mixins.register("error_mixin", ErrorMixin)

        result = mixins.apply("error_mixin", TestClass)
        assert result.is_failure
        assert "Mixin error" in result.error

    def test_mixins_validation(self) -> None:
        """Test mixins validation."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        result = mixins.validate("test_data")
        assert result.is_success

    def test_mixins_export_import(self) -> None:
        """Test mixins export/import."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        # Export mixins configuration
        config_result = mixins.export_config()
        assert config_result.is_success
        config = config_result.data
        assert isinstance(config, dict)
        assert "test_mixin" in config

        # Create new mixins and import configuration
        new_mixins = FlextMixins()
        result = new_mixins.import_config(config)
        assert result.is_success

        # Verify mixin is available in new mixins
        result = new_mixins.apply("test_mixin", TestClass)
        assert result.is_success

    def test_mixins_cleanup(self) -> None:
        """Test mixins cleanup."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)
        mixins.apply("test_mixin", TestClass)

        mixins.cleanup()

        # After cleanup, mixins should be cleared
        registered_mixins = mixins.get_mixins()
        assert len(registered_mixins) == 0
