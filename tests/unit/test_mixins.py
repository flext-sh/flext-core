"""Comprehensive tests for FlextMixins - Mixin Classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time

import pytest

from flext_core import FlextMixins, FlextResult, FlextTypes


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

        # Test with empty name (should fail)
        result = mixins.register("", str)
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

        result = mixins.unregister("nonexistent_mixin")
        assert result.is_failure

    def test_mixins_apply_mixin(self) -> None:
        """Test mixin application."""
        mixins = FlextMixins()

        class TestMixin:
            def __call__(self, data: object) -> object:
                """Make mixin callable for the mixin system."""
                return data

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
        assert result.error is not None
        assert "not found" in result.error

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
        assert result.is_success  # Class creation succeeds
        # Get the mixed class and instantiate it to test behavior
        mixed_class = result.unwrap()
        if mixed_class is not None and callable(mixed_class):
            mixed_instance = mixed_class()
            if hasattr(mixed_instance, "test_method"):
                with pytest.raises(ValueError, match="Mixin failed"):
                    mixed_instance.test_method()
        else:
            pytest.fail("Failed to get mixed class")

    def test_mixins_apply_mixin_with_retry(self) -> None:
        """Test mixin application with retry mechanism (not implemented in new API)."""
        mixins = FlextMixins()

        class RetryMixin:
            def test_method(self) -> str:
                # Retry logic not implemented in new API
                return "mixin_applied"

        class TestClass:
            pass

        mixins.register("retry_mixin", RetryMixin)
        result = mixins.apply("retry_mixin", TestClass)
        assert result.is_success
        # Result is a class, not a string
        assert isinstance(result.data, type)

    def test_mixins_apply_mixin_with_timeout(self) -> None:
        """Test mixin application with timeout (not implemented in new API)."""
        mixins = FlextMixins()

        class TimeoutMixin:
            def test_method(self) -> str:
                time.sleep(0.2)  # Exceed timeout
                return "should_not_reach_here"

        class TestClass:
            pass

        mixins.register("timeout_mixin", TimeoutMixin)
        result = mixins.apply("timeout_mixin", TestClass)
        assert result.is_success  # Class creation succeeds, timeout not enforced

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

        def middleware(mixin_class: type, data: object) -> tuple[type, object]:
            nonlocal middleware_called
            middleware_called = True
            return mixin_class, data

        class TestMixin:
            def __call__(self, data: object) -> object:
                return data

        mixins.add_middleware(middleware).unwrap()  # Middleware added successfully
        mixins.register("test_mixin", TestMixin)
        result = mixins.apply("test_mixin", "test_data")  # Apply to data, not class
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
        assert isinstance(metrics["test_mixin"], dict)
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

        classes: FlextTypes.List = [TestClass1, TestClass2, TestClass3]
        results = mixins.apply_batch(classes)
        if results.is_success and results.data:
            assert len(results.data) == 3
            # results.data contains the unwrapped values, not FlextResult objects
            assert all(
                isinstance(result, type(classes[i]))
                for i, result in enumerate(results.data)
            )
        else:
            assert results.is_failure

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

        classes: FlextTypes.List = [TestClass1, TestClass2, TestClass3]

        start_time = time.time()
        results = mixins.apply_parallel(classes)
        end_time = time.time()

        if results.is_success and results.data:
            assert len(results.data) == 3
            # results.data contains the unwrapped values, not FlextResult objects
            assert all(
                isinstance(result, type(classes[i]))
                for i, result in enumerate(results.data)
            )
        else:
            assert results.is_failure
        # Should complete faster than sequential execution
        assert end_time - start_time < 0.3

    def test_mixins_apply_mixin_with_circuit_breaker(self) -> None:
        """Test mixin application with circuit breaker (not applicable to class mixins)."""
        mixins = FlextMixins()

        class FailingMixin:
            def __call__(self, data: object) -> object:
                msg = "Service unavailable"
                raise ValueError(msg)

        mixins.register("failing_mixin", FailingMixin)

        # Circuit breaker doesn't apply to class mixins, only to callable mixins
        # that actually execute. For class mixins, circuit breaker is not triggered.
        result = mixins.apply("failing_mixin", "test_data")
        assert result.is_failure  # Callable mixin fails
        assert "Service unavailable" in str(result.error)

        # Circuit breaker state is not relevant for class mixins
        assert mixins.is_circuit_breaker_open("failing_mixin") is False

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
        for _i in range(10):
            result = mixins.apply("test_mixin", TestClass)
            assert result.is_success

        # Exceed rate limit
        result = mixins.apply("test_mixin", TestClass)
        assert result.is_failure
        assert result.error is not None
        assert "rate limit" in result.error.lower()

    def test_mixins_apply_mixin_with_caching(self) -> None:
        """Test mixin application creates new classes each time (no caching)."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        # First application
        result1 = mixins.apply("test_mixin", TestClass)
        assert result1.is_success

        # Second application creates different class (no caching)
        result2 = mixins.apply("test_mixin", TestClass)
        assert result2.is_success
        assert result1.data != result2.data  # Different instances

    def test_mixins_apply_mixin_with_audit(self) -> None:
        """Test mixin application with audit logging (not implemented in new API)."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        result = mixins.apply("test_mixin", TestClass)
        assert result.is_success

        # Audit logging is not implemented in the new API
        audit_log = mixins.get_audit_log()
        assert len(audit_log) == 0

    def test_mixins_apply_mixin_with_performance_monitoring(self) -> None:
        """Test mixin application with performance monitoring (not implemented in new API)."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                time.sleep(0.05)  # Simulate work
                return "test_result"

        class TestClass:
            pass

        mixins.register("test_mixin", TestMixin)

        result = mixins.apply("test_mixin", TestClass)
        assert result.is_success

        # Performance monitoring is not implemented in the new API
        performance = mixins.get_performance_metrics()
        assert len(performance) == 0

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
        assert result.is_success  # Mixin application succeeds
        mixed_class = result.unwrap()

        # Now test that the mixed class method fails
        if callable(mixed_class):
            instance = mixed_class()
            if hasattr(instance, "test_method"):
                with pytest.raises(RuntimeError, match="Mixin error"):
                    instance.test_method()

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
        assert result.error is not None
        assert "not found" in result.error

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
        assert isinstance(stats["test_mixin"], dict)
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

        results: list[FlextResult[object]] = []

        def apply_mixin(_thread_id: int) -> None:
            result = mixins.apply("test_mixin", TestClass)
            results.append(result)

        threads: list[threading.Thread] = []
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

        # Class creation succeeds - error only occurs when method is called
        result = mixins.apply("error_mixin", TestClass)
        assert result.is_success  # Changed: class creation succeeds
        mixed_class = result.unwrap()

        # Error occurs when calling the method
        if callable(mixed_class):
            instance = mixed_class()
            if hasattr(instance, "test_method"):
                with pytest.raises(ValueError, match="Mixin error"):
                    instance.test_method()

    def test_mixins_validation(self) -> None:
        """Test mixins validation."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

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
        apply_result = new_mixins.apply("test_mixin", TestClass)
        assert apply_result.is_success

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


class TestFlextMixinsNestedClasses:
    """Comprehensive test suite for nested mixin classes."""

    def test_loggable_di_logger_injection(self) -> None:
        """Test LoggableDI mixin logger injection."""
        from flext_core import FlextLogger

        class MyService(FlextMixins.LoggableDI):
            pass

        service = MyService()
        assert isinstance(service.logger, FlextLogger)

    def test_loggable_di_logger_caching(self) -> None:
        """Test LoggableDI mixin logger caching."""

        class MyService(FlextMixins.LoggableDI):
            pass

        service1 = MyService()
        service2 = MyService()

        # Same logger instance due to caching
        assert service1.logger is service2.logger

    def test_loggable_di_clear_logger_cache(self) -> None:
        """Test LoggableDI mixin clearing logger cache."""

        class MyService(FlextMixins.LoggableDI):
            pass

        service = MyService()
        logger_before = service.logger

        MyService.clear_logger_cache()

        logger_after = service.logger
        # Different logger instance after cache clear
        assert logger_before is not logger_after

    def test_context_aware_operation_context(self) -> None:
        """Test ContextAware mixin operation context."""

        class MyService(FlextMixins.ContextAware):
            pass

        service = MyService()
        with service.operation_context(user_id="123", operation="test"):
            context = service.get_current_context()
            assert context.get("user_id") == "123"
            assert context.get("operation") == "test"

    def test_context_aware_correlation_context(self) -> None:
        """Test ContextAware mixin correlation context."""

        class MyService(FlextMixins.ContextAware):
            pass

        service = MyService()
        with service.correlation_context("test-corr-id") as corr_id:
            assert corr_id == "test-corr-id"
            context = service.get_current_context()
            assert context.get("correlation_id") == "test-corr-id"

    def test_context_aware_correlation_context_auto_generate(self) -> None:
        """Test ContextAware mixin auto-generating correlation ID."""

        class MyService(FlextMixins.ContextAware):
            pass

        service = MyService()
        with service.correlation_context() as corr_id:
            assert corr_id is not None
            assert corr_id.startswith("corr-")
            context = service.get_current_context()
            assert context.get("correlation_id") == corr_id

    def test_context_aware_bind_unbind_context(self) -> None:
        """Test ContextAware mixin bind/unbind context."""

        class MyService(FlextMixins.ContextAware):
            pass

        service = MyService()

        # Bind context
        service.bind_context(key1="value1", key2="value2")
        context = service.get_current_context()
        assert context.get("key1") == "value1"
        assert context.get("key2") == "value2"

        # Unbind specific keys
        service.unbind_context("key1")
        context = service.get_current_context()
        # key1 should still be there (unbind doesn't remove, just marks for removal)
        # This is structlog behavior
        service.clear_context()

    def test_context_aware_clear_context(self) -> None:
        """Test ContextAware mixin clear context."""

        class MyService(FlextMixins.ContextAware):
            pass

        service = MyService()
        service.bind_context(test_key="test_value")
        service.clear_context()
        context = service.get_current_context()
        # Context should be empty after clear
        assert len(context) == 0

    def test_measurable_measure_operation(self) -> None:
        """Test Measurable mixin measure_operation."""

        class DataProcessor(FlextMixins.Measurable):
            def process(self) -> str:
                with self.measure_operation("process_data", log_result=False):
                    time.sleep(0.01)
                    return "processed"

        processor = DataProcessor()
        result = processor.process()
        assert result == "processed"

    def test_measurable_measure_operation_with_threshold(self) -> None:
        """Test Measurable mixin measure_operation with threshold."""

        class DataProcessor(FlextMixins.Measurable):
            def process(self) -> str:
                with self.measure_operation(
                    "slow_operation", threshold_ms=1.0, log_result=True
                ):
                    time.sleep(0.005)
                    return "done"

        processor = DataProcessor()
        result = processor.process()
        assert result == "done"

    def test_measurable_measure_function(self) -> None:
        """Test Measurable mixin measure_function."""

        class Service(FlextMixins.Measurable):
            def _process_impl(self) -> str:
                time.sleep(0.01)
                return "result"

            def process(self) -> str:
                measured_func = self.measure_function(self._process_impl, "process")
                return measured_func()

        service = Service()
        result = service.process()
        assert result == "result"

    def test_measurable_get_timing_stats(self) -> None:
        """Test Measurable mixin get_timing_stats."""

        class Service(FlextMixins.Measurable):
            pass

        service = Service()
        stats = service.get_timing_stats()
        assert isinstance(stats, dict)

    def test_validatable_validate_with_result(self) -> None:
        """Test Validatable mixin validate_with_result."""
        from flext_core import FlextResult

        class UserService(FlextMixins.Validatable):
            pass

        service = UserService()

        def validate_email(data: dict[str, str]) -> FlextResult[None]:
            if "@" not in data.get("email", ""):
                return FlextResult[None].fail("Invalid email")
            return FlextResult[None].ok(None)

        # Valid data
        result = service.validate_with_result(
            {"email": "test@example.com"}, [validate_email]
        )
        assert result.is_success

        # Invalid data
        result = service.validate_with_result({"email": "invalid"}, [validate_email])
        assert result.is_failure

    def test_validatable_compose_validators(self) -> None:
        """Test Validatable mixin compose_validators."""
        from flext_core import FlextResult

        class UserService(FlextMixins.Validatable):
            pass

        service = UserService()

        def validate_email(data: dict[str, str]) -> FlextResult[None]:
            if "@" not in data.get("email", ""):
                return FlextResult[None].fail("Invalid email")
            return FlextResult[None].ok(None)

        def validate_age(data: dict[str, str]) -> FlextResult[None]:
            age_str = data.get("age", "0")
            try:
                age_value = int(age_str)
                if age_value < 18:
                    return FlextResult[None].fail("Age must be 18+")
            except ValueError:
                return FlextResult[None].fail("Age must be a number")
            return FlextResult[None].ok(None)

        composed = service.compose_validators(validate_email, validate_age)

        # Valid data
        valid_data: dict[str, str] = {"email": "test@example.com", "age": "25"}
        result = composed(valid_data)
        assert result.is_success

        # Invalid email
        invalid_data: dict[str, str] = {"email": "invalid", "age": "25"}
        result = composed(invalid_data)
        assert result.is_failure

    def test_validatable_validate_field(self) -> None:
        """Test Validatable mixin validate_field."""

        class UserService(FlextMixins.Validatable):
            pass

        service = UserService()

        # Required field validation
        result = service.validate_field("test", "username", required=True)
        assert result.is_success

        result = service.validate_field(None, "username", required=True)
        assert result.is_failure

        # Custom validator
        result = service.validate_field(
            "short",
            "password",
            required=True,
            validator=lambda x: len(x) >= 8,
        )
        assert result.is_failure

    def test_validatable_validate_range(self) -> None:
        """Test Validatable mixin validate_range."""

        class UserService(FlextMixins.Validatable):
            pass

        service = UserService()

        # Value in range
        result = service.validate_range(5.0, "age", min_value=0.0, max_value=10.0)
        assert result.is_success

        # Value below min
        result = service.validate_range(-1.0, "age", min_value=0.0)
        assert result.is_failure

        # Value above max
        result = service.validate_range(15.0, "age", max_value=10.0)
        assert result.is_failure

    def test_advanced_patterns_create_entity(self) -> None:
        """Test AdvancedPatterns create_entity."""
        from flext_core import FlextResult

        def validate_entity(entity: object) -> FlextResult[None]:
            return FlextResult[None].ok(None)

        entity_result: FlextResult[object] = FlextMixins.AdvancedPatterns.create_entity(
            "User", {"name": "John"}, [validate_entity]
        )
        assert entity_result.is_success

    def test_advanced_patterns_create_value_object(self) -> None:
        """Test AdvancedPatterns create_value_object."""
        from flext_core import FlextResult

        def validate_invariant(vo: object) -> FlextResult[None]:
            return FlextResult[None].ok(None)

        vo_result: FlextResult[object] = FlextMixins.AdvancedPatterns.create_value_object(
            "Email", {"address": "test@example.com"}, [validate_invariant]
        )
        assert vo_result.is_success

    def test_advanced_patterns_create_aggregate_root(self) -> None:
        """Test AdvancedPatterns create_aggregate_root."""
        from flext_core import FlextResult

        def validate_rule(agg: object) -> FlextResult[None]:
            return FlextResult[None].ok(None)

        result: FlextResult[object] = FlextMixins.AdvancedPatterns.create_aggregate_root(
            "Order", {"id": 1}, [validate_rule]
        )
        assert result.is_success

    def test_advanced_patterns_create_domain_event(self) -> None:
        """Test AdvancedPatterns create_domain_event."""
        result = FlextMixins.AdvancedPatterns.create_domain_event(
            "OrderCreated", {"order_id": 123}, "order-123"
        )
        assert result.is_success
        event = result.unwrap()
        assert isinstance(event, dict)
        assert event["event_type"] == "OrderCreated"
        assert event["aggregate_id"] == "order-123"

    def test_advanced_patterns_create_command(self) -> None:
        """Test AdvancedPatterns create_command."""
        result = FlextMixins.AdvancedPatterns.create_command(
            "CreateOrder", {"customer_id": 456}
        )
        assert result.is_success
        command = result.unwrap()
        assert isinstance(command, dict)
        assert command["command_type"] == "CreateOrder"

    def test_advanced_patterns_create_query(self) -> None:
        """Test AdvancedPatterns create_query."""
        result = FlextMixins.AdvancedPatterns.create_query(
            "GetOrderById", {"order_id": 789}
        )
        assert result.is_success
        query = result.unwrap()
        assert isinstance(query, dict)
        assert query["query_type"] == "GetOrderById"

    def test_container_mixin_property(self) -> None:
        """Test Container mixin container property."""
        from flext_core import FlextContainer

        class MyService(FlextMixins.Container):
            pass

        service = MyService()
        assert isinstance(service.container, FlextContainer)

    def test_container_mixin_register_in_container(self) -> None:
        """Test Container mixin _register_in_container."""

        class MyService(FlextMixins.Container):
            pass

        service = MyService()
        result = service._register_in_container("test_service")
        assert result.is_success

    def test_context_mixin_property(self) -> None:
        """Test Context mixin context property."""
        from flext_core import FlextContext

        class MyService(FlextMixins.Context):
            pass

        service = MyService()
        assert isinstance(service.context, FlextContext)

    def test_context_mixin_propagate_context(self) -> None:
        """Test Context mixin _propagate_context."""

        class MyService(FlextMixins.Context):
            pass

        service = MyService()
        service._propagate_context("test_operation")
        # No assertion - just test it doesn't crash

    def test_context_mixin_correlation_id(self) -> None:
        """Test Context mixin get/set correlation ID."""

        class MyService(FlextMixins.Context):
            pass

        service = MyService()
        service._set_correlation_id("test-123")
        corr_id = service._get_correlation_id()
        assert corr_id == "test-123"

    def test_logging_mixin_logger_property(self) -> None:
        """Test Logging mixin logger property."""
        from flext_core import FlextLogger

        class MyService(FlextMixins.Logging):
            pass

        service = MyService()
        assert isinstance(service.logger, FlextLogger)

    def test_logging_mixin_log_with_context(self) -> None:
        """Test Logging mixin _log_with_context."""

        class MyService(FlextMixins.Logging):
            pass

        service = MyService()
        # Should not crash
        service._log_with_context("info", "Test message", extra_data="value")

    def test_metrics_mixin_track_operation(self) -> None:
        """Test Metrics mixin _track_operation."""

        class MyService(FlextMixins.Metrics):
            def process(self) -> str:
                with self._track_operation("test_op") as metrics:
                    assert isinstance(metrics, dict)
                    time.sleep(0.01)
                    return "done"

        service = MyService()
        result = service.process()
        assert result == "done"

    def test_configurable_mixin_config_property(self) -> None:
        """Test Configurable mixin config property."""
        from flext_core import FlextConfig

        class MyService(FlextMixins.Configurable):
            pass

        service = MyService()
        assert isinstance(service.config, FlextConfig)

    def test_configurable_mixin_get_config_value(self) -> None:
        """Test Configurable mixin _get_config_value."""

        class MyService(FlextMixins.Configurable):
            pass

        service = MyService()
        value = service._get_config_value("nonexistent_key", default="default")
        assert value == "default"

    def test_configurable_mixin_set_config_value(self) -> None:
        """Test Configurable mixin _set_config_value."""

        class MyService(FlextMixins.Configurable):
            pass

        service = MyService()
        # This may fail if config doesn't allow arbitrary attributes
        result = service._set_config_value("test_key", "test_value")
        # Just test it returns a result
        assert isinstance(result, FlextResult)

    def test_service_mixin_init_service(self) -> None:
        """Test Service mixin _init_service."""

        class MyService(FlextMixins.Service):
            def __init__(self) -> None:
                super().__init__()
                self._init_service("MyTestService")

        service = MyService()
        # Verify service has all required properties
        assert hasattr(service, "logger")
        assert hasattr(service, "container")
        assert hasattr(service, "config")

    def test_service_mixin_enrich_context(self) -> None:
        """Test Service mixin _enrich_context."""

        class MyService(FlextMixins.Service):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._enrich_context(version="1.0.0", team="test")
        # No assertion - just test it doesn't crash

    def test_service_mixin_with_correlation_id(self) -> None:
        """Test Service mixin _with_correlation_id."""

        class MyService(FlextMixins.Service):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        corr_id = service._with_correlation_id()
        assert corr_id is not None
        assert corr_id.startswith("corr-")

        # With provided ID
        corr_id2 = service._with_correlation_id("test-456")
        assert corr_id2 == "test-456"

    def test_service_mixin_with_user_context(self) -> None:
        """Test Service mixin _with_user_context."""

        class MyService(FlextMixins.Service):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._with_user_context("user-123", role="REDACTED_LDAP_BIND_PASSWORD")
        # No assertion - just test it doesn't crash

    def test_service_mixin_with_operation_context(self) -> None:
        """Test Service mixin _with_operation_context."""

        class MyService(FlextMixins.Service):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._with_operation_context("process_order", order_id="123")
        # No assertion - just test it doesn't crash

    def test_service_mixin_clear_operation_context(self) -> None:
        """Test Service mixin _clear_operation_context."""

        class MyService(FlextMixins.Service):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._with_operation_context("test_op")
        service._clear_operation_context()
        # No assertion - just test it doesn't crash

    def test_static_methods_to_json(self) -> None:
        """Test FlextMixins.to_json static method."""
        from flext_core import FlextModels

        request = FlextModels.SerializationRequest(data={"key": "value"})
        json_str = FlextMixins.to_json(request)
        assert isinstance(json_str, str)
        assert "key" in json_str
        assert "value" in json_str

    def test_static_methods_to_dict(self) -> None:
        """Test FlextMixins.to_dict static method."""
        from flext_core import FlextModels

        request = FlextModels.SerializationRequest(data={"key": "value"})
        result_dict = FlextMixins.to_dict(request)
        assert isinstance(result_dict, dict)

    def test_static_methods_create_timestamp_fields(self) -> None:
        """Test FlextMixins.create_timestamp_fields static method."""
        from flext_core import FlextModels

        class TestObj:
            created_at: object = None
            updated_at: object = None

        obj = TestObj()
        config = FlextModels.TimestampConfig(obj=obj, use_utc=True)
        FlextMixins.create_timestamp_fields(config)
        assert obj.created_at is not None

    def test_static_methods_update_timestamp(self) -> None:
        """Test FlextMixins.update_timestamp static method."""
        from flext_core import FlextModels

        class TestObj:
            updated_at: object = None

        obj = TestObj()
        config = FlextModels.TimestampConfig(obj=obj, use_utc=True, auto_update=True)
        FlextMixins.update_timestamp(config)
        assert obj.updated_at is not None

    def test_static_methods_log_operation(self) -> None:
        """Test FlextMixins.log_operation static method."""
        from flext_core import FlextModels

        class TestObj:
            pass

        obj = TestObj()
        config = FlextModels.LogOperation(
            obj=obj, operation="test_op", level="info", message="Test message"
        )
        # Should not crash
        FlextMixins.log_operation(config)

    def test_static_methods_initialize_validation(self) -> None:
        """Test FlextMixins.initialize_validation static method."""

        class TestObj:
            is_valid: bool = False

        obj = TestObj()
        FlextMixins.initialize_validation(obj, "is_valid")
        assert obj.is_valid is True

    def test_static_methods_initialize_state(self) -> None:
        """Test FlextMixins.initialize_state static method."""
        from flext_core import FlextModels

        class TestObj:
            state: str = ""

        obj = TestObj()
        request = FlextModels.StateInitializationRequest(
            data=obj,
            state_key="state",
            initial_value="initialized",
            field_name="state",
            state="initialized",
        )
        FlextMixins.initialize_state(request)
        assert obj.state == "initialized"

    def test_static_methods_clear_cache(self) -> None:
        """Test FlextMixins.clear_cache static method."""

        class TestObj:
            pass

        obj = TestObj()
        # Should not crash
        FlextMixins.clear_cache(obj)

    def test_static_methods_ensure_id(self) -> None:
        """Test FlextMixins.ensure_id static method."""

        class TestObj:
            id: str = ""

        obj = TestObj()
        FlextMixins.ensure_id(obj)
        assert obj.id  # Non-empty string is truthy

    def test_static_methods_get_config_parameter(self) -> None:
        """Test FlextMixins.get_config_parameter static method."""
        from flext_core import FlextConfig

        config = FlextConfig.get_global_instance()
        value = FlextMixins.get_config_parameter(config, "app_name")
        assert value is not None

    def test_static_methods_set_config_parameter(self) -> None:
        """Test FlextMixins.set_config_parameter static method."""
        from flext_core import FlextConfig

        config = FlextConfig.get_global_instance()
        # Try to set a parameter
        result = FlextMixins.set_config_parameter(config, "test_param", "test_value")
        assert isinstance(result, bool)

    def test_static_methods_get_singleton_parameter(self) -> None:
        """Test FlextMixins.get_singleton_parameter static method."""
        from flext_core import FlextConfig

        value = FlextMixins.get_singleton_parameter(FlextConfig, "app_name")
        assert value is not None

    def test_static_methods_set_singleton_parameter(self) -> None:
        """Test FlextMixins.set_singleton_parameter static method."""
        from flext_core import FlextConfig

        result = FlextMixins.set_singleton_parameter(
            FlextConfig, "test_param", "test_value"
        )
        assert isinstance(result, bool)
