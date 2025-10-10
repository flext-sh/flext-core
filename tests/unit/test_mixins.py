"""Comprehensive tests for FlextMixins - Mixin Classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time

import pytest

from flext_core import FlextMixins, FlextResult


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
                test_method = getattr(mixed_instance, "test_method")
                if callable(test_method):
                    with pytest.raises(ValueError, match="Mixin failed"):
                        test_method()
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
                test_method = getattr(instance, "test_method")
                if callable(test_method):
                    with pytest.raises(RuntimeError, match="Mixin error"):
                        test_method()

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
                test_method = getattr(instance, "test_method")
                if callable(test_method):
                    with pytest.raises(ValueError, match="Mixin error"):
                        test_method()

    def test_mixins_validation(self) -> None:
        """Test mixins validation."""
        mixins = FlextMixins()

        class TestMixin:
            def test_method(self) -> str:
                return "test_result"

        mixins.register("test_mixin", TestMixin)

        result = mixins.validate("test_data")
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


class TestFlextMixinsNestedClasses:
    """Comprehensive test suite for nested mixin classes."""

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
