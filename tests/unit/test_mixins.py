"""Tests for FlextMixins infrastructure - Container, Context, Logging, Metrics, Service.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time

from flext_core import FlextContext, FlextMixins


class TestFlextMixinsNestedClasses:
    """Comprehensive test suite for nested mixin classes."""

    def test_container_mixin_register_in_container(self) -> None:
        """Test Container mixin _register_in_container."""

        class MyService(FlextMixins):
            pass

        service = MyService()
        result = service._register_in_container("test_service")
        assert result.is_success

    def test_context_mixin_property(self) -> None:
        """Test Context mixin context property."""

        class MyService(FlextMixins):
            pass

        service = MyService()

        assert isinstance(service.context, FlextContext)

    def test_context_mixin_propagate_context(self) -> None:
        """Test Context mixin _propagate_context."""

        class MyService(FlextMixins):
            pass

        service = MyService()
        service._propagate_context("test_operation")
        # No assertion - just test it doesn't crash

    def test_context_mixin_correlation_id(self) -> None:
        """Test Context mixin get/set correlation ID."""

        class MyService(FlextMixins):
            pass

        service = MyService()
        service._set_correlation_id("test-123")
        corr_id = service._get_correlation_id()
        assert corr_id == "test-123"

    def test_logging_mixin_log_with_context(self) -> None:
        """Test Logging mixin _log_with_context."""

        class MyService(FlextMixins):
            pass

        service = MyService()
        # Should not crash
        service._log_with_context("info", "Test message", extra_data="value")

    def test_metrics_mixintrack(self) -> None:
        """Test Metrics mixin track."""

        class MyService(FlextMixins):
            def process(self) -> str:
                with self.track("test_op") as metrics:
                    assert isinstance(metrics, dict)
                    time.sleep(0.01)
                    return "done"

        service = MyService()
        result = service.process()
        assert result == "done"

    def test_service_mixin_init_service(self) -> None:
        """Test Service mixin _init_service."""

        class MyService(FlextMixins):
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

        class MyService(FlextMixins):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._enrich_context(version="1.0.0", team="test")
        # No assertion - just test it doesn't crash

    def test_service_mixin_with_operation_context(self) -> None:
        """Test Service mixin _with_operation_context."""

        class MyService(FlextMixins):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._with_operation_context("process_order", order_id="123")
        # No assertion - just test it doesn't crash

    def test_service_mixin_clear_operation_context(self) -> None:
        """Test Service mixin _clear_operation_context."""

        class MyService(FlextMixins):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._with_operation_context("test_op")
        service._clear_operation_context()
        # No assertion - just test it doesn't crash

    def test_model_conversion_to_dict_with_basemodel(self) -> None:
        """Test ModelConversion.to_dict() with Pydantic BaseModel."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = FlextMixins.ModelConversion.to_dict(model)

        assert isinstance(result, dict)
        assert result == {"name": "test", "value": 42}

    def test_model_conversion_to_dict_with_dict(self) -> None:
        """Test ModelConversion.to_dict() with plain dict."""
        input_dict = {"key": "value", "number": 123}
        result = FlextMixins.ModelConversion.to_dict(input_dict)

        assert result is input_dict  # Should return same dict
        assert result == {"key": "value", "number": 123}

    def test_model_conversion_to_dict_with_none(self) -> None:
        """Test ModelConversion.to_dict() with None."""
        result = FlextMixins.ModelConversion.to_dict(None)

        assert isinstance(result, dict)
        assert result == {}  # Should return empty dict

    def test_result_handling_ensure_result_with_raw_value(self) -> None:
        """Test ResultHandling.ensure_result() with raw value."""
        from flext_core import FlextResult

        result = FlextMixins.ResultHandling.ensure_result(42)

        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.value == 42

    def test_result_handling_ensure_result_with_existing_result(self) -> None:
        """Test ResultHandling.ensure_result() with existing FlextResult."""
        from flext_core import FlextResult

        original = FlextResult.ok(100)
        result = FlextMixins.ResultHandling.ensure_result(original)

        assert result is original  # Should return same instance
        assert result.is_success
        assert result.value == 100

    def test_result_handling_ensure_result_preserves_type(self) -> None:
        """Test ResultHandling.ensure_result() preserves generic type."""
        # Test with different types
        int_result = FlextMixins.ResultHandling.ensure_result(42)
        str_result = FlextMixins.ResultHandling.ensure_result("hello")
        list_result = FlextMixins.ResultHandling.ensure_result([1, 2, 3])

        assert int_result.value == 42
        assert str_result.value == "hello"
        assert list_result.value == [1, 2, 3]
