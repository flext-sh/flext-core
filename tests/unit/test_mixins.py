"""Tests for FlextCore.Mixins infrastructure - Container, Context, Logging, Metrics, Service.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time

from flext_core import FlextCore


class TestFlextMixinsNestedClasses:
    """Comprehensive test suite for nested mixin classes."""

    def test_container_mixin_register_in_container(self) -> None:
        """Test Container mixin _register_in_container."""

        class MyService(FlextCore.Mixins):
            pass

        service = MyService()
        result = service._register_in_container("test_service")
        assert result.is_success

    def test_context_mixin_property(self) -> None:
        """Test Context mixin context property."""
        from flext_core import FlextCore

        class MyService(FlextCore.Mixins):
            pass

        service = MyService()
        from flext_core.context import FlextContext

        assert isinstance(service.context, FlextContext)

    def test_context_mixin_propagate_context(self) -> None:
        """Test Context mixin _propagate_context."""

        class MyService(FlextCore.Mixins):
            pass

        service = MyService()
        service._propagate_context("test_operation")
        # No assertion - just test it doesn't crash

    def test_context_mixin_correlation_id(self) -> None:
        """Test Context mixin get/set correlation ID."""

        class MyService(FlextCore.Mixins):
            pass

        service = MyService()
        service._set_correlation_id("test-123")
        corr_id = service._get_correlation_id()
        assert corr_id == "test-123"

    def test_logging_mixin_log_with_context(self) -> None:
        """Test Logging mixin _log_with_context."""

        class MyService(FlextCore.Mixins):
            pass

        service = MyService()
        # Should not crash
        service._log_with_context("info", "Test message", extra_data="value")

    def test_metrics_mixintrack(self) -> None:
        """Test Metrics mixin track."""

        class MyService(FlextCore.Mixins):
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

        class MyService(FlextCore.Mixins):
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

        class MyService(FlextCore.Mixins):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._enrich_context(version="1.0.0", team="test")
        # No assertion - just test it doesn't crash

    def test_service_mixin_with_operation_context(self) -> None:
        """Test Service mixin _with_operation_context."""

        class MyService(FlextCore.Mixins):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._with_operation_context("process_order", order_id="123")
        # No assertion - just test it doesn't crash

    def test_service_mixin_clear_operation_context(self) -> None:
        """Test Service mixin _clear_operation_context."""

        class MyService(FlextCore.Mixins):
            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._with_operation_context("test_op")
        service._clear_operation_context()
        # No assertion - just test it doesn't crash
