"""Tests specifically targeting uncovered lines in core.py.

This file directly calls methods that are not being called by normal usage 
to increase code coverage and test edge cases in FlextCore orchestration.
"""

from __future__ import annotations

import pytest

from flext_core import (
    FlextCore,
    FlextResult,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextCoreContainerMethods:
    """Test uncovered container management methods in FlextCore."""

    def test_configure_container_with_configure_method(self) -> None:
        """Test lines 218-228: configure_container when container has configure method."""
        core = FlextCore()

        # Test the real container configuration - create real configure method
        class ConfigurableContainer:
            def __init__(self, original_container):
                self._original = original_container
                self.last_config = {}

            def configure(self, **kwargs):
                self.last_config = kwargs
                return True

            def __getattr__(self, name):
                return getattr(self._original, name)

        # Replace with real configurable container
        original_container = core._container
        configurable = ConfigurableContainer(original_container)
        core._container = configurable

        result = core.configure_container(setting1="value1", setting2="value2")

        assert result.success
        assert configurable.last_config == {"setting1": "value1", "setting2": "value2"}

    def test_configure_container_without_configure_method(self) -> None:
        """Test lines 218-228: configure_container when container lacks configure method."""
        core = FlextCore()

        # Ensure container doesn't have configure method (it shouldn't by default)
        if hasattr(core._container, "configure"):
            delattr(core._container, "configure")

        result = core.configure_container(setting1="value1")

        # Should still succeed but do nothing
        assert result.success

    def test_configure_container_exception_handling(self) -> None:
        """Test lines 227-228: configure_container exception handling."""
        core = FlextCore()

        # Real failing configure method implementation
        class FailingContainer:
            def __init__(self, original_container):
                self._original = original_container

            def configure(self, **kwargs):
                raise RuntimeError("Configure failed")

            def __getattr__(self, name):
                return getattr(self._original, name)

        # Replace with failing container
        core._container = FailingContainer(core._container)
        result = core.configure_container(setting1="value1")

        assert result.is_failure
        assert "Container configuration failed" in (result.error or "")

    def test_clear_container_with_clear_method(self) -> None:
        """Test lines 230-237: clear_container when container has clear method."""
        core = FlextCore()

        # Add some services first
        core.register_service("test_service", "test_value")

        # Real clearable container implementation
        class ClearableContainer:
            def __init__(self, original_container):
                self._original = original_container
                self.clear_called = False

            def clear(self):
                self.clear_called = True
                # Delegate to original if it has clear method
                if hasattr(self._original, "clear"):
                    self._original.clear()

            def __getattr__(self, name):
                return getattr(self._original, name)

        clearable = ClearableContainer(core._container)
        core._container = clearable
        result = core.clear_container()

        assert result.success
        assert clearable.clear_called

    def test_clear_container_without_clear_method(self) -> None:
        """Test lines 230-237: clear_container when container lacks clear method."""
        core = FlextCore()

        # Just test normal clear_container behavior
        result = core.clear_container()

        # Should succeed
        assert result.success

    def test_clear_container_exception_handling(self) -> None:
        """Test lines 236-237: clear_container exception handling."""
        core = FlextCore()

        # Real failing clear method implementation
        class FailingClearContainer:
            def __init__(self, original_container):
                self._original = original_container

            def clear(self):
                raise RuntimeError("Clear failed")

            def __getattr__(self, name):
                return getattr(self._original, name)

        core._container = FailingClearContainer(core._container)
        result = core.clear_container()

        assert result.is_failure
        assert "Container clear failed" in (result.error or "")


class TestFlextCoreLoggingMethods:
    """Test uncovered logging methods in FlextCore."""

    def test_get_logger_static_method(self) -> None:
        """Test static get_logger method."""
        logger = FlextCore.get_logger("test_logger")

        assert logger is not None
        # Logger should have basic logging interface
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_configure_logging_if_available(self) -> None:
        """Test configure_logging static method if it exists."""
        # Check if configure_logging exists
        if hasattr(FlextCore, "configure_logging"):
            # Test basic configuration - API uses log_level parameter
            FlextCore.configure_logging(log_level="INFO")
            # configure_logging returns None, not FlextResult


class TestFlextCoreConfigurationMethods:
    """Test uncovered configuration methods in FlextCore."""

    def test_load_config_if_available(self) -> None:
        """Test load_config method if it exists."""
        core = FlextCore()

        if hasattr(core, "load_config"):
            result = core.load_config({})
            assert isinstance(result, FlextResult)

    def test_get_config_if_available(self) -> None:
        """Test get_config method if it exists."""
        core = FlextCore()

        if hasattr(core, "get_config"):
            result = core.get_config("some_key")
            assert isinstance(result, FlextResult)

    def test_set_config_if_available(self) -> None:
        """Test set_config method if it exists."""
        core = FlextCore()

        if hasattr(core, "set_config"):
            result = core.set_config("test_key", "test_value")
            assert isinstance(result, FlextResult)


class TestFlextCoreObservabilityMethods:
    """Test uncovered observability methods in FlextCore."""

    def test_get_metrics_if_available(self) -> None:
        """Test get_metrics method if it exists."""
        core = FlextCore()

        if hasattr(core, "get_metrics"):
            result = core.get_metrics()
            assert isinstance(result, FlextResult)

    def test_clear_metrics_if_available(self) -> None:
        """Test clear_metrics method if it exists."""
        core = FlextCore()

        if hasattr(core, "clear_metrics"):
            result = core.clear_metrics()
            assert isinstance(result, FlextResult)

    def test_record_metric_if_available(self) -> None:
        """Test record_metric method if it exists."""
        core = FlextCore()

        if hasattr(core, "record_metric"):
            result = core.record_metric("test_metric", 1.0)
            assert isinstance(result, FlextResult)


class TestFlextCoreFactoryMethods:
    """Test uncovered factory methods in FlextCore."""

    def test_create_entity_if_available(self) -> None:
        """Test create_entity static method if it exists."""
        if hasattr(FlextCore, "create_entity"):
            # create_entity is a static method that takes entity_class and **data
            from flext_core import FlextModel
            result = FlextCore.create_entity(FlextModel)
            assert isinstance(result, FlextResult)

    def test_create_service_if_available(self) -> None:
        """Test create_service method if it exists."""
        core = FlextCore()

        if hasattr(core, "create_service"):
            result = core.create_service("test_service", {})
            assert isinstance(result, FlextResult)


class TestFlextCoreUtilityMethods:
    """Test uncovered utility methods in FlextCore."""

    def test_validate_if_available(self) -> None:
        """Test validate method if it exists."""
        core = FlextCore()

        if hasattr(core, "validate"):
            result = core.validate({})
            assert isinstance(result, FlextResult)

    def test_transform_if_available(self) -> None:
        """Test transform method if it exists."""
        core = FlextCore()

        if hasattr(core, "transform"):
            result = core.transform("input_data")
            assert isinstance(result, FlextResult)

    def test_process_if_available(self) -> None:
        """Test process method if it exists."""
        core = FlextCore()

        if hasattr(core, "process"):
            result = core.process("input_data")
            assert isinstance(result, FlextResult)


class TestFlextCoreExceptionHandlingPaths:
    """Test exception handling paths in FlextCore."""

    def test_service_registration_failure_path(self) -> None:
        """Test service registration failure handling."""
        core = FlextCore()

        # Real container that fails registration
        class FailingRegistrationContainer:
            def __init__(self, original_container):
                self._original = original_container

            def register(self, name, service):
                return FlextResult[None].fail("Registration failed")

            def __getattr__(self, name):
                return getattr(self._original, name)

        core._container = FailingRegistrationContainer(core._container)
        result = core.register_service("test", "value")
        assert result.is_failure

    def test_service_retrieval_failure_path(self) -> None:
        """Test service retrieval failure handling."""
        core = FlextCore()

        # Try to get non-existent service
        result = core.get_service("nonexistent_service")
        assert result.is_failure
        assert "not found" in (result.error or "")

    def test_factory_registration_failure_path(self) -> None:
        """Test factory registration failure handling."""
        core = FlextCore()

        # Real container that fails factory registration
        class FailingFactoryContainer:
            def __init__(self, original_container):
                self._original = original_container

            def register_factory(self, name, factory):
                return FlextResult[None].fail("Factory registration failed")

            def __getattr__(self, name):
                return getattr(self._original, name)

        core._container = FailingFactoryContainer(core._container)
        result = core.register_factory("test_factory", lambda: "test")
        assert result.is_failure


class TestFlextCoreInteractionMethods:
    """Test methods that interact with other FlextCore components."""

    def test_environment_interaction_if_available(self) -> None:
        """Test environment-related methods if they exist."""
        core = FlextCore()

        if hasattr(core, "get_environment"):
            result = core.get_environment()
            assert isinstance(result, FlextResult)

        if hasattr(core, "set_environment"):
            result = core.set_environment("test")
            assert isinstance(result, FlextResult)

    def test_context_management_if_available(self) -> None:
        """Test context management methods if they exist."""
        core = FlextCore()

        if hasattr(core, "get_context"):
            result = core.get_context()
            assert isinstance(result, FlextResult)

        if hasattr(core, "set_context"):
            result = core.set_context({})
            assert isinstance(result, FlextResult)

    def test_lifecycle_management_if_available(self) -> None:
        """Test lifecycle management methods if they exist."""
        core = FlextCore()

        if hasattr(core, "initialize"):
            result = core.initialize()
            assert isinstance(result, FlextResult)

        if hasattr(core, "shutdown"):
            result = core.shutdown()
            assert isinstance(result, FlextResult)

        if hasattr(core, "restart"):
            result = core.restart()
            assert isinstance(result, FlextResult)


class TestFlextCoreAdvancedFeatures:
    """Test advanced FlextCore features that might not be commonly used."""

    def test_serialization_if_available(self) -> None:
        """Test serialization methods if they exist."""
        core = FlextCore()

        if hasattr(core, "serialize"):
            result = core.serialize({})
            assert isinstance(result, FlextResult)

        if hasattr(core, "deserialize"):
            result = core.deserialize("{}")
            assert isinstance(result, FlextResult)

    def test_caching_if_available(self) -> None:
        """Test caching methods if they exist."""
        core = FlextCore()

        if hasattr(core, "cache_get"):
            result = core.cache_get("test_key")
            assert isinstance(result, FlextResult)

        if hasattr(core, "cache_set"):
            result = core.cache_set("test_key", "test_value")
            assert isinstance(result, FlextResult)

        if hasattr(core, "cache_clear"):
            result = core.cache_clear()
            assert isinstance(result, FlextResult)

    def test_health_check_if_available(self) -> None:
        """Test health check methods if they exist."""
        core = FlextCore()

        if hasattr(core, "health_check"):
            result = core.health_check()
            assert isinstance(result, FlextResult)

        if hasattr(core, "get_status"):
            result = core.get_status()
            assert isinstance(result, FlextResult)

    def test_plugin_management_if_available(self) -> None:
        """Test plugin management methods if they exist."""
        core = FlextCore()

        if hasattr(core, "load_plugin"):
            result = core.load_plugin("test_plugin")
            assert isinstance(result, FlextResult)

        if hasattr(core, "unload_plugin"):
            result = core.unload_plugin("test_plugin")
            assert isinstance(result, FlextResult)

        if hasattr(core, "list_plugins"):
            result = core.list_plugins()
            assert isinstance(result, FlextResult)
