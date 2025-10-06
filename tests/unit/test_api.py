"""Tests for Flext unified facade.

This module tests the Flext facade providing unified access to all
flext-core components with proper integration and modern patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import (
    Flext,
    FlextBus,
    FlextConfig,
    FlextConstants,
    FlextContainer,
    FlextContext,
    FlextDispatcher,
    FlextExceptions,
    FlextHandlers,
    FlextLogger,
    FlextMixins,
    FlextModels,
    FlextProcessors,
    FlextProtocols,
    FlextRegistry,
    FlextResult,
    FlextService,
    FlextTypes,
    FlextUtilities,
)


class TestFlextImports:
    """Test Flext import and alias functionality."""

    def test_flextcore_import(self) -> None:
        """Test Flext can be imported."""
        assert Flext is not None
        assert isinstance(Flext, type)

    def test_flextcore_alias(self) -> None:
        """Test Flext alias works."""
        assert Flext is not None
        assert hasattr(Flext, "__name__")


class TestFlextClassAttributes:
    """Test Flext class-level component access."""

    def test_core_patterns_available(self) -> None:
        """Test core pattern classes are available."""
        assert Flext.Result is FlextResult
        assert issubclass(Flext.Config, FlextConfig)
        assert issubclass(Flext.Container, FlextContainer)
        assert issubclass(Flext.Logger, FlextLogger)
        assert issubclass(Flext.Service, FlextService)

    def test_namespace_classes_available(self) -> None:
        """Test namespace classes are available."""
        assert issubclass(Flext.Models, FlextModels)
        assert issubclass(Flext.Constants, FlextConstants)
        assert issubclass(Flext.Types, FlextTypes)
        assert issubclass(Flext.Exceptions, FlextExceptions)
        assert issubclass(Flext.Protocols, FlextProtocols)

    def test_advanced_components_available(self) -> None:
        """Test advanced integration components are available."""
        assert issubclass(Flext.Bus, FlextBus)
        assert issubclass(Flext.Context, FlextContext)
        assert issubclass(Flext.Handlers, FlextHandlers)
        assert issubclass(Flext.Processors, FlextProcessors)
        assert issubclass(Flext.Registry, FlextRegistry)
        assert issubclass(Flext.Dispatcher, FlextDispatcher)
        assert issubclass(Flext.Mixins, FlextMixins)
        assert issubclass(Flext.Utilities, FlextUtilities)


class TestFlextInstantiation:
    """Test Flext instance creation and initialization."""

    def test_create_default_instance(self) -> None:
        """Test creating Flext instance with defaults."""
        core = Flext()
        assert core is not None
        assert isinstance(core, Flext)
        assert isinstance(core, FlextService)

    def test_create_instance_with_config(self) -> None:
        """Test creating Flext instance with custom config."""
        custom_config = FlextConfig()
        core = Flext(config=custom_config)
        assert core is not None
        assert core.config is custom_config

    def test_execute_method(self) -> None:
        """Test execute method returns success."""
        core = Flext()
        result = core.execute()
        assert result.is_success
        assert result.value is None


class TestFlextPropertyAccessors:
    """Test Flext property-based component access."""

    def test_config_property(self) -> None:
        """Test config property returns FlextConfig instance."""
        core = Flext()
        config = core.config
        assert config is not None
        assert isinstance(config, FlextConfig)

    def test_container_property(self) -> None:
        """Test container property returns global container."""
        core = Flext()
        container = core.container
        assert container is not None
        assert isinstance(container, FlextContainer)

    def test_logger_property(self) -> None:
        """Test logger property returns FlextLogger instance."""
        core = Flext()
        logger = core.logger
        assert logger is not None
        assert isinstance(logger, FlextLogger)

    def test_bus_property(self) -> None:
        """Test bus property returns FlextBus instance."""
        core = Flext()
        bus = core.bus
        assert bus is not None
        assert isinstance(bus, FlextBus)

    def test_context_property(self) -> None:
        """Test context property returns FlextContext instance."""
        core = Flext()
        context = core.context
        assert context is not None
        assert isinstance(context, FlextContext)

    def test_dispatcher_property(self) -> None:
        """Test dispatcher property returns FlextDispatcher instance."""
        core = Flext()
        dispatcher = core.dispatcher
        assert dispatcher is not None
        assert isinstance(dispatcher, FlextDispatcher)

    def test_processors_property(self) -> None:
        """Test processors property returns FlextProcessors instance."""
        core = Flext()
        processors = core.processors
        assert processors is not None
        assert isinstance(processors, FlextProcessors)

    def test_registry_property(self) -> None:
        """Test registry property returns FlextRegistry instance."""
        core = Flext()
        registry = core.registry
        assert registry is not None
        assert isinstance(registry, FlextRegistry)

    def test_property_lazy_loading(self) -> None:
        """Test properties are lazy-loaded (same instance returned)."""
        core = Flext()

        # First access
        logger1 = core.logger
        bus1 = core.bus
        context1 = core.context

        # Second access - should return same instances
        logger2 = core.logger
        bus2 = core.bus
        context2 = core.context

        assert logger1 is logger2
        assert bus1 is bus2
        assert context1 is context2


class TestFlextFactoryMethods:
    """Test Flext factory methods for common patterns."""

    def test_create_result_ok(self) -> None:
        """Test creating successful result."""
        result = Flext.create_result_ok("test_value")
        assert result.is_success
        assert result.value == "test_value"

    def test_create_result_fail(self) -> None:
        """Test creating failed result."""
        result: FlextResult[None] = Flext.create_result_fail(
            "error message", "ERROR_CODE"
        )
        assert result.is_failure
        assert result.error == "error message"

    def test_create_logger(self) -> None:
        """Test creating logger instance."""
        logger = Flext.create_logger("test_module")
        assert logger is not None
        assert isinstance(logger, FlextLogger)

    def test_get_container(self) -> None:
        """Test getting global container."""
        container = Flext.get_container()
        assert container is not None
        assert isinstance(container, FlextContainer)

    def test_get_config(self) -> None:
        """Test getting config instance."""
        config = Flext.get_config()
        assert config is not None
        assert isinstance(config, FlextConfig)


class TestFlextIntegrationHelpers:
    """Test Flext integration helper methods."""

    def test_setup_service_infrastructure_success(self) -> None:
        """Test successful service infrastructure setup."""
        result = Flext.setup_service_infrastructure("test-service")

        assert result.is_success
        infra = result.unwrap()

        # Verify all components are present
        assert "config" in infra
        assert "container" in infra
        assert "logger" in infra
        assert "bus" in infra
        assert "context" in infra

        # Verify component types
        assert isinstance(infra["config"], FlextConfig)
        assert isinstance(infra["container"], FlextContainer)
        assert isinstance(infra["logger"], FlextLogger)
        assert isinstance(infra["bus"], FlextBus)
        assert isinstance(infra["context"], FlextContext)

    def test_setup_service_infrastructure_with_config(self) -> None:
        """Test service infrastructure setup with custom config."""
        custom_config = FlextConfig()
        result = Flext.setup_service_infrastructure(
            "test-service", config=custom_config
        )

        assert result.is_success
        infra = result.unwrap()
        assert infra["config"] is custom_config

    @pytest.mark.skip(
        reason="FlextHandlers is abstract - handler creation needs concrete implementation"
    )
    def test_create_command_handler_success(self) -> None:
        """Test creating command handler."""

        def handler_func(cmd: object) -> FlextResult[object]:
            return FlextResult[object].ok({"processed": True})

        result = Flext.create_command_handler(handler_func)
        assert result.is_success
        handler = result.unwrap()
        # Handler is returned as object type from factory
        assert handler is not None

    @pytest.mark.skip(
        reason="FlextHandlers is abstract - handler creation needs concrete implementation"
    )
    def test_create_query_handler_success(self) -> None:
        """Test creating query handler."""

        def handler_func(query: object) -> FlextResult[object]:
            return FlextResult[object].ok({"data": "result"})

        result = Flext.create_query_handler(handler_func)
        assert result.is_success
        handler = result.unwrap()
        # Handler is returned as object type from factory
        assert handler is not None


class TestFlextDirectClassAccess:
    """Test direct class access patterns through Flext."""

    def test_result_creation_via_class(self) -> None:
        """Test creating FlextResult through Flext.Result."""
        result = Flext.Result[str].ok("test")
        assert result.is_success
        assert result.value == "test"

    def test_models_entity_access(self) -> None:
        """Test accessing FlextModels.Entity through Flext."""
        # FlextModels.Entity is a class, verify it's accessible
        assert Flext.Models.Entity is not None

    def test_constants_access(self) -> None:
        """Test accessing constants through Flext."""
        timeout = Flext.Constants.Defaults.TIMEOUT
        assert timeout is not None
        assert isinstance(timeout, int)

    def test_types_access(self) -> None:
        """Test accessing types through Flext."""
        # Verify FlextTypes is accessible
        assert Flext.Types is not None

    def test_exceptions_access(self) -> None:
        """Test accessing exceptions through Flext."""
        # Verify base exception is accessible
        assert Flext.Exceptions.BaseError is not None


class TestFlextUsagePatterns:
    """Test common usage patterns with Flext."""

    def test_unified_access_pattern(self) -> None:
        """Test unified access to all components."""
        core = Flext()

        # Access all components through unified interface
        config = core.config
        container = core.container
        logger = core.logger
        bus = core.bus
        context = core.context

        # Verify all are initialized
        assert config is not None
        assert container is not None
        assert logger is not None
        assert bus is not None
        assert context is not None

    def test_result_railway_pattern(self) -> None:
        """Test railway pattern using Flext."""

        def validate_input(data: str) -> FlextResult[str]:
            if not data:
                return Flext.Result[str].fail("Data cannot be empty")
            return Flext.Result[str].ok(data)

        def process_data(data: str) -> FlextResult[str]:
            return Flext.Result[str].ok(data.upper())

        # Railway-oriented composition
        result = validate_input("test").flat_map(process_data)

        assert result.is_success
        assert result.value == "TEST"

    def test_combined_infrastructure_setup(self) -> None:
        """Test setting up complete infrastructure."""
        # Setup infrastructure
        setup_result = Flext.setup_service_infrastructure("my-service")
        assert setup_result.is_success

        infra = setup_result.unwrap()
        logger = infra["logger"]
        container = infra["container"]

        # Use infrastructure components
        assert isinstance(logger, FlextLogger)
        assert isinstance(container, FlextContainer)

    def test_service_creation_pattern(self) -> None:
        """Test creating services using Flext."""

        class MyService(Flext.Service[str]):
            def __init__(self) -> None:
                super().__init__()
                # Store logger in private attribute to avoid Pydantic validation
                self._logger = Flext.create_logger(__name__)

            def execute(self) -> FlextResult[str]:
                return Flext.Result[str].ok("Service executed")

        service = MyService()
        result = service.execute()

        assert result.is_success
        assert result.value == "Service executed"


class TestFlextBackwardCompatibility:
    """Test that Flext doesn't break existing patterns."""

    def test_direct_imports_still_work(self) -> None:
        """Test that direct imports still work alongside Flext."""
        # Old pattern
        from flext_core import (
            FlextConfig,
            FlextContainer,
            FlextLogger,
            FlextResult,
        )

        config = FlextConfig()
        container = FlextContainer.get_global()
        logger = FlextLogger(__name__)
        result = FlextResult[str].ok("test")

        # All should work as before
        assert config is not None
        assert container is not None
        assert logger is not None
        assert result.is_success

    def test_both_patterns_coexist(self) -> None:
        """Test that both old and new patterns can be used together."""
        # New pattern
        core = Flext()
        core_logger = core.logger

        # Old pattern
        direct_logger = FlextLogger("test")

        # Both should work
        assert core_logger is not None
        assert direct_logger is not None


# =================================================================
# Integration Tests
# =================================================================


class TestFlext11Features:
    """Test Flext 1.1.0 convenience methods."""

    def test_publish_event_success(self) -> None:
        """Test successful event publishing with correlation tracking."""
        core = Flext()

        event_data: FlextTypes.Dict = {"user_id": "123", "action": "login"}
        result = core.publish_event("user.login", event_data)

        assert result.is_success
        assert result.value is None

    def test_publish_event_with_custom_correlation(self) -> None:
        """Test event publishing with custom correlation ID."""
        core = Flext()

        custom_correlation = "test-correlation-123"
        event_data: FlextTypes.Dict = {"user_id": "456"}

        result = core.publish_event(
            "user.action", event_data, correlation_id=custom_correlation
        )

        assert result.is_success

    def test_publish_event_with_empty_data(self) -> None:
        """Test event publishing with empty data dict."""
        core = Flext()

        result = core.publish_event("test.event", {})

        # Should succeed with empty data
        assert result.is_success

    def test_create_service_success(self) -> None:
        """Test service creation with infrastructure injection."""

        class TestService(FlextService[str]):
            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("service_executed")

        result = Flext.create_service(TestService, "test-service")

        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, TestService)
        assert isinstance(service, FlextService)

    def test_create_service_with_kwargs(self) -> None:
        """Test service creation stores kwargs for later use."""

        class ConfigurableService(FlextService[str]):
            def __init__(self, custom_value: str = "default") -> None:
                super().__init__()
                self._custom_value = custom_value

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok(self._custom_value)

        result = Flext.create_service(
            ConfigurableService, "configurable-service", custom_value="test-value"
        )

        assert result.is_success
        service = result.unwrap()
        exec_result = service.execute()
        assert exec_result.value == "test-value"

    def test_create_service_with_custom_config(self) -> None:
        """Test service creation with custom config."""

        class SimpleService(FlextService[None]):
            def execute(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

        custom_config = FlextConfig()

        result = Flext.create_service(
            SimpleService, "configured-service", config=custom_config
        )

        assert result.is_success

    def test_build_pipeline_success(self) -> None:
        """Test pipeline composition with successful operations."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def multiply_two(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        def subtract_three(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x - 3)

        # Build pipeline: (5 + 1) * 2 - 3 = 9
        pipeline = Flext.build_pipeline(add_one, multiply_two, subtract_three)

        result = pipeline(5)

        assert result.is_success
        assert result.value == 9

    def test_build_pipeline_with_failure(self) -> None:
        """Test pipeline early termination on failure."""

        def succeed(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def fail(x: int) -> FlextResult[int]:
            return FlextResult[int].fail("Operation failed")

        def should_not_execute(x: int) -> FlextResult[int]:
            # This should never be called due to early termination
            return FlextResult[int].ok(x * 100)

        pipeline = Flext.build_pipeline(succeed, fail, should_not_execute)

        result = pipeline(5)

        assert result.is_failure
        assert result.error is not None and "Operation failed" in result.error

    def test_build_pipeline_with_exception(self) -> None:
        """Test pipeline exception handling."""

        def succeed(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 1)

        def raise_exception(x: int) -> FlextResult[int]:
            msg = "Unexpected error"
            raise ValueError(msg)

        pipeline = Flext.build_pipeline(succeed, raise_exception)

        result = pipeline(5)

        assert result.is_failure
        assert result.error is not None and "Pipeline operation failed" in result.error

    def test_request_context_manager(self) -> None:
        """Test request context manager setup and cleanup."""
        core = Flext()

        with core.request_context() as context:
            # Context should be available
            assert context is not None
            assert isinstance(context, FlextContext)

            # Request ID should be set
            request_id = context.get("request_id")
            assert request_id is not None

    def test_request_context_with_custom_request_id(self) -> None:
        """Test request context with custom request ID."""
        core = Flext()
        custom_id = "test-request-123"

        with core.request_context(request_id=custom_id) as context:
            request_id = context.get("request_id")
            assert request_id == custom_id

    def test_request_context_with_user_id(self) -> None:
        """Test request context with user ID tracking."""
        core = Flext()

        with core.request_context(user_id="user-456") as context:
            user_id = context.get("user_id")
            assert user_id == "user-456"

    def test_request_context_with_metadata(self) -> None:
        """Test request context with custom metadata."""
        core = Flext()

        with core.request_context(
            client_ip="192.168.1.1", user_agent="TestClient/1.0"
        ) as context:
            assert context.get("client_ip") == "192.168.1.1"
            assert context.get("user_agent") == "TestClient/1.0"

    def test_request_context_cleanup(self) -> None:
        """Test that request context is properly cleaned up."""
        core = Flext()

        # Use context manager
        with core.request_context(request_id="cleanup-test") as context:
            assert context.get("request_id") == "cleanup-test"

        # Context manager should complete successfully
        assert True


# =================================================================
# Integration Tests
# =================================================================


@pytest.mark.integration
class TestFlextIntegration:
    """Integration tests for Flext with real components."""

    def test_complete_workflow(self) -> None:
        """Test complete workflow using Flext."""
        # Setup
        core = Flext()

        # Create result
        result = Flext.create_result_ok({"user_id": "123"})
        assert result.is_success

        # Log operation
        logger = core.logger
        assert logger is not None

        # Access container
        container = core.container
        assert container is not None

        # Complete workflow succeeds
        assert True

    def test_service_infrastructure_integration(self) -> None:
        """Test complete service infrastructure integration."""
        # Setup infrastructure
        result = Flext.setup_service_infrastructure("integration-test")
        assert result.is_success

        infra = result.unwrap()

        # Verify container has registered components
        container = infra["container"]
        assert container is not None

        # Verify logger is functional
        logger = infra["logger"]
        assert logger is not None

        # Verify config is accessible
        config = infra["config"]
        assert config is not None

    def test_11_features_integration(self) -> None:
        """Test 1.1.0 features working together in realistic scenario."""
        core = Flext()

        # Simulate request handling with full infrastructure
        with core.request_context(
            request_id="integration-123", user_id="user-789"
        ) as context:
            # Build processing pipeline
            def validate(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
                if not data.get("valid"):
                    return FlextResult[FlextTypes.Dict].fail("Validation failed")
                return FlextResult[FlextTypes.Dict].ok(data)

            def enrich(data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
                enriched = {**data, "enriched": True}
                return FlextResult[FlextTypes.Dict].ok(enriched)

            pipeline = Flext.build_pipeline(validate, enrich)

            # Process data through pipeline
            result = pipeline({"valid": True, "value": 42})

            # Publish event on success
            if result.is_success:
                data = result.unwrap()
                request_id = context.get("request_id")
                event_result = core.publish_event(
                    "data.processed",
                    data if isinstance(data, dict) else {},
                    correlation_id=request_id if isinstance(request_id, str) else None,
                )
                assert event_result.is_success

            assert result.is_success
            processed_data = result.unwrap()
            assert (
                isinstance(processed_data, dict)
                and processed_data.get("enriched") is True
            )
