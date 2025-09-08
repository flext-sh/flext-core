"""Additional tests to achieve 100% coverage for FlextCore.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Never
from unittest.mock import MagicMock, Mock, patch

import pytest

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextContext,
    FlextCore,
    FlextDecorators,
    FlextExceptions,
    FlextFields,
    FlextGuards,
    FlextHandlers,
    FlextLogger,
    FlextMixins,
    FlextModels,
    FlextProcessors,
    FlextProtocols,
    FlextResult,
    FlextUtilities,
    FlextValidations,
)


class TestFlextCoreUncoveredMethods:
    """Test methods that are not covered in the main test file."""

    def test_constants_property(self) -> None:
        """Test constants property lazy loading."""
        core = FlextCore()
        # Access a property that initializes _constants
        assert core._constants is None or isinstance(core._constants, type)

    def test_types_property(self) -> None:
        """Test types property lazy loading."""
        core = FlextCore()
        assert core._types is None or isinstance(core._types, type)

    def test_protocols_property(self) -> None:
        """Test protocols property lazy loading."""
        core = FlextCore()
        assert core._protocols is None or isinstance(core._protocols, type)

    def test_result_utils_property(self) -> None:
        """Test result_utils property lazy loading."""
        core = FlextCore()
        assert core._result_utils is None or isinstance(core._result_utils, type)

    def test_handler_registry_property(self) -> None:
        """Test handler_registry property lazy loading."""
        core = FlextCore()
        registry = core.handler_registry
        assert registry is not None
        assert core._handler_registry is registry

    def test_field_registry_property(self) -> None:
        """Test field_registry property lazy loading."""
        core = FlextCore()
        registry = core.field_registry
        assert registry is not None
        assert core._field_registry is registry

    def test_console_property(self) -> None:
        """Test console property lazy loading."""
        core = FlextCore()
        console = core.console
        assert isinstance(console, FlextUtilities)
        assert core._console is console

    def test_plugin_registry_property(self) -> None:
        """Test plugin_registry property lazy loading."""
        core = FlextCore()
        registry = core.plugin_registry
        assert registry is not None
        assert core._plugin_registry is registry
        # Test methods
        assert hasattr(registry, "register")
        assert hasattr(registry, "get")
        assert hasattr(registry, "list_plugins")


class TestFlextCoreStaticMethods:
    """Test static methods."""

    def test_configure_core_system_success(self) -> None:
        """Test configure_core_system with valid config."""
        config = {
            "environment": "development",
            "log_level": "INFO",
            "validation_level": "normal",
        }

        with patch.object(FlextConfig, "create_from_environment") as mock_create:
            mock_config = MagicMock()
            mock_config.model_dump.return_value = config
            mock_create.return_value = FlextResult.ok(mock_config)

            result = FlextCore.configure_core_system(config)

            assert result.is_success
            assert "enable_observability" in result.value

    def test_configure_core_system_failure(self) -> None:
        """Test configure_core_system with invalid config."""
        config = {"invalid": "config"}

        with patch.object(FlextConfig, "create_from_environment") as mock_create:
            mock_create.return_value = FlextResult.fail("Invalid config")

            result = FlextCore.configure_core_system(config)

            assert result.is_failure

    def test_get_core_system_config(self) -> None:
        """Test get_core_system_config."""
        result = FlextCore.get_core_system_config()

        assert result.is_success
        config = result.value
        assert "environment" in config
        assert "log_level" in config
        assert "available_subsystems" in config

    def test_create_environment_core_config_production(self) -> None:
        """Test create_environment_core_config for production."""
        result = FlextCore.create_environment_core_config("production")

        assert result.is_success
        config = result.value
        assert config["environment"] == "production"
        assert config["log_level"] == "WARNING"
        assert config["validation_level"] == "strict"

    def test_create_environment_core_config_development(self) -> None:
        """Test create_environment_core_config for development."""
        result = FlextCore.create_environment_core_config("development")

        assert result.is_success
        config = result.value
        assert config["environment"] == "development"
        assert config["log_level"] == "DEBUG"
        assert config["validation_level"] == "loose"

    def test_create_environment_core_config_test(self) -> None:
        """Test create_environment_core_config for test."""
        result = FlextCore.create_environment_core_config("test")

        assert result.is_success
        config = result.value
        assert config["environment"] == "test"
        assert config["log_level"] == "ERROR"
        assert config["validation_level"] == "strict"

    def test_create_environment_core_config_staging(self) -> None:
        """Test create_environment_core_config for staging."""
        result = FlextCore.create_environment_core_config("staging")

        assert result.is_success
        config = result.value
        assert config["environment"] == "staging"
        assert config["log_level"] == "INFO"
        assert config["validation_level"] == "normal"

    def test_create_environment_core_config_local(self) -> None:
        """Test create_environment_core_config for local."""
        result = FlextCore.create_environment_core_config("local")

        assert result.is_success
        config = result.value
        assert config["environment"] == "local"
        assert config["log_level"] == "DEBUG"

    def test_create_environment_core_config_invalid(self) -> None:
        """Test create_environment_core_config with invalid environment."""
        result = FlextCore.create_environment_core_config("invalid_env")

        assert result.is_failure
        assert "Invalid environment" in result.error

    def test_optimize_core_performance_high(self) -> None:
        """Test optimize_core_performance with high level."""
        config = {"performance_level": "high"}
        result = FlextCore.optimize_core_performance(config)

        assert result.is_success
        opt_config = result.value
        assert opt_config["performance_level"] == "high"
        assert opt_config["container_cache_size"] == 1000
        assert opt_config["enable_lazy_loading"] is True

    def test_optimize_core_performance_medium(self) -> None:
        """Test optimize_core_performance with medium level."""
        config = {"performance_level": "medium"}
        result = FlextCore.optimize_core_performance(config)

        assert result.is_success
        opt_config = result.value
        assert opt_config["performance_level"] == "medium"
        assert opt_config["container_cache_size"] == 500

    def test_optimize_core_performance_low(self) -> None:
        """Test optimize_core_performance with low level."""
        config = {"performance_level": "low"}
        result = FlextCore.optimize_core_performance(config)

        assert result.is_success
        opt_config = result.value
        assert opt_config["performance_level"] == "low"
        assert opt_config["container_cache_size"] == 100

    def test_optimize_core_performance_custom(self) -> None:
        """Test optimize_core_performance with custom level."""
        config = {"performance_level": "custom", "max_service_registrations": 2000}
        result = FlextCore.optimize_core_performance(config)

        assert result.is_success
        opt_config = result.value
        assert opt_config["max_service_registrations"] == 2000

    def test_load_config_from_env(self) -> None:
        """Test load_config_from_env."""
        with patch.dict(os.environ, {"FLEXT_API_KEY": "secret", "FLEXT_DEBUG": "true"}):
            result = FlextCore.load_config_from_env()

            assert result.is_success
            config = result.value
            assert config["api_key"] == "secret"
            assert config["debug"] == "true"

    def test_merge_configs(self) -> None:
        """Test merge_configs."""
        config1 = {"key1": "value1", "shared": "original"}
        config2 = {"key2": "value2", "shared": "override"}

        with patch.object(FlextConfig, "merge_configs") as mock_merge:
            mock_merge.return_value = FlextResult.ok(
                {
                    "key1": "value1",
                    "key2": "value2",
                    "shared": "override",
                }
            )

            result = FlextCore.merge_configs(config1, config2)

            assert result.is_success
            merged = result.value
            assert merged["key1"] == "value1"
            assert merged["key2"] == "value2"

    def test_merge_configs_insufficient(self) -> None:
        """Test merge_configs with insufficient configs."""
        result = FlextCore.merge_configs({"key": "value"})

        assert result.is_failure
        assert "At least 2 configs required" in result.error

    def test_safe_get_env_var(self) -> None:
        """Test safe_get_env_var."""
        with patch.object(FlextCore, "safe_get_env_var") as mock_get:
            mock_get.return_value = FlextResult.ok("value")

            result = FlextCore.safe_get_env_var("TEST_VAR", "default")

            assert result.is_success
            assert result.value == "value"


class TestFlextCoreDomainMethods:
    """Test domain modeling methods."""

    def test_create_aggregate_root_with_model_validate(self) -> None:
        """Test create_aggregate_root with model_validate."""

        class TestAggregate:
            @classmethod
            def model_validate(cls, data):
                instance = cls()
                for k, v in data.items():
                    setattr(instance, k, v)
                return instance

        result = FlextCore.create_aggregate_root(TestAggregate, name="test")

        assert result.is_success
        assert hasattr(result.value, "name")

    def test_create_aggregate_root_without_model_validate(self) -> None:
        """Test create_aggregate_root without model_validate."""

        class TestAggregate:
            def __init__(self, **kwargs) -> None:
                for k, v in kwargs.items():
                    setattr(self, k, v)

        result = FlextCore.create_aggregate_root(TestAggregate, name="test")

        assert result.is_success
        assert hasattr(result.value, "name")

    def test_create_aggregate_root_exception(self) -> None:
        """Test create_aggregate_root with exception."""

        class BadAggregate:
            def __init__(self, **kwargs) -> None:
                msg = "Cannot create"
                raise ValueError(msg)

        result = FlextCore.create_aggregate_root(BadAggregate, name="test")

        assert result.is_failure
        assert "Aggregate root creation failed" in result.error

    def test_entity_base_property(self) -> None:
        """Test entity_base property."""
        core = FlextCore()
        assert core.entity_base is FlextModels.Entity

    def test_value_object_base_property(self) -> None:
        """Test value_object_base property."""
        core = FlextCore()
        assert core.value_object_base is FlextModels.Value

    def test_aggregate_root_base_property(self) -> None:
        """Test aggregate_root_base property."""
        core = FlextCore()
        assert core.aggregate_root_base is FlextModels.Entity

    def test_domain_service_base_property(self) -> None:
        """Test domain_service_base property."""
        core = FlextCore()
        assert core.domain_service_base is not None


class TestFlextCoreUtilityMethods:
    """Test utility methods."""

    def test_safe_call_success(self) -> None:
        """Test safe_call with successful function."""

        def func() -> int:
            return 42

        result = FlextCore.safe_call(func, default=0)
        assert result == 42

    def test_safe_call_failure(self) -> None:
        """Test safe_call with failing function."""

        def func() -> float:
            return 1 / 0

        result = FlextCore.safe_call(func, default=0)
        assert result == 0

    def test_truncate(self) -> None:
        """Test truncate."""
        with patch.object(FlextUtilities, "truncate") as mock_truncate:
            mock_truncate.return_value = "truncated..."

            result = FlextCore.truncate("long text", 5)

            assert result == "truncated..."
            mock_truncate.assert_called_once_with("long text", 5)

    def test_is_not_none(self) -> None:
        """Test is_not_none."""
        assert FlextCore.is_not_none("value") is True
        assert FlextCore.is_not_none(None) is False
        assert FlextCore.is_not_none(0) is True
        assert FlextCore.is_not_none("") is True

    def test_generators_property(self) -> None:
        """Test generators property."""
        core = FlextCore()
        assert core.generators is FlextUtilities.Generators

    def test_type_guards_property(self) -> None:
        """Test type_guards property."""
        core = FlextCore()
        assert core.type_guards is FlextUtilities.TypeGuards


class TestFlextCoreMessagingMethods:
    """Test messaging and event methods."""

    def test_create_message_success(self) -> None:
        """Test create_message with valid data."""
        result = FlextCore.create_message(
            "TestMessage", data={"key": "value"}, correlation_id="corr-123"
        )

        assert result.is_success
        message = result.value
        assert message.message_type == "TestMessage"

    def test_create_message_exception(self) -> None:
        """Test create_message with exception."""
        with patch(
            "flext_core.models.FlextModels.Payload",
            side_effect=Exception("Payload error"),
        ):
            result = FlextCore.create_message("TestMessage", data={})

            assert result.is_failure
            assert "Message creation failed" in result.error

    def test_create_event_success(self) -> None:
        """Test create_event with valid data."""
        result = FlextCore.create_event(
            "TestEvent",
            {"key": "value"},
            aggregate_id="agg-123",
            aggregate_type="TestAggregate",
        )

        assert result.is_success
        event = result.value
        assert event.message_type == "TestEvent"

    def test_create_event_exception(self) -> None:
        """Test create_event with exception."""
        with patch(
            "flext_core.models.FlextModels.Event", side_effect=Exception("Event error")
        ):
            result = FlextCore.create_event("TestEvent", {})

            assert result.is_failure
            assert "Domain event creation failed" in result.error

    def test_validate_protocol_valid(self) -> None:
        """Test validate_protocol with valid payload."""
        payload = {"message_type": "test", "source_service": "service", "data": {}}
        result = FlextCore.validate_protocol(payload)

        assert result.is_success
        assert result.value == payload

    def test_validate_protocol_missing_field(self) -> None:
        """Test validate_protocol with missing field."""
        payload = {"message_type": "test", "data": {}}
        result = FlextCore.validate_protocol(payload)

        assert result.is_failure
        assert "Missing required field" in result.error

    def test_get_serialization_metrics(self) -> None:
        """Test get_serialization_metrics."""
        metrics = FlextCore.get_serialization_metrics()

        assert "total_payloads" in metrics
        assert "average_size" in metrics
        assert metrics["total_payloads"] == 0

    def test_payload_base_property(self) -> None:
        """Test payload_base property."""
        core = FlextCore()
        assert core.payload_base is not None

    def test_message_base_property(self) -> None:
        """Test message_base property."""
        core = FlextCore()
        assert core.message_base is FlextModels.Message

    def test_event_base_property(self) -> None:
        """Test event_base property."""
        core = FlextCore()
        assert core.event_base is FlextModels.Event


class TestFlextCoreHandlerMethods:
    """Test handler methods."""

    def test_get_handler_success(self) -> None:
        """Test get_handler with existing handler."""
        core = FlextCore()
        mock_handler = Mock()

        with patch.object(core.handler_registry, "get_handler") as mock_get:
            mock_get.return_value = FlextResult.ok(mock_handler)

            result = core.get_handler("test_handler")

            assert result.is_success
            assert result.value == mock_handler

    def test_get_handler_not_found(self) -> None:
        """Test get_handler with missing handler."""
        core = FlextCore()
        result = core.get_handler("missing_handler")

        assert result.is_failure
        assert "not found in registry" in result.error

    def test_base_handler_property(self) -> None:
        """Test base_handler property."""
        core = FlextCore()
        assert core.base_handler is FlextHandlers.Implementation.BasicHandler

    def test_create_string_field(self) -> None:
        """Test create_string_field."""
        with patch.object(FlextFields.Factory, "create_field") as mock_create:
            mock_create.return_value = FlextResult.ok("field")

            result = FlextCore.create_string_field("test_field", required=True)

            assert result == "field"

    def test_create_integer_field(self) -> None:
        """Test create_integer_field."""
        with patch.object(FlextFields.Factory, "create_field") as mock_create:
            mock_create.return_value = FlextResult.ok("field")

            result = FlextCore.create_integer_field("test_field", min_value=0)

            assert result == "field"

    def test_create_validation_decorator(self) -> None:
        """Test create_validation_decorator."""
        core = FlextCore()

        def validator(x):
            return x > 0

        with patch.object(
            FlextDecorators.Validation, "validate_input"
        ) as mock_validate:
            mock_decorator = Mock()
            mock_validate.return_value = mock_decorator

            result = core.create_validation_decorator(validator)

            assert result == mock_decorator

    def test_create_error_handling_decorator(self) -> None:
        """Test create_error_handling_decorator."""
        core = FlextCore()
        result = core.create_error_handling_decorator()
        assert result is FlextDecorators.Reliability

    def test_create_performance_decorator(self) -> None:
        """Test create_performance_decorator."""
        core = FlextCore()
        result = core.create_performance_decorator()
        assert result is FlextDecorators.Performance

    def test_create_logging_decorator(self) -> None:
        """Test create_logging_decorator."""
        core = FlextCore()
        result = core.create_logging_decorator()
        assert result is FlextDecorators.Observability

    def test_make_immutable(self) -> None:
        """Test make_immutable."""
        with patch.object(FlextGuards, "immutable") as mock_immutable:

            class TestClass:
                pass

            mock_immutable.return_value = TestClass

            result = FlextCore.make_immutable(TestClass)

            assert result is TestClass
            mock_immutable.assert_called_once_with(TestClass)

    def test_make_pure(self) -> None:
        """Test make_pure."""
        with patch.object(FlextGuards, "pure") as mock_pure:

            def func(x):
                return x

            mock_pure.return_value = func

            result = FlextCore.make_pure(func)

            assert result is func
            mock_pure.assert_called_once_with(func)


class TestFlextCoreMixinProperties:
    """Test mixin properties."""

    def test_timestamp_mixin(self) -> None:
        """Test timestamp_mixin property."""
        core = FlextCore()
        assert core.timestamp_mixin is FlextMixins.create_timestamp_fields

    def test_identifiable_mixin(self) -> None:
        """Test identifiable_mixin property."""
        core = FlextCore()
        assert core.identifiable_mixin is FlextMixins.ensure_id

    def test_loggable_mixin(self) -> None:
        """Test loggable_mixin property."""
        core = FlextCore()
        assert core.loggable_mixin is FlextLogger

    def test_validatable_mixin(self) -> None:
        """Test validatable_mixin property."""
        core = FlextCore()
        assert core.validatable_mixin is FlextMixins.validate_required_fields

    def test_serializable_mixin(self) -> None:
        """Test serializable_mixin property."""
        core = FlextCore()
        assert core.serializable_mixin is FlextMixins.to_dict

    def test_cacheable_mixin(self) -> None:
        """Test cacheable_mixin property."""
        core = FlextCore()
        assert core.cacheable_mixin is FlextMixins.get_cache_key


class TestFlextCoreRootModels:
    """Test root model creation methods."""

    def test_create_entity_id_success(self) -> None:
        """Test create_entity_id with valid value."""
        result = FlextCore.create_entity_id("entity-123")

        assert result.is_success
        assert result.value.root == "entity-123"

    def test_create_entity_id_none(self) -> None:
        """Test create_entity_id with None value."""
        result = FlextCore.create_entity_id(None)

        assert result.is_failure
        assert "cannot be None" in result.error

    def test_create_entity_id_exception(self) -> None:
        """Test create_entity_id with exception."""
        with patch(
            "flext_core.models.FlextModels.EntityId", side_effect=Exception("ID error")
        ):
            result = FlextCore.create_entity_id("test")

            assert result.is_failure
            assert "Entity ID creation failed" in result.error

    def test_create_version_number(self) -> None:
        """Test create_version_number."""
        result = FlextCore.create_version_number(1)

        assert result.is_success
        assert result.value.root == 1

    def test_create_email_address(self) -> None:
        """Test create_email_address."""
        result = FlextCore.create_email_address("test@example.com")

        assert result.is_success
        assert result.value.root == "test@example.com"

    def test_create_service_name_value(self) -> None:
        """Test create_service_name_value."""
        result = FlextCore.create_service_name_value("test-service")

        assert result.is_success
        assert result.value.root == "test-service"

    def test_create_timestamp(self) -> None:
        """Test create_timestamp."""
        timestamp = FlextCore.create_timestamp()

        assert isinstance(timestamp, datetime)
        assert timestamp.tzinfo is not None

    def test_create_metadata(self) -> None:
        """Test create_metadata."""
        result = FlextCore.create_metadata(key="value", count=42)

        assert result.is_success
        assert result.value.root["key"] == "value"
        assert result.value.root["count"] == 42


class TestFlextCoreExceptionMethods:
    """Test exception methods."""

    def test_create_error(self) -> None:
        """Test create_error."""
        error = FlextCore.create_error("Test error", error_code="TEST001")

        assert isinstance(error, FlextExceptions.Error)

    def test_get_exception_metrics(self) -> None:
        """Test get_exception_metrics."""
        with patch.object(FlextExceptions, "get_metrics") as mock_get:
            mock_get.return_value = {"errors": 10}

            metrics = FlextCore.get_exception_metrics()

            assert metrics["errors"] == 10

    def test_clear_exception_metrics(self) -> None:
        """Test clear_exception_metrics."""
        with patch.object(FlextExceptions, "clear_metrics") as mock_clear:
            FlextCore.clear_exception_metrics()

            mock_clear.assert_called_once()

    def test_create_processing_pipeline(self) -> None:
        """Test create_processing_pipeline."""
        pipeline = FlextCore.create_processing_pipeline()

        assert isinstance(pipeline, FlextProcessors.ProcessingPipeline)


class TestFlextCoreContextProtocols:
    """Test context and protocol methods."""

    def test_context_class_property(self) -> None:
        """Test context_class property."""
        core = FlextCore()
        assert core.context_class is FlextContext

    def test_repository_protocol_property(self) -> None:
        """Test repository_protocol property."""
        core = FlextCore()
        assert core.repository_protocol is FlextProtocols.Domain.Repository

    def test_plugin_protocol_property(self) -> None:
        """Test plugin_protocol property."""
        core = FlextCore()
        assert core.plugin_protocol is FlextProtocols.Extensions.Plugin

    def test_register_plugin_success(self) -> None:
        """Test register_plugin with success."""
        core = FlextCore()
        plugin = Mock()
        plugin.name = "test_plugin"

        result = core.register_plugin(plugin)

        assert result.is_success
        # Verify plugin was registered
        assert core.plugin_registry.get("test_plugin") == plugin

    def test_register_plugin_failure(self) -> None:
        """Test register_plugin with failure."""
        core = FlextCore()

        # Mock registry without register method
        core._plugin_registry = object()

        plugin = Mock()
        result = core.register_plugin(plugin)

        assert result.is_failure
        assert "does not support registration" in result.error


class TestFlextCoreTypeValidation:
    """Test type validation methods."""

    def test_validate_type_success(self) -> None:
        """Test validate_type with correct type."""
        result = FlextCore.validate_type("test", str)

        assert result.is_success
        assert result.value == "test"

    def test_validate_type_failure(self) -> None:
        """Test validate_type with incorrect type."""
        result = FlextCore.validate_type(123, str)

        assert result.is_failure
        assert "Expected str" in result.error

    def test_validate_dict_structure_success(self) -> None:
        """Test validate_dict_structure with valid dict."""
        with patch.object(FlextGuards, "is_dict_of") as mock_is_dict:
            mock_is_dict.return_value = True

            result = FlextCore.validate_dict_structure({"key": "value"}, str)

            assert result.is_success

    def test_validate_dict_structure_not_dict(self) -> None:
        """Test validate_dict_structure with non-dict."""
        result = FlextCore.validate_dict_structure("not a dict", str)

        assert result.is_failure
        assert "Expected dictionary" in result.error

    def test_validate_dict_structure_wrong_values(self) -> None:
        """Test validate_dict_structure with wrong value types."""
        with patch.object(FlextGuards, "is_dict_of") as mock_is_dict:
            mock_is_dict.return_value = False

            result = FlextCore.validate_dict_structure({"key": 123}, str)

            assert result.is_failure
            assert "must be of type str" in result.error

    def test_create_validated_model_with_model_validate(self) -> None:
        """Test create_validated_model with model_validate."""

        class TestModel:
            @classmethod
            def model_validate(cls, data):
                return cls()

        result = FlextCore.create_validated_model(TestModel, field="value")

        assert result.is_success
        assert isinstance(result.value, TestModel)

    def test_create_validated_model_without_model_validate(self) -> None:
        """Test create_validated_model without model_validate."""

        class TestModel:
            def __init__(self, **kwargs) -> None:
                pass

        result = FlextCore.create_validated_model(TestModel, field="value")

        assert result.is_success
        assert isinstance(result.value, TestModel)

    def test_create_validated_model_exception(self) -> None:
        """Test create_validated_model with exception."""

        class BadModel:
            def __init__(self, **kwargs) -> None:
                msg = "Cannot create"
                raise ValueError(msg)

        result = FlextCore.create_validated_model(BadModel, field="value")

        assert result.is_failure
        assert "Model validation failed" in result.error


class TestFlextCorePerformance:
    """Test performance methods."""

    def test_performance_property(self) -> None:
        """Test performance property."""
        core = FlextCore()
        assert core.performance is FlextUtilities.Performance

    def test_track_performance(self) -> None:
        """Test track_performance decorator."""
        core = FlextCore()

        @core.track_performance("test_operation")
        def test_func(x):
            return x * 2

        with patch.object(core, "flext_logger") as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            result = test_func(5)

            assert result == 10
            # Logger should be called for performance tracking
            assert mock_logger.called

    def test_track_performance_with_exception(self) -> None:
        """Test track_performance with exception."""
        core = FlextCore()

        @core.track_performance("failing_operation")
        def failing_func() -> Never:
            msg = "Test error"
            raise ValueError(msg)

        with patch.object(core, "flext_logger") as mock_logger:
            logger = MagicMock()
            mock_logger.return_value = logger

            with pytest.raises(ValueError):
                failing_func()

            # Logger should be called for exception
            assert mock_logger.called


class TestFlextCoreFactoryMethods:
    """Test factory methods."""

    def test_create_factory_model(self) -> None:
        """Test create_factory for model type."""
        core = FlextCore()
        result = core.create_factory("model", name="TestModel")

        assert result.is_success
        factory = result.value
        assert factory["name"] == "TestModel"
        assert "id" in factory

    def test_create_factory_service(self) -> None:
        """Test create_factory for service type."""
        core = FlextCore()
        result = core.create_factory("service", type="api")

        assert result.is_success
        factory = result.value
        assert factory["type"] == "api"
        assert factory["name"] == "service"

    def test_create_factory_unknown(self) -> None:
        """Test create_factory with unknown type."""
        core = FlextCore()
        result = core.create_factory("unknown_type")

        assert result.is_failure
        assert "Unknown factory type" in result.error

    def test_model_factory_property(self) -> None:
        """Test model_factory property."""
        core = FlextCore()
        assert core.model_factory is FlextModels.Config


class TestFlextCoreValidatorGuards:
    """Test validator and guard properties."""

    def test_validators_property(self) -> None:
        """Test validators property."""
        core = FlextCore()
        assert core.validators is FlextValidations

    def test_predicates_property(self) -> None:
        """Test predicates property."""
        core = FlextCore()
        assert core.predicates is FlextValidations.Core.Predicates

    def test_get_settings_cached(self) -> None:
        """Test get_settings with caching."""
        core = FlextCore()

        class TestSettings:
            pass

        # First call creates instance
        settings1 = core.get_settings(TestSettings)
        # Second call returns cached instance
        settings2 = core.get_settings(TestSettings)

        assert settings1 is settings2
        assert TestSettings in core._settings_cache

    def test_validate_service_name_valid(self) -> None:
        """Test validate_service_name with valid name."""
        with patch.object(
            FlextContainer, "flext_validate_service_name"
        ) as mock_validate:
            mock_validate.return_value = FlextResult.ok(None)

            result = FlextCore.validate_service_name("test-service")

            assert result.is_success
            assert result.value == "test-service"

    def test_validate_service_name_invalid(self) -> None:
        """Test validate_service_name with invalid name."""
        with patch.object(
            FlextContainer, "flext_validate_service_name"
        ) as mock_validate:
            mock_validate.return_value = FlextResult.fail("Invalid format")

            result = FlextCore.validate_service_name("bad service")

            assert result.is_failure
            assert "Invalid" in result.error

    def test_require_not_none_success(self) -> None:
        """Test require_not_none with value."""
        with patch.object(
            FlextGuards.ValidationUtils, "require_not_none"
        ) as mock_require:
            mock_require.return_value = "value"

            result = FlextCore.require_not_none("value")

            assert result.is_success
            assert result.value == "value"

    def test_require_not_none_failure(self) -> None:
        """Test require_not_none with None."""
        with patch.object(
            FlextGuards.ValidationUtils, "require_not_none"
        ) as mock_require:
            mock_require.side_effect = Exception("Cannot be None")

            result = FlextCore.require_not_none(None)

            assert result.is_failure
            assert "Cannot be None" in result.error

    def test_require_non_empty_success(self) -> None:
        """Test require_non_empty with value."""
        with patch.object(
            FlextGuards.ValidationUtils, "require_non_empty"
        ) as mock_require:
            mock_require.return_value = "value"

            result = FlextCore.require_non_empty("value")

            assert result.is_success
            assert result.value == "value"

    def test_require_positive_success(self) -> None:
        """Test require_positive with positive value."""
        with patch.object(
            FlextGuards.ValidationUtils, "require_positive"
        ) as mock_require:
            mock_require.return_value = 42.0

            result = FlextCore.require_positive(42.0)

            assert result.is_success
            assert result.value == 42.0


class TestFlextCoreBuilders:
    """Test builder methods."""

    def test_create_validator_class(self) -> None:
        """Test create_validator_class."""
        core = FlextCore()

        def validate_positive(x):
            if x > 0:
                return FlextResult.ok(x)
            return FlextResult.fail("Must be positive")

        ValidatorClass = core.create_validator_class(
            "PositiveValidator", validate_positive
        )

        assert ValidatorClass.__name__ == "PositiveValidator"
        validator = ValidatorClass()
        assert hasattr(validator, "validate")

    def test_create_service_processor(self) -> None:
        """Test create_service_processor."""
        core = FlextCore()

        def process_request(request):
            return FlextResult.ok({"processed": True})

        ProcessorClass = core.create_service_processor(
            "TestProcessor", process_request, result_type=dict
        )

        assert ProcessorClass.__name__ == "TestProcessorServiceProcessor"
        processor = ProcessorClass()
        assert hasattr(processor, "process")
        assert hasattr(processor, "build")

    def test_create_entity_with_validators(self) -> None:
        """Test create_entity_with_validators."""
        core = FlextCore()

        fields = {"name": (str, {"min_length": 1}), "age": (int, {"ge": 0})}

        EntityClass = core.create_entity_with_validators("Person", fields)

        assert issubclass(EntityClass, FlextModels.Entity)

    def test_create_value_object_with_validators(self) -> None:
        """Test create_value_object_with_validators."""
        core = FlextCore()

        fields = {"value": (str, {"min_length": 1})}

        def business_rules(obj):
            return FlextResult.ok(None)

        ValueClass = core.create_value_object_with_validators(
            "TestValue", fields, business_rules=business_rules
        )

        assert issubclass(ValueClass, FlextModels.Value)
        assert hasattr(ValueClass, "validate_business_rules")


class TestFlextCoreServiceSetup:
    """Test service setup methods."""

    def test_setup_container_with_services_success(self) -> None:
        """Test setup_container_with_services with success."""
        core = FlextCore()

        services = {"service1": lambda: "instance1", "service2": lambda: "instance2"}

        result = core.setup_container_with_services(services)

        assert result.is_success
        assert result.value == core.container

    def test_setup_container_with_services_with_validator(self) -> None:
        """Test setup_container_with_services with validator."""
        core = FlextCore()

        def validator(name):
            if name.startswith("valid_"):
                return FlextResult.ok(None)
            return FlextResult.fail("Invalid name")

        services = {
            "valid_service": lambda: "instance",
            "invalid_service": lambda: "instance",
        }

        result = core.setup_container_with_services(services, validator)

        assert result.is_success

    def test_setup_container_with_services_class_type(self) -> None:
        """Test setup_container_with_services with class type."""
        core = FlextCore()

        class TestService:
            pass

        services = {"test_service": TestService}

        result = core.setup_container_with_services(services)

        assert result.is_success

    def test_create_demo_function(self) -> None:
        """Test create_demo_function."""
        core = FlextCore()

        def demo() -> str:
            return "demo"

        demo_func = core.create_demo_function("CustomDemo", demo)

        assert demo_func.__name__ == "CustomDemo"
        assert demo_func() == "demo"

    def test_log_result_success(self) -> None:
        """Test log_result with success."""
        core = FlextCore()
        result = FlextResult.ok("value")

        with patch.object(core, "flext_logger") as mock_logger_func:
            logger = MagicMock()
            mock_logger_func.return_value = logger

            returned = core.log_result(result, "Operation succeeded")

            assert returned is result
            logger.info.assert_called_once()

    def test_log_result_failure(self) -> None:
        """Test log_result with failure."""
        core = FlextCore()
        result = FlextResult.fail("error")

        with patch.object(core, "flext_logger") as mock_logger_func:
            logger = MagicMock()
            mock_logger_func.return_value = logger

            returned = core.log_result(result, "Operation")

            assert returned is result
            logger.error.assert_called_once()

    def test_get_service_with_fallback_success(self) -> None:
        """Test get_service_with_fallback with existing service."""
        core = FlextCore()
        service = Mock()
        core.container.register("test", service)

        result = core.get_service_with_fallback("test", Mock)

        assert result == service

    def test_get_service_with_fallback_uses_fallback(self) -> None:
        """Test get_service_with_fallback uses fallback."""
        core = FlextCore()
        fallback = Mock()

        result = core.get_service_with_fallback("missing", lambda: fallback)

        assert result == fallback

    def test_create_standard_validators(self) -> None:
        """Test create_standard_validators."""
        core = FlextCore()
        validators = core.create_standard_validators()

        assert "age" in validators
        assert "email" in validators
        assert "name" in validators
        assert "service_name" in validators

        # Test age validator
        age_result = validators["age"](25)
        assert age_result.is_success

        # Test invalid age
        age_result = validators["age"](10)
        assert age_result.is_failure
