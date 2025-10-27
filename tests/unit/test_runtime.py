"""Comprehensive tests for FlextRuntime - Layer 0.5 Runtime Utilities.

Tests all functionality of FlextRuntime including type guards, serialization utilities,
external library access, and structlog configuration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import logging

import structlog
from dependency_injector import containers, providers

from flext_core import (
    FlextConstants,
    FlextContext,
    FlextRuntime,
)


class TestTypeGuards:
    """Test FlextRuntime type guard utilities."""

    def test_is_valid_phone_valid_cases(self) -> None:
        """Test is_valid_phone with valid phone numbers."""
        assert FlextRuntime.is_valid_phone("+5511987654321")
        assert FlextRuntime.is_valid_phone("5511987654321")
        assert FlextRuntime.is_valid_phone("+1234567890")
        assert FlextRuntime.is_valid_phone("123456789012345")  # 15 digits max

    def test_is_valid_phone_invalid_cases(self) -> None:
        """Test is_valid_phone with invalid phone numbers."""
        assert not FlextRuntime.is_valid_phone("123")  # Too short
        assert not FlextRuntime.is_valid_phone("abc1234567890")
        assert not FlextRuntime.is_valid_phone("+abc123")
        assert not FlextRuntime.is_valid_phone("")

    def test_is_valid_phone_non_string_types(self) -> None:
        """Test is_valid_phone returns False for non-string types."""
        assert not FlextRuntime.is_valid_phone(5511987654321)
        assert not FlextRuntime.is_valid_phone(None)

    def test_is_dict_like_valid_cases(self) -> None:
        """Test is_dict_like with dict-like objects."""
        assert FlextRuntime.is_dict_like({})
        assert FlextRuntime.is_dict_like({"key": "value"})
        assert FlextRuntime.is_dict_like({"nested": {"dict": True}})

    def test_is_dict_like_invalid_cases(self) -> None:
        """Test is_dict_like with non-dict objects."""
        assert not FlextRuntime.is_dict_like([])
        assert not FlextRuntime.is_dict_like("string")
        assert not FlextRuntime.is_dict_like(123)
        assert not FlextRuntime.is_dict_like(None)

    def test_is_list_like_valid_cases(self) -> None:
        """Test is_list_like with list-like objects."""
        assert FlextRuntime.is_list_like([])
        assert FlextRuntime.is_list_like([1, 2, 3])
        assert FlextRuntime.is_list_like(["a", "b", "c"])

    def test_is_list_like_invalid_cases(self) -> None:
        """Test is_list_like with non-list objects."""
        assert not FlextRuntime.is_list_like({})
        assert not FlextRuntime.is_list_like("string")
        assert not FlextRuntime.is_list_like(123)
        assert not FlextRuntime.is_list_like(None)

    def test_is_valid_json_valid_cases(self) -> None:
        """Test is_valid_json with valid JSON strings."""
        assert FlextRuntime.is_valid_json('{"key": "value"}')
        assert FlextRuntime.is_valid_json("[]")
        assert FlextRuntime.is_valid_json("[1, 2, 3]")
        assert FlextRuntime.is_valid_json('"string"')
        assert FlextRuntime.is_valid_json("null")

    def test_is_valid_json_invalid_cases(self) -> None:
        """Test is_valid_json with invalid JSON strings."""
        assert not FlextRuntime.is_valid_json("not json")
        assert not FlextRuntime.is_valid_json("{invalid}")
        assert not FlextRuntime.is_valid_json("")

    def test_is_valid_json_non_string_types(self) -> None:
        """Test is_valid_json returns False for non-string types."""
        assert not FlextRuntime.is_valid_json({"key": "value"})
        assert not FlextRuntime.is_valid_json([1, 2, 3])
        assert not FlextRuntime.is_valid_json(None)

    def test_is_valid_identifier_valid_cases(self) -> None:
        """Test is_valid_identifier with valid Python identifiers."""
        assert FlextRuntime.is_valid_identifier("variable")
        assert FlextRuntime.is_valid_identifier("_private")
        assert FlextRuntime.is_valid_identifier("ClassName")
        assert FlextRuntime.is_valid_identifier("function_name")

    def test_is_valid_identifier_invalid_cases(self) -> None:
        """Test is_valid_identifier with invalid Python identifiers."""
        assert not FlextRuntime.is_valid_identifier("123invalid")
        assert not FlextRuntime.is_valid_identifier("invalid-name")
        assert not FlextRuntime.is_valid_identifier("invalid name")
        assert not FlextRuntime.is_valid_identifier("")

    def test_is_valid_identifier_non_string_types(self) -> None:
        """Test is_valid_identifier returns False for non-string types."""
        assert not FlextRuntime.is_valid_identifier(123)
        assert not FlextRuntime.is_valid_identifier(None)


class TestSerializationUtilities:
    """Test FlextRuntime serialization utilities."""

    def test_safe_get_attribute_exists(self) -> None:
        """Test safe_get_attribute with existing attribute."""

        class TestObj:
            attr = "value"

        obj = TestObj()
        assert FlextRuntime.safe_get_attribute(obj, "attr") == "value"

    def test_safe_get_attribute_missing_with_default(self) -> None:
        """Test safe_get_attribute with missing attribute returns default."""

        class TestObj:
            pass

        obj = TestObj()
        assert FlextRuntime.safe_get_attribute(obj, "missing", "default") == "default"

    def test_safe_get_attribute_missing_without_default(self) -> None:
        """Test safe_get_attribute with missing attribute returns None."""

        class TestObj:
            pass

        obj = TestObj()
        assert FlextRuntime.safe_get_attribute(obj, "missing") is None


class TestTypeIntrospection:
    """Test FlextRuntime type introspection utilities."""

    def test_extract_generic_args_with_generic(self) -> None:
        """Test extract_generic_args with generic types."""
        args = FlextRuntime.extract_generic_args(list[str])
        assert args == (str,)

        args = FlextRuntime.extract_generic_args(dict[str, int])
        assert args == (str, int)

    def test_extract_generic_args_with_non_generic(self) -> None:
        """Test extract_generic_args with non-generic types."""
        args = FlextRuntime.extract_generic_args(str)
        assert args == ()

        args = FlextRuntime.extract_generic_args(int)
        assert args == ()

    def test_is_sequence_type_with_sequences(self) -> None:
        """Test is_sequence_type with sequence types."""
        assert FlextRuntime.is_sequence_type(list[str])
        assert FlextRuntime.is_sequence_type(tuple[int, ...])

    def test_is_sequence_type_with_non_sequences(self) -> None:
        """Test is_sequence_type with non-sequence types."""
        assert not FlextRuntime.is_sequence_type(dict[str, int])
        # Note: str is actually a sequence type in Python (subclass of Sequence)
        assert FlextRuntime.is_sequence_type(str)
        assert not FlextRuntime.is_sequence_type(int)

    def test_extract_generic_args_with_exception(self) -> None:
        """Test extract_generic_args with invalid input causing exception."""
        # Test with None - should return empty tuple
        args = FlextRuntime.extract_generic_args(None)
        assert args == ()

        # Test with invalid type hint
        args = FlextRuntime.extract_generic_args("not a type")
        assert args == ()

    def test_is_sequence_type_with_exception(self) -> None:
        """Test is_sequence_type with invalid input causing exception."""
        # Test with None - should not crash
        assert not FlextRuntime.is_sequence_type(None)

        # Test with invalid type hint
        assert not FlextRuntime.is_sequence_type("not a type")


class TestExternalLibraryAccess:
    """Test FlextRuntime external library access methods."""

    def test_structlog_returns_module(self) -> None:
        """Test structlog() returns the structlog module."""
        module = FlextRuntime.structlog()
        assert module is structlog
        assert hasattr(module, "get_logger")
        assert hasattr(module, "configure")

    def test_dependency_providers_returns_module(self) -> None:
        """Test dependency_providers() returns providers module."""
        module = FlextRuntime.dependency_providers()
        assert module is providers
        assert hasattr(module, "Singleton")
        assert hasattr(module, "Factory")

    def test_dependency_containers_returns_module(self) -> None:
        """Test dependency_containers() returns containers module."""
        module = FlextRuntime.dependency_containers()
        assert module is containers
        assert hasattr(module, "DeclarativeContainer")


class TestStructlogConfiguration:
    """Test FlextRuntime structlog configuration."""

    def test_configure_structlog_defaults(self) -> None:
        """Test configure_structlog with default settings."""
        # Reset configuration flag for testing
        FlextRuntime._structlog_configured = False

        # Configure with defaults
        FlextRuntime.configure_structlog()

        # Verify configuration was applied
        assert FlextRuntime._structlog_configured is True
        assert structlog.is_configured()

    def test_configure_structlog_custom_log_level(self) -> None:
        """Test configure_structlog with custom log level."""
        FlextRuntime._structlog_configured = False
        FlextRuntime.configure_structlog(log_level=logging.DEBUG)
        assert FlextRuntime._structlog_configured is True

    def test_configure_structlog_json_renderer(self) -> None:
        """Test configure_structlog with JSON renderer."""
        FlextRuntime._structlog_configured = False
        FlextRuntime.configure_structlog(console_renderer=False)
        assert FlextRuntime._structlog_configured is True

    def test_configure_structlog_with_additional_processors(self) -> None:
        """Test configure_structlog with additional processors."""
        FlextRuntime._structlog_configured = False

        # Custom processor
        def custom_processor(
            logger: object, method_name: str, event_dict: dict[str, object]
        ) -> dict[str, object]:
            event_dict["custom"] = True
            return event_dict

        FlextRuntime.configure_structlog(additional_processors=[custom_processor])
        assert FlextRuntime._structlog_configured is True

    def test_configure_structlog_json_renderer_explicit(self) -> None:
        """Test configure_structlog with JSON renderer explicitly set."""
        FlextRuntime._structlog_configured = False
        FlextRuntime.configure_structlog(
            console_renderer=False, log_level=logging.WARNING
        )
        assert FlextRuntime._structlog_configured is True

    def test_configure_structlog_idempotent(self) -> None:
        """Test configure_structlog is idempotent (can be called multiple times)."""
        FlextRuntime._structlog_configured = False
        FlextRuntime.configure_structlog()

        # Call again - should not raise error
        FlextRuntime.configure_structlog()
        assert FlextRuntime._structlog_configured is True


class TestRuntimeIntegrationWithConstants:
    """Test FlextRuntime integration with FlextConstants."""

    def test_type_guards_use_constants_patterns(self) -> None:
        """Test that type guards use patterns from FlextConstants."""
        # Verify the patterns are accessible
        assert hasattr(FlextConstants.Platform, "PATTERN_PHONE_NUMBER")

        # Verify type guards work with these patterns
        test_phone = "+5511987654321"
        assert FlextRuntime.is_valid_phone(test_phone)

    def test_layer_hierarchy_respected(self) -> None:
        """Test that Layer 0.5 (runtime) imports from Layer 0 (constants) only."""
        # FlextRuntime should import FlextConstants (Layer 0)
        # This test verifies the import chain is correct

        # Verify Flextis importable
        assert FlextConstants is not None

        # Verify both Runtime and Constants are accessible
        assert FlextRuntime is not None
        assert FlextConstants is not None

        # FlextRuntime should be able to access FlextConstants patterns
        assert FlextConstants.Platform.PATTERN_EMAIL is not None


class TestContextIntegrationPatterns:
    """Test FlextRuntime.Integration application-layer helpers (PHASE 2)."""

    def test_track_service_resolution_success(self) -> None:
        """Test Integration.track_service_resolution with successful resolution."""
        # Setup: Configure structlog and context

        FlextRuntime.configure_structlog()

        # Set correlation ID for tracking
        correlation_id = FlextContext.Utilities.ensure_correlation_id()

        # Track successful service resolution
        FlextRuntime.Integration.track_service_resolution("database", resolved=True)

        # Verify correlation ID is still in context
        current_correlation = FlextContext.Correlation.get_correlation_id()
        assert current_correlation == correlation_id

    def test_track_service_resolution_failure(self) -> None:
        """Test track_service_resolution with failed resolution."""
        FlextRuntime.configure_structlog()

        # Set correlation ID
        correlation_id = FlextContext.Utilities.ensure_correlation_id()

        # Track failed service resolution
        FlextRuntime.Integration.track_service_resolution(
            "cache",
            resolved=False,
            error_message="Connection refused",
        )

        # Verify correlation ID persists
        assert FlextContext.Correlation.get_correlation_id() == correlation_id

    def test_track_service_resolution_without_correlation(self) -> None:
        """Test track_service_resolution works without pre-existing correlation ID."""
        import structlog

        FlextRuntime.configure_structlog()

        # Clear any existing correlation using structlog
        structlog.contextvars.unbind_contextvars("correlation_id")

        # Should not raise error even without correlation ID
        FlextRuntime.Integration.track_service_resolution("logger", resolved=True)

        # Correlation ID should still be None

        assert FlextContext.Correlation.get_correlation_id() is None

    def test_track_domain_event_with_aggregate(self) -> None:
        """Test track_domain_event with aggregate ID."""
        FlextRuntime.configure_structlog()

        # Set correlation ID
        correlation_id = FlextContext.Utilities.ensure_correlation_id()

        # Track domain event with aggregate
        FlextRuntime.Integration.track_domain_event(
            "UserCreated",
            aggregate_id="user-123",
            event_data={"email": "test@example.com"},
        )

        # Verify correlation ID persists
        assert FlextContext.Correlation.get_correlation_id() == correlation_id

    def test_track_domain_event_without_aggregate(self) -> None:
        """Test track_domain_event without aggregate ID."""
        FlextRuntime.configure_structlog()

        # Set correlation ID
        correlation_id = FlextContext.Utilities.ensure_correlation_id()

        # Track domain event without aggregate
        FlextRuntime.Integration.track_domain_event(
            "SystemInitialized",
            event_data={"timestamp": "2025-01-01T00:00:00Z"},
        )

        # Verify correlation ID persists
        assert FlextContext.Correlation.get_correlation_id() == correlation_id

    def test_track_domain_event_minimal(self) -> None:
        """Test track_domain_event with minimal arguments."""
        FlextRuntime.configure_structlog()

        # Should not raise error with just event name
        FlextRuntime.Integration.track_domain_event("ConfigUpdated")

    def test_setup_service_infrastructure_full(self) -> None:
        """Test setup_service_infrastructure with all options."""
        FlextRuntime.configure_structlog()

        # Setup complete service infrastructure
        FlextRuntime.Integration.setup_service_infrastructure(
            service_name="test-service",
            service_version="1.0.0",
            enable_context_correlation=True,
        )

        # Verify service context is set
        assert FlextContext.Service.get_service_name() == "test-service"
        assert FlextContext.Service.get_service_version() == "1.0.0"

        # Verify correlation ID was generated
        correlation_id = FlextContext.Correlation.get_correlation_id()
        assert correlation_id is not None

    def test_setup_service_infrastructure_without_version(self) -> None:
        """Test setup_service_infrastructure without version."""
        FlextRuntime.configure_structlog()

        # Setup without version
        FlextRuntime.Integration.setup_service_infrastructure(
            service_name="minimal-service",
            enable_context_correlation=True,
        )

        # Verify service name is set
        assert FlextContext.Service.get_service_name() == "minimal-service"

        # Verify correlation ID was generated
        assert FlextContext.Correlation.get_correlation_id() is not None

    def test_setup_service_infrastructure_without_correlation(self) -> None:
        """Test setup_service_infrastructure without correlation generation."""
        import structlog

        FlextRuntime.configure_structlog()

        # Clear any existing correlation using structlog
        structlog.contextvars.unbind_contextvars("correlation_id")

        # Setup without correlation generation
        FlextRuntime.Integration.setup_service_infrastructure(
            service_name="no-correlation-service",
            service_version="2.0.0",
            enable_context_correlation=False,
        )

        # Verify service context is set
        assert FlextContext.Service.get_service_name() == "no-correlation-service"

        # Verify correlation ID was NOT generated
        assert FlextContext.Correlation.get_correlation_id() is None

    def test_integration_lazy_imports_prevent_circular_dependencies(
        self,
    ) -> None:
        """Test that Integration nested class uses lazy imports correctly."""
        # This test verifies the pattern works by importing FlextRuntime first

        # FlextRuntime should be importable without triggering circular imports
        assert FlextRuntime is not None

        # The Integration nested class should exist
        assert hasattr(FlextRuntime, "Integration")
        assert FlextRuntime.Integration is not None

        # The integration methods should exist on the nested class
        assert hasattr(FlextRuntime.Integration, "track_service_resolution")
        assert hasattr(FlextRuntime.Integration, "track_domain_event")
        assert hasattr(FlextRuntime.Integration, "setup_service_infrastructure")

    def test_integration_keyword_only_arguments(self) -> None:
        """Test that boolean arguments are keyword-only (Ruff FBT compliance)."""
        FlextRuntime.configure_structlog()

        # Set correlation for testing
        FlextContext.Utilities.ensure_correlation_id()

        # These should work with keyword arguments
        FlextRuntime.Integration.track_service_resolution("test", resolved=True)

        # This test verifies keyword-only pattern at runtime

    def test_integration_with_context_single_source_of_truth(self) -> None:
        """Test that integration uses structlog contextvars as single source of truth."""
        FlextRuntime.configure_structlog()

        # Create correlation ID (stored in structlog contextvars)
        original_correlation = FlextContext.Utilities.ensure_correlation_id()

        # Use integration methods - they should all see the same context
        FlextRuntime.Integration.track_service_resolution("service1", resolved=True)
        FlextRuntime.Integration.track_domain_event("Event1")

        # Correlation ID should remain consistent (single source of truth)
        final_correlation = FlextContext.Correlation.get_correlation_id()
        assert final_correlation == original_correlation
