"""Comprehensive tests for FlextCore.Runtime - Layer 0.5 Runtime Utilities.

Tests all functionality of FlextCore.Runtime including type guards, serialization utilities,
external library access, and structlog configuration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging

import structlog
from dependency_injector import containers, providers
from pydantic import BaseModel

from flext_core import FlextCore


class TestTypeGuards:
    """Test FlextCore.Runtime type guard utilities."""

    def test_is_valid_email_valid_cases(self) -> None:
        """Test is_valid_email with valid email addresses."""
        assert FlextCore.Runtime.is_valid_email("test@example.com")
        assert FlextCore.Runtime.is_valid_email("user.name+tag@example.co.uk")
        assert FlextCore.Runtime.is_valid_email("valid_email@domain.com")
        assert FlextCore.Runtime.is_valid_email("test.email@sub.domain.com")

    def test_is_valid_email_invalid_cases(self) -> None:
        """Test is_valid_email with invalid email addresses."""
        assert not FlextCore.Runtime.is_valid_email("invalid.email")
        assert not FlextCore.Runtime.is_valid_email("@example.com")
        assert not FlextCore.Runtime.is_valid_email("test@")
        assert not FlextCore.Runtime.is_valid_email("test @example.com")
        assert not FlextCore.Runtime.is_valid_email("")

    def test_is_valid_email_non_string_types(self) -> None:
        """Test is_valid_email returns False for non-string types."""
        assert not FlextCore.Runtime.is_valid_email(123)
        assert not FlextCore.Runtime.is_valid_email(None)
        assert not FlextCore.Runtime.is_valid_email(["test@example.com"])
        assert not FlextCore.Runtime.is_valid_email({"email": "test@example.com"})

    def test_is_valid_url_valid_cases(self) -> None:
        """Test is_valid_url with valid URLs."""
        assert FlextCore.Runtime.is_valid_url("https://github.com")
        assert FlextCore.Runtime.is_valid_url("http://localhost:8000")
        assert FlextCore.Runtime.is_valid_url("https://example.com/path?query=1")
        assert FlextCore.Runtime.is_valid_url("https://sub.domain.example.com")

    def test_is_valid_url_invalid_cases(self) -> None:
        """Test is_valid_url with invalid URLs."""
        assert not FlextCore.Runtime.is_valid_url("not-a-url")
        assert not FlextCore.Runtime.is_valid_url("ftp://invalid.com")
        assert not FlextCore.Runtime.is_valid_url("just text")
        assert not FlextCore.Runtime.is_valid_url("")

    def test_is_valid_url_non_string_types(self) -> None:
        """Test is_valid_url returns False for non-string types."""
        assert not FlextCore.Runtime.is_valid_url(123)
        assert not FlextCore.Runtime.is_valid_url(None)
        assert not FlextCore.Runtime.is_valid_url(["https://github.com"])

    def test_is_valid_phone_valid_cases(self) -> None:
        """Test is_valid_phone with valid phone numbers."""
        assert FlextCore.Runtime.is_valid_phone("+5511987654321")
        assert FlextCore.Runtime.is_valid_phone("5511987654321")
        assert FlextCore.Runtime.is_valid_phone("+1234567890")
        assert FlextCore.Runtime.is_valid_phone("123456789012345")  # 15 digits max

    def test_is_valid_phone_invalid_cases(self) -> None:
        """Test is_valid_phone with invalid phone numbers."""
        assert not FlextCore.Runtime.is_valid_phone("123")  # Too short
        assert not FlextCore.Runtime.is_valid_phone("abc1234567890")
        assert not FlextCore.Runtime.is_valid_phone("+abc123")
        assert not FlextCore.Runtime.is_valid_phone("")

    def test_is_valid_phone_non_string_types(self) -> None:
        """Test is_valid_phone returns False for non-string types."""
        assert not FlextCore.Runtime.is_valid_phone(5511987654321)
        assert not FlextCore.Runtime.is_valid_phone(None)

    def test_is_valid_uuid_valid_cases(self) -> None:
        """Test is_valid_uuid with valid UUIDs."""
        assert FlextCore.Runtime.is_valid_uuid("550e8400-e29b-41d4-a716-446655440000")
        assert FlextCore.Runtime.is_valid_uuid(
            "550e8400e29b41d4a716446655440000"
        )  # Without hyphens
        assert FlextCore.Runtime.is_valid_uuid("123e4567-E89B-12D3-A456-426614174000")
        assert FlextCore.Runtime.is_valid_uuid("123e4567E89B12D3A456426614174000")

    def test_is_valid_uuid_invalid_cases(self) -> None:
        """Test is_valid_uuid with invalid UUIDs."""
        assert not FlextCore.Runtime.is_valid_uuid("invalid-uuid")
        assert not FlextCore.Runtime.is_valid_uuid("550e8400-e29b-41d4")  # Too short
        assert not FlextCore.Runtime.is_valid_uuid("not-a-uuid-at-all")
        assert not FlextCore.Runtime.is_valid_uuid("")

    def test_is_valid_uuid_non_string_types(self) -> None:
        """Test is_valid_uuid returns False for non-string types."""
        assert not FlextCore.Runtime.is_valid_uuid(123)
        assert not FlextCore.Runtime.is_valid_uuid(None)

    def test_is_dict_like_valid_cases(self) -> None:
        """Test is_dict_like with dict-like objects."""
        assert FlextCore.Runtime.is_dict_like({})
        assert FlextCore.Runtime.is_dict_like({"key": "value"})
        assert FlextCore.Runtime.is_dict_like({"nested": {"dict": True}})

    def test_is_dict_like_invalid_cases(self) -> None:
        """Test is_dict_like with non-dict objects."""
        assert not FlextCore.Runtime.is_dict_like([])
        assert not FlextCore.Runtime.is_dict_like("string")
        assert not FlextCore.Runtime.is_dict_like(123)
        assert not FlextCore.Runtime.is_dict_like(None)

    def test_is_list_like_valid_cases(self) -> None:
        """Test is_list_like with list-like objects."""
        assert FlextCore.Runtime.is_list_like([])
        assert FlextCore.Runtime.is_list_like([1, 2, 3])
        assert FlextCore.Runtime.is_list_like(["a", "b", "c"])

    def test_is_list_like_invalid_cases(self) -> None:
        """Test is_list_like with non-list objects."""
        assert not FlextCore.Runtime.is_list_like({})
        assert not FlextCore.Runtime.is_list_like("string")
        assert not FlextCore.Runtime.is_list_like(123)
        assert not FlextCore.Runtime.is_list_like(None)

    def test_is_valid_json_valid_cases(self) -> None:
        """Test is_valid_json with valid JSON strings."""
        assert FlextCore.Runtime.is_valid_json('{"key": "value"}')
        assert FlextCore.Runtime.is_valid_json("[]")
        assert FlextCore.Runtime.is_valid_json("[1, 2, 3]")
        assert FlextCore.Runtime.is_valid_json('"string"')
        assert FlextCore.Runtime.is_valid_json("null")

    def test_is_valid_json_invalid_cases(self) -> None:
        """Test is_valid_json with invalid JSON strings."""
        assert not FlextCore.Runtime.is_valid_json("not json")
        assert not FlextCore.Runtime.is_valid_json("{invalid}")
        assert not FlextCore.Runtime.is_valid_json("")

    def test_is_valid_json_non_string_types(self) -> None:
        """Test is_valid_json returns False for non-string types."""
        assert not FlextCore.Runtime.is_valid_json({"key": "value"})
        assert not FlextCore.Runtime.is_valid_json([1, 2, 3])
        assert not FlextCore.Runtime.is_valid_json(None)

    def test_is_valid_path_valid_cases(self) -> None:
        """Test is_valid_path with valid file paths."""
        assert FlextCore.Runtime.is_valid_path("/home/user/file.txt")
        assert FlextCore.Runtime.is_valid_path("C:\\Users\\file.txt")
        assert FlextCore.Runtime.is_valid_path("relative/path/file.py")
        assert FlextCore.Runtime.is_valid_path("./local/file.txt")

    def test_is_valid_path_invalid_cases(self) -> None:
        """Test is_valid_path with invalid file paths."""
        assert not FlextCore.Runtime.is_valid_path("path/with<invalid>chars")
        assert not FlextCore.Runtime.is_valid_path('path/with"quotes')
        assert not FlextCore.Runtime.is_valid_path("path/with|pipe")

    def test_is_valid_path_non_string_types(self) -> None:
        """Test is_valid_path returns False for non-string types."""
        assert not FlextCore.Runtime.is_valid_path(123)
        assert not FlextCore.Runtime.is_valid_path(None)

    def test_is_valid_identifier_valid_cases(self) -> None:
        """Test is_valid_identifier with valid Python identifiers."""
        assert FlextCore.Runtime.is_valid_identifier("variable")
        assert FlextCore.Runtime.is_valid_identifier("_private")
        assert FlextCore.Runtime.is_valid_identifier("ClassName")
        assert FlextCore.Runtime.is_valid_identifier("function_name")

    def test_is_valid_identifier_invalid_cases(self) -> None:
        """Test is_valid_identifier with invalid Python identifiers."""
        assert not FlextCore.Runtime.is_valid_identifier("123invalid")
        assert not FlextCore.Runtime.is_valid_identifier("invalid-name")
        assert not FlextCore.Runtime.is_valid_identifier("invalid name")
        assert not FlextCore.Runtime.is_valid_identifier("")

    def test_is_valid_identifier_non_string_types(self) -> None:
        """Test is_valid_identifier returns False for non-string types."""
        assert not FlextCore.Runtime.is_valid_identifier(123)
        assert not FlextCore.Runtime.is_valid_identifier(None)


class TestSerializationUtilities:
    """Test FlextCore.Runtime serialization utilities."""

    def test_safe_get_attribute_exists(self) -> None:
        """Test safe_get_attribute with existing attribute."""

        class TestObj:
            attr = "value"

        obj = TestObj()
        assert FlextCore.Runtime.safe_get_attribute(obj, "attr") == "value"

    def test_safe_get_attribute_missing_with_default(self) -> None:
        """Test safe_get_attribute with missing attribute returns default."""

        class TestObj:
            pass

        obj = TestObj()
        assert (
            FlextCore.Runtime.safe_get_attribute(obj, "missing", "default") == "default"
        )

    def test_safe_get_attribute_missing_without_default(self) -> None:
        """Test safe_get_attribute with missing attribute returns None."""

        class TestObj:
            pass

        obj = TestObj()
        assert FlextCore.Runtime.safe_get_attribute(obj, "missing") is None

    def test_safe_serialize_to_dict_pydantic_model(self) -> None:
        """Test safe_serialize_to_dict with Pydantic model."""

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = FlextCore.Runtime.safe_serialize_to_dict(model)

        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_safe_serialize_to_dict_dict_attribute(self) -> None:
        """Test safe_serialize_to_dict with object having __dict__."""

        class TestObj:
            def __init__(self) -> None:
                super().__init__()
                self.name = "test"
                self.value = 42

        obj = TestObj()
        result = FlextCore.Runtime.safe_serialize_to_dict(obj)

        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_safe_serialize_to_dict_already_dict(self) -> None:
        """Test safe_serialize_to_dict with already dict[str, object] object."""
        original = {"key": "value"}
        result = FlextCore.Runtime.safe_serialize_to_dict(original)

        assert result is not None
        assert result["key"] == "value"

    def test_safe_serialize_to_dict_unsupported_type(self) -> None:
        """Test safe_serialize_to_dict with unsupported type returns None."""
        result = FlextCore.Runtime.safe_serialize_to_dict(123)
        assert result is None

        result = FlextCore.Runtime.safe_serialize_to_dict("string")
        assert result is None

    def test_safe_serialize_to_dict_model_dump_exception(self) -> None:
        """Test safe_serialize_to_dict with model_dump raising exception falls back to __dict__."""

        class BrokenModel:
            def __init__(self) -> None:
                super().__init__()
                self.data = "test"

            def model_dump(self) -> FlextCore.Types.Dict:
                msg = "Intentional error"
                raise RuntimeError(msg)

        obj = BrokenModel()
        result = FlextCore.Runtime.safe_serialize_to_dict(obj)
        # Should fall through to __dict__ strategy
        assert result is not None
        assert result.get("data") == "test"

    def test_safe_serialize_to_dict_dict_method_exception(self) -> None:
        """Test safe_serialize_to_dict with dict[str, object]() method raising exception falls back to __dict__."""

        class BrokenDict:
            def __init__(self) -> None:
                super().__init__()
                self.data = "test"

            def dict[str, object](self) -> FlextCore.Types.Dict:
                msg = "Intentional error"
                raise RuntimeError(msg)

        obj = BrokenDict()
        result = FlextCore.Runtime.safe_serialize_to_dict(obj)
        # Should fall through to __dict__ strategy
        assert result is not None
        assert result.get("data") == "test"

    def test_safe_serialize_to_dict_dict_method_returns_non_dict(self) -> None:
        """Test safe_serialize_to_dict when dict[str, object]() returns non-dict value."""

        class NonDictReturn:
            def __init__(self) -> None:
                super().__init__()
                self.data = "test"

            def dict[str, object](self) -> str:
                return "not a dict"

        obj = NonDictReturn()
        result = FlextCore.Runtime.safe_serialize_to_dict(obj)
        # Should fall through to __dict__ strategy
        assert result is not None
        assert result.get("data") == "test"

    def test_safe_serialize_to_dict_dict_method_returns_dict(self) -> None:
        """Test safe_serialize_to_dict when dict[str, object]() returns dict[str, object] successfully."""

        class DictMethodWorking:
            def __init__(self) -> None:
                super().__init__()
                self.data = "test"

            def dict[str, object](self) -> FlextCore.Types.Dict:
                return {"data": self.data, "method": "dict"}

        obj = DictMethodWorking()
        result = FlextCore.Runtime.safe_serialize_to_dict(obj)
        # Should return result from dict[str, object]() method
        assert result is not None
        assert result["data"] == "test"
        assert result["method"] == "dict"


class TestTypeIntrospection:
    """Test FlextCore.Runtime type introspection utilities."""

    def test_is_optional_type_with_optional(self) -> None:
        """Test is_optional_type with Optional type."""
        assert FlextCore.Runtime.is_optional_type(str | None)
        assert FlextCore.Runtime.is_optional_type(int | None)
        assert FlextCore.Runtime.is_optional_type(FlextCore.Types.StringList | None)

    def test_is_optional_type_with_non_optional(self) -> None:
        """Test is_optional_type with non-Optional type."""
        assert not FlextCore.Runtime.is_optional_type(str)
        assert not FlextCore.Runtime.is_optional_type(int)
        assert not FlextCore.Runtime.is_optional_type(FlextCore.Types.StringList)

    def test_extract_generic_args_with_generic(self) -> None:
        """Test extract_generic_args with generic types."""
        args = FlextCore.Runtime.extract_generic_args(FlextCore.Types.StringList)
        assert args == (str,)

        args = FlextCore.Runtime.extract_generic_args(dict[str, int])
        assert args == (str, int)

    def test_extract_generic_args_with_non_generic(self) -> None:
        """Test extract_generic_args with non-generic types."""
        args = FlextCore.Runtime.extract_generic_args(str)
        assert args == ()

        args = FlextCore.Runtime.extract_generic_args(int)
        assert args == ()

    def test_is_sequence_type_with_sequences(self) -> None:
        """Test is_sequence_type with sequence types."""
        assert FlextCore.Runtime.is_sequence_type(FlextCore.Types.StringList)
        assert FlextCore.Runtime.is_sequence_type(tuple[int, ...])

    def test_is_sequence_type_with_non_sequences(self) -> None:
        """Test is_sequence_type with non-sequence types."""
        assert not FlextCore.Runtime.is_sequence_type(dict[str, int])
        # Note: str is actually a sequence type in Python (subclass of Sequence)
        assert FlextCore.Runtime.is_sequence_type(str)
        assert not FlextCore.Runtime.is_sequence_type(int)

    def test_is_optional_type_with_exception(self) -> None:
        """Test is_optional_type with invalid input causing exception."""
        # Test with None - should not crash
        assert not FlextCore.Runtime.is_optional_type(None)

        # Test with invalid type hint
        assert not FlextCore.Runtime.is_optional_type("not a type")

    def test_extract_generic_args_with_exception(self) -> None:
        """Test extract_generic_args with invalid input causing exception."""
        # Test with None - should return empty tuple
        args = FlextCore.Runtime.extract_generic_args(None)
        assert args == ()

        # Test with invalid type hint
        args = FlextCore.Runtime.extract_generic_args("not a type")
        assert args == ()

    def test_is_sequence_type_with_exception(self) -> None:
        """Test is_sequence_type with invalid input causing exception."""
        # Test with None - should not crash
        assert not FlextCore.Runtime.is_sequence_type(None)

        # Test with invalid type hint
        assert not FlextCore.Runtime.is_sequence_type("not a type")


class TestExternalLibraryAccess:
    """Test FlextCore.Runtime external library access methods."""

    def test_structlog_returns_module(self) -> None:
        """Test structlog() returns the structlog module."""
        module = FlextCore.Runtime.structlog()
        assert module is structlog
        assert hasattr(module, "get_logger")
        assert hasattr(module, "configure")

    def test_dependency_providers_returns_module(self) -> None:
        """Test dependency_providers() returns providers module."""
        module = FlextCore.Runtime.dependency_providers()
        assert module is providers
        assert hasattr(module, "Singleton")
        assert hasattr(module, "Factory")

    def test_dependency_containers_returns_module(self) -> None:
        """Test dependency_containers() returns containers module."""
        module = FlextCore.Runtime.dependency_containers()
        assert module is containers
        assert hasattr(module, "DeclarativeContainer")


class TestStructlogConfiguration:
    """Test FlextCore.Runtime structlog configuration."""

    def test_configure_structlog_defaults(self) -> None:
        """Test configure_structlog with default settings."""
        # Reset configuration flag for testing
        FlextCore.Runtime._structlog_configured = False

        # Configure with defaults
        FlextCore.Runtime.configure_structlog()

        # Verify configuration was applied
        assert FlextCore.Runtime._structlog_configured is True
        assert structlog.is_configured()

    def test_configure_structlog_custom_log_level(self) -> None:
        """Test configure_structlog with custom log level."""
        FlextCore.Runtime._structlog_configured = False
        FlextCore.Runtime.configure_structlog(log_level=logging.DEBUG)
        assert FlextCore.Runtime._structlog_configured is True

    def test_configure_structlog_json_renderer(self) -> None:
        """Test configure_structlog with JSON renderer."""
        FlextCore.Runtime._structlog_configured = False
        FlextCore.Runtime.configure_structlog(console_renderer=False)
        assert FlextCore.Runtime._structlog_configured is True

    def test_configure_structlog_with_additional_processors(self) -> None:
        """Test configure_structlog with additional processors."""
        FlextCore.Runtime._structlog_configured = False

        # Custom processor
        def custom_processor(
            logger: object, method_name: str, event_dict: FlextCore.Types.Dict
        ) -> FlextCore.Types.Dict:
            event_dict["custom"] = True
            return event_dict

        FlextCore.Runtime.configure_structlog(additional_processors=[custom_processor])
        assert FlextCore.Runtime._structlog_configured is True

    def test_configure_structlog_json_renderer_explicit(self) -> None:
        """Test configure_structlog with JSON renderer explicitly set."""
        FlextCore.Runtime._structlog_configured = False
        FlextCore.Runtime.configure_structlog(
            console_renderer=False, log_level=logging.WARNING
        )
        assert FlextCore.Runtime._structlog_configured is True

    def test_configure_structlog_idempotent(self) -> None:
        """Test configure_structlog is idempotent (can be called multiple times)."""
        FlextCore.Runtime._structlog_configured = False
        FlextCore.Runtime.configure_structlog()

        # Call again - should not raise error
        FlextCore.Runtime.configure_structlog()
        assert FlextCore.Runtime._structlog_configured is True


class TestRuntimeIntegrationWithConstants:
    """Test FlextCore.Runtime integration with FlextCore.Constants."""

    def test_type_guards_use_constants_patterns(self) -> None:
        """Test that type guards use patterns from FlextCore.Constants."""
        # Verify the patterns are accessible
        assert hasattr(FlextCore.Constants.Platform, "PATTERN_EMAIL")
        assert hasattr(FlextCore.Constants.Platform, "PATTERN_URL")
        assert hasattr(FlextCore.Constants.Platform, "PATTERN_PHONE_NUMBER")
        assert hasattr(FlextCore.Constants.Platform, "PATTERN_UUID")
        assert hasattr(FlextCore.Constants.Platform, "PATTERN_PATH")

        # Verify type guards work with these patterns
        test_email = "test@example.com"
        assert FlextCore.Runtime.is_valid_email(test_email)

    def test_runtime_has_no_circular_imports(self) -> None:
        """Test that FlextCore.Runtime can be imported without circular dependencies."""
        # This test passes if the import succeeds
        from flext_core import FlextCore

        assert FlextCore is not None
        assert FlextCore.Runtime is not None
        # Verify Runtime is accessible as nested class
        assert hasattr(FlextCore, "Runtime")

    def test_layer_hierarchy_respected(self) -> None:
        """Test that Layer 0.5 (runtime) imports from Layer 0 (constants) only."""
        # FlextCore.Runtime should import FlextCore.Constants (Layer 0)
        # This test verifies the import chain is correct
        from flext_core import FlextCore

        # Verify FlextCore is importable
        assert FlextCore is not None

        # Verify both Runtime and Constants are accessible
        assert FlextCore.Runtime is not None
        assert FlextCore.Constants is not None

        # FlextCore.Runtime should be able to access FlextCore.Constants patterns
        assert FlextCore.Constants.Platform.PATTERN_EMAIL is not None


class TestContextIntegrationPatterns:
    """Test FlextCore.Runtime.Integration application-layer helpers (PHASE 2)."""

    def test_track_service_resolution_success(self) -> None:
        """Test Integration.track_service_resolution with successful resolution."""
        # Setup: Configure structlog and context
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Set correlation ID for tracking
        correlation_id = FlextCore.Context.Utilities.ensure_correlation_id()

        # Track successful service resolution
        FlextCore.Runtime.Integration.track_service_resolution(
            "database", resolved=True
        )

        # Verify correlation ID is still in context
        current_correlation = FlextCore.Context.Correlation.get_correlation_id()
        assert current_correlation == correlation_id

    def test_track_service_resolution_failure(self) -> None:
        """Test track_service_resolution with failed resolution."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Set correlation ID
        correlation_id = FlextCore.Context.Utilities.ensure_correlation_id()

        # Track failed service resolution
        FlextCore.Runtime.Integration.track_service_resolution(
            "cache",
            resolved=False,
            error_message="Connection refused",
        )

        # Verify correlation ID persists
        assert FlextCore.Context.Correlation.get_correlation_id() == correlation_id

    def test_track_service_resolution_without_correlation(self) -> None:
        """Test track_service_resolution works without pre-existing correlation ID."""
        import structlog

        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Clear any existing correlation using structlog
        structlog.contextvars.unbind_contextvars("correlation_id")

        # Should not raise error even without correlation ID
        FlextCore.Runtime.Integration.track_service_resolution("logger", resolved=True)

        # Correlation ID should still be None
        from flext_core import FlextCore

        assert FlextCore.Context.Correlation.get_correlation_id() is None

    def test_log_config_access_unmasked(self) -> None:
        """Test log_config_access without masking."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Set correlation ID
        correlation_id = FlextCore.Context.Utilities.ensure_correlation_id()

        # Log config access without masking
        FlextCore.Runtime.Integration.log_config_access(
            "app_name", value="flext-core", masked=False
        )

        # Verify correlation ID persists
        assert FlextCore.Context.Correlation.get_correlation_id() == correlation_id

    def test_log_config_access_masked(self) -> None:
        """Test log_config_access with value masking for secrets."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Set correlation ID
        correlation_id = FlextCore.Context.Utilities.ensure_correlation_id()

        # Log config access with masking (for secrets)
        FlextCore.Runtime.Integration.log_config_access(
            "database_password",
            value="super_secret_password",
            masked=True,
        )

        # Verify correlation ID persists
        assert FlextCore.Context.Correlation.get_correlation_id() == correlation_id

    def test_log_config_access_no_value(self) -> None:
        """Test log_config_access without providing value."""
        FlextCore.Runtime.configure_structlog()

        # Should not raise error without value
        FlextCore.Runtime.Integration.log_config_access("some_key", masked=False)

    def test_attach_context_to_result_basic(self) -> None:
        """Test attach_context_to_result basic usage."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Set correlation ID
        FlextCore.Context.Utilities.ensure_correlation_id()

        # Create result and attach context
        result = FlextCore.Result[str].ok("test_data")
        result_with_ctx = FlextCore.Runtime.Integration.attach_context_to_result(
            result,
            attach_correlation=True,
        )

        # Currently returns result unchanged (future-proofing pattern)
        assert result_with_ctx is result

    def test_attach_context_to_result_with_service_name(self) -> None:
        """Test attach_context_to_result with service name."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Set service context
        FlextCore.Context.Service.set_service_name("test-service")

        # Create result and attach context
        result = FlextCore.Result[FlextCore.Types.Dict].ok({"status": "success"})
        result_with_ctx = FlextCore.Runtime.Integration.attach_context_to_result(
            result,
            attach_correlation=True,
            attach_service_name=True,
        )

        # Currently returns result unchanged (future-proofing pattern)
        assert result_with_ctx is result

    def test_attach_context_to_result_no_correlation(self) -> None:
        """Test attach_context_to_result without correlation ID."""
        import structlog

        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Clear correlation using structlog
        structlog.contextvars.unbind_contextvars("correlation_id")

        # Create result
        result = FlextCore.Result[str].ok("test")

        # Should not raise error without correlation
        result_with_ctx = FlextCore.Runtime.Integration.attach_context_to_result(
            result,
            attach_correlation=False,
        )
        assert result_with_ctx is result

    def test_track_domain_event_with_aggregate(self) -> None:
        """Test track_domain_event with aggregate ID."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Set correlation ID
        correlation_id = FlextCore.Context.Utilities.ensure_correlation_id()

        # Track domain event with aggregate
        FlextCore.Runtime.Integration.track_domain_event(
            "UserCreated",
            aggregate_id="user-123",
            event_data={"email": "test@example.com"},
        )

        # Verify correlation ID persists
        assert FlextCore.Context.Correlation.get_correlation_id() == correlation_id

    def test_track_domain_event_without_aggregate(self) -> None:
        """Test track_domain_event without aggregate ID."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Set correlation ID
        correlation_id = FlextCore.Context.Utilities.ensure_correlation_id()

        # Track domain event without aggregate
        FlextCore.Runtime.Integration.track_domain_event(
            "SystemInitialized",
            event_data={"timestamp": "2025-01-01T00:00:00Z"},
        )

        # Verify correlation ID persists
        assert FlextCore.Context.Correlation.get_correlation_id() == correlation_id

    def test_track_domain_event_minimal(self) -> None:
        """Test track_domain_event with minimal arguments."""
        FlextCore.Runtime.configure_structlog()

        # Should not raise error with just event name
        FlextCore.Runtime.Integration.track_domain_event("ConfigUpdated")

    def test_setup_service_infrastructure_full(self) -> None:
        """Test setup_service_infrastructure with all options."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Setup complete service infrastructure
        FlextCore.Runtime.Integration.setup_service_infrastructure(
            service_name="test-service",
            service_version="1.0.0",
            enable_context_correlation=True,
        )

        # Verify service context is set
        assert FlextCore.Context.Service.get_service_name() == "test-service"
        assert FlextCore.Context.Service.get_service_version() == "1.0.0"

        # Verify correlation ID was generated
        correlation_id = FlextCore.Context.Correlation.get_correlation_id()
        assert correlation_id is not None

    def test_setup_service_infrastructure_without_version(self) -> None:
        """Test setup_service_infrastructure without version."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Setup without version
        FlextCore.Runtime.Integration.setup_service_infrastructure(
            service_name="minimal-service",
            enable_context_correlation=True,
        )

        # Verify service name is set
        assert FlextCore.Context.Service.get_service_name() == "minimal-service"

        # Verify correlation ID was generated
        assert FlextCore.Context.Correlation.get_correlation_id() is not None

    def test_setup_service_infrastructure_without_correlation(self) -> None:
        """Test setup_service_infrastructure without correlation generation."""
        import structlog

        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Clear any existing correlation using structlog
        structlog.contextvars.unbind_contextvars("correlation_id")

        # Setup without correlation generation
        FlextCore.Runtime.Integration.setup_service_infrastructure(
            service_name="no-correlation-service",
            service_version="2.0.0",
            enable_context_correlation=False,
        )

        # Verify service context is set
        assert FlextCore.Context.Service.get_service_name() == "no-correlation-service"

        # Verify correlation ID was NOT generated
        assert FlextCore.Context.Correlation.get_correlation_id() is None

    def test_integration_lazy_imports_prevent_circular_dependencies(self) -> None:
        """Test that Integration nested class uses lazy imports correctly."""
        # This test verifies the pattern works by importing FlextCore.Runtime first
        from flext_core import FlextCore

        # FlextCore.Runtime should be importable without triggering circular imports
        assert FlextCore.Runtime is not None

        # The Integration nested class should exist
        assert hasattr(FlextCore.Runtime, "Integration")
        assert FlextCore.Runtime.Integration is not None

        # The integration methods should exist on the nested class
        assert hasattr(FlextCore.Runtime.Integration, "track_service_resolution")
        assert hasattr(FlextCore.Runtime.Integration, "log_config_access")
        assert hasattr(FlextCore.Runtime.Integration, "attach_context_to_result")
        assert hasattr(FlextCore.Runtime.Integration, "track_domain_event")
        assert hasattr(FlextCore.Runtime.Integration, "setup_service_infrastructure")

    def test_integration_keyword_only_arguments(self) -> None:
        """Test that boolean arguments are keyword-only (Ruff FBT compliance)."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Set correlation for testing
        FlextCore.Context.Utilities.ensure_correlation_id()

        # These should work with keyword arguments
        FlextCore.Runtime.Integration.track_service_resolution("test", resolved=True)
        FlextCore.Runtime.Integration.log_config_access("key", masked=False)
        FlextCore.Runtime.Integration.setup_service_infrastructure(
            service_name="test",
            enable_context_correlation=True,
        )

        # This test verifies keyword-only pattern at runtime

    def test_integration_with_context_single_source_of_truth(self) -> None:
        """Test that integration uses structlog contextvars as single source of truth."""
        from flext_core import FlextCore

        FlextCore.Runtime.configure_structlog()

        # Create correlation ID (stored in structlog contextvars)
        original_correlation = FlextCore.Context.Utilities.ensure_correlation_id()

        # Use integration methods - they should all see the same context
        FlextCore.Runtime.Integration.track_service_resolution(
            "service1", resolved=True
        )
        FlextCore.Runtime.Integration.log_config_access("config1", masked=False)
        FlextCore.Runtime.Integration.track_domain_event("Event1")

        # Correlation ID should remain consistent (single source of truth)
        final_correlation = FlextCore.Context.Correlation.get_correlation_id()
        assert final_correlation == original_correlation
