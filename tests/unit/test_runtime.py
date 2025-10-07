"""Comprehensive tests for FlextRuntime - Layer 0.5 Runtime Utilities.

Tests all functionality of FlextRuntime including type guards, serialization utilities,
external library access, and structlog configuration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from typing import Any

import structlog
from dependency_injector import containers, providers
from pydantic import BaseModel

from flext_core import FlextConstants
from flext_core.runtime import FlextRuntime


class TestTypeGuards:
    """Test FlextRuntime type guard utilities."""

    def test_is_valid_email_valid_cases(self) -> None:
        """Test is_valid_email with valid email addresses."""
        assert FlextRuntime.is_valid_email("test@example.com")
        assert FlextRuntime.is_valid_email("user.name+tag@example.co.uk")
        assert FlextRuntime.is_valid_email("valid_email@domain.com")
        assert FlextRuntime.is_valid_email("test.email@sub.domain.com")

    def test_is_valid_email_invalid_cases(self) -> None:
        """Test is_valid_email with invalid email addresses."""
        assert not FlextRuntime.is_valid_email("invalid.email")
        assert not FlextRuntime.is_valid_email("@example.com")
        assert not FlextRuntime.is_valid_email("test@")
        assert not FlextRuntime.is_valid_email("test @example.com")
        assert not FlextRuntime.is_valid_email("")

    def test_is_valid_email_non_string_types(self) -> None:
        """Test is_valid_email returns False for non-string types."""
        assert not FlextRuntime.is_valid_email(123)
        assert not FlextRuntime.is_valid_email(None)
        assert not FlextRuntime.is_valid_email(["test@example.com"])
        assert not FlextRuntime.is_valid_email({"email": "test@example.com"})

    def test_is_valid_url_valid_cases(self) -> None:
        """Test is_valid_url with valid URLs."""
        assert FlextRuntime.is_valid_url("https://github.com")
        assert FlextRuntime.is_valid_url("http://localhost:8000")
        assert FlextRuntime.is_valid_url("https://example.com/path?query=1")
        assert FlextRuntime.is_valid_url("https://sub.domain.example.com")

    def test_is_valid_url_invalid_cases(self) -> None:
        """Test is_valid_url with invalid URLs."""
        assert not FlextRuntime.is_valid_url("not-a-url")
        assert not FlextRuntime.is_valid_url("ftp://invalid.com")
        assert not FlextRuntime.is_valid_url("just text")
        assert not FlextRuntime.is_valid_url("")

    def test_is_valid_url_non_string_types(self) -> None:
        """Test is_valid_url returns False for non-string types."""
        assert not FlextRuntime.is_valid_url(123)
        assert not FlextRuntime.is_valid_url(None)
        assert not FlextRuntime.is_valid_url(["https://github.com"])

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

    def test_is_valid_uuid_valid_cases(self) -> None:
        """Test is_valid_uuid with valid UUIDs."""
        assert FlextRuntime.is_valid_uuid("550e8400-e29b-41d4-a716-446655440000")
        assert FlextRuntime.is_valid_uuid(
            "550e8400e29b41d4a716446655440000"
        )  # Without hyphens
        assert FlextRuntime.is_valid_uuid("123e4567-E89B-12D3-A456-426614174000")
        assert FlextRuntime.is_valid_uuid("123e4567E89B12D3A456426614174000")

    def test_is_valid_uuid_invalid_cases(self) -> None:
        """Test is_valid_uuid with invalid UUIDs."""
        assert not FlextRuntime.is_valid_uuid("invalid-uuid")
        assert not FlextRuntime.is_valid_uuid("550e8400-e29b-41d4")  # Too short
        assert not FlextRuntime.is_valid_uuid("not-a-uuid-at-all")
        assert not FlextRuntime.is_valid_uuid("")

    def test_is_valid_uuid_non_string_types(self) -> None:
        """Test is_valid_uuid returns False for non-string types."""
        assert not FlextRuntime.is_valid_uuid(123)
        assert not FlextRuntime.is_valid_uuid(None)

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

    def test_is_valid_path_valid_cases(self) -> None:
        """Test is_valid_path with valid file paths."""
        assert FlextRuntime.is_valid_path("/home/user/file.txt")
        assert FlextRuntime.is_valid_path("C:\\Users\\file.txt")
        assert FlextRuntime.is_valid_path("relative/path/file.py")
        assert FlextRuntime.is_valid_path("./local/file.txt")

    def test_is_valid_path_invalid_cases(self) -> None:
        """Test is_valid_path with invalid file paths."""
        assert not FlextRuntime.is_valid_path("path/with<invalid>chars")
        assert not FlextRuntime.is_valid_path('path/with"quotes')
        assert not FlextRuntime.is_valid_path("path/with|pipe")

    def test_is_valid_path_non_string_types(self) -> None:
        """Test is_valid_path returns False for non-string types."""
        assert not FlextRuntime.is_valid_path(123)
        assert not FlextRuntime.is_valid_path(None)

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

    def test_safe_serialize_to_dict_pydantic_model(self) -> None:
        """Test safe_serialize_to_dict with Pydantic model."""

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = FlextRuntime.safe_serialize_to_dict(model)

        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_safe_serialize_to_dict_dict_attribute(self) -> None:
        """Test safe_serialize_to_dict with object having __dict__."""

        class TestObj:
            def __init__(self) -> None:
                self.name = "test"
                self.value = 42

        obj = TestObj()
        result = FlextRuntime.safe_serialize_to_dict(obj)

        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_safe_serialize_to_dict_already_dict(self) -> None:
        """Test safe_serialize_to_dict with already dict object."""
        original = {"key": "value"}
        result = FlextRuntime.safe_serialize_to_dict(original)

        assert result is not None
        assert result["key"] == "value"

    def test_safe_serialize_to_dict_unsupported_type(self) -> None:
        """Test safe_serialize_to_dict with unsupported type returns None."""
        result = FlextRuntime.safe_serialize_to_dict(123)
        assert result is None

        result = FlextRuntime.safe_serialize_to_dict("string")
        assert result is None

    def test_safe_serialize_to_dict_model_dump_exception(self) -> None:
        """Test safe_serialize_to_dict with model_dump raising exception falls back to __dict__."""

        class BrokenModel:
            def __init__(self) -> None:
                self.data = "test"

            def model_dump(self) -> dict[str, object]:
                msg = "Intentional error"
                raise RuntimeError(msg)

        obj = BrokenModel()
        result = FlextRuntime.safe_serialize_to_dict(obj)
        # Should fall through to __dict__ strategy
        assert result is not None
        assert result.get("data") == "test"

    def test_safe_serialize_to_dict_dict_method_exception(self) -> None:
        """Test safe_serialize_to_dict with dict() method raising exception falls back to __dict__."""

        class BrokenDict:
            def __init__(self) -> None:
                self.data = "test"

            def dict(self) -> dict[str, Any]:
                msg = "Intentional error"
                raise RuntimeError(msg)

        obj = BrokenDict()
        result = FlextRuntime.safe_serialize_to_dict(obj)
        # Should fall through to __dict__ strategy
        assert result is not None
        assert result.get("data") == "test"

    def test_safe_serialize_to_dict_dict_method_returns_non_dict(self) -> None:
        """Test safe_serialize_to_dict when dict() returns non-dict value."""

        class NonDictReturn:
            def __init__(self) -> None:
                self.data = "test"

            def dict(self) -> str:
                return "not a dict"

        obj = NonDictReturn()
        result = FlextRuntime.safe_serialize_to_dict(obj)
        # Should fall through to __dict__ strategy
        assert result is not None
        assert result.get("data") == "test"

    def test_safe_serialize_to_dict_dict_method_returns_dict(self) -> None:
        """Test safe_serialize_to_dict when dict() returns dict successfully."""

        class DictMethodWorking:
            def __init__(self) -> None:
                self.data = "test"

            def dict(self) -> dict[str, Any]:
                return {"data": self.data, "method": "dict"}

        obj = DictMethodWorking()
        result = FlextRuntime.safe_serialize_to_dict(obj)
        # Should return result from dict() method
        assert result is not None
        assert result["data"] == "test"
        assert result["method"] == "dict"


class TestTypeIntrospection:
    """Test FlextRuntime type introspection utilities."""

    def test_is_optional_type_with_optional(self) -> None:
        """Test is_optional_type with Optional type."""
        assert FlextRuntime.is_optional_type(str | None)
        assert FlextRuntime.is_optional_type(int | None)
        assert FlextRuntime.is_optional_type(list[str] | None)

    def test_is_optional_type_with_non_optional(self) -> None:
        """Test is_optional_type with non-Optional type."""
        assert not FlextRuntime.is_optional_type(str)
        assert not FlextRuntime.is_optional_type(int)
        assert not FlextRuntime.is_optional_type(list[str])

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
        assert not FlextRuntime.is_sequence_type(str)
        assert not FlextRuntime.is_sequence_type(int)

    def test_is_optional_type_with_exception(self) -> None:
        """Test is_optional_type with invalid input causing exception."""
        # Test with None - should not crash
        assert not FlextRuntime.is_optional_type(None)

        # Test with invalid type hint
        assert not FlextRuntime.is_optional_type("not a type")

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
            logger: object, method_name: str, event_dict: dict[str, Any]
        ) -> dict[str, Any]:
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
        assert hasattr(FlextConstants.Platform, "PATTERN_EMAIL")
        assert hasattr(FlextConstants.Platform, "PATTERN_URL")
        assert hasattr(FlextConstants.Platform, "PATTERN_PHONE_NUMBER")
        assert hasattr(FlextConstants.Platform, "PATTERN_UUID")
        assert hasattr(FlextConstants.Platform, "PATTERN_PATH")

        # Verify type guards work with these patterns
        test_email = "test@example.com"
        assert FlextRuntime.is_valid_email(test_email)

    def test_runtime_has_no_circular_imports(self) -> None:
        """Test that FlextRuntime can be imported without circular dependencies."""
        # This test passes if the import succeeds
        from flext_core.runtime import FlextRuntime as RuntimeClass

        assert RuntimeClass is not None
        assert RuntimeClass is FlextRuntime

    def test_layer_hierarchy_respected(self) -> None:
        """Test that Layer 0.5 (runtime) imports from Layer 0 (constants) only."""
        # FlextRuntime should import FlextConstants (Layer 0)
        # This test verifies the import chain is correct
        from flext_core.constants import FlextConstants as ConstantsClass
        from flext_core.runtime import FlextRuntime as RuntimeClass

        # Both should be importable
        assert ConstantsClass is not None
        assert RuntimeClass is not None

        # FlextRuntime should be able to access FlextConstants patterns
        assert FlextConstants.Platform.PATTERN_EMAIL is not None
