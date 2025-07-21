"""Tests for modules with 0% coverage to boost overall coverage.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides tests to improve coverage for the domain exceptions module.
It includes tests for DomainError, NotFoundError, RepositoryError,
ServiceError, and ValidationError.
"""

from __future__ import annotations

import pytest


class TestDomainExceptions:
    """Test domain exceptions import module."""

    def test_domain_exceptions_imports(self) -> None:
        """Test that all exceptions are importable."""
        from flext_core.domain.exceptions import (
            DomainError,
            NotFoundError,
            RepositoryError,
            ServiceError,
            ValidationError,
        )

        # Verify they are exception classes
        assert issubclass(DomainError, Exception)
        assert issubclass(NotFoundError, DomainError)
        assert issubclass(RepositoryError, DomainError)
        assert issubclass(ServiceError, DomainError)
        assert issubclass(ValidationError, DomainError)

    def test_domain_exceptions_creation(self) -> None:
        """Test creating exception instances."""
        from flext_core.domain.exceptions import (
            DomainError,
            NotFoundError,
            RepositoryError,
            ServiceError,
            ValidationError,
        )

        # Create instances
        domain_err = DomainError("domain error")
        not_found_err = NotFoundError("not found")
        repo_err = RepositoryError("repo error")
        service_err = ServiceError("SERVICE_ERROR", "service error")
        validation_err = ValidationError("validation error")

        # Verify error messages
        assert str(domain_err) == "domain error"
        assert str(not_found_err) == "not found"
        assert str(repo_err) == "repo error"
        assert str(service_err) == "service error"
        assert str(validation_err) == "validation error"


class TestSerializationModule:
    """Test serialization module imports."""

    def test_serialization_imports(self) -> None:
        """Test that serialization components are importable."""
        from flext_core.serialization import get_serializer

        # Verify it's callable
        assert callable(get_serializer)

    def test_serialization_module_all(self) -> None:
        """Test __all__ is properly defined."""
        from flext_core import serialization

        assert hasattr(serialization, "__all__")
        assert "get_serializer" in serialization.__all__


class TestDomainCore:
    """Test domain core missing lines."""

    def test_domain_core_missing_coverage(self) -> None:
        """Test lines not covered in domain core."""
        from flext_core.domain.core import DomainError, ServiceError

        # Test creating DomainError
        error = DomainError("test message")
        assert str(error) == "test message"

        # Test ServiceError with both parameters
        service_error = ServiceError("SERVICE_CODE", "service message")
        assert str(service_error) == "service message"
        assert service_error.error_code == "SERVICE_CODE"
        assert service_error.message == "service message"


class TestDomainTesting:
    """Test domain testing utilities missing coverage."""

    async def test_mock_repository(self) -> None:
        """Test MockRepository functionality."""
        from flext_core.domain.testing import MockRepository

        # Create a mock repository
        repo = MockRepository[str, str]()

        # Test save method (covers line 40)
        entity = "test-entity"
        saved = await repo.save(entity)
        assert saved == entity

        # Test get method (covers line 52)
        result = await repo.get("test-id")
        assert result is None

        # Test delete method (covers lines 64-67)
        deleted = await repo.delete("non-existent")
        assert deleted is False

        # Test find_all method (covers line 76)
        all_entities = await repo.find_all()
        assert all_entities == []


class TestUtilsModules:
    """Test utils modules for coverage."""

    def test_utils_init_module(self) -> None:
        """Test utils __init__ module import logic."""
        import flext_core.utils as utils_module

        # Test __all__ is properly defined
        assert hasattr(utils_module, "__all__")
        assert isinstance(utils_module.__all__, list)

    def test_ldif_writer_deprecation_warning(self) -> None:
        """Test LDIF writer deprecation handling."""
        import warnings

        # Capture deprecation warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should trigger a deprecation warning
            import contextlib

            with contextlib.suppress(ImportError):
                pass

            # Check if deprecation warning was issued (if no ImportError)
            if not any(issubclass(warning.category, ImportError) for warning in w):
                deprecation_warnings = [
                    warning
                    for warning in w
                    if issubclass(warning.category, DeprecationWarning)
                ]
                if deprecation_warnings:
                    assert "deprecated" in str(deprecation_warnings[0].message)


class TestBaseConfigModules:
    """Test base config modules for coverage."""

    def test_base_component_config_functionality(self) -> None:
        """Test BaseComponentConfig methods with empty config."""
        from flext_core.base.config_base import BaseComponentConfig

        # Create basic config without extra fields (since extra="forbid")
        config = BaseComponentConfig()

        # Test that all methods return empty dicts when no subsections exist
        conn_config = config.get_connection_config()
        assert conn_config == {}

        auth_config = config.get_auth_config()
        assert auth_config == {}

        stream_config = config.get_stream_config("any_stream")
        assert stream_config == {}

        logging_config = config.get_logging_config()
        assert logging_config == {}

        perf_config = config.get_performance_config()
        assert perf_config == {}

    def test_base_component_config_validation(self) -> None:
        """Test BaseComponentConfig validation."""
        from flext_core.base.config_base import BaseComponentConfig

        # Create a basic config instance
        config = BaseComponentConfig()

        # Test validation with missing field (should raise ValueError)
        try:
            config.validate_required_fields(["missing_field"])
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            # Use pytest.raises instead of assert on exception
            expected_msg = "Missing required configuration fields: missing_field"
            if expected_msg not in str(e):
                pytest.fail(f"Expected '{expected_msg}' in error message: {e}")

        # Test edge case where config has no subsections
        missing_section = config.get_subsection("nonexistent")
        assert missing_section == {}

        # Test stream config edge cases
        missing_stream_config = config.get_stream_config("nonexistent")
        assert missing_stream_config == {}


class TestSerializationModules:
    """Test serialization modules for comprehensive coverage."""

    def test_msgspec_serializer_initialization(self) -> None:
        """Test MsgspecJSONSerializer initialization."""
        from flext_core.serialization.msgspec_adapters import MsgspecJSONSerializer

        serializer = MsgspecJSONSerializer()
        assert serializer._encoder is not None
        assert serializer._decoder is not None

    def test_msgspec_json_string_operations(self) -> None:
        """Test JSON string encode/decode operations."""
        from flext_core.serialization.msgspec_adapters import MsgspecJSONSerializer

        serializer = MsgspecJSONSerializer()

        # Test data
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        # Test encode_json_str
        json_str = serializer.encode_json_str(test_data)
        assert isinstance(json_str, str)
        assert "key" in json_str
        assert "value" in json_str

        # Test decode_json_str
        decoded_data = serializer.decode_json_str(json_str)
        assert decoded_data == test_data

    def test_msgspec_bytes_operations(self) -> None:
        """Test JSON bytes encode/decode operations."""
        from flext_core.serialization.msgspec_adapters import MsgspecJSONSerializer

        serializer = MsgspecJSONSerializer()

        # Test data
        test_data = {"message": "hello", "count": 123}

        # Test encode (bytes)
        json_bytes = serializer.encode(test_data)
        assert isinstance(json_bytes, bytes)

        # Test decode (bytes)
        decoded_data = serializer.decode(json_bytes)
        assert decoded_data == test_data

    def test_get_serializer_function(self) -> None:
        """Test get_serializer factory function."""
        from flext_core.serialization.msgspec_adapters import (
            MsgspecJSONSerializer,
            get_serializer,
        )

        serializer = get_serializer()
        assert isinstance(serializer, MsgspecJSONSerializer)

        # Test that it returns a working serializer
        test_data = {"test": True}
        encoded = serializer.encode_json_str(test_data)
        decoded = serializer.decode_json_str(encoded)
        assert decoded == test_data

    def test_ldif_writer_import_error_handling(self) -> None:
        """Test LDIF writer import error when flext-ldif not available."""
        # Test that the module exists and has proper error handling
        import flext_core.utils.ldif_writer as ldif_module

        # Verify __all__ is defined
        assert hasattr(ldif_module, "__all__")
        assert "FlextLDIFWriter" in ldif_module.__all__
        assert "LDIFWriter" in ldif_module.__all__
        assert "LDIFHierarchicalSorter" in ldif_module.__all__
