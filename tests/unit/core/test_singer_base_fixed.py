"""Fixed comprehensive tests for singer_base module.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Tests all Singer base exception classes with correct string format expectations.
"""

from __future__ import annotations

import pytest

from flext_core.singer_base import (
    FlextSingerError,
    FlextTapError,
    FlextTargetError,
    FlextTransformError,
)


class TestFlextSingerError:
    """Test base Singer error class."""

    def test_basic_init(self) -> None:
        """Test basic Singer error initialization."""
        error = FlextSingerError()
        assert "Singer operation error" in str(error)
        assert error.error_code == "SINGER_ERROR"

    def test_init_with_all_context(self) -> None:
        """Test Singer error with all context parameters."""
        error = FlextSingerError(
            "Complex error",
            component_type="target",
            stream_name="orders",
            custom_field="custom_value",
        )
        assert "Complex error" in str(error)
        assert error.context["component_type"] == "target"
        assert error.context["stream_name"] == "orders"
        assert error.context["custom_field"] == "custom_value"


class TestFlextTapError:
    """Test Singer tap error class."""

    def test_basic_init(self) -> None:
        """Test basic tap error initialization."""
        error = FlextTapError()
        assert "Tap operation error" in str(error)
        assert error.context["component_type"] == "tap"

    def test_init_with_source_system(self) -> None:
        """Test tap error with source system."""
        error = FlextTapError("Tap error", source_system="postgresql")
        assert error.context["source_system"] == "postgresql"
        assert error.context["component_type"] == "tap"


class TestFlextTargetError:
    """Test Singer target error class."""

    def test_basic_init(self) -> None:
        """Test basic target error initialization."""
        error = FlextTargetError()
        assert "Target operation error" in str(error)
        assert error.context["component_type"] == "target"


class TestFlextTransformError:
    """Test Singer transform error class."""

    def test_basic_init(self) -> None:
        """Test basic transform error initialization."""
        error = FlextTransformError()
        assert "Transform operation error" in str(error)
        assert error.context["component_type"] == "transform"


class TestInheritanceAndHierarchy:
    """Test exception inheritance hierarchy."""

    def test_singer_error_inheritance(self) -> None:
        """Test that Singer error inherits from FlextError."""
        from flext_core.exceptions import FlextError

        error = FlextSingerError()
        assert isinstance(error, FlextError)

    def test_component_error_inheritance(self) -> None:
        """Test that component errors inherit from FlextSingerError."""
        tap_error = FlextTapError()
        target_error = FlextTargetError()
        transform_error = FlextTransformError()

        assert isinstance(tap_error, FlextSingerError)
        assert isinstance(target_error, FlextSingerError)
        assert isinstance(transform_error, FlextSingerError)


class TestErrorRaising:
    """Test that errors can be raised and caught properly."""

    def test_raise_singer_error(self) -> None:
        """Test raising and catching Singer error."""
        test_error_message = "Test error"
        with pytest.raises(FlextSingerError) as exc_info:
            raise FlextSingerError(test_error_message)

        assert test_error_message in str(exc_info.value)

    def test_catch_as_base_singer_error(self) -> None:
        """Test catching component error as base Singer error."""
        target_error_message = "Target failed"
        with pytest.raises(FlextSingerError):
            raise FlextTargetError(target_error_message)


class TestModuleExports:
    """Test module exports and public API."""

    def test_all_exports_exist(self) -> None:
        """Test that all declared exports exist."""
        from flext_core import singer_base

        expected_exports = [
            "FlextSingerAuthenticationError",
            "FlextSingerConfigurationError",
            "FlextSingerConnectionError",
            "FlextSingerError",
            "FlextSingerProcessingError",
            "FlextSingerValidationError",
            "FlextTapError",
            "FlextTargetError",
            "FlextTransformError",
        ]

        for export_name in expected_exports:
            assert hasattr(singer_base, export_name)
            assert export_name in singer_base.__all__
