"""Tests for gRPC base service functionality.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from google.protobuf.timestamp_pb2 import Timestamp

from flext_core.infrastructure.grpc_base import BaseGrpcService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class TestBaseGrpcService:
    """Test BaseGrpcService functionality."""

    def test_initialization(self) -> None:
        """Test service initialization."""
        service = BaseGrpcService("test-service")

        assert service.service_name == "test-service"
        assert service.logger is not None

    def test_generate_id(self) -> None:
        """Test UUID generation."""
        service = BaseGrpcService("test-service")

        generated_id = service.generate_id()

        # Should be a valid UUID string
        assert isinstance(generated_id, str)
        UUID(generated_id)  # Should not raise exception

    def test_get_current_timestamp(self) -> None:
        """Test current timestamp generation."""
        service = BaseGrpcService("test-service")

        timestamp = service.get_current_timestamp()

        assert isinstance(timestamp, Timestamp)
        assert timestamp.seconds > 0

    def test_datetime_to_timestamp_with_datetime(self) -> None:
        """Test datetime to timestamp conversion."""
        service = BaseGrpcService("test-service")
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

        timestamp = service.datetime_to_timestamp(dt)

        assert isinstance(timestamp, Timestamp)
        assert timestamp.seconds > 0

    def test_datetime_to_timestamp_with_none(self) -> None:
        """Test datetime to timestamp conversion with None."""
        service = BaseGrpcService("test-service")

        timestamp = service.datetime_to_timestamp(None)

        assert isinstance(timestamp, Timestamp)
        assert timestamp.seconds == 0

    def test_timestamp_to_datetime_with_timestamp(self) -> None:
        """Test timestamp to datetime conversion."""
        service = BaseGrpcService("test-service")
        timestamp = Timestamp()
        timestamp.FromDatetime(datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC))

        dt = service.timestamp_to_datetime(timestamp)

        assert isinstance(dt, datetime)
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 1
        assert dt.tzinfo == UTC

    def test_timestamp_to_datetime_with_none(self) -> None:
        """Test timestamp to datetime conversion with None."""
        service = BaseGrpcService("test-service")

        dt = service.timestamp_to_datetime(None)

        assert dt is None

    def test_build_success_response(self) -> None:
        """Test success response building."""
        service = BaseGrpcService("test-service")

        response = service.build_success_response(data="test", count=5)

        assert response["success"] is True
        assert isinstance(response["timestamp"], Timestamp)
        assert response["data"] == "test"
        assert response["count"] == 5

    def test_build_error_response_default(self) -> None:
        """Test error response building with defaults."""
        service = BaseGrpcService("test-service")

        response = service.build_error_response("Something went wrong")

        assert response["success"] is False
        assert response["error"] == "Something went wrong"
        assert response["error_code"] == "INTERNAL_ERROR"
        assert isinstance(response["timestamp"], Timestamp)

    def test_build_error_response_custom(self) -> None:
        """Test error response building with custom values."""
        service = BaseGrpcService("test-service")

        response = service.build_error_response(
            "Validation failed", error_code="VALIDATION_ERROR", field="name"
        )

        assert response["success"] is False
        assert response["error"] == "Validation failed"
        assert response["error_code"] == "VALIDATION_ERROR"
        assert response["field"] == "name"
        assert isinstance(response["timestamp"], Timestamp)

    def test_get_utc_now(self) -> None:
        """Test UTC now retrieval."""
        service = BaseGrpcService("test-service")

        now = service.get_utc_now()

        assert isinstance(now, datetime)
        assert now.tzinfo == UTC

    def test_log_operation_without_entity_id(self) -> None:
        """Test operation logging without entity ID."""
        service = BaseGrpcService("test-service")

        # Should not raise any exceptions
        service.log_operation("test_operation", user="admin")

    def test_log_operation_with_string_entity_id(self) -> None:
        """Test operation logging with string entity ID."""
        service = BaseGrpcService("test-service")

        service.log_operation("test_operation", entity_id="test-id-123")

    def test_log_operation_with_uuid_entity_id(self) -> None:
        """Test operation logging with UUID entity ID."""
        service = BaseGrpcService("test-service")
        entity_id = uuid4()

        service.log_operation("test_operation", entity_id=entity_id)

    def test_log_error_without_entity_id(self) -> None:
        """Test error logging without entity ID."""
        service = BaseGrpcService("test-service")
        error = ValueError("Test error")

        service.log_error("test_operation", error, user="admin")

    def test_log_error_with_entity_id(self) -> None:
        """Test error logging with entity ID."""
        service = BaseGrpcService("test-service")
        error = RuntimeError("Test runtime error")
        entity_id = uuid4()

        service.log_error("test_operation", error, entity_id=entity_id)

    def test_validate_required_field_valid(self) -> None:
        """Test required field validation with valid value."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        result = service.validate_required_field("valid_value", "test_field", context)

        assert result is True
        context.set_code.assert_not_called()
        context.set_details.assert_not_called()

    def test_validate_required_field_empty_string(self) -> None:
        """Test required field validation with empty string."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        result = service.validate_required_field("", "test_field", context)

        assert result is False
        context.set_code.assert_called_once()
        context.set_details.assert_called_once_with(
            "Missing required field: test_field"
        )

    def test_validate_required_field_none(self) -> None:
        """Test required field validation with None."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        result = service.validate_required_field(None, "test_field", context)

        assert result is False
        context.set_code.assert_called_once()
        context.set_details.assert_called_once_with(
            "Missing required field: test_field"
        )

    def test_handle_not_found(self) -> None:
        """Test not found error handling."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        service.handle_not_found("pipeline", "test-id-123", context)

        context.set_code.assert_called_once()
        context.set_details.assert_called_once_with("pipeline test-id-123 not found")


class TestBaseGrpcServiceErrorHandling:
    """Test error handling functionality."""

    @pytest.mark.asyncio
    async def test_execute_with_error_handling_success(self) -> None:
        """Test successful operation execution."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        async def successful_handler() -> str:
            return "success_result"

        def error_factory() -> str:
            return "error_result"

        result = await service.execute_with_error_handling(
            "test_operation", successful_handler, context, error_factory
        )

        assert result == "success_result"
        context.set_code.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_with_error_handling_value_error(self) -> None:
        """Test execution with ValueError."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        async def failing_handler() -> str:
            raise ValueError("Invalid value")

        def error_factory() -> str:
            return "error_result"

        result = await service.execute_with_error_handling(
            "test_operation", failing_handler, context, error_factory
        )

        assert result == "error_result"
        context.set_code.assert_called_once()
        context.set_details.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_error_handling_permission_error(self) -> None:
        """Test execution with PermissionError."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        async def failing_handler() -> str:
            raise PermissionError("Access denied")

        def error_factory() -> str:
            return "error_result"

        result = await service.execute_with_error_handling(
            "test_operation", failing_handler, context, error_factory
        )

        assert result == "error_result"
        context.set_code.assert_called_once()
        context.set_details.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_error_handling_file_not_found_error(self) -> None:
        """Test execution with FileNotFoundError."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        async def failing_handler() -> str:
            raise FileNotFoundError("File not found")

        def error_factory() -> str:
            return "error_result"

        result = await service.execute_with_error_handling(
            "test_operation", failing_handler, context, error_factory
        )

        assert result == "error_result"
        context.set_code.assert_called_once()
        context.set_details.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_error_handling_generic_exception(self) -> None:
        """Test execution with generic exception."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        async def failing_handler() -> str:
            raise RuntimeError("Something went wrong")

        def error_factory() -> str:
            return "error_result"

        result = await service.execute_with_error_handling(
            "test_operation", failing_handler, context, error_factory
        )

        assert result == "error_result"
        context.set_code.assert_called_once()
        context.set_details.assert_called_once()


class TestBaseGrpcServiceStreaming:
    """Test streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_with_error_handling_success(self) -> None:
        """Test successful streaming operation."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        async def stream_generator() -> AsyncGenerator[str]:
            yield "item1"
            yield "item2"
            yield "item3"

        results = [
            item
            async for item in service.stream_with_error_handling(
                "test_stream", stream_generator, context
            )
        ]

        assert results == ["item1", "item2", "item3"]
        context.set_code.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_with_error_handling_exception(self) -> None:
        """Test streaming operation with exception."""
        service = BaseGrpcService("test-service")
        context = MagicMock()

        async def failing_stream_generator() -> AsyncGenerator[str]:
            yield "item1"
            raise RuntimeError("Stream failed")

        results = [
            item
            async for item in service.stream_with_error_handling(
                "test_stream", failing_stream_generator, context
            )
        ]

        assert results == ["item1"]
        context.set_code.assert_called_once()
        context.set_details.assert_called_once()


class TestBaseGrpcServiceWithMocks:
    """Test service with mocked dependencies."""

    @patch("flext_core.infrastructure.grpc_base.uuid.uuid4")
    def test_generate_id_with_mock(self, mock_uuid4: MagicMock) -> None:
        """Test ID generation with mocked UUID."""
        mock_uuid = MagicMock()
        mock_uuid.configure_mock(**{"__str__.return_value": "test-uuid-123"})
        mock_uuid4.return_value = mock_uuid

        service = BaseGrpcService("test-service")
        generated_id = service.generate_id()

        assert generated_id == "test-uuid-123"
        mock_uuid4.assert_called_once()
