"""Base gRPC service class for FLEXT components.

Provides common patterns and utilities for gRPC service implementations,
eliminating code duplication across different gRPC services.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from collections.abc import Awaitable
    from collections.abc import Callable
    from uuid import UUID

    from flext_core.infrastructure.protocols import LoggingProtocol

T = TypeVar("T")

# Default logger fallback
_default_logger = logging.getLogger(__name__)


class BaseGrpcService:
    """Base class for gRPC service implementations with DIP compliance.

    Provides common functionality like error handling, logging,
    timestamp conversion, and response building patterns.
    Depends on abstractions (LoggingProtocol) not concretions.
    """

    def __init__(
        self,
        service_name: str,
        logger: LoggingProtocol | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize base gRPC service with dependency injection.

        Args:
            service_name: Name of the service for logging
            logger: Logging implementation (injected dependency)
            **kwargs: Additional service-specific initialization parameters

        """
        self.service_name = service_name
        # DIP compliance - depend on abstraction, use fallback if not injected
        # Store kwargs for subclass initialization
        self._init_kwargs = kwargs
        self.logger = logger if logger is not None else _default_logger
        self.logger.info("%s gRPC service initialized", service_name)

    async def execute_with_error_handling(
        self,
        operation: str,
        handler: Callable[[], Awaitable[T]],
        context: Any,  # grpc.aio.ServicerContext
        error_response_factory: Callable[[], T],
    ) -> T:
        """Execute operation with standardized error handling.

        Args:
            operation: Description of the operation for logging
            handler: The actual operation handler function
            context: gRPC service context
            error_response_factory: Function to create error response

        Returns:
            Operation result or error response

        """
        try:
            self.logger.debug("Starting %s", operation)
            result = await handler()
            self.logger.info("%s completed successfully", operation)
        except ValueError as e:
            self.logger.warning("%s validation failed: %s", operation, e)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"{operation} validation failed: {e}")
            return error_response_factory()

        except PermissionError as e:
            self.logger.warning("%s permission denied: %s", operation, e)
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(f"{operation} permission denied: {e}")
            return error_response_factory()

        except FileNotFoundError as e:
            self.logger.warning("%s resource not found: %s", operation, e)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"{operation} resource not found: {e}")
            return error_response_factory()
        except Exception as e:
            self.logger.exception("%s failed", operation)
            context.set_code(internal.invalid)
            context.set_details(f"{operation} failed: {e}")
            return error_response_factory()
        else:
            return result

    def generate_id(self) -> str:
        """Generate a new UUID string.

        Returns:
            UUID string for new entities

        """
        return str(uuid.uuid4())

    def get_current_timestamp(self) -> Timestamp:
        """Get current timestamp as protobuf Timestamp.

        Returns:
            Current timestamp in protobuf format

        """
        timestamp = Timestamp()
        timestamp.GetCurrentTime()
        return timestamp

    def datetime_to_timestamp(self, dt: datetime | None) -> Timestamp:
        """Convert datetime to protobuf Timestamp.

        Args:
            dt: Datetime to convert (can be None)

        Returns:
            Protobuf Timestamp object

        """
        timestamp = Timestamp()
        if dt is not None:
            timestamp.FromDatetime(dt)
        return timestamp

    def timestamp_to_datetime(self, timestamp: Timestamp | None) -> datetime | None:
        """Convert protobuf Timestamp to datetime.

        Args:
            timestamp: Protobuf Timestamp to convert (can be None)

        Returns:
            Datetime object or None

        """
        if timestamp is None:
            return None
        return timestamp.ToDatetime().replace(tzinfo=UTC)

    def build_success_response(self, **kwargs: Any) -> dict[str, Any]:
        """Build a success response with common structure.

        Args:
            **kwargs: Additional fields for the response

        Returns:
            Success response dictionary

        """
        return {
            "success": True,
            "timestamp": self.get_current_timestamp(),
            **kwargs,
        }

    def build_error_response(
        self,
        error_message: str,
        error_code: str = "INTERNAL_ERROR",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build an error response with common structure.

        Args:
            error_message: Error message description
            error_code: Error code identifier
            **kwargs: Additional fields for the response

        Returns:
            Error response dictionary

        """
        return {
            "success": False,
            "error": error_message,
            "error_code": error_code,
            "timestamp": self.get_current_timestamp(),
            **kwargs,
        }

    def get_utc_now(self) -> datetime:
        """Get current UTC datetime.

        Returns:
            Current UTC datetime

        """
        return datetime.now(UTC)

    def log_operation(
        self,
        operation: str,
        entity_id: str | UUID | None = None,
        **extra_context: Any,
    ) -> None:
        """Log operation with structured context.

        Args:
            operation: Operation description
            entity_id: Optional entity ID
            **extra_context: Additional context for logging

        """
        log_context = {
            "service": self.service_name,
            "operation": operation,
            **extra_context,
        }

        if entity_id:
            log_context["entity_id"] = str(entity_id)

        self.logger.info("%s", operation, extra=log_context)

    def log_error(
        self,
        operation: str,
        error: Exception,
        entity_id: str | UUID | None = None,
        **extra_context: Any,
    ) -> None:
        """Log error with structured context.

        Args:
            operation: Operation that failed
            error: Exception that occurred
            entity_id: Optional entity ID
            **extra_context: Additional context for logging

        """
        log_context = {
            "service": self.service_name,
            "operation": operation,
            "error": str(error),
            "error_type": type(error).__name__,
            **extra_context,
        }

        if entity_id:
            log_context["entity_id"] = str(entity_id)

        self.logger.error("%s failed", operation, extra=log_context)

    async def stream_with_error_handling(
        self,
        operation: str,
        stream_generator: Callable[[], AsyncIterator[T]],
        context: Any,  # grpc.aio.ServicerContext
    ) -> AsyncIterator[T]:
        """Execute streaming operation with error handling.

        Args:
            operation: Description of the streaming operation
            stream_generator: Async generator function
            context: gRPC service context

        Yields:
            Stream items or handles errors

        """
        try:
            self.logger.debug("Starting %s stream", operation)
            async for item in stream_generator():
                yield item
            self.logger.info("%s stream completed", operation)
        except Exception as e:
            self.logger.exception("%s stream failed", operation)
            context.set_code(internal.invalid)
            context.set_details(f"{operation} stream failed: {e}")
            # For streaming, we can't return an error response,
            # the client will get the error via context
            return

    def validate_required_field(
        self,
        field_value: Any,
        field_name: str,
        context: Any,  # grpc.aio.ServicerContext
    ) -> bool:
        """Validate that a required field has a value.

        Args:
            field_value: Value to validate
            field_name: Name of the field for error messages
            context: gRPC service context

        Returns:
            True if valid, False if invalid (and sets error context)

        """
        if not field_value:
            self.logger.warning("Missing required field: %s", field_name)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Missing required field: {field_name}")
            return False
        return True

    def handle_not_found(
        self,
        resource_type: str,
        resource_id: str,
        context: Any,  # grpc.aio.ServicerContext
    ) -> None:
        """Handle resource not found error.

        Args:
            resource_type: Type of resource (e.g., "pipeline")
            resource_id: ID of the resource
            context: gRPC service context

        """
        self.logger.warning("%s not found: %s", resource_type, resource_id)
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"{resource_type} {resource_id} not found")
