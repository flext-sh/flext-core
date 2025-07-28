"""FLEXT Core Payload Module.

Comprehensive type-safe payload system for structured data transport with validation
and metadata management. Implements consolidated architecture pattern with mixin
inheritance.

Architecture:
    - Type-safe payload containers with generic type support
    - Pydantic-based validation with strict immutability
    - Multiple inheritance from specialized mixin base classes
    - Railway-oriented programming with FlextResult integration
    - Specialized payload types for common message patterns

Payload System Components:
    - FlextPayload[T]: Generic payload container with metadata support
    - FlextMessage: Specialized string message payload with level validation
    - FlextEvent: Domain event payload with aggregate tracking
    - Factory methods: Type-safe payload creation with validation
    - Metadata management: Key-value metadata with type safety

Maintenance Guidelines:
    - Add new specialized payload types by inheriting from FlextPayload[T]
    - Use FlextResult pattern for all factory methods that can fail
    - Maintain immutability through Pydantic frozen configuration
    - Keep validation logic consistent with base validation patterns
    - Integrate logging through FlextLoggableMixin for observability

Design Decisions:
    - Generic type parameter [T] for type-safe data transport
    - Frozen Pydantic models for immutability and thread safety
    - Multiple inheritance from mixin classes for behavior composition
    - Factory method pattern for validated payload creation
    - Metadata as separate dict for flexible extension
    - Railway programming pattern for error handling

Transport Features:
    - Type-safe generic payload container with compile-time type checking
    - Automatic validation through Pydantic with comprehensive error reporting
    - Immutable payload objects preventing accidental modification
    - Rich metadata support for transport context and debugging
    - Structured logging integration for payload lifecycle tracking
    - Serialization support through SerializableMixin inheritance

Dependencies:
    - pydantic: Data validation and immutable model configuration
    - mixins: Serializable, validatable, and loggable behavior patterns
    - result: FlextResult pattern for consistent error handling
    - types: Generic type variables and payload-specific type aliases
    - validation: Base validation utilities for data integrity

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from flext_core.flext_types import FlextTypes
from flext_core.loggings import FlextLoggerFactory
from flext_core.mixins import (
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextValidatableMixin,
)
from flext_core.result import FlextResult
from flext_core.validation import FlextValidators

# Use FlextTypes for type variables
T = FlextTypes.T


class FlextPayload[T](
    BaseModel,
    FlextSerializableMixin,
    FlextValidatableMixin,
    FlextLoggableMixin,
):
    """Generic type-safe payload container for structured data transport and validation.

    Comprehensive payload implementation providing immutable data transport with
    automatic validation, serialization, and metadata management. Combines Pydantic
    validation with mixin functionality for complete data integrity.

    Architecture:
        - Generic type parameter [T] for compile-time type safety
        - Pydantic BaseModel for automatic validation and serialization
        - Multiple inheritance from specialized mixin classes
        - Frozen configuration for immutability and thread safety
        - Rich metadata support for transport context

    Transport Features:
        - Type-safe data encapsulation with generic constraints
        - Automatic validation through Pydantic field validation
        - Immutable payload objects preventing modification after creation
        - Metadata dictionary for transport context and debugging information
        - Structured logging integration through FlextLoggableMixin
        - Serialization support through FlextSerializableMixin

    Validation Integration:
        - Automatic field validation through Pydantic configuration
        - Custom validation through FlextValidatableMixin methods
        - Railway-oriented creation through factory methods
        - Comprehensive error reporting for validation failures

    Metadata Management:
        - Key-value metadata storage with type safety
        - Immutable metadata updates through copy-on-write pattern
        - Metadata querying and existence checking methods
        - Integration with logging for observability

    Usage Patterns:
        # Basic payload creation
        payload = FlextPayload(data={"user_id": "123"})

        # Type-safe payload
        user_payload: FlextPayload[User] = FlextPayload(data=user_instance)

        # Payload with metadata
        order_payload = FlextPayload(
            data=order_data,
            metadata={
                "version": "1.0",
                "source": "api",
                "timestamp": _BaseGenerators.generate_timestamp(),
            },
        )

        # Factory method with validation
        result = FlextPayload.create(
            data=complex_data,
            version="2.0",
            source="batch_processor"
        )
        if result.is_success:
            validated_payload = result.data

        # Metadata operations
        enhanced_payload = payload.with_metadata(
            processed_at=_BaseGenerators.generate_timestamp(),
            processor_id="worker_001"
        )

        if enhanced_payload.has_metadata("version"):
            version = enhanced_payload.get_metadata("version")

    Type Safety:
        - Generic type parameter constrains data type at compile time
        - Type checkers can verify payload content type compatibility
        - Runtime type validation through Pydantic field constraints
        - Safe metadata access with default value support
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={
            "description": "Type-safe payload container",
            "examples": [
                {"data": {"id": 1}, "metadata": {"version": "1.0"}},
                {"data": "simple string", "metadata": {"type": "text"}},
            ],
        },
    )

    data: T = Field(description="Payload data")
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Optional metadata",
    )

    @classmethod
    def create(cls, data: T, **metadata: object) -> FlextResult[FlextPayload[T]]:
        """Create payload with validation.

        Args:
            data: Payload data
            **metadata: Optional metadata fields

        Returns:
            Result containing payload or error

        """
        # Import logger directly for class methods

        logger = FlextLoggerFactory.get_logger(f"{cls.__module__}.{cls.__name__}")

        logger.debug(
            "Creating payload",
            data_type=type(data).__name__,
            metadata_keys=list(metadata.keys()),
        )

        try:
            # Keys in **metadata are always strings, so no validation needed
            payload = cls(data=data, metadata=metadata)
            logger.debug("Payload created successfully", payload_id=id(payload))
            return FlextResult.ok(payload)
        except (TypeError, ValueError, ValidationError) as e:
            logger.exception("Failed to create payload")
            return FlextResult.fail(f"Failed to create payload: {e}")

    def with_metadata(self, **additional: object) -> FlextPayload[T]:
        """Create new payload with additional metadata.

        Args:
            **additional: Metadata to add/update

        Returns:
            New payload with updated metadata

        """
        self.logger.debug(
            "Adding metadata to payload",
            additional_keys=list(additional.keys()),
        )

        # Keys in **additional are always strings, so merge directly
        new_metadata = {**self.metadata, **additional}
        return FlextPayload(data=self.data, metadata=new_metadata)

    def get_metadata(self, key: str, default: object | None = None) -> object | None:
        """Get metadata value by key.

        Args:
            key: Metadata key
            default: Default if key not found

        Returns:
            Metadata value or default

        """
        return self.metadata.get(key, default)

    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists.

        Args:
            key: Metadata key to check

        Returns:
            True if key exists

        """
        return key in self.metadata

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with data and metadata

        """
        return {
            "data": self.data,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        data_repr = repr(self.data)
        max_repr_length = 50
        if len(data_repr) > max_repr_length:
            data_repr = f"{data_repr[:47]}..."
        meta_count = len(self.metadata)
        return f"FlextPayload(data={data_repr}, metadata_keys={meta_count})"


# =============================================================================
# SPECIALIZED PAYLOAD TYPES - Message and Event patterns
# =============================================================================


class FlextMessage(FlextPayload[str]):
    """Specialized string message payload with level validation and source tracking.

    Purpose-built payload for text messages with structured metadata including
    message level classification and source identification. Extends FlextPayload[str]
    with message-specific validation and factory methods.

    Architecture:
        - Inherits from FlextPayload[str] for string-specific type safety
        - Level-based message classification with validation
        - Source tracking for message origin identification
        - Factory method pattern for validated message creation
        - Integration with logging system for message lifecycle tracking

    Message Classification:
        - Supports standard logging levels: info, warning, error, debug, critical
        - Automatic level validation with fallback to 'info' for invalid levels
        - Level-specific metadata enrichment for message categorization
        - Source attribution for message traceability

    Validation Features:
        - Non-empty string validation for message content
        - Level validation against predefined valid values
        - Source validation when provided (optional parameter)
        - Comprehensive error reporting through FlextResult pattern

    Usage Patterns:
        # Basic message creation
        result = FlextMessage.create_message("User logged in successfully")
        if result.is_success:
            message = result.data

        # Message with level and source
        error_result = FlextMessage.create_message(
            "Database connection failed",
            level="error",
            source="database_service"
        )

        # Access message properties
        if message.has_metadata("level"):
            level = message.get_metadata("level")  # Returns message level

        # Extend with additional metadata
        enriched_message = message.with_metadata(
            timestamp=_BaseGenerators.generate_timestamp(),
            user_id="user_123"
        )
    """

    @classmethod
    def create_message(
        cls,
        message: str,
        *,
        level: str = "info",
        source: str | None = None,
    ) -> FlextResult[FlextMessage]:
        """Create message payload.

        Args:
            message: Message text
            level: Message level (info, warning, error)
            source: Message source

        Returns:
            Result containing message payload

        """
        # Import logger directly for class methods

        logger = FlextLoggerFactory.get_logger(f"{cls.__module__}.{cls.__name__}")

        # Validate message using FlextValidation
        if not FlextValidators.is_non_empty_string(message):
            logger.error("Invalid message - empty or not string")
            return FlextResult.fail("Message cannot be empty")

        # Validate level
        valid_levels = ["info", "warning", "error", "debug", "critical"]
        if level not in valid_levels:
            logger.warning("Invalid message level, using 'info'", level=level)
            level = "info"

        metadata: dict[str, object] = {"level": level}
        if source:
            metadata["source"] = source

        logger.debug("Creating message payload", level=level, source=source)

        # Create FlextMessage instance directly
        try:
            instance = cls(data=message, metadata=metadata)
            return FlextResult.ok(instance)
        except (TypeError, ValueError, ValidationError) as e:
            return FlextResult.fail(f"Failed to create message: {e}")


class FlextEvent(FlextPayload[Mapping[str, object]]):
    """Domain event payload with aggregate tracking and versioning support.

    Specialized payload for domain-driven design events with comprehensive metadata
    for event sourcing, aggregate identification, and version tracking. Extends
    FlextPayload[Mapping[str, object]] for structured event data transport.

    Architecture:
        - Inherits from FlextPayload[Mapping[str, object]] for structured event data
        - Event type classification with validation requirements
        - Aggregate identification for domain entity correlation
        - Version tracking for event ordering and conflict resolution
        - Factory method pattern for validated event creation

    Event Sourcing Features:
        - Event type identification for event handler routing
        - Aggregate ID correlation for entity reconstruction
        - Version tracking for optimistic concurrency control
        - Structured event data with key-value mapping constraint
        - Comprehensive validation for event integrity

    Domain-Driven Design Integration:
        - Event type naming conventions for domain clarity
        - Aggregate boundary respect through ID correlation
        - Event versioning for evolution and backward compatibility
        - Rich event data structure supporting complex domain information
        - Metadata enrichment for event processing context

    Validation Requirements:
        - Non-empty string validation for event type classification
        - Aggregate ID validation when provided (must be non-empty string)
        - Version validation ensuring non-negative integer values
        - Event data structure validation through Mapping constraint
        - Factory method validation with comprehensive error reporting

    Usage Patterns:
        # Basic domain event
        result = FlextEvent.create_event(
            event_type="UserRegistered",
            event_data={"user_id": "123", "email": "user@example.com"}
        )

        # Event with aggregate tracking
        order_event = FlextEvent.create_event(
            event_type="OrderCreated",
            event_data={"order_id": "456", "amount": 100.00, "items": [...]},
            aggregate_id="order_456",
            version=1
        )

        # Access event metadata
        event_type = event.get_metadata("event_type")
        aggregate_id = event.get_metadata("aggregate_id")
        version = event.get_metadata("version")

        # Event data access
        event_data = event.data  # Returns Mapping[str, object]
        user_id = event_data.get("user_id")

        # Extend event with processing metadata
        processed_event = event.with_metadata(
            processed_at=_BaseGenerators.generate_timestamp(),
            processor_version="1.2.3",
            correlation_id="req_789"
        )
    """

    @classmethod
    def create_event(
        cls,
        event_type: str,
        event_data: Mapping[str, object],
        *,
        aggregate_id: str | None = None,
        version: int | None = None,
    ) -> FlextResult[FlextEvent]:
        """Create event payload.

        Args:
            event_type: Type of event
            event_data: Event data
            aggregate_id: Optional aggregate ID
            version: Optional event version

        Returns:
            Result containing event payload

        """
        # Import logger directly for class methods

        logger = FlextLoggerFactory.get_logger(f"{cls.__module__}.{cls.__name__}")

        # Validate event_type using FlextValidation
        if not FlextValidators.is_non_empty_string(event_type):
            logger.error("Invalid event type - empty or not string")
            return FlextResult.fail("Event type cannot be empty")

        # Validate aggregate_id if provided
        if aggregate_id and not FlextValidators.is_non_empty_string(aggregate_id):
            logger.error("Invalid aggregate ID - empty or not string")
            return FlextResult.fail("Invalid aggregate ID")

        # Validate version if provided
        if version is not None and version < 0:
            logger.error("Invalid event version", version=version)
            return FlextResult.fail("Event version must be non-negative")

        metadata: dict[str, object] = {"event_type": event_type}
        if aggregate_id:
            metadata["aggregate_id"] = aggregate_id
        if version is not None:
            metadata["version"] = version

        logger.debug(
            "Creating event payload",
            event_type=event_type,
            aggregate_id=aggregate_id,
            version=version,
        )
        # Create FlextEvent instance directly for correct return type
        try:
            instance = cls(data=dict(event_data), metadata=metadata)
            return FlextResult.ok(instance)
        except (TypeError, ValueError, ValidationError) as e:
            return FlextResult.fail(f"Failed to create event: {e}")


# Export API
__all__ = [
    "FlextEvent",
    "FlextMessage",
    "FlextPayload",
]
