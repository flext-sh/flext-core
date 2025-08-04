"""FLEXT Core Payload - Configuration Layer Data Transport System.

Enterprise-grade type-safe payload containers for structured data transport with
comprehensive validation, metadata management, and serialization across the 32-project
FLEXT ecosystem. Foundation for messaging, events, and data pipeline communication.

Module Role in Architecture:
    Configuration Layer → Data Transport → Message Communication

    This module provides unified payload patterns for data transport throughout FLEXT:
    - FlextPayload[T] generic containers for type-safe data transport
    - FlextMessage specialized payloads for logging and notification systems
    - FlextEvent domain event payloads for event sourcing and CQRS patterns
    - Immutable data containers preventing accidental modification in pipelines

Payload Architecture Patterns:
    Generic Type Safety: FlextPayload[T] with compile-time type checking
    Immutable Transport: Pydantic frozen models preventing modification
    Metadata Enrichment: Flexible key-value metadata for transport context
    Factory Validation: Type-safe creation with comprehensive error handling

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Generic payloads, message/event specializations, validation
    🚧 Active Development: Event sourcing integration (Priority 1 - September 2025)
    📋 TODO Integration: Cross-service serialization for Go bridge (Priority 4)

Specialized Payload Types:
    FlextPayload[T]: Generic type-safe container with metadata support
    FlextMessage: String message payload with level classification and source tracking
    FlextEvent: Domain event payload with aggregate tracking and versioning
    Factory Methods: Validated creation with FlextResult error handling

Ecosystem Usage Patterns:
    # FLEXT Service Communication
    user_payload = FlextPayload.create(user_data, version="1.0", source="api")

    # Singer Tap/Target Messages
    message_result = FlextMessage.create_message(
        "Oracle extraction completed",
        level="info",
        source="flext-tap-oracle"
    )

    # Domain Events (DDD/Event Sourcing)
    event_result = FlextEvent.create_event(
        "UserRegistered",
        {"user_id": "123", "email": "user@example.com"},
        aggregate_id="user_123",
        version=1
    )

    # Go Service Integration
    payload_dict = payload.to_dict()  # JSON serialization for FlexCore bridge

Transport and Serialization Features:
    - Immutable payload objects ensuring data integrity in concurrent processing
    - Rich metadata support for correlation IDs, versioning, and debugging context
    - JSON serialization compatibility for cross-service communication
    - Type-safe generic containers preventing runtime type errors

Quality Standards:
    - All payload creation must use factory methods with validation
    - Payload objects must be immutable after creation
    - Metadata must support JSON serialization for cross-service transport
    - Generic type parameters must be preserved for compile-time safety

See Also:
    docs/TODO.md: Priority 1 - Event sourcing implementation
    mixins.py: Serializable, validatable, and loggable behavior patterns
    result.py: FlextResult pattern for consistent error handling

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from flext_core.exceptions import FlextAttributeError, FlextValidationError
from flext_core.flext_types import T, TAnyDict  # noqa: TC001
from flext_core.loggings import FlextLoggerFactory
from flext_core.mixins import (
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextValidatableMixin,
)
from flext_core.result import FlextResult
from flext_core.validation import FlextValidators


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
                "timestamp": time.time(),
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
            processed_at=time.time(),
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
        str_strip_whitespace=False,  # Preserve whitespace in extra fields
        extra="allow",  # Allow arbitrary extra fields
        json_schema_extra={
            "description": "Type-safe payload container",
            "examples": [
                {"data": {"id": 1}, "metadata": {"version": "1.0"}},
                {"data": "simple string", "metadata": {"type": "text"}},
            ],
        },
    )

    data: T | None = Field(default=None, description="Payload data")
    metadata: TAnyDict = Field(
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
        except (ValidationError, FlextValidationError) as e:
            logger.exception("Failed to create payload")
            return FlextResult.fail(f"Failed to create payload: {e}")

    def with_metadata(self, **additional: object) -> FlextPayload[T]:
        """Create new payload with additional metadata.

        Args:
            **additional: Metadata to add/update

        Returns:
            New payload with updated metadata

        """
        # Keys in **additional are always strings, so merge directly
        new_metadata = {**self.metadata, **additional}
        return FlextPayload(data=self.data, metadata=new_metadata)

    def enrich_metadata(self, additional: dict[str, object]) -> FlextPayload[T]:
        """Create new payload with enriched metadata from dictionary.

        Args:
            additional: Dictionary of metadata to add/update

        Returns:
            New payload with updated metadata

        """
        # Merge existing metadata with additional metadata
        new_metadata = {**self.metadata, **additional}
        return FlextPayload(data=self.data, metadata=new_metadata)

    @classmethod
    def from_dict(
        cls,
        data_dict: object,
    ) -> FlextResult[FlextPayload[object]]:
        """Create payload from dictionary.

        Args:
            data_dict: Dictionary containing data and metadata keys

        Returns:
            FlextResult containing new payload instance

        """
        # Validate input is actually a dictionary first
        if not isinstance(data_dict, dict):
            return FlextResult.fail(
                "Failed to create payload from dict: Input is not a dictionary",
            )

        try:
            payload_data = data_dict.get("data")
            payload_metadata = data_dict.get("metadata", {})
            if not isinstance(payload_metadata, dict):
                payload_metadata = {}
            # Cast to proper type for the generic class
            payload = cls(data=payload_data, metadata=payload_metadata)
            return FlextResult.ok(payload)
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            # Broad exception handling for API contract safety in payload creation
            return FlextResult.fail(f"Failed to create payload from dict: {e}")

    def has_data(self) -> bool:
        """Check if payload has non-None data.

        Returns:
            True if data is not None

        """
        return self.data is not None

    def get_data(self) -> FlextResult[T]:
        """Get payload data with type safety.

        Returns:
            FlextResult containing data or error if None

        """
        if self.data is None:
            return FlextResult.fail("Payload data is None")
        return FlextResult.ok(self.data)

    def get_data_or_default(self, default: T) -> T:
        """Get payload data or return default if None.

        Args:
            default: Default value to return if data is None

        Returns:
            Payload data or default value

        """
        return self.data if self.data is not None else default

    def transform_data(
        self,
        transformer: Callable[[T], object],
    ) -> FlextResult[FlextPayload[object]]:
        """Transform payload data using a function.

        Args:
            transformer: Function to transform the data

        Returns:
            FlextResult containing new payload with transformed data

        """
        if self.data is None:
            return FlextResult.fail("Cannot transform None data")

        try:
            transformed_data = transformer(self.data)
            new_payload = FlextPayload(data=transformed_data, metadata=self.metadata)
            return FlextResult.ok(new_payload)
        except (RuntimeError, ValueError, TypeError) as e:
            # Broad exception handling for transformer function safety
            return FlextResult.fail(f"Data transformation failed: {e}")

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

    def to_dict(self) -> TAnyDict:
        """Convert payload to dictionary representation.

        Returns:
            Dictionary representation of payload

        """
        return {
            "data": self.data,
            "metadata": self.metadata,
        }

    def to_dict_basic(self) -> TAnyDict:
        """Convert to basic dictionary representation."""
        result: TAnyDict = {}

        # Get all attributes that don't start with __
        for attr_name in dir(self):
            if not attr_name.startswith("__"):
                # Skip mixin attributes that might not be initialized yet
                if attr_name in {"_validation_errors", "_is_valid", "_logger"}:
                    continue

                # Skip Pydantic internal attributes that cause deprecation warnings
                if attr_name in {"model_computed_fields", "model_fields"}:
                    continue

                # Skip callable attributes
                if callable(getattr(self, attr_name)):
                    continue

                try:
                    value = getattr(self, attr_name)
                    serialized_value = self._serialize_value(value)
                    if serialized_value is not None:
                        result[attr_name] = serialized_value
                except (AttributeError, TypeError):
                    # Skip attributes that can't be accessed or serialized
                    continue

        return result

    def _serialize_value(self, value: object) -> object | None:
        """Serialize a single value for dict conversion."""
        # Simple types
        if isinstance(value, str | int | float | bool | type(None)):
            return value

        # Collections
        if isinstance(value, list | tuple):
            return self._serialize_collection(value)

        if isinstance(value, dict):
            return self._serialize_dict(value)

        # Objects with serialization method
        if hasattr(value, "to_dict_basic"):
            to_dict_method = value.to_dict_basic
            if callable(to_dict_method):
                result = to_dict_method()
                return result if isinstance(result, dict) else None

        return None

    def _serialize_collection(
        self,
        collection: list[object] | tuple[object, ...],
    ) -> list[object]:
        """Serialize list or tuple values."""
        serialized_list: list[object] = []
        for item in collection:
            if isinstance(item, str | int | float | bool | type(None)):
                serialized_list.append(item)
            elif hasattr(item, "to_dict_basic"):
                to_dict_method = item.to_dict_basic
                if callable(to_dict_method):
                    result = to_dict_method()
                    if isinstance(result, dict):
                        serialized_list.append(result)
        return serialized_list

    def _serialize_dict(self, dict_value: TAnyDict) -> TAnyDict:
        """Serialize dictionary values."""
        serialized_dict: TAnyDict = {}
        for k, v in dict_value.items():
            if isinstance(v, str | int | float | bool | type(None)):
                serialized_dict[str(k)] = v
        return serialized_dict

    def __repr__(self) -> str:
        """Return string representation."""
        data_repr = repr(self.data)
        max_repr_length = 50
        if len(data_repr) > max_repr_length:
            data_repr = f"{data_repr[:47]}..."
        meta_count = len(self.metadata)
        return f"FlextPayload(data={data_repr}, metadata_keys={meta_count})"

    def __getattr__(self, name: str) -> object:
        """Get attribute from extra fields.

        Args:
            name: Field name to get

        Returns:
            Field value

        Raises:
            AttributeError: If field doesn't exist

        """
        # Handle mixin attributes that need lazy initialization
        mixin_attrs: dict[str, tuple[type | object, object]] = {
            "_validation_errors": (list[str], []),
            "_is_valid": (bool | None, None),
            "_logger": (object, None),
        }

        if name in mixin_attrs:
            _attr_type, default_value = mixin_attrs[name]
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                if name == "_logger":
                    logger_name = (
                        f"{self.__class__.__module__}.{self.__class__.__name__}"
                    )
                    self._logger = FlextLoggerFactory.get_logger(logger_name)
                    return self._logger
                setattr(self, name, default_value)
                return default_value

        # Handle extra fields
        if (
            hasattr(self, "__pydantic_extra__")
            and self.__pydantic_extra__
            and name in self.__pydantic_extra__
        ):
            return self.__pydantic_extra__[name]

        error_msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        available_fields = (
            list(self.__pydantic_extra__.keys())
            if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__
            else []
        )
        raise FlextAttributeError(
            error_msg,
            attribute_context={
                "class_name": self.__class__.__name__,
                "attribute_name": name,
                "available_fields": available_fields,
            },
        )

    def __contains__(self, key: str) -> bool:
        """Check if key exists in extra fields.

        Args:
            key: Key to check

        Returns:
            True if key exists

        """
        return self.has(key)

    def __hash__(self) -> int:
        """Create hash from payload data and extra fields.

        Returns:
            Hash value based on data and extra fields

        """
        # Create hash from data field if it exists and is hashable
        data_hash = 0
        if self.data is not None:
            try:
                data_hash = hash(self.data)
            except TypeError:
                # If data is not hashable, use its string representation
                data_hash = hash(str(self.data))

        # Create hash from extra fields by converting to sorted tuple
        extra_hash = 0
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            # Sort items to ensure consistent hash for same content
            sorted_items = tuple(sorted(self.__pydantic_extra__.items()))
            try:
                extra_hash = hash(sorted_items)
            except TypeError:
                # If some values are not hashable, use string representation
                str_items = tuple((k, str(v)) for k, v in sorted_items)
                extra_hash = hash(str_items)

        # Create hash from metadata
        metadata_hash = 0
        if self.metadata:
            sorted_metadata = tuple(sorted(self.metadata.items()))
            try:
                metadata_hash = hash(sorted_metadata)
            except TypeError:
                str_metadata = tuple((k, str(v)) for k, v in sorted_metadata)
                metadata_hash = hash(str_metadata)

        # Combine all hashes
        return hash((data_hash, extra_hash, metadata_hash))

    def has(self, key: str) -> bool:
        """Check if field exists in extra fields.

        Args:
            key: Field name to check

        Returns:
            True if field exists

        """
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            return key in self.__pydantic_extra__
        return False

    def get(self, key: str, default: object | None = None) -> object | None:
        """Get field value from extra fields with default.

        Args:
            key: Field name to get
            default: Default value if key not found

        Returns:
            Field value or default

        """
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            return self.__pydantic_extra__.get(key, default)
        return default

    def keys(self) -> list[str]:
        """Get list of extra field names.

        Returns:
            List of field names

        """
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            return list(self.__pydantic_extra__.keys())
        return []

    def items(self) -> list[tuple[str, object]]:
        """Get list of (key, value) pairs from extra fields.

        Returns:
            List of (key, value) tuples

        """
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            return list(self.__pydantic_extra__.items())
        return []


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
            timestamp=time.time(),
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

        metadata: TAnyDict = {"level": level}
        if source:
            metadata["source"] = source

        logger.debug("Creating message payload", level=level, source=source)

        # Create FlextMessage instance directly
        try:
            instance = cls(data=message, metadata=metadata)
            return FlextResult.ok(instance)
        except (ValidationError, FlextValidationError) as e:
            return FlextResult.fail(f"Failed to create message: {e}")

    @property
    def level(self) -> str:
        """Get message level."""
        level = self.get_metadata("level", "info")
        return str(level) if level is not None else "info"

    @property
    def source(self) -> str | None:
        """Get message source."""
        source = self.get_metadata("source")
        return str(source) if source is not None else None

    @property
    def correlation_id(self) -> str | None:
        """Get message correlation ID."""
        corr_id = self.get_metadata("correlation_id")
        return str(corr_id) if corr_id is not None else None

    @property
    def text(self) -> str | None:
        """Get message text."""
        return self.data


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
            processed_at=time.time(),
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

        # Validate aggregate_id if provided (not None)
        if aggregate_id is not None and not FlextValidators.is_non_empty_string(
            aggregate_id,
        ):
            logger.error("Invalid aggregate ID - empty or not string")
            return FlextResult.fail("Invalid aggregate ID")

        # Validate version if provided
        if version is not None and version < 0:
            logger.error("Invalid event version", version=version)
            return FlextResult.fail("Event version must be non-negative")

        metadata: TAnyDict = {"event_type": event_type}
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
        except (ValidationError, FlextValidationError) as e:
            return FlextResult.fail(f"Failed to create event: {e}")

    @property
    def event_type(self) -> str | None:
        """Get event type."""
        event_type = self.get_metadata("event_type")
        return str(event_type) if event_type is not None else None

    @property
    def aggregate_id(self) -> str | None:
        """Get aggregate ID."""
        agg_id = self.get_metadata("aggregate_id")
        return str(agg_id) if agg_id is not None else None

    @property
    def aggregate_type(self) -> str | None:
        """Get aggregate type."""
        agg_type = self.get_metadata("aggregate_type")
        return str(agg_type) if agg_type is not None else None

    @property
    def version(self) -> int | None:
        """Get event version."""
        version = self.get_metadata("version")
        if version is None:
            return None
        try:
            return int(str(version))
        except (ValueError, TypeError):
            return None

    @property
    def correlation_id(self) -> str | None:
        """Get event correlation ID."""
        corr_id = self.get_metadata("correlation_id")
        return str(corr_id) if corr_id is not None else None


# Export API
__all__ = [
    "FlextEvent",
    "FlextMessage",
    "FlextPayload",
]
