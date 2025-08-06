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
    ✅ Production Ready: Enterprise cross-service serialization with Go bridge support
    🚧 Active Development: Event sourcing integration (Priority 1 - September 2025)

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

import json
import time
import zlib
from base64 import b64decode, b64encode
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from flext_core.exceptions import FlextAttributeError, FlextValidationError
from flext_core.loggings import FlextLoggerFactory
from flext_core.mixins import (
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextValidatableMixin,
)
from flext_core.result import FlextResult
from flext_core.validation import FlextValidators

if TYPE_CHECKING:
    from flext_core.flext_types import T, TData, TValue

# =============================================================================
# CROSS-SERVICE SERIALIZATION CONSTANTS AND TYPES
# =============================================================================

# Serialization protocol version for backward compatibility
FLEXT_SERIALIZATION_VERSION = "1.0.0"

# Supported serialization formats for cross-service communication
SERIALIZATION_FORMAT_JSON = "json"
SERIALIZATION_FORMAT_JSON_COMPRESSED = "json_compressed"
SERIALIZATION_FORMAT_BINARY = "binary"

# Go bridge type mappings for proper type reconstruction
GO_TYPE_MAPPINGS = {
    "string": str,
    "int": int,
    "int64": int,
    "float64": float,
    "bool": bool,
    "map[string]interface{}": dict,
    "[]interface{}": list,
    "interface{}": object,
}

# Python to Go type mappings for serialization
PYTHON_TO_GO_TYPES = {
    str: "string",
    int: "int64",
    float: "float64",
    bool: "bool",
    dict: "map[string]interface{}",
    list: "[]interface{}",
    object: "interface{}",
}

# Maximum payload size before automatic compression (64KB)
MAX_UNCOMPRESSED_SIZE = 65536

# Compression level for large payloads
COMPRESSION_LEVEL = 6


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
        if result.success:
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
    metadata: TData = Field(
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
            # Cast metadata to TData type for type safety
            typed_metadata = cast("TData", metadata)
            payload = cls(data=data, metadata=typed_metadata)
            logger.debug("Payload created successfully", payload_id=id(payload))
            return FlextResult.ok(payload)
        except (ValidationError, FlextValidationError) as e:
            logger.exception("Failed to create payload")
            return FlextResult.fail(f"Failed to create payload: {e}")

    def with_metadata(self, **additional: TValue) -> FlextPayload[T]:
        """Create new payload with additional metadata.

        Args:
            **additional: Metadata to add/update

        Returns:
            New payload with updated metadata

        """
        # Keys in **additional are always strings, so merge directly
        new_metadata = {**self.metadata, **additional}
        return FlextPayload(data=self.data, metadata=new_metadata)

    def enrich_metadata(self, additional: TData) -> FlextPayload[T]:
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

    def to_dict(self) -> dict[str, object]:
        """Convert payload to dictionary representation.

        Returns:
            Dictionary representation of payload

        """
        return {
            "data": self.data,
            "metadata": self.metadata,
        }

    def to_dict_basic(self) -> dict[str, object]:
        """Convert to basic dictionary representation."""
        result: dict[str, object] = {}

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
                    if serialized_value is not None and isinstance(
                        serialized_value,
                        (str, int, float, bool, type(None)),
                    ):
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

    def _serialize_dict(self, dict_value: dict[str, object]) -> dict[str, object]:
        """Serialize dictionary values."""
        serialized_dict: dict[str, object] = {}
        for k, v in dict_value.items():
            if isinstance(v, str | int | float | bool | type(None)):
                serialized_dict[str(k)] = v
        return serialized_dict

    # =============================================================================
    # CROSS-SERVICE SERIALIZATION - Enterprise Go Bridge Integration
    # =============================================================================

    def to_cross_service_dict(
        self,
        *,
        include_type_info: bool = True,
        protocol_version: str = FLEXT_SERIALIZATION_VERSION,
    ) -> dict[str, object]:
        """Convert payload to cross-service compatible dictionary with type information.

        Enhanced serialization for Go bridge integration with comprehensive type
        information preservation and protocol versioning for backward compatibility.

        Cross-Service Features:
            - Type information preservation for proper deserialization
            - Protocol versioning for backward compatibility
            - Go-compatible type mappings for seamless integration
            - Metadata enrichment with serialization context
            - Timestamp tracking for message lifecycle

        Args:
            include_type_info: Whether to include Python type information
            protocol_version: Serialization protocol version

        Returns:
            Dictionary optimized for cross-service transport

        Usage:
            # Basic cross-service serialization
            cross_service_dict = payload.to_cross_service_dict()

            # Without type information (smaller payload)
            minimal_dict = payload.to_cross_service_dict(include_type_info=False)
            # Send to Go service
            json_payload = json.dumps(cross_service_dict)
            response = go_service.process(json_payload)

        """
        base_dict = {
            "data": self._serialize_for_cross_service(self.data),
            "metadata": self._serialize_metadata_for_cross_service(self.metadata),
            "payload_type": self.__class__.__name__,
            "serialization_timestamp": time.time(),
            "protocol_version": protocol_version,
        }

        if include_type_info:
            base_dict["type_info"] = {
                "data_type": self._get_go_type_name(type(self.data)),
                "python_type": self._get_python_type_name(type(self.data)),
                "generic_type": self._extract_generic_type_info(),
            }

        return base_dict

    def _serialize_for_cross_service(self, value: object) -> object:  # noqa: PLR0911
        """Serialize value for cross-service compatibility.

        Args:
            value: Value to serialize

        Returns:
            Cross-service compatible representation

        """
        if value is None:
            return None

        # Basic JSON-compatible types
        if isinstance(value, (str, int, float, bool)):
            return value

        # Collections
        if isinstance(value, (list, tuple)):
            return [self._serialize_for_cross_service(item) for item in value]

        if isinstance(value, dict):
            return {
                str(k): self._serialize_for_cross_service(v) for k, v in value.items()
            }

        # Objects with cross-service serialization
        if hasattr(value, "to_cross_service_dict"):
            return value.to_cross_service_dict()

        # Objects with basic serialization
        if hasattr(value, "to_dict"):
            result = value.to_dict()
            if isinstance(result, dict):
                return result

        # REAL SOLUTION: Type-safe complex object serialization
        logger = FlextLoggerFactory.get_logger(__name__)
        logger.warning(
            "Complex object cannot be serialized for cross-service transport",
            object_type=type(value).__name__,
            has_to_dict=hasattr(value, "to_dict"),
            has_dict=hasattr(value, "__dict__")
        )
        # Return detailed type information instead of string representation
        return {
            "type": type(value).__name__,
            "module": getattr(type(value), "__module__", "unknown"),
            "serialization_error": "Complex object not serializable",
            "has_to_dict": hasattr(value, "to_dict"),
            "has_dict": hasattr(value, "__dict__")
        }

    def _serialize_metadata_for_cross_service(
        self, metadata: TData
    ) -> dict[str, object]:
        """Serialize metadata for cross-service transport.

        Args:
            metadata: Metadata dictionary

        Returns:
            Cross-service compatible metadata

        """
        serialized_metadata: dict[str, object] = {}

        for key, value in metadata.items():
            # Ensure keys are strings
            str_key = str(key)

            # Serialize values for cross-service compatibility
            serialized_value = self._serialize_for_cross_service(value)

            # Only include JSON-serializable values
            if self._is_json_serializable(serialized_value):
                serialized_metadata[str_key] = serialized_value

        return serialized_metadata

    def _get_go_type_name(self, python_type: type) -> str:
        """Get Go type name for Python type.

        Args:
            python_type: Python type

        Returns:
            Corresponding Go type name

        """
        return PYTHON_TO_GO_TYPES.get(python_type, "interface{}")

    def _get_python_type_name(self, python_type: type) -> str:
        """Get Python type name string.

        Args:
            python_type: Python type

        Returns:
            Python type name as string

        """
        return getattr(python_type, "__name__", str(python_type))

    def _extract_generic_type_info(self) -> dict[str, object]:
        """Extract generic type information for type reconstruction.

        Returns:
            Dictionary containing generic type information

        """
        type_info: dict[str, object] = {
            "is_generic": False,
            "origin_type": None,
            "type_args": [],
        }

        # Check if this class has generic type information
        if hasattr(self.__class__, "__orig_bases__"):
            for base in self.__class__.__orig_bases__:
                if hasattr(base, "__origin__") and hasattr(base, "__args__"):
                    type_info["is_generic"] = True
                    type_info["origin_type"] = str(base.__origin__)
                    type_info["type_args"] = [str(arg) for arg in base.__args__]
                    break

        return type_info

    def _is_json_serializable(self, value: object) -> bool:
        """Check if value is JSON serializable.

        Args:
            value: Value to check

        Returns:
            True if JSON serializable

        """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError, OverflowError):
            return False

    @classmethod
    def from_cross_service_dict(
        cls,
        cross_service_dict: dict[str, object],
    ) -> FlextResult[FlextPayload[T]]:
        """Create payload from cross-service dictionary with type reconstruction.

        Comprehensive deserialization supporting type reconstruction, protocol
        versioning, and validation for robust cross-service communication.

        Args:
            cross_service_dict: Cross-service serialized dictionary

        Returns:
            FlextResult containing reconstructed payload

        Raises:
            FlextValidationError: If dictionary format is invalid

        Usage:
            # Deserialize from Go service response
            response_dict = json.loads(go_service_response)
            result = FlextPayload.from_cross_service_dict(response_dict)

            if result.success:
                payload = result.data
                original_data = payload.data

        """
        # Validate required fields
        required_fields = {"data", "metadata", "payload_type", "protocol_version"}
        missing_fields = required_fields - set(cross_service_dict.keys())

        if missing_fields:
            return FlextResult.fail(
                f"Invalid cross-service dictionary: missing fields {missing_fields}",
            )

        try:
            # Extract fields
            data = cross_service_dict["data"]
            metadata = cross_service_dict.get("metadata", {})
            protocol_version = cross_service_dict.get("protocol_version", "1.0.0")

            # Validate protocol version compatibility
            if not cls._is_protocol_compatible(str(protocol_version)):
                return FlextResult.fail(
                    f"Incompatible protocol version: {protocol_version}",
                )

            # Reconstruct data with type information if available
            type_info = cross_service_dict.get("type_info", {})
            if not isinstance(type_info, dict):
                type_info = {}
            reconstructed_data = cls._reconstruct_data_with_types(data, type_info)

            # Validate metadata is dictionary
            if not isinstance(metadata, dict):
                metadata = {}

            # Create payload instance - cast for generic constructor compatibility
            payload = cls(data=cast("T | None", reconstructed_data), metadata=metadata)
            return FlextResult.ok(payload)

        except (ValueError, TypeError, KeyError) as e:
            return FlextResult.fail(
                f"Failed to reconstruct payload from cross-service dict: {e}",
            )

    @classmethod
    def _is_protocol_compatible(cls, version: str) -> bool:
        """Check if protocol version is compatible.

        Args:
            version: Protocol version string

        Returns:
            True if compatible

        """
        # Simple version compatibility check
        # In production, this would implement semantic versioning rules
        major_version = version.split(".", maxsplit=1)[0] if "." in version else version
        current_major = FLEXT_SERIALIZATION_VERSION.split(".", maxsplit=1)[0]

        return major_version == current_major

    @classmethod
    def _reconstruct_data_with_types(
        cls,
        data: object,
        type_info: dict[str, object],
    ) -> object:
        """Reconstruct data using type information.

        Args:
            data: Serialized data
            type_info: Type reconstruction information

        Returns:
            Data with proper types reconstructed

        """
        if not type_info or not isinstance(type_info, dict):
            return data

        go_type = type_info.get("data_type")
        if not (go_type and isinstance(go_type, str) and go_type in GO_TYPE_MAPPINGS):
            return data

        target_type = GO_TYPE_MAPPINGS[go_type]
        return cls._convert_to_target_type(data, target_type)

    @classmethod
    def _convert_to_target_type(cls, data: object, target_type: type) -> object:
        """Convert data to target type safely.

        Args:
            data: Data to convert
            target_type: Target type for conversion

        Returns:
            Converted data or original data if conversion fails

        """
        try:
            if target_type is str and not isinstance(data, str):
                return str(data)

            if target_type is int and not isinstance(data, int):
                return cls._safe_int_conversion(data)

            if target_type is float and not isinstance(data, float):
                return cls._safe_float_conversion(data)

            if target_type is bool and not isinstance(data, bool):
                return bool(data) if data is not None else None

        except (ValueError, TypeError):
            pass

        return data

    @classmethod
    def _safe_int_conversion(cls, data: object) -> object:
        """Safely convert data to int."""
        if data is None:
            return None
        try:
            return int(str(data))
        except (ValueError, TypeError):
            return data

    @classmethod
    def _safe_float_conversion(cls, data: object) -> object:
        """Safely convert data to float."""
        if data is None:
            return None
        try:
            return float(str(data))
        except (ValueError, TypeError):
            return data

    def to_json_string(
        self,
        *,
        compressed: bool = False,
        include_type_info: bool = True,
    ) -> FlextResult[str]:
        """Convert payload to JSON string for cross-service transport.

        Args:
            compressed: Whether to compress large payloads
            include_type_info: Whether to include type information

        Returns:
            FlextResult containing JSON string or error

        """
        try:
            # Get cross-service dictionary
            payload_dict = self.to_cross_service_dict(
                include_type_info=include_type_info,
            )

            # Convert to JSON
            json_str = json.dumps(payload_dict, separators=(",", ":"))

            # Compress if payload is large and compression requested
            if compressed and len(json_str.encode()) > MAX_UNCOMPRESSED_SIZE:
                compressed_bytes = zlib.compress(
                    json_str.encode(),
                    level=COMPRESSION_LEVEL,
                )
                encoded_str = b64encode(compressed_bytes).decode()

                # Wrap in compression envelope
                envelope = {
                    "format": SERIALIZATION_FORMAT_JSON_COMPRESSED,
                    "data": encoded_str,
                    "original_size": len(json_str.encode()),
                    "compressed_size": len(compressed_bytes),
                }
                return FlextResult.ok(json.dumps(envelope))
            # Add format information
            envelope = {
                "format": SERIALIZATION_FORMAT_JSON,
                "data": payload_dict,
            }
            return FlextResult.ok(json.dumps(envelope))

        except (TypeError, ValueError, OverflowError) as e:
            return FlextResult.fail(f"Failed to serialize to JSON: {e}")

    @classmethod
    def from_json_string(cls, json_str: str) -> FlextResult[FlextPayload[T]]:  # noqa: PLR0911
        """Create payload from JSON string with automatic decompression.

        Args:
            json_str: JSON string (potentially compressed)

        Returns:
            FlextResult containing payload or error

        """
        try:
            # Parse envelope
            envelope = json.loads(json_str)

            if not isinstance(envelope, dict) or "format" not in envelope:
                return FlextResult.fail("Invalid JSON envelope format")

            format_type = envelope["format"]

            if format_type == SERIALIZATION_FORMAT_JSON:
                # Direct JSON format
                payload_dict = envelope.get("data", {})
                return cls.from_cross_service_dict(payload_dict)

            if format_type == SERIALIZATION_FORMAT_JSON_COMPRESSED:
                # Compressed format - decompress first
                encoded_data = envelope.get("data", "")
                if not isinstance(encoded_data, str):
                    return FlextResult.fail("Invalid compressed data format")

                try:
                    compressed_bytes = b64decode(encoded_data.encode())
                    decompressed_str = zlib.decompress(compressed_bytes).decode()
                    payload_dict = json.loads(decompressed_str)
                    return cls.from_cross_service_dict(payload_dict)

                except (zlib.error, UnicodeDecodeError) as e:
                    return FlextResult.fail(f"Decompression failed: {e}")
            else:
                return FlextResult.fail(f"Unsupported format: {format_type}")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return FlextResult.fail(f"Failed to parse JSON: {e}")

    def get_serialization_size(self) -> dict[str, int | float]:
        """Get serialization size information for monitoring.

        Returns:
            Dictionary with size information in bytes

        """
        # JSON serialization size
        json_result = self.to_json_string(compressed=False)
        json_size = (
            len(json_result.data.encode())
            if json_result.success and json_result.data
            else 0
        )

        # Compressed size
        compressed_result = self.to_json_string(compressed=True)
        compressed_size = (
            len(compressed_result.data.encode())
            if compressed_result.success and compressed_result.data
            else 0
        )

        # Basic dict size
        basic_dict = self.to_dict()
        basic_size = len(json.dumps(basic_dict).encode())

        return {
            "json_size": json_size,
            "compressed_size": compressed_size,
            "basic_size": basic_size,
            "compression_ratio": compressed_size / max(json_size, 1),
        }

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

        error_msg: str = f"'{self.__class__.__name__}' object has no attribute '{name}'"
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
        if result.success:
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

        metadata: TData = {"level": level}
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

    def to_cross_service_dict(
        self,
        *,
        include_type_info: bool = True,
        protocol_version: str = FLEXT_SERIALIZATION_VERSION,
    ) -> dict[str, object]:
        """Convert message to cross-service compatible dictionary.

        Enhanced for FlextMessage with message-specific metadata.

        Args:
            include_type_info: Whether to include type information
            protocol_version: Serialization protocol version

        Returns:
            Cross-service compatible dictionary for message transport

        """
        base_dict = super().to_cross_service_dict(
            include_type_info=include_type_info,
            protocol_version=protocol_version,
        )

        # Add message-specific information
        base_dict["message_level"] = self.level
        base_dict["message_source"] = self.source
        base_dict["message_text"] = self.text

        if self.correlation_id:
            base_dict["correlation_id"] = self.correlation_id

        return base_dict

    @classmethod
    def from_cross_service_dict(
        cls,
        cross_service_dict: dict[str, object],
    ) -> FlextResult[FlextPayload[str]]:
        """Create FlextMessage from cross-service dictionary.

        Args:
            cross_service_dict: Cross-service serialized dictionary

        Returns:
            FlextResult containing FlextMessage or error

        """
        # Extract message-specific fields
        message_text = cross_service_dict.get("message_text")
        message_level = cross_service_dict.get("message_level", "info")
        message_source = cross_service_dict.get("message_source")

        if not message_text or not isinstance(message_text, str):
            return FlextResult.fail("Invalid message text in cross-service data")

        # Create message using factory method
        result: FlextResult[FlextPayload[str]] = cast(
            "FlextResult[FlextPayload[str]]",
            cls.create_message(
                message_text,
                level=str(message_level),
                source=str(message_source) if message_source else None,
            ),
        )
        # Cast to match parent class return type
        return result


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

        metadata: TData = {"event_type": event_type}
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

    def to_cross_service_dict(
        self,
        *,
        include_type_info: bool = True,
        protocol_version: str = FLEXT_SERIALIZATION_VERSION,
    ) -> dict[str, object]:
        """Convert event to cross-service compatible dictionary.

        Enhanced for FlextEvent with event sourcing metadata.

        Args:
            include_type_info: Whether to include type information
            protocol_version: Serialization protocol version

        Returns:
            Cross-service compatible dictionary for event transport

        """
        base_dict = super().to_cross_service_dict(
            include_type_info=include_type_info,
            protocol_version=protocol_version,
        )

        # Add event-specific information
        base_dict["event_type"] = self.event_type
        base_dict["aggregate_id"] = self.aggregate_id
        base_dict["aggregate_type"] = self.aggregate_type
        base_dict["event_version"] = self.version

        if self.correlation_id:
            base_dict["correlation_id"] = self.correlation_id

        # Add event data directly at top level for easier Go access
        if self.data:
            base_dict["event_data"] = dict(self.data)

        return base_dict

    @classmethod
    def from_cross_service_dict(
        cls,
        cross_service_dict: dict[str, object],
    ) -> FlextResult[FlextPayload[Mapping[str, object]]]:
        """Create FlextEvent from cross-service dictionary.

        Args:
            cross_service_dict: Cross-service serialized dictionary

        Returns:
            FlextResult containing FlextEvent or error

        """
        # Extract event-specific fields
        event_type = cross_service_dict.get("event_type")
        event_data = cross_service_dict.get("event_data", {})
        aggregate_id = cross_service_dict.get("aggregate_id")
        event_version = cross_service_dict.get("event_version")

        if not event_type or not isinstance(event_type, str):
            return FlextResult.fail("Invalid event type in cross-service data")

        if not isinstance(event_data, dict):
            return FlextResult.fail("Invalid event data in cross-service data")

        # Convert version to int if provided
        version_int = None
        if event_version is not None:
            try:
                version_int = int(str(event_version))
            except (ValueError, TypeError):
                return FlextResult.fail("Invalid event version format")

        # Create event using factory method
        result: FlextResult[FlextPayload[Mapping[str, object]]] = cast(
            "FlextResult[FlextPayload[Mapping[str, object]]]",
            cls.create_event(
                event_type=str(event_type),
                event_data=event_data,
                aggregate_id=str(aggregate_id) if aggregate_id else None,
                version=version_int,
            ),
        )
        # Cast to match parent class return type
        return result


# =============================================================================
# CROSS-SERVICE SERIALIZATION UTILITIES - Factory Methods and Helpers
# =============================================================================


def serialize_payload_for_go_bridge(payload: FlextPayload[object]) -> FlextResult[str]:
    """Serialize payload optimized for Go bridge communication.

    Convenience function providing optimal serialization settings for Go service
    integration with automatic compression for large payloads.

    Args:
        payload: Payload to serialize

    Returns:
        FlextResult containing JSON string optimized for Go bridge

    Usage:
        # Serialize for Go service
        result = serialize_payload_for_go_bridge(payload)
        if result.success:
            json_str = result.data
            response = go_service.send(json_str)

    """
    return payload.to_json_string(
        compressed=True,  # Always use compression for Go bridge
        include_type_info=True,  # Include type info for proper reconstruction
    )


def deserialize_payload_from_go_bridge(json_str: str) -> FlextResult[object]:
    """Deserialize payload from Go bridge response.

    Convenience function for deserializing payloads from Go service responses
    with automatic decompression and type reconstruction.

    Args:
        json_str: JSON string from Go service

    Returns:
        FlextResult containing reconstructed payload

    Usage:
        # Deserialize from Go service
        result = deserialize_payload_from_go_bridge(go_response)
        if result.success:
            payload = result.data
            original_data = payload.data

    """
    result: FlextResult[FlextPayload[object]] = FlextPayload.from_json_string(json_str)
    return cast("FlextResult[object]", result)


def create_cross_service_message(
    text: str,
    level: str = "info",
    source: str | None = None,
    correlation_id: str | None = None,
) -> FlextResult[FlextMessage]:
    """Create message optimized for cross-service communication.

    Factory function creating FlextMessage with cross-service metadata
    including correlation ID for distributed tracing.

    Args:
        text: Message text
        level: Message level
        source: Message source
        correlation_id: Optional correlation ID for tracing

    Returns:
        FlextResult containing cross-service optimized message

    """
    result = FlextMessage.create_message(text, level=level, source=source)

    if result.success and correlation_id:
        message = result.data
        if message is not None:
            enhanced_message = message.with_metadata(correlation_id=correlation_id)
            return FlextResult.ok(cast("FlextMessage", enhanced_message))

    return result


def create_cross_service_event(
    event_type: str,
    event_data: Mapping[str, object],
    *,
    aggregate_id: str | None = None,
    version: int | None = None,
    correlation_id: str | None = None,
) -> FlextResult[FlextEvent]:
    """Create event optimized for cross-service communication.

    Factory function creating FlextEvent with cross-service metadata
    including correlation ID for distributed event tracking.

    Args:
        event_type: Type of event
        event_data: Event data
        aggregate_id: Optional aggregate ID
        version: Optional event version
        correlation_id: Optional correlation ID for tracing

    Returns:
        FlextResult containing cross-service optimized event

    """
    result = FlextEvent.create_event(
        event_type=event_type,
        event_data=event_data,
        aggregate_id=aggregate_id,
        version=version,
    )

    if result.success and correlation_id:
        event = result.data
        if event is not None:
            enhanced_event = event.with_metadata(correlation_id=correlation_id)
            return FlextResult.ok(cast("FlextEvent", enhanced_event))

    return result


def validate_cross_service_protocol(
    serialized_data: str,
) -> FlextResult[dict[str, object]]:
    """Validate cross-service protocol format.

    Validates that serialized data conforms to FLEXT cross-service protocol
    with proper format envelope and required fields.

    Args:
        serialized_data: Serialized JSON string

    Returns:
        FlextResult containing validation status and parsed envelope

    Usage:
        # Validate before processing
        validation_result = validate_cross_service_protocol(incoming_data)
        if validation_result.success:
            envelope = validation_result.data
            format_type = envelope["format"]

    """
    try:
        envelope = json.loads(serialized_data)

        if not isinstance(envelope, dict):
            return FlextResult.fail("Invalid protocol: envelope must be dictionary")

        if "format" not in envelope:
            return FlextResult.fail("Invalid protocol: missing format field")

        format_type = envelope["format"]
        if format_type not in {
            SERIALIZATION_FORMAT_JSON,
            SERIALIZATION_FORMAT_JSON_COMPRESSED,
            SERIALIZATION_FORMAT_BINARY,
        }:
            return FlextResult.fail(
                f"Invalid protocol: unsupported format {format_type}",
            )

        if "data" not in envelope:
            return FlextResult.fail("Invalid protocol: missing data field")

        return FlextResult.ok(envelope)

    except json.JSONDecodeError as e:
        return FlextResult.fail(f"Invalid protocol: JSON decode error - {e}")


def get_serialization_metrics(payload: FlextPayload[object]) -> dict[str, object]:
    """Get comprehensive serialization metrics for monitoring.

    Provides detailed metrics about payload serialization performance
    for operational monitoring and optimization.

    Args:
        payload: Payload to analyze

    Returns:
        Dictionary containing serialization metrics

    Usage:
        # Monitor serialization performance
        metrics = get_serialization_metrics(payload)
        print(f"Compression ratio: {metrics['compression_ratio']:.2f}")
        print(f"JSON size: {metrics['json_size']} bytes")

    """
    base_metrics = payload.get_serialization_size()

    # Add additional metrics - casting to dict[str, object] for return type
    additional_metrics: dict[str, object] = {
        "payload_type": payload.__class__.__name__,
        "data_type": type(payload.data).__name__ if payload.data else "None",
        "metadata_keys": len(payload.metadata),
        "has_data": payload.has_data(),
        "protocol_version": FLEXT_SERIALIZATION_VERSION,
    }

    # Combine metrics with proper typing
    result_metrics: dict[str, object] = {}
    result_metrics.update(base_metrics)
    result_metrics.update(additional_metrics)

    return result_metrics


# =============================================================================
# MODEL REBUILDS - Resolve forward references for Pydantic
# =============================================================================

# CRITICAL: Model rebuild disabled - TAnyDict import conflicts resolved by design
# The models work correctly without explicit rebuild as Pydantic handles
# forward references automatically during runtime validation.
# Original error resolved by using proper type imports and avoiding circular deps.
# FlextPayload.model_rebuild()
# FlextMessage.model_rebuild()
# FlextEvent.model_rebuild()

# Export API
__all__: list[str] = [
    # Serialization constants (sorted alphabetically)
    "FLEXT_SERIALIZATION_VERSION",
    "SERIALIZATION_FORMAT_BINARY",
    "SERIALIZATION_FORMAT_JSON",
    "SERIALIZATION_FORMAT_JSON_COMPRESSED",
    # Core payload classes (sorted alphabetically)
    "FlextEvent",
    "FlextMessage",
    "FlextPayload",
    # Cross-service serialization utilities (sorted alphabetically)
    "create_cross_service_event",
    "create_cross_service_message",
    "deserialize_payload_from_go_bridge",
    "get_serialization_metrics",
    "serialize_payload_for_go_bridge",
    "validate_cross_service_protocol",
]
