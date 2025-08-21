"""Type-safe payload containers for data transport."""

from __future__ import annotations

import json
import time
import zlib
from base64 import b64decode, b64encode
from collections.abc import Callable, Mapping
from typing import TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    model_serializer,
)

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextAttributeError, FlextValidationError
from flext_core.loggings import FlextLoggerFactory, flext_get_logger
from flext_core.mixins import (
    FlextLoggableMixin,
    FlextSerializableMixin,
)
from flext_core.result import FlextResult
from flext_core.typings import TValue
from flext_core.validation import FlextValidators

# =============================================================================
# CROSS-SERVICE SERIALIZATION CONSTANTS AND TYPES
# =============================================================================

# Serialization protocol version for cross-service communication
# Constants moved to constants.py following SOLID Single Responsibility Principle

FLEXT_SERIALIZATION_VERSION = FlextConstants.Observability.FLEXT_SERIALIZATION_VERSION

# Supported serialization formats for cross-service communication
SERIALIZATION_FORMAT_JSON = FlextConstants.Observability.SERIALIZATION_FORMAT_JSON
SERIALIZATION_FORMAT_JSON_COMPRESSED = (
    FlextConstants.Observability.SERIALIZATION_FORMAT_JSON_COMPRESSED
)
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


T = TypeVar("T")


class FlextPayload[T](
    BaseModel,
    FlextSerializableMixin,
    FlextLoggableMixin,
):
    """Generic type-safe payload container for structured data transport and validation.

    Comprehensive payload implementation providing immutable data transport with
    automatic validation, serialization, and metadata management. Combines Pydantic
    validation with mixin functionality for complete data integrity.
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

    data: T | None = Field(
        default=None,
        description="Payload data",
        serialization_alias="payloadData",
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Optional metadata",
        serialization_alias="payloadMetadata",
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
        logger = flext_get_logger(__name__)

        logger.debug(
            "Creating payload",
            data_type=type(data).__name__,
            metadata_keys=list(metadata.keys()),
        )

        try:
            # Metadata is already dict[str, object] from **kwargs
            payload = cls(data=data, metadata=metadata)
            logger.debug("Payload created successfully", payload_id=id(payload))
            return FlextResult[FlextPayload[T]].ok(payload)
        except (ValidationError, FlextValidationError) as e:
            logger.exception("Failed to create payload")
            return FlextResult[FlextPayload[T]].fail(f"Failed to create payload: {e}")

    def with_metadata(self, **additional: TValue) -> FlextPayload[T]:
        """Create a new payload with additional metadata.

        Args:
            **additional: Metadata to add/update

        Returns:
            New payload with updated metadata

        """
        # Keys in **additional are always strings, so merge directly
        new_metadata = {**self.metadata, **additional}
        return FlextPayload(data=self.data, metadata=new_metadata)

    def enrich_metadata(self, additional: dict[str, object]) -> FlextPayload[T]:
        """Create a new payload with enriched metadata from dictionary.

        Args:
            additional: Dictionary of metadata to add/update

        Returns:
            New payload with updated metadata

        """
        # Merge existing metadata with additional metadata
        new_metadata = {**self.metadata, **additional}
        return FlextPayload(data=self.data, metadata=new_metadata)

    @classmethod
    def create_from_dict(
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
        match data_dict:
            case dict():
                # Type narrowing: data_dict is now known to be a dict
                validated_dict = cast("dict[str, object]", data_dict)
            case _:
                return FlextResult[FlextPayload[object]].fail(
                    "Failed to create payload from dict: Input is not a dictionary",
                )

        try:
            payload_data = validated_dict.get("data")
            payload_metadata_raw = validated_dict.get("metadata", {})
            if isinstance(payload_metadata_raw, dict):
                payload_metadata = cast("dict[str, object]", payload_metadata_raw)
            else:
                payload_metadata = {}
            # Cast to proper type for the generic class
            payload = cls(
                data=cast("T | None", payload_data), metadata=payload_metadata
            )
            return FlextResult[FlextPayload[object]].ok(
                cast("FlextPayload[object]", payload)
            )
        except (RuntimeError, ValueError, TypeError, AttributeError) as e2:
            # Broad exception handling for API contract safety in payload creation
            return FlextResult[FlextPayload[object]].fail(
                f"Failed to create payload from dict: {e2}"
            )

    @classmethod
    def from_dict(
        cls,
        data_dict: dict[str, object] | Mapping[str, object] | object,
    ) -> FlextResult[FlextPayload[object]]:
        """Convenience wrapper ; returns FlextResult.

        Accepts dict-like inputs to satisfy broader call sites and delegates
        to ``create_from_dict`` after minimal normalization.
        """
        if isinstance(data_dict, Mapping):
            data_obj = dict(cast("Mapping[str, object]", data_dict))
        else:
            data_obj = cast("dict[str, object]", data_dict)
        return cls.create_from_dict(data_obj)

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
            return FlextResult[T].fail("Payload data is None")
        return FlextResult[T].ok(self.data)

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
            return FlextResult[FlextPayload[object]].fail("Cannot transform None data")

        try:
            transformed_data = transformer(self.data)
            new_payload = FlextPayload(data=transformed_data, metadata=self.metadata)
            return FlextResult[FlextPayload[object]].ok(new_payload)
        except (RuntimeError, ValueError, TypeError) as e3:
            # Broad exception handling for transformer function safety
            return FlextResult[FlextPayload[object]].fail(
                f"Data transformation failed: {e3}"
            )

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
        """Check if a metadata key exists.

        Args:
            key: Metadata key to check

        Returns:
            True if key exists

        """
        return key in self.metadata

    @field_serializer("data", when_used="json")
    def serialize_data_for_json(self, value: T | None) -> object:
        """Custom field serializer for data in JSON mode."""
        if value is None:
            return None
        #  JSON serialization with type information
        return {
            "value": value,
            "type": type(value).__name__,
            "serialized_at": time.time(),
        }

    @field_serializer("metadata", when_used="always")
    def serialize_metadata(
        self,
        value: dict[str, object],
    ) -> dict[str, object]:
        """Metadata serialization with payload context."""
        metadata = dict(value)
        metadata["_payload_type"] = self.__class__.__name__
        metadata["_serialization_timestamp"] = time.time()
        return metadata

    @model_serializer(mode="wrap", when_used="json")
    def serialize_payload_for_api(
        self,
        serializer: Callable[[FlextPayload[T]], dict[str, object] | object],
        info: object,
    ) -> dict[str, object] | object:
        """Model serializer for API output with comprehensive payload metadata."""
        _ = info  # Acknowledge parameter for future use
        data = serializer(self)
        if isinstance(data, dict):
            # Add comprehensive payload API metadata
            data["_payload"] = {
                "type": self.__class__.__name__,
                "has_data": self.has_data(),
                "metadata_keys": list(self.metadata.keys()),
                "serialization_format": "json",
                "api_version": "v2",
                "cross_service_ready": True,
            }
        return cast("dict[str, object]", data)

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
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        # Collections
        if isinstance(value, (list, tuple)):
            return self._serialize_collection(
                cast("list[object] | tuple[object, ...]", value)
            )
        if isinstance(value, dict):
            return self._serialize_dict(cast("dict[str, object]", value))
        # Objects with serialization method
        to_dict_method = getattr(value, "to_dict_basic", None)
        if callable(to_dict_method):
            result = to_dict_method()
            if isinstance(result, dict):
                return cast("dict[str, object]", result)
            return None
        return None

    @staticmethod
    def _serialize_collection(
        collection: list[object] | tuple[object, ...],
    ) -> list[object]:
        """Serialize list or tuple values."""
        serialized_list: list[object] = []
        for item in collection:
            if isinstance(item, (str, int, float, bool)) or item is None:
                serialized_list.append(item)
            else:
                to_dict_method = getattr(item, "to_dict_basic", None)
                if callable(to_dict_method):
                    result = to_dict_method()
                    if isinstance(result, dict):
                        serialized_list.append(cast("dict[str, object]", result))
                    else:
                        pass  # Skip non-dict results
        return serialized_list

    @staticmethod
    def _serialize_dict(dict_value: dict[str, object]) -> dict[str, object]:
        """Serialize dictionary values."""
        serialized_dict: dict[str, object] = {}
        for k, v in dict_value.items():
            match v:
                case str() | int() | float() | bool() | None:
                    serialized_dict[str(k)] = v
                case _:
                    # Handle other types with to_dict_basic if available
                    pass
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
        """Convert payload to cross-service dictionary with type information.

         serialization for Go bridge integration with comprehensive type
        information preservation and protocol versioning for cross-service communication.

        Args:
            include_type_info: Whether to include Python type information
            protocol_version: Serialization protocol version

        Returns:
            Dictionary optimized for cross-service transport.

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

    def _serialize_for_cross_service(self, value: object) -> object:
        """Serialize value for cross-service communication.

        Args:
            value: Value to serialize

        Returns:
            Cross-service representation

        """
        logger = flext_get_logger(__name__)

        # Delegate to helper method to reduce complexity
        result = self._get_serializable_value(value)
        if result is not None or value is None:
            return result

        # REAL SOLUTION: Type-safe complex object serialization
        logger.warning(
            "Complex object cannot be serialized for cross-service transport",
            object_type=type(value).__name__,
            has_to_dict=hasattr(value, "to_dict"),
            has_dict=hasattr(value, "__dict__"),
        )
        # Return detailed type information instead of string representation
        return {
            "type": type(value).__name__,
            "module": getattr(type(value), "__module__", "unknown"),
            "serialization_error": "Complex object not serializable",
            "has_to_dict": hasattr(value, "to_dict"),
            "has_dict": hasattr(value, "__dict__"),
        }

    def _get_serializable_value(self, value: object) -> object | None:
        """Get serializable value."""
        # Handle None and basic types
        basic_result = self._handle_basic_types(value)
        if basic_result is not None or value is None:
            return basic_result

        # Handle collections
        collection_result = self._handle_collections(value)
        if collection_result is not None:
            return collection_result

        # Handle objects with serialization methods
        return self._handle_serializable_objects(value)

    @staticmethod
    def _handle_basic_types(value: object) -> object | None:
        """Handle basic JSON-serializable types."""
        match value:
            case None:
                return None
            case str() | int() | float() | bool() as basic_value:
                return basic_value
            case _:
                return None

    def _handle_collections(self, value: object) -> object | None:
        """Handle collection types."""
        if isinstance(value, (list, tuple)):
            typed_collection = cast("list[object] | tuple[object, ...]", value)
            return [
                self._serialize_for_cross_service(item) for item in typed_collection
            ]
        if isinstance(value, dict):
            typed_dict = cast("dict[str, object]", value)
            return {
                str(k): self._serialize_for_cross_service(v)
                for k, v in typed_dict.items()
            }
        return None

    @staticmethod
    def _handle_serializable_objects(value: object) -> object | None:
        """Handle objects with serialization methods."""
        # Objects with cross-service serialization
        cross_service_method = getattr(value, "to_cross_service_dict", None)
        if callable(cross_service_method):
            result = cross_service_method()
            return result if result is not None else None

        # Objects with basic serialization
        to_dict_method = getattr(value, "to_dict", None)
        if callable(to_dict_method):
            result = to_dict_method()
            if isinstance(result, dict):
                return cast("dict[str, object]", result)

        # Return None to indicate no serialization found
        return None

    def _serialize_metadata_for_cross_service(
        self,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        """Serialize metadata for cross-service transport.

        Args:
            metadata: Metadata dictionary

        Returns:
            Cross-service metadata

        """
        serialized_metadata: dict[str, object] = {}

        for key, value in metadata.items():
            # Ensure keys are strings
            str_key = str(key)

            # Serialize values for cross-service communication
            serialized_value = self._serialize_for_cross_service(value)

            # Only include JSON-serializable values
            if self._is_json_serializable(serialized_value):
                serialized_metadata[str_key] = serialized_value

        return serialized_metadata

    @staticmethod
    def _get_go_type_name(python_type: type) -> str:
        """Get Go type name for Python type.

        Args:
            python_type: Python type

        Returns:
            Corresponding Go type name

        """
        return PYTHON_TO_GO_TYPES.get(python_type, "interface{}")

    @staticmethod
    def _get_python_type_name(python_type: type) -> str:
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
        orig_bases = getattr(self.__class__, "__orig_bases__", None)
        if orig_bases is not None:
            for base in orig_bases:
                origin = getattr(base, "__origin__", None)
                args = getattr(base, "__args__", None)
                if origin is not None and args is not None:
                    type_info["is_generic"] = True
                    type_info["origin_type"] = str(base.__origin__)
                    type_info["type_args"] = [str(arg) for arg in base.__args__]
                    break

        return type_info

    @staticmethod
    def _is_json_serializable(value: object) -> bool:
        """Check if the value is JSON serializable.

        Args:
            value: Value to check

        Returns:
            True if JSON serializable

        """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError, OverflowError) as e4:
            logger = flext_get_logger(__name__)
            logger.warning(
                f"Value not JSON serializable: {type(value).__name__} - {e4}",
            )
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
            FlextResult containing reconstructed payload.

        Raises:
            FlextValidationError: If a dictionary format is invalid.

        """
        # Validate required fields
        required_fields = {"data", "metadata", "payload_type", "protocol_version"}
        missing_fields = required_fields - set(cross_service_dict.keys())

        if missing_fields:
            return FlextResult[FlextPayload[T]].fail(
                f"Invalid cross-service dictionary: missing fields {missing_fields}",
            )

        try:
            # Extract fields
            data = cross_service_dict["data"]
            metadata = cross_service_dict.get("metadata", {})
            protocol_version = cross_service_dict.get("protocol_version", "1.0.0")

            # Validate protocol version support
            if not cls._is_protocol_supported(str(protocol_version)):
                return FlextResult[FlextPayload[T]].fail(
                    f"Unsupported protocol version: {protocol_version}",
                )

            # Reconstruct data with type information if available
            type_info_raw = cross_service_dict.get("type_info", {})
            if isinstance(type_info_raw, dict):
                type_info = cast("dict[str, object]", type_info_raw)
            else:
                type_info = {}
            reconstructed_data = cls._reconstruct_data_with_types(data, type_info)

            # Validate metadata is dictionary
            if not isinstance(metadata, dict):
                metadata = {}

            # Create payload instance - cast for generic constructor
            payload = cls(
                data=cast("T | None", reconstructed_data),
                metadata=cast("dict[str, object]", metadata),
            )
            return FlextResult[FlextPayload[T]].ok(payload)

        except (ValueError, TypeError, KeyError) as e5:
            return FlextResult[FlextPayload[T]].fail(
                f"Failed to reconstruct payload from cross-service dict: {e5}",
            )

    @classmethod
    def _is_protocol_supported(cls, version: str) -> bool:
        """Check if a protocol version is supported.

        Args:
            version: Protocol version string

        Returns:
            True if supported

        """
        # Simple version support check
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
        if not type_info:
            return data

        go_type = type_info.get("data_type")
        if isinstance(go_type, str) and go_type in GO_TYPE_MAPPINGS:
            target_type = GO_TYPE_MAPPINGS[go_type]
            return cls._convert_to_target_type(data, target_type)
        return data

    @classmethod
    def _convert_to_target_type(cls, data: object, target_type: type) -> object:
        """Convert data to a target type safely.

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

        except (ValueError, TypeError) as e6:
            logger = flext_get_logger(__name__)
            logger.warning(
                f"Type conversion failed for {type(data).__name__} "
                f"to {target_type.__name__}: {e6}",
            )

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
            # Get a cross-service dictionary
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
                return FlextResult[str].ok(json.dumps(envelope))
            # Add format information
            envelope = {
                "format": SERIALIZATION_FORMAT_JSON,
                "data": payload_dict,
            }
            return FlextResult[str].ok(json.dumps(envelope))

        except (TypeError, ValueError, OverflowError) as e7:
            return FlextResult[str].fail(f"Failed to serialize to JSON: {e7}")

    @classmethod
    def from_json_string(cls, json_str: str) -> FlextResult[FlextPayload[T]]:
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
                return FlextResult[FlextPayload[T]].fail("Invalid JSON envelope format")

            # Delegate to helper method to reduce complexity
            return cls._process_json_envelope(cast("dict[str, object]", envelope))

        except (json.JSONDecodeError, KeyError, TypeError) as e8:
            return FlextResult[FlextPayload[T]].fail(f"Failed to parse JSON: {e8}")

    @classmethod
    def _process_json_envelope(
        cls,
        envelope: dict[str, object],
    ) -> FlextResult[FlextPayload[T]]:
        """Process JSON envelope."""
        format_type = envelope["format"]

        if format_type == SERIALIZATION_FORMAT_JSON:
            # Direct JSON format
            payload_data = envelope.get("data", {})
            if isinstance(payload_data, dict):
                return cls.from_cross_service_dict(
                    cast("dict[str, object]", payload_data)
                )
            msg = f"Expected dict for JSON format, got {type(payload_data)}"
            raise ValueError(msg)

        if format_type == SERIALIZATION_FORMAT_JSON_COMPRESSED:
            # Compressed format - decompress first
            return cls._process_compressed_json(envelope)

        return FlextResult[FlextPayload[T]].fail(f"Unsupported format: {format_type}")

    @classmethod
    def _process_compressed_json(
        cls,
        envelope: dict[str, object],
    ) -> FlextResult[FlextPayload[T]]:
        """Process compressed JSON."""
        encoded_data = envelope.get("data", "")
        if not isinstance(encoded_data, str):
            return FlextResult[FlextPayload[T]].fail("Invalid compressed data format")

        try:
            compressed_bytes = b64decode(encoded_data.encode())
            decompressed_str = zlib.decompress(compressed_bytes).decode()
            payload_dict = json.loads(decompressed_str)
            return cls.from_cross_service_dict(payload_dict)

        except (zlib.error, UnicodeDecodeError) as e9:
            return FlextResult[FlextPayload[T]].fail(f"Decompression failed: {e9}")

    def get_serialization_size(self) -> dict[str, int | float]:
        """Get serialization size information for monitoring.

        Returns:
            Dictionary with size information in bytes

        """
        # JSON serialization size
        json_result = self.to_json_string(compressed=False)
        json_size = (
            len(json_result.value.encode())
            if json_result.is_success and json_result.value
            else 0
        )

        # Compressed size
        compressed_result = self.to_json_string(compressed=True)
        compressed_size = (
            len(compressed_result.value.encode())
            if compressed_result.is_success and compressed_result.value
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
            # Sort items to ensure consistent hash for the same content
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
        """Get field value from extra fields with default."""
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            return cast("object | None", self.__pydantic_extra__.get(key, default))
        return default

    def keys(self) -> list[str]:
        """Get a list of extra field names.

        Returns:
            List of field names

        """
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            return list(self.__pydantic_extra__.keys())
        return []

    def items(self) -> list[tuple[str, object]]:
        """Get a list of (key, value) pairs from extra fields.

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
        logger = flext_get_logger(__name__)

        # Validate message using FlextValidation
        if not FlextValidators.is_non_empty_string(message):
            logger.error("Invalid message - empty or not string")
            return FlextResult[FlextMessage].fail("Message cannot be empty")

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
            return FlextResult[FlextMessage].ok(instance)
        except (ValidationError, FlextValidationError) as e:
            return FlextResult[FlextMessage].fail(f"Failed to create message: {e}")

    @property
    def level(self) -> str:
        """Get message level."""
        level = self.get_metadata("level", "info")
        return str(level) if level is not None else "info"

    @property
    def source(self) -> str | None:
        """Get a message source."""
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
        """Convert a message to cross-service dictionary.

         for FlextMessage with message-specific metadata.

        Args:
            include_type_info: Whether to include type information
            protocol_version: Serialization protocol version

        Returns:
            Cross-service dictionary for message transport

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
            return FlextResult[FlextPayload[str]].fail(
                "Invalid message text in cross-service data",
            )

        # Create a message using factory method
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
    """Domain event payload with aggregate tracking and versioning.

    Specialized payload for DDD events with event sourcing support,
    aggregate correlation, and version tracking for event ordering.
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
        logger = flext_get_logger(__name__)

        # Validate event_type using FlextValidation
        if not FlextValidators.is_non_empty_string(event_type):
            logger.error("Invalid event type - empty or not string")
            return FlextResult[FlextEvent].fail("Event type cannot be empty")

        # Validate aggregate_id if provided (not None)
        if aggregate_id is not None and not FlextValidators.is_non_empty_string(
            aggregate_id,
        ):
            logger.error("Invalid aggregate ID - empty or not string")
            return FlextResult[FlextEvent].fail("Invalid aggregate ID")

        # Validate version if provided
        if version is not None and version < 0:
            logger.error("Invalid event version", version=version)
            return FlextResult[FlextEvent].fail("Event version must be non-negative")

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
            return FlextResult[FlextEvent].ok(instance)
        except (ValidationError, FlextValidationError) as e12:
            # Create failed result directly with correct type
            return FlextResult[FlextEvent](error=f"Failed to create event: {e12}")

    @property
    def event_type(self) -> str | None:
        """Get an event type."""
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
        """Get an event version."""
        version = self.get_metadata("version")
        if version is None:
            return None
        try:
            return int(str(version))
        except (ValueError, TypeError) as e:
            logger = flext_get_logger(__name__)
            logger.warning(f"Failed to convert version to int: {version} - {e}")
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
        """Convert event to cross-service dictionary.

         for FlextEvent with event sourcing metadata.

        Args:
            include_type_info: Whether to include type information
            protocol_version: Serialization protocol version

        Returns:
            Cross-service dictionary for event transport

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
            return FlextResult[FlextPayload[Mapping[str, object]]].fail(
                "Invalid event type in cross-service data",
            )

        if not isinstance(event_data, dict):
            return FlextResult[FlextPayload[Mapping[str, object]]].fail(
                "Invalid event data in cross-service data",
            )

        # Convert a version to int if provided
        version_int = None
        if event_version is not None:
            try:
                version_int = int(str(event_version))
            except (ValueError, TypeError):
                return FlextResult[FlextPayload[Mapping[str, object]]].fail(
                    "Invalid event version format",
                )

        # Create event using factory method
        result: FlextResult[FlextPayload[Mapping[str, object]]] = cast(
            "FlextResult[FlextPayload[Mapping[str, object]]]",
            cls.create_event(
                event_type=str(event_type),
                event_data=cast("dict[str, object]", event_data),
                aggregate_id=str(aggregate_id) if aggregate_id else None,
                version=version_int,
            ),
        )
        # Cast to match parent class return type
        return result


# =============================================================================
# MIGRATION NOTICE - Helper functions moved to legacy.py
# =============================================================================


# IMPORTANT: Cross-service helper functions have been moved to legacy.py
#
# Migration guide:
# OLD: from flext_core.payload import serialize_payload_for_go_bridge
# NEW: from flext_core.legacy import serialize_payload_for_go_bridge
#      (with transition warning)
# MODERN: use payload.to_json_string() directly with appropriate settings
#
# Helper functions moved:
# - serialize_payload_for_go_bridge()
# - deserialize_payload_from_go_bridge()
# - create_cross_service_message()
# - create_cross_service_event()
# - validate_cross_service_protocol()
# - get_serialization_metrics()
#
# For new code, use the FlextPayload, FlextMessage, and FlextEvent
# class methods directly
#
# TEST CONVENIENCE: Re-export selected helpers from legacy.py here so
# imports like `from flext_core.payload import create_cross_service_event` keep
# working in tests. These are thin wrappers around implementations.
def create_cross_service_event(
    event_type: str,
    event_data: dict[str, object],
    correlation_id: str | None = None,
    **kwargs: object,
) -> FlextResult[FlextEvent]:
    """Create a cross-service event."""
    try:
        # Extract known parameters for create_event
        aggregate_id = kwargs.pop("aggregate_id", None)
        version = kwargs.pop("version", None)

        # Create event with supported parameters only - add type casts for safety
        aggregate_id_str = aggregate_id if isinstance(aggregate_id, str) else None
        version_int = version if isinstance(version, int) else None

        result = FlextEvent.create_event(
            event_type,
            event_data,
            aggregate_id=aggregate_id_str,
            version=version_int,
        )

        if result.is_success and correlation_id and result.value:
            # Add correlation_id to metadata (correlation_id is a property that
            # reads from metadata)
            event = result.value
            new_event = event.with_metadata(correlation_id=correlation_id)
            return FlextResult[FlextEvent].ok(cast("FlextEvent", new_event))

        return result
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        return FlextResult[FlextEvent].fail(f"Cross-service event creation failed: {e}")


def create_cross_service_message(
    message_text: str,
    correlation_id: str | None = None,
    **kwargs: object,
) -> FlextResult[FlextMessage]:
    """Create a cross-service message."""
    try:
        # Extract known parameters for create_message with type safety
        level = kwargs.pop("level", "info")
        source = kwargs.pop("source", None)

        # Ensure proper types
        level_str = level if isinstance(level, str) else "info"
        source_str = source if isinstance(source, str) else None

        # Create a message with supported parameters only
        result = FlextMessage.create_message(
            message_text,
            level=level_str,
            source=source_str,
        )

        if result.is_success and correlation_id and result.value:
            # Add correlation_id to metadata (correlation_id is a property that
            # reads from metadata)
            message = result.value
            new_message = message.with_metadata(correlation_id=correlation_id)
            return FlextResult[FlextMessage].ok(cast("FlextMessage", new_message))

        return result
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        return FlextResult[FlextMessage].fail(
            f"Cross-service message creation failed: {e}"
        )


def get_serialization_metrics(
    payload: object | None = None,
) -> dict[str, object]:
    """Get serialization metrics for payload."""
    metrics: dict[str, object] = {
        "payload_type": type(payload).__name__ if payload is not None else "NoneType",
        "data_type": "unknown",
    }

    # Try to get data type from payload
    if hasattr(payload, "data"):
        data_obj = payload.data
        metrics["data_type"] = type(data_obj).__name__
    elif isinstance(payload, dict) and "data" in payload:
        data_obj = cast("dict[str, object]", payload).get("data")
        metrics["data_type"] = type(data_obj).__name__

    return metrics


def validate_cross_service_protocol(payload: object) -> FlextResult[None]:
    """Validate cross-service protocol."""
    try:
        # Basic validation for cross-service protocol
        if isinstance(payload, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict) and "format" in parsed:
                    return FlextResult[None].ok(None)
            except (json.JSONDecodeError, TypeError):
                return FlextResult[None].fail("Invalid JSON format")

        if isinstance(payload, dict) and ("format" in payload or "data" in payload):
            # Check for minimum required fields
            return FlextResult[None].ok(None)

        return FlextResult[None].fail("Invalid protocol format")
    except (TypeError, ValueError, AttributeError, KeyError) as e:
        return FlextResult[None].fail(f"Protocol validation error: {e}")


# =============================================================================
# MODEL REBUILDS - Resolve forward references for Pydantic
# =============================================================================

# CRITICAL: Model rebuild enabled for test functionality
# The models need explicit rebuild to work correctly with generic types
# in test environment to avoid "not fully defined" errors.
try:
    FlextPayload.model_rebuild()
    FlextMessage.model_rebuild()
    FlextEvent.model_rebuild()
except (
    TypeError,
    ValueError,
    AttributeError,
    ImportError,
    RuntimeError,
) as except_data:
    # Log rebuild errors but maintain runtime functionality
    logger = flext_get_logger(__name__)
    logger.warning(
        "Model rebuild failed, continuing with runtime functionality",
        error=str(except_data),
        models=["FlextPayload", "FlextMessage", "FlextEvent"],
    )

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
    # Cross-service helper functions for test convenience
    "create_cross_service_event",
    "create_cross_service_message",
    "get_serialization_metrics",
    "validate_cross_service_protocol",
]
