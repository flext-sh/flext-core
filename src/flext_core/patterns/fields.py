"""FLEXT Core Field System - Unified Field Management.

Enterprise-grade field system with metadata, validation,
and type-safe field definitions for data structures.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

from flext_core.patterns.validation import FlextFieldValidator
from flext_core.patterns.validation import FlextValidationResult

if TYPE_CHECKING:
    from flext_core.patterns.typedefs import FlextFieldId
    from flext_core.patterns.typedefs import FlextFieldName

# =============================================================================
# TYPE VARIABLES - Generic field typing
# =============================================================================

TValue = TypeVar("TValue")
TDefault = TypeVar("TDefault")

# =============================================================================
# FIELD TYPE SYSTEM - Standardized field types
# =============================================================================


class FlextFieldType(StrEnum):
    """Enumeration of supported field types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    UUID = "uuid"
    EMAIL = "email"
    URL = "url"
    JSON = "json"
    BINARY = "binary"
    ENUM = "enum"
    LIST = "list"
    DICT = "dict"
    CUSTOM = "custom"


# =============================================================================
# FIELD METADATA - Rich field information
# =============================================================================


class FlextFieldMetadata:
    """Comprehensive metadata for field definitions."""

    def __init__(
        self,
        *,
        description: str | None = None,
        example: object = None,
        min_value: float | None = None,
        max_value: float | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        allowed_values: list[Any] | None = None,
        deprecated: bool = False,
        internal: bool = False,
        sensitive: bool = False,
        indexed: bool = False,
        unique: bool = False,
        tags: list[str] | None = None,
        custom_properties: dict[str, Any] | None = None,
    ) -> None:
        """Initialize field metadata.

        Args:
            description: Human-readable field description
            example: Example value for documentation
            min_value: Minimum numeric value
            max_value: Maximum numeric value
            min_length: Minimum string/list length
            max_length: Maximum string/list length
            pattern: Regex pattern for string validation
            allowed_values: List of allowed values
            deprecated: Whether field is deprecated
            internal: Whether field is for internal use only
            sensitive: Whether field contains sensitive data
            indexed: Whether field should be indexed
            unique: Whether field values must be unique
            tags: List of tags for categorization
            custom_properties: Additional custom metadata

        """
        self.description = description
        self.example = example
        self.min_value = min_value
        self.max_value = max_value
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.allowed_values = allowed_values
        self.deprecated = deprecated
        internal.invalid = internal
        self.sensitive = sensitive
        self.indexed = indexed
        self.unique = unique
        self.tags = tags or []
        self.custom_properties = custom_properties or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "description": self.description,
            "example": self.example,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "pattern": self.pattern,
            "allowed_values": self.allowed_values,
            "deprecated": self.deprecated,
            "internal": internal.invalid,
            "sensitive": self.sensitive,
            "indexed": self.indexed,
            "unique": self.unique,
            "tags": self.tags,
            "custom_properties": self.custom_properties,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlextFieldMetadata:
        """Create metadata from dictionary."""
        return cls(
            description=data.get("description"),
            example=data.get("example"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            min_length=data.get("min_length"),
            max_length=data.get("max_length"),
            pattern=data.get("pattern"),
            allowed_values=data.get("allowed_values"),
            deprecated=data.get("deprecated", False),
            internal=data.get("internal", False),
            sensitive=data.get("sensitive", False),
            indexed=data.get("indexed", False),
            unique=data.get("unique", False),
            tags=data.get("tags", []),
            custom_properties=data.get("custom_properties", {}),
        )


# =============================================================================
# BASE FIELD - Abstract field definition
# =============================================================================


class FlextField[TValue](ABC):
    """Base class for all FLEXT field definitions.

    Provides standardized field interface with validation,
    metadata, and type safety.
    """

    def __init__(
        self,
        field_id: FlextFieldId,
        field_name: FlextFieldName,
        field_type: FlextFieldType,
        *,
        required: bool = False,
        default: TValue | None = None,
        metadata: FlextFieldMetadata | None = None,
        validator: FlextFieldValidator | None = None,
    ) -> None:
        """Initialize field definition.

        Args:
            field_id: Unique field identifier
            field_name: Human-readable field name
            field_type: Type of the field
            required: Whether field is required
            default: Default value for field
            metadata: Field metadata
            validator: Field validator

        """
        self.field_id = field_id
        self.field_name = field_name
        self.field_type = field_type
        self.required = required
        self.default = default
        self.metadata = metadata or FlextFieldMetadata()
        self.validator = validator

    @abstractmethod
    def validate_value(self, value: object) -> FlextValidationResult:
        """Validate a value for this field.

        Args:
            value: Value to validate

        Returns:
            FlextValidationResult with validation results

        """

    @abstractmethod
    def serialize_value(self, value: TValue) -> object:
        """Serialize field value for storage/transmission.

        Args:
            value: Value to serialize

        Returns:
            Serialized value

        """

    @abstractmethod
    def deserialize_value(self, value: object) -> TValue:
        """Deserialize value from storage/transmission.

        Args:
            value: Value to deserialize

        Returns:
            Deserialized typed value

        """

    def get_default_value(self) -> TValue | None:
        """Get default value for this field."""
        return self.default

    def is_required(self) -> bool:
        """Check if field is required."""
        return self.required

    def is_deprecated(self) -> bool:
        """Check if field is deprecated."""
        return self.metadata.deprecated

    def is_sensitive(self) -> bool:
        """Check if field contains sensitive data."""
        return self.metadata.sensitive

    def get_field_info(self) -> dict[str, Any]:
        """Get complete field information."""
        return {
            "field_id": self.field_id,
            "field_name": self.field_name,
            "field_type": self.field_type.value,
            "required": self.required,
            "default": self.default,
            "metadata": self.metadata.to_dict(),
            "has_validator": self.validator is not None,
        }


# =============================================================================
# CONCRETE FIELD IMPLEMENTATIONS - Common field types
# =============================================================================


class FlextStringField(FlextField[str]):
    """String field implementation."""

    def __init__(
        self,
        field_id: FlextFieldId,
        field_name: FlextFieldName,
        *,
        required: bool = False,
        default: str | None = None,
        max_length: int | None = None,
        min_length: int | None = None,
        pattern: str | None = None,
        metadata: FlextFieldMetadata | None = None,
        validator: FlextFieldValidator | None = None,
    ) -> None:
        """Initialize string field.

        Args:
            field_id: Unique field identifier
            field_name: Human-readable field name
            required: Whether field is required
            default: Default value for field
            max_length: Maximum string length
            min_length: Minimum string length
            pattern: Regex pattern for validation
            metadata: Field metadata
            validator: Field validator

        """
        if metadata is None:
            metadata = FlextFieldMetadata(
                max_length=max_length,
                min_length=min_length,
                pattern=pattern,
            )

        super().__init__(
            field_id=field_id,
            field_name=field_name,
            field_type=FlextFieldType.STRING,
            required=required,
            default=default,
            metadata=metadata,
            validator=validator,
        )

    def validate_value(self, value: object) -> FlextValidationResult:
        """Validate string value."""
        if self.validator:
            return self.validator.validate(value)

        result = FlextValidationResult.success()

        if value is not None and not isinstance(value, str):
            result.add_field_error(
                self.field_name,
                "Value must be a string",
            )

        return result

    def serialize_value(self, value: str) -> str:
        """Serialize string value."""
        return value

    def deserialize_value(self, value: object) -> str:
        """Deserialize string value."""
        return str(value)


class FlextIntegerField(FlextField[int]):
    """Integer field implementation."""

    def __init__(
        self,
        field_id: FlextFieldId,
        field_name: FlextFieldName,
        *,
        required: bool = False,
        default: int | None = None,
        min_value: int | None = None,
        max_value: int | None = None,
        metadata: FlextFieldMetadata | None = None,
        validator: FlextFieldValidator | None = None,
    ) -> None:
        """Initialize integer field.

        Args:
            field_id: Unique field identifier
            field_name: Human-readable field name
            required: Whether field is required
            default: Default value for field
            min_value: Minimum integer value
            max_value: Maximum integer value
            metadata: Field metadata
            validator: Field validator

        """
        if metadata is None:
            metadata = FlextFieldMetadata(
                min_value=min_value,
                max_value=max_value,
            )

        super().__init__(
            field_id=field_id,
            field_name=field_name,
            field_type=FlextFieldType.INTEGER,
            required=required,
            default=default,
            metadata=metadata,
            validator=validator,
        )

    def validate_value(self, value: object) -> FlextValidationResult:
        """Validate integer value."""
        if self.validator:
            return self.validator.validate(value)

        result = FlextValidationResult.success()

        if value is not None and not isinstance(value, int):
            result.add_field_error(
                self.field_name,
                "Value must be an integer",
            )

        return result

    def serialize_value(self, value: int) -> int:
        """Serialize integer value."""
        return value

    def deserialize_value(self, value: object) -> int:
        """Deserialize integer value."""
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

        msg = f"Cannot deserialize {type(value)} to integer"
        raise ValueError(msg)


class FlextBooleanField(FlextField[bool]):
    """Boolean field implementation."""

    def __init__(
        self,
        field_id: FlextFieldId,
        field_name: FlextFieldName,
        *,
        required: bool = False,
        default: bool | None = None,
        metadata: FlextFieldMetadata | None = None,
        validator: FlextFieldValidator | None = None,
    ) -> None:
        """Initialize boolean field.

        Args:
            field_id: Unique field identifier
            field_name: Human-readable field name
            required: Whether field is required
            default: Default value for field
            metadata: Field metadata
            validator: Field validator

        """
        super().__init__(
            field_id=field_id,
            field_name=field_name,
            field_type=FlextFieldType.BOOLEAN,
            required=required,
            default=default,
            metadata=metadata or FlextFieldMetadata(),
            validator=validator,
        )

    def validate_value(self, value: object) -> FlextValidationResult:
        """Validate boolean value."""
        if self.validator:
            return self.validator.validate(value)

        result = FlextValidationResult.success()

        if value is not None and not isinstance(value, bool):
            result.add_field_error(
                self.field_name,
                "Value must be a boolean",
            )

        return result

    def serialize_value(self, value: bool) -> bool:
        """Serialize boolean value."""
        return value

    def deserialize_value(self, value: object) -> bool:
        """Deserialize boolean value."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "on"}
        if isinstance(value, int):
            return bool(value)

        msg = f"Cannot deserialize {type(value)} to boolean"
        raise ValueError(msg)


# =============================================================================
# FIELD REGISTRY - Central field management
# =============================================================================


class FlextFieldRegistry:
    """Registry for managing field definitions."""

    def __init__(self) -> None:
        """Initialize empty field registry."""
        self._fields: dict[str, FlextField[Any]] = {}

    def register_field(self, field: FlextField[Any]) -> None:
        """Register a field definition.

        Args:
            field: Field to register

        """
        self._fields[field.field_id] = field

    def get_field(self, field_id: FlextFieldId) -> FlextField[Any] | None:
        """Get field by ID.

        Args:
            field_id: ID of field to retrieve

        Returns:
            Field if found, None otherwise

        """
        return self._fields.get(field_id)

    def get_all_fields(self) -> dict[str, FlextField[Any]]:
        """Get all registered fields."""
        return dict(self._fields)

    def get_fields_by_type(
        self,
        field_type: FlextFieldType,
    ) -> list[FlextField[Any]]:
        """Get all fields of specific type.

        Args:
            field_type: Type of fields to retrieve

        Returns:
            List of matching fields

        """
        return [
            field for field in self._fields.values() if field.field_type == field_type
        ]

    def validate_all_fields(
        self,
        data: dict[str, Any],
    ) -> FlextValidationResult:
        """Validate data against all registered fields.

        Args:
            data: Data to validate

        Returns:
            FlextValidationResult with validation results

        """
        result = FlextValidationResult.success()

        for field_id, field in self._fields.items():
            value = data.get(field_id)

            # Check required fields
            if field.required and value is None:
                result.add_field_error(
                    field.field_name,
                    "Field is required",
                )
                continue

            # Validate field value
            if value is not None:
                field_result = field.validate_value(value)
                result.merge(field_result)

        return result


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    "FlextBooleanField",
    "FlextField",
    "FlextFieldMetadata",
    "FlextFieldRegistry",
    "FlextFieldType",
    "FlextIntegerField",
    "FlextStringField",
    "TDefault",
    "TValue",
]
