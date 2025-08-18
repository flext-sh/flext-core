"""Field definition and validation system with metadata support."""

from __future__ import annotations

import re
from contextlib import suppress
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core.constants import FlextFieldType
from flext_core.exceptions import FlextValidationError
from flext_core.loggings import FlextLoggerFactory
from flext_core.result import FlextResult
from flext_core.typings import (
    FlextFieldId,
    FlextFieldName,
    FlextFieldTypeStr,
    FlextValidator,
    TFieldInfo,
)

# Type alias for JSON dictionary
JsonDict = dict[str, object]

# =============================================================================
# TYPE DEFINITIONS - Consolidates without underscore
# =============================================================================

# Type aliases moved to typings.py for centralization
# Import from flext_core.typings: FlextFieldId, FlextFieldName, FlextFieldTypeStr


# =============================================================================
# FLEXT FIELD CORE - Consolidates using Pydantic
# =============================================================================


class FlextFieldCore(
    BaseModel,
):
    """Immutable field definition with validation and metadata.

    Thread-safe field definition supporting registry integration
    and comprehensive validation rules.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",
    )

    # Core identification
    field_id: FlextFieldId
    field_name: FlextFieldName
    field_type: FlextFieldTypeStr

    # Core behavior
    required: bool = True
    default_value: str | int | float | bool | None = None

    # Validation constraints
    min_value: int | float | None = Field(
        default=None,
        description="Minimum numeric value",
    )
    max_value: int | float | None = Field(
        default=None,
        description="Maximum numeric value",
    )
    min_length: int | None = Field(
        default=None,
        ge=0,
        description="Minimum string length",
    )
    max_length: int | None = Field(
        default=None,
        ge=1,
        description="Maximum string length",
    )
    pattern: str | None = Field(
        default=None,
        description="Regex pattern for validation",
    )
    allowed_values: list[object] = Field(
        default_factory=lambda: cast("list[object]", []),
        description="Allowed value list",
    )

    # Metadata
    description: str | None = Field(default=None, description="Field description")
    example: str | int | float | bool | None = Field(
        default=None,
        description="Example",
    )
    deprecated: bool = Field(default=False, description="Is field deprecated")
    sensitive: bool = Field(default=False, description="Contains sensitive data")
    indexed: bool = Field(default=False, description="Should be indexed")
    tags: list[str] = Field(default_factory=list, description="Field tags")

    # Custom validator function for field validation
    validator: FlextValidator | None = Field(
        default=None,
        description="Callable validator function for custom field validation",
    )

    # Mixin functionality is now inherited properly:
    # - Validation methods from FlextValidatableMixin
    # - Serialization methods from FlextSerializableMixin

    # =========================================================================
    # MIXIN FUNCTIONALITY - Now delegated through __getattr__ method
    # =========================================================================
    # All validation and serialization methods delegated to:
    # - FlextValidatableMixin: validation_errors, is_valid, add_validation_error, etc.
    # - FlextSerializableMixin: to_dict_basic, _serialize_value, etc.

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, v: str | None) -> str | None:
        """Validate regex pattern is compilable."""
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                error_msg: str = f"Invalid regex pattern: {e}"
                raise FlextValidationError(
                    error_msg,
                    validation_details={"field": "pattern", "value": v},
                ) from e
        return v

    @field_validator("max_length")
    @classmethod
    def _validate_max_length(cls, v: int | None, info: object) -> int | None:
        """Validate max_length > min_length."""
        # FieldValidationInfo.data is a mapping of sibling fields available during
        # validation. Use safe access and explicit typing to keep linters happy.
        if v is not None:
            data = getattr(info, "data", None)
            if isinstance(data, dict) and "min_length" in data:
                typed_data = cast("dict[str, object]", data)
                min_length = typed_data.get("min_length")
                if min_length is not None:
                    min_length_int = _safe_cast_int(min_length)
                    if min_length_int is not None and v <= min_length_int:
                        error_msg = "max_length must be greater than min_length"
                        raise FlextValidationError(
                            error_msg,
                            validation_details={
                                "field": "max_length",
                                "value": v,
                                "min_length": min_length_int,
                            },
                        )
        return v

    def validate_field_value(self, value: object) -> tuple[bool, str | None]:
        """Validate value against field rules.

        Returns:
            Tuple of (is_valid, error_message)

        """
        # Handle None values
        if value is None:
            if self.required and self.default_value is None:
                return False, f"Field '{self.field_name}' is required"
            return True, None

        # Type-specific validation
        # Support both enum and string representations of field_type
        if isinstance(self.field_type, FlextFieldType):
            field_type_value = self.field_type.value
        else:
            field_type_value = str(self.field_type)

        if field_type_value == "string":
            return self._validate_string_value(value)
        if field_type_value == "integer":
            return self._validate_integer_value(value)
        if field_type_value == "boolean":
            return self._validate_boolean_value(value)

        return True, None

    # Backwards-compatible method expected by tests: returns FlextResult
    def validate_value(self, value: object) -> FlextResult[object]:
        """Compatibility wrapper returning FlextResult for legacy tests."""
        is_valid, err = self.validate_field_value(value)
        if is_valid:
            return FlextResult[object].ok(value)
        return FlextResult[object].fail(err or "Validation failed")

    # Backwards-compat convenience methods expected by tests
    def get_default_value(self) -> str | int | float | bool | None:
        """Return the configured default value for this field."""
        return self.default_value

    def is_required(self) -> bool:
        """Return whether the field is required."""
        return bool(self.required)

    def is_deprecated(self) -> bool:
        """Return whether the field is deprecated."""
        return bool(self.deprecated)

    def is_sensitive(self) -> bool:
        """Return whether the field is sensitive."""
        return bool(self.sensitive)

    def get_field_info(self) -> TFieldInfo:
        """Return a dictionary with field information for compatibility."""
        info = self.get_field_schema()
        # Attach metadata as a plain dict (tests accept dict or object)
        info["metadata"] = (
            self.metadata.to_dict()
            if hasattr(self.metadata, "to_dict")
            else self.metadata
        )
        return cast("TFieldInfo", info)

    def serialize_value(self, value: object) -> object:
        """Serialize a value according to the field type (compatibility wrapper)."""
        # Simple default serialization behavior
        if value is None:
            return None
        if isinstance(self.field_type, FlextFieldType):
            t = self.field_type.value
        else:
            t = str(self.field_type)

        if t == "integer":
            try:
                return int(value)
            except (TypeError, ValueError):
                return str(value)
        if t == "boolean":
            try:
                return self._deserialize_boolean_value(value)
            except TypeError:
                return bool(value)
        # default: string
        return str(value)

    def deserialize_value(self, value: object) -> object:
        """Deserialize a value according to the field type (compat wrapper)."""
        # For compatibility just coerce to the corresponding type
        if value is None:
            return None
        if isinstance(self.field_type, FlextFieldType):
            t = self.field_type.value
        else:
            t = str(self.field_type)

        if t == "integer":
            try:
                return int(value)
            except (TypeError, ValueError):
                msg = f"Cannot convert {value!r} to integer"
                raise TypeError(msg) from None
        if t == "boolean":
            return self._deserialize_boolean_value(value)
        # default: string
        return str(value)

    def _validate_string_value(self, value: object) -> tuple[bool, str | None]:
        """Validate string value with all constraints."""
        match value:
            case str() as str_value:
                pass  # Continue with validation
            case _:
                return False, f"Expected string, got {type(value).__name__}"

        # Length validation using simple checks
        if self.min_length is not None and len(str_value) < self.min_length:
            return False, f"String too short: {len(str_value)} < {self.min_length}"

        if self.max_length is not None and len(str_value) > self.max_length:
            return False, f"String too long: {len(str_value)} > {self.max_length}"

        # Pattern validation using simple regex
        if self.pattern is not None and not re.match(self.pattern, str_value):
            return False, f"String does not match pattern: {self.pattern}"

        # Allowed values validation
        if self.allowed_values and str_value not in [
            str(v) for v in self.allowed_values
        ]:
            return False, f"Value not in allowed list: {self.allowed_values}"

        return True, None

    def _validate_integer_value(self, value: object) -> tuple[bool, str | None]:
        """Validate integer value with all constraints."""
        match value:
            case int() as int_value:
                pass  # Continue with validation
            case _:
                return False, f"Expected integer, got {type(value).__name__}"

        # Range validation
        if self.min_value is not None and int_value < self.min_value:
            return False, f"Value too small: {int_value} < {self.min_value}"

        if self.max_value is not None and int_value > self.max_value:
            return False, f"Value too large: {int_value} > {self.max_value}"

        # Allowed values validation
        if self.allowed_values and int_value not in self.allowed_values:
            return False, f"Value not in allowed list: {self.allowed_values}"

        return True, None

    def _validate_boolean_value(self, value: object) -> tuple[bool, str | None]:
        """Validate boolean value with all constraints."""
        match value:
            case bool() as bool_value:
                pass  # Continue with validation
            case _:
                return False, f"Expected boolean, got {type(value).__name__}"

        # Allowed values validation
        if self.allowed_values and bool_value not in self.allowed_values:
            return False, f"Value not in allowed list: {self.allowed_values}"

        return True, None

    def _deserialize_boolean_value(self, value: object) -> bool:
        """Deserialize boolean value from various formats."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ("true", "1", "yes", "on"):
                return True
            if lower_value in ("false", "0", "no", "off"):
                return False
            error_msg = f"Cannot convert string '{value}' to boolean"
            raise TypeError(error_msg)
        if isinstance(value, int):
            if value == 1:
                return True
            if value == 0:
                return False
            error_msg = f"Cannot convert integer {value} to boolean"
            raise TypeError(error_msg)
        error_msg = f"Cannot convert {type(value).__name__} to boolean"
        raise TypeError(error_msg)

    def get_field_schema(self) -> JsonDict:
        """Get field schema as dictionary."""
        return {
            "field_id": self.field_id,
            "field_name": self.field_name,
            "field_type": self.field_type,
            "required": self.required,
            "default_value": self.default_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "pattern": self.pattern,
            "allowed_values": self.allowed_values,
            "description": self.description,
            "example": self.example,
            "deprecated": self.deprecated,
            "sensitive": self.sensitive,
            "indexed": self.indexed,
            "tags": self.tags,
            "validator": self.validator is not None,
        }

    @property
    def metadata(self) -> FlextFieldMetadata:
        """Get field metadata as dictionary."""
        # Return metadata object for backward-compatibility (tests expect an object)
        return FlextFieldMetadata(
            field_id=self.field_id,
            field_name=self.field_name,
            field_type=self.field_type,
            description=(
                str(self.description) if self.description is not None else None
            ),
            required=self.required,
            default_value=_safe_cast_default_value(self.default_value),
            allowed_values=_safe_cast_list(self.allowed_values),
            min_length=_safe_cast_int(self.min_length),
            max_length=_safe_cast_int(self.max_length),
            min_value=_safe_cast_numeric(self.min_value),
            max_value=_safe_cast_numeric(self.max_value),
            pattern=(str(self.pattern) if self.pattern is not None else None),
            validator=self.validator,
            example=_safe_cast_default_value(self.example),
        )


# =============================================================================
# FLEXT FIELD METADATA - Metadata container for field information
# =============================================================================


class FlextFieldMetadata(BaseModel):
    """Metadata container for field information."""

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",
    )

    # Core identification (provide defaults for easy instantiation in tests)
    field_id: FlextFieldId = ""
    field_name: FlextFieldName = ""
    field_type: FlextFieldTypeStr = FlextFieldType.STRING.value

    # Core behavior
    required: bool = True
    default_value: str | int | float | bool | None = None

    # Validation constraints
    min_value: int | float | None = None
    max_value: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    allowed_values: list[object] = Field(
        default_factory=lambda: cast("list[object]", [])
    )

    # Metadata
    description: str | None = None
    example: str | int | float | bool | None = None
    deprecated: bool = False
    sensitive: bool = False
    indexed: bool = False
    tags: list[str] = Field(default_factory=list)

    # Custom validator function for field validation
    validator: FlextValidator | None = None

    # Back-compat extra fields used by tests
    internal: bool = False
    unique: bool = False
    custom_properties: dict[str, object] | None = Field(
        default_factory=lambda: cast("dict[str, object]", {}),
    )

    def to_dict(self) -> JsonDict:
        """Convert metadata to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> FlextFieldMetadata:
        """Create metadata from a plain dictionary with defaults for missing keys."""
        data = data or {}
        # Normalize allowed values and tags if present
        allowed = data.get("allowed_values", [])
        tags = data.get("tags", [])
        return cls(
            field_id=str(data.get("field_id", "")),
            field_name=str(data.get("field_name", "")),
            field_type=str(data.get("field_type", FlextFieldType.STRING.value)),
            required=bool(data.get("required", True)),
            default_value=_safe_cast_default_value(data.get("default_value")),
            min_value=_safe_cast_numeric(data.get("min_value")),
            max_value=_safe_cast_numeric(data.get("max_value")),
            min_length=_safe_cast_int(data.get("min_length")),
            max_length=_safe_cast_int(data.get("max_length")),
            pattern=(
                str(data.get("pattern")) if data.get("pattern") is not None else None
            ),
            allowed_values=_safe_cast_list(allowed),
            description=(
                str(data.get("description"))
                if data.get("description") is not None
                else None
            ),
            example=_safe_cast_default_value(data.get("example")),
            deprecated=bool(data.get("deprecated", False)),
            sensitive=bool(data.get("sensitive", False)),
            indexed=bool(data.get("indexed", False)),
            tags=_safe_cast_string_list(tags),
            validator=None,
            internal=bool(data.get("internal", False)),
            unique=bool(data.get("unique", False)),
            custom_properties=cast(
                "dict[str, object]", data.get("custom_properties", {})
            ),
        )

    @classmethod
    def from_field(cls, field: FlextFieldCore) -> FlextFieldMetadata:
        """Create metadata from a FlextFieldCore instance."""
        return cls(
            field_id=field.field_id,
            field_name=field.field_name,
            field_type=field.field_type,
            required=field.required,
            default_value=field.default_value,
            min_value=field.min_value,
            max_value=field.max_value,
            min_length=field.min_length,
            max_length=field.max_length,
            pattern=field.pattern,
            allowed_values=field.allowed_values,
            description=field.description,
            example=field.example,
            deprecated=field.deprecated,
            sensitive=field.sensitive,
            indexed=field.indexed,
            tags=field.tags,
            validator=field.validator,
            custom_properties=cast(
                "dict[str, object]",
                field.metadata.model_dump().get("custom_properties", {}),
            ),
        )

    def get_field_info(self) -> TFieldInfo:
        """Get field info as dictionary."""
        return {
            "field_id": self.field_id,
            "field_name": self.field_name,
            "field_type": self.field_type,
            "required": self.required,
            "default_value": self.default_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "pattern": self.pattern,
            "allowed_values": self.allowed_values,
            "description": self.description,
            "example": self.example,
            "deprecated": self.deprecated,
            "sensitive": self.sensitive,
            "indexed": self.indexed,
            "tags": self.tags,
            "validator": self.validator is not None,
        }

    # Provide a compatibility alias used by some tests
    def to_dict_basic(self) -> JsonDict:
        return self.to_dict()


# =============================================================================
# FLEXT FIELD REGISTRY - Central registry for field management
# =============================================================================


class FlextFieldRegistry:
    """Central registry for field management and lookup."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._fields: dict[str, FlextFieldCore] = {}
        self._logger = FlextLoggerFactory.get_logger(__name__)

    def register_field(self, field: FlextFieldCore) -> JsonDict:
        """Register field in registry.

        Backwards compatible: return FlextResult with the registered field on success.
        """
        try:
            self._fields[field.field_id] = field
            self._logger.info(f"Registered field: {field.field_id}")
            return FlextResult[FlextFieldCore].ok(field)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.exception(f"Failed to register field {field.field_id}")
            return FlextResult[FlextFieldCore].fail(str(exc))

    def get_field_by_id(self, field_id: str) -> FlextResult[FlextFieldCore]:
        """Get field by ID from registry.

        Returns FlextResult with the field on success or failure message.
        """
        if field_id not in self._fields:
            return FlextResult[FlextFieldCore].fail(f"Field not found: {field_id}")
        return FlextResult[FlextFieldCore].ok(self._fields[field_id])

    def get_field_by_name(self, field_name: str) -> FlextResult[FlextFieldCore]:
        """Get field by name from registry.

        Returns FlextResult with the field on success or failure message.
        """
        for field in self._fields.values():
            if field.field_name == field_name:
                return FlextResult[FlextFieldCore].ok(field)
        return FlextResult[FlextFieldCore].fail(f"Field not found: {field_name}")

    def list_field_names(self) -> list[str]:
        """List all registered field names."""
        return [field.field_name for field in self._fields.values()]

    def list_field_ids(self) -> list[str]:
        """List all registered field ids (compatibility)."""
        return [field.field_id for field in self._fields.values()]

    def get_field_count(self) -> int:
        """Get the total number of registered fields."""
        return len(self._fields)

    def clear_registry(self) -> None:
        """Clear all registered fields."""
        self._fields.clear()
        self._logger.info("Registry cleared")

    # Legacy compatibility: prefer dict mapping id -> field. The dict variant
    # is exposed below as `get_all_fields` which returns a mapping expected by
    # existing tests and typing stubs.

    def remove_field(self, field_id: str) -> bool:
        """Remove field from registry by id; return True if removed."""
        if field_id in self._fields:
            del self._fields[field_id]
            self._logger.info(f"Removed field: {field_id}")
            return True
        return False

    def get_fields_by_type(self, type_str: str) -> list[FlextFieldCore]:
        """Get all fields of a specific type."""
        matching_fields: list[FlextFieldCore] = []
        for field in self._fields.values():
            if isinstance(field.field_type, FlextFieldType):
                field_type_str = field.field_type.value
            else:
                field_type_str = str(field.field_type)
            if field_type_str == type_str:
                matching_fields.append(field)

        return matching_fields

    def get_field(self, field_id: str) -> FlextFieldCore | None:
        # Backwards compatible variant expected in older tests: return the raw
        # field object or None. Keep a thin wrapper method `get_field_by_id`
        # that returns FlextResult.
        return self._fields.get(field_id)

    def get_all_fields(self) -> dict[str, FlextFieldCore]:
        return dict(self._fields)

    def validate_all_fields(self, data: dict[str, object]) -> FlextResult[None]:
        # Validate each registered field against provided data
        for fid, field in self._fields.items():
            value = data.get(fid)
            res = field.validate_value(value)
            if res.is_failure:
                return FlextResult[None].fail(res.error or "Validation failed")
        return FlextResult[None].ok(None)


# =============================================================================
# HELPER FUNCTIONS - Type-safe conversion utilities
# =============================================================================


def _safe_cast_default_value(value: object) -> str | int | float | bool | None:
    """Safely cast default value to appropriate type."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _safe_cast_numeric(value: object) -> int | float | None:
    """Safely cast value to numeric type."""
    result: int | float | None = None

    if value is not None:
        if isinstance(value, (int, float)):
            result = value
        elif isinstance(value, str):
            with suppress(ValueError, TypeError):
                result = float(value) if "." in value else int(value)
        else:
            # Try to convert using __float__ or __int__ attributes
            with suppress(ValueError, TypeError):
                if hasattr(value, "__float__"):
                    result = float(str(value))
                elif hasattr(value, "__int__"):
                    result = int(str(value))

    return result


def _safe_cast_string_list(value: object) -> list[str]:
    """Safely cast value to string list."""
    if value is None:
        return []
    if isinstance(value, list):
        as_objects: list[object] = cast("list[object]", value)
        result: list[str] = [str(item) for item in as_objects if item is not None]
        return result
    if isinstance(value, (str, int, float, bool)):
        return [str(value)]
    return []


def _safe_cast_int(value: object) -> int | None:
    """Safely cast value to int."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        if isinstance(value, (float, str)):
            return int(value)
        # Use explicit casting with type checking
        if hasattr(value, "__int__"):
            return int(str(value))
        return None
    except (ValueError, TypeError):
        return None


def _safe_cast_list(value: object) -> list[object]:
    """Safely cast value to list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: list[object] = list(cast("list[object] | tuple[object, ...]", value))
        return result
    return [value]


# =============================================================================
# FLEXT FIELDS - Consolidados com factory methods + API pública
# =============================================================================


class FlextFields:
    """Consolidated fields factory and management interface.

    Serves as the primary public API for field creation, registration, and
    management. Combines factory methods with registry operations in a
    unified interface.

    Comprehensive field factory providing type-safe field creation,
    registration, and lookup capabilities following SOLID principles.
    """

    # Registry singleton
    _registry = FlextFieldRegistry()

    @classmethod
    def create_string_field(
        cls,
        field_id: FlextFieldId,
        field_name: FlextFieldName,
        **field_config: object,
    ) -> FlextFieldCore:
        """Create string field with validation."""
        # Extract and validate field configuration values with type safety
        min_length_val = field_config.get("min_length")
        max_length_val = field_config.get("max_length")
        pattern_val = field_config.get("pattern")
        allowed_values_val = field_config.get("allowed_values", [])
        description_val = field_config.get("description")
        example_val = field_config.get("example")
        required_val = field_config.get("required", True)
        deprecated_val = field_config.get("deprecated")
        sensitive_val = field_config.get("sensitive")
        indexed_val = field_config.get("indexed")
        tags_val = field_config.get("tags", [])
        default_val = field_config.get("default_value")

        # Type-safe conversion with helper functions
        min_length_int: int | None = _safe_cast_int(min_length_val)
        max_length_int: int | None = _safe_cast_int(max_length_val)
        pattern_str: str | None = str(pattern_val) if pattern_val is not None else None
        allowed_values_list: list[object] = _safe_cast_list(allowed_values_val)
        description_str: str | None = (
            str(description_val) if description_val is not None else None
        )
        required_bool: bool = bool(required_val)
        deprecated_bool: bool = (
            bool(deprecated_val) if deprecated_val is not None else False
        )
        sensitive_bool: bool = (
            bool(sensitive_val) if sensitive_val is not None else False
        )
        indexed_bool: bool = bool(indexed_val) if indexed_val is not None else False
        tags_list: list[str] = _safe_cast_string_list(tags_val)
        default_value_safe: str | int | float | bool | None = _safe_cast_default_value(
            default_val
        )
        example_safe: str | int | float | bool | None = _safe_cast_default_value(
            example_val
        )

        return FlextFieldCore(
            field_id=field_id,
            field_name=field_name,
            field_type=FlextFieldType.STRING.value,
            required=required_bool,
            default_value=default_value_safe,
            min_length=min_length_int,
            max_length=max_length_int,
            pattern=pattern_str,
            allowed_values=allowed_values_list,
            description=description_str,
            example=example_safe,
            deprecated=deprecated_bool,
            sensitive=sensitive_bool,
            indexed=indexed_bool,
            tags=tags_list,
        )

    @classmethod
    def create_integer_field(
        cls,
        field_id: FlextFieldId,
        field_name: FlextFieldName,
        **field_config: object,
    ) -> FlextFieldCore:
        """Create integer field with validation."""
        # Extract and validate field configuration values with type safety
        min_value_val = field_config.get("min_value")
        max_value_val = field_config.get("max_value")
        description_val = field_config.get("description")
        example_val = field_config.get("example")
        required_val = field_config.get("required", True)
        deprecated_val = field_config.get("deprecated")
        sensitive_val = field_config.get("sensitive")
        indexed_val = field_config.get("indexed")
        tags_val = field_config.get("tags", [])
        default_val = field_config.get("default_value")

        # Type-safe conversion with helper functions
        min_value_float = _safe_cast_numeric(min_value_val)
        max_value_float = _safe_cast_numeric(max_value_val)
        description_str = str(description_val) if description_val is not None else None
        required_bool = bool(required_val)
        deprecated_bool = bool(deprecated_val) if deprecated_val is not None else False
        sensitive_bool = bool(sensitive_val) if sensitive_val is not None else False
        indexed_bool = bool(indexed_val) if indexed_val is not None else False
        tags_list = _safe_cast_string_list(tags_val)
        default_value_safe = _safe_cast_default_value(default_val)
        example_safe = _safe_cast_default_value(example_val)

        return FlextFieldCore(
            field_id=field_id,
            field_name=field_name,
            field_type=FlextFieldType.INTEGER.value,
            required=required_bool,
            default_value=default_value_safe,
            min_value=min_value_float,
            max_value=max_value_float,
            description=description_str,
            example=example_safe,
            deprecated=deprecated_bool,
            sensitive=sensitive_bool,
            indexed=indexed_bool,
            tags=tags_list,
        )

    @classmethod
    def create_boolean_field(
        cls,
        field_id: FlextFieldId,
        field_name: FlextFieldName,
        **field_config: object,
    ) -> FlextFieldCore:
        """Create boolean field with validation."""
        # Extract and validate field configuration values with type safety
        description_val = field_config.get("description")
        example_val = field_config.get("example")
        required_val = field_config.get("required", True)
        deprecated_val = field_config.get("deprecated")
        sensitive_val = field_config.get("sensitive")
        indexed_val = field_config.get("indexed")
        tags_val = field_config.get("tags", [])
        default_val = field_config.get("default_value")

        # Type-safe conversion with helper functions
        description_str = str(description_val) if description_val is not None else None
        required_bool = bool(required_val)
        deprecated_bool = bool(deprecated_val) if deprecated_val is not None else False
        sensitive_bool = bool(sensitive_val) if sensitive_val is not None else False
        indexed_bool = bool(indexed_val) if indexed_val is not None else False
        tags_list = _safe_cast_string_list(tags_val)
        default_value_safe = _safe_cast_default_value(default_val)
        example_safe = _safe_cast_default_value(example_val)

        return FlextFieldCore(
            field_id=field_id,
            field_name=field_name,
            field_type=FlextFieldType.BOOLEAN.value,
            required=required_bool,
            default_value=default_value_safe,
            description=description_str,
            example=example_safe,
            deprecated=deprecated_bool,
            sensitive=sensitive_bool,
            indexed=indexed_bool,
            tags=tags_list,
        )

    @classmethod
    def register_field(cls, field: FlextFieldCore) -> FlextResult[FlextFieldCore]:
        """Register field in global registry (returns FlextResult)."""
        return cls._registry.register_field(field)

    @classmethod
    def get_field_by_id(cls, field_id: FlextFieldId) -> FlextResult[FlextFieldCore]:
        """Get field by ID from global registry (returns FlextResult)."""
        return cls._registry.get_field_by_id(field_id)

    @classmethod
    def get_field_by_name(
        cls,
        field_name: FlextFieldName,
    ) -> FlextResult[FlextFieldCore]:
        """Get field by name from global registry (returns FlextResult)."""
        return cls._registry.get_field_by_name(field_name)

    @classmethod
    def list_field_names(cls) -> list[str]:
        """List all registered field names."""
        return cls._registry.list_field_names()

    @classmethod
    def get_field_count(cls) -> int:
        """Get the total number of registered fields."""
        return cls._registry.get_field_count()

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered fields."""
        cls._registry.clear_registry()

    # Convenience wrappers expected by tests
    @classmethod
    def string_field(cls, name: str, **kwargs: object) -> FlextFieldCore:
        """Create string field with name (convenience method)."""
        return cls.create_string_field(field_id=name, field_name=name, **kwargs)

    @classmethod
    def integer_field(cls, name: str, **kwargs: object) -> FlextFieldCore:
        """Create integer field with name (convenience method)."""
        return cls.create_integer_field(field_id=name, field_name=name, **kwargs)

    @classmethod
    def boolean_field(cls, name: str, **kwargs: object) -> FlextFieldCore:
        """Create boolean field with name (convenience method)."""
        return cls.create_boolean_field(field_id=name, field_name=name, **kwargs)


# =============================================================================
# ALIASES FOR CONVENIENCE
# =============================================================================

# No legacy aliases - only Flext prefixed classes


# =============================================================================
# MIGRATION NOTICE - Legacy convenience functions moved to legacy.py
# =============================================================================

# IMPORTANT: Legacy convenience functions have been moved to legacy.py
#
# Migration guide:
# OLD: from flext_core.fields import flext_create_string_field
# NEW: from flext_core.legacy import flext_create_string_field
#      (with deprecation warning)
# MODERN: from flext_core import FlextFields; FlextFields.create_string_field()
#
# For new code, use FlextFields factory methods directly


# =============================================================================
# CONVENIENCE FIELD CLASSES REMOVED - DRY principle (use factory methods)
# =============================================================================

# Removed FlextStringField, FlextIntegerField, FlextBooleanField classes
# to remove duplication. Use FlextFields.create_string_field() instead
# or flext_create_string_field() helper function.


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Essential for existing tests
# =============================================================================

# Backward compatibility for renamed classes
FlextFieldCoreMetadata = FlextFieldMetadata

# =============================================================================
# LEGACY FIELD CREATION FUNCTIONS - Backward compatibility
# =============================================================================


def flext_create_string_field(name: str, **kwargs: object) -> FlextFieldCore:
    """Create string field (legacy function)."""
    return FlextFields.string_field(name, **kwargs)


def flext_create_integer_field(name: str, **kwargs: object) -> FlextFieldCore:
    """Create integer field (legacy function)."""
    return FlextFields.integer_field(name, **kwargs)


def flext_create_boolean_field(name: str, **kwargs: object) -> FlextFieldCore:
    """Create boolean field (legacy function)."""
    return FlextFields.boolean_field(name, **kwargs)


# =============================================================================
# EXPORTS - Clean public API following guidelines
# =============================================================================

# Export API
__all__: list[str] = [
    "FlextFieldCore",
    "FlextFieldMetadata",
    "FlextFieldRegistry",
    "FlextFields",
    "JsonDict",
]
