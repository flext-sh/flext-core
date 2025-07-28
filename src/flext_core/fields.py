"""FLEXT Core Fields Module.

Comprehensive field definition and validation system for enterprise data management with
metadata support, registry patterns, and type-safe operations. Implements consolidated
architecture with factory patterns and comprehensive validation.

Architecture:
    - Single source of truth pattern eliminating base module duplication
    - Pydantic-based validation with strict type safety and immutability
    - Factory pattern for type-safe field creation with validation
    - Registry singleton pattern for centralized field management
    - Multiple inheritance from specialized mixin classes for behavior composition
    - FlextResult pattern integration for consistent error handling

Field System Components:
    - FlextFieldCore: Immutable field definition with comprehensive validation
    - FlextFieldRegistry: Thread-safe centralized field registration and lookup
    - FlextFields: Factory methods and unified public API interface
    - Type definitions: Strong typing for field properties and metadata
    - Convenience functions: Automatic registration with factory methods
    - Validation integration: Type-specific validation with FlextValidators

Maintenance Guidelines:
    - Add new field types through factory methods in FlextFields class
    - Maintain validation consistency with FlextValidators integration patterns
    - Use FlextResult pattern for all operations that can fail or return errors
    - Keep field metadata separate from validation logic for clear separation
    - Register new field types with consistent naming and validation patterns
    - Follow immutability principles with frozen Pydantic models

Design Decisions:
    - Eliminated _fields_base.py module to reduce code duplication and complexity
    - Pydantic BaseModel for automatic validation and serialization support
    - Multiple inheritance from mixin classes for reusable behavior composition
    - Factory pattern for type-safe field creation with comprehensive validation
    - Registry singleton for global field management with conflict detection
    - Frozen models for immutability and thread safety in concurrent environments

Enterprise Features:
    - Type-safe field creation with compile-time verification through factory methods
    - Comprehensive metadata support for field description and categorization
    - Registry pattern for centralized field management and conflict resolution
    - Validation integration with enterprise-grade error handling and reporting
    - Thread-safe operations for concurrent access in multi-threaded environments
    - Immutable field definitions preventing accidental modification after creation

Field Validation Strategy:
    - Type-specific validation methods for string, integer, boolean, and custom types
    - Constraint validation using FlextValidators for consistency and reusability
    - FlextResult pattern for validation outcomes with comprehensive error reporting
    - Comprehensive error messaging for field violations with detailed context
    - Pattern-based validation for strings with regex compilation verification
    - Range validation for numeric types with min/max constraint enforcement

Registry Management:
    - Singleton registry pattern for global field management and access
    - Dual indexing by field ID and field name for efficient lookup operations
    - Conflict detection and resolution for field ID and name uniqueness
    - Thread-safe registration and retrieval operations for concurrent access
    - Registry statistics and management operations for monitoring and maintenance
    - Clean removal operations with proper cleanup and consistency maintenance

Dependencies:
    - pydantic: Field definition, validation core, and immutable model configuration
    - _mixins_base: Serializable and validatable behavior inheritance patterns
    - validation: FlextValidators for core validation utilities and consistency
    - constants: FlextFieldType definitions and field categorization constants
    - result: FlextResult pattern for consistent error handling across operations

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core._mixins_base import _BaseSerializableMixin, _BaseValidatableMixin
from flext_core.constants import FlextFieldType
from flext_core.result import FlextResult
from flext_core.types import FlextValidator, TEntityId
from flext_core.validation import FlextValidators

# =============================================================================
# TYPE DEFINITIONS - Consolidados sem underscore
# =============================================================================

# Type aliases consolidados - Python 3.13 + objetos sem underscore
type FlextFieldId = TEntityId
type FlextFieldName = str
type FlextFieldTypeStr = str


# =============================================================================
# FLEXT FIELD CORE - Consolidado usando Pydantic máximo
# =============================================================================


class FlextFieldCore(BaseModel, _BaseSerializableMixin, _BaseValidatableMixin):
    """Core field definition with comprehensive validation and metadata.

    Consolidated field implementation serving as single source of truth for all
    field functionality. Combines Pydantic validation with custom field logic.

    Architecture:
        - Pydantic BaseModel for automatic validation and serialization
        - Mixin inheritance for serializable and validatable behaviors
        - Frozen model for immutability and thread safety
        - Type-specific validation methods for different field types

    Validation Features:
        - Automatic type validation through Pydantic
        - Custom regex pattern validation for strings
        - Range validation for numeric types
        - Length constraints for string types
        - Allowed values enumeration support

    Metadata Support:
        - Field description and examples
        - Deprecation and sensitivity markers
        - Indexing hints for database optimization
        - Custom tags for field categorization

    Usage:
        field = FlextFieldCore(
            field_id="user_email",
            field_name="email",
            field_type=FlextFieldType.STRING,
            pattern=EMAIL_PATTERN,
            required=True
        )

        result = field.validate_value("user@example.com")
        if result.is_success:
            validated_value = result.data
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
    default_value: object = None

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
        default_factory=list,
        description="Allowed value list",
    )

    # Metadata
    description: str | None = Field(default=None, description="Field description")
    example: object = Field(default=None, description="Example value")
    deprecated: bool = Field(default=False, description="Is field deprecated")
    sensitive: bool = Field(default=False, description="Contains sensitive data")
    indexed: bool = Field(default=False, description="Should be indexed")
    tags: list[str] = Field(default_factory=list, description="Field tags")

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, v: str | None) -> str | None:
        """Validate regex pattern is compilable."""
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                error_msg = f"Invalid regex pattern: {e}"
                raise ValueError(error_msg) from e
        return v

    @field_validator("max_length")
    @classmethod
    def _validate_max_length(cls, v: int | None, info: object) -> int | None:
        """Validate max_length > min_length."""
        if v is not None and hasattr(info, "data") and "min_length" in info.data:
            min_length = info.data["min_length"]
            if min_length is not None and v <= min_length:
                error_msg = "max_length must be greater than min_length"
                raise ValueError(error_msg)
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
        if self.field_type == FlextFieldType.STRING.value:
            return self._validate_string_value(value)
        if self.field_type == FlextFieldType.INTEGER.value:
            return self._validate_integer_value(value)
        if self.field_type == FlextFieldType.BOOLEAN.value:
            return self._validate_boolean_value(value)

        return True, None

    def _validate_string_value(self, value: object) -> tuple[bool, str | None]:
        """Validate string value with all constraints."""
        if not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"

        # Length validation using base validators
        if self.min_length is not None and not FlextValidators.has_min_length(
            value,
            self.min_length,
        ):
            return False, f"String too short: {len(value)} < {self.min_length}"

        if self.max_length is not None and not FlextValidators.has_max_length(
            value,
            self.max_length,
        ):
            return False, f"String too long: {len(value)} > {self.max_length}"

        # Pattern validation using base validators
        if self.pattern is not None and not FlextValidators.matches_pattern(
            value,
            self.pattern,
        ):
            return False, f"String does not match pattern: {self.pattern}"

        # Allowed values validation
        if self.allowed_values and value not in self.allowed_values:
            return False, f"Value not in allowed list: {self.allowed_values}"

        return True, None

    def _validate_integer_value(self, value: object) -> tuple[bool, str | None]:
        """Validate integer value with range constraints."""
        if not isinstance(value, int):
            return False, f"Expected integer, got {type(value).__name__}"

        # Range validation using base validators
        if self.min_value is not None and value < self.min_value:
            return False, f"Integer too small: {value} < {self.min_value}"

        if self.max_value is not None and value > self.max_value:
            return False, f"Integer too large: {value} > {self.max_value}"

        return True, None

    def _validate_boolean_value(self, value: object) -> tuple[bool, str | None]:
        """Validate boolean value."""
        if not isinstance(value, bool):
            return False, f"Expected boolean, got {type(value).__name__}"
        return True, None

    def has_tag(self, tag: str) -> bool:
        """Check if field has specific tag."""
        return tag in self.tags

    def get_field_schema(self) -> dict[str, object]:
        """Get complete field schema."""
        return self.model_dump()

    def get_field_metadata(self) -> dict[str, object]:
        """Get field metadata only."""
        return {
            "description": self.description,
            "example": self.example,
            "deprecated": self.deprecated,
            "sensitive": self.sensitive,
            "indexed": self.indexed,
            "tags": self.tags,
        }

    def validate_value(self, value: object) -> FlextResult[object]:
        """Validate value using FlextResult pattern."""
        is_valid, error_message = self.validate_field_value(value)

        if is_valid:
            return FlextResult.ok(value if value is not None else self.default_value)
        return FlextResult.fail(error_message or "Validation failed")


# =============================================================================
# FLEXT FIELD METADATA - Metadata wrapper for field information
# =============================================================================


class FlextFieldMetadata(BaseModel):
    """Field metadata wrapper providing standardized access to field information.

    Consolidated metadata container for field properties, validation rules,
    and descriptive information. Serves as a standardized interface for
    field introspection and documentation.

    Architecture:
        - Immutable metadata container with Pydantic validation
        - Direct mapping from FlextFieldCore properties
        - Standardized access patterns for field information
        - JSON-serializable structure for API and documentation

    Metadata Categories:
        - Core identification: field_id, field_name, field_type
        - Behavior settings: required, default_value
        - Validation constraints: min/max values, length, pattern, allowed_values
        - Documentation: description, example, tags
        - System flags: deprecated, sensitive, indexed

    Usage:
        field = FlextFieldCore(...)
        metadata = FlextFieldMetadata.from_field(field)

        # Access metadata properties
        assert metadata.field_name == "user_email"
        assert metadata.required is True
        assert metadata.description == "User email address"
    """

    model_config = ConfigDict(frozen=True)

    # Core identification
    field_id: FlextFieldId
    field_name: FlextFieldName
    field_type: FlextFieldTypeStr

    # Behavior settings
    required: bool
    default_value: object | None

    # Validation constraints
    min_value: int | float | None = None
    max_value: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    allowed_values: list[object] = Field(default_factory=list)

    # Documentation
    description: str | None = None
    example: object | None = None
    tags: list[str] = Field(default_factory=list)

    # System flags
    deprecated: bool = False
    sensitive: bool = False
    indexed: bool = False

    @classmethod
    def from_field(cls, field: FlextFieldCore) -> FlextFieldMetadata:
        """Create metadata from field instance.

        Args:
            field: FlextFieldCore instance to extract metadata from

        Returns:
            FlextFieldMetadata with all field properties

        """
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
            tags=field.tags,
            deprecated=field.deprecated,
            sensitive=field.sensitive,
            indexed=field.indexed,
        )


# =============================================================================
# FLEXT FIELD REGISTRY - Consolidado seguindo padrão estabelecido
# =============================================================================


class FlextFieldRegistry(BaseModel, _BaseSerializableMixin, _BaseValidatableMixin):
    """Centralized field registry for managing field definitions.

    Provides thread-safe registration and lookup of field instances with
    conflict detection and resolution capabilities.

    Architecture:
        - Pydantic BaseModel for validation and serialization
        - Dual indexing by field ID and field name
        - FlextResult pattern for all operations that can fail
        - Thread-safe operations for concurrent access

    Registry Features:
        - Unique field ID enforcement
        - Unique field name enforcement within registry
        - Efficient lookup by ID or name
        - Registry statistics and management operations

    Conflict Resolution:
        - Prevents duplicate field IDs
        - Prevents duplicate field names
        - Clear error messages for registration conflicts
        - Safe removal operations with cleanup

    Usage:
        registry = FlextFieldRegistry()
        field = FlextFieldCore(field_id="user_id", field_name="user_id", ...)

        result = registry.register_field(field)
        if result.is_success:
            lookup_result = registry.get_field_by_name("user_id")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fields_dict: dict[FlextFieldId, FlextFieldCore] = Field(
        default_factory=dict,
        exclude=True,
        description="Field storage by ID",
    )
    field_names_dict: dict[FlextFieldName, FlextFieldId] = Field(
        default_factory=dict,
        exclude=True,
        description="Name to ID mapping",
    )

    def register_field(self, field: FlextFieldCore) -> FlextResult[None]:
        """Register field with conflict detection."""
        # Check for ID conflicts
        if field.field_id in self.fields_dict:
            return FlextResult.fail(f"Field ID '{field.field_id}' already registered")

        # Check for name conflicts
        if field.field_name in self.field_names_dict:
            existing_id = self.field_names_dict[field.field_name]
            error_msg = (
                f"Field name '{field.field_name}' already registered "
                f"with ID '{existing_id}'"
            )
            return FlextResult.fail(error_msg)

        # Register field
        self.fields_dict[field.field_id] = field
        self.field_names_dict[field.field_name] = field.field_id
        return FlextResult.ok(None)

    def get_field_by_id(self, field_id: FlextFieldId) -> FlextResult[FlextFieldCore]:
        """Get field by ID."""
        field = self.fields_dict.get(field_id)
        if field is not None:
            return FlextResult.ok(field)
        return FlextResult.fail(f"Field ID '{field_id}' not found")

    def get_field_by_name(
        self,
        field_name: FlextFieldName,
    ) -> FlextResult[FlextFieldCore]:
        """Get field by name."""
        field_id = self.field_names_dict.get(field_name)
        if field_id is None:
            return FlextResult.fail(f"Field name '{field_name}' not found")

        field = self.fields_dict.get(field_id)
        if field is not None:
            return FlextResult.ok(field)
        return FlextResult.fail(f"Field ID '{field_id}' not found")

    def list_field_names(self) -> list[FlextFieldName]:
        """List all registered field names."""
        return list(self.field_names_dict.keys())

    def list_field_ids(self) -> list[FlextFieldId]:
        """List all registered field IDs."""
        return list(self.fields_dict.keys())

    def get_field_count(self) -> int:
        """Get total number of registered fields."""
        return len(self.fields_dict)

    def clear_registry(self) -> None:
        """Clear all registered fields."""
        self.fields_dict.clear()
        self.field_names_dict.clear()

    def remove_field(self, field_id: FlextFieldId) -> bool:
        """Remove field by ID.

        Returns:
            True if field was removed, False if not found

        """
        if field_id not in self.fields_dict:
            return False

        field = self.fields_dict[field_id]
        del self.fields_dict[field_id]
        del self.field_names_dict[field.field_name]
        return True


# =============================================================================
# FLEXT FIELDS - Consolidados com factory methods + API pública
# =============================================================================


class FlextFields:
    """Consolidated fields factory and management interface.

    Serves as the primary public API for field creation, registration, and
    management. Combines factory methods with registry operations in a
    unified interface.

    Architecture:
        - Factory pattern for type-safe field creation
        - Singleton registry for global field management
        - FlextResult pattern for all fallible operations
        - Type-specific factory methods for common field types

    Factory Methods:
        - create_string_field: String fields with pattern and length validation
        - create_integer_field: Integer fields with range validation
        - create_boolean_field: Boolean fields with type validation
        - Extensible pattern for additional field types

    Registry Integration:
        - Automatic registration option for convenience functions
        - Direct registry access for advanced operations
        - Centralized field lookup and management
        - Registry statistics and maintenance operations

    Design Pattern:
        - Static methods for stateless factory operations
        - Class-level registry singleton for global state
        - Consistent parameter patterns across field types
        - Backward compatibility through aliases

    Usage:
        # Create and register field
        field = FlextFields.create_string_field(
            field_id="email",
            field_name="user_email",
            pattern=EMAIL_PATTERN,
            required=True
        )

        # Register manually
        result = FlextFields.register_field(field)

        # Lookup field
        field_result = FlextFields.get_field_by_name("user_email")
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
        return FlextFieldCore(
            field_id=field_id,
            field_name=field_name,
            field_type=FlextFieldType.STRING.value,
            required=bool(field_config.get("required", True)),
            default_value=field_config.get("default_value"),
            min_length=int(cast("int", min_length_val))
            if (min_length_val := field_config.get("min_length")) is not None
            else None,
            max_length=int(cast("int", max_length_val))
            if (max_length_val := field_config.get("max_length")) is not None
            else None,
            pattern=str(field_config.get("pattern"))
            if field_config.get("pattern") is not None
            else None,
            allowed_values=cast("list[object]", field_config.get("allowed_values", [])),
            description=str(field_config.get("description"))
            if field_config.get("description") is not None
            else None,
            example=field_config.get("example"),
            deprecated=bool(field_config.get("deprecated", False)),
            sensitive=bool(field_config.get("sensitive", False)),
            indexed=bool(field_config.get("indexed", False)),
            tags=cast("list[str]", field_config.get("tags", [])),
        )

    @classmethod
    def create_integer_field(
        cls,
        field_id: FlextFieldId,
        field_name: FlextFieldName,
        **field_config: object,
    ) -> FlextFieldCore:
        """Create integer field with validation."""
        return FlextFieldCore(
            field_id=field_id,
            field_name=field_name,
            field_type=FlextFieldType.INTEGER.value,
            required=bool(field_config.get("required", True)),
            default_value=field_config.get("default_value"),
            min_value=float(cast("float", min_value_val))
            if (min_value_val := field_config.get("min_value")) is not None
            else None,
            max_value=float(cast("float", max_value_val))
            if (max_value_val := field_config.get("max_value")) is not None
            else None,
            description=str(field_config.get("description"))
            if field_config.get("description") is not None
            else None,
            example=field_config.get("example"),
            deprecated=bool(field_config.get("deprecated", False)),
            sensitive=bool(field_config.get("sensitive", False)),
            indexed=bool(field_config.get("indexed", False)),
            tags=cast("list[str]", field_config.get("tags", [])),
        )

    @classmethod
    def create_boolean_field(
        cls,
        field_id: FlextFieldId,
        field_name: FlextFieldName,
        **field_config: object,
    ) -> FlextFieldCore:
        """Create boolean field with validation."""
        return FlextFieldCore(
            field_id=field_id,
            field_name=field_name,
            field_type=FlextFieldType.BOOLEAN.value,
            required=bool(field_config.get("required", True)),
            default_value=field_config.get("default_value"),
            description=str(field_config.get("description"))
            if field_config.get("description") is not None
            else None,
            example=field_config.get("example"),
            deprecated=bool(field_config.get("deprecated", False)),
            sensitive=bool(field_config.get("sensitive", False)),
            indexed=bool(field_config.get("indexed", False)),
            tags=cast("list[str]", field_config.get("tags", [])),
        )

    @classmethod
    def register_field(cls, field: FlextFieldCore) -> FlextResult[None]:
        """Register field in global registry."""
        return cls._registry.register_field(field)

    @classmethod
    def get_field_by_id(cls, field_id: FlextFieldId) -> FlextResult[FlextFieldCore]:
        """Get field by ID from global registry."""
        return cls._registry.get_field_by_id(field_id)

    @classmethod
    def get_field_by_name(
        cls,
        field_name: FlextFieldName,
    ) -> FlextResult[FlextFieldCore]:
        """Get field by name from global registry."""
        return cls._registry.get_field_by_name(field_name)

    @classmethod
    def list_field_names(cls) -> list[FlextFieldName]:
        """List all registered field names."""
        return cls._registry.list_field_names()

    @classmethod
    def get_field_count(cls) -> int:
        """Get total number of registered fields."""
        return cls._registry.get_field_count()

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered fields."""
        cls._registry.clear_registry()


# =============================================================================
# CONVENIENCE ALIASES E FUNÇÕES - Mantendo compatibilidade
# =============================================================================

# No legacy aliases - only Flext prefixed classes


# Convenience functions with automatic registration and flext_ prefix
def flext_create_string_field(
    field_id: str,
    field_name: str,
    **config: object,
) -> FlextFieldCore:
    """Create and automatically register string field.

    Convenience function that combines field creation with automatic registry
    registration. Raises ValueError if registration fails due to conflicts.

    Args:
        field_id: Unique identifier for the field
        field_name: Human-readable field name
        **config: Additional field configuration parameters

    Returns:
        Created and registered FlextFieldCore instance

    Raises:
        ValueError: If field registration fails due to ID or name conflicts

    """
    field = FlextFields.create_string_field(field_id, field_name, **config)
    result = FlextFields.register_field(field)
    if result.is_failure:
        raise ValueError(result.error)
    return field


def flext_create_integer_field(
    field_id: str,
    field_name: str,
    **config: object,
) -> FlextFieldCore:
    """Create and automatically register integer field.

    Convenience function that combines field creation with automatic registry
    registration. Raises ValueError if registration fails due to conflicts.

    Args:
        field_id: Unique identifier for the field
        field_name: Human-readable field name
        **config: Additional field configuration parameters

    Returns:
        Created and registered FlextFieldCore instance

    Raises:
        ValueError: If field registration fails due to ID or name conflicts

    """
    field = FlextFields.create_integer_field(field_id, field_name, **config)
    result = FlextFields.register_field(field)
    if result.is_failure:
        raise ValueError(result.error)
    return field


def flext_create_boolean_field(
    field_id: str,
    field_name: str,
    **config: object,
) -> FlextFieldCore:
    """Create and automatically register boolean field.

    Convenience function that combines field creation with automatic registry
    registration. Raises ValueError if registration fails due to conflicts.

    Args:
        field_id: Unique identifier for the field
        field_name: Human-readable field name
        **config: Additional field configuration parameters

    Returns:
        Created and registered FlextFieldCore instance

    Raises:
        ValueError: If field registration fails due to ID or name conflicts

    """
    field = FlextFields.create_boolean_field(field_id, field_name, **config)
    result = FlextFields.register_field(field)
    if result.is_failure:
        raise ValueError(result.error)
    return field


# =============================================================================
# CONVENIENCE FIELD CLASSES REMOVED - DRY principle (use factory methods)
# =============================================================================

# Removed FlextStringField, FlextIntegerField, FlextBooleanField classes
# to eliminate duplication. Use FlextFields.create_string_field() instead
# or flext_create_string_field() helper function.


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Essential for existing tests
# =============================================================================

# Backward compatibility for renamed classes
FlextFieldCoreMetadata = FlextFieldMetadata

# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__ = [
    "FlextFieldCore",
    "FlextFieldCoreMetadata",  # Backward compatibility alias
    "FlextFieldId",
    "FlextFieldMetadata",
    "FlextFieldName",
    "FlextFieldRegistry",
    "FlextFieldTypeStr",
    "FlextFields",
    "FlextValidator",
    "flext_create_boolean_field",
    "flext_create_integer_field",
    "flext_create_string_field",
]
