"""Comprehensive field definition and validation system for the FLEXT ecosystem.

This module provides enterprise-grade field validation, metadata management, and schema
processing capabilities following Clean Architecture principles and SOLID patterns.

Built for Python 3.13+ with strict typing enforcement, the module uses hierarchical
nested classes to organize functionality by domain and prevent circular imports.

Key Features:
    - Hierarchical FlextFields class with nested domain organization
    - Type-safe field definitions with comprehensive validation
    - Registry-based field management with performance optimization
    - Schema processing and metadata extraction capabilities
    - Railway-oriented programming with FlextResult integration
    - Zero circular dependencies through proper layering

Architecture:
    - FlextFields: Main hierarchical class containing all field functionality
    - FlextFields.Core: Foundation field types and basic operations
    - FlextFields.Validation: Field validation and constraint checking
    - FlextFields.Registry: Field registration and management
    - FlextFields.Schema: Schema processing and metadata extraction
    - FlextFields.Factory: Factory patterns for field creation

Examples:
    Basic field creation and validation::

        from flext_core.fields import FlextFields

        # Create and validate a string field
        field = FlextFields.Core.StringField(
            name="username", min_length=3, max_length=20
        )

        result = field.validate("john_doe")
        if result.success:
            print(f"Valid: {result.value}")

    Field registry operations::

        # Register a field type
        registry = FlextFields.Registry.FieldRegistry()
        result = registry.register_field("email", FlextFields.Core.EmailField())

        # Get registered field
        field_result = registry.get_field("email")
        if field_result.success:
            field = field_result.value

    Schema processing::

        # Process field schema
        processor = FlextFields.Schema.FieldProcessor()
        schema_result = processor.process_field_schema(field_definition)

Note:
    This module enforces Python 3.13+ requirements and uses modern nested class
    patterns to avoid circular imports. All functionality is organized within
    the FlextFields hierarchical structure following SOLID principles.

"""

from __future__ import annotations

import contextlib
import re
import uuid
from abc import abstractmethod
from datetime import UTC, datetime
from typing import Final, cast, override

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# =============================================================================
# FLEXT FIELDS HIERARCHICAL SYSTEM - Complete field management ecosystem
# =============================================================================


class FlextFields:
    """Hierarchical field system organizing all field functionality by domain.

    This is the consolidated class for all FLEXT field operations, following the
    FLEXT hierarchical pattern. All field functionality is organized into nested
    classes by domain and responsibility, preventing circular imports while
    maintaining clean architecture.

    The field system is organized into the following domains:
        - Core: Basic field types and fundamental operations
        - Validation: Field validation logic and constraint checking
        - Registry: Field registration, management and lookup
        - Schema: Schema processing and metadata extraction
        - Factory: Factory patterns for field creation and builders
        - Metadata: Field metadata management and introspection

    Architecture Principles:
        - Single Responsibility: Each nested class handles one domain
        - Open/Closed: Extensible through inheritance and composition
        - Liskov Substitution: All field types implement base contracts
        - Interface Segregation: Minimal interfaces for specific needs
        - Dependency Inversion: Depends on abstractions, not concretions

    Examples:
        Field creation and validation::

            # Create field through Core nested class
            field = FlextFields.Core.StringField("username", min_length=3)

            # Validate through Validation nested class
            result = FlextFields.Validation.validate_field(field, "john")
            if result.success:
                print(f"Valid: {result.value}")

        Registry operations::

            registry = FlextFields.Registry.FieldRegistry()
            registry.register_field("user_email", email_field)
            field_result = registry.get_field("user_email")

        Factory pattern::

            builder = FlextFields.Factory.FieldBuilder("string", "name")
            field = builder.with_length(1, 50).build().unwrap()

    """

    # ==========================================================================
    # CORE FIELD TYPES - Foundation field implementations
    # ==========================================================================

    class Core:
        """Core field types and fundamental field operations.

        This nested class contains all basic field type implementations and
        core field operations. All field types inherit from BaseField and
        implement the standard validation contract.

        The core system provides:
            - BaseField abstract base class with validation protocol
            - Concrete field implementations for common data types
            - Type-safe field validation with FlextResult integration
            - Metadata extraction and field introspection capabilities
            - Immutable field configurations following value object patterns
        """

        class BaseField[T]:
            """Abstract base class for all field types.

            Defines the fundamental contract that all field implementations must
            follow. Uses FlextResult for error handling and FlextTypes for
            type annotations.

            Args:
                name: Field name identifier
                required: Whether field is required (default True)
                default: Default value for optional fields
                description: Human-readable field description

            Raises:
                ValueError: If field name is empty or invalid

            """

            def __init__(
                self,
                name: FlextTypes.Fields.String,
                *,
                required: FlextTypes.Fields.Boolean = True,
                default: T | None = None,
                description: FlextTypes.Fields.String = "",
            ) -> None:
                if not name or not name.strip():
                    error_msg = FlextConstants.Messages.VALIDATION_FAILED
                    raise ValueError(error_msg)

                self.name = name.strip()
                self.required = required
                self.default = default
                self.description = description

            @abstractmethod
            def validate(self, value: FlextTypes.Fields.Object) -> FlextResult[T]:
                """Validate field value - must be implemented by subclasses.

                Args:
                    value: Value to validate

                Returns:
                    FlextResult with validated value or error

                """

            @property
            def field_type(self) -> FlextTypes.Fields.String:
                """Get field type identifier.

                Returns:
                    String identifying the field type

                """
                return self.__class__.__name__.replace("Field", "").lower()

            def get_metadata(self) -> FlextTypes.Fields.Dict:
                """Extract field metadata for introspection.

                Returns:
                    Dictionary containing field configuration and metadata

                """
                return {
                    "name": self.name,
                    "type": self.field_type,
                    "required": self.required,
                    "default": self.default,
                    "description": self.description,
                }

        class StringField(BaseField[str]):
            """String field with length and pattern validation."""

            def __init__(
                self,
                name: FlextTypes.Fields.String,
                *,
                required: FlextTypes.Fields.Boolean = True,
                default: FlextTypes.Fields.String | None = None,
                description: FlextTypes.Fields.String = "",
                min_length: FlextTypes.Fields.Integer | None = None,
                max_length: FlextTypes.Fields.Integer | None = None,
                pattern: FlextTypes.Fields.String | None = None,
            ) -> None:
                super().__init__(
                    name, required=required, default=default, description=description
                )
                self.min_length = min_length
                self.max_length = max_length
                self.pattern = re.compile(pattern) if pattern else None

            @override
            def validate(
                self, value: FlextTypes.Fields.Object
            ) -> FlextResult[FlextTypes.Fields.String]:
                """Validate string value with length and pattern constraints.

                Args:
                    value: Value to validate as string

                Returns:
                    FlextResult with validated string or validation error

                """
                if value is None and not self.required:
                    return FlextResult[str].ok(self.default or "")

                if not isinstance(value, str):
                    return FlextResult[str].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                if self.min_length is not None and len(value) < self.min_length:
                    return FlextResult[str].fail(
                        f"String too short: {len(value)} < {self.min_length}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if self.max_length is not None and len(value) > self.max_length:
                    return FlextResult[str].fail(
                        f"String too long: {len(value)} > {self.max_length}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if self.pattern and not self.pattern.match(value):
                    return FlextResult[str].fail(
                        f"String does not match pattern: {self.pattern.pattern}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[str].ok(value)

        class IntegerField(BaseField[int]):
            """Integer field with range validation."""

            def __init__(
                self,
                name: FlextTypes.Fields.String,
                *,
                required: FlextTypes.Fields.Boolean = True,
                default: FlextTypes.Fields.Integer | None = None,
                description: FlextTypes.Fields.String = "",
                min_value: FlextTypes.Fields.Integer | None = None,
                max_value: FlextTypes.Fields.Integer | None = None,
            ) -> None:
                super().__init__(
                    name, required=required, default=default, description=description
                )
                self.min_value = min_value
                self.max_value = max_value

            @override
            def validate(
                self, value: FlextTypes.Fields.Object
            ) -> FlextResult[FlextTypes.Fields.Integer]:
                """Validate integer value with range constraints.

                Args:
                    value: Value to validate as integer

                Returns:
                    FlextResult with validated integer or validation error

                """
                if value is None and not self.required:
                    return FlextResult[int].ok(self.default or 0)

                if not isinstance(value, int) or isinstance(value, bool):
                    return FlextResult[int].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                if self.min_value is not None and value < self.min_value:
                    return FlextResult[int].fail(
                        f"Value too small: {value} < {self.min_value}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if self.max_value is not None and value > self.max_value:
                    return FlextResult[int].fail(
                        f"Value too large: {value} > {self.max_value}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[int].ok(value)

        class FloatField(BaseField[float]):
            """Float field with range and precision validation."""

            def __init__(
                self,
                name: FlextTypes.Fields.String,
                *,
                required: FlextTypes.Fields.Boolean = True,
                default: float | None = None,
                description: FlextTypes.Fields.String = "",
                min_value: float | None = None,
                max_value: float | None = None,
                precision: int | None = None,
            ) -> None:
                super().__init__(
                    name, required=required, default=default, description=description
                )
                self.min_value = min_value
                self.max_value = max_value
                self.precision = precision

            @override
            def validate(
                self, value: FlextTypes.Fields.Object
            ) -> FlextResult[FlextTypes.Fields.Float]:
                """Validate float value with range and precision constraints.

                Args:
                    value: Value to validate as float

                Returns:
                    FlextResult with validated float or validation error

                """
                if value is None and not self.required:
                    return FlextResult[float].ok(self.default or 0.0)

                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return FlextResult[float].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                float_value = float(value)

                if self.min_value is not None and float_value < self.min_value:
                    return FlextResult[float].fail(
                        f"Value too small: {float_value} < {self.min_value}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if self.max_value is not None and float_value > self.max_value:
                    return FlextResult[float].fail(
                        f"Value too large: {float_value} > {self.max_value}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if self.precision is not None:
                    rounded_value = round(float_value, self.precision)
                    return FlextResult[float].ok(rounded_value)

                return FlextResult[float].ok(float_value)

        class BooleanField(BaseField[bool]):
            """Boolean field with flexible value conversion."""

            def __init__(
                self,
                name: FlextTypes.Fields.String,
                *,
                required: FlextTypes.Fields.Boolean = True,
                default: FlextTypes.Fields.Boolean | None = None,
                description: FlextTypes.Fields.String = "",
            ) -> None:
                super().__init__(
                    name, required=required, default=default, description=description
                )

            @override
            def validate(
                self, value: FlextTypes.Fields.Object
            ) -> FlextResult[FlextTypes.Fields.Boolean]:
                """Validate boolean value with flexible conversion.

                Args:
                    value: Value to validate and convert to boolean

                Returns:
                    FlextResult with validated boolean or validation error

                """
                if value is None and not self.required:
                    return FlextResult[bool].ok(self.default or False)

                if isinstance(value, bool):
                    return FlextResult[bool].ok(value)

                # Convert string representations
                if isinstance(value, str):
                    lower_value = value.lower().strip()
                    if lower_value in ("true", "yes", "1", "on", "enabled"):
                        return FlextResult[bool].ok(data=True)
                    if lower_value in ("false", "no", "0", "off", "disabled"):
                        return FlextResult[bool].ok(data=False)

                # Convert numeric values
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return FlextResult[bool].ok(bool(value))

                return FlextResult[bool].fail(
                    FlextConstants.Messages.TYPE_MISMATCH,
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

        class EmailField(StringField):
            """Email field with email format validation."""

            def __init__(
                self,
                name: FlextTypes.Fields.String,
                *,
                required: FlextTypes.Fields.Boolean = True,
                default: FlextTypes.Fields.String | None = None,
                description: FlextTypes.Fields.String = "",
            ) -> None:
                # Use FlextConstants for email pattern
                super().__init__(
                    name,
                    required=required,
                    default=default,
                    description=description,
                    pattern=FlextConstants.Patterns.EMAIL_PATTERN,
                )

            @override
            def validate(self, value: object) -> FlextResult[str]:
                """Validate email address format.

                Args:
                    value: Email address to validate

                Returns:
                    FlextResult with validated email or validation error

                """
                # First run string validation
                string_result = super().validate(value)
                if not string_result.success:
                    return string_result

                # Additional email-specific validation can be added here
                return string_result

        class UuidField(BaseField[str]):
            """UUID field with UUID format validation."""

            def __init__(
                self,
                name: FlextTypes.Fields.String,
                *,
                required: FlextTypes.Fields.Boolean = True,
                default: FlextTypes.Fields.String | None = None,
                description: FlextTypes.Fields.String = "",
            ) -> None:
                super().__init__(
                    name, required=required, default=default, description=description
                )

            @override
            def validate(
                self, value: FlextTypes.Fields.Object
            ) -> FlextResult[FlextTypes.Fields.String]:
                """Validate UUID format.

                Args:
                    value: UUID string to validate

                Returns:
                    FlextResult with validated UUID string or validation error

                """
                if value is None and not self.required:
                    return FlextResult[str].ok(self.default or str(uuid.uuid4()))

                if not isinstance(value, str):
                    return FlextResult[str].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                try:
                    # Validate UUID format by attempting to parse it
                    uuid.UUID(value)
                    return FlextResult[str].ok(value)
                except ValueError:
                    return FlextResult[str].fail(
                        f"Invalid UUID format: {value}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

        class DateTimeField(BaseField[datetime]):
            """DateTime field with format and range validation."""

            def __init__(
                self,
                name: FlextTypes.Fields.String,
                *,
                required: FlextTypes.Fields.Boolean = True,
                default: datetime | None = None,
                description: FlextTypes.Fields.String = "",
                date_format: FlextTypes.Fields.String = "%Y-%m-%d %H:%M:%S",
                min_date: datetime | None = None,
                max_date: datetime | None = None,
            ) -> None:
                super().__init__(
                    name, required=required, default=default, description=description
                )
                self.date_format = date_format
                self.min_date = min_date
                self.max_date = max_date

            @override
            def validate(
                self, value: FlextTypes.Fields.Object
            ) -> FlextResult[datetime]:
                """Validate datetime value with format and range constraints.

                Args:
                    value: DateTime value to validate (string or datetime)

                Returns:
                    FlextResult with validated datetime or validation error

                """
                if value is None and not self.required:
                    return FlextResult[datetime].ok(self.default or datetime.now(UTC))

                # Handle datetime objects
                if isinstance(value, datetime):
                    dt_value = value
                # Handle string representations
                elif isinstance(value, str):
                    try:
                        # Try ISO format first (handles timezone info automatically)
                        if "T" in value or "Z" in value or "+" in value:
                            dt_value = datetime.fromisoformat(value)
                        else:
                            # Fallback to strptime for custom formats,
                            # then make timezone-aware
                            dt_value = datetime.strptime(
                                value, self.date_format
                            ).replace(tzinfo=UTC)
                    except ValueError:
                        return FlextResult[datetime].fail(
                            f"Invalid date format: {value} "
                            f"(expected {self.date_format})",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )
                else:
                    return FlextResult[datetime].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )

                # Range validation
                if self.min_date and dt_value < self.min_date:
                    return FlextResult[datetime].fail(
                        f"Date too early: {dt_value} < {self.min_date}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                if self.max_date and dt_value > self.max_date:
                    return FlextResult[datetime].fail(
                        f"Date too late: {dt_value} > {self.max_date}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[datetime].ok(dt_value)

    # ==========================================================================
    # VALIDATION SUBSYSTEM - Field validation logic and utilities
    # ==========================================================================

    class Validation:
        """Field validation logic and constraint checking utilities.

        This nested class provides validation utilities and constraint checking
        logic that can be used across different field types and validation
        scenarios.
        """

        @staticmethod
        def validate_field(
            field: FlextFields.Core.BaseField[FlextTypes.Fields.Object],
            value: FlextTypes.Fields.Object,
        ) -> FlextResult[FlextTypes.Fields.Object]:
            """Validate a field value using the field's validation logic.

            Args:
                field: Field instance to use for validation
                value: Value to validate

            Returns:
                FlextResult with validated value or validation error

            """
            return field.validate(value)

        @staticmethod
        def validate_multiple_fields(
            fields: list[FlextFields.Core.BaseField[FlextTypes.Fields.Object]],
            values: dict[str, FlextTypes.Fields.Object],
        ) -> FlextResult[dict[str, FlextTypes.Fields.Object]]:
            """Validate multiple fields with their corresponding values.

            Args:
                fields: List of field instances
                values: Dictionary mapping field names to values

            Returns:
                FlextResult with validated values dict or validation errors

            """
            validated_values: dict[str, FlextTypes.Fields.Object] = {}
            errors: list[str] = []

            for field in fields:
                field_value = values.get(field.name)
                result = field.validate(field_value)

                if result.success:
                    validated_values[field.name] = result.value
                else:
                    errors.append(f"{field.name}: {result.error}")

            if errors:
                return FlextResult[dict[str, FlextTypes.Fields.Object]].fail(
                    "; ".join(errors),
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return FlextResult[dict[str, FlextTypes.Fields.Object]].ok(validated_values)

    # ==========================================================================
    # REGISTRY SUBSYSTEM - Field registration and management
    # ==========================================================================

    class Registry:
        """Field registration, management and lookup functionality.

        This nested class provides field registry capabilities for managing
        field types and instances at runtime.
        """

        class FieldRegistry:
            """Registry for managing field instances and types.

            Provides centralized storage and retrieval for field definitions
            and field type factories.
            """

            def __init__(self) -> None:
                self._fields: dict[str, FlextFields.Core.BaseField[object]] = {}
                self._field_types: dict[
                    str, type[FlextFields.Core.BaseField[object]]
                ] = {}

            def register_field(
                self,
                name: FlextTypes.Fields.String,
                field: FlextFields.Core.BaseField[object],
            ) -> FlextResult[None]:
                """Register a field instance by name.

                Args:
                    name: Name to register the field under
                    field: Field instance to register

                Returns:
                    FlextResult indicating success or failure

                """
                if not name or not name.strip():
                    return FlextResult[None].fail(
                        "Field name cannot be empty",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Type is already enforced by parameter annotation

                self._fields[name.strip()] = field
                return FlextResult[None].ok(None)

            def get_field(
                self, name: str
            ) -> FlextResult[FlextFields.Core.BaseField[object]]:
                """Get registered field by name.

                Args:
                    name: Name of field to retrieve

                Returns:
                    FlextResult with field instance or error if not found

                """
                if not name or not name.strip():
                    return FlextResult[FlextFields.Core.BaseField[object]].fail(
                        "Field name cannot be empty",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                field = self._fields.get(name.strip())
                if field is None:
                    return FlextResult[FlextFields.Core.BaseField[object]].fail(
                        f"Field not found: {name}",
                        error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
                    )

                return FlextResult[FlextFields.Core.BaseField[object]].ok(field)

            def register_field_type(
                self,
                type_name: str,
                field_type: type[FlextFields.Core.BaseField[object]],
            ) -> FlextResult[None]:
                """Register a field type for dynamic creation.

                Args:
                    type_name: Name to register the field type under
                    field_type: Field class to register

                Returns:
                    FlextResult indicating success or failure

                """
                if not type_name or not type_name.strip():
                    return FlextResult[None].fail(
                        "Field type name cannot be empty",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Type is already enforced by parameter annotation

                self._field_types[type_name.strip()] = field_type
                return FlextResult[None].ok(None)

            def get_field_type(
                self, type_name: str
            ) -> FlextResult[type[FlextFields.Core.BaseField[object]]]:
                """Get registered field type by name.

                Args:
                    type_name: Name of field type to retrieve

                Returns:
                    FlextResult with field type class or error if not found

                """
                field_type = self._field_types.get(type_name)
                if field_type is None:
                    return FlextResult[type[FlextFields.Core.BaseField[object]]].fail(
                        f"Field type not found: {type_name}",
                        error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
                    )

                return FlextResult[type[FlextFields.Core.BaseField[object]]].ok(
                    field_type
                )

            def list_fields(self) -> list[str]:
                """Get list of all registered field names.

                Returns:
                    List of registered field names

                """
                return list(self._fields.keys())

            def list_field_types(self) -> list[str]:
                """Get list of all registered field type names.

                Returns:
                    List of registered field type names

                """
                return list(self._field_types.keys())

            def get_field_metadata(self, name: str) -> FlextResult[dict[str, object]]:
                """Get metadata for a registered field.

                Args:
                    name: Name of field to get metadata for

                Returns:
                    FlextResult with field metadata dictionary or error

                """
                field_result = self.get_field(name)
                if not field_result.success:
                    return FlextResult[dict[str, object]].fail(
                        field_result.error or "Unknown error"
                    )

                metadata = field_result.value.get_metadata()
                return FlextResult[dict[str, object]].ok(metadata)

            def clear(self) -> None:
                """Clear all registered fields and types."""
                self._fields.clear()
                self._field_types.clear()

    # ==========================================================================
    # SCHEMA SUBSYSTEM - Schema processing and metadata extraction
    # ==========================================================================

    class Schema:
        """Schema processing and metadata extraction capabilities.

        This nested class provides schema processing utilities for working
        with field definitions, extracting metadata, and processing field
        schemas.
        """

        class FieldProcessor:
            """Processor for field schema operations and metadata extraction."""

            def process_field_schema(
                self,
                schema: dict[str, object],
            ) -> FlextResult[dict[str, object]]:
                """Process field schema definition and extract metadata.

                Args:
                    schema: Schema definition dictionary

                Returns:
                    FlextResult with processed schema or validation error

                """
                # Type is already enforced by parameter annotation

                processed_schema: dict[str, object] = {
                    "fields": list[object](),
                    "metadata": {},
                    "validation_rules": [],
                }

                # Extract field definitions
                fields_data_raw = schema.get("fields", [])
                if isinstance(fields_data_raw, list):
                    fields_list = processed_schema["fields"]
                    if isinstance(fields_list, list):
                        # Type cast the list to help Pyright inference
                        fields_data = cast("list[dict[str, object]]", fields_data_raw)
                        for field_def_raw in fields_data:
                            # field_def_raw is guaranteed dict[str, object] from cast
                            field_result = self._extract_field_definition(field_def_raw)
                            if field_result.success:
                                # Type cast for Pyright compatibility
                                typed_value = cast("object", field_result.value)
                                fields_list.append(typed_value)

                # Extract metadata
                metadata = schema.get("metadata", {})
                if isinstance(metadata, dict):
                    processed_schema["metadata"] = metadata

                return FlextResult[dict[str, object]].ok(processed_schema)

            def _extract_field_definition(
                self,
                field_def: dict[str, object],
            ) -> FlextResult[dict[str, object]]:
                """Extract field definition from schema entry.

                Args:
                    field_def: Field definition dictionary

                Returns:
                    FlextResult with extracted field definition or error

                """
                # Type guard to ensure field_def is properly typed
                field_def_dict = dict[str, object](field_def)

                required_keys = ["name", "type"]
                for key in required_keys:
                    if key not in field_def_dict:
                        return FlextResult[dict[str, object]].fail(
                            f"Missing required key: {key}",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )

                # Extract field-specific schema information
                extracted = {
                    "name": field_def_dict["name"],
                    "type": field_def_dict["type"],
                    "required": field_def_dict.get("required", True),
                    "default": field_def_dict.get("default"),
                    "constraints": field_def_dict.get("constraints", {}),
                    "metadata": field_def_dict.get("metadata", {}),
                }

                return FlextResult[dict[str, object]].ok(extracted)

            def process_multiple_fields_schema(
                self,
                schemas: list[dict[str, object]],
            ) -> FlextResult[list[dict[str, object]]]:
                """Process multiple field schemas.

                Args:
                    schemas: List of schema definitions

                Returns:
                    FlextResult with list of processed schemas or error

                """
                processed_schemas: list[dict[str, object]] = []
                errors: list[str] = []

                for i, schema in enumerate(schemas):
                    result = self.process_field_schema(schema)
                    if result.success:
                        processed_schemas.append(result.value)
                    else:
                        errors.append(f"Schema {i}: {result.error}")

                if errors:
                    return FlextResult[list[dict[str, object]]].fail(
                        "; ".join(errors),
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                return FlextResult[list[dict[str, object]]].ok(processed_schemas)

    # ==========================================================================
    # FACTORY SUBSYSTEM - Factory patterns for field creation
    # ==========================================================================

    class Factory:
        """Factory patterns for field creation and builders.

        This nested class provides factory methods and builder patterns for
        creating field instances with fluent interfaces and configuration.
        """

        @staticmethod
        def create_field(
            field_type: FlextTypes.Fields.String,
            name: FlextTypes.Fields.String,
            **config: FlextTypes.Fields.Object,
        ) -> FlextResult[FlextTypes.Fields.Instance]:
            """Create field instance using factory pattern.

            Args:
                field_type: Type of field to create
                name: Name for the field
                **config: Configuration parameters for the field

            Returns:
                FlextResult with created field instance or error

            """
            field_types = {
                "string": FlextFields.Core.StringField,
                "integer": FlextFields.Core.IntegerField,
                "float": FlextFields.Core.FloatField,
                "boolean": FlextFields.Core.BooleanField,
                "email": FlextFields.Core.EmailField,
                "uuid": FlextFields.Core.UuidField,
                "datetime": FlextFields.Core.DateTimeField,
            }

            field_class = field_types.get(field_type.lower())
            if not field_class:
                return FlextResult[FlextTypes.Fields.Instance].fail(
                    f"Unknown field type: {field_type}",
                    error_code=FlextConstants.Errors.NOT_FOUND,
                )

            try:
                # Create field with specific parameter handling for each type
                field = FlextFields.Factory._create_field_with_params(
                    field_class, name, field_type, config
                )
                return FlextResult[FlextTypes.Fields.Instance].ok(field)
            except Exception as e:
                return FlextResult[FlextTypes.Fields.Instance].fail(
                    f"Failed to create field: {e!s}",
                    error_code=FlextConstants.Errors.GENERIC_ERROR,
                )

        @staticmethod
        def _prepare_field_config(  # noqa: PLR0912, PLR0915
            field_type: str,
            config: dict[str, object],
        ) -> dict[str, object]:
            """Prepare field configuration with proper type narrowing.

            Args:
                field_type: Type of field being created
                config: Raw configuration dictionary

            Returns:
                Processed configuration with proper types for the specific field type

            """
            processed_config: dict[str, object] = {}

            # Common field parameters
            if "required" in config:
                value = config["required"]
                if isinstance(value, bool):
                    processed_config["required"] = value
                elif isinstance(value, str):
                    processed_config["required"] = value.lower() in (
                        "true",
                        "yes",
                        "1",
                        "on",
                    )
                else:
                    processed_config["required"] = bool(value)

            if "description" in config:
                value = config["description"]
                if isinstance(value, str):
                    processed_config["description"] = value

            if "default" in config:
                processed_config["default"] = config["default"]

            # Field-specific parameters with proper type narrowing
            if field_type.lower() == "integer":
                if "min_value" in config:
                    value = config["min_value"]
                    if isinstance(value, int):
                        processed_config["min_value"] = value
                    elif isinstance(value, str) and value.isdigit():
                        processed_config["min_value"] = int(value)

                if "max_value" in config:
                    value = config["max_value"]
                    if isinstance(value, int):
                        processed_config["max_value"] = value
                    elif isinstance(value, str) and value.isdigit():
                        processed_config["max_value"] = int(value)

            elif field_type.lower() == "float":
                if "min_value" in config:
                    value = config["min_value"]
                    if isinstance(value, (int, float)):
                        processed_config["min_value"] = float(value)
                    elif isinstance(value, str):
                        with contextlib.suppress(ValueError):
                            processed_config["min_value"] = float(value)

                if "max_value" in config:
                    value = config["max_value"]
                    if isinstance(value, (int, float)):
                        processed_config["max_value"] = float(value)
                    elif isinstance(value, str):
                        with contextlib.suppress(ValueError):
                            processed_config["max_value"] = float(value)

                if "precision" in config:
                    value = config["precision"]
                    if isinstance(value, int):
                        processed_config["precision"] = value
                    elif isinstance(value, str) and value.isdigit():
                        processed_config["precision"] = int(value)

            elif field_type.lower() == "string":
                if "min_length" in config:
                    value = config["min_length"]
                    if isinstance(value, int):
                        processed_config["min_length"] = value
                    elif isinstance(value, str) and value.isdigit():
                        processed_config["min_length"] = int(value)

                if "max_length" in config:
                    value = config["max_length"]
                    if isinstance(value, int):
                        processed_config["max_length"] = value
                    elif isinstance(value, str) and value.isdigit():
                        processed_config["max_length"] = int(value)

                if "pattern" in config:
                    value = config["pattern"]
                    if isinstance(value, str):
                        processed_config["pattern"] = value

            elif field_type.lower() == "datetime":
                if "date_format" in config:
                    value = config["date_format"]
                    if isinstance(value, str):
                        processed_config["date_format"] = value

                if "min_date" in config:
                    value = config["min_date"]
                    if isinstance(value, str):
                        with contextlib.suppress(ValueError):
                            processed_config["min_date"] = datetime.fromisoformat(value)

                if "max_date" in config:
                    value = config["max_date"]
                    if isinstance(value, str):
                        with contextlib.suppress(ValueError):
                            processed_config["max_date"] = datetime.fromisoformat(value)

            return processed_config

        @staticmethod
        def _create_field_with_params(  # noqa: PLR0911, PLR0912, PLR0915
            field_class: type[
                FlextFields.Core.BaseField[object]
                | FlextFields.Core.StringField
                | FlextFields.Core.IntegerField
                | FlextFields.Core.FloatField
                | FlextFields.Core.BooleanField
                | FlextFields.Core.EmailField
                | FlextFields.Core.UuidField
                | FlextFields.Core.DateTimeField
            ],
            name: FlextTypes.Fields.String,
            field_type: FlextTypes.Fields.String,
            config: dict[str, FlextTypes.Fields.Object],
        ) -> FlextTypes.Fields.Instance:
            """Create field with specific parameter handling for each type.

            Args:
                field_class: Field class to instantiate
                name: Field name
                field_type: Type of field being created
                config: Configuration parameters

            Returns:
                Created field instance

            """
            # Extract common parameters with type safety
            required = config.get("required", True)
            if isinstance(required, str):
                required = required.lower() in ("true", "yes", "1", "on")
            elif not isinstance(required, bool):
                required = bool(required)

            description = config.get("description", "")
            if not isinstance(description, str):
                description = ""

            default = config.get("default")

            # Create field based on type with specific parameters
            if field_type.lower() == "string":
                min_length = config.get("min_length")
                if isinstance(min_length, str) and min_length.isdigit():
                    min_length = int(min_length)
                elif not isinstance(min_length, int):
                    min_length = None

                max_length = config.get("max_length")
                if isinstance(max_length, str) and max_length.isdigit():
                    max_length = int(max_length)
                elif not isinstance(max_length, int):
                    max_length = None

                pattern = config.get("pattern")
                if not isinstance(pattern, str):
                    pattern = None

                # Cast default to string if it's not None
                string_default = None
                if default is not None:
                    if isinstance(default, str):
                        string_default = default
                    else:
                        string_default = str(default)

                return FlextFields.Core.StringField(
                    name,
                    required=required,
                    default=string_default,
                    description=description,
                    min_length=min_length,
                    max_length=max_length,
                    pattern=pattern,
                )

            if field_type.lower() == "integer":
                min_value = config.get("min_value")
                if isinstance(min_value, str) and min_value.isdigit():
                    min_value = int(min_value)
                elif not isinstance(min_value, int):
                    min_value = None

                max_value = config.get("max_value")
                if isinstance(max_value, str) and max_value.isdigit():
                    max_value = int(max_value)
                elif not isinstance(max_value, int):
                    max_value = None

                # Cast default to int if it's not None
                int_default = None
                if default is not None:
                    if isinstance(default, int):
                        int_default = default
                    elif isinstance(default, str) and default.isdigit():
                        int_default = int(default)
                    else:
                        int_default = 0

                return FlextFields.Core.IntegerField(
                    name,
                    required=required,
                    default=int_default,
                    description=description,
                    min_value=min_value,
                    max_value=max_value,
                )

            if field_type.lower() == "float":
                min_value = config.get("min_value")
                if isinstance(min_value, (int, float)):
                    min_value = float(min_value)
                elif isinstance(min_value, str):
                    try:
                        min_value = float(min_value)
                    except ValueError:
                        min_value = None
                else:
                    min_value = None

                max_value = config.get("max_value")
                if isinstance(max_value, (int, float)):
                    max_value = float(max_value)
                elif isinstance(max_value, str):
                    try:
                        max_value = float(max_value)
                    except ValueError:
                        max_value = None
                else:
                    max_value = None

                precision = config.get("precision")
                if isinstance(precision, str) and precision.isdigit():
                    precision = int(precision)
                elif not isinstance(precision, int):
                    precision = None

                # Cast default to float if it's not None
                float_default = None
                if default is not None:
                    if isinstance(default, (int, float)):
                        float_default = float(default)
                    elif isinstance(default, str):
                        try:
                            float_default = float(default)
                        except ValueError:
                            float_default = 0.0
                    else:
                        float_default = 0.0

                FlextFields.Core.FloatField(
                    name,
                    required=required,
                    default=float_default,
                    description=description,
                    min_value=min_value,
                    max_value=max_value,
                    precision=precision,
                )

            if field_type.lower() == "datetime":
                date_format = config.get("date_format")
                if not isinstance(date_format, str):
                    date_format = "%Y-%m-%d %H:%M:%S"

                min_date = config.get("min_date")
                if isinstance(min_date, str):
                    try:
                        min_date = datetime.fromisoformat(min_date)
                    except ValueError:
                        min_date = None
                elif not isinstance(min_date, datetime):
                    min_date = None

                max_date = config.get("max_date")
                if isinstance(max_date, str):
                    try:
                        max_date = datetime.fromisoformat(max_date)
                    except ValueError:
                        max_date = None
                elif not isinstance(max_date, datetime):
                    max_date = None

                # Cast default to datetime if it's not None
                datetime_default = None
                if default is not None:
                    if isinstance(default, datetime):
                        datetime_default = default
                    elif isinstance(default, str):
                        try:
                            datetime_default = datetime.fromisoformat(default)
                        except ValueError:
                            datetime_default = datetime.now(UTC)
                    else:
                        datetime_default = datetime.now(UTC)

                return FlextFields.Core.DateTimeField(
                    name,
                    required=required,
                    default=datetime_default,
                    description=description,
                    date_format=date_format,
                    min_date=min_date,
                    max_date=max_date,
                )

            if field_type.lower() == "boolean":
                # Cast default to boolean if it's not None
                bool_default = None
                if default is not None:
                    if isinstance(default, bool):
                        bool_default = default
                    elif isinstance(default, str):
                        bool_default = default.lower() in ("true", "yes", "1", "on")
                    elif isinstance(default, (int, float)):
                        bool_default = bool(default)
                    else:
                        bool_default = False

                return FlextFields.Core.BooleanField(
                    name,
                    required=required,
                    default=bool_default,
                    description=description,
                )

            if field_type.lower() == "email":
                # Cast default to string if it's not None
                email_default = None
                if default is not None:
                    if isinstance(default, str):
                        email_default = default
                    else:
                        email_default = str(default)

                return FlextFields.Core.EmailField(
                    name,
                    required=required,
                    default=email_default,
                    description=description,
                )

            if field_type.lower() == "uuid":
                # Cast default to string if it's not None
                uuid_default = None
                if default is not None:
                    uuid_default = default if isinstance(default, str) else str(default)

                return FlextFields.Core.UuidField(
                    name,
                    required=required,
                    default=uuid_default,
                    description=description,
                )

            # Fallback to base field with common parameters only
            # Cast default to a compatible type
            fallback_default = None
            if default is not None:
                if isinstance(default, (str, int, float, bool)):
                    fallback_default = default
                else:
                    fallback_default = str(default)

            # Type-safe field creation with explicit casting
            return cast(
                "FlextTypes.Fields.Instance",
                field_class(
                    name,
                    required=required,
                    default=fallback_default,  # type: ignore[arg-type]
                    description=description,
                ),
            )

        @staticmethod
        def create_fields_from_schema(
            schema: dict[str, object],
        ) -> FlextResult[list[FlextFields.Core.BaseField[object]]]:
            """Create multiple fields from schema definition.

            Args:
                schema: Schema containing field definitions

            Returns:
                FlextResult with list of created fields or error

            """
            if "fields" not in schema:
                return FlextResult[list[FlextFields.Core.BaseField[object]]].fail(
                    "Schema must contain 'fields' key",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            fields: list[FlextFields.Core.BaseField[object]] = []
            errors: list[str] = []

            schema_fields = schema["fields"]
            if not isinstance(schema_fields, list):
                return FlextResult[list[FlextFields.Core.BaseField[object]]].fail(
                    "Schema 'fields' must be a list",
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

            # Type annotation for schema_fields iteration
            typed_schema_fields: list[object] = schema_fields
            for field_def in typed_schema_fields:
                if not isinstance(field_def, dict):
                    errors.append("Field definition must be dictionary")
                    continue

                # Type cast after isinstance check for Pyright compatibility
                field_def_dict = cast("dict[str, object]", field_def)

                field_type = field_def_dict.get("type")
                field_name = field_def_dict.get("name")

                if not field_type or not field_name:
                    errors.append("Field definition must have 'type' and 'name'")
                    continue

                # Type guard to ensure field_type and field_name are strings
                if not isinstance(field_type, str) or not isinstance(field_name, str):
                    errors.append("Field 'type' and 'name' must be strings")
                    continue

                # Extract configuration
                config = {
                    k: v for k, v in field_def_dict.items() if k not in {"type", "name"}
                }

                result = FlextFields.Factory.create_field(
                    field_type, field_name, **config
                )
                if result.success:
                    fields.append(
                        cast("FlextFields.Core.BaseField[object]", result.value)
                    )
                else:
                    errors.append(f"{field_name}: {result.error}")

            if errors:
                return FlextResult[list[FlextFields.Core.BaseField[object]]].fail(
                    "; ".join(errors),
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return FlextResult[list[FlextFields.Core.BaseField[object]]].ok(fields)

        class FieldBuilder:
            """Builder pattern for fluent field creation."""

            def __init__(
                self,
                field_type: str,
                name: FlextTypes.Fields.String,
            ) -> None:
                self.field_type = field_type
                self.name = name
                self.config: dict[str, object] = {}

            def with_required(
                self, *, required: FlextTypes.Fields.Boolean
            ) -> FlextFields.Factory.FieldBuilder:
                """Set field as required or optional.

                Args:
                    required: Whether field is required

                Returns:
                    Builder instance for method chaining

                """
                self.config["required"] = required
                return self

            def with_default(
                self, default_value: object
            ) -> FlextFields.Factory.FieldBuilder:
                """Set default value for field.

                Args:
                    default_value: Default value for field

                Returns:
                    Builder instance for method chaining

                """
                self.config["default"] = default_value
                return self

            def with_description(
                self, description: str
            ) -> FlextFields.Factory.FieldBuilder:
                """Set field description.

                Args:
                    description: Human-readable description

                Returns:
                    Builder instance for method chaining

                """
                self.config["description"] = description
                return self

            def with_length(
                self,
                min_length: FlextTypes.Fields.Integer | None = None,
                max_length: FlextTypes.Fields.Integer | None = None,
            ) -> FlextFields.Factory.FieldBuilder:
                """Set length constraints for string fields.

                Args:
                    min_length: Minimum string length
                    max_length: Maximum string length

                Returns:
                    Builder instance for method chaining

                """
                if min_length is not None:
                    self.config["min_length"] = min_length
                if max_length is not None:
                    self.config["max_length"] = max_length
                return self

            def with_range(
                self,
                min_value: float | None = None,
                max_value: float | None = None,
            ) -> FlextFields.Factory.FieldBuilder:
                """Set range constraints for numeric fields.

                Args:
                    min_value: Minimum numeric value
                    max_value: Maximum numeric value

                Returns:
                    Builder instance for method chaining

                """
                if min_value is not None:
                    self.config["min_value"] = min_value
                if max_value is not None:
                    self.config["max_value"] = max_value
                return self

            def with_pattern(self, pattern: str) -> FlextFields.Factory.FieldBuilder:
                """Set regex pattern for string validation.

                Args:
                    pattern: Regular expression pattern

                Returns:
                    Builder instance for method chaining

                """
                self.config["pattern"] = pattern
                return self

            def build(
                self,
            ) -> FlextResult[FlextTypes.Fields.Instance]:
                """Build the field instance with configured parameters.

                Returns:
                    FlextResult with created field instance or error

                """
                return FlextFields.Factory.create_field(
                    self.field_type, self.name, **self.config
                )

    # ==========================================================================
    # METADATA SUBSYSTEM - Field metadata management and introspection
    # ==========================================================================

    class Metadata:
        """Field metadata management and introspection capabilities.

        This nested class provides metadata extraction, analysis, and
        introspection capabilities for field instances and schemas.
        """

        @staticmethod
        def analyze_field(
            field: FlextFields.Core.BaseField[object],
        ) -> FlextResult[dict[str, object]]:
            """Analyze field and extract comprehensive metadata.

            Args:
                field: Field instance to analyze

            Returns:
                FlextResult with field analysis data or error

            """
            # Type is already enforced by parameter annotation

            analysis = {
                "basic_metadata": field.get_metadata(),
                "field_class": field.__class__.__name__,
                "field_module": field.__class__.__module__,
                "capabilities": FlextFields.Metadata._analyze_field_capabilities(field),
                "constraints": {
                    "has_default": field.default is not None,
                    "default_value": field.default,
                    "is_required": field.required,
                    "description_available": bool(field.description),
                },
            }

            # Add type-specific analysis
            if isinstance(field, FlextFields.Core.StringField):
                analysis["string_constraints"] = {
                    "min_length": field.min_length,
                    "max_length": field.max_length,
                    "pattern": field.pattern.pattern if field.pattern else None,
                }
            elif isinstance(
                field, (FlextFields.Core.IntegerField, FlextFields.Core.FloatField)
            ):
                analysis["numeric_constraints"] = {
                    "min_value": getattr(field, "min_value", None),
                    "max_value": getattr(field, "max_value", None),
                    "precision": getattr(field, "precision", None),
                }

            # Cast analysis to object dict for FlextResult compatibility
            analysis_obj: dict[str, object] = analysis  # type: ignore[assignment]
            return FlextResult[dict[str, object]].ok(analysis_obj)

        @staticmethod
        def _analyze_field_capabilities(
            field: FlextFields.Core.BaseField[object],
        ) -> dict[str, bool]:
            """Analyze field capabilities and features.

            Args:
                field: Field instance to analyze

            Returns:
                Dictionary of capability flags

            """
            capabilities = {
                "validates_type": hasattr(field, "validate"),
                "has_metadata": hasattr(field, "get_metadata"),
                "supports_default": True,
                "supports_required": True,
                "supports_description": True,
            }

            # Type-specific capabilities
            if isinstance(field, FlextFields.Core.StringField):
                capabilities.update(
                    {
                        "supports_length_validation": True,
                        "supports_pattern_validation": field.pattern is not None,
                    }
                )
            elif isinstance(
                field, (FlextFields.Core.IntegerField, FlextFields.Core.FloatField)
            ):
                capabilities.update(
                    {
                        "supports_range_validation": True,
                        "supports_precision": isinstance(
                            field, FlextFields.Core.FloatField
                        ),
                    }
                )
            elif isinstance(field, FlextFields.Core.EmailField):
                capabilities["validates_email_format"] = True
            elif isinstance(field, FlextFields.Core.UuidField):
                capabilities["validates_uuid_format"] = True
            elif isinstance(field, FlextFields.Core.DateTimeField):
                capabilities.update(
                    {
                        "supports_date_format": True,
                        "supports_date_range": True,
                    }
                )

            return capabilities

        @staticmethod
        def get_field_summary(
            fields: list[FlextFields.Core.BaseField[object]],
        ) -> FlextResult[dict[str, object]]:
            """Generate summary of multiple fields.

            Args:
                fields: List of fields to summarize

            Returns:
                FlextResult with field summary data or error

            """
            # Type is already enforced by parameter annotation

            # Use properly typed variables
            field_types: dict[str, int] = {}
            required_fields = 0
            optional_fields = 0
            fields_with_defaults = 0
            validation_capabilities: set[str] = set()

            for field in fields:
                # Count by type
                field_type = field.field_type
                field_types[field_type] = field_types.get(field_type, 0) + 1

                # Count required/optional
                if field.required:
                    required_fields += 1
                else:
                    optional_fields += 1

                # Count defaults
                if field.default is not None:
                    fields_with_defaults += 1

                # Collect capabilities
                capabilities = FlextFields.Metadata._analyze_field_capabilities(field)
                for capability, enabled in capabilities.items():
                    if enabled:
                        validation_capabilities.add(capability)

            # Build result dictionary with proper typing
            summary: dict[str, object] = {
                "total_fields": len(fields),
                "field_types": field_types,
                "required_fields": required_fields,
                "optional_fields": optional_fields,
                "fields_with_defaults": fields_with_defaults,
                "validation_capabilities": list(validation_capabilities),
            }

            return FlextResult[dict[str, object]].ok(summary)


# =============================================================================
# CONVENIENCE FUNCTIONS - Module-level functions for common operations
# =============================================================================


def create_field(
    field_type: str,
    name: str,
    **config: object,
) -> FlextResult[FlextFields.Core.BaseField[object]]:
    """Create field instance using factory pattern.

    Args:
        field_type: Type of field to create
        name: Name for the field
        **config: Configuration parameters for the field

    Returns:
        FlextResult with created field instance or error

    """
    result = FlextFields.Factory.create_field(field_type, name, **config)
    if result.success:
        return FlextResult[FlextFields.Core.BaseField[object]].ok(
            cast("FlextFields.Core.BaseField[object]", result.value)
        )
    return FlextResult[FlextFields.Core.BaseField[object]].fail(
        result.error or "Unknown error", error_code=result.error_code
    )


def validate_field_value(
    field: FlextFields.Core.BaseField[object],
    value: object,
) -> FlextResult[object]:
    """Validate value using field's validation logic.

    Args:
        field: Field instance to use for validation
        value: Value to validate

    Returns:
        FlextResult with validated value or error

    """
    return FlextFields.Validation.validate_field(field, value)


class _RegistrySingleton:
    """Thread-safe singleton for field registry."""

    _instance: FlextFields.Registry.FieldRegistry | None = None

    @classmethod
    def get_instance(cls) -> FlextFields.Registry.FieldRegistry:
        if cls._instance is None:
            cls._instance = FlextFields.Registry.FieldRegistry()
        return cls._instance


def get_global_field_registry() -> FlextFields.Registry.FieldRegistry:
    """Get global field registry instance.

    Returns:
        Global field registry instance

    """
    return _RegistrySingleton.get_instance()


# =============================================================================
# BUILDER SHORTCUTS - Convenience functions for common field types
# =============================================================================


def string_field(name: str) -> FlextFields.Factory.FieldBuilder:
    """Create string field builder.

    Args:
        name: Field name

    Returns:
        String field builder

    """
    return FlextFields.Factory.FieldBuilder("string", name)


def integer_field(name: str) -> FlextFields.Factory.FieldBuilder:
    """Create integer field builder.

    Args:
        name: Field name

    Returns:
        Integer field builder

    """
    return FlextFields.Factory.FieldBuilder("integer", name)


def float_field(name: str) -> FlextFields.Factory.FieldBuilder:
    """Create float field builder.

    Args:
        name: Field name

    Returns:
        Float field builder

    """
    return FlextFields.Factory.FieldBuilder("float", name)


def boolean_field(name: str) -> FlextFields.Factory.FieldBuilder:
    """Create boolean field builder.

    Args:
        name: Field name

    Returns:
        Boolean field builder

    """
    return FlextFields.Factory.FieldBuilder("boolean", name)


def email_field(name: str) -> FlextFields.Factory.FieldBuilder:
    """Create email field builder.

    Args:
        name: Field name

    Returns:
        Email field builder

    """
    return FlextFields.Factory.FieldBuilder("email", name)


def uuid_field(name: str) -> FlextFields.Factory.FieldBuilder:
    """Create UUID field builder.

    Args:
        name: Field name

    Returns:
        UUID field builder

    """
    return FlextFields.Factory.FieldBuilder("uuid", name)


def datetime_field(name: str) -> FlextFields.Factory.FieldBuilder:
    """Create datetime field builder.

    Args:
        name: Field name

    Returns:
        Datetime field builder

    """
    return FlextFields.Factory.FieldBuilder("datetime", name)


# =============================================================================
# EXPORTS - Comprehensive field system
# =============================================================================

__all__: Final[list[str]] = [
    "FlextFields",  # ONLY main class exported
]
