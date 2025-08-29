"""FLEXT Fields - Comprehensive field definition and validation system with enterprise patterns.

Type-safe field validation, metadata management, and schema processing system using hierarchical
nested classes for domain organization, registry-based field management, and FlextResult
integration following Clean Architecture principles with zero circular dependencies.

Module Role in Architecture:
    FlextFields provides comprehensive field validation and metadata management for all
    FLEXT ecosystem components, organized in hierarchical layers from Core to Factory,
    enabling type-safe schema processing and railway-oriented validation patterns.

Classes and Methods:
    FlextFields:                            # Hierarchical field management system
        # Core Layer - Foundation Field Types:
        Core.StringField                            # String validation with length constraints
        Core.NumericField                           # Numeric validation with range constraints
        Core.EmailField                             # Email format validation
        Core.DateField                              # Date validation and parsing
        Core.TimeField                              # Time validation and parsing
        Core.DateTimeField                          # DateTime validation with timezone support
        Core.BooleanField                           # Boolean value validation
        Core.ListField                              # List validation with item constraints
        Core.DictField                              # Dictionary validation with key/value constraints

        # Validation Layer - Field Validation and Constraints:
        Validation.FieldValidator                   # Base field validator with constraint checking
        Validation.StringValidator                  # String-specific validation rules
        Validation.NumericValidator                 # Numeric range and constraint validation
        Validation.EmailValidator                   # Email format and domain validation
        Validation.ListValidator                    # List length and item validation
        Validation.DictValidator                    # Dictionary structure validation

        # Registry Layer - Field Registration and Management:
        Registry.FieldRegistry                      # Central field type registry
        Registry.register_field(name, field_type) -> FlextResult[None] # Register field type
        Registry.get_field(name) -> FlextResult[FieldType] # Retrieve registered field
        Registry.list_registered_fields() -> FlextResult[list[str]] # List all field names
        Registry.unregister_field(name) -> FlextResult[None] # Remove field registration

        # Schema Layer - Schema Processing and Metadata:
        Schema.SchemaProcessor                      # Schema validation and processing
        Schema.extract_field_metadata(field) -> dict # Extract field metadata
        Schema.validate_schema(schema_dict) -> FlextResult[dict] # Validate schema structure
        Schema.generate_field_docs(field) -> str   # Generate field documentation
        Schema.merge_schemas(*schemas) -> FlextResult[dict] # Merge multiple schemas

        # Factory Layer - Field Creation Patterns:
        Factory.FieldFactory                        # Factory for field creation
        Factory.create_string_field(name, **constraints) -> FlextResult[StringField] # Create string field
        Factory.create_numeric_field(name, **constraints) -> FlextResult[NumericField] # Create numeric field
        Factory.create_email_field(name) -> FlextResult[EmailField] # Create email field
        Factory.create_list_field(name, item_type) -> FlextResult[ListField] # Create list field
        Factory.create_dict_field(name, key_type, value_type) -> FlextResult[DictField] # Create dict field

        # Configuration Methods:
        configure_fields_system(config) -> FlextResult[ConfigDict] # Configure field system
        get_fields_system_config() -> FlextResult[ConfigDict] # Get current field config
        create_environment_fields_config(environment) -> FlextResult[ConfigDict] # Environment config
        optimize_fields_performance(performance_level) -> FlextResult[ConfigDict] # Performance optimization

Usage Examples:
    Basic field validation:
        string_field = FlextFields.Core.StringField(min_length=3, max_length=20)
        result = string_field.validate("username")
        if result.success:
            validated_value = result.value

    Field registry usage:
        registry = FlextFields.Registry.FieldRegistry()
        registry.register_field("email", FlextFields.Core.EmailField())
        field_result = registry.get_field("email")

    Schema processing:
        schema = {"username": "string", "age": "numeric", "email": "email"}
        result = FlextFields.Schema.validate_schema(schema)

    Configuration:
        config = {
            "environment": "production",
            "validation_level": "strict",
            "enable_field_caching": True,
        }
        FlextFields.configure_fields_system(config)

Integration:
    FlextFields integrates with FlextResult for error handling, FlextTypes.Config
    for configuration, FlextConstants for validation limits, providing comprehensive
    field validation and metadata management for the entire FLEXT ecosystem.

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
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Final, cast, override

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

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
                    if lower_value in {"true", "yes", "1", "on", "enabled"}:
                        return FlextResult[bool].ok(data=True)
                    if lower_value in {"false", "no", "0", "off", "disabled"}:
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
                    # Use centralized UUID generation
                    if self.default:
                        return FlextResult[str].ok(self.default)
                    return FlextResult[str].ok(
                        FlextUtilities.Generators.generate_uuid()
                    )

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
                    # Create a properly typed list for field definitions
                    field_definitions: list[object] = []

                    # Type cast the list to help Pyright inference
                    fields_data = cast("list[dict[str, object]]", fields_data_raw)
                    for field_def_raw in fields_data:
                        # field_def_raw is guaranteed dict[str, object] from cast
                        field_result = self._extract_field_definition(field_def_raw)
                        if field_result.success:
                            field_definitions.append(field_result.value)

                    # Assign the typed list to the processed schema
                    processed_schema["fields"] = field_definitions

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
                    processed_config["required"] = value.lower() in {
                        "true",
                        "yes",
                        "1",
                        "on",
                    }
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
                required = required.lower() in {"true", "yes", "1", "on"}
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
                        bool_default = default.lower() in {"true", "yes", "1", "on"}
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

            # Fallback for unknown field types - use field_class directly
            # Determine appropriate default conversion based on the field_class
            converted_default: object = None
            if default is not None:
                # Check the field class name to determine appropriate conversion
                field_class_name = getattr(field_class, "__name__", str(field_class))

                if (
                    "String" in field_class_name
                    or "Email" in field_class_name
                    or "Uuid" in field_class_name
                    or field_type.lower() == "string"
                    or "string" in field_type.lower()
                ):
                    # String-based fields
                    converted_default = (
                        default if isinstance(default, str) else str(default)
                    )
                elif (
                    "Integer" in field_class_name
                    or "Int" in field_class_name
                    or field_type.lower() == "integer"
                    or "int" in field_type.lower()
                ):
                    # Integer fields
                    converted_default = FlextUtilities.safe_int(default)
                elif (
                    "Float" in field_class_name
                    or field_type.lower() == "float"
                    or "float" in field_type.lower()
                ):
                    # Float fields
                    converted_default = FlextUtilities.Conversions.safe_float(default)
                elif (
                    "Boolean" in field_class_name
                    or "Bool" in field_class_name
                    or field_type.lower() == "boolean"
                    or "bool" in field_type.lower()
                ):
                    # Boolean fields
                    converted_default = FlextUtilities.safe_bool_conversion(default)
                else:
                    # Generic fallback - try string conversion as safest option
                    converted_default = (
                        default if isinstance(default, str) else str(default)
                    )

            return cast(
                "FlextTypes.Fields.Instance",
                field_class(
                    name,
                    required=required,
                    default=converted_default,  # type: ignore[arg-type]
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

            # Extract field definitions using the same pattern as process_field_schema
            schema_fields_raw = schema["fields"]
            if not isinstance(schema_fields_raw, list):
                return FlextResult[list[FlextFields.Core.BaseField[object]]].fail(
                    "Schema 'fields' must be a list",
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )

            # Type-narrowed after isinstance check - Pylance can now infer this is list[object]
            schema_fields = cast("list[dict[str, object]]", schema_fields_raw)

            for field_definition in schema_fields:
                # field_definition is now guaranteed to be dict[str, object] from cast
                field_def: dict[str, object] = field_definition
                field_type_obj = field_def.get("type")
                field_name_obj = field_def.get("name")

                # Checagem de None
                if field_type_obj is None or field_name_obj is None:
                    errors.append("Field definition must have 'type' and 'name'")
                    continue

                # Checagem de tipo
                if not isinstance(field_type_obj, str) or not isinstance(
                    field_name_obj, str
                ):
                    errors.append("Field 'type' and 'name' must be strings")
                    continue

                field_type: str = field_type_obj
                field_name: str = field_name_obj

                # Extração de configuração com tipos explícitos
                config: dict[str, object] = {
                    k: v for k, v in field_def.items() if k not in {"type", "name"}
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
                capabilities.update({
                    "supports_length_validation": True,
                    "supports_pattern_validation": field.pattern is not None,
                })
            elif isinstance(
                field, (FlextFields.Core.IntegerField, FlextFields.Core.FloatField)
            ):
                capabilities.update({
                    "supports_range_validation": True,
                    "supports_precision": isinstance(
                        field, FlextFields.Core.FloatField
                    ),
                })
            elif isinstance(field, FlextFields.Core.EmailField):
                capabilities["validates_email_format"] = True
            elif isinstance(field, FlextFields.Core.UuidField):
                capabilities["validates_uuid_format"] = True
            elif isinstance(field, FlextFields.Core.DateTimeField):
                capabilities.update({
                    "supports_date_format": True,
                    "supports_date_range": True,
                })

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

    # =========================================================================
    # CONFIGURATION MANAGEMENT - FLEXT TYPES INTEGRATION
    # =========================================================================

    @classmethod
    def configure_fields_system(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure fields system using FlextTypes.Config with StrEnum validation."""
        try:
            # Validate environment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )

            # Validate log level
            if "log_level" in config:
                log_value = config["log_level"]
                valid_log_levels = [
                    level.value for level in FlextConstants.Config.LogLevel
                ]
                if log_value not in valid_log_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_value}'. Valid options: {valid_log_levels}"
                    )

            # Validate validation level
            if "validation_level" in config:
                val_value = config["validation_level"]
                valid_validation_levels = [
                    v.value for v in FlextConstants.Config.ValidationLevel
                ]
                if val_value not in valid_validation_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid validation_level '{val_value}'. Valid options: {valid_validation_levels}"
                    )

            # Build validated configuration with defaults
            validated_config: FlextTypes.Config.ConfigDict = {
                "environment": config.get(
                    "environment",
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                ),
                "log_level": config.get(
                    "log_level", FlextConstants.Config.LogLevel.DEBUG.value
                ),
                "validation_level": config.get(
                    "validation_level",
                    FlextConstants.Config.ValidationLevel.NORMAL.value,
                ),
                "enable_field_validation": config.get("enable_field_validation", True),
                "enable_type_checking": config.get("enable_type_checking", True),
                "enable_constraint_validation": config.get(
                    "enable_constraint_validation", True
                ),
                "max_field_cache_size": config.get("max_field_cache_size", 500),
                "enable_field_metadata": config.get("enable_field_metadata", True),
                "enable_schema_validation": config.get(
                    "enable_schema_validation", True
                ),
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Fields system configuration failed: {e}"
            )

    @classmethod
    def get_fields_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current fields system configuration with runtime information."""
        try:
            config: FlextTypes.Config.ConfigDict = {
                # Current system state
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                # Fields-specific settings
                "enable_field_validation": True,
                "enable_type_checking": True,
                "enable_constraint_validation": True,
                "max_field_cache_size": 500,
                "enable_field_metadata": True,
                "enable_schema_validation": True,
                # Runtime metrics
                "registered_field_types": 7,  # string, integer, float, boolean, email, uuid, datetime
                "active_field_instances": 0,  # Would be tracked in registry
                "validation_errors_count": 0,  # Would be tracked in validation
                "schema_processing_count": 0,  # Would be tracked in schema processor
                # Available field types
                "available_field_types": [
                    "string",
                    "integer",
                    "float",
                    "boolean",
                    "email",
                    "uuid",
                    "datetime",
                    "base",
                ],
                # Field capabilities
                "supported_constraints": [
                    "required",
                    "default",
                    "min_length",
                    "max_length",
                    "min_value",
                    "max_value",
                    "pattern",
                    "precision",
                    "date_format",
                    "min_date",
                    "max_date",
                ],
                # Performance settings
                "enable_field_caching": True,
                "cache_validation_results": False,
                "enable_performance_monitoring": False,
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get fields system configuration: {e}"
            )

    @classmethod
    def create_environment_fields_config(
        cls, environment: FlextTypes.Config.Environment
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific fields system configuration."""
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            # Base configuration
            config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
            }

            # Environment-specific settings
            if environment == "production":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "enable_field_validation": True,  # Strict validation in production
                    "enable_type_checking": True,  # Type checking for safety
                    "enable_constraint_validation": True,  # All constraints in production
                    "max_field_cache_size": 1000,  # Larger cache for production
                    "enable_field_metadata": False,  # Minimal metadata for performance
                    "enable_schema_validation": True,  # Schema validation for safety
                    "cache_validation_results": True,  # Cache for performance
                    "enable_performance_monitoring": True,  # Performance monitoring
                })
            elif environment == "development":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "enable_field_validation": True,  # Validation for development
                    "enable_type_checking": True,  # Type checking for catching issues
                    "enable_constraint_validation": True,  # Full constraints for development
                    "max_field_cache_size": 200,  # Smaller cache for development
                    "enable_field_metadata": True,  # Full metadata for debugging
                    "enable_schema_validation": True,  # Schema validation for development
                    "cache_validation_results": False,  # No caching for fresh results
                    "enable_detailed_error_messages": True,  # Detailed errors for debugging
                })
            elif environment == "test":
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.ERROR.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "enable_field_validation": True,  # Strict validation for tests
                    "enable_type_checking": True,  # Type checking for test accuracy
                    "enable_constraint_validation": True,  # Full constraints for tests
                    "max_field_cache_size": 50,  # Minimal cache for tests
                    "enable_field_metadata": True,  # Metadata for test inspection
                    "enable_schema_validation": True,  # Schema validation for tests
                    "cache_validation_results": False,  # No caching in tests
                    "enable_test_utilities": True,  # Special test utilities
                })
            else:  # staging, local, etc.
                config.update({
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "enable_field_validation": True,  # Standard validation
                    "enable_type_checking": True,  # Standard type checking
                    "enable_constraint_validation": True,  # Standard constraints
                    "max_field_cache_size": 500,  # Standard cache size
                    "enable_field_metadata": True,  # Standard metadata
                    "enable_schema_validation": True,  # Standard schema validation
                })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment fields config: {e}"
            )

    @classmethod
    def optimize_fields_performance(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize fields system performance based on configuration."""
        try:
            # Extract performance level or determine from config
            performance_level = config.get("performance_level", "medium")

            # Base optimization settings
            optimized_config: FlextTypes.Config.ConfigDict = {
                "performance_level": performance_level,
                "optimization_enabled": True,
            }

            # Performance level specific optimizations
            if performance_level == "high":
                optimized_config.update({
                    "max_field_cache_size": config.get("max_field_cache_size", 2000),
                    "enable_field_caching": True,
                    "cache_validation_results": True,
                    "enable_lazy_validation": True,
                    "batch_validation_size": 100,
                    "enable_concurrent_validation": True,
                    "memory_optimization": "aggressive",
                    "enable_field_pooling": True,
                    "precompile_constraints": True,
                })
            elif performance_level == "medium":
                optimized_config.update({
                    "max_field_cache_size": config.get("max_field_cache_size", 1000),
                    "enable_field_caching": True,
                    "cache_validation_results": False,
                    "enable_lazy_validation": False,
                    "batch_validation_size": 50,
                    "enable_concurrent_validation": False,
                    "memory_optimization": "balanced",
                    "enable_field_pooling": False,
                    "precompile_constraints": False,
                })
            elif performance_level == "low":
                optimized_config.update({
                    "max_field_cache_size": config.get("max_field_cache_size", 200),
                    "enable_field_caching": False,
                    "cache_validation_results": False,
                    "enable_lazy_validation": False,
                    "batch_validation_size": 10,
                    "enable_concurrent_validation": False,
                    "memory_optimization": "conservative",
                    "enable_field_pooling": False,
                    "precompile_constraints": False,
                })
            else:
                # Default/custom performance level
                optimized_config.update({
                    "max_field_cache_size": config.get("max_field_cache_size", 500),
                    "enable_field_caching": config.get("enable_field_caching", True),
                    "memory_optimization": "balanced",
                })

            # Merge with original config
            optimized_config.update({
                key: value
                for key, value in config.items()
                if key not in optimized_config
            })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Fields performance optimization failed: {e}"
            )

    # =============================================================================
    # LEGACY COMPATIBILITY METHODS - Support for old API patterns
    # =============================================================================

    @classmethod
    def create_string_field(cls, **kwargs: object) -> FlextResult[object]:
        """Create string field with validation - legacy compatibility method."""
        try:
            # Convert kwargs to proper types for StringField
            name = str(kwargs.get("name", ""))
            required = bool(kwargs.get("required", True))
            default = (
                str(kwargs["default"]) if kwargs.get("default") is not None else None
            )
            description = str(kwargs.get("description", ""))
            min_length = (
                FlextUtilities.safe_int(kwargs["min_length"])
                if kwargs.get("min_length") is not None
                else None
            )

            field = cls.Core.StringField(
                name=name,
                required=required,
                default=default,
                description=description,
                min_length=min_length,
            )
            return FlextResult[object].ok(field)
        except Exception as e:
            return FlextResult[object].fail(f"Failed to create string field: {e}")

    @classmethod
    def create_integer_field(cls, **kwargs: object) -> FlextResult[object]:
        """Create integer field with validation - legacy compatibility method."""
        try:
            # Convert kwargs to proper types for IntegerField
            name = str(kwargs.get("name", ""))
            required = bool(kwargs.get("required", True))
            default = (
                FlextUtilities.safe_int(kwargs["default"])
                if kwargs.get("default") is not None
                else None
            )
            description = str(kwargs.get("description", ""))
            min_value = (
                FlextUtilities.safe_int(kwargs["min_value"])
                if kwargs.get("min_value") is not None
                else None
            )

            field = cls.Core.IntegerField(
                name=name,
                required=required,
                default=default,
                description=description,
                min_value=min_value,
            )
            return FlextResult[object].ok(field)
        except Exception as e:
            return FlextResult[object].fail(f"Failed to create integer field: {e}")

    @classmethod
    def create_boolean_field(cls, **kwargs: object) -> FlextResult[object]:
        """Create boolean field with validation - legacy compatibility method."""
        try:
            # Convert kwargs to proper types for BooleanField
            name = str(kwargs.get("name", ""))
            required = bool(kwargs.get("required", True))
            default = (
                FlextUtilities.safe_bool_conversion(kwargs["default"])
                if kwargs.get("default") is not None
                else None
            )
            description = str(kwargs.get("description", ""))

            field = cls.Core.BooleanField(
                name=name,
                required=required,
                default=default,
                description=description,
            )
            return FlextResult[object].ok(field)
        except Exception as e:
            return FlextResult[object].fail(f"Failed to create boolean field: {e}")

    @property
    def string_field(self) -> type:
        """Legacy compatibility property for string field access."""
        return self.Core.StringField

    @property
    def integer_field(self) -> type:
        """Legacy compatibility property for integer field access."""
        return self.Core.IntegerField

    @property
    def boolean_field(self) -> type:
        """Legacy compatibility property for boolean field access."""
        return self.Core.BooleanField


# =============================================================================
# COMPATIBILITY FUNCTIONS - Legacy support for test code
# =============================================================================


def flext_create_string_field(**kwargs: object) -> FlextResult[object]:
    """Create string field with validation - compatibility function."""
    try:
        # Convert kwargs to proper types for StringField
        name = str(kwargs.get("name", ""))
        required = bool(kwargs.get("required", True))
        default = str(kwargs["default"]) if kwargs.get("default") is not None else None
        description = str(kwargs.get("description", ""))
        min_length = (
            FlextUtilities.safe_int(kwargs["min_length"])
            if kwargs.get("min_length") is not None
            else None
        )

        field = FlextFields.Core.StringField(
            name=name,
            required=required,
            default=default,
            description=description,
            min_length=min_length,
        )
        return FlextResult[object].ok(field)
    except Exception as e:
        return FlextResult[object].fail(f"Failed to create string field: {e}")


def flext_create_integer_field(**kwargs: object) -> FlextResult[object]:
    """Create integer field with validation - compatibility function."""
    try:
        # Convert kwargs to proper types for IntegerField
        name = str(kwargs.get("name", ""))
        required = bool(kwargs.get("required", True))
        default = (
            FlextUtilities.safe_int(kwargs["default"])
            if kwargs.get("default") is not None
            else None
        )
        description = str(kwargs.get("description", ""))
        min_value = (
            FlextUtilities.safe_int(kwargs["min_value"])
            if kwargs.get("min_value") is not None
            else None
        )

        field = FlextFields.Core.IntegerField(
            name=name,
            required=required,
            default=default,
            description=description,
            min_value=min_value,
        )
        return FlextResult[object].ok(field)
    except Exception as e:
        return FlextResult[object].fail(f"Failed to create integer field: {e}")


def flext_create_boolean_field(**kwargs: object) -> FlextResult[object]:
    """Create boolean field with validation - compatibility function."""
    try:
        # Convert kwargs to proper types for BooleanField
        name = str(kwargs.get("name", ""))
        required = bool(kwargs.get("required", True))
        default = (
            FlextUtilities.safe_bool_conversion(kwargs["default"])
            if kwargs.get("default") is not None
            else None
        )
        description = str(kwargs.get("description", ""))

        field = FlextFields.Core.BooleanField(
            name=name,
            required=required,
            default=default,
            description=description,
        )
        return FlextResult[object].ok(field)
    except Exception as e:
        return FlextResult[object].fail(f"Failed to create boolean field: {e}")


# Compatibility wrapper for legacy field API
class _FlextFieldCoreCompat(FlextFields.Core.BaseField[object]):
    """Compatibility wrapper for legacy FlextFieldCore API."""

    def __init__(
        self,
        field_id: str = "default",
        field_name: str = "default",
        field_type: str = "string",
        **kwargs: str | float | bool | None,
    ) -> None:
        # Map old API to new API parameters
        name = str(kwargs.get("name", field_name))
        required = bool(kwargs.get("required", True))
        default = kwargs.get("default")
        description = str(
            kwargs.get("description", f"Field {field_name} of type {field_type}")
        )

        super().__init__(
            name=name,
            required=required,
            default=default,
            description=description,
        )

        # Store legacy attributes for compatibility
        self._legacy_field_id = field_id
        self._legacy_field_name = field_name
        self._legacy_field_type = field_type

        # Store validation constraints for compatibility
        min_length_val = kwargs.get("min_length")
        self._min_length: int | None = (
            FlextUtilities.safe_int(min_length_val)
            if min_length_val is not None
            else None
        )

        max_length_val = kwargs.get("max_length")
        self._max_length: int | None = (
            FlextUtilities.safe_int(max_length_val)
            if max_length_val is not None
            else None
        )

        min_value_val = kwargs.get("min_value")
        self._min_value: int | float | None = (
            FlextUtilities.Conversions.safe_float(min_value_val)
            if min_value_val is not None
            else None
        )

        max_value_val = kwargs.get("max_value")
        self._max_value: int | float | None = (
            FlextUtilities.Conversions.safe_float(max_value_val)
            if max_value_val is not None
            else None
        )

        allowed_values_val = kwargs.get("allowed_values")
        self._allowed_values: list[object] | None = None
        if allowed_values_val is not None:
            # Check if it's a list or tuple using try/except to avoid type narrowing issues
            try:
                if hasattr(allowed_values_val, "__iter__") and not isinstance(
                    allowed_values_val, (str, bytes)
                ):
                    # It's an iterable that's not a string - convert to list
                    # Cast to iterable after checking __iter__ exists
                    self._allowed_values = list(cast("Iterable[object]", allowed_values_val))
            except (TypeError, ValueError):
                # If conversion fails, leave as None
                pass

        # Store kwargs for property access
        self._kwargs = kwargs

    @property
    def field_id(self) -> str:
        """Legacy field_id property."""
        return self._legacy_field_id

    @property
    def field_name(self) -> str:
        """Legacy field_name property."""
        return self._legacy_field_name

    @property
    def min_length(self) -> int | None:
        """Legacy min_length property."""
        return self._min_length

    @property
    def max_length(self) -> int | None:
        """Legacy max_length property."""
        return self._max_length

    @property
    def min_value(self) -> int | float | None:
        """Legacy min_value property."""
        return self._min_value

    @property
    def max_value(self) -> int | float | None:
        """Legacy max_value property."""
        return self._max_value

    @property
    def allowed_values(self) -> list[object] | None:
        """Legacy allowed_values property."""
        return self._allowed_values

    def validate(self, value: object) -> FlextResult[object]:
        """Enhanced validate implementation with basic constraint checking."""
        # String length validation
        if self._legacy_field_type == "string" and isinstance(value, str):
            if self._min_length is not None and len(value) < self._min_length:
                return FlextResult[object].fail(
                    f"Value length {len(value)} is below minimum {self._min_length}"
                )
            if self._max_length is not None and len(value) > self._max_length:
                return FlextResult[object].fail(
                    f"Value length {len(value)} is above maximum {self._max_length}"
                )

        # Numeric value validation
        if self._legacy_field_type == "integer" and isinstance(value, (int, float)):
            if self._min_value is not None and value < self._min_value:
                return FlextResult[object].fail(
                    f"Value {value} is below minimum {self._min_value}"
                )
            if self._max_value is not None and value > self._max_value:
                return FlextResult[object].fail(
                    f"Value {value} is above maximum {self._max_value}"
                )

        # Allowed values validation
        if self._allowed_values is not None and value not in self._allowed_values:
            return FlextResult[object].fail(
                f"Value {value} is not in allowed values {self._allowed_values}"
            )

        return FlextResult[object].ok(value)

    def validate_value(self, value: object) -> object:
        """Legacy validate_value method - maps to validate."""
        return self.validate(value)


# Compatibility aliases for test code
FlextFieldCore = _FlextFieldCoreCompat
# Use FlextFields.Metadata and FlextFields.Registry directly - no aliases


# Note: Legacy factory functions have been consolidated into FlextFields class methods
# Use FlextFields.create_string_field, FlextFields.create_integer_field, etc. instead


# Enum-like object for field types with .value attributes for backward compatibility
class _FlextFieldTypeEnum:
    """Enum-like object providing backward compatibility for field types."""

    class STRING:
        value = "string"

    class INTEGER:
        value = "integer"

    class BOOLEAN:
        value = "boolean"

    class FLOAT:
        value = "float"

    class EMAIL:
        value = "email"

    class UUID:
        value = "uuid"

    class DATETIME:
        value = "datetime"


FlextFieldType = _FlextFieldTypeEnum()


# =============================================================================
# EXPORTS - Comprehensive field system
# =============================================================================

__all__: Final[list[str]] = [
    "FlextFields",  # ONLY main class exported
    # Legacy compatibility aliases moved to flext_core.legacy to avoid type conflicts
    # Import from flext_core.legacy if you need FlextFieldRegistry, flext_create_*_field, etc.
]
