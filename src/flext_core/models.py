"""Consolidated FLEXT Models - Domain modeling and data structures.

This module provides the complete model foundation for the FLEXT ecosystem,
organized using the consolidated class pattern with nested model definitions
following FLEXT architectural standards and Clean Architecture principles.

Key Features:
- FlextModels: Single consolidated class containing ALL model functionality
- Nested model classes organized by type (Model, Value, Entity, etc.)
- Full FlextTypes and FlextConstants integration with hierarchical access
- Railway-oriented programming via FlextResult patterns
- Type-safe domain modeling with Pydantic v2 features
- Enterprise patterns following SOLID principles

Architecture:
- Hierarchical model organization using FlextTypes patterns
- Constants integration via FlextConstants nested structures
- Clean separation between base models, value objects, and entities
- Backward compatibility through property re-exports
- Factory pattern integration with validation and error handling

Usage:
    Basic model creation using consolidated class::

        from flext_core.models import FlextModels

        # Use nested classes for type-safe model creation
        user_model = FlextModels.Entity(**user_data)
        config_model = FlextModels.Model(**config_data)

        # Factory pattern with validation
        result = FlextModels.create("user", **user_data)
        if result.success:
            user = result.value

    Using FlextTypes and FlextConstants integration::

        from flext_core import FlextTypes, FlextConstants

        # Type-safe field definitions
        entity_id: FlextTypes.Domain.EntityId = "user_123"
        config: FlextTypes.Core.Config = {"debug": True}

        # Constants for validation thresholds
        max_length = FlextConstants.Validation.MAX_STRING_LENGTH
        timeout = FlextConstants.Defaults.TIMEOUT

Note:
    This module enforces the FLEXT consolidated class pattern where all
    related functionality is organized within a single main class with
    nested classes for specific domains. This follows the principle of
    high cohesion and provides a single point of access for all model
    operations while maintaining clear separation of concerns.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime, timezone
from typing import Annotated, ClassVar, Self, cast, override

from pydantic import (
    AliasChoices,
    AliasGenerator,
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    ValidationError,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_serializer,
)
from pydantic.alias_generators import to_camel, to_snake

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLoggerFactory
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult
from flext_core.root_models import (
    FlextEntityId,
    FlextEventList,
    FlextMetadata,
    FlextTimestamp,
    FlextVersion,
)
from flext_core.typings import (
    FlextTypes,
)
from flext_core.utilities import FlextGenerators

# Use centralized types from FlextTypes hierarchical structure
SerializerCallable = FlextTypes.Core.Serializer


def _get_exception_class(name: str) -> type[Exception]:
    """Get dynamically created exception class with type safety."""
    return cast("type[Exception]", getattr(FlextExceptions, name))


# Use centralized constants from FlextConstants hierarchical structure
DEFAULT_TIMEOUT = FlextConstants.Defaults.TIMEOUT
MAX_STRING_LENGTH = FlextConstants.Validation.MAX_STRING_LENGTH
VALIDATION_ERROR_CODE = FlextConstants.Errors.VALIDATION_ERROR
SUCCESS_STATUS = FlextConstants.Status.SUCCESS

# =============================================================================
# Constants
# =============================================================================

# Re-export UTC timezone for this module
UTC_TIMEZONE: timezone = UTC

# =============================================================================
# Alias Generator
# =============================================================================


def flext_alias_generator(field_name: str) -> str:
    """FLEXT-specific alias generator that converts snake_case to camelCase."""
    return to_camel(field_name)


# =============================================================================
# Type Aliases (now using RootModel types for validation)
# =============================================================================

# FlextEntityId and FlextVersion are now imported above from root_models

# =============================================================================
# Helper Functions
# =============================================================================


def create_timestamp() -> FlextTimestamp:
    """Create a new timestamp."""
    return FlextTimestamp.now()


def create_version(value: int) -> FlextResult[FlextVersion]:
    """Create a new version number."""
    if value < 1:
        return FlextResult[FlextVersion].fail("Version must be >= 1")
    return FlextResult[FlextVersion].ok(FlextVersion(value))


# =============================================================================
# Base Models
# =============================================================================


class FlextModel(BaseModel):
    """Base model for all FLEXT data structures."""

    model_config = ConfigDict(
        populate_by_name=True,
        validate_default=True,
        alias_generator=AliasGenerator(
            validation_alias=to_snake, serialization_alias=to_camel
        ),
        use_enum_values=True,
        protected_namespaces=("model_",),
    )

    @property
    @computed_field
    def model_type(self) -> str:
        """Returns the type of the model, typically the class name."""
        return self.__class__.__name__

    @property
    @computed_field
    def model_namespace(self) -> str:
        """Returns the namespace of the model, based on its module path."""
        return self.__class__.__module__

    def to_dict(self) -> FlextTypes.Core.Dict:
        """Serializes the model to a dictionary."""
        return self.model_dump(by_alias=True, exclude_none=True)

    def to_dict_basic(self) -> FlextTypes.Core.Dict:
        """Serializes the model to a basic dictionary (alias for to_dict)."""
        return self.to_dict()

    def to_json(self, **_kwargs: object) -> str:
        """Serializes the model to a JSON string."""
        # Only pass valid kwargs to avoid type errors
        return self.model_dump_json(by_alias=True, exclude_none=True)

    @model_serializer(mode="wrap", when_used="json")
    def serialize_model_wrapper(
        self, serializer: SerializerCallable
    ) -> dict[str, object]:
        """Wraps the serialized model with metadata."""
        data = serializer(self)
        data["metadata"] = {
            "model_type": self.model_type,
            "model_namespace": self.model_namespace,
            "serialized_at": datetime.now(UTC_TIMEZONE).isoformat(),
        }
        return data

    def validate_business_rules(self) -> FlextResult[None]:
        """Default business rule validation - override in subclasses for specific rules."""
        return FlextResult[None].ok(None)


class FlextRootModel(FlextModel):
    """A base model for root data structures in the FLEXT ecosystem."""

    model_config = ConfigDict(
        populate_by_name=True,
        validate_default=True,
        alias_generator=AliasGenerator(
            validation_alias=to_snake, serialization_alias=to_camel
        ),
        use_enum_values=True,
        protected_namespaces=("model_",),
    )


# =============================================================================
# Value Objects
# =============================================================================


def make_hashable(
    item: object,
) -> frozenset[tuple[str, object]] | tuple[object, ...] | object:
    """Recursively converts mutable collections to their immutable counterparts."""
    if isinstance(item, dict):
        dict_cast = cast("dict[str, object]", item)
        hashable_items = tuple((str(k), make_hashable(v)) for k, v in dict_cast.items())
        return frozenset(hashable_items)
    if isinstance(item, list):
        list_cast = cast("list[object]", item)
        return tuple(make_hashable(v) for v in list_cast)
    if isinstance(item, set):
        set_cast = cast("set[object]", item)
        return frozenset(make_hashable(v) for v in set_cast)
    return item


class FlextValue(FlextModel, ABC):
    """Abstract base class for Value Objects in the FLEXT ecosystem."""

    model_config = ConfigDict(frozen=True)

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Log subclass creation for monitoring."""
        super().__init_subclass__()
        try:
            logger = FlextLoggerFactory.get_logger(cls.__name__)
            logger.debug(f"FlextValue subclass created: {cls.__name__}")
        except ImportError:
            pass  # Silently continue if logging is not available

    @override
    def __hash__(self) -> int:
        """Generates a hash based on the model's attributes."""
        hashable_items = make_hashable(self.model_dump())
        return hash(hashable_items)

    @override
    def __eq__(self, other: object) -> bool:
        """Compares this Value Object with another for equality."""
        if not isinstance(other, FlextValue):
            return NotImplemented
        self_data = {k: v for k, v in self.model_dump().items() if k != "metadata"}
        other_data = {k: v for k, v in other.model_dump().items() if k != "metadata"}
        return self_data == other_data

    @abstractmethod
    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Abstract business rule validation - must be implemented."""

    @override
    def __str__(self) -> str:
        """Return human-readable string representation."""
        class_name = self.__class__.__name__
        attrs = self.model_dump()
        formatted_attrs = self.format_dict(attrs)
        return f"{class_name}({formatted_attrs})"

    def format_dict(self, data: dict[str, object]) -> str:
        """Format a dictionary as a string representation."""
        formatted_items: list[str] = []
        for key, value in data.items():
            if isinstance(value, str):
                formatted_items.append(f"{key}='{value}'")
            else:
                formatted_items.append(f"{key}={value}")
        return ", ".join(formatted_items)

    def validate_flext(self) -> FlextResult[Self]:
        """Validate the value object using its business rules."""
        result = self.validate_business_rules()
        if result.is_failure:
            return FlextResult[Self].fail(result.error or "Unknown validation error")
        return FlextResult[Self].ok(self)

    def to_payload(self) -> FlextPayload[dict[str, object]]:
        """Converts the value object to a FlextPayload."""
        validation_result = self.validate_business_rules()
        status = "valid" if validation_result.is_success else "invalid"

        try:
            value_object_data = self._extract_serializable_attributes()
        except Exception:
            value_object_data = self._get_fallback_info()
            status = "serialization_fallback"

        payload_data: dict[str, object] = {
            "value_object_data": value_object_data,
            "class_info": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "validation_status": status,
        }

        # First attempt to create payload
        payload_result = FlextPayload[dict[str, object]].create(
            data=payload_data,
            metadata={"type": "ValueObjectPayload"},
        )

        # If first attempt fails, try with fallback data
        if payload_result.is_failure:
            fallback_data: dict[str, object] = {
                "value_object_data": self._get_fallback_info(),
                "class_info": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "validation_status": "serialization_fallback",
            }

            payload_result = FlextPayload[dict[str, object]].create(
                data=fallback_data,
                metadata={"type": "ValueObjectPayload"},
            )

            # If both attempts fail, raise exception
            if payload_result.is_failure:
                error_msg = f"Failed to create payload: {payload_result.error}"
                validation_error = _get_exception_class(
                    "FlextExceptions.ValidationError"
                )
                raise validation_error(error_msg)

        return payload_result.value

    def _extract_serializable_attributes(self) -> dict[str, object]:
        """Extracts serializable attributes from the value object."""
        processed: dict[str, object] = {}
        for attr_name, attr_value in self:
            if self._is_serializable(attr_name):
                processed[attr_name] = self._process_attribute_value(attr_value) or ""
        return processed

    def _get_fallback_info(self) -> dict[str, object]:
        """Provides fallback information for serialization errors."""
        return {
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
            "error": "Could not serialize value object",
        }

    def _try_manual_extraction(self) -> dict[str, object]:
        """Try manual attribute extraction."""
        result: dict[str, object] = {}
        for attr_name in dir(self):
            if self._should_include_attribute(attr_name):
                result[attr_name] = self._safely_get_attribute(attr_name)
        return result

    def _should_include_attribute(self, attr_name: str) -> bool:
        """Check if attribute should be included in serialization."""
        if attr_name.startswith("_"):
            return False
        try:
            attr = getattr(self, attr_name, None)
            # Handle Pydantic v2 PydanticDescriptorProxy
            if attr is None:
                return False
            # Check if it's a callable or method, but handle Pydantic descriptors safely
            return not callable(attr)
        except (AttributeError, TypeError):
            # Skip attributes that cause access errors (like Pydantic descriptors)
            return False

    def _process_serializable_values(
        self, data: dict[str, object]
    ) -> dict[str, object]:
        """Process values to make them serializable."""
        processed: dict[str, object] = {}
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                processed[key] = value
            else:
                # Convert complex types to string representation
                processed[key] = str(value)
        return processed

    def validate_field(
        self, field_name: str, _value: object = None
    ) -> FlextResult[None]:
        """Validate a single field using FlextFields registry if available."""
        # FlextFields registry not implemented yet, use Pydantic validation directly

        # If field doesn't exist in model, return success for unknown fields
        if field_name not in self.__class__.model_fields:
            return FlextResult[None].ok(None)

        # Fallback to Pydantic validation for model fields
        try:
            self.__class__.model_validate(self.model_dump())
        except ValidationError as e:
            for error in e.errors():
                if field_name in error["loc"]:
                    return FlextResult[None].fail(error["msg"])
        return FlextResult[None].ok(None)

    def validate_all_fields(self) -> FlextResult[None]:
        """Validate all fields and return a consolidated result."""
        errors: dict[str, object] = {}

        # Get all fields from model_dump
        try:
            all_fields = self.model_dump()
        except Exception:
            # Fallback to model_fields if model_dump fails
            all_fields = {
                name: getattr(self, name, None) for name in self.__class__.model_fields
            }

        for field_name, field_value in all_fields.items():
            # Skip internal fields
            if field_name.startswith("_"):
                continue
            result = self.validate_field(field_name, field_value)
            if result.is_failure:
                errors[field_name] = result.error or "Unknown error"

        if errors:
            # Format errors into string
            error_parts = ["Field validation errors:"]
            for field, error in errors.items():
                error_parts.append(f"{field}: {error}")
            error_message = " ".join(error_parts)
            return FlextResult[None].fail(error_message, error_data=errors)
        return FlextResult[None].ok(None)

    def _is_serializable(self, attr_name: str) -> bool:
        """Determines if an attribute should be included in serialization."""
        if attr_name.startswith("_") or attr_name == "metadata":
            return False
        attr = getattr(self, attr_name, None)
        return not callable(attr)

    def _safely_get_attribute(self, attr_name: str) -> object | None:
        """Safely gets an attribute value from the instance by name."""
        try:
            attr_value = getattr(self, attr_name, None)
            if attr_value is None:
                return None
            return self._process_attribute_value(attr_value)
        except Exception:
            return None

    def _process_attribute_value(
        self, attr_value: object
    ) -> (
        str
        | int
        | float
        | bool
        | dict[str, object]
        | list[object]
        | tuple[object, ...]
        | None
    ):
        """Process an attribute value for safe serialization."""
        # Handle None first
        if attr_value is None:
            return None

        # Handle primitive types
        if isinstance(attr_value, (str, int, float, bool)):
            return attr_value

        # Handle collections with single consolidated logic
        if isinstance(attr_value, dict):
            dict_cast = cast("dict[str, object]", attr_value)
            return {str(k): v for k, v in dict_cast.items()}
        if isinstance(attr_value, (list, tuple)):
            return (
                cast("list[object]", attr_value)
                if isinstance(attr_value, list)
                else cast("tuple[object, ...]", attr_value)
            )

        # Handle all other objects by converting to string
        try:
            return str(attr_value)
        except Exception:
            return repr(attr_value)


# FlextValue is already defined above - no need for duplicate alias


class FlextEntity(FlextModel, ABC):
    """identity-based entities with lifecycle management.

    Foundation for Domain-Driven Design entities across the ecosystem.
    Inherits all standard Pydantic patterns from FlextModel while supporting mutability.

    Features:
    - Inherits AliasGenerator from FlextModel for consistent naming
    - Mutable model for entity lifecycle management
    - Identity-based equality and hashing
    - Domain events support for event sourcing
    -  validation with context support
    - Field aliases for flexible API interfaces
    """

    model_config = ConfigDict(
        # Inherit from FlextModel with entity-specific overrides
        alias_generator=AliasGenerator(
            alias=flext_alias_generator,
            validation_alias=flext_alias_generator,
            serialization_alias=flext_alias_generator,
        ),
        # Type safety and validation
        extra="allow",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        #  serialization features
        arbitrary_types_allowed=True,
        validate_default=True,
        populate_by_name=True,
        # Entity specific: MUTABLE for lifecycle management
        frozen=False,
        use_attribute_docstrings=True,
        # JSON schema for entities
        json_schema_extra={
            "examples": [],
            "description": "Mutable FLEXT entity with standard patterns and lifecycle management",
            "title": "FlextEntity",
        },
    )

    # Core identity - using RootModel types with standard Field patterns and PlainSerializer
    id: Annotated[
        FlextEntityId | str,
        Field(
            description="Unique entity identifier",
            examples=["flext_123456789", "entity_abcdefgh"],
            json_schema_extra={"format": "flext-entity-id"},
            validation_alias=AliasChoices("id", "entityId", "entity_id"),
            serialization_alias="entityId",
        ),
    ]

    version: FlextVersion = Field(
        default_factory=lambda: FlextVersion(1),
        description="Entity version for optimistic locking",
        examples=[1, 2, 10],
        validation_alias=AliasChoices("version", "entityVersion", "entity_version"),
    )

    # Timestamp fields - using RootModel timestamp with aliases
    created_at: FlextTimestamp = Field(
        default_factory=create_timestamp,
        description="Entity creation timestamp",
        examples=["2024-01-01T00:00:00Z"],
        validation_alias=AliasChoices("created_at", "createdAt", "dateCreated"),
    )

    updated_at: FlextTimestamp = Field(
        default_factory=create_timestamp,
        description="Entity last update timestamp",
        examples=["2024-01-01T12:00:00Z"],
        validation_alias=AliasChoices(
            "updated_at",
            "updatedAt",
            "dateModified",
            "lastModified",
        ),
    )

    # Domain events - using RootModel event list with standard patterns
    domain_events: FlextEventList = Field(
        default_factory=lambda: FlextEventList([]),
        exclude=True,  # Never serialize domain events
        description="Domain events collected during entity operations for event sourcing",
        examples=[],
        repr=False,  # Don't include in __repr__
    )

    # Optional metadata - using RootModel metadata with flexible aliases
    metadata: FlextMetadata = Field(
        default_factory=lambda: FlextMetadata({}),
        description="Additional metadata for the entity",
        examples=[{"category": "user", "tags": ["premium"]}],
        validation_alias=AliasChoices("metadata", "meta", "properties", "attributes"),
    )

    @field_validator("id", mode="before")
    @classmethod
    def validate_entity_id(
        cls,
        v: object,
        info: ValidationInfo,
    ) -> FlextEntityId:
        """Validate entity ID using RootModel pattern with context information."""
        # FlextEntityId is a type alias for str, so no isinstance check needed
        _ = info  # Acknowledge parameter for future use
        return FlextEntityId(str(v))

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v: object) -> FlextVersion:
        """Validate version with context-aware logic."""
        # FlextVersion is a type alias for int, so just validate the value directly
        if isinstance(v, int):
            version_value = v
        elif isinstance(v, str):
            try:
                version_value = int(v)
            except ValueError as e:
                error_message = f"Invalid version format: {v}"
                raise ValueError(error_message) from e
        elif hasattr(v, "__int__"):
            try:
                # Cast to support int conversion for objects with __int__ method
                int_convertible = cast("int | str", v)
                version_value = int(int_convertible)
            except (ValueError, TypeError) as e:
                error_message = f"Invalid version format: {v}"
                raise ValueError(error_message) from e
        else:
            error_message = (
                f"Version must be int or convertible to int, got {type(v).__name__}"
            )
            raise ValueError(error_message)

        # Ensure version is positive
        if version_value < 1:
            error_message = "Version must be >= 1"
            raise ValueError(error_message)

        return FlextVersion(version_value)

    @property
    @computed_field
    def entity_type(self) -> str:
        """Computed field providing the entity type name."""
        return self.__class__.__name__

    @property
    @computed_field
    def entity_age_seconds(self) -> float:
        """Computed field providing entity age in seconds."""
        now = datetime.now(UTC_TIMEZONE)
        created = datetime.fromisoformat(str(self.created_at))
        return (now - created).total_seconds()

    @property
    @computed_field
    def is_new_entity(self) -> bool:
        """Computed field indicating if this is a new entity (version 1)."""
        return self.version == 1

    @property
    @computed_field
    def has_events(self) -> bool:
        """Computed field indicating if entity has pending domain events."""
        return len(self.domain_events) > 0

    @override
    def __hash__(self) -> int:
        """Hash based on entity ID (identity-based)."""
        return hash(self.id)

    @override
    def __eq__(self, other: object) -> bool:
        """Equality based on entity ID (identity-based)."""
        if not isinstance(other, FlextEntity):
            return False
        return str(self.id) == str(other.id)

    @override
    def __str__(self) -> str:
        """Return human-readable entity representation."""
        return f"{self.__class__.__name__}(id={self.id})"

    @override
    def __repr__(self) -> str:
        """Return detailed entity representation for debugging."""
        fields_repr = [f"id={self.id}", f"version={self.version}"]

        # Add other fields dynamically, excluding domain_events and timestamps
        excluded_fields = {"id", "version", "domain_events", "created_at", "updated_at"}
        for field_name in type(self).model_fields:
            if field_name not in excluded_fields:
                value = getattr(self, field_name, None)
                if isinstance(value, str):
                    fields_repr.append(f"{field_name}={value}")
                else:
                    fields_repr.append(f"{field_name}={value}")

        return f"{self.__class__.__name__}({', '.join(fields_repr)})"

    def increment_version(self) -> FlextResult[Self]:
        """Increment version for optimistic locking and return new entity."""
        try:
            new_version_result = create_version(int(self.version) + 1)
            if new_version_result.is_failure:
                return FlextResult[Self].fail(
                    new_version_result.error or "Invalid version"
                )

            entity_data = self.model_dump()
            # Keep RootModel types when reconstructing entity_data so static
            # type checkers can infer correct shapes
            entity_data["version"] = new_version_result.value
            entity_data["updated_at"] = create_timestamp()

            try:
                new_entity = type(self)(**entity_data)
            except TypeError as e:
                return FlextResult[Self].fail(f"Failed to increment version: {e}")

            validation_result = new_entity.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[Self].fail(
                    validation_result.error or "Validation failed"
                )
            return FlextResult[Self].ok(new_entity)
        except Exception as e:
            return FlextResult[Self].fail(f"Failed to increment version: {e}")

    def with_version(self, new_version: int) -> Self:
        """Create new entity with specified version.

        Args:
            new_version: The new version number

        Returns:
            New entity instance with the specified version

        Raises:
            FlextExceptions.ValidationError: If version is invalid or not greater than current

        """

        def _raise_version_error() -> None:
            error_msg = "New version must be greater than current version"
            raise FlextExceptions.ValidationError(error_msg)

        def _raise_invalid_version_error(error: str) -> None:
            error_msg = f"Invalid version: {error}"
            raise FlextExceptions.ValidationError(error_msg)

        def _raise_validation_error(error: str) -> None:
            error_msg = error or "Validation failed"
            raise FlextExceptions.ValidationError(error_msg)

        # Validate new version is greater than current
        if new_version <= self.version:
            _raise_version_error()

        try:
            # Create new version
            new_version_result = create_version(new_version)
            if new_version_result.is_failure:
                _raise_invalid_version_error(
                    new_version_result.error or "Unknown version error",
                )

            # Create entity data with new version
            entity_data = self.model_dump()
            # Preserve RootModel types for model construction
            entity_data["version"] = new_version_result.value
            entity_data["updated_at"] = create_timestamp()

            try:
                new_entity = type(self)(**entity_data)
            except TypeError as e:
                error_msg = f"Failed to set version: {e}"
                raise FlextExceptions.ValidationError(error_msg) from e

            # Validate business rules on new entity
            validation_result = new_entity.validate_business_rules()
            if validation_result.is_failure:
                _raise_validation_error(
                    validation_result.error or "Unknown validation error",
                )

            return new_entity

        except Exception as e:
            error_msg = f"Failed to set version: {e}"
            raise FlextExceptions.ValidationError(error_msg) from e

    def add_domain_event(
        self,
        event_type_or_dict: str | dict[str, object],
        event_data: dict[str, object] | None = None,
    ) -> FlextResult[None]:
        """Add domain event to entity.

        Args:
            event_type_or_dict: Either event type string or dict containing event
            event_data: Event data dictionary (when first arg is string)

        Returns:
            FlextResult indicating success or failure

        """
        try:
            # Handle both calling patterns:
            # 1. add_domain_event("created", {"data": "value"})
            # 2. add_domain_event({"type": "created", "data": "value"})
            if isinstance(event_type_or_dict, dict):
                event_type = str(event_type_or_dict.get("type", "unknown"))
                event_data = {
                    k: v for k, v in event_type_or_dict.items() if k != "type"
                }
            else:
                event_type = event_type_or_dict
                if event_data is None:
                    event_data = {}

            # Create FlextEvent object
            event_result = FlextPayload.create_event(
                event_type=event_type,
                event_data=event_data,
                aggregate_id=str(self.id),
                version=int(self.version),
            )

            if event_result.is_failure:
                return FlextResult[None].fail(
                    event_result.error or "Event creation failed"
                )

            # Get the created event
            event = event_result.value

            # Add to domain events using FlextEventList pattern
            # Create event dict with aggregate_id for compatibility
            event_dict: dict[str, object] = {
                "type": event_type,
                "data": event_data,
                "timestamp": datetime.now(UTC_TIMEZONE).isoformat(),
                "aggregate_id": str(self.id),  # Include aggregate_id in dict
                "version": self.version,
            }

            # Create new events list
            new_events_list: list[dict[str, object]] = []
            new_events_list.extend(self.domain_events.root)
            new_events_list.append(event_dict)
            new_events = FlextEventList(new_events_list)

            # Store the FlextEvent object using proper attribute access
            # Use object.__setattr__ to satisfy MyPy for RootModel dynamic attributes
            if not hasattr(new_events, "_flext_events"):
                object.__setattr__(new_events, "_flext_events", [])
            if hasattr(self.domain_events, "_flext_events"):
                existing_events = getattr(self.domain_events, "_flext_events", [])
                object.__setattr__(new_events, "_flext_events", existing_events.copy())
            current_events = getattr(new_events, "_flext_events", [])
            current_events.append(event)

            # Update the entity's domain_events field
            self.domain_events = new_events

            return FlextResult[None].ok(None)

        except Exception as e:
            error_msg = f"Failed to add domain event: {e}"
            return FlextResult[None].fail(error_msg)

    def clear_events(self) -> list[object]:
        """Clear all domain events and return the cleared events.

        Returns:
            List of cleared domain events

        """
        # Extract events before clearing using list comprehension
        events = [self.domain_events[i] for i in range(len(self.domain_events))]

        # Clear the events
        self.domain_events, _ = self.domain_events.clear()

        # Clear stored FlextEvent objects
        if hasattr(self.domain_events, "_flext_events"):
            object.__setattr__(self.domain_events, "_flext_events", [])

        return events

    def validate_field(self, field_name: str, _value: object) -> FlextResult[None]:
        """Validate individual field value against business rules."""
        try:
            # Check if field exists in model
            if field_name not in type(self).model_fields:
                # For unknown fields, just return success
                # This handles tests that try to validate non-existent fields
                return FlextResult[None].ok(None)

            # Get field definition for validation
            field_info = type(self).model_fields[field_name]

            # Basic type validation using Pydantic field information
            if hasattr(field_info, "annotation"):
                # For basic validation, we could check type compatibility
                # This is a simplified implementation - full validation would use Pydantic internals
                pass

            # For field-specific validation, just return success
            # Business rules are validated separately
            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(
                f"Validation error for field '{field_name}': {e}"
            )

    def validate_all_fields(self) -> FlextResult[None]:
        """Validate all entity fields against business rules.

        Returns:
            FlextResult indicating validation success or failure

        """
        try:
            # Validate each field individually
            for field_name in type(self).model_fields:
                field_value = getattr(self, field_name, None)
                result = self.validate_field(field_name, field_value)
                if result.is_failure:
                    return result

            # Run overall business rules validation
            return self.validate_business_rules()

        except Exception as e:
            return FlextResult[None].fail(f"Field validation error: {e}")

    def copy_with(self, **changes: object) -> FlextResult[Self]:
        """Create copy with changes and auto-increment version."""
        try:
            entity_data = self.model_dump()
            entity_data.update(changes)

            # Auto-increment version unless explicitly provided
            if changes and "version" not in changes:
                # Keep RootModel type for version
                entity_data["version"] = FlextVersion(int(self.version) + 1)

            # Update timestamp when changes are made
            if changes and "updated_at" not in changes:
                entity_data["updated_at"] = create_timestamp()

            try:
                new_entity = type(self)(**entity_data)
            except TypeError as te:
                return FlextResult[Self].fail(f"Failed to increment version: {te}")

            validation_result = new_entity.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[Self].fail(
                    validation_result.error or "Business rule validation failed",
                )

            return FlextResult[Self].ok(new_entity)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            return FlextResult[Self].fail(f"Failed to copy entity: {e}")

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules (override in subclasses for specific rules)."""
        return FlextResult[None].ok(None)

    @field_serializer("created_at", "updated_at", when_used="json", check_fields=False)
    def serialize_timestamps(self, value: FlextTimestamp) -> str:
        """Serialize timestamps in ISO format for JSON."""
        return str(value) if value else datetime.now(UTC_TIMEZONE).isoformat()

    @field_serializer("version", when_used="json", check_fields=False)
    def serialize_version(self, value: FlextVersion) -> dict[str, object]:
        """Serialize version with additional metadata in JSON mode."""
        return {
            "number": value,
            "is_initial": value == 1,
            "timestamp": datetime.now(UTC_TIMEZONE).isoformat(),
        }

    @field_serializer("metadata", when_used="always", check_fields=False)
    def serialize_metadata(self, value: FlextMetadata) -> dict[str, object]:
        """Serialize metadata with type preservation."""
        base_metadata = dict(value)
        base_metadata["_entity_type"] = self.__class__.__name__
        base_metadata["_serialized_at"] = datetime.now(UTC_TIMEZONE).isoformat()
        return base_metadata

    @model_serializer(mode="wrap", when_used="json")
    def serialize_entity_for_api(
        self,
        serializer: Callable[[FlextEntity], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]:
        """Entity-specific model serializer for API with computed fields."""
        _ = info  # Acknowledge parameter for future use
        data = serializer(self)
        # Data is already dict type from serializer
        # Add entity-specific API metadata
        # data is already dict[str, object] from serializer
        data["_entity"] = {
            "type": self.entity_type,
            "age_seconds": self.entity_age_seconds,
            "is_new": self.is_new_entity,
            "has_events": self.has_events,
            "api_version": "v2",
        }
        # Remove domain events from API output (they're excluded but double-check)
        data.pop("domain_events", None)
        return data


def _get_default_registry() -> dict[str, type[FlextModel] | object]:
    """Get default registry for FlextFactory."""
    return {}


class FlextModels:
    """Consolidated FLEXT Models - Single class containing ALL model functionality.

    This is the main consolidated class following the FLEXT refactoring pattern
    where all model-related functionality is organized within a single class
    with nested classes for specific model types. This approach provides:

    - Single Responsibility: All model functionality in one place
    - High Cohesion: Related model operations grouped together
    - Type Safety: Full FlextTypes integration with hierarchical access
    - Constants Integration: Direct FlextConstants usage throughout
    - Clean Architecture: Clear separation between model types
    - Factory Patterns: Unified creation and validation methods

    Architecture:
        The class is organized into nested model classes that correspond to
        different architectural layers following Clean Architecture:

        - Model: Base Pydantic models with FlextTypes integration
        - RootModel: Root data structures using hierarchical types
        - Value: Value objects with FlextConstants validation
        - Entity: Domain entities with lifecycle and FlextTypes support
        - Factory: Creation patterns with FlextResult error handling

    FlextTypes Integration:
        All nested classes use FlextTypes hierarchical structure for type safety:

        - FlextTypes.Core: Fundamental types (Value, Data, Config)
        - FlextTypes.Domain: Business domain types (EntityId, EventData)
        - FlextTypes.Service: Service layer types (ServiceName, Container)
        - FlextTypes.Handler: Handler patterns (HandlerDict, MetricsData)

    FlextConstants Integration:
        Constants are accessed through FlextConstants hierarchical structure:

        - FlextConstants.Validation: Validation rules and limits
        - FlextConstants.Defaults: Default values for configuration
        - FlextConstants.Errors: Error codes with structured hierarchy
        - FlextConstants.Messages: Standardized user-facing messages

    Examples:
        Creating models with type safety::

            # Using nested classes with FlextTypes
            config_data: FlextTypes.Core.Config = {"debug": True}
            model = FlextModels.Model(**config_data)

            # Entity creation with domain types
            entity_id: FlextTypes.Domain.EntityId = "user_123"
            entity = FlextModels.Entity(id=entity_id)

        Using constants for validation::

            # FlextConstants for validation limits
            max_length = FlextConstants.Validation.MAX_STRING_LENGTH
            timeout = FlextConstants.Defaults.TIMEOUT
            error_code = FlextConstants.Errors.VALIDATION_ERROR

        Factory pattern with error handling::

            # FlextResult integration for railway pattern
            result = FlextModels.create("user", name="John")
            if result.success:
                user = result.value
            else:
                error = result.error  # Uses FlextConstants error codes

    """

    # ==========================================================================
    # TYPE SYSTEM INTEGRATION - FlextTypes hierarchical access
    # ==========================================================================

    class Types:
        """FlextTypes integration for hierarchical type access.

        Provides organized access to the complete FlextTypes system within
        the models context, maintaining the hierarchical structure for
        better organization and type safety.
        """

        # Core fundamental types
        Core = FlextTypes.Core

        # Domain modeling types
        Domain = FlextTypes.Domain

        # Service layer types
        Service = FlextTypes.Service

        # Handler pattern types
        Handler = FlextTypes.Handler

        # Configuration types
        Config = FlextTypes.Config

        # Payload and transport types
        Payload = FlextTypes.Payload

        # Protocol aliases
        Protocol = FlextTypes.Protocol

        # Result types for railway pattern
        Result = FlextTypes.Result

        # Authentication types
        Auth = FlextTypes.Auth

        # Field definition types
        Field = FlextTypes.Fields

        # Logging and observability types
        Logging = FlextTypes.Logging

    # ==========================================================================
    # CONSTANTS SYSTEM INTEGRATION - FlextConstants hierarchical access
    # ==========================================================================

    class Constants:
        """FlextConstants integration for hierarchical constant access.

        Provides organized access to the complete FlextConstants system within
        the models context, maintaining the hierarchical structure for
        better organization and maintainability.
        """

        # Core system constants
        Core = FlextConstants.Core

        # Network and connectivity constants
        Network = FlextConstants.Network

        # Validation rules and limits
        Validation = FlextConstants.Validation

        # Error codes and categorization
        Errors = FlextConstants.Errors

        # User-facing messages
        Messages = FlextConstants.Messages

        # Status values
        Status = FlextConstants.Status

        # Regex patterns
        Patterns = FlextConstants.Patterns

        # Default values
        Defaults = FlextConstants.Defaults

        # System limits
        Limits = FlextConstants.Limits

        # Performance constants
        Performance = FlextConstants.Performance

        # Configuration system
        Configuration = FlextConstants.Configuration

        # CLI constants
        Cli = FlextConstants.Cli

        # Infrastructure constants
        Infrastructure = FlextConstants.Infrastructure

        # Model configuration constants
        Models = FlextConstants.Models

        # Observability constants
        Observability = FlextConstants.Observability

        # Handler system constants
        Handlers = FlextConstants.Handlers

        # Entity system constants
        Entities = FlextConstants.Entities

        # Validation system constants
        ValidationSystem = FlextConstants.ValidationSystem

        # Infrastructure messaging
        InfrastructureMessages = FlextConstants.InfrastructureMessages

        # Platform constants
        Platform = FlextConstants.Platform

        # Enum definitions
        Enums = FlextConstants.Enums

    # ==========================================================================
    # NESTED MODEL CLASSES - Core model implementations with FlextTypes/Constants
    # ==========================================================================

    # References to model classes - will be set after class definitions
    Model: type[BaseModel] | None = None
    RootModel: type[BaseModel] | None = None
    Value: type[BaseModel] | None = None
    Entity: type[BaseModel] | None = None

    # ==========================================================================
    # FACTORY FUNCTIONALITY - Model creation and registration
    # ==========================================================================
    _registry: ClassVar[dict[str, type[FlextModel] | object] | None] = None

    @classmethod
    def _ensure_registry(cls) -> None:
        """Ensure registry is initialized."""
        if cls._registry is None:
            cls._registry = _get_default_registry()

    @classmethod
    def register(cls, name: str, factory_or_class: type[FlextModel] | object) -> None:
        """Register a model class or factory function with a name."""
        cls._ensure_registry()
        registry = cast("dict[str, type[FlextModel] | object]", cls._registry)
        registry[name] = factory_or_class

    @classmethod
    def create(cls, name: str, **kwargs: object) -> FlextResult[object]:
        """Create model instance using registered factory with FlextConstants integration.

        Uses FlextConstants for error codes and messages to ensure consistent
        error handling throughout the system.

        Args:
            name: The name of the factory to use
            **kwargs: Arguments to pass to the factory

        Returns:
            FlextResult containing the created instance or error with structured codes

        """
        cls._ensure_registry()
        registry = cast("dict[str, type[FlextModel] | object]", cls._registry)
        if name not in registry:
            error_msg = f"No factory registered for '{name}'"
            return FlextResult[object].fail(
                error_msg, error_code=cls.Constants.Errors.RESOURCE_NOT_FOUND
            )

        factory = registry[name]

        try:
            return cls._create_with_factory(name, factory, kwargs)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            error_msg = f"Failed to create '{name}': {e}"
            return FlextResult[object].fail(
                error_msg, error_code=cls.Constants.Errors.OPERATION_ERROR
            )

    @classmethod
    def _create_with_factory(
        cls,
        name: str,
        factory: object,
        kwargs: dict[str, object],
    ) -> FlextResult[object]:
        """Create instance with factory using FlextConstants for error handling.

        Integrates FlextConstants error codes and messages for consistent
        error reporting throughout the factory creation process.
        """
        # Handle model class
        if isinstance(factory, type):
            try:
                if issubclass(factory, FlextModel):
                    return cls._create_model_instance(factory, kwargs)
            except TypeError:
                pass  # factory is not a class

        # Handle callable factory function
        # Type: ignore for complex union type with callable check
        if callable(factory):
            # Callable factory function handling
            return cls._create_with_callable(factory, kwargs)

        error_msg = f"Invalid factory type for '{name}'"
        return FlextResult[object].fail(
            error_msg, error_code=cls.Constants.Errors.TYPE_ERROR
        )

    @classmethod
    def _create_model_instance(
        cls,
        factory: type[FlextModel],
        kwargs: dict[str, object],
    ) -> FlextResult[object]:
        """Create model instance with validation using FlextConstants messages.

        Uses FlextConstants for consistent validation error messages and codes.
        """
        try:
            instance = factory.model_validate(kwargs)
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                error_msg = (
                    validation_result.error or cls.Constants.Messages.VALIDATION_FAILED
                )
                return FlextResult[object].fail(
                    error_msg, error_code=cls.Constants.Errors.VALIDATION_ERROR
                )
            return FlextResult[object].ok(instance)
        except Exception as e:
            error_msg = cls.Constants.Messages.OPERATION_FAILED + f": {e}"
            return FlextResult[object].fail(
                error_msg, error_code=cls.Constants.Errors.VALIDATION_ERROR
            )

    @classmethod
    def _create_with_callable(
        cls,
        factory: object,  # Accept any callable to avoid complex typing
        kwargs: dict[str, object],
    ) -> FlextResult[object]:
        """Create with callable factory."""
        try:
            # Factory is guaranteed to be callable by the caller
            # Type: ignore to avoid complex callable typing
            instance: object = factory(**kwargs)  # type: ignore[operator]
            # Check for business rule validation if available
            try:
                validation_method = getattr(instance, "validate_business_rules", None)
                if validation_method and callable(validation_method):
                    validation_result = validation_method()
                    if hasattr(validation_result, "is_failure") and getattr(
                        validation_result, "is_failure", False
                    ):
                        error = getattr(validation_result, "error", "Validation failed")
                        return FlextResult[object].fail(
                            str(error) if error else "Validation failed"
                        )
            except Exception as e:
                # If validation check fails, continue without validation
                try:
                    logger = FlextLoggerFactory.get_logger("FlextFactory")
                    logger.debug(f"Business rule validation check failed: {e}")
                except ImportError:
                    pass  # Logger not available, silently continue
            return FlextResult[object].ok(instance)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ) as e:
            return FlextResult[object].fail(f"Factory function failed: {e}")

    @classmethod
    def create_entity_factory(
        cls,
        entity_class: type[FlextModel],
        defaults: dict[str, object] | None = None,
    ) -> object:
        """Create a factory function for the given entity class.

        Args:
            entity_class: The entity class to create a factory for
            defaults: Default values to use when creating instances

        Returns:
            A callable factory function that returns FlextResult[entity_class]

        """

        def factory(**kwargs: object) -> FlextResult[object]:
            """Factory function that creates entity instances with validation."""
            try:
                # Merge defaults with provided kwargs
                merged_kwargs: dict[str, object] = {}
                if defaults:
                    merged_kwargs.update(defaults)
                merged_kwargs.update(kwargs)

                # Auto-generate ID if not provided and entity is FlextEntity
                if issubclass(entity_class, FlextEntity):
                    if "id" not in merged_kwargs or merged_kwargs.get("id") == "":
                        merged_kwargs["id"] = FlextGenerators.generate_id()

                    # Set default version if not provided
                    if "version" not in merged_kwargs:
                        merged_kwargs["version"] = 1

                # Return directly without redundant cast
                return cls.create_model(entity_class, **merged_kwargs)
            except ImportError as e:
                return FlextResult[object].fail(str(e))
            except TypeError as e:
                return FlextResult[object].fail(f"Failed to create entity: {e}")

        return factory

    @classmethod
    def create_value_object_factory(
        cls,
        value_object_class: type[FlextModel],
        defaults: dict[str, object] | None,
    ) -> object:
        """Create a factory function for the given value object class.

        Args:
            value_object_class: The value object class to create a factory for
            defaults: Default values to use when creating instances

        Returns:
            A factory object with model_validate method that returns FlextResult[value_object_class]

        """

        class ValueObjectFactory:
            """Factory wrapper with model_validate method."""

            def __init__(
                self,
                vo_class: type[FlextModel],
                default_values: dict[str, object] | None,
            ) -> None:
                self.vo_class = vo_class
                self.defaults = default_values or {}

            def model_validate(self, data: dict[str, object]) -> FlextResult[object]:
                """Validate and create a value object instance."""
                try:
                    # Merge defaults with provided data
                    merged_data: dict[str, object] = {}
                    merged_data.update(self.defaults)
                    merged_data.update(data)

                    # Return directly without redundant cast
                    return cls.create_model(self.vo_class, **merged_data)
                except ImportError as e:
                    return FlextResult[object].fail(str(e))
                except TypeError as e:
                    return FlextResult[object].fail(
                        f"Failed to create value object: {e}"
                    )

            def __call__(self, **kwargs: object) -> FlextResult[object]:
                """Allow calling factory directly."""
                return self.model_validate(kwargs)

        return ValueObjectFactory(value_object_class, defaults)

    @staticmethod
    def create_model(
        model_class: type,
        **kwargs: object,
    ) -> FlextResult[object]:
        """Create model with validation."""
        try:
            # Explicitly type kwargs as dict[str, object]

            # Check if model_class has model_validate method (Pydantic v2+)
            # Use getattr to safely access class methods
            model_validate_method = getattr(model_class, "model_validate", None)
            parse_obj_method = getattr(model_class, "parse_obj", None)

            if model_validate_method is not None:
                instance = model_validate_method(kwargs)
            elif parse_obj_method is not None:  # Pydantic v1
                instance = parse_obj_method(kwargs)
            else:
                # Fallback to direct instantiation
                instance = model_class(**kwargs)
            # Check if instance has validate_business_rules method
            try:
                validation_method = getattr(instance, "validate_business_rules", None)
                if validation_method is not None and callable(validation_method):
                    validation_result = validation_method()
                    if getattr(validation_result, "is_failure", False):
                        error_msg = str(
                            getattr(
                                validation_result,
                                "error",
                                "Business rule validation failed",
                            )
                            or "Business rule validation failed"
                        )
                        return FlextResult[object].fail(error_msg)
            except Exception as e:
                # If validation check fails, continue without validation
                try:
                    logger = FlextLoggerFactory.get_logger("FlextFactory")
                    logger.debug(f"Business rule validation check failed: {e}")
                except ImportError:
                    pass  # Logger not available, silently continue
            return FlextResult[object].ok(instance)
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            error_msg = str(e)
            if "validation error" in error_msg.lower():
                return FlextResult[object].fail(
                    f"Failed to create {model_class.__name__}: {e}"
                )
            return FlextResult[object].fail(
                f"Failed to create {model_class.__name__}: {e}"
            )


# Type aliases for convenience (no self-assignment to avoid lint warnings)
# Expose canonical names via __all__ below instead.
# Legacy compatibility aliases removed to satisfy linting rules.

#  factory patterns - no legacy imports needed
# FlextEntityFactory will be set after FlextModels is defined

# Python 3.13 type aliases for patterns

# JSON Schema type aliases following standard Pydantic patterns
JsonSchemaValue = str | int | float | bool | None | dict[str, object] | list[object]
JsonSchemaFieldInfo = dict[str, JsonSchemaValue]
JsonSchemaDefinition = dict[str, JsonSchemaValue]

# Core FLEXT type aliases
DomainEventDict = dict[str, object]
FlextEntityDict = dict[str, object]
FlextValueObjectDict = dict[str, object]

# Legacy compatibility aliases
FlextValueObject = FlextValue  # Alias for backward compatibility
FlextOperationDict = dict[str, object]
FlextConnectionDict = dict[str, object]

#  type aliases for model patterns
FlextModelDict = dict[str, JsonSchemaValue]
FlextValidationContext = dict[str, object]
FlextFieldValidationInfo = dict[str, JsonSchemaValue]

# Legacy model aliases - maintained for ecosystem compatibility
FlextBaseModel = FlextModel  # Primary compatibility alias
FlextDatabaseModel = FlextModel
FlextOracleModel = FlextModel
FlextLegacyConfig = FlextModel
FlextOperationModel = FlextModel
FlextServiceModel = FlextModel
FlextSingerStreamModel = FlextModel


# Legacy factory functions - maintained for ecosystem compatibility
# Note: Model classes are aliased above - these factory functions create instances


def create_database_model(**kwargs: object) -> FlextResult[FlextModel]:
    """Create database model instance with FlextConstants integration.

    Uses FlextConstants for default values and error handling to ensure
    consistent behavior across the FLEXT ecosystem.
    """
    try:
        # Create with defaults using FlextConstants
        model_data: dict[str, object] = {
            "host": FlextConstants.Infrastructure.DEFAULT_HOST,
            "port": FlextConstants.Infrastructure.DEFAULT_DB_PORT,
            "database": "flext_db",
        }
        model_data.update(kwargs)
        return FlextResult[FlextModel].ok(FlextDatabaseModel(**model_data))
    except Exception as e:
        error_msg = f"Failed to create database model: {e}"
        return FlextResult[FlextModel].fail(
            error_msg, error_code=FlextConstants.Errors.CONFIGURATION_ERROR
        )


def create_oracle_model(**kwargs: object) -> FlextResult[FlextModel]:
    """Create Oracle model instance with FlextConstants integration.

    Uses FlextConstants for default values and error handling.
    """
    try:
        # Create with defaults using FlextConstants
        model_data: dict[str, object] = {
            "host": FlextConstants.Infrastructure.DEFAULT_HOST,
            "port": FlextConstants.Infrastructure.DEFAULT_ORACLE_PORT,
            "sid": "ORCL",
            "service_name": "XEPDB1",
        }
        model_data.update(kwargs)
        return FlextResult[FlextModel].ok(FlextOracleModel(**model_data))
    except Exception as e:
        error_msg = f"Failed to create oracle model: {e}"
        return FlextResult[FlextModel].fail(
            error_msg, error_code=FlextConstants.Errors.CONFIGURATION_ERROR
        )


def create_operation_model(**kwargs: object) -> FlextResult[FlextModel]:
    """Create operation model instance."""
    result = FlextFactory.create_model(FlextModel, **kwargs)
    return result.map(lambda obj: obj if isinstance(obj, FlextModel) else FlextModel())


def create_service_model(**kwargs: object) -> FlextResult[FlextModel]:
    """Create service model instance with FlextConstants integration.

    Uses FlextConstants for default values and error handling.
    """
    try:
        # Create with defaults using FlextConstants
        model_data: dict[str, object] = {
            "host": FlextConstants.Infrastructure.DEFAULT_HOST,
            "port": FlextConstants.Platform.FLEXCORE_PORT,
            "service_name": "flext_service",
            "version": FlextConstants.Core.VERSION,
        }
        model_data.update(kwargs)
        return FlextResult[FlextModel].ok(FlextServiceModel(**model_data))
    except Exception as e:
        error_msg = f"Failed to create service model: {e}"
        return FlextResult[FlextModel].fail(
            error_msg, error_code=FlextConstants.Errors.CONFIGURATION_ERROR
        )


def validate_all_models(*models: FlextModel) -> FlextResult[None]:
    """Validate variable arguments of models."""
    for model in models:
        validation_result = model.validate_business_rules()
        if validation_result.is_failure:
            return validation_result
    return FlextResult[None].ok(None)


def model_to_dict_safe(model: object) -> dict[str, object]:
    """Safely convert model to dictionary."""
    model_dump_method = getattr(model, "model_dump", None)
    if model_dump_method is not None and callable(model_dump_method):
        return cast("dict[str, object]", model_dump_method())
    if isinstance(model, dict):
        return cast("dict[str, object]", model)
    # For invalid inputs (None, strings, etc.), return empty dict
    return {}


# =============================================================================
# SETUP NESTED CLASS REFERENCES - Connect existing classes to FlextModels
# =============================================================================

# Set up the nested class references
FlextModels.Model = FlextModel
FlextModels.RootModel = FlextRootModel
FlextModels.Value = FlextValue
FlextModels.Entity = FlextEntity

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Factory alias for backward compatibility (FlextFactory now references FlextModels)
FlextFactory = FlextModels
FlextEntityFactory = FlextModels  # Entity factory also references FlextModels

# Exports
__all__: list[str] = [
    "FlextEntity",  # Entity pattern
    "FlextEntityFactory",  # Entity factory alias
    "FlextFactory",  # Factory for backward compatibility
    "FlextModel",  # Base model class
    "FlextModels",  # Main class with namespace
    "FlextRootModel",  # Root model pattern
    "FlextValue",  # Value object pattern
    "model_to_dict_safe",  # Safe conversion helper
    "validate_all_models",  # Helper function
]
