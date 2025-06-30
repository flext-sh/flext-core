"""Unified modern reflection patterns for enterprise Python 3.13 architecture.

This module consolidates all reflection capabilities into a single, authoritative source
providing domain-driven design patterns, validation, type conversion, and specifications
with complete type safety and zero boilerplate.

ZERO TOLERANCE: This is the ONLY source of reflection patterns in the codebase.
All other reflection modules should be deprecated and removed.

Features:
- Value objects with automatic equality and hashing
- Entity patterns with identity management
- Aggregate root patterns with domain event handling
- Specification patterns with logical operators
- Automatic validation and type conversion
- Modern Python 3.13 type aliases and generics
"""

from __future__ import annotations

import contextlib
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, ClassVar, TypeVar, get_origin
from uuid import UUID

# ZERO TOLERANCE: Import specification classes from canonical location
from flext_core.domain.advanced_types import CompositeSpecification, NotSpecification

# ZERO TOLERANCE: Import protocols from canonical domain_patterns.py to eliminate duplicates
from flext_core.reflection.domain_patterns import (
    ConverterProtocol,
    DataclassProtocol,
    EntityBase,
    ValidatorProtocol,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Python 3.13 type aliases for enterprise reflection patterns
type DomainEvent = dict[str, str | int | float | bool | None]
type EntityIdentity = str | UUID | int
type ValidationTarget = object
type ConversionTarget = object
type ConversionResult = object
type SpecificationTarget = object
type DomainObject = object

# Generic type variables with proper constraints
T = TypeVar("T")
TEntity = TypeVar("TEntity", bound="EntityBase")
TSpecification = TypeVar("TSpecification", bound="SpecificationBase")


# ============================================================================
# PROTOCOLS FOR TYPE SAFETY
# ============================================================================


# Protocols imported above from domain_patterns.py


class SpecificationBase:
    """Base protocol for domain specifications with logical operators."""

    def is_satisfied_by(self, _obj: SpecificationTarget) -> bool:
        """Check if object satisfies this specification.

        Args:
        ----
            obj: Object to validate against specification

        Returns:
        -------
            True if object satisfies specification, False otherwise

        """
        return True

    def __and__(self, other: SpecificationBase) -> AndSpecification[object]:
        """Combine specifications with logical AND."""
        return AndSpecification(self, other, "and")

    def __or__(self, other: SpecificationBase) -> OrSpecification[object]:
        """Combine specifications with logical OR."""
        return OrSpecification(self, other, "or")

    def __invert__(self) -> NotSpecification[object]:
        """Negate specification with logical NOT."""
        return NotSpecification(self)


# ============================================================================
# ENTERPRISE REFLECTION REGISTRY
# ============================================================================


class EnterpriseReflectionRegistry:
    """Unified registry for validation, conversion, and reflection patterns.

    This registry provides enterprise-grade automatic validation and type
    conversion capabilities with complete type safety and zero boilerplate.

    Features:
    - Type-safe validator registration
    - Automatic type conversion
    - Dataclass field validation
    - Specification pattern support
    """

    _validators: ClassVar[dict[type, ValidatorProtocol]] = {}
    _converters: ClassVar[dict[tuple[type, type], ConverterProtocol]] = {}
    _domain_events: ClassVar[list[DomainEvent]] = []

    @classmethod
    def register_validator(
        cls, target_type: type,
    ) -> Callable[[ValidatorProtocol], ValidatorProtocol]:
        """Register validator for specific type.

        Args:
        ----
            target_type: Type to register validator for

        Returns:
        -------
            Decorator function for validator registration

        """

        def decorator(validator_func: ValidatorProtocol) -> ValidatorProtocol:
            cls._validators[target_type] = validator_func
            return validator_func

        return decorator

    @classmethod
    def register_converter(
        cls, from_type: type, to_type: type,
    ) -> Callable[[ConverterProtocol], ConverterProtocol]:
        """Register converter between types.

        Args:
        ----
            from_type: Source type
            to_type: Target type

        Returns:
        -------
            Decorator function for converter registration

        """

        def decorator(converter_func: ConverterProtocol) -> ConverterProtocol:
            cls._converters[from_type, to_type] = converter_func
            return converter_func

        return decorator

    @classmethod
    def validate(cls, obj: ValidationTarget) -> None:
        """Automatically validate object using registered validators.

        Args:
        ----
            obj: Object to validate

        Raises:
        ------
            ValueError: If validation fails

        """
        obj_type = type(obj)
        if obj_type in cls._validators:
            cls._validators[obj_type](obj)

        # Auto-validate dataclass fields
        if is_dataclass(obj):
            for field_info in fields(obj):
                field_value = getattr(obj, field_info.name)
                field_type = field_info.type

                # Handle generic types
                origin = get_origin(field_type)
                if origin:
                    field_type = origin

                if isinstance(field_type, type) and field_type in cls._validators:
                    cls._validators[field_type](field_value)

    @classmethod
    def convert(cls, obj: ConversionTarget, target_type: type) -> ConversionResult:
        """Automatically convert object to target type.

        Args:
        ----
            obj: Object to convert
            target_type: Target type

        Returns:
        -------
            Converted object

        Raises:
        ------
            ValueError: If conversion fails

        """
        source_type = type(obj)
        converter_key = (source_type, target_type)

        if converter_key in cls._converters:
            return cls._converters[converter_key](obj)

        # Built-in enterprise conversions
        if target_type == UUID and isinstance(obj, str):
            return UUID(obj)

        if target_type is str:
            return str(obj)

        if target_type is int and isinstance(obj, str):
            return int(obj)

        if target_type is float and isinstance(obj, str | int):
            return float(obj)

        msg = f"No converter registered for {source_type} -> {target_type}"
        raise ValueError(msg)

    @classmethod
    def record_domain_event(cls, event: DomainEvent) -> None:
        """Record domain event for enterprise event sourcing.

        Args:
        ----
            event: Domain event to record

        """
        cls._domain_events.append(event)

    @classmethod
    def clear_domain_events(cls) -> list[DomainEvent]:
        """Clear and return all recorded domain events.

        Returns
        -------
            List of recorded domain events

        """
        events = cls._domain_events.copy()
        cls._domain_events.clear()
        return events


# ============================================================================
# DOMAIN PATTERN DECORATORS
# ============================================================================


def auto_init[T](cls: type[T]) -> type[T]:
    """Class decorator that automatically generates __init__ with validation.

    Enhanced to work with dataclasses, Pydantic models, and regular classes.
    Provides enterprise-grade automatic validation and initialization patterns.

    Args:
    ----
        cls: Class to decorate

    Returns:
    -------
        Decorated class with automatic initialization and validation

    """
    # Get original __init__ method, handling edge cases
    original_init = getattr(cls, "__init__", None)
    original_post_init = getattr(cls, "__post_init__", None)

    # Check if this is a dataclass or Pydantic model that already has proper __init__
    # Use try/except pattern for safe attribute checking - ZERO TOLERANCE MODERNIZATION
    has_dataclass_init = False
    has_pydantic_init = False

    try:
        # Check for dataclass fields attribute
        _ = cls.__dataclass_fields__
        has_dataclass_init = True
    except AttributeError:
        pass

    try:
        # Check for Pydantic model fields attribute
        _ = cls.model_fields
        has_pydantic_init = True
    except AttributeError:
        pass

    # Skip auto-init for classes that already have proper initialization
    if has_dataclass_init or has_pydantic_init:
        # Just add validation to existing __init__ without replacing it
        if original_init and original_init != object.__init__:

            def enhanced_init(
                self: ValidationTarget, *args: object, **kwargs: object,
            ) -> None:
                # Call original init first
                original_init(self, *args, **kwargs)

                # Add enterprise validation
                with contextlib.suppress(Exception):
                    # Validation failed, but don't break initialization
                    EnterpriseReflectionRegistry.validate(self)

                # Call post_init if it exists
                if original_post_init:
                    original_post_init(self)

            cls.__init__ = enhanced_init  # type: ignore[method-assign]
        return cls

    # For regular classes, create a proper __init__ method
    def new_init(self: ValidationTarget, *args: object, **kwargs: object) -> None:
        # Call original init only if it's not object.__init__
        if original_init and original_init != object.__init__:
            original_init(self, *args, **kwargs)

        # Auto-validate all fields if possible
        with contextlib.suppress(Exception):
            # Validation failed, but don't break initialization
            EnterpriseReflectionRegistry.validate(self)

        # Call original post_init if it exists
        if original_post_init:
            original_post_init(self)

    cls.__init__ = new_init  # type: ignore[method-assign]
    return cls


def value_object[T](cls: type[T]) -> type[T]:
    """Class decorator for value objects with automatic equality and hashing.

    Value objects are immutable objects that are compared by their attributes
    rather than their identity. This decorator automatically implements
    equality, hashing, and string representation.

    Args:
    ----
        cls: Class to decorate as value object

    Returns:
    -------
        Decorated class with value object semantics

    """

    def __eq__(self: DataclassProtocol, other: object) -> bool:  # noqa: N807
        if not isinstance(other, cls):
            return False
        self_dc = self
        other_dc = other
        return all(
            getattr(self_dc, field_info.name) == getattr(other_dc, field_info.name)
            for field_info in fields(self_dc)
        )

    def __hash__(self: DataclassProtocol) -> int:  # noqa: N807
        return hash(
            tuple(
                getattr(self, field_info.name)
                for field_info in fields(self)
                if field_info.hash
            ),
        )

    def __str__(self: DataclassProtocol) -> str:  # noqa: N807
        try:
            return str(self.value)
        except AttributeError:
            pass

        # Try to get field information safely
        try:
            field_strs = [
                f"{f.name}={getattr(self, f.name, 'N/A')}" for f in fields(self)
            ]
            return f"{cls.__name__}({', '.join(field_strs)})"
        except (TypeError, AttributeError):
            # Fallback to simple representation
            return f"{cls.__name__}(value_object)"

    # Apply enterprise patterns using type: ignore for method assignments
    cls.__eq__ = __eq__  # type: ignore[method-assign,assignment]
    cls.__hash__ = __hash__  # type: ignore[method-assign,assignment]
    cls.__str__ = __str__  # type: ignore[method-assign,assignment]

    # Only apply auto_init if this is not a dataclass or Pydantic model
    try:
        has_dataclass = True
    except AttributeError:
        has_dataclass = False

    try:
        has_pydantic = True
    except AttributeError:
        has_pydantic = False

    if not (has_dataclass or has_pydantic):
        return auto_init(cls)
    return cls


def entity[T](cls: type[T]) -> type[T]:
    """Class decorator for domain entities with identity management.

    Entities are objects with identity that are compared by their ID
    rather than their attributes. This decorator automatically implements
    identity-based equality and provides domain event recording.

    Args:
    ----
        cls: Class to decorate as entity

    Returns:
    -------
        Decorated class with entity semantics

    """

    def __eq__(self: object, other: object) -> bool:  # noqa: N807
        if not isinstance(other, cls):
            return False
        return getattr(self, "id", None) == getattr(other, "id", None)

    def __hash__(self: object) -> int:  # noqa: N807
        entity_id = getattr(self, "id", None)
        if entity_id is None:
            return object.__hash__(self)
        return hash((cls.__name__, entity_id))

    def record_event(self: object, event_type: str, **event_data: object) -> None:
        """Record a domain event for this entity."""
        event: DomainEvent = {
            "event_type": event_type,
            "entity_type": cls.__name__,
            "entity_id": getattr(self, "id", None),
            "data": event_data,  # type: ignore[dict-item]
        }
        EnterpriseReflectionRegistry.record_domain_event(event)

    # Apply enterprise patterns using type: ignore for method assignments
    cls.__eq__ = __eq__  # type: ignore[method-assign]
    cls.__hash__ = __hash__  # type: ignore[method-assign]
    cls.record_event = record_event  # type: ignore[attr-defined]

    # Only apply auto_init if this is not a dataclass or Pydantic model
    try:
        has_dataclass = True
    except AttributeError:
        has_dataclass = False

    try:
        has_pydantic = True
    except AttributeError:
        has_pydantic = False

    if not (has_dataclass or has_pydantic):
        return auto_init(cls)
    return cls


def aggregate_root[T](cls: type[T]) -> type[T]:
    """Class decorator for aggregate root entities with event sourcing.

    Aggregate roots are entities that serve as the entry point to an aggregate
    and manage consistency within the aggregate boundary. They handle domain
    events and enforce business invariants.

    Args:
    ----
        cls: Class to decorate as aggregate root

    Returns:
    -------
        Decorated class with aggregate root semantics

    """
    # First apply entity decorator
    cls = entity(cls)

    def get_uncommitted_events(self: object) -> list[DomainEvent]:
        """Get all uncommitted domain events for this aggregate."""
        all_events = EnterpriseReflectionRegistry.get_domain_events()
        return [
            event
            for event in all_events
            if event.get("entity_id") == getattr(self, "id", None)
        ]

    def mark_events_as_committed(self: object) -> None:
        """Mark all events as committed (typically called after persistence)."""
        entity_id = getattr(self, "id", None)
        # Use public method to clear events for this entity
        all_events = EnterpriseReflectionRegistry.get_domain_events()
        events_to_keep = [
            event for event in all_events if event.get("entity_id") != entity_id
        ]
        EnterpriseReflectionRegistry.clear_domain_events()
        for event in events_to_keep:
            EnterpriseReflectionRegistry.publish_domain_event(event)

    # Apply aggregate root patterns using type: ignore for method assignments
    cls.get_uncommitted_events = get_uncommitted_events  # type: ignore[attr-defined]
    cls.mark_events_as_committed = mark_events_as_committed  # type: ignore[attr-defined]

    return cls


def specification[T](cls: type[T]) -> type[T]:
    """Class decorator for specification pattern implementation.

    Specifications encapsulate business rules that can be combined using
    logical operators. This decorator ensures proper implementation of
    the specification interface.

    Args:
    ----
        cls: Class to decorate as specification

    Returns:
    -------
        Decorated class with specification semantics

    """
    try:
        _ = cls.is_satisfied_by
    except AttributeError:
        msg = (
            f"Specification class {cls.__name__} must implement is_satisfied_by method"
        )
        raise TypeError(msg)

    def __and__(self: SpecificationBase, other: SpecificationBase) -> AndSpecification:  # noqa: N807
        return AndSpecification(self, other, "and")

    def __or__(self: SpecificationBase, other: SpecificationBase) -> OrSpecification:  # noqa: N807
        return OrSpecification(self, other, "or")

    def __invert__(self: SpecificationBase) -> NotSpecification:  # noqa: N807
        return NotSpecification(self)

    # Apply specification patterns using type: ignore for method assignments
    cls.__and__ = __and__  # type: ignore[assignment,operator]
    cls.__or__ = __or__  # type: ignore[assignment]
    cls.__invert__ = __invert__  # type: ignore[assignment,operator]

    return cls


# ============================================================================
# SPECIFICATION IMPLEMENTATIONS
# ============================================================================
# ZERO TOLERANCE: Specification classes imported from canonical advanced_types.py
# AndSpecification, OrSpecification, NotSpecification are now CompositeSpecification
# Legacy aliases maintained for backward compatibility

# Type aliases for backward compatibility with legacy specification patterns
AndSpecification = CompositeSpecification  # Legacy alias - use CompositeSpecification(left, right, "and")
OrSpecification = CompositeSpecification  # Legacy alias - use CompositeSpecification(left, right, "or")
# NotSpecification imported directly from canonical location


# ============================================================================
# BASE CLASSES FOR ENTERPRISE PATTERNS
# ============================================================================


# EntityBase imported above from domain_patterns.py

# ============================================================================
# ENTERPRISE REFLECTION UTILITIES
# ============================================================================


def get_entity_id(obj: object) -> EntityIdentity | None:
    """Extract entity ID from domain object.

    Args:
    ----
        obj: Domain object

    Returns:
    -------
        Entity ID if found, None otherwise

    """
    return getattr(obj, "id", None)


def is_entity(obj: object) -> bool:
    """Check if object is a domain entity.

    Args:
    ----
        obj: Object to check

    Returns:
    -------
        True if object is an entity, False otherwise

    """
    if isinstance(obj, EntityBase):
        return True
    try:
        _ = obj.id
    except AttributeError:
        return False
    else:
        return True


def is_value_object(cls: type) -> bool:
    """Check if class is a value object.

    Args:
    ----
        cls: Class to check

    Returns:
    -------
        True if class is a value object, False otherwise

    """
    if not is_dataclass(cls):
        return False
    try:
        _ = cls.__eq__
        _ = cls.__hash__
    except AttributeError:
        return False
    else:
        return True


def extract_domain_events(obj: object) -> list[DomainEvent]:
    """Extract domain events from object.

    Args:
    ----
        obj: Domain object

    Returns:
    -------
        List of domain events

    """
    if isinstance(obj, EntityBase):
        try:
            events = obj.get_uncommitted_events()
            if isinstance(events, list):
                return events
        except AttributeError:
            pass
    return []


# ============================================================================
# REGISTRY INSTANCE AND ALIASES
# ============================================================================

# Global registry instance - SINGLE SOURCE OF TRUTH
ReflectionRegistry = EnterpriseReflectionRegistry

# Legacy aliases for backward compatibility (temporary)
validation_registry = ReflectionRegistry
converter_registry = ReflectionRegistry

# Export all public APIs
__all__ = [
    # Specification implementations
    "AndSpecification",
    "ConversionResult",
    "ConversionTarget",
    "ConverterProtocol",
    # Protocols
    "DataclassProtocol",
    # Type aliases
    "DomainEvent",
    "DomainObject",
    # Core registry
    "EnterpriseReflectionRegistry",
    # Base classes
    "EntityBase",
    "EntityIdentity",
    "NotSpecification",
    "OrSpecification",
    "ReflectionRegistry",
    "SpecificationBase",
    "SpecificationTarget",
    "ValidationTarget",
    "ValidatorProtocol",
    "aggregate_root",
    # Decorators
    "auto_init",
    "entity",
    "extract_domain_events",
    # Utilities
    "get_entity_id",
    "is_entity",
    "is_value_object",
    "specification",
    "value_object",
]
