"""FLEXT Core Value Objects Module.

Comprehensive Domain-Driven Design (DDD) value object implementation with immutability,
validation, and rich behavior composition. Implements consolidated architecture with
Pydantic validation, mixin inheritance, and type-safe operations.

Architecture:
    - Domain-Driven Design value object patterns with attribute-based equality
    - Pydantic BaseModel integration for automatic validation and serialization
    - Multiple inheritance from specialized mixin classes for behavior composition
    - Immutable value objects with comprehensive validation and formatting
    - Payload integration for transport and persistence with metadata enrichment
    - Utility inheritance for formatting and generation capabilities

Value Object System Components:
    - FlextValueObject: Abstract base value object with validation and behavior
    - FlextValueObjectFactory: Factory pattern for type-safe value object creation
    - Validation integration: Domain rule validation with FlextResult patterns
    - Payload conversion: Rich payload creation with metadata and correlation
    - Utility inheritance: Direct access to formatting and generation utilities

Maintenance Guidelines:
    - Create domain value objects by inheriting from FlextValueObject abstract base
    - Implement validate_domain_rules method for value-specific business validation
    - Use immutable value objects with attribute-based equality comparisons
    - Leverage inherited utilities for formatting and data generation needs
    - Implement payload conversion for transport and persistence scenarios
    - Follow DDD principles with rich value behaviors and encapsulated logic

Design Decisions:
    - Abstract base class pattern enforcing domain validation implementation
    - Pydantic frozen models for immutability and thread safety
    - Multiple inheritance from utility classes for direct method access
    - Payload integration for structured transport with rich metadata
    - Factory pattern for validated value object creation with defaults
    - FlextResult pattern integration for type-safe error handling

Domain-Driven Design Features:
    - Attribute-based equality following DDD value object principles
    - Immutable value objects preventing modification after creation
    - Rich domain behaviors through multiple inheritance composition
    - Business rule validation through abstract validate_domain_rules method
    - Value object equality based on structural content rather than identity
    - Comprehensive validation ensuring value object integrity

Utility Integration:
    - Direct inheritance from FlextFormatters for data presentation
    - Direct inheritance from FlextGenerators for ID and timestamp generation
    - Mixin inheritance for value object specific behaviors
    - Logging integration through FlextLoggableMixin
    - Validation integration through FlextValueObjectMixin

Dependencies:
    - pydantic: Data validation and immutable model configuration
    - mixins: Value object specific behavior inheritance and logging
    - payload: FlextPayload integration for transport and persistence
    - result: FlextResult pattern for consistent error handling
    - utilities: Direct inheritance of formatting and generation utilities

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict

from flext_core.fields import FlextFields
from flext_core.loggings import FlextLoggerFactory
from flext_core.mixins import FlextLoggableMixin, FlextValueObjectMixin
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult
from flext_core.types import TAnyDict
from flext_core.utilities import FlextFormatters, FlextGenerators

if TYPE_CHECKING:
    from flext_core.types import TAnyDict

# FlextLogger imported for class methods only - instance methods use FlextLoggableMixin


# =============================================================================
# FLEXT VALUE OBJECT - Public DDD Value Object implementation
# =============================================================================


class FlextValueObject(
    BaseModel,
    FlextValueObjectMixin,
    FlextLoggableMixin,
    FlextFormatters,
    FlextGenerators,
    ABC,
):
    """Abstract DDD value object with immutability, validation, and rich behavior.

    Comprehensive value object implementation providing attribute-based equality,
    immutable design, and rich behavior composition through multiple inheritance.
    Combines Pydantic validation with DDD principles and utility integration.

    Architecture:
        - Abstract base class enforcing domain validation implementation
        - Pydantic BaseModel for automatic validation and serialization
        - Multiple inheritance from specialized mixin and utility classes
        - Frozen configuration for immutability and thread safety
        - Rich behavior composition through utility class inheritance

    Value Object Principles:
        - Attribute-based equality rather than identity-based comparison
        - Immutability preventing modification after creation
        - Value semantics with structural equality comparisons
        - Rich domain behaviors through method composition
        - Business rule validation through abstract method enforcement

    Multiple Inheritance Composition:
        - FlextValueObjectMixin: Value object specific behaviors and equality
        - FlextLoggableMixin: Structured logging with context management
        - FlextFormatters: Direct access to data formatting utilities
        - FlextGenerators: Direct access to ID and timestamp generation
        - ABC: Abstract base class pattern for interface enforcement

    Validation Integration:
        - Abstract validate_domain_rules method enforcing implementation
        - validate_flext method for comprehensive validation with logging
        - FlextResult pattern for type-safe error handling
        - Automatic validation integration with factory methods
        - Business rule validation separate from data validation

    Usage Patterns:
        # Define domain value object
        class EmailAddress(FlextValueObject):
            email: str

            def validate_domain_rules(self) -> FlextResult[None]:
                if not self.email or "@" not in self.email:
                    return FlextResult.fail("Invalid email format")
                if len(self.email) > 254:
                    return FlextResult.fail("Email too long")
                return FlextResult.ok(None)

        class Money(FlextValueObject):
            amount: Decimal
            currency: str = "USD"

            def validate_domain_rules(self) -> FlextResult[None]:
                if self.amount < 0:
                    return FlextResult.fail("Amount cannot be negative")
                if self.currency not in ["USD", "EUR", "GBP"]:
                    return FlextResult.fail("Unsupported currency")
                return FlextResult.ok(None)

            def add(self, other: Money) -> FlextResult[Money]:
                if self.currency != other.currency:
                    return FlextResult.fail("Currency mismatch")
                return FlextResult.ok(Money(
                    amount=self.amount + other.amount,
                    currency=self.currency
                ))

        # Create value objects
        email_result = EmailAddress(email="user@example.com").validate_flext()
        if email_result.is_success:
            email = email_result.data

        # Value object equality
        money1 = Money(amount=Decimal("10.00"))
        money2 = Money(amount=Decimal("10.00"))
        if money1 != money2:  # Attribute-based equality
            raise AssertionError(
                f"Expected {money2}, got {money1}"
            )

        # Payload conversion for transport
        payload = email.to_payload()
        # Rich metadata and formatting included

    Utility Integration:
        - Direct access to formatting methods from FlextFormatters
        - Direct access to generation methods from FlextGenerators
        - Inherited string representation with formatted output
        - Logging integration with automatic context management
        - Validation integration with comprehensive error reporting

    Payload Conversion:
        - Rich payload creation with metadata enrichment
        - Correlation ID generation for request tracking
        - Validation status inclusion in payload metadata
        - Formatted data presentation through inherited utilities
        - Fallback payload creation for error scenarios
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        extra="forbid",
    )

    def __hash__(self) -> int:
        """Generate hash from hashable field values.

        Handles unhashable types like lists and dicts by converting them to hashable
        representations.
        """

        def make_hashable(item: object) -> object:
            """Convert item to hashable representation."""
            if isinstance(item, dict):
                return frozenset((k, make_hashable(v)) for k, v in item.items())
            if isinstance(item, list):
                return tuple(make_hashable(i) for i in item)
            if isinstance(item, set):
                return frozenset(make_hashable(i) for i in item)
            return item

        # Get all field values excluding domain_events if present
        values = []
        for name, value in self.model_dump().items():
            if name != "domain_events":  # Exclude non-hashable internal fields
                values.append((name, make_hashable(value)))

        return hash(tuple(sorted(values)))

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Track ValueObject subclasses for type safety.

        Args:
            **kwargs: Additional keyword arguments

        """
        super().__init_subclass__()
        # Use FlextLogger directly for class methods
        logger = FlextLoggerFactory.get_logger(f"{cls.__module__}.{cls.__name__}")
        logger.debug(
            "ValueObject subclass created",
            class_name=cls.__name__,
            module=cls.__module__,
        )

    @abstractmethod
    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate value object-specific business rules.

        Must return FlextResult for consistent error handling.
        """

    def validate_flext(self) -> FlextResult[Self]:
        """Validate value object with FLEXT validation system.

        Renamed from 'validate' to avoid conflict with Pydantic's validate method.

        Returns:
            FlextResult with validation result

        """
        # Use FlextValidation for comprehensive validation
        validation_result = self.validate_domain_rules()
        if validation_result.is_failure:
            self.logger.warning(
                "ValueObject validation failed",
                value_object_type=self.__class__.__name__,
                error=validation_result.error,
            )
            return FlextResult.fail(validation_result.error or "Validation failed")

        self.logger.debug(
            "ValueObject validated successfully",
            value_object_type=self.__class__.__name__,
        )
        return FlextResult.ok(self)

    def validate_field(self, field_name: str, field_value: object) -> FlextResult[None]:
        """Validate a specific field using the fields system.

        Args:
            field_name: Name of the field to validate
            field_value: Value to validate

        Returns:
            Result of field validation

        """
        try:
            # Try to get field definition from registry
            field_result = FlextFields.get_field_by_name(field_name)
            if field_result.is_success:
                field_def = field_result.unwrap()
                validation_result = field_def.validate_value(field_value)
                if validation_result.is_success:
                    return FlextResult.ok(None)
                return FlextResult.fail(
                    validation_result.error or "Field validation failed",
                )

            # If no field definition found, return success (allow other validation)
            return FlextResult.ok(None)
        except (ImportError, AttributeError, ValueError) as e:
            return FlextResult.fail(f"Field validation error: {e}")

    def validate_all_fields(self) -> FlextResult[None]:
        """Validate all value object fields using the fields system.

        Automatically validates all model fields that have corresponding
        field definitions in the fields registry.

        Returns:
            Result of comprehensive field validation

        """
        errors = []

        # Get all model fields and their values
        model_data = self.model_dump()

        for field_name, field_value in model_data.items():
            # Skip internal fields
            if field_name.startswith("_"):
                continue

            validation_result = self.validate_field(field_name, field_value)
            if validation_result.is_failure:
                errors.append(f"{field_name}: {validation_result.error}")

        if errors:
            return FlextResult.fail(f"Field validation errors: {'; '.join(errors)}")

        return FlextResult.ok(None)

    def format_dict(self, data: dict[str, object]) -> str:
        """Format dictionary for string representation."""
        formatted_items = []
        for key, value in data.items():
            if isinstance(value, str):
                formatted_items.append(f"{key}='{value}'")
            else:
                formatted_items.append(f"{key}={value}")
        return ", ".join(formatted_items)

    def __str__(self) -> str:
        """Return string representation using inherited formatters."""
        # Use inherited formatting method directly
        fields = self.format_dict(self.model_dump())
        return f"{self.__class__.__name__}({fields})"

    def to_payload(self) -> FlextPayload[TAnyDict]:
        """Convert to FlextPayload for transport using orchestrated patterns.

        This demonstrates complex functionality using multiple base modules
        rather than simple delegation.
        """
        # COMPLEX ORCHESTRATION: Multiple base patterns combined

        # 1. Use base validators for pre-validation
        domain_validation = self.validate_domain_rules()
        if domain_validation.is_failure:
            self.logger.warning(
                "Value object validation failed during payload conversion",
                error=domain_validation.error,
            )
            # Continue but mark as potentially invalid

        # 2. Generate comprehensive metadata using base generators
        payload_metadata = {
            "type": f"ValueObject.{self.__class__.__name__}",
            "timestamp": FlextGenerators.generate_timestamp(),
            "correlation_id": FlextGenerators.generate_correlation_id(),
            "validated": domain_validation.is_success,
        }

        # 3. Use base formatters for data preparation
        raw_data = self.model_dump()
        formatted_data = self.format_dict(raw_data)

        # 4. Build comprehensive payload data
        payload_data: TAnyDict = {
            "value_object_data": formatted_data,
            "metadata": payload_metadata,
            "class_info": {
                "name": self.__class__.__name__,
                "module": self.__class__.__module__,
            },
        }

        # 5. Create payload with validation
        payload_result = FlextPayload.create(
            data=payload_data,
            **payload_metadata,
        )

        if payload_result.is_failure:
            self.logger.error(
                "Failed to create payload for value object",
                error=payload_result.error,
            )
            # Fallback to minimal payload
            fallback_data: TAnyDict = {
                "error": "Payload creation failed",
                "raw_data": raw_data,
            }
            return FlextPayload.create(data=fallback_data).unwrap()

        return payload_result.unwrap()


# =============================================================================
# FACTORY METHODS - Convenience builders for ValueObjects
# =============================================================================


class FlextValueObjectFactory:
    """Factory pattern for type-safe value object creation with validation and defaults.

    Comprehensive factory implementation providing type-safe value object creation with
    default value management and domain validation. Implements factory pattern with
    FlextResult integration for consistent error handling and reliability.

    Architecture:
        - Static factory methods for stateless value object creation
        - Default value support for consistent value object initialization
        - Type-safe factory functions with generic return types
        - Domain validation integration through value object validate_domain_rules
        - FlextResult pattern integration for error handling

    Factory Features:
        - Dynamic factory function creation for any value object type
        - Default value merging with parameter override capability
        - Automatic domain validation execution before value object return
        - Error handling with detailed failure messages and type information
        - Type-safe creation with compile-time verification

    Value Object Creation Process:
        - Default value application for consistent initialization
        - Parameter override support for customization and specialization
        - Value object instantiation with merged parameters
        - Domain validation execution ensuring business rule compliance
        - FlextResult wrapping for type-safe error handling and reporting
        - Comprehensive error messages including type and validation information

    Usage Patterns:
        # Create factory for EmailAddress value object
        email_factory = FlextValueObjectFactory.create_value_object_factory(
            EmailAddress,
            defaults={
                "domain_validation": True,
                "normalize": True
            }
        )

        # Use factory to create value objects
        email_result = email_factory(
            email="john@example.com"
        )

        if email_result.is_success:
            email_address = email_result.data
            # Domain validation already executed
            # Defaults applied automatically

        # Factory for Money value object with currency defaults
        money_factory = FlextValueObjectFactory.create_value_object_factory(
            Money,
            defaults={
                "currency": "USD",
                "precision": 2
            }
        )

        # Create money with default currency
        price_result = money_factory(amount=Decimal("29.99"))

        # Override defaults when needed
        euro_result = money_factory(
            amount=Decimal("25.50"),
            currency="EUR"  # Overrides USD default
        )

        # Handle factory creation errors
        if price_result.is_failure:
            logger.error(f"Failed to create price: {price_result.error}")

    Factory Pattern Benefits:
        - Consistent value object creation with validation
        - Default value management across value object instances
        - Type-safe creation with compile-time verification
        - Error handling with detailed failure information
        - Reduced boilerplate code for value object instantiation
        - Centralized validation and creation logic
    """

    @staticmethod
    def create_value_object_factory(
        value_object_class: type[FlextValueObject],
        defaults: TAnyDict | None = None,
    ) -> object:
        """Create a factory function for value objects.

        Args:
            value_object_class: Value object class to create
            defaults: Default values for the factory

        Returns:
            Factory function that returns FlextResult

        """

        def factory(
            **kwargs: object,
        ) -> FlextResult[FlextValueObject]:
            try:
                data = {**(defaults or {}), **kwargs}
                instance = value_object_class(**data)
                validation_result = instance.validate_domain_rules()
                if validation_result.is_failure:
                    return FlextResult.fail(
                        validation_result.error or "Validation failed",
                    )
                return FlextResult.ok(instance)
            except (TypeError, ValueError) as e:
                return FlextResult.fail(
                    f"Failed to create {value_object_class.__name__}: {e}",
                )

        return factory


# Export API
__all__ = ["FlextValueObject", "FlextValueObjectFactory"]
