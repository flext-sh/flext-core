"""Immutable value objects with attribute-based equality.

Provides DDD value object implementation with immutability,
rich behaviors, and comprehensive validation patterns.

Classes:
    FlextValueObject: Abstract base for immutable values.
    FlextValueObjectFactory: Type-safe creation with validation.
    ValueObjectValidator: Validation patterns for values.
    ValueObjectFormatter: Formatting utilities.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict

from flext_core.exceptions import FlextValidationError
from flext_core.fields import FlextFields
from flext_core.loggings import FlextLoggerFactory
from flext_core.mixins import FlextLoggableMixin, FlextValueObjectMixin
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult
from flext_core.utilities import FlextFormatters, FlextGenerators

if TYPE_CHECKING:
    from flext_core.typings import TAnyDict


# =============================================================================
# FLEXT VALUE OBJECT - Public DDD Value Object implementation
# =============================================================================


class FlextValueObject(  # type: ignore[misc]
    BaseModel,
    FlextValueObjectMixin,
    FlextLoggableMixin,
    FlextFormatters,
    FlextGenerators,
    ABC,
):
    """Abstract DDD value object with immutability, validation, and rich behavior.

    Comprehensive value object implementation providing attribute-based equality,
    immutable design, and rich behavior composition. Combines Pydantic validation
    with DDD principles and utility integration for enterprise domain modeling.
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
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate value object-specific business rules.

        Must return FlextResult for consistent error handling.
        """

    def validate_flext(self) -> FlextResult[Self]:
        """Validate value object with FLEXT validation system.

        Renamed from 'validate' to avoid conflict with Pydantic's validate method.

        Returns:
            FlextResult with a validation result

        """
        # Use FlextValidation for comprehensive validation
        validation_result = self.validate_business_rules()
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

    @staticmethod
    def validate_field(field_name: str, field_value: object) -> FlextResult[None]:
        """Validate a specific field using the field's system.

        Args:
            field_name: Name of the field to validate
            field_value: Value to validate

        Returns:
            Result of field validation

        """
        # Get field definition from registry
        field_result = FlextFields.get_field_by_name(field_name)
        if field_result.success:
            field_def = field_result.unwrap()
            validation_result = field_def.validate_value(field_value)
            if validation_result.success:
                return FlextResult.ok(None)
            return FlextResult.fail(
                validation_result.error or "Field validation failed",
            )

        # If no field definition found, return success (allow another validation)
        return FlextResult.ok(None)

    def validate_all_fields(self) -> FlextResult[None]:
        """Validate all value object fields using the fields system.

        Automatically validates all model fields that have corresponding
        field definitions in the field's registry.

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

    @staticmethod
    def format_dict(data: dict[str, object]) -> str:
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

    # Back-compatible helper relied on by tests for simple dict view
    def to_dict_basic(self) -> dict[str, object]:
        """Return basic dict view of the value object (public fields only)."""
        return {k: v for k, v in self.model_dump().items() if not k.startswith("_")}

    def to_payload(self) -> FlextPayload[dict[str, str | int | float | bool | None]]:
        """Convert to FlextPayload for transport using orchestrated patterns.

        This demonstrates complex functionality using multiple base modules
        rather than simple delegation.
        """
        # COMPLEX ORCHESTRATION: Multiple base patterns combined

        # 1. Use base validators for pre-validation
        domain_validation = self.validate_business_rules()
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
            "validated": domain_validation.success,
        }

        # 3. Use base formatters for data preparation
        raw_data = self.model_dump()
        formatted_data = self.format_dict(raw_data)

        # 4. Create metadata with only compatible types
        compatible_metadata: dict[str, str | int | float | bool | None] = {}
        compatible_metadata.update(
            {
                k: v
                for k, v in payload_metadata.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            },
        )

        # 5. Build comprehensive payload data (only with compatible types)
        payload_data: dict[str, str | int | float | bool | None] = {
            "value_object_data": formatted_data,
            "class_info": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "validation_status": "valid" if domain_validation.success else "invalid",
        }

        # 6. Create payload with validation
        payload_result = FlextPayload.create(
            data=payload_data,
        )

        if payload_result.is_failure:
            self.logger.error(
                "Failed to create payload for value object",
                error=payload_result.error,
                value_object_type=self.__class__.__name__,
            )
            # REAL SOLUTION: Fix the underlying serialization issue
            # Extract only serializable attributes using proper introspection
            serializable_data = self._extract_serializable_attributes()
            corrected_result = FlextPayload.create(data=serializable_data)
            if corrected_result.is_failure:
                # If still failing, there's a deeper architectural issue
                error_msg = (
                    f"Cannot serialize value object {self.__class__.__name__}: "
                    f"{corrected_result.error}"
                )
                raise FlextValidationError(error_msg)
            return corrected_result.unwrap()

        return payload_result.unwrap()

    def _extract_serializable_attributes(
        self,
    ) -> dict[str, str | int | float | bool | None]:
        """Extract only serializable attributes from a value object.

        Proper implementation without fallback - uses type-safe introspection.
        """
        # Try Pydantic serialization first
        pydantic_result = self._try_pydantic_serialization()
        if pydantic_result:
            return pydantic_result

        # Fallback to manual extraction
        manual_result = self._try_manual_extraction()
        return manual_result or self._get_fallback_info()

    def _try_pydantic_serialization(
        self,
    ) -> dict[str, str | int | float | bool | None] | None:
        """Try Pydantic model serialization."""
        if not hasattr(self, "model_dump"):
            return None

        try:
            dumped = self.model_dump()
            return self._process_serializable_values(dumped)
        except Exception as e:
            logger = FlextLoggerFactory.get_logger(__name__)
            logger.debug("Failed to serialize Pydantic attributes: %s", e)
            return None

    def _try_manual_extraction(
        self,
    ) -> dict[str, str | int | float | bool | None]:
        """Try manual attribute extraction."""
        serializable: dict[str, str | int | float | bool | None] = {}

        for attr_name in dir(self):
            if self._should_include_attribute(attr_name):
                value = self._safely_get_attribute(attr_name)
                if value is not None:
                    serializable[attr_name] = value

        return serializable

    @staticmethod
    def _process_serializable_values(
        data: dict[str, object],
    ) -> dict[str, str | int | float | bool | None]:
        """Process values to ensure they are serializable."""
        serializable: dict[str, str | int | float | bool | None] = {}

        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                serializable[key] = value
            else:
                # Convert complex types to string representation
                serializable[key] = str(value)

        return serializable

    def _should_include_attribute(self, attr_name: str) -> bool:
        """Check if attribute should be included in serialization."""
        return not attr_name.startswith("_") and not callable(
            getattr(self, attr_name, None),
        )

    def _safely_get_attribute(self, attr_name: str) -> str | int | float | bool | None:
        """Safely get and convert attribute value."""
        try:
            value = getattr(self, attr_name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if hasattr(value, "__str__"):
                return str(value)
        except Exception as e:
            logger = FlextLoggerFactory.get_logger(__name__)
            logger.debug("Failed to extract attribute %s: %s", attr_name, e)

        return None

    def _get_fallback_info(self) -> dict[str, str | int | float | bool | None]:
        """Get fallback information when no attributes can be extracted."""
        return {
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
        }


# =============================================================================
# FACTORY METHODS - Convenience builders for ValueObjects
# =============================================================================


class FlextValueObjectFactory:
    """Factory pattern for type-safe value object creation with validation and defaults.

    Comprehensive factory implementation providing type-safe value object creation with
    default value management and domain validation. Implements factory pattern with
    FlextResult integration for consistent error handling and reliability.
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
                validation_result = instance.validate_business_rules()
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


# =============================================================================
# MODEL REBUILDS - Resolve forward references for Pydantic
# =============================================================================

# Rebuild models to resolve forward references after import
# FlextValueObject.model_rebuild()  # Disabled due to TAnyDict import issues

# Export API
__all__: list[str] = ["FlextValueObject", "FlextValueObjectFactory"]
