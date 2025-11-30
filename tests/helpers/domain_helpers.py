"""Domain-specific test helpers for FlextUtilities.Domain testing.

Provides reusable classes and methods for testing domain utilities,
reducing code duplication and improving maintainability.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from flext_core import FlextModels, FlextUtilities


class DomainTestType(StrEnum):
    """Enumeration of domain test operation types."""

    COMPARE_ENTITIES_BY_ID = "compare_entities_by_id"
    HASH_ENTITY_BY_ID = "hash_entity_by_id"
    COMPARE_VALUE_OBJECTS_BY_VALUE = "compare_value_objects_by_value"
    HASH_VALUE_OBJECT_BY_VALUE = "hash_value_object_by_value"
    VALIDATE_ENTITY_HAS_ID = "validate_entity_has_id"
    VALIDATE_VALUE_OBJECT_IMMUTABLE = "validate_value_object_immutable"


@dataclass(frozen=True, slots=True)
class DomainTestCase:
    """Test case data structure for domain utilities."""

    test_type: DomainTestType
    description: str
    input_data: dict[str, Any]
    expected_result: Any
    expected_success: bool = True
    id_attr: str = "unique_id"


class DomainTestEntity(FlextModels.Entity):
    """Test entity for domain tests."""

    name: str
    value: int


class DomainTestValue(FlextModels.Value):
    """Test value object for domain tests."""

    data: str
    count: int


class CustomEntity:
    """Custom entity with configurable ID attribute."""

    def __init__(self, custom_id: str | None = None) -> None:
        """Initialize custom entity with ID."""
        self.custom_id = custom_id


class SimpleValue:
    """Simple value object without model_dump."""

    def __init__(self, data: str) -> None:
        """Initialize simple value object."""
        self.data = data


class BadModelDump:
    """Value object that raises exception in model_dump."""

    def model_dump(self) -> dict[str, object]:
        msg = "model_dump failed"
        raise AttributeError(msg)


class ComplexValue:
    """Value object with non-hashable attributes."""

    def __init__(self, data: str, items: list[str]) -> None:
        """Initialize complex value with non-hashable items."""
        self.data = data
        self.items = items  # list is not hashable


class NoDict:
    """Object without __dict__, using __slots__."""

    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        """Initialize object without __dict__."""
        object.__setattr__(self, "value", value)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"NoDict({getattr(self, 'value', None)})"


class MutableObj:
    """Mutable object for immutability testing."""

    def __init__(self, value: int) -> None:
        """Initialize mutable object."""
        self.value = value


class ImmutableObj:
    """Immutable object with custom __setattr__."""

    _frozen: bool = True

    def __init__(self, value: int) -> None:
        """Initialize immutable object."""
        object.__setattr__(self, "value", value)

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent attribute setting if frozen."""
        if self._frozen:
            msg = "Object is frozen"
            raise AttributeError(msg)
        object.__setattr__(self, name, value)


class BadConfig:
    """Object with problematic model_config."""

    @property
    def model_config(self) -> dict[str, object]:
        msg = "Config access failed"
        raise AttributeError(msg)


class BadConfigTypeError:
    """Object with model_config that raises TypeError."""

    @property
    def model_config(self) -> dict[str, object]:
        msg = "Config type error"
        raise TypeError(msg)


class NoConfigNoSetattr:
    """Object without model_config or __setattr__."""


class NoSetattr:
    """Object without __setattr__."""


class DomainTestHelpers:
    """Generic helpers for domain utility testing."""

    @staticmethod
    def create_entity(
        name: str, value: int, *, with_id: bool = True,
    ) -> DomainTestEntity:
        """Create a test entity with optional ID."""
        entity = DomainTestEntity(name=name, value=value)
        if not with_id:
            delattr(entity, "unique_id")
        return entity

    @staticmethod
    def create_value(data: str, count: int) -> DomainTestValue:
        """Create a test value object."""
        return DomainTestValue(data=data, count=count)

    @staticmethod
    def create_custom_entity(custom_id: str) -> CustomEntity:
        """Create a custom entity with specified ID."""
        return CustomEntity(custom_id)

    @staticmethod
    def execute_domain_test(test_case: DomainTestCase) -> object:
        """Execute a domain test case and return result."""
        domain = FlextUtilities.Domain

        match test_case.test_type:
            case DomainTestType.COMPARE_ENTITIES_BY_ID:
                entity1 = test_case.input_data["entity1"]
                entity2 = test_case.input_data["entity2"]
                return domain.compare_entities_by_id(
                    entity1, entity2, id_attr=test_case.id_attr,
                )

            case DomainTestType.HASH_ENTITY_BY_ID:
                entity = test_case.input_data["entity"]
                return domain.hash_entity_by_id(entity, id_attr=test_case.id_attr)

            case DomainTestType.COMPARE_VALUE_OBJECTS_BY_VALUE:
                obj1 = test_case.input_data["obj1"]
                obj2 = test_case.input_data["obj2"]
                return domain.compare_value_objects_by_value(obj1, obj2)

            case DomainTestType.HASH_VALUE_OBJECT_BY_VALUE:
                obj = test_case.input_data["obj"]
                return domain.hash_value_object_by_value(obj)

            case DomainTestType.VALIDATE_ENTITY_HAS_ID:
                entity = test_case.input_data["entity"]
                return domain.validate_entity_has_id(entity, id_attr=test_case.id_attr)

            case DomainTestType.VALIDATE_VALUE_OBJECT_IMMUTABLE:
                obj = test_case.input_data["obj"]
                return domain.validate_value_object_immutable(obj)

        msg = f"Unknown test type: {test_case.test_type}"
        raise ValueError(msg)
