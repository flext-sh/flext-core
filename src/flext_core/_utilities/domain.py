"""Domain helper utilities for entities, value objects, and aggregates.

The helpers consolidate common DDD checks so domain services and dispatcher
handlers can validate identity and immutability without duplicating boilerplate
logic.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Hashable

from flext_core.constants import c
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextUtilitiesDomain:
    """Reusable DDD helpers for dispatcher-driven domain workflows."""

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance with all logging methods (debug, info, warning, error, etc).
        Uses same structure/config as FlextLogger but without circular import.
        """
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def compare_entities_by_id(
        entity_a: p.HasModelDump,
        entity_b: p.HasModelDump,
        id_attr: str = c.Mixins.FIELD_ID,
    ) -> bool:
        """Compare two entities by their unique ID attribute.

        Generic comparison for DDD entities - compares by identity, not by value.

        Args:
            entity_a: First entity
            entity_b: Second entity
            id_attr: Attribute name for unique ID (default: "unique_id")

        Returns:
            True if same entity (same ID and type), False otherwise

        Example:
            >>> user1 = User(unique_id="123", name="Alice")
            >>> user2 = User(unique_id="123", name="Bob")  # Same ID
            >>> FlextUtilitiesDomain.compare_entities_by_id(user1, user2)
            True

        """
        if not isinstance(entity_b, entity_a.__class__):
            return False

        id_a = getattr(entity_a, id_attr, None)
        id_b = getattr(entity_b, id_attr, None)

        return id_a is not None and id_a == id_b

    @staticmethod
    def hash_entity_by_id(
        entity: p.HasModelDump,
        id_attr: str = c.Mixins.FIELD_ID,
    ) -> int:
        """Generate hash for entity based on unique ID and type.

        Generic hashing for DDD entities - uses identity (ID + type), not value.

        Args:
            entity: Entity to hash
            id_attr: Attribute name for unique ID (default: "unique_id")

        Returns:
            Hash value based on entity ID and type

        Example:
            >>> user = User(unique_id="123", name="Alice")
            >>> hash_val = FlextUtilitiesDomain.hash_entity_by_id(user)

        """
        entity_id = getattr(entity, id_attr, None)
        if entity_id is None:
            return hash(id(entity))  # Fallback to object ID if no unique_id

        # Combine type and ID for hash
        return hash((entity.__class__.__name__, entity_id))

    @staticmethod
    def compare_value_objects_by_value(obj_a: object, obj_b: object) -> bool:
        """Compare two value objects by their values (all attributes).

        Generic comparison for DDD Value Objects - compares by value, not identity.

        Args:
            obj_a: First value object
            obj_b: Second value object

        Returns:
            True if same values (same type and all attributes equal)

        Example:
            >>> addr1 = Address(street="123 Main", city="NYC")
            >>> addr2 = Address(street="123 Main", city="NYC")
            >>> FlextUtilitiesDomain.compare_value_objects_by_value(addr1, addr2)
            True

        """
        if not isinstance(obj_b, obj_a.__class__):
            return False

        # Try Pydantic model_dump first (using protocol for type safety)
        if isinstance(obj_a, p.HasModelDump) and isinstance(
            obj_b,
            p.HasModelDump,
        ):
            try:
                dump_a = obj_a.model_dump()
                dump_b = obj_b.model_dump()
                return bool(dump_a == dump_b)
            except (AttributeError, TypeError, RuntimeError):
                pass

        # Try __dict__ comparison
        try:
            return obj_a.__dict__ == obj_b.__dict__
        except (AttributeError, TypeError):
            # Fallback to repr comparison
            return repr(obj_a) == repr(obj_b)

    @staticmethod
    def hash_value_object_by_value(obj: object) -> int:
        """Generate hash for value object based on all attribute values.

        Generic hashing for DDD Value Objects - uses values, not identity.

        Args:
            obj: Value object to hash

        Returns:
            Hash value based on all object attributes

        Example:
            >>> addr = Address(street="123 Main", city="NYC")
            >>> hash_val = FlextUtilitiesDomain.hash_value_object_by_value(addr)

        """
        # Try Pydantic model_dump first (using protocol for type safety)
        if isinstance(obj, p.HasModelDump):
            try:
                data = obj.model_dump()
                # Convert to hashable tuple of items
                return hash(tuple(sorted(data.items())))
            except (AttributeError, TypeError, RuntimeError):
                pass

        # Try __dict__
        try:
            obj_dict = obj.__dict__
            # Filter out non-hashable values and convert to tuple
            hashable_items: list[tuple[str, object]] = []
            for key, value in sorted(obj_dict.items()):
                if isinstance(value, Hashable):
                    hashable_items.append((key, value))
                else:
                    # Use repr for non-hashable values
                    hashable_items.append((key, repr(value)))

            return hash(tuple(hashable_items))
        except (AttributeError, TypeError):
            # Fallback to repr hash
            return hash(repr(obj))

    @staticmethod
    def validate_entity_has_id(
        entity: object,
        id_attr: str = c.Mixins.FIELD_ID,
    ) -> bool:
        """Validate that entity has a non-None unique ID.

        Args:
            entity: Entity to validate
            id_attr: Attribute name for unique ID (default: "unique_id")

        Returns:
            True if entity has non-None ID, False otherwise

        """
        entity_id = getattr(entity, id_attr, None)
        return bool(entity_id)

    @staticmethod
    def validate_value_object_immutable(obj: t.GeneralValueType) -> bool:
        """Check if value object appears to be immutable (frozen).

        Args:
            obj: Value object to check

        Returns:
            True if appears immutable (frozen=True or no __setattr__)

        """
        # Check Pydantic frozen config
        if hasattr(obj, "model_config"):
            try:
                config = getattr(obj, "model_config", {})
                if FlextRuntime.is_dict_like(config) and config.get("frozen"):
                    return True
            except (AttributeError, TypeError):
                pass

        # Check if __setattr__ is overridden to prevent mutation
        if hasattr(obj, "__setattr__"):
            # If __setattr__ is from object class, it's mutable
            # Custom __setattr__ might enforce immutability
            setattr_method = getattr(type(obj), "__setattr__", None)
            return setattr_method is not object.__setattr__

        return False


__all__ = [
    "FlextUtilitiesDomain",
]
