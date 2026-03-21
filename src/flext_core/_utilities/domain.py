"""Domain helper utilities for entities, value objects, and aggregates.

The helpers consolidate common DDD checks so domain services and dispatcher
handlers can validate identity and immutability without duplicating boilerplate
logic.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextRuntime, c, p, t


class FlextUtilitiesDomain:
    """Reusable DDD helpers for dispatcher-driven domain workflows."""

    @property
    def logger(self) -> p.Logger:
        """Get structlog logger via FlextRuntime (infrastructure-level, no FlextLogger)."""
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def same_type(
        obj_a: t.RuntimeData,
        obj_b: t.RuntimeData,
    ) -> bool:
        return isinstance(obj_a, type(obj_b))

    @staticmethod
    def compare_entities_by_id(
        entity_a: t.RuntimeData,
        entity_b: t.RuntimeData,
        id_attr: str = c.FIELD_ID,
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
        if not FlextUtilitiesDomain.same_type(entity_b, entity_a):
            return False
        id_a = getattr(entity_a, id_attr, None)
        id_b = getattr(entity_b, id_attr, None)
        return id_a is not None and id_a == id_b

    @staticmethod
    def compare_value_objects_by_value(
        obj_a: t.RuntimeData,
        obj_b: t.RuntimeData,
    ) -> bool:
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
        if not FlextUtilitiesDomain.same_type(obj_b, obj_a):
            return False
        try:
            return obj_a.__dict__ == obj_b.__dict__
        except (AttributeError, TypeError):
            return repr(obj_a) == repr(obj_b)

    @staticmethod
    def hash_entity_by_id(
        entity: t.RuntimeData,
        id_attr: str = c.FIELD_ID,
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
            return hash(id(entity))
        return hash((entity.__class__.__name__, entity_id))

    @staticmethod
    def hash_value_object_by_value(obj: t.RuntimeData) -> int:
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
        try:
            obj_dict = obj.__dict__
            hashable_items: list[tuple[str, t.NormalizedValue]] = []
            for key, value in sorted(obj_dict.items()):
                key_str = str(key)
                if isinstance(value, (str, int, float, bool, type(None))):
                    hashable_items.append((key_str, value))
                else:
                    hashable_items.append((key_str, value.__class__.__name__))
            return hash(tuple(hashable_items))
        except (AttributeError, TypeError):
            return hash(repr(obj))

    @staticmethod
    def validate_entity_has_id(
        entity: t.RuntimeData,
        id_attr: str = c.FIELD_ID,
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
    def validate_value_object_immutable(
        obj: t.RuntimeData,
    ) -> bool:
        """Check if value object appears to be immutable (frozen).

        Args:
            obj: Value object to check

        Returns:
            True if appears immutable (frozen=True or no __setattr__)

        """
        if hasattr(obj, "model_config"):
            try:
                config = getattr(obj, "model_config", {})
                if FlextRuntime.is_dict_like(config) and config.get("frozen"):
                    return True
            except (AttributeError, TypeError):
                pass
        if hasattr(obj, "__setattr__"):
            setattr_method = getattr(obj.__class__, "__setattr__", None)
            return setattr_method is not object.__setattr__
        return False


__all__ = ["FlextUtilitiesDomain"]
