"""Domain helper utilities for entities, value objects, and aggregates.

The helpers consolidate common DDD checks so domain services and dispatcher
handlers can validate identity and immutability without duplicating boilerplate
logic.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_core.constants import c
from flext_core.typings import t


class FlextUtilitiesDomain:
    """Reusable DDD helpers for dispatcher-driven domain workflows."""

    @staticmethod
    def same_type(obj_a: object, obj_b: object) -> bool:
        """Exact-type identity comparison (no MRO traversal).

        Returns True only when both objects are the exact same concrete type.
        """
        return type(obj_a) is type(obj_b)

    @staticmethod
    def _get_obj_dict(obj: t.RuntimeData) -> Mapping[str, t.NormalizedValue] | None:
        """Extract __dict__ safely, returning None on failure."""
        try:
            return obj.__dict__
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _to_hashable(value: t.NormalizedValue) -> t.NormalizedValue:
        """Coerce a value to something hashable for dict-based hashing."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return value.__class__.__name__

    @staticmethod
    def compare_entities_by_id(
        entity_a: t.RuntimeData,
        entity_b: t.RuntimeData,
        id_attr: str = c.FIELD_ID,
    ) -> bool:
        """Compare two entities by unique ID (identity, not value).

        Returns True if both entities have same type and ID.
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
        """Compare two value objects by all attributes (value, not identity).

        Returns True if same type and all attributes equal.
        """
        if not FlextUtilitiesDomain.same_type(obj_b, obj_a):
            return False
        dict_a = FlextUtilitiesDomain._get_obj_dict(obj_a)
        dict_b = FlextUtilitiesDomain._get_obj_dict(obj_b)
        if dict_a is not None and dict_b is not None:
            return dict_a == dict_b
        return repr(obj_a) == repr(obj_b)

    @staticmethod
    def hash_entity_by_id(
        entity: t.RuntimeData,
        id_attr: str = c.FIELD_ID,
    ) -> int:
        """Hash entity by ID + type. Falls back to identity hash if ID missing."""
        entity_id = getattr(entity, id_attr, None)
        if entity_id is None:
            return hash(id(entity))
        return hash((entity.__class__.__name__, entity_id))

    @staticmethod
    def hash_value_object_by_value(obj: t.RuntimeData) -> int:
        """Hash value object by all attributes. Falls back to repr hash."""
        obj_dict = FlextUtilitiesDomain._get_obj_dict(obj)
        if obj_dict is None:
            return hash(repr(obj))
        items: Sequence[tuple[str, t.NormalizedValue]] = [
            (str(k), FlextUtilitiesDomain._to_hashable(v))
            for k, v in sorted(obj_dict.items())
        ]
        return hash(tuple(items))


__all__ = ["FlextUtilitiesDomain"]
