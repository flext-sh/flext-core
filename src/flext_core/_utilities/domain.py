"""Domain helper utilities for entities, value objects, and aggregates.

The helpers consolidate common DDD checks so domain services and dispatcher
handlers can validate identity and immutability without duplicating boilerplate
logic.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING

from flext_core import (
    FlextModelsBase as m,
    FlextModelsContainers as mc,
    FlextModelsDomainEvent as mde,
    FlextUtilitiesGuards as u,
    c,
    t,
)

if TYPE_CHECKING:
    from flext_core import FlextProtocolsBase as pb


class FlextUtilitiesDomain:
    """Reusable DDD helpers for dispatcher-driven domain workflows."""

    @staticmethod
    def same_type(obj_a: t.JsonPayload, obj_b: t.JsonPayload) -> bool:
        """Exact-type identity comparison (no MRO traversal).

        Returns True only when both objects are the exact same concrete type.
        """
        return obj_a.__class__ is obj_b.__class__

    @staticmethod
    def _get_obj_dict(obj: t.JsonPayload) -> t.JsonMapping | None:
        """Extract __dict__ safely, returning None on failure."""
        try:
            return obj.__dict__
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _to_hashable(value: t.JsonValue) -> t.JsonValue:
        """Coerce a value to something hashable for dict-based hashing."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return value.__class__.__name__

    @staticmethod
    def compare_entities_by_id(
        entity_a: t.JsonPayload,
        entity_b: t.JsonPayload,
        id_attr: str = c.FIELD_ID,
    ) -> bool:
        """Compare two entities by unique ID (identity, not value).

        Returns True if both entities have same type and ID.
        """
        invalid_entity = u.scalar(entity_a) or isinstance(entity_a, (Sequence, Mapping))
        invalid_other = u.scalar(entity_b) or isinstance(entity_b, (Sequence, Mapping))
        if (
            invalid_entity
            or invalid_other
            or not FlextUtilitiesDomain.same_type(entity_b, entity_a)
        ):
            result = False
        else:
            id_a = getattr(entity_a, id_attr, None)
            id_b = getattr(entity_b, id_attr, None)
            result = id_a is not None and id_a == id_b
        return result

    @staticmethod
    def compare_value_objects_by_value(
        obj_a: t.JsonPayload,
        obj_b: t.JsonPayload,
    ) -> bool:
        """Compare two value objects by all attributes (value, not identity).

        Returns True if same type and all attributes equal.
        """
        if u.scalar(obj_a):
            result = obj_a == obj_b
        elif u.scalar(obj_b):
            result = False
        else:
            obj_a_iterable = hasattr(obj_a, "__iter__") and not hasattr(
                obj_a, "model_dump"
            )
            obj_b_iterable = hasattr(obj_b, "__iter__") and not hasattr(
                obj_b, "model_dump"
            )
            if obj_a_iterable or obj_b_iterable:
                result = obj_a == obj_b
            elif not FlextUtilitiesDomain.same_type(obj_b, obj_a):
                result = False
            elif isinstance(obj_a, m.EnforcedModel) and isinstance(
                obj_b, m.EnforcedModel
            ):
                result = obj_a.model_dump() == obj_b.model_dump()
            else:
                dict_a = FlextUtilitiesDomain._get_obj_dict(obj_a)
                dict_b = FlextUtilitiesDomain._get_obj_dict(obj_b)
                result = (
                    dict_a == dict_b
                    if dict_a is not None and dict_b is not None
                    else repr(obj_a) == repr(obj_b)
                )
        return result

    @staticmethod
    def hash_entity_by_id(
        entity: t.JsonPayload,
        id_attr: str = c.FIELD_ID,
    ) -> int:
        """Hash entity by ID + type. Falls back to identity hash if ID missing."""
        if u.scalar(entity):
            return hash(entity)
        entity_id = getattr(entity, id_attr, None)
        if entity_id is None:
            return hash(id(entity))
        return hash((entity.__class__.__name__, entity_id))

    @staticmethod
    def hash_value_object_by_value(obj: t.JsonPayload) -> int:
        """Hash value object by all attributes. Falls back to repr hash."""
        if u.scalar(obj):
            return hash(obj)
        if isinstance(obj, m.EnforcedModel):
            data = obj.model_dump()
            return hash(tuple(sorted((str(k), str(v)) for k, v in data.items())))
        if hasattr(obj, "__iter__"):
            return hash(repr(obj))
        obj_dict = FlextUtilitiesDomain._get_obj_dict(obj)
        if obj_dict is None:
            return hash(repr(obj))
        items: Sequence[t.Pair[str, t.JsonValue]] = [
            (str(k), FlextUtilitiesDomain._to_hashable(v))
            for k, v in sorted(obj_dict.items())
        ]
        return hash(tuple(items))

    @staticmethod
    def add_domain_event(
        entity: pb.HasDomainEvents,
        event_type: str,
        data: mc.ConfigMap | Mapping[str, t.JsonPayload | None] | None = None,
        aggregate_id: str | None = None,
    ) -> mde.Entry:
        """Create a domain event and append it to the entity's event buffer.

        Pass ``aggregate_id`` explicitly when the entity's stable identity
        differs from ``unique_id`` (e.g. a surrogate ``id`` field). Pydantic's
        ``BeforeValidator`` on ``Entry.data`` handles all normalization.
        """
        if data is None:
            normalized_data = mc.ConfigMap(root={})
        elif isinstance(data, mc.ConfigMap):
            normalized_data = data
        else:
            normalized_data = mc.ConfigMap.model_validate(data)
        entry = mde.Entry(
            event_type=event_type,
            aggregate_id=aggregate_id if aggregate_id is not None else entity.unique_id,
            data=normalized_data,
        )
        entity.domain_events.append(entry)
        return entry


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesDomain"]
