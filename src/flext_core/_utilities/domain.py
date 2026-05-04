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
    FlextConstants as c,
    FlextTypes as t,
)
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.containers import FlextModelsContainers as mc
from flext_core._models.domain_event import FlextModelsDomainEvent as mde
from flext_core._utilities.guards import FlextUtilitiesGuards as u

if TYPE_CHECKING:
    from flext_core import FlextProtocolsBase as pb


class FlextUtilitiesDomain:
    """Reusable DDD helpers for dispatcher-driven domain workflows."""

    @staticmethod
    def same_type(obj_a: t.JsonPayload, obj_b: t.JsonPayload) -> bool:
        """Exact-type identity comparison (no MRO traversal).

        Returns True only when both objects are the exact same concrete type.
        """
        return type(obj_a) is type(obj_b)

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
                try:
                    dict_a = vars(obj_a)
                except c.EXC_ATTR_TYPE:
                    dict_a = None
                try:
                    dict_b = vars(obj_b)
                except c.EXC_ATTR_TYPE:
                    dict_b = None
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
        return hash((u.type_name(entity), entity_id))

    @staticmethod
    def hash_value_object_by_value(obj: t.JsonPayload) -> int:
        """Hash value object by all attributes. Falls back to repr hash."""
        if u.scalar(obj):
            return hash(obj)
        if isinstance(obj, m.EnforcedModel):
            data = obj.model_dump()
            return hash(tuple(sorted((k, str(v)) for k, v in data.items())))
        if hasattr(obj, "__iter__"):
            return hash(repr(obj))
        try:
            obj_dict = vars(obj)
        except c.EXC_ATTR_TYPE:
            obj_dict = None
        if obj_dict is None:
            return hash(repr(obj))
        items: t.SequenceOf[t.Pair[str, t.JsonValue]] = [
            (
                k,
                v
                if isinstance(v, (str, int, float, bool, type(None)))
                else u.type_name(v),
            )
            for k, v in sorted(obj_dict.items())
        ]
        return hash(tuple(items))

    @staticmethod
    def add_domain_event(
        entity: pb.HasDomainEvents,
        event_type: str,
        data: mc.ConfigMap | t.MappingKV[str, t.JsonPayload | None] | None = None,
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
