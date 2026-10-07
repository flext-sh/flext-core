"""Domain helper utilities for entities, value objects, and aggregates.

The helpers consolidate common DDD checks so domain services and dispatcher
handlers can validate identity and immutability without duplicating boilerplate
logic.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core import c, t
from flext_core._models import (
    FlextModelsBase,
    FlextModelsContainers,
    FlextModelsDomainEvent,
)
from flext_core._protocols import FlextProtocolsResult
from flext_core._utilities import FlextUtilitiesGuards

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsBase


class FlextUtilitiesDomain:
    """Reusable DDD helpers for dispatcher-driven domain workflows."""

    @staticmethod
    def same_type(
        obj_a: t.JsonPayload | FlextProtocolsResult.HasModelDump,
        obj_b: t.JsonPayload | FlextProtocolsResult.HasModelDump,
    ) -> bool:
        """Exact-type identity comparison (no MRO traversal).

        Returns True only when both objects are the exact same concrete type.

        Returns:
            The resulting ``bool``.

        """
        return type(obj_a) is type(obj_b)

    @staticmethod
    def compare_entities_by_id(
        entity_a: t.JsonPayload | FlextProtocolsResult.HasModelDump,
        entity_b: t.JsonPayload | FlextProtocolsResult.HasModelDump,
        id_attr: str = c.FIELD_ID,
    ) -> bool:
        """Compare two entities by unique ID (identity, not value).

        Returns True if both entities have same type and ID.

        Returns:
            The resulting ``bool``.

        """
        invalid_entity = FlextUtilitiesGuards.scalar(entity_a) or isinstance(
            entity_a,
            (Sequence, Mapping),
        )
        invalid_other = FlextUtilitiesGuards.scalar(entity_b) or isinstance(
            entity_b,
            (Sequence, Mapping),
        )
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
        obj_a: t.JsonPayload | FlextProtocolsResult.HasModelDump,
        obj_b: t.JsonPayload | FlextProtocolsResult.HasModelDump,
    ) -> bool:
        """Compare two value objects by all attributes (value, not identity).

        Returns True if same type and all attributes equal.

        Returns:
            The resulting ``bool``.

        """
        result: bool
        if isinstance(obj_a, c.SCALAR_TYPES):
            result = obj_a == obj_b if isinstance(obj_b, c.SCALAR_TYPES) else False
        elif FlextUtilitiesGuards.scalar(obj_b):
            result = False
        else:
            obj_a_iterable = hasattr(obj_a, "__iter__") and not hasattr(
                obj_a,
                "model_dump",
            )
            obj_b_iterable = hasattr(obj_b, "__iter__") and not hasattr(
                obj_b,
                "model_dump",
            )
            if obj_a_iterable or obj_b_iterable:
                if isinstance(obj_a, Mapping) and isinstance(obj_b, Mapping):
                    result = dict(obj_a.items()) == dict(obj_b.items())
                elif isinstance(obj_a, Sequence) and isinstance(obj_b, Sequence):
                    result = tuple(obj_a) == tuple(obj_b)
                else:
                    result = False
            elif not FlextUtilitiesDomain.same_type(obj_b, obj_a):
                result = False
            elif isinstance(obj_a, FlextModelsBase.EnforcedModel) and isinstance(
                obj_b,
                FlextModelsBase.EnforcedModel,
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
        entity: t.JsonPayload | FlextProtocolsResult.HasModelDump,
        id_attr: str = c.FIELD_ID,
    ) -> int:
        """Hash entity by ID + type. Falls back to identity hash if ID missing.

        Returns:
            The resulting ``int``.

        """
        if FlextUtilitiesGuards.scalar(entity):
            return hash(entity)
        entity_id = getattr(entity, id_attr, None)
        if entity_id is None:
            return hash(id(entity))
        return hash((FlextUtilitiesGuards.type_name(entity), entity_id))

    @staticmethod
    def hash_value_object_by_value(
        obj: t.JsonPayload | FlextProtocolsResult.HasModelDump,
    ) -> int:
        """Hash value object by all attributes. Falls back to repr hash.

        Returns:
            The resulting ``int``.

        """
        if FlextUtilitiesGuards.scalar(obj):
            return hash(obj)
        if isinstance(obj, FlextModelsBase.EnforcedModel):
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
                else FlextUtilitiesGuards.type_name(v),
            )
            for k, v in sorted(obj_dict.items())
        ]
        return hash(tuple(items))

    @staticmethod
    def add_domain_event(
        entity: FlextProtocolsBase.HasDomainEvents,
        event_type: str,
        data: FlextModelsContainers.ConfigMap
        | t.MappingKV[str, t.JsonPayload | None]
        | None = None,
        aggregate_id: str | None = None,
    ) -> FlextModelsDomainEvent.DomainEvent:
        """Create a domain event and append it to the entity's event buffer.

        Pass ``aggregate_id`` explicitly when the entity's stable identity
        differs from ``unique_id`` (e.g. a surrogate ``id`` field). Pydantic's
        ``BeforeValidator`` on ``DomainEvent.data`` handles all normalization.

        Returns:
            The resulting ``mde.DomainEvent``.

        """
        if data is None:
            normalized_data = FlextModelsContainers.ConfigMap(root={})
        elif isinstance(data, FlextModelsContainers.ConfigMap):
            normalized_data = data
        else:
            normalized_data = FlextModelsContainers.ConfigMap.model_validate(data)
        entry = FlextModelsDomainEvent.DomainEvent(
            event_type=event_type,
            aggregate_id=aggregate_id if aggregate_id is not None else entity.unique_id,
            data=normalized_data,
        )
        entity.domain_events.append(entry)
        return entry


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesDomain"]
