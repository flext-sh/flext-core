"""Domain helper utilities for entities, value objects, and aggregates.

The helpers consolidate common DDD checks so domain services and dispatcher
handlers can validate identity and immutability without duplicating boilerplate
logic.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence

from flext_core import c, m, p, t
from flext_core._utilities import FlextUtilitiesGuardsTypeCore


class FlextUtilitiesDomain:
    """Reusable DDD helpers for dispatcher-driven domain workflows."""

    @staticmethod
    def same_type(
        obj_a: t.JsonPayload | p.HasModelDump,
        obj_b: t.JsonPayload | p.HasModelDump,
    ) -> bool:
        """Exact-type identity comparison (no MRO traversal).

        Returns True only when both objects are the exact same concrete type.

        Returns:
            The resulting ``bool``.

        """
        return type(obj_a) is type(obj_b)

    @staticmethod
    def _iterable_members_equal(
        obj_a: t.JsonPayload | p.HasModelDump,
        obj_b: t.JsonPayload | p.HasModelDump,
    ) -> bool | None:
        """Compare two iterable payloads member-wise.

        Returns:
            The comparison outcome, or ``None`` when neither payload is a
            plain iterable.

        """
        obj_a_iterable = hasattr(obj_a, "__iter__") and not hasattr(
            obj_a,
            "model_dump",
        )
        obj_b_iterable = hasattr(obj_b, "__iter__") and not hasattr(
            obj_b,
            "model_dump",
        )
        if not (obj_a_iterable or obj_b_iterable):
            return None
        if isinstance(obj_a, Mapping) and isinstance(obj_b, Mapping):
            return dict(obj_a.items()) == dict(obj_b.items())
        if isinstance(obj_a, Sequence) and isinstance(obj_b, Sequence):
            return tuple(obj_a) == tuple(obj_b)
        return False

    @staticmethod
    def _plain_objects_equal(
        obj_a: t.JsonPayload | p.HasModelDump,
        obj_b: t.JsonPayload | p.HasModelDump,
    ) -> bool:
        """Compare plain objects by ``__dict__``, falling back to ``repr``.

        Returns:
            The resulting ``bool``.

        """
        try:
            dict_a = vars(obj_a)
        except c.EXC_ATTR_TYPE:
            dict_a = None
        try:
            dict_b = vars(obj_b)
        except c.EXC_ATTR_TYPE:
            dict_b = None
        return (
            dict_a == dict_b
            if dict_a is not None and dict_b is not None
            else repr(obj_a) == repr(obj_b)
        )

    @staticmethod
    def compare_value_objects_by_value(
        obj_a: t.JsonPayload | p.HasModelDump,
        obj_b: t.JsonPayload | p.HasModelDump,
    ) -> bool:
        """Compare two value objects by all attributes (value, not identity).

        Returns True if same type and all attributes equal.

        Returns:
            The resulting ``bool``.

        """
        result: bool
        if isinstance(obj_a, c.SCALAR_TYPES):
            result = obj_a == obj_b if isinstance(obj_b, c.SCALAR_TYPES) else False
        else:
            result = False
            if not FlextUtilitiesGuardsTypeCore.scalar(obj_b):
                iterable_result = FlextUtilitiesDomain._iterable_members_equal(
                    obj_a,
                    obj_b,
                )
                if iterable_result is not None:
                    result = iterable_result
                elif FlextUtilitiesDomain.same_type(obj_b, obj_a):
                    if isinstance(obj_a, m.EnforcedModel) and isinstance(
                        obj_b,
                        m.EnforcedModel,
                    ):
                        result = obj_a.model_dump() == obj_b.model_dump()
                    else:
                        result = FlextUtilitiesDomain._plain_objects_equal(
                            obj_a,
                            obj_b,
                        )
        return result

    @staticmethod
    def hash_value_object_by_value(
        obj: t.JsonPayload | p.HasModelDump,
    ) -> int:
        """Hash value object by all attributes. Falls back to repr hash.

        Returns:
            The resulting ``int``.

        """
        if FlextUtilitiesGuardsTypeCore.scalar(obj):
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
        items: Sequence[tuple[str, t.JsonValue]] = [
            (
                k,
                v
                if isinstance(v, (str, int, float, bool, type(None)))
                else FlextUtilitiesGuardsTypeCore.type_name(v),
            )
            for k, v in sorted(obj_dict.items())
        ]
        return hash(tuple(items))

    @staticmethod
    def add_domain_event(
        entity: p.HasDomainEvents,
        event_type: str,
        data: m.ConfigMap | Mapping[str, t.JsonPayload | None] | None = None,
        aggregate_id: str | None = None,
    ) -> m.DomainEvent:
        """Create a domain event and append it to the entity's event buffer.

        Pass ``aggregate_id`` explicitly when the entity's stable identity
        differs from ``unique_id`` (e.g. a surrogate ``id`` field). Pydantic's
        ``BeforeValidator`` on ``DomainEvent.data`` handles all normalization.

        Returns:
            The resulting ``mde.DomainEvent``.

        """
        if data is None:
            normalized_data = m.ConfigMap(root={})
        elif isinstance(data, m.ConfigMap):
            normalized_data = data
        else:
            normalized_data = m.ConfigMap.model_validate(data)
        entry = m.DomainEvent(
            event_type=event_type,
            aggregate_id=aggregate_id if aggregate_id is not None else entity.unique_id,
            data=normalized_data,
        )
        entity.domain_events.append(entry)
        return entry


__all__: MutableSequence[str] = ["FlextUtilitiesDomain"]
