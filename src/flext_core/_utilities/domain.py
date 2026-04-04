"""Domain helper utilities for entities, value objects, and aggregates.

The helpers consolidate common DDD checks so domain services and dispatcher
handlers can validate identity and immutability without duplicating boilerplate
logic.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from flext_core import FlextUtilitiesGuardsTypeCore, c, t


class FlextUtilitiesDomain:
    """Reusable DDD helpers for dispatcher-driven domain workflows."""

    @staticmethod
    def same_type(obj_a: object, obj_b: object) -> bool:
        """Exact-type identity comparison (no MRO traversal).

        Returns True only when both objects are the exact same concrete type.
        """
        return type(obj_a) is type(obj_b)

    @staticmethod
    def _get_obj_dict(obj: t.RuntimeData) -> Mapping[str, t.RecursiveContainer] | None:
        """Extract __dict__ safely, returning None on failure."""
        try:
            return obj.__dict__
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _to_hashable(value: t.RecursiveContainer) -> t.RecursiveContainer:
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
        if FlextUtilitiesGuardsTypeCore.is_scalar(entity_a):
            return False
        if isinstance(entity_a, (Sequence, Mapping)):
            return False
        if FlextUtilitiesGuardsTypeCore.is_scalar(entity_b):
            return False
        if isinstance(entity_b, (Sequence, Mapping)):
            return False
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
        if FlextUtilitiesGuardsTypeCore.is_scalar(obj_a):
            return obj_a == obj_b
        if FlextUtilitiesGuardsTypeCore.is_scalar(obj_b):
            return False
        if hasattr(obj_a, "__iter__") and not hasattr(obj_a, "model_dump"):
            return obj_a == obj_b
        if hasattr(obj_b, "__iter__") and not hasattr(obj_b, "model_dump"):
            return obj_a == obj_b
        if not FlextUtilitiesDomain.same_type(obj_b, obj_a):
            return False
        if isinstance(obj_a, BaseModel) and isinstance(obj_b, BaseModel):
            return obj_a.model_dump() == obj_b.model_dump()
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
        if FlextUtilitiesGuardsTypeCore.is_scalar(entity):
            return hash(entity)
        entity_id = getattr(entity, id_attr, None)
        if entity_id is None:
            return hash(id(entity))
        return hash((entity.__class__.__name__, entity_id))

    @staticmethod
    def hash_value_object_by_value(obj: t.RuntimeData) -> int:
        """Hash value object by all attributes. Falls back to repr hash."""
        if FlextUtilitiesGuardsTypeCore.is_scalar(obj):
            return hash(obj)
        if isinstance(obj, BaseModel):
            data = obj.model_dump()
            return hash(tuple(sorted((str(k), str(v)) for k, v in data.items())))
        if hasattr(obj, "__iter__"):
            return hash(repr(obj))
        obj_dict = FlextUtilitiesDomain._get_obj_dict(obj)
        if obj_dict is None:
            return hash(repr(obj))
        items: Sequence[tuple[str, t.RecursiveContainer]] = [
            (str(k), FlextUtilitiesDomain._to_hashable(v))
            for k, v in sorted(obj_dict.items())
        ]
        return hash(tuple(items))

    @staticmethod
    def normalize_recursive_metadata_value(
        item: t.MetadataOrValue | None,
    ) -> t.RecursiveContainer:
        """Normalize metadata-like values while preserving recursive shape."""
        if item is None:
            return None
        if isinstance(item, (bool, str, int, float, datetime)):
            return item
        if isinstance(item, Mapping):
            normalized_map: t.MutableContainerMapping = {}
            for key, value in item.items():
                normalized_map[str(key)] = (
                    FlextUtilitiesDomain.normalize_recursive_metadata_value(
                        value
                        if FlextUtilitiesGuardsTypeCore.is_scalar(value)
                        else str(value)
                    )
                )
            return normalized_map
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [
                FlextUtilitiesDomain.normalize_recursive_metadata_value(
                    value
                    if FlextUtilitiesGuardsTypeCore.is_scalar(value)
                    else str(value)
                )
                for value in item
            ]
        return str(item)

    @staticmethod
    def normalize_domain_event_data(
        value: t.ValueOrModel | Mapping[str, t.ValueOrModel] | None,
    ) -> Mapping[str, t.RecursiveContainer]:
        """Normalize domain event payloads into comparable plain mappings."""
        if value is None:
            return {}
        if not isinstance(value, (t.ConfigMap, Mapping)):
            msg = "Domain event data must be a dictionary or None"
            raise TypeError(msg)
        raw_source = value.root if isinstance(value, t.ConfigMap) else value
        typed_source = t.dict_str_metadata_adapter().validate_python(raw_source)
        normalized: MutableMapping[str, t.RecursiveContainer] = {}
        for key, item in typed_source.items():
            normalized[str(key)] = (
                FlextUtilitiesDomain.normalize_recursive_metadata_value(item)
            )
        return normalized

    @staticmethod
    def aggregate_metadata_values(
        existing: t.MetadataValue | None,
        value: t.MetadataValue | None,
    ) -> t.MetadataValue | None:
        """Resolve metadata conflicts using stable aggregation rules."""
        non_none: Sequence[t.MetadataValue] = [
            item for item in (existing, value) if item is not None
        ]
        if not non_none:
            return None
        first_val = non_none[0]
        if isinstance(first_val, bool):
            return non_none[-1]
        if isinstance(first_val, (int, float)) and not isinstance(first_val, bool):
            numeric_values: Sequence[t.Numeric] = [
                item
                for item in non_none
                if isinstance(item, (int, float)) and not isinstance(item, bool)
            ]
            return sum(numeric_values) if numeric_values else non_none[-1]
        if isinstance(first_val, list):
            combined: MutableSequence[t.Scalar] = []
            for item in non_none:
                if isinstance(item, list):
                    combined.extend(item)
            return combined
        if isinstance(first_val, Mapping):
            merged: MutableMapping[str, t.Scalar | t.ScalarList] = {}
            for item in non_none:
                if isinstance(item, Mapping):
                    for key, nested_value in item.items():
                        if isinstance(
                            nested_value,
                            (str, int, float, bool, datetime, list),
                        ):
                            merged[str(key)] = nested_value
            return merged
        return non_none[-1]

    @staticmethod
    def aggregate_dumped_models[
        TModel: BaseModel,
    ](
        items: Sequence[TModel],
    ) -> Mapping[str, t.MetadataValue]:
        """Aggregate dumped model fields into one comparable metadata mapping."""
        if not items:
            return {}
        aggregated: MutableMapping[str, t.MetadataValue | None] = {}
        for item in items:
            for key, value in item.model_dump().items():
                aggregated[key] = FlextUtilitiesDomain.aggregate_metadata_values(
                    aggregated.get(key),
                    value,
                )
        return {key: value for key, value in aggregated.items() if value is not None}

    @staticmethod
    def append_metadata_sequence_item(
        metadata: t.Dict,
        key: Literal["failed_items", "warning_items"],
        item: t.ValueOrModel,
    ) -> None:
        """Append one normalized item to a metadata sequence bucket."""
        raw_items = metadata.root.get(key)
        result_list: t.MutableContainerList = []
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if isinstance(
                    raw_item,
                    (str, int, float, bool, datetime, Path, list, dict, tuple),
                ):
                    result_list.append(raw_item)
                elif raw_item is not None:
                    result_list.append(str(raw_item))
        if isinstance(item, (str, int, float, bool, datetime, Path, list, dict, tuple)):
            result_list.append(item)
        elif item is not None:
            result_list.append(str(item))
        metadata.root[key] = result_list

    @staticmethod
    def upsert_skip_reason(
        metadata: t.Dict,
        item: t.ValueOrModel,
        reason: str,
    ) -> None:
        """Store one skip reason keyed by the stringified item representation."""
        raw_reasons = metadata.root.get("skip_reasons", {})
        reasons: t.MutableStrMapping = {}
        if isinstance(raw_reasons, Mapping):
            reasons = {str(key): str(value) for key, value in raw_reasons.items()}
        reasons[str(item)] = reason
        metadata.root["skip_reasons"] = reasons


__all__ = ["FlextUtilitiesDomain"]
