"""Domain event model for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, runtime only.
Defines the DomainEvent model in its own module to avoid forward reference
issues when referenced by Entity (in entity.py).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Annotated, override

from pydantic import BaseModel, BeforeValidator, Field

from flext_core import c, t
from flext_core._models.base import FlextModelFoundation
from flext_core._models.containers import FlextModelsContainers

_V = FlextModelFoundation.Validators


def _metadata_to_normalized(item: t.MetadataValue | None) -> t.NormalizedValue:
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, int):
        return item
    if isinstance(item, float):
        return item
    if isinstance(item, bool):
        return item
    if isinstance(item, datetime):
        return item
    if isinstance(item, Mapping):
        normalized_map: dict[str, t.NormalizedValue] = {}
        for key, value in item.items():
            normalized_map[str(key)] = _metadata_to_normalized(
                value
                if isinstance(value, (str, int, float, bool, datetime))
                else str(value)
            )
        return normalized_map
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
        return [
            _metadata_to_normalized(
                value
                if isinstance(value, (str, int, float, bool, datetime))
                else str(value)
            )
            for value in item
        ]
    return str(item)


class _ComparableConfigMap(FlextModelsContainers.ConfigMap):
    """ConfigMap with equality support for domain event data."""

    @override
    def __eq__(self, other) -> bool:
        if isinstance(other, dict):
            return self.root == other
        if isinstance(other, Mapping):
            typed_other = _V.dict_str_metadata_adapter().validate_python(other)
            other_mapping = FlextModelsContainers.ConfigMap(
                root={
                    key: _metadata_to_normalized(value)
                    for key, value in typed_other.items()
                }
            ).root
            return self.root == other_mapping
        return super().__eq__(other)

    __hash__ = FlextModelsContainers.ConfigMap.__hash__


def _normalize_event_data(value: t.NormalizedValue | BaseModel) -> _ComparableConfigMap:
    """BeforeValidator: normalize event data to _ComparableConfigMap."""
    if isinstance(value, _ComparableConfigMap):
        return value
    if isinstance(value, FlextModelsContainers.ConfigMap):
        return _ComparableConfigMap(root=dict(value.items()))
    if isinstance(value, dict):
        typed_value = _V.dict_str_metadata_adapter().validate_python(value)
        intermediate = FlextModelsContainers.ConfigMap(
            root={
                key: _metadata_to_normalized(item) for key, item in typed_value.items()
            }
        )
        return _ComparableConfigMap(root=intermediate.root)
    if isinstance(value, Mapping):
        typed_mapping = _V.dict_str_metadata_adapter().validate_python(value)
        intermediate = FlextModelsContainers.ConfigMap(
            root={
                key: _metadata_to_normalized(item)
                for key, item in typed_mapping.items()
            }
        )
        return _ComparableConfigMap(root=intermediate.root)
    if value is None:
        return _ComparableConfigMap(root={})
    msg = "Domain event data must be a dictionary or None"
    raise TypeError(msg)


class FlextModelsDomainEvent:
    """Namespace for domain event models.

    Contains DomainEvent and helper utilities for event data normalization.
    Split into its own module so Entity can import without forward references.
    """

    ComparableConfigMap = _ComparableConfigMap

    @staticmethod
    def _normalize_event_data(
        value: t.NormalizedValue | BaseModel,
    ) -> _ComparableConfigMap:
        """BeforeValidator: normalize event data to _ComparableConfigMap."""
        return _normalize_event_data(value)

    @staticmethod
    def to_config_map(
        data: FlextModelsContainers.ConfigMap | None,
    ) -> _ComparableConfigMap:
        """Convert optional ConfigMap to a comparable variant."""
        if not data:
            return _ComparableConfigMap(root={})
        return _ComparableConfigMap(
            root={
                str(key): (
                    value if isinstance(value, (str, int, float, bool)) else str(value)
                )
                for key, value in data.items()
            }
        )

    class Entry(
        FlextModelFoundation.ArbitraryTypesModel,
        FlextModelFoundation.IdentifiableMixin,
        FlextModelFoundation.TimestampableMixin,
    ):
        """Base class for domain events."""

        message_type: Annotated[
            c.Cqrs.EventMessageTypeLiteral,
            Field(
                default="event",
                frozen=True,
                description="Message type discriminator for union routing - always 'event'",
            ),
        ] = "event"
        event_type: Annotated[str, Field(min_length=c.Reliability.RETRY_COUNT_MIN)]
        aggregate_id: Annotated[str, Field(min_length=c.Reliability.RETRY_COUNT_MIN)]
        data: Annotated[
            _ComparableConfigMap, BeforeValidator(_normalize_event_data)
        ] = Field(
            default_factory=_ComparableConfigMap, description="Event data container"
        )


__all__ = ["FlextModelsDomainEvent"]
