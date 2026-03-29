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
from typing import override

from pydantic import Field, field_validator

from flext_core._models.base import FlextModelFoundation
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core.typings import t


class FlextModelsDomainEvent:
    """Namespace for domain event models.

    Contains DomainEvent and helper utilities for event data normalization.
    Split into its own module so Entity can import without forward references.
    """

    class ComparableConfigMap(t.ConfigMap):
        """ConfigMap with equality support for domain event data."""

        @override
        def __eq__(self, other: object) -> bool:
            if isinstance(other, dict):
                return self.root == other
            if isinstance(other, Mapping):
                typed_other = FlextModelFoundation.Validators.dict_str_metadata_adapter().validate_python(
                    other,
                )
                other_mapping = t.ConfigMap(
                    root={
                        key: FlextModelsDomainEvent.metadata_to_normalized(value)
                        for key, value in typed_other.items()
                    },
                ).root
                return self.root == other_mapping
            return super().__eq__(other)

        __hash__ = t.ConfigMap.__hash__

    @staticmethod
    def metadata_to_normalized(
        item: t.MetadataOrValue | None,
    ) -> t.NormalizedValue:
        if item is None:
            return None
        if isinstance(item, bool):
            return item
        if isinstance(item, str):
            return item
        if isinstance(item, int):
            return item
        if isinstance(item, float):
            return item
        if isinstance(item, datetime):
            return item
        if isinstance(item, Mapping):
            normalized_map: t.MutableContainerMapping = {}
            for key, value in item.items():
                normalized_map[str(key)] = (
                    FlextModelsDomainEvent.metadata_to_normalized(
                        value
                        if FlextUtilitiesGuardsTypeCore.is_scalar(value)
                        else str(value),
                    )
                )
            return normalized_map
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [
                FlextModelsDomainEvent.metadata_to_normalized(
                    value
                    if FlextUtilitiesGuardsTypeCore.is_scalar(value)
                    else str(value),
                )
                for value in item
            ]
        return str(item)

    @staticmethod
    def _normalize_event_data(
        value: t.ValueOrModel,
    ) -> FlextModelsDomainEvent.ComparableConfigMap:
        """BeforeValidator: normalize event data to FlextModelsDomainEvent.ComparableConfigMap."""
        if isinstance(value, FlextModelsDomainEvent.ComparableConfigMap):
            return value
        if isinstance(value, t.ConfigMap):
            return FlextModelsDomainEvent.ComparableConfigMap(root=dict(value.items()))
        if isinstance(value, dict):
            typed_value = FlextModelFoundation.Validators.dict_str_metadata_adapter().validate_python(
                value,
            )
            intermediate = t.ConfigMap(
                root={
                    key: FlextModelsDomainEvent.metadata_to_normalized(item)
                    for key, item in typed_value.items()
                },
            )
            return FlextModelsDomainEvent.ComparableConfigMap(root=intermediate.root)
        if isinstance(value, Mapping):
            typed_mapping = FlextModelFoundation.Validators.dict_str_metadata_adapter().validate_python(
                value,
            )
            intermediate = t.ConfigMap(
                root={
                    key: FlextModelsDomainEvent.metadata_to_normalized(item)
                    for key, item in typed_mapping.items()
                },
            )
            return FlextModelsDomainEvent.ComparableConfigMap(root=intermediate.root)
        if value is None:
            return FlextModelsDomainEvent.ComparableConfigMap(root={})
        msg = "Domain event data must be a dictionary or None"
        raise TypeError(msg)

    @staticmethod
    def to_config_map(
        data: t.ConfigMap | None,
    ) -> FlextModelsDomainEvent.ComparableConfigMap:
        """Convert optional ConfigMap to a comparable variant."""
        if not data:
            return FlextModelsDomainEvent.ComparableConfigMap(root={})
        return FlextModelsDomainEvent.ComparableConfigMap(
            root={
                str(key): (
                    value
                    if FlextUtilitiesGuardsTypeCore.is_primitive(value)
                    else str(value)
                )
                for key, value in data.items()
            },
        )

    class Entry(
        FlextModelFoundation.ArbitraryTypesModel,
        FlextModelFoundation.IdentifiableMixin,
        FlextModelFoundation.TimestampableMixin,
    ):
        """Base class for domain events."""

        message_type: str = Field(
            default="event",
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
        )
        event_type: t.NonEmptyStr
        aggregate_id: t.NonEmptyStr
        data: FlextModelsDomainEvent.ComparableConfigMap = Field(
            validate_default=True,
            description="Event data container",
            default_factory=lambda: FlextModelsDomainEvent.ComparableConfigMap(
                root={},
            ),
        )

        @field_validator("data", mode="before")
        @classmethod
        def _validate_data(
            cls,
            value: t.ValueOrModel,
        ) -> FlextModelsDomainEvent.ComparableConfigMap:
            return FlextModelsDomainEvent._normalize_event_data(value)

    # Canonical alias: tests use m.DomainEvent, which resolves to Entry
    DomainEvent = Entry


__all__ = ["FlextModelsDomainEvent"]
