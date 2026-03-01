"""Domain event model for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, runtime only.
Defines the DomainEvent model in its own module to avoid forward reference
issues when referenced by Entity (in entity.py).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Self, override

from pydantic import BeforeValidator, Field, model_validator

from flext_core import FlextRuntime, c, t
from flext_core._models.base import FlextModelFoundation


class _ComparableConfigMap(t.ConfigMap):
    """ConfigMap with equality support for domain event data."""

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return self.root == other
        if isinstance(other, Mapping):
            return self.root == dict(other.items())
        return super().__eq__(other)

    __hash__ = t.ConfigMap.__hash__


def _normalize_event_data(
    value: t.GuardInputValue,
) -> _ComparableConfigMap:
    """BeforeValidator: normalize event data to _ComparableConfigMap."""
    if isinstance(value, _ComparableConfigMap):
        return value
    if isinstance(value, t.ConfigMap):
        return _ComparableConfigMap(root=dict(value.items()))
    if isinstance(value, dict):
        normalized: t.ConfigMap = t.ConfigMap(
            root={
                str(k): FlextRuntime.normalize_to_metadata_value(v)
                for k, v in value.items()
            },
        )
        return _ComparableConfigMap(root=normalized.root)
    if isinstance(value, Mapping):
        normalized = t.ConfigMap(
            root={
                str(k): FlextRuntime.normalize_to_metadata_value(v)
                for k, v in value.items()
            },
        )
        return _ComparableConfigMap(root=normalized.root)
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
    def to_config_map(data: t.ConfigMap | None) -> _ComparableConfigMap:
        """Convert optional ConfigMap to a comparable variant."""
        if not data:
            return _ComparableConfigMap(root={})
        return _ComparableConfigMap(
            root={
                str(key): FlextRuntime.normalize_to_metadata_value(value)
                for key, value in data.items()
            },
        )

    @staticmethod
    def _normalize_event_data(
        value: t.GuardInputValue,
    ) -> _ComparableConfigMap:
        """BeforeValidator: normalize event data to _ComparableConfigMap."""
        return _normalize_event_data(value)

    class Entry(
        FlextModelFoundation.ArbitraryTypesModel,
        FlextModelFoundation.IdentifiableMixin,
        FlextModelFoundation.TimestampableMixin,
    ):
        """Base class for domain events."""

        message_type: c.Cqrs.EventMessageTypeLiteral = Field(
            default="event",
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
        )

        event_type: str
        aggregate_id: str
        data: Annotated[
            _ComparableConfigMap,
            BeforeValidator(_normalize_event_data),
        ] = Field(
            default_factory=_ComparableConfigMap,
            description="Event data container",
        )

        @model_validator(mode="after")
        def validate_domain_event(self) -> Self:
            if not self.event_type:
                msg = "Domain event event_type must be a non-empty string"
                raise ValueError(msg)
            if not self.aggregate_id:
                msg = "Domain event aggregate_id must be a non-empty string"
                raise ValueError(msg)
            return self


__all__ = ["FlextModelsDomainEvent"]
