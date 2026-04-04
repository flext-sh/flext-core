"""Domain event model for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, runtime only.
Defines the DomainEvent model in its own module to avoid forward reference
issues when referenced by Entity (in entity.py).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from typing import Annotated, override

from pydantic import BeforeValidator, Field

from flext_core import (
    FlextModelsBase,
    FlextUtilitiesDomain,
    t,
)


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
                other_mapping = t.ConfigMap(
                    root=dict(
                        FlextUtilitiesDomain.normalize_domain_event_data(
                            typing.cast("Mapping[str, t.ValueOrModel]", other)
                        )
                    ),
                ).root
                return self.root == other_mapping
            return super().__eq__(other)

        __hash__ = t.ConfigMap.__hash__

    class Entry(
        FlextModelsBase.IdentifiableMixin,
        FlextModelsBase.TimestampedModel,
    ):
        """Base class for domain events."""

        message_type: str = Field(
            default="event",
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
        )
        event_type: Annotated[
            t.NonEmptyStr,
            Field(description="Domain event type identifier for subscriber routing."),
        ]
        aggregate_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Identifier of the aggregate root that produced this event."
            ),
        ]
        data: Annotated[
            FlextModelsDomainEvent.ComparableConfigMap,
            BeforeValidator(FlextUtilitiesDomain.normalize_domain_event_data),
        ] = Field(
            validate_default=True,
            description="Event data container",
            default_factory=lambda: FlextModelsDomainEvent.ComparableConfigMap(
                root={},
            ),
        )

    # Canonical alias: tests use m.DomainEvent, which resolves to Entry
    DomainEvent = Entry


__all__ = ["FlextModelsDomainEvent"]
