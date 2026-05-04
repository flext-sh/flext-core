"""Domain event model for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, runtime only.
Defines the DomainEvent model in its own module to avoid forward reference
issues when referenced by Entity (in entity.py).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import (
    FlextModelsPydantic as mp,
    FlextTypes as t,
    FlextUtilitiesCollection as u,
)
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.containers import FlextModelsContainers as mc


class FlextModelsDomainEvent:
    """Namespace for domain event models.

    Contains DomainEvent and helper utilities for event data normalization.
    Split into its own module so Entity can import without forward references.
    """

    class Entry(
        m.IdentifiableMixin,
        m.TimestampedModel,
    ):
        """Base class for domain events."""

        message_type: str = mp.Field(
            "event",
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
            validate_default=True,
        )
        event_type: Annotated[
            t.NonEmptyStr,
            mp.Field(
                description="Domain event type identifier for subscriber routing."
            ),
        ]
        aggregate_id: Annotated[
            t.NonEmptyStr,
            mp.Field(
                description="Identifier of the aggregate root that produced this event.",
            ),
        ]
        data: Annotated[
            mc.ConfigMap,
            mp.BeforeValidator(u.normalize_domain_event_data),
        ] = mp.Field(
            validate_default=True,
            description="Event data container",
            default_factory=lambda: mc.ConfigMap(root={}),
        )

    DomainEvent = Entry


__all__: t.MutableSequenceOf[str] = ["FlextModelsDomainEvent"]
