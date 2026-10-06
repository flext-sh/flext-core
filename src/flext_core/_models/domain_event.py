"""Domain event model for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, runtime only.
Defines the DomainEvent model in its own module to avoid forward reference
issues when referenced by Entity (in entity.py).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import t
from flext_core._models.base import FlextModelsBase
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities.collection import FlextUtilitiesCollection


class FlextModelsDomainEvent:
    """Namespace for domain event models.

    Contains DomainEvent and helper utilities for event data normalization.
    Split into its own module so Entity can import without forward references.
    """

    class DomainEvent(
        FlextModelsBase.IdentifiableMixin,
        FlextModelsBase.TimestampedModel,
    ):
        """Base class for domain events."""

        message_type: str = FlextModelsPydantic.Field(
            "event",
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
            validate_default=True,
        )
        event_type: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(
                description="Domain event type identifier for subscriber routing.",
            ),
        ]
        aggregate_id: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(
                description="Identifier of the aggregate root that produced this event.",
            ),
        ]
        data: Annotated[
            FlextModelsContainers.ConfigMap,
            FlextModelsPydantic.BeforeValidator(
                FlextUtilitiesCollection.normalize_domain_event_data,
            ),
        ] = FlextModelsPydantic.Field(
            validate_default=True,
            description="Event data container",
            default_factory=lambda: FlextModelsContainers.ConfigMap(root={}),
        )


__all__: t.MutableSequenceOf[str] = ["FlextModelsDomainEvent"]
