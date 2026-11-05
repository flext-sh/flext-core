"""Domain event model for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, runtime only.
Defines the DomainEvent model in its own module to avoid forward reference
issues when referenced by Entity (in entity.py).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING, Annotated, Literal

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._models import (
    FlextModelsBase,
    FlextModelsContainers,
    FlextModelsPydantic,
)


class FlextModelsDomainEvent:
    """Namespace for domain event models.

    Contains the typed DomainEvent contract.
    Split into its own module so Entity can import without forward references.
    """

    class DomainEvent(
        FlextModelsBase.IdentifiableMixin,
        FlextModelsBase.TimestampedModel,
    ):
        """Base class for domain events."""

        message_type: Literal[flext_core.c.HandlerType.EVENT] = (
            FlextModelsPydantic.Field(
                default=flext_core.c.HandlerType.EVENT,
                frozen=True,
                description="Immutable event discriminator for union routing.",
                validate_default=True,
            )
        )
        event_type: Annotated[
            flext_core.t.NonEmptyStr,
            FlextModelsPydantic.Field(
                description="Domain event type identifier for subscriber routing.",
            ),
        ]
        aggregate_id: Annotated[
            flext_core.t.NonEmptyStr,
            FlextModelsPydantic.Field(
                description=(
                    "Identifier of the aggregate root that produced this event."
                ),
            ),
        ]
        data: FlextModelsContainers.ConfigMap = FlextModelsPydantic.Field(
            validate_default=True,
            description="Event data container",
            default_factory=lambda: FlextModelsContainers.ConfigMap(root={}),
        )


__all__: MutableSequence[str] = ["FlextModelsDomainEvent"]
