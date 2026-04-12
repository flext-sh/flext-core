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

from flext_core import FlextModelsBase, FlextUtilitiesDomain, t
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities.pydantic import FlextUtilitiesPydantic


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
                other_mapping = type(self)(
                    root=dict(
                        FlextUtilitiesDomain.normalize_domain_event_data(
                            typing.cast("Mapping[str, t.ValueOrModel]", other),
                        ),
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

        message_type: str = FlextUtilitiesPydantic.Field(
            "event",
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
            validate_default=True,
        )
        event_type: Annotated[
            t.NonEmptyStr,
            FlextUtilitiesPydantic.Field(
                description="Domain event type identifier for subscriber routing."
            ),
        ]
        aggregate_id: Annotated[
            t.NonEmptyStr,
            FlextUtilitiesPydantic.Field(
                description="Identifier of the aggregate root that produced this event.",
            ),
        ]
        data: Annotated[
            t.ConfigMap,
            FlextModelsPydantic.BeforeValidator(
                FlextUtilitiesDomain.normalize_domain_event_data
            ),
        ] = FlextUtilitiesPydantic.Field(
            validate_default=True,
            description="Event data container",
            default_factory=lambda: FlextModelsDomainEvent.ComparableConfigMap(
                root={},
            ),
        )

    DomainEvent = Entry


__all__: list[str] = ["FlextModelsDomainEvent"]
