"""Entity and DDD patterns for FLEXT ecosystem.

TIER 1: Uses base.py (Tier 0) + constants, typings, protocols, runtime only.
Imports m and adds only DDD-specific data classes.

The DomainEvent is defined in domain_event.py and imported here to
avoid Pydantic forward-reference issues with nested class annotations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Hashable, MutableSequence
from typing import Annotated, override

from pydantic import Field

from flext_core._models import (
    FlextModelsBase,
    FlextModelsDomainEvent,
    FlextModelsPydantic,
)
from flext_core._typings import FlextTypesServices, FlextTypingBase
from flext_core._utilities import FlextUtilitiesDomain, FlextUtilitiesGenerators


class FlextModelsEntity:
    """Entity and DDD pattern container class.

    This class acts as a namespace container for Entity and related DDD patterns.
    Uses m for all base classes (Tier 0).

    DomainEvent is imported from FlextModelsDomainEvent to break
    the forward-reference cycle that Pydantic cannot resolve.
    """

    class Entity(
        FlextModelsBase.TimestampedModel,
        FlextModelsBase.IdentifiableMixin,
        FlextModelsBase.VersionableMixin,
        Hashable,
    ):
        """Entity implementation - base class for domain entities with identity.

        Combines TimestampedModel, IdentifiableMixin, and VersionableMixin to provide:
        - unique_id: Unique identifier (from IdentifiableMixin)
        - created_at/updated_at: Timestamps (from TimestampedModel)
        - version: Optimistic locking (from VersionableMixin)
        - domain_events: Event sourcing support

        ``domain_events`` is an intentionally mutable in-process buffer; the
        field contract is a ``MutableSequence`` because the event-sourcing API
        appends new entries during the entity lifecycle.
        """

        # Why assigned-value form, not ``Annotated`` metadata: pyright's
        # ``dataclass_transform`` synthesis recognizes ``default_factory``
        # default-ness only from the field specifier call assigned to the
        # class variable; a specifier inside ``Annotated`` metadata synthesizes
        # a REQUIRED ``__init__`` parameter (verified against plain pydantic).
        domain_events: MutableSequence[FlextModelsDomainEvent.DomainEvent] = Field(
            default_factory=list[FlextModelsDomainEvent.DomainEvent],
            validate_default=True,
            description="List of uncommitted domain events for event sourcing",
        )

        @override
        def __eq__(self, other: object) -> bool:
            """Identity-based equality for entities.

            Returns:
                The resulting ``bool``.

            """
            if not isinstance(other, FlextModelsBase.EnforcedModel):
                return NotImplemented
            return FlextUtilitiesDomain.compare_entities_by_id(self, other)

        def __hash__(self) -> int:
            """Identity-based hash for entities.

            Returns:
                The resulting ``int``.

            """
            return FlextUtilitiesDomain.hash_entity_by_id(self)

        @override
        def model_post_init(
            self,
            __context: FlextTypingBase.ScalarMapping | None,
            /,
        ) -> None:
            _ = __context
            """Post-initialization hook to set updated_at timestamp when absent."""
            if self.updated_at is None:
                self.updated_at = FlextUtilitiesGenerators.generate_datetime_utc()

    class Value(FlextModelsBase.FrozenValueModel):
        """Base class for value objects - immutable and compared by value."""

    class AggregateRoot(Entity):
        """Aggregate-root marker class (DDD consistency boundary)."""

    class TransformOptions(Value):
        """Validated option envelope for the ``transform`` value pipeline."""

        normalize: Annotated[
            bool,
            FlextModelsPydantic.Field(description="Normalize values to metadata form"),
        ] = False
        strip_none: Annotated[
            bool, FlextModelsPydantic.Field(description="Strip ``None`` values")
        ] = False
        strip_empty: Annotated[
            bool, FlextModelsPydantic.Field(description="Strip empty values")
        ] = False
        map_keys: Annotated[
            FlextTypingBase.StrMapping | None,
            FlextModelsPydantic.Field(
                description="Optional old-key to new-key mapping"
            ),
        ] = None

    class ExtractResolvePathPartContext(Value):
        """Validated context envelope for one path-part extraction step."""

        path_context: Annotated[
            str, FlextModelsPydantic.Field(description="Resolved parent path context")
        ] = ""
        default: Annotated[
            FlextTypesServices.JsonPayload | None,
            FlextModelsPydantic.Field(description="Default fallback payload"),
        ] = None
        required: Annotated[
            bool,
            FlextModelsPydantic.Field(description="Whether missing values are fatal"),
        ] = False


__all__: list[str] = ["FlextModelsEntity"]
