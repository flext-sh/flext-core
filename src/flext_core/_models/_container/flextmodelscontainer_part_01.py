"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated

from flext_core import c, p, t
from flext_core._models import (
    FlextModelsBase,
    FlextModelsContainers,
    FlextModelsPydantic,
)


class FlextModelsContainerPart01:
    """Container models namespace for DI and service registry."""

    class RegistrationEnvelope(FlextModelsBase.ArbitraryTypesModel):
        """Shared registration timestamp and validated metadata boundary."""

        registration_time: Annotated[
            datetime,
            FlextModelsPydantic.Field(
                description=(
                    "Timestamp when service was registered (configured timezone)"
                ),
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=lambda: datetime.now(UTC),
        )
        metadata: Annotated[
            FlextModelsBase.Metadata | FlextModelsContainers.ConfigMap | None,
            FlextModelsPydantic.Field(
                None,
                description="Additional service metadata (JSON-serializable)",
            ),
        ] = None

        @FlextModelsPydantic.field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata_model_input(
            cls, value: p.MetadataInput
        ) -> p.Metadata | None:
            """Keep metadata identity and validate raw registration attributes.

            Returns:
                The validated metadata or None.

            """
            if value is None:
                return None
            if isinstance(value, FlextModelsBase.Metadata):
                return value
            return FlextModelsBase.Metadata.model_validate({
                c.FIELD_ATTRIBUTES: (
                    value.model_dump(mode="json")
                    if isinstance(value, p.HasModelDump)
                    else value
                )
            })

    class ServiceRegistration(RegistrationEnvelope):
        """Service registry entry following the declared service contract."""

        name: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(..., description="Service identifier/name"),
        ]
        service: Annotated[
            t.SkipJsonSchema[p.RegisterableService],
            FlextModelsPydantic.Field(
                ...,
                description="Service instance (protocols, models, callables)",
                exclude=True,
            ),
        ]
        tags: Annotated[
            Sequence[str],
            FlextModelsPydantic.Field(
                description="Service tags for categorization", validate_default=True
            ),
        ] = ()

        @FlextModelsPydantic.computed_field
        @property
        def service_type(self) -> str:
            """Type name of the registered service, derived from the service."""
            return type(self.service).__name__

    class FactoryRegistration(RegistrationEnvelope):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: m.ConfigMap for factory tracking.
        """

        name: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(..., description="Factory identifier/name"),
        ]
        factory: Annotated[
            p.FactoryCallable,
            FlextModelsPydantic.Field(
                ...,
                description="Factory function that creates service instances",
            ),
        ]


__all__: list[str] = ["FlextModelsContainerPart01"]
