"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Annotated

from typing_extensions import TypeForm

from flext_core import c, p, t
from flext_core._models import (
    FlextModelsBase,
    FlextModelsContainers,
    FlextModelsPydantic,
)
from flext_core._utilities import FlextUtilitiesGenerators, FlextUtilitiesGuardsTypeCore


class FlextModelsContainer:
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
        ] = FlextModelsPydantic.Field(default_factory=FlextUtilitiesGenerators.now)
        metadata: Annotated[
            FlextModelsBase.Metadata | FlextModelsContainers.ConfigMap | None,
            FlextModelsPydantic.Field(
                None,
                description="Additional service metadata (JSON-serializable)",
            ),
        ] = None

        @FlextModelsPydantic.field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata_model_input(cls, value: p.MetadataInput) -> p.Metadata:
            """Keep existing metadata identity and validate raw registration attributes."""
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
        """Service registry entry whose payload follows the declared service contract."""

        name: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(..., description="Service identifier/name"),
        ]
        service: Annotated[
            p.RegisterableService,
            FlextModelsPydantic.Field(
                ...,
                description="Service instance (protocols, models, callables)",
            ),
        ]
        tags: Annotated[
            t.StrSequence,
            FlextModelsPydantic.Field(description="Service tags for categorization"),
        ] = FlextModelsPydantic.Field(
            default_factory=FlextModelsPydantic.empty(TypeForm(t.StrSequence)),
        )

        @FlextModelsPydantic.computed_field
        @property
        def service_type(self) -> str:
            """Type name of the registered service, derived from the service."""
            return FlextUtilitiesGuardsTypeCore.type_name(self.service)

        @FlextModelsPydantic.field_validator("service", mode="before")
        @classmethod
        def validate_service(
            cls,
            value: p.RegisterableService,
        ) -> (
            p.RegisterableService
            | FlextModelsContainers.ConfigMap
            | FlextModelsContainers.ObjectList
        ):
            if isinstance(value, Mapping):
                return FlextModelsContainers.ConfigMap.model_validate(value)
            if isinstance(value, Sequence) and not isinstance(
                value, c.STR_BINARY_TYPES
            ):
                return FlextModelsContainers.ObjectList.model_validate(value)
            return value

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


__all__: list[str] = ["FlextModelsContainer"]
