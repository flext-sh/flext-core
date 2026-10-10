"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from typing_extensions import TypeForm

from flext_core import t
from flext_core._models.base import FlextModelsBase
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._runtime._container import FlextRuntimeContainer
from flext_core._utilities import FlextUtilitiesGenerators, FlextUtilitiesGuardsTypeCore


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    class ServiceRegistration(FlextModelsBase.ArbitraryTypesModel):
        """Model for service registry entries.

        Implements metadata for registered service instances in the DI container.
        Replaces: m.ConfigMap for service tracking.
        """

        name: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(..., description="Service identifier/name"),
        ]
        service: Annotated[
            t.RegisterableService,
            FlextModelsPydantic.Field(
                ...,
                description="Service instance (protocols, models, callables)",
            ),
        ]
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
            FlextModelsPydantic.BeforeValidator(
                lambda value: FlextRuntimeContainer.validate_metadata_model_input(
                    value,
                    FlextModelsBase.Metadata,
                ),
            ),
            FlextModelsPydantic.Field(
                None,
                description="Additional service metadata (JSON-serializable)",
            ),
        ] = None
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
            value: t.RegisterableService,
        ) -> (
            t.RegisterableService
            | FlextModelsContainers.ConfigMap
            | FlextModelsContainers.ObjectList
        ):
            return FlextRuntimeContainer.normalize_registerable_service(value)

    class FactoryRegistration(FlextModelsBase.ArbitraryTypesModel):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: m.ConfigMap for factory tracking.
        """

        name: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(..., description="Factory identifier/name"),
        ]
        factory: Annotated[
            t.FactoryCallable,
            FlextModelsPydantic.Field(
                ...,
                description="Factory function that creates service instances",
            ),
        ]
        registration_time: Annotated[
            datetime,
            FlextModelsPydantic.Field(
                description=(
                    "Timestamp when factory was registered (configured timezone)"
                ),
            ),
        ] = FlextModelsPydantic.Field(default_factory=FlextUtilitiesGenerators.now)
        metadata: Annotated[
            FlextModelsBase.Metadata | FlextModelsContainers.ConfigMap | None,
            FlextModelsPydantic.BeforeValidator(
                lambda value: FlextRuntimeContainer.validate_metadata_model_input(
                    value,
                    FlextModelsBase.Metadata,
                ),
            ),
            FlextModelsPydantic.Field(
                None,
                description="Additional factory metadata (JSON-serializable)",
            ),
        ] = None


__all__: list[str] = ["FlextModelsContainer"]
