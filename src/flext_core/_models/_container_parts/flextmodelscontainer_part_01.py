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

from flext_core import FlextConstants as c, FlextTypes as t
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._runtime._container import FlextRuntimeContainer as FlextRuntime
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._utilities.generators import FlextUtilitiesGenerators as ug
from flext_core._utilities.pydantic import FlextUtilitiesPydantic as up


class FlextModelsContainer:
    """Container models namespace for DI and service registry."""

    class ServiceRegistration(
        m.ArbitraryTypesModel,
    ):
        """Model for service registry entries.

        Implements metadata for registered service instances in the DI container.
        Replaces: m.ConfigMap for service tracking.
        """

        name: Annotated[
            t.NonEmptyStr,
            mp.Field(
                ...,
                description="Service identifier/name",
            ),
        ]
        service: Annotated[
            t.RegisterableService,
            tp.SkipValidation,
            mp.Field(
                ..., description="Service instance (protocols, models, callables)"
            ),
        ]
        registration_time: Annotated[
            datetime,
            mp.Field(
                default_factory=lambda: ug.now(),
                description="Timestamp when service was registered (configured timezone)",
            ),
        ]
        metadata: Annotated[
            m.Metadata | FlextModelsContainers.ConfigMap | None,
            mp.BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    m.Metadata,
                ),
            ),
            mp.Field(
                None,
                description="Additional service metadata (JSON-serializable)",
            ),
        ] = None
        service_type: Annotated[
            str | None,
            mp.Field(
                None,
                description="Service type name (e.g., 'DatabaseService')",
            ),
        ] = None
        tags: Annotated[
            t.StrSequence,
            mp.Field(
                default_factory=tuple, description="Service tags for categorization"
            ),
        ]

        @up.field_validator("service", mode="before")
        @classmethod
        def validate_service(
            cls,
            value: t.RegisterableService,
        ) -> (
            t.RegisterableService
            | FlextModelsContainers.ConfigMap
            | FlextModelsContainers.ObjectList
        ):
            return FlextRuntime.normalize_registerable_service(value)

    class FactoryRegistration(
        m.ArbitraryTypesModel,
    ):
        """Model for factory registry entries.

        Implements metadata for registered factory functions in the DI container.
        Replaces: m.ConfigMap for factory tracking.
        """

        name: Annotated[
            t.NonEmptyStr,
            mp.Field(
                ...,
                description="Factory identifier/name",
            ),
        ]
        factory: Annotated[
            t.FactoryCallable,
            tp.SkipValidation,
            mp.Field(
                ..., description="Factory function that creates service instances"
            ),
        ]
        registration_time: Annotated[
            datetime,
            mp.Field(
                default_factory=lambda: ug.now(),
                description="Timestamp when factory was registered (configured timezone)",
            ),
        ]
        is_singleton: Annotated[
            bool,
            mp.Field(
                False,
                description="Whether factory creates singleton instances",
            ),
        ] = False
        cached_instance: Annotated[
            t.RegisterableService | None,
            tp.SkipValidation,
            mp.Field(
                None,
                description="Cached singleton instance (if is_singleton=True)",
            ),
        ] = None
        metadata: Annotated[
            m.Metadata | FlextModelsContainers.ConfigMap | None,
            mp.BeforeValidator(
                lambda value: FlextRuntime.validate_metadata_model_input(
                    value,
                    m.Metadata,
                ),
            ),
            mp.Field(
                None,
                description="Additional factory metadata (JSON-serializable)",
            ),
        ] = None
        invocation_count: Annotated[
            t.NonNegativeInt,
            mp.Field(
                c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of times factory has been invoked",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES


__all__: list[str] = ["FlextModelsContainer"]
