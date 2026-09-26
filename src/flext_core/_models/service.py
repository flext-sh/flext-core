"""Domain service runtime models extracted from FlextModels.

This module contains the FlextModelsService class with the service runtime and
bootstrap option models as nested classes. It should NOT be imported directly -
use FlextModels.Service instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import p, t

from .._typings.pydantic import FlextTypesPydantic as tp
from .base import FlextModelsBase as m
from .pydantic import FlextModelsPydantic as mp


class FlextModelsService:
    """Domain service pattern container class.

    This class acts as a namespace container for domain service patterns.
    All nested classes are accessed via FlextModels.Service.* in the main models.py.
    """

    class ServiceRuntime(m.ArbitraryTypesModel):
        """Shared runtime state for services and infrastructure collaborators.

        Every collaborator is a validated port: construction rejects a value that
        does not satisfy its protocol, so a runtime can never carry a fake
        settings, context, container or dispatcher.
        """

        settings: tp.Port[p.Settings] = mp.Field(
            exclude=True,
            description="Service configuration settings for runtime behavior.",
        )
        context: tp.Port[p.Context] = mp.Field(
            exclude=True,
            description="Execution context carrying correlation and tracing metadata.",
        )
        container: tp.Port[p.Container] = mp.Field(
            exclude=True,
            description="Dependency injection container scoped to this runtime.",
        )
        dispatcher: tp.Port[p.Dispatcher] = mp.Field(
            exclude=True,
            description="Dispatcher resolved for CQRS routing in this runtime.",
        )

    class RuntimeBootstrapOptions(m.ArbitraryTypesModel):
        """Options a service base declares to build its runtime.

        Every field is optional: an absent value means the runtime derives it
        from its canonical owner (the settings class, a fresh context, the
        container's command bus).
        """

        settings: tp.Port[p.Settings | None] = mp.Field(
            None,
            exclude=True,
            description="Pre-built settings instance used directly for the runtime.",
        )
        settings_type: Annotated[t.SettingsClass | None, tp.SkipJsonSchema()] = (
            mp.Field(
                None,
                exclude=True,
                description="FlextSettings class used to load runtime settings.",
            )
        )
        settings_overrides: t.ScalarMapping | None = mp.Field(
            None,
            description="Key-value overrides applied on top of the loaded configuration.",
        )
        context: tp.Port[p.Context | None] = mp.Field(
            None,
            exclude=True,
            description="Pre-built execution context to inject into the runtime.",
        )
        dispatcher: tp.Port[p.Dispatcher | None] = mp.Field(
            None,
            exclude=True,
            description="Pre-built dispatcher injected into the runtime.",
        )


__all__: t.MutableSequenceOf[str] = ["FlextModelsService"]
