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
from flext_core._models.base import FlextModelsBase
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._typings.pydantic import FlextTypesPydantic


class FlextModelsService:
    """Domain service pattern container class.

    This class acts as a namespace container for domain service patterns.
    All nested classes are accessed via FlextModels.Service.* in the main models.py.
    """

    class ServiceRuntime(FlextModelsBase.ArbitraryTypesModel):
        """Shared runtime state for services and infrastructure collaborators.

        Every collaborator is a validated port: construction rejects a value that
        does not satisfy its protocol, so a runtime can never carry a fake
        settings, context, container or dispatcher.
        """

        settings: FlextTypesPydantic.Port[p.Settings] = FlextModelsPydantic.Field(
            exclude=True,
            description="Service configuration settings for runtime behavior.",
        )
        context: FlextTypesPydantic.Port[p.Context] = FlextModelsPydantic.Field(
            exclude=True,
            description="Execution context carrying correlation and tracing metadata.",
        )
        container: FlextTypesPydantic.Port[p.Container] = FlextModelsPydantic.Field(
            exclude=True,
            description="Dependency injection container scoped to this runtime.",
        )
        dispatcher: FlextTypesPydantic.Port[p.Dispatcher] = FlextModelsPydantic.Field(
            exclude=True,
            description="Dispatcher resolved for CQRS routing in this runtime.",
        )

    class RuntimeBootstrapOptions(FlextModelsBase.ArbitraryTypesModel):
        """Options a service base declares to build its runtime.

        Every field is optional: an absent value means the runtime derives it
        from its canonical owner (the settings class, a fresh context, the
        container's command bus).
        """

        settings: FlextTypesPydantic.Port[p.Settings | None] = (
            FlextModelsPydantic.Field(
                None,
                exclude=True,
                description=(
                    "Pre-built settings instance used directly for the runtime."
                ),
            )
        )
        settings_type: Annotated[
            t.SettingsClass | None,
            FlextTypesPydantic.SkipJsonSchema(),
        ] = FlextModelsPydantic.Field(
            None,
            exclude=True,
            description="FlextSettings class used to load runtime settings.",
        )
        settings_overrides: t.ScalarMapping | None = FlextModelsPydantic.Field(
            None,
            description=(
                "Key-value overrides applied on top of the loaded configuration."
            ),
        )
        context: FlextTypesPydantic.Port[p.Context | None] = FlextModelsPydantic.Field(
            None,
            exclude=True,
            description="Pre-built execution context to inject into the runtime.",
        )
        dispatcher: FlextTypesPydantic.Port[p.Dispatcher | None] = (
            FlextModelsPydantic.Field(
                None,
                exclude=True,
                description="Pre-built dispatcher injected into the runtime.",
            )
        )

    class ServiceOperation(FlextModelsBase.FrozenModel):
        """One typed operation of a service, as ``u.service_operations`` reports it.

        An operation is a public instance method declared below ``FlextService``
        that takes nothing or one Pydantic request model and returns
        ``p.Result``; its one-line docstring is the summary.
        """

        name: str = FlextModelsPydantic.Field(
            description="Method name that implements the operation.",
        )
        summary: str = FlextModelsPydantic.Field(
            description="First docstring line of the operation.",
        )
        request: t.ModelClass[t.BaseModelType] | None = FlextModelsPydantic.Field(
            description="Pydantic request model, or None for an input-less operation.",
        )


__all__: t.MutableSequenceOf[str] = ["FlextModelsService"]
