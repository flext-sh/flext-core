"""Dependency-injector runtime bridge types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import ModuleType
from typing import Annotated, ClassVar

from dependency_injector import containers, providers
from pydantic import BaseModel, ConfigDict

from flext_core._models.containers import FlextModelsContainers as mc
from flext_core._typings.base import FlextTypingBase as tb
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._typings.services import FlextTypesServices as ts


class FlextRuntimeDependencyTypes:
    """Type owners for dependency-injector runtime bridge."""

    class DynamicContainerWithConfig(containers.DynamicContainer):
        """Dynamic container with declared configuration provider."""

        settings: providers.Configuration = providers.Configuration()

    class BridgeContainer(containers.DeclarativeContainer):
        """Declarative container grouping settings and resource modules."""

        settings = providers.Configuration()
        services = providers.Object(containers.DynamicContainer())
        resources = providers.Object(containers.DynamicContainer())

    class ContainerCreationOptions(BaseModel):
        """Validated options for dependency container creation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

        settings: mc.ConfigMap | None = None
        services: (
            tb.MappingKV[str, Annotated[ts.RegisterableService, tp.SkipValidation]]
            | None
        ) = None
        factories: (
            tb.MappingKV[str, Annotated[ts.FactoryCallable, tp.SkipValidation]] | None
        ) = None
        resources: (
            tb.MappingKV[str, Annotated[ts.ResourceCallable, tp.SkipValidation]] | None
        ) = None
        wire_modules: tb.SequenceOf[ModuleType] | None = None
        wire_packages: tb.StrSequence | None = None
        wire_classes: tb.SequenceOf[type] | None = None
        factory_cache: bool = True

    _OPTION_FIELDS: ClassVar[tb.StrSequence] = (
        "settings",
        "services",
        "factories",
        "resources",
        "wire_modules",
        "wire_packages",
        "wire_classes",
    )


__all__: list[str] = ["FlextRuntimeDependencyTypes"]
