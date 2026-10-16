"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import t
from flext_core._models import FlextModelsBase, FlextModelsPydantic
from flext_core._models._container_parts import flextmodelscontainer_part_03


class FlextModelsContainer(flextmodelscontainer_part_03.FlextModelsContainer):
    class FactoryDecoratorConfig(FlextModelsBase.ImmutableValueModel):
        """Configuration extracted from @d.factory() decorator.

        Used by factory discovery to auto-register factories with FlextContainer.
        Stores metadata about factory name, singleton behavior, and lazy loading.

        Attributes:
            name: The name to register this factory under in the container.
            singleton: Whether the factory creates singleton instances. Default: False.
            lazy: Whether to defer factory invocation until first use. Default: True.

        Examples:
            >>> settings = FlextModelsContainers.FactoryDecoratorConfig(
            ...     name="database_service", singleton=True, lazy=False
            ... )
            >>> settings.name
            'database_service'
            >>> settings.singleton
            True

        """

        name: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(
                ...,
                description="Name to register this factory under in the container",
            ),
        ]
        singleton: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=False,
                description="Whether factory creates singleton instances",
            ),
        ] = False
        lazy: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=True,
                description="Whether to defer factory invocation until first use",
            ),
        ] = True


__all__: list[str] = ["FlextModelsContainer"]
