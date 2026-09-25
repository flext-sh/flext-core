"""Factory discovery implementation for auto-registration.

This module provides factory discovery functionality that can be used by
container and decorators without creating circular dependencies.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from flext_core import c, t

from .._models.container import FlextModelsContainer

if TYPE_CHECKING:
    from types import ModuleType


class FlextUtilitiesDiscovery:
    """Auto-discovery for @factory() decorated functions in modules."""

    @staticmethod
    def _factory_config_for(
        module: ModuleType, name: str
    ) -> FlextModelsContainer.FactoryDecoratorConfig | None:
        func = vars(module).get(name)
        if func is None or not callable(func):
            return None
        config_raw = vars(func).get(c.FACTORY_ATTR)
        if not isinstance(config_raw, FlextModelsContainer.FactoryDecoratorConfig):
            return None
        return config_raw

    @staticmethod
    def scan_module(
        module: ModuleType,
    ) -> t.SequenceOf[tuple[str, FlextModelsContainer.FactoryDecoratorConfig]]:
        """Scan module for @factory()-decorated functions, sorted by name."""
        return sorted(
            [
                (name, config)
                for name in dir(module)
                if not name.startswith("_")
                and (
                    config := FlextUtilitiesDiscovery._factory_config_for(module, name)
                )
                is not None
            ],
            key=operator.itemgetter(0),
        )


__all__: list[str] = ["FlextUtilitiesDiscovery"]
