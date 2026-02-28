"""Factory discovery implementation for auto-registration.

This module provides factory discovery functionality that can be used by
container and decorators without creating circular dependencies.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from types import ModuleType

from flext_core import c
from flext_core._models.container import FlextModelsContainer


class FactoryDecoratorsDiscovery:
    """Auto-discovery mechanism for factory decorators.

    Scans modules and classes for functions decorated with @factory() and provides
    utilities for finding and analyzing factory configurations.

    This class enables zero-config factory registration in FlextContainer
    by automatically discovering decorated functions at initialization time.
    """

    @staticmethod
    def scan_module(
        module: ModuleType,
    ) -> list[tuple[str, FlextModelsContainer.FactoryDecoratorConfig]]:
        """Scan module for functions decorated with @factory().

        Introspects the module to find all functions with factory configuration
        metadata, returning them sorted by name for consistent ordering.

        Args:
            module: Module object to scan for factory decorators

        Returns:
            List of tuples (function_name, FactoryDecoratorConfig) sorted by name

        Example:
            >>> from flext_core._decorators import FactoryDecoratorsDiscovery
            >>> factories = FactoryDecoratorsDiscovery.scan_module(my_module)
            >>> for func_name, config in factories:
            ...     print(f"{func_name}: singleton={config.singleton}")

        """
        factories: list[tuple[str, FlextModelsContainer.FactoryDecoratorConfig]] = []
        for name in dir(module):
            if name.startswith("_"):
                continue
            func = getattr(module, name, None)
            if callable(func) and hasattr(func, c.Discovery.FACTORY_ATTR):
                config: FlextModelsContainer.FactoryDecoratorConfig = getattr(
                    func,
                    c.Discovery.FACTORY_ATTR,
                )
                factories.append((name, config))

        # Sort by name for consistent ordering
        return sorted(factories, key=operator.itemgetter(0))

    @staticmethod
    def has_factories(module: ModuleType) -> bool:
        """Check if module has any factory-decorated functions."""
        return any(
            hasattr(getattr(module, name, None), c.Discovery.FACTORY_ATTR)
            for name in dir(module)
            if not name.startswith("_") and callable(getattr(module, name, None))
        )
