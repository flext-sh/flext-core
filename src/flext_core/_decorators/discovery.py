"""Factory discovery implementation for auto-registration.

This module provides factory discovery functionality that can be used by
container and decorators without creating circular dependencies.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator

from flext_core.constants import c
from flext_core.models import m


class FactoryDecoratorsDiscovery:
    """Auto-discovery mechanism for factory decorators.

    Scans modules and classes for functions decorated with @factory() and provides
    utilities for finding and analyzing factory configurations.

    This class enables zero-config factory registration in FlextContainer
    by automatically discovering decorated functions at initialization time.
    """

    @staticmethod
    def scan_module(
        module: object,
    ) -> list[tuple[str, m.ContainerFactoryDecoratorConfig]]:
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
        factories: list[tuple[str, m.ContainerFactoryDecoratorConfig]] = []
        for name in dir(module):
            if name.startswith("_"):
                continue
            func = getattr(module, name, None)
            if callable(func) and hasattr(func, c.Discovery.FACTORY_ATTR):
                config: m.ContainerFactoryDecoratorConfig = getattr(
                    func,
                    c.Discovery.FACTORY_ATTR,
                )
                factories.append((name, config))

        # Sort by name for consistent ordering
        return sorted(factories, key=operator.itemgetter(0))

    @staticmethod
    def has_factories(module: object) -> bool:
        """Check if module has any factory-decorated functions.

        Efficiently checks if a module contains any functions marked with
        the @factory() decorator without scanning all items.

        Args:
            module: Module object to check for factories

        Returns:
            True if module has at least one factory, False otherwise

        Example:
            >>> from flext_core._decorators import FactoryDecoratorsDiscovery
            >>> if FactoryDecoratorsDiscovery.has_factories(my_module):
            ...     # Auto-register factories in container
            ...     container.auto_register_factories(my_module)

        """
        return any(
            hasattr(getattr(module, name, None), c.Discovery.FACTORY_ATTR)
            for name in dir(module)
            if not name.startswith("_") and callable(getattr(module, name, None))
        )
