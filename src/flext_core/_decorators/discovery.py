"""Factory discovery implementation for auto-registration.

This module provides factory discovery functionality that can be used by
container and decorators without creating circular dependencies.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from types import ModuleType

from flext_core.constants import c
from flext_core.models import m


class FactoryDecoratorsDiscovery:
    """Auto-discovery mechanism for factory decorators (namespace-only, no MRO base).

    Governance: Pure namespace class with only @staticmethod members. No state,
    no instantiation, no fields. BaseModel inheritance not required per §3.1.

    Scans modules and classes for functions decorated with @factory() and provides
    utilities for finding and analyzing factory configurations.

    This class enables zero-config factory registration in FlextContainer
    by automatically discovering decorated functions at initialization time.
    """

    @staticmethod
    def has_factories(module: ModuleType) -> bool:
        """Check if module has any factory-decorated functions."""
        for name in dir(module):
            if name.startswith("_"):
                continue
            candidate = vars(module).get(name)
            if candidate is None:
                continue
            if callable(candidate) and hasattr(candidate, c.Discovery.FACTORY_ATTR):
                return True
        return False

    @staticmethod
    def scan_module(module: ModuleType) -> list[tuple[str, m.FactoryDecoratorConfig]]:
        """Scan module for functions decorated with @factory().

        Introspects the module to find all functions with factory configuration
        metadata, returning them sorted by name for consistent ordering.

        Args:
            module: Module object to scan for factory decorators

        Returns:
            List of tuples (function_name, FactoryDecoratorConfig) sorted by name

        Example:
            >>> from flext_core import FactoryDecoratorsDiscovery
            >>> factories = FactoryDecoratorsDiscovery.scan_module(my_module)
            >>> for func_name, config in factories:
            ...     print(f"{func_name}: singleton={config.singleton}")

        """
        factories: list[tuple[str, m.FactoryDecoratorConfig]] = []
        for name in dir(module):
            if name.startswith("_"):
                continue
            func = vars(module).get(name)
            if func is None:
                continue
            if callable(func) and hasattr(func, c.Discovery.FACTORY_ATTR):
                config_raw = vars(func).get(c.Discovery.FACTORY_ATTR)
                if not isinstance(config_raw, m.FactoryDecoratorConfig):
                    continue
                config: m.FactoryDecoratorConfig = config_raw
                factories.append((name, config))
        return sorted(factories, key=operator.itemgetter(0))
