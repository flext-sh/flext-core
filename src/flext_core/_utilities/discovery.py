"""Factory discovery implementation for auto-registration.

This module provides factory discovery functionality that can be used by
container and decorators without creating circular dependencies.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from collections.abc import MutableSequence, Sequence
from types import ModuleType

from flext_core import FlextModelsContainer, c, t


class FlextUtilitiesDiscovery:
    """Auto-discovery for @factory() decorated functions in modules."""

    @staticmethod
    def scan_module(
        module: ModuleType,
    ) -> Sequence[tuple[str, FlextModelsContainer.FactoryDecoratorConfig]]:
        """Scan module for @factory()-decorated functions, sorted by name."""
        return sorted(
            [
                (name, config_raw)
                for name in dir(module)
                if not name.startswith("_")
                and (func := vars(module).get(name)) is not None
                and callable(func)
                and hasattr(func, c.FACTORY_ATTR)
                and isinstance(
                    (config_raw := vars(func).get(c.FACTORY_ATTR)),
                    FlextModelsContainer.FactoryDecoratorConfig,
                )
            ],
            key=operator.itemgetter(0),
        )

    @staticmethod
    def resolve_wire_targets(
        wire_modules: Sequence[ModuleType | str] | None,
        wire_packages: t.StrSequence | None,
        wire_classes: Sequence[type] | None,
    ) -> tuple[
        Sequence[ModuleType] | None,
        t.StrSequence | None,
        Sequence[type] | None,
    ]:
        """Separate mixed wire_modules into actual modules vs package name strings."""
        resolved_modules: Sequence[ModuleType] | None = None
        resolved_packages: t.StrSequence | None = None
        resolved_classes: Sequence[type] | None = wire_classes

        if wire_modules is not None:
            modules_list: MutableSequence[ModuleType] = []
            packages_list: MutableSequence[str] = []
            for item in wire_modules:
                match item:
                    case str():
                        packages_list.append(item)
                    case _:
                        modules_list.append(item)
            resolved_modules = modules_list
            if packages_list:
                resolved_packages = packages_list

        if wire_packages is not None:
            current = list(resolved_packages or [])
            current.extend(wire_packages)
            resolved_packages = current

        return resolved_modules, resolved_packages, resolved_classes


__all__: list[str] = ["FlextUtilitiesDiscovery"]
