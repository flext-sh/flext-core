"""Factory discovery implementation for auto-registration.

This module provides factory discovery functionality that can be used by
container and decorators without creating circular dependencies.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from flext_core import FlextConstants as c, FlextTypes as t
from flext_core._models.container import FlextModelsContainer

if TYPE_CHECKING:
    from collections.abc import MutableSequence
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

    @staticmethod
    def resolve_wire_targets(
        wire_modules: t.SequenceOf[ModuleType | str] | None,
        wire_packages: t.StrSequence | None,
        wire_classes: t.SequenceOf[type] | None,
    ) -> tuple[
        t.SequenceOf[ModuleType] | None, t.StrSequence | None, t.SequenceOf[type] | None
    ]:
        """Separate mixed wire_modules into actual modules vs package name strings."""
        resolved_modules: t.SequenceOf[ModuleType] | None = None
        resolved_packages: t.StrSequence | None = None
        resolved_classes: t.SequenceOf[type] | None = wire_classes

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
