# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Factory decorator discovery utilities.

This module provides factory discovery functionality that can be used by
container and decorators without creating circular dependencies.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core._utilities.discovery import FlextUtilitiesDiscovery
    from flext_core.typings import FlextTypes

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FactoryDecoratorsDiscovery": (
        "flext_core._utilities.discovery",
        "FlextUtilitiesDiscovery",
    ),
}

__all__ = [
    "FactoryDecoratorsDiscovery",
    "FlextUtilitiesDiscovery",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


# Backward compatibility alias
FactoryDecoratorsDiscovery = __getattr__("FactoryDecoratorsDiscovery")

cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
