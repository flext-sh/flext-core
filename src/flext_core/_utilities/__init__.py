# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Internal utilities package - implementation modules only.

This package contains implementation modules for u.
External code MUST NOT import from this package directly.
Use u from flext_core instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.cache import FlextUtilitiesCache
    from flext_core._utilities.checker import FlextUtilitiesChecker
    from flext_core._utilities.collection import FlextUtilitiesCollection
    from flext_core._utilities.configuration import FlextUtilitiesConfiguration
    from flext_core._utilities.context import FlextUtilitiesContext
    from flext_core._utilities.conversion import FlextUtilitiesConversion
    from flext_core._utilities.deprecation import FlextUtilitiesDeprecation
    from flext_core._utilities.discovery import FlextUtilitiesDiscovery
    from flext_core._utilities.domain import FlextUtilitiesDomain
    from flext_core._utilities.enum import FlextUtilitiesEnum
    from flext_core._utilities.file_ops import FlextUtilitiesFileOps
    from flext_core._utilities.generators import FlextUtilitiesGenerators
    from flext_core._utilities.guards import FlextUtilitiesGuards
    from flext_core._utilities.guards_ensure import FlextUtilitiesGuardsEnsure
    from flext_core._utilities.guards_type import FlextUtilitiesGuardsType
    from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
    from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
    from flext_core._utilities.guards_type_protocol import (
        FlextUtilitiesGuardsTypeProtocol,
    )
    from flext_core._utilities.mapper import FlextUtilitiesMapper
    from flext_core._utilities.model import FlextUtilitiesModel
    from flext_core._utilities.pagination import FlextUtilitiesPagination
    from flext_core._utilities.parser import FlextUtilitiesParser
    from flext_core._utilities.reliability import FlextUtilitiesReliability
    from flext_core._utilities.result_helpers import FlextUtilitiesResultHelpers
    from flext_core._utilities.text import FlextUtilitiesText

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextUtilitiesArgs": ("flext_core._utilities.args", "FlextUtilitiesArgs"),
    "FlextUtilitiesCache": ("flext_core._utilities.cache", "FlextUtilitiesCache"),
    "FlextUtilitiesChecker": ("flext_core._utilities.checker", "FlextUtilitiesChecker"),
    "FlextUtilitiesCollection": (
        "flext_core._utilities.collection",
        "FlextUtilitiesCollection",
    ),
    "FlextUtilitiesConfiguration": (
        "flext_core._utilities.configuration",
        "FlextUtilitiesConfiguration",
    ),
    "FlextUtilitiesContext": ("flext_core._utilities.context", "FlextUtilitiesContext"),
    "FlextUtilitiesConversion": (
        "flext_core._utilities.conversion",
        "FlextUtilitiesConversion",
    ),
    "FlextUtilitiesDeprecation": (
        "flext_core._utilities.deprecation",
        "FlextUtilitiesDeprecation",
    ),
    "FlextUtilitiesDiscovery": (
        "flext_core._utilities.discovery",
        "FlextUtilitiesDiscovery",
    ),
    "FlextUtilitiesDomain": ("flext_core._utilities.domain", "FlextUtilitiesDomain"),
    "FlextUtilitiesEnum": ("flext_core._utilities.enum", "FlextUtilitiesEnum"),
    "FlextUtilitiesFileOps": (
        "flext_core._utilities.file_ops",
        "FlextUtilitiesFileOps",
    ),
    "FlextUtilitiesGenerators": (
        "flext_core._utilities.generators",
        "FlextUtilitiesGenerators",
    ),
    "FlextUtilitiesGuards": ("flext_core._utilities.guards", "FlextUtilitiesGuards"),
    "FlextUtilitiesGuardsEnsure": (
        "flext_core._utilities.guards_ensure",
        "FlextUtilitiesGuardsEnsure",
    ),
    "FlextUtilitiesGuardsType": (
        "flext_core._utilities.guards_type",
        "FlextUtilitiesGuardsType",
    ),
    "FlextUtilitiesGuardsTypeCore": (
        "flext_core._utilities.guards_type_core",
        "FlextUtilitiesGuardsTypeCore",
    ),
    "FlextUtilitiesGuardsTypeModel": (
        "flext_core._utilities.guards_type_model",
        "FlextUtilitiesGuardsTypeModel",
    ),
    "FlextUtilitiesGuardsTypeProtocol": (
        "flext_core._utilities.guards_type_protocol",
        "FlextUtilitiesGuardsTypeProtocol",
    ),
    "FlextUtilitiesMapper": ("flext_core._utilities.mapper", "FlextUtilitiesMapper"),
    "FlextUtilitiesModel": ("flext_core._utilities.model", "FlextUtilitiesModel"),
    "FlextUtilitiesPagination": (
        "flext_core._utilities.pagination",
        "FlextUtilitiesPagination",
    ),
    "FlextUtilitiesParser": ("flext_core._utilities.parser", "FlextUtilitiesParser"),
    "FlextUtilitiesReliability": (
        "flext_core._utilities.reliability",
        "FlextUtilitiesReliability",
    ),
    "FlextUtilitiesResultHelpers": (
        "flext_core._utilities.result_helpers",
        "FlextUtilitiesResultHelpers",
    ),
    "FlextUtilitiesText": ("flext_core._utilities.text", "FlextUtilitiesText"),
}

__all__ = [
    "FlextUtilitiesArgs",
    "FlextUtilitiesCache",
    "FlextUtilitiesChecker",
    "FlextUtilitiesCollection",
    "FlextUtilitiesConfiguration",
    "FlextUtilitiesContext",
    "FlextUtilitiesConversion",
    "FlextUtilitiesDeprecation",
    "FlextUtilitiesDiscovery",
    "FlextUtilitiesDomain",
    "FlextUtilitiesEnum",
    "FlextUtilitiesFileOps",
    "FlextUtilitiesGenerators",
    "FlextUtilitiesGuards",
    "FlextUtilitiesGuardsEnsure",
    "FlextUtilitiesGuardsType",
    "FlextUtilitiesGuardsTypeCore",
    "FlextUtilitiesGuardsTypeModel",
    "FlextUtilitiesGuardsTypeProtocol",
    "FlextUtilitiesMapper",
    "FlextUtilitiesModel",
    "FlextUtilitiesPagination",
    "FlextUtilitiesParser",
    "FlextUtilitiesReliability",
    "FlextUtilitiesResultHelpers",
    "FlextUtilitiesText",
]


_LAZY_CACHE: dict[str, FlextTypes.ModuleExport] = {}


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562).

    A local cache ``_LAZY_CACHE`` persists resolved objects across repeated
    accesses during process lifetime.

    Args:
        name: Attribute name requested by dir()/import.

    Returns:
        Lazy-loaded module export type.

    Raises:
        AttributeError: If attribute not registered.

    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    value = lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)
    _LAZY_CACHE[name] = value
    return value


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete.

    Returns:
        List of public names from module exports.

    """
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
