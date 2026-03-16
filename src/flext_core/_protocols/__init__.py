# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Internal module for FlextProtocols nested classes.

This module contains extracted nested classes from FlextProtocols to improve
maintainability.

All classes are re-exported through FlextProtocols in protocols.py - users should
NEVER import from this module directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core._protocols.base import FlextProtocolsBase
    from flext_core._protocols.config import FlextProtocolsConfig
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.di import FlextProtocolsDI
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.introspection import (
        _METACLASS_STRICT,
        _ProtocolIntrospection,
    )
    from flext_core._protocols.logging import FlextProtocolsLogging
    from flext_core._protocols.metaclass import (
        FlextProtocolsMetaclassUtilities,
        ProtocolModel,
        ProtocolModelMeta,
        ProtocolSettings,
        _CombinedModelMeta,
    )
    from flext_core._protocols.metrics import FlextProtocolsMetrics
    from flext_core._protocols.result import FlextProtocolsResult
    from flext_core._protocols.service import FlextProtocolsService
    from flext_core.typings import FlextTypes

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextProtocolsBase": ("flext_core._protocols.base", "FlextProtocolsBase"),
    "FlextProtocolsConfig": ("flext_core._protocols.config", "FlextProtocolsConfig"),
    "FlextProtocolsContext": ("flext_core._protocols.context", "FlextProtocolsContext"),
    "FlextProtocolsDI": ("flext_core._protocols.di", "FlextProtocolsDI"),
    "FlextProtocolsHandler": ("flext_core._protocols.handler", "FlextProtocolsHandler"),
    "FlextProtocolsLogging": ("flext_core._protocols.logging", "FlextProtocolsLogging"),
    "FlextProtocolsMetaclassUtilities": (
        "flext_core._protocols.metaclass",
        "FlextProtocolsMetaclassUtilities",
    ),
    "FlextProtocolsMetrics": ("flext_core._protocols.metrics", "FlextProtocolsMetrics"),
    "FlextProtocolsResult": ("flext_core._protocols.result", "FlextProtocolsResult"),
    "FlextProtocolsService": ("flext_core._protocols.service", "FlextProtocolsService"),
    "ProtocolModel": ("flext_core._protocols.metaclass", "ProtocolModel"),
    "ProtocolModelMeta": ("flext_core._protocols.metaclass", "ProtocolModelMeta"),
    "ProtocolSettings": ("flext_core._protocols.metaclass", "ProtocolSettings"),
    "_CombinedModelMeta": ("flext_core._protocols.metaclass", "_CombinedModelMeta"),
    "_METACLASS_STRICT": ("flext_core._protocols.introspection", "_METACLASS_STRICT"),
    "_ProtocolIntrospection": (
        "flext_core._protocols.introspection",
        "_ProtocolIntrospection",
    ),
}

__all__ = [
    "_METACLASS_STRICT",
    "FlextProtocolsBase",
    "FlextProtocolsConfig",
    "FlextProtocolsContext",
    "FlextProtocolsDI",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsMetaclassUtilities",
    "FlextProtocolsMetrics",
    "FlextProtocolsResult",
    "FlextProtocolsService",
    "ProtocolModel",
    "ProtocolModelMeta",
    "ProtocolSettings",
    "_CombinedModelMeta",
    "_ProtocolIntrospection",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
