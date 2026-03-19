# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Internal module for FlextProtocols nested classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core._protocols.base import FlextProtocolsBase
    from flext_core._protocols.config import FlextProtocolsConfig
    from flext_core._protocols.container import FlextProtocolsContainer
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.logging import FlextProtocolsLogging
    from flext_core._protocols.registry import FlextProtocolsRegistry
    from flext_core._protocols.result import FlextProtocolsResult
    from flext_core._protocols.service import FlextProtocolsService
    from flext_core.typings import FlextTypes

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextProtocolsBase": ("flext_core._protocols.base", "FlextProtocolsBase"),
    "FlextProtocolsConfig": ("flext_core._protocols.config", "FlextProtocolsConfig"),
    "FlextProtocolsContainer": ("flext_core._protocols.container", "FlextProtocolsContainer"),
    "FlextProtocolsContext": ("flext_core._protocols.context", "FlextProtocolsContext"),
    "FlextProtocolsHandler": ("flext_core._protocols.handler", "FlextProtocolsHandler"),
    "FlextProtocolsLogging": ("flext_core._protocols.logging", "FlextProtocolsLogging"),
    "FlextProtocolsRegistry": ("flext_core._protocols.registry", "FlextProtocolsRegistry"),
    "FlextProtocolsResult": ("flext_core._protocols.result", "FlextProtocolsResult"),
    "FlextProtocolsService": ("flext_core._protocols.service", "FlextProtocolsService"),
}

__all__ = [
    "FlextProtocolsBase",
    "FlextProtocolsConfig",
    "FlextProtocolsContainer",
    "FlextProtocolsContext",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsRegistry",
    "FlextProtocolsResult",
    "FlextProtocolsService",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
