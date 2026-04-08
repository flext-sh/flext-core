# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Protocols package."""

from __future__ import annotations

from flext_core.lazy import install_lazy_exports

_LAZY_IMPORTS = {
    "FlextProtocolsBase": ("flext_core._protocols.base", "FlextProtocolsBase"),
    "FlextProtocolsConfig": ("flext_core._protocols.config", "FlextProtocolsConfig"),
    "FlextProtocolsContainer": (
        "flext_core._protocols.container",
        "FlextProtocolsContainer",
    ),
    "FlextProtocolsContext": ("flext_core._protocols.context", "FlextProtocolsContext"),
    "FlextProtocolsHandler": ("flext_core._protocols.handler", "FlextProtocolsHandler"),
    "FlextProtocolsLogging": ("flext_core._protocols.logging", "FlextProtocolsLogging"),
    "FlextProtocolsRegistry": (
        "flext_core._protocols.registry",
        "FlextProtocolsRegistry",
    ),
    "FlextProtocolsResult": ("flext_core._protocols.result", "FlextProtocolsResult"),
    "FlextProtocolsService": ("flext_core._protocols.service", "FlextProtocolsService"),
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
