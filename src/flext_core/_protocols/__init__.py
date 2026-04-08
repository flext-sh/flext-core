# AUTO-GENERATED FILE — Regenerate with: make gen
"""Protocols package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".base": ("FlextProtocolsBase",),
        ".config": ("FlextProtocolsConfig",),
        ".container": ("FlextProtocolsContainer",),
        ".context": ("FlextProtocolsContext",),
        ".handler": ("FlextProtocolsHandler",),
        ".logging": ("FlextProtocolsLogging",),
        ".registry": ("FlextProtocolsRegistry",),
        ".result": ("FlextProtocolsResult",),
        ".service": ("FlextProtocolsService",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
