# AUTO-GENERATED FILE — Regenerate with: make gen
"""Protocols package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._container_parts": ("_container_parts",),
        "._container_parts.flextprotocolscontainer_part_03": (
            "FlextProtocolsContainer",
        ),
        "._context_parts": ("_context_parts",),
        "._context_parts.flextprotocolscontext_part_03": ("FlextProtocolsContext",),
        "._logging_parts": ("_logging_parts",),
        "._logging_parts.flextprotocolslogging_part_03": ("FlextProtocolsLogging",),
        "._result_parts": ("_result_parts",),
        "._result_parts.flextprotocolsresult_part_04": ("FlextProtocolsResult",),
        ".base": ("FlextProtocolsBase",),
        ".handler": ("FlextProtocolsHandler",),
        ".pydantic": ("FlextProtocolsPydantic",),
        ".registry": ("FlextProtocolsRegistry",),
        ".service": ("FlextProtocolsService",),
        ".settings": ("FlextProtocolsSettings",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
