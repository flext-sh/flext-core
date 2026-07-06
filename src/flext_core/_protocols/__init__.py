# AUTO-GENERATED FILE — Regenerate with: make gen
"""Protocols package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._container_parts.flextprotocolscontainer_part_03 import (
        FlextProtocolsContainer,
    )
    from flext_core._protocols._context_parts.flextprotocolscontext_part_03 import (
        FlextProtocolsContext,
    )
    from flext_core._protocols._logging_parts.flextprotocolslogging_part_03 import (
        FlextProtocolsLogging,
    )
    from flext_core._protocols._result_parts.flextprotocolsresult_part_04 import (
        FlextProtocolsResult,
    )
    from flext_core._protocols.base import FlextProtocolsBase
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.pydantic import FlextProtocolsPydantic
    from flext_core._protocols.registry import FlextProtocolsRegistry
    from flext_core._protocols.service import FlextProtocolsService
    from flext_core._protocols.settings import FlextProtocolsSettings
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


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
