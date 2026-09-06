# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import (
        _container_parts as _container_parts,
        _context_parts as _context_parts,
        _logging_parts as _logging_parts,
    )
    from .base import FlextProtocolsBase
    from .config import FlextProtocolsConfig
    from .container import FlextProtocolsContainer
    from .context import FlextProtocolsContext
    from .handler import FlextProtocolsHandler
    from .logging import FlextProtocolsLogging
    from .project_metadata import FlextProtocolsProjectMetadata
    from .pydantic import FlextProtocolsPydantic
    from .registry import FlextProtocolsRegistry
    from .result import FlextProtocolsResult
    from .service import FlextProtocolsService
    from .settings import FlextProtocolsSettings
__all__: tuple[str, ...] = (
    "FlextProtocolsBase",
    "FlextProtocolsConfig",
    "FlextProtocolsContainer",
    "FlextProtocolsContext",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsProjectMetadata",
    "FlextProtocolsPydantic",
    "FlextProtocolsRegistry",
    "FlextProtocolsResult",
    "FlextProtocolsService",
    "FlextProtocolsSettings",
    "_container_parts",
    "_context_parts",
    "_logging_parts",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._container_parts": ("_container_parts",),
            "._context_parts": ("_context_parts",),
            "._logging_parts": ("_logging_parts",),
            ".base": ("FlextProtocolsBase",),
            ".config": ("FlextProtocolsConfig",),
            ".container": ("FlextProtocolsContainer",),
            ".context": ("FlextProtocolsContext",),
            ".handler": ("FlextProtocolsHandler",),
            ".logging": ("FlextProtocolsLogging",),
            ".project_metadata": ("FlextProtocolsProjectMetadata",),
            ".pydantic": ("FlextProtocolsPydantic",),
            ".registry": ("FlextProtocolsRegistry",),
            ".result": ("FlextProtocolsResult",),
            ".service": ("FlextProtocolsService",),
            ".settings": ("FlextProtocolsSettings",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
