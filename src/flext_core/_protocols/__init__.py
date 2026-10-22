# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols import (
        _container as _container,
        _context as _context,
        _logging as _logging,
    )
    from flext_core._protocols.base import FlextProtocolsBase
    from flext_core._protocols.config import FlextProtocolsConfig
    from flext_core._protocols.container import FlextProtocolsContainer
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.loggings import FlextProtocolsLogging
    from flext_core._protocols.project_metadata import FlextProtocolsProjectMetadata
    from flext_core._protocols.pydantic import FlextProtocolsPydantic
    from flext_core._protocols.registry import FlextProtocolsRegistry
    from flext_core._protocols.result import FlextProtocolsResult
    from flext_core._protocols.service import FlextProtocolsService
    from flext_core._protocols.settings import FlextProtocolsSettings


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
    "_container",
    "_context",
    "_logging",
)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextProtocolsBase": ".base",
        "FlextProtocolsConfig": ".config",
        "FlextProtocolsContainer": ".container",
        "FlextProtocolsContext": ".context",
        "FlextProtocolsHandler": ".handler",
        "FlextProtocolsLogging": ".loggings",
        "FlextProtocolsProjectMetadata": ".project_metadata",
        "FlextProtocolsPydantic": ".pydantic",
        "FlextProtocolsRegistry": ".registry",
        "FlextProtocolsResult": ".result",
        "FlextProtocolsService": ".service",
        "FlextProtocolsSettings": ".settings",
        "_container": "._container",
        "_context": "._context",
        "_logging": "._logging",
    }),
    public_exports=__all__,
)[0]
