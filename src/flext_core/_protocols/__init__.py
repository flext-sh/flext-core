# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Protocols package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import flext_core._protocols.base as _flext_core__protocols_base

    base = _flext_core__protocols_base
    import flext_core._protocols.config as _flext_core__protocols_config
    from flext_core._protocols.base import FlextProtocolsBase

    config = _flext_core__protocols_config
    import flext_core._protocols.container as _flext_core__protocols_container
    from flext_core._protocols.config import FlextProtocolsConfig

    container = _flext_core__protocols_container
    import flext_core._protocols.context as _flext_core__protocols_context
    from flext_core._protocols.container import FlextProtocolsContainer

    context = _flext_core__protocols_context
    import flext_core._protocols.handler as _flext_core__protocols_handler
    from flext_core._protocols.context import FlextProtocolsContext

    handler = _flext_core__protocols_handler
    import flext_core._protocols.logging as _flext_core__protocols_logging
    from flext_core._protocols.handler import FlextProtocolsHandler

    logging = _flext_core__protocols_logging
    import flext_core._protocols.registry as _flext_core__protocols_registry
    from flext_core._protocols.logging import FlextProtocolsLogging

    registry = _flext_core__protocols_registry
    import flext_core._protocols.result as _flext_core__protocols_result
    from flext_core._protocols.registry import FlextProtocolsRegistry

    result = _flext_core__protocols_result
    import flext_core._protocols.service as _flext_core__protocols_service
    from flext_core._protocols.result import FlextProtocolsResult

    service = _flext_core__protocols_service
    from flext_core._protocols.service import FlextProtocolsService
_LAZY_IMPORTS = {
    "FlextProtocolsBase": "flext_core._protocols.base",
    "FlextProtocolsConfig": "flext_core._protocols.config",
    "FlextProtocolsContainer": "flext_core._protocols.container",
    "FlextProtocolsContext": "flext_core._protocols.context",
    "FlextProtocolsHandler": "flext_core._protocols.handler",
    "FlextProtocolsLogging": "flext_core._protocols.logging",
    "FlextProtocolsRegistry": "flext_core._protocols.registry",
    "FlextProtocolsResult": "flext_core._protocols.result",
    "FlextProtocolsService": "flext_core._protocols.service",
    "base": "flext_core._protocols.base",
    "config": "flext_core._protocols.config",
    "container": "flext_core._protocols.container",
    "context": "flext_core._protocols.context",
    "handler": "flext_core._protocols.handler",
    "logging": "flext_core._protocols.logging",
    "registry": "flext_core._protocols.registry",
    "result": "flext_core._protocols.result",
    "service": "flext_core._protocols.service",
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
    "base",
    "config",
    "container",
    "context",
    "handler",
    "logging",
    "registry",
    "result",
    "service",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
