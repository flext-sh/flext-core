# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Internal module for FlextModels nested classes.

This module contains extracted nested classes from FlextModels to improve
maintainability.

All classes are re-exported through FlextModels in models.py - users should
NEVER import from this module directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr


if TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_core._models._context._data import FlextModelsContextData
    from flext_core._models._context._export import FlextModelsContextExport
    from flext_core._models._context._metadata import FlextModelsContextMetadata
    from flext_core._models._context._proxy_var import FlextModelsContextProxyVar
    from flext_core._models._context._scope import FlextModelsContextScope
    from flext_core._models._context._tokens import FlextModelsContextTokens
    from flext_core._models.base import FlextModelFoundation
    from flext_core._models.collections import FlextModelsCollections
    from flext_core._models.container import FlextModelsContainer
    from flext_core._models.containers import FlextModelsContainers
    from flext_core._models.context import FlextModelsContext
    from flext_core._models.cqrs import FlextModelsCqrs
    from flext_core._models.decorators import FlextModelsDecorators
    from flext_core._models.dispatcher import FlextModelsDispatcher
    from flext_core._models.domain_event import FlextModelsDomainEvent
    from flext_core._models.entity import FlextModelsEntity
    from flext_core._models.generic import FlextGenericModels
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.result import FlextModelsResult
    from flext_core._models.service import FlextModelsService
    from flext_core._models.settings import FlextModelsConfig

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "FlextGenericModels": ["flext_core._models.generic", "FlextGenericModels"],
    "FlextModelFoundation": ["flext_core._models.base", "FlextModelFoundation"],
    "FlextModelsCollections": ["flext_core._models.collections", "FlextModelsCollections"],
    "FlextModelsConfig": ["flext_core._models.settings", "FlextModelsConfig"],
    "FlextModelsContainer": ["flext_core._models.container", "FlextModelsContainer"],
    "FlextModelsContainers": ["flext_core._models.containers", "FlextModelsContainers"],
    "FlextModelsContext": ["flext_core._models.context", "FlextModelsContext"],
    "FlextModelsContextData": ["flext_core._models._context._data", "FlextModelsContextData"],
    "FlextModelsContextExport": ["flext_core._models._context._export", "FlextModelsContextExport"],
    "FlextModelsContextMetadata": ["flext_core._models._context._metadata", "FlextModelsContextMetadata"],
    "FlextModelsContextProxyVar": ["flext_core._models._context._proxy_var", "FlextModelsContextProxyVar"],
    "FlextModelsContextScope": ["flext_core._models._context._scope", "FlextModelsContextScope"],
    "FlextModelsContextTokens": ["flext_core._models._context._tokens", "FlextModelsContextTokens"],
    "FlextModelsCqrs": ["flext_core._models.cqrs", "FlextModelsCqrs"],
    "FlextModelsDecorators": ["flext_core._models.decorators", "FlextModelsDecorators"],
    "FlextModelsDispatcher": ["flext_core._models.dispatcher", "FlextModelsDispatcher"],
    "FlextModelsDomainEvent": ["flext_core._models.domain_event", "FlextModelsDomainEvent"],
    "FlextModelsEntity": ["flext_core._models.entity", "FlextModelsEntity"],
    "FlextModelsHandler": ["flext_core._models.handler", "FlextModelsHandler"],
    "FlextModelsResult": ["flext_core._models.result", "FlextModelsResult"],
    "FlextModelsService": ["flext_core._models.service", "FlextModelsService"],
}

__all__ = [
    "FlextGenericModels",
    "FlextModelFoundation",
    "FlextModelsCollections",
    "FlextModelsConfig",
    "FlextModelsContainer",
    "FlextModelsContainers",
    "FlextModelsContext",
    "FlextModelsContextData",
    "FlextModelsContextExport",
    "FlextModelsContextMetadata",
    "FlextModelsContextProxyVar",
    "FlextModelsContextScope",
    "FlextModelsContextTokens",
    "FlextModelsCqrs",
    "FlextModelsDecorators",
    "FlextModelsDispatcher",
    "FlextModelsDomainEvent",
    "FlextModelsEntity",
    "FlextModelsHandler",
    "FlextModelsResult",
    "FlextModelsService",
]


_LAZY_CACHE: MutableMapping[str, FlextTypes.ModuleExport] = {}


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


def __dir__() -> Sequence[str]:
    """Return list of available attributes for dir() and autocomplete.

    Returns:
        List of public names from module exports.
    """
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)