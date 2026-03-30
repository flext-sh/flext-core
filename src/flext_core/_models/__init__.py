# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
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

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models import (
        base as base,
        collections as collections,
        container as container,
        containers as containers,
        context as context,
        cqrs as cqrs,
        decorators as decorators,
        dispatcher as dispatcher,
        domain_event as domain_event,
        entity as entity,
        errors as errors,
        exception_params as exception_params,
        generic as generic,
        handler as handler,
        service as service,
        settings as settings,
    )
    from flext_core._models._context._data import (
        FlextModelsContextData as FlextModelsContextData,
    )
    from flext_core._models._context._export import (
        FlextModelsContextExport as FlextModelsContextExport,
    )
    from flext_core._models._context._metadata import (
        FlextModelsContextMetadata as FlextModelsContextMetadata,
    )
    from flext_core._models._context._proxy_var import (
        FlextModelsContextProxyVar as FlextModelsContextProxyVar,
    )
    from flext_core._models._context._scope import (
        FlextModelsContextScope as FlextModelsContextScope,
    )
    from flext_core._models._context._tokens import (
        FlextModelsContextTokens as FlextModelsContextTokens,
    )
    from flext_core._models.base import FlextModelFoundation as FlextModelFoundation
    from flext_core._models.collections import (
        FlextModelsCollections as FlextModelsCollections,
    )
    from flext_core._models.container import (
        FlextModelsContainer as FlextModelsContainer,
    )
    from flext_core._models.containers import (
        FlextModelsContainers as FlextModelsContainers,
    )
    from flext_core._models.context import FlextModelsContext as FlextModelsContext
    from flext_core._models.cqrs import FlextModelsCqrs as FlextModelsCqrs
    from flext_core._models.decorators import (
        FlextModelsDecorators as FlextModelsDecorators,
    )
    from flext_core._models.dispatcher import (
        FlextModelsDispatcher as FlextModelsDispatcher,
    )
    from flext_core._models.domain_event import (
        FlextModelsDomainEvent as FlextModelsDomainEvent,
    )
    from flext_core._models.entity import FlextModelsEntity as FlextModelsEntity
    from flext_core._models.errors import FlextModelsErrors as FlextModelsErrors
    from flext_core._models.exception_params import (
        FlextModelsExceptionParams as FlextModelsExceptionParams,
    )
    from flext_core._models.generic import FlextGenericModels as FlextGenericModels
    from flext_core._models.handler import FlextModelsHandler as FlextModelsHandler
    from flext_core._models.service import FlextModelsService as FlextModelsService
    from flext_core._models.settings import FlextModelsConfig as FlextModelsConfig

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "FlextGenericModels": ["flext_core._models.generic", "FlextGenericModels"],
    "FlextModelFoundation": ["flext_core._models.base", "FlextModelFoundation"],
    "FlextModelsCollections": [
        "flext_core._models.collections",
        "FlextModelsCollections",
    ],
    "FlextModelsConfig": ["flext_core._models.settings", "FlextModelsConfig"],
    "FlextModelsContainer": ["flext_core._models.container", "FlextModelsContainer"],
    "FlextModelsContainers": ["flext_core._models.containers", "FlextModelsContainers"],
    "FlextModelsContext": ["flext_core._models.context", "FlextModelsContext"],
    "FlextModelsContextData": [
        "flext_core._models._context._data",
        "FlextModelsContextData",
    ],
    "FlextModelsContextExport": [
        "flext_core._models._context._export",
        "FlextModelsContextExport",
    ],
    "FlextModelsContextMetadata": [
        "flext_core._models._context._metadata",
        "FlextModelsContextMetadata",
    ],
    "FlextModelsContextProxyVar": [
        "flext_core._models._context._proxy_var",
        "FlextModelsContextProxyVar",
    ],
    "FlextModelsContextScope": [
        "flext_core._models._context._scope",
        "FlextModelsContextScope",
    ],
    "FlextModelsContextTokens": [
        "flext_core._models._context._tokens",
        "FlextModelsContextTokens",
    ],
    "FlextModelsCqrs": ["flext_core._models.cqrs", "FlextModelsCqrs"],
    "FlextModelsDecorators": ["flext_core._models.decorators", "FlextModelsDecorators"],
    "FlextModelsDispatcher": ["flext_core._models.dispatcher", "FlextModelsDispatcher"],
    "FlextModelsDomainEvent": [
        "flext_core._models.domain_event",
        "FlextModelsDomainEvent",
    ],
    "FlextModelsEntity": ["flext_core._models.entity", "FlextModelsEntity"],
    "FlextModelsErrors": ["flext_core._models.errors", "FlextModelsErrors"],
    "FlextModelsExceptionParams": [
        "flext_core._models.exception_params",
        "FlextModelsExceptionParams",
    ],
    "FlextModelsHandler": ["flext_core._models.handler", "FlextModelsHandler"],
    "FlextModelsService": ["flext_core._models.service", "FlextModelsService"],
    "base": ["flext_core._models.base", ""],
    "collections": ["flext_core._models.collections", ""],
    "container": ["flext_core._models.container", ""],
    "containers": ["flext_core._models.containers", ""],
    "context": ["flext_core._models.context", ""],
    "cqrs": ["flext_core._models.cqrs", ""],
    "decorators": ["flext_core._models.decorators", ""],
    "dispatcher": ["flext_core._models.dispatcher", ""],
    "domain_event": ["flext_core._models.domain_event", ""],
    "entity": ["flext_core._models.entity", ""],
    "errors": ["flext_core._models.errors", ""],
    "exception_params": ["flext_core._models.exception_params", ""],
    "generic": ["flext_core._models.generic", ""],
    "handler": ["flext_core._models.handler", ""],
    "service": ["flext_core._models.service", ""],
    "settings": ["flext_core._models.settings", ""],
}

_EXPORTS: Sequence[str] = [
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
    "FlextModelsErrors",
    "FlextModelsExceptionParams",
    "FlextModelsHandler",
    "FlextModelsService",
    "base",
    "collections",
    "container",
    "containers",
    "context",
    "cqrs",
    "decorators",
    "dispatcher",
    "domain_event",
    "entity",
    "errors",
    "exception_params",
    "generic",
    "handler",
    "service",
    "settings",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
