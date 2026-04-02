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
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_core._models import (
        _context,
        base,
        collections,
        container,
        containers,
        context,
        cqrs,
        decorators,
        dispatcher,
        domain_event,
        entity,
        errors,
        exception_params,
        generic,
        handler,
        service,
        settings,
    )
    from flext_core._models._context import (
        FlextModelsContextData,
        FlextModelsContextExport,
        FlextModelsContextMetadata,
        FlextModelsContextProxyVar,
        FlextModelsContextScope,
        FlextModelsContextTokens,
    )
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
    from flext_core._models.errors import FlextModelsErrors
    from flext_core._models.exception_params import FlextModelsExceptionParams
    from flext_core._models.generic import FlextGenericModels
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.service import FlextModelsService
    from flext_core._models.settings import FlextModelsConfig

_LAZY_IMPORTS: FlextTypes.LazyImportIndex = merge_lazy_imports(
    ("flext_core._models._context",),
    {
        "FlextGenericModels": "flext_core._models.generic",
        "FlextModelFoundation": "flext_core._models.base",
        "FlextModelsCollections": "flext_core._models.collections",
        "FlextModelsConfig": "flext_core._models.settings",
        "FlextModelsContainer": "flext_core._models.container",
        "FlextModelsContainers": "flext_core._models.containers",
        "FlextModelsContext": "flext_core._models.context",
        "FlextModelsCqrs": "flext_core._models.cqrs",
        "FlextModelsDecorators": "flext_core._models.decorators",
        "FlextModelsDispatcher": "flext_core._models.dispatcher",
        "FlextModelsDomainEvent": "flext_core._models.domain_event",
        "FlextModelsEntity": "flext_core._models.entity",
        "FlextModelsErrors": "flext_core._models.errors",
        "FlextModelsExceptionParams": "flext_core._models.exception_params",
        "FlextModelsHandler": "flext_core._models.handler",
        "FlextModelsService": "flext_core._models.service",
        "_context": "flext_core._models._context",
        "base": "flext_core._models.base",
        "collections": "flext_core._models.collections",
        "container": "flext_core._models.container",
        "containers": "flext_core._models.containers",
        "context": "flext_core._models.context",
        "cqrs": "flext_core._models.cqrs",
        "decorators": "flext_core._models.decorators",
        "dispatcher": "flext_core._models.dispatcher",
        "domain_event": "flext_core._models.domain_event",
        "entity": "flext_core._models.entity",
        "errors": "flext_core._models.errors",
        "exception_params": "flext_core._models.exception_params",
        "generic": "flext_core._models.generic",
        "handler": "flext_core._models.handler",
        "service": "flext_core._models.service",
        "settings": "flext_core._models.settings",
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
