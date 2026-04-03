# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Models package."""

from __future__ import annotations

import typing as _t

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
from flext_core._models.errors import FlextModelsErrors
from flext_core._models.exception_params import FlextModelsExceptionParams
from flext_core._models.generic import FlextGenericModels
from flext_core._models.handler import FlextModelsHandler
from flext_core._models.service import FlextModelsService
from flext_core._models.settings import FlextModelsConfig
from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _t.TYPE_CHECKING:
    import flext_core._models._context as _flext_core__models__context

    _context = _flext_core__models__context
    import flext_core._models.base as _flext_core__models_base

    base = _flext_core__models_base
    import flext_core._models.collections as _flext_core__models_collections

    collections = _flext_core__models_collections
    import flext_core._models.container as _flext_core__models_container

    container = _flext_core__models_container
    import flext_core._models.containers as _flext_core__models_containers

    containers = _flext_core__models_containers
    import flext_core._models.context as _flext_core__models_context

    context = _flext_core__models_context
    import flext_core._models.cqrs as _flext_core__models_cqrs

    cqrs = _flext_core__models_cqrs
    import flext_core._models.decorators as _flext_core__models_decorators

    decorators = _flext_core__models_decorators
    import flext_core._models.dispatcher as _flext_core__models_dispatcher

    dispatcher = _flext_core__models_dispatcher
    import flext_core._models.domain_event as _flext_core__models_domain_event

    domain_event = _flext_core__models_domain_event
    import flext_core._models.entity as _flext_core__models_entity

    entity = _flext_core__models_entity
    import flext_core._models.errors as _flext_core__models_errors

    errors = _flext_core__models_errors
    import flext_core._models.exception_params as _flext_core__models_exception_params

    exception_params = _flext_core__models_exception_params
    import flext_core._models.generic as _flext_core__models_generic

    generic = _flext_core__models_generic
    import flext_core._models.handler as _flext_core__models_handler

    handler = _flext_core__models_handler
    import flext_core._models.service as _flext_core__models_service

    service = _flext_core__models_service
    import flext_core._models.settings as _flext_core__models_settings

    settings = _flext_core__models_settings

    _ = (
        FlextGenericModels,
        FlextModelFoundation,
        FlextModelsCollections,
        FlextModelsConfig,
        FlextModelsContainer,
        FlextModelsContainers,
        FlextModelsContext,
        FlextModelsContextData,
        FlextModelsContextExport,
        FlextModelsContextMetadata,
        FlextModelsContextProxyVar,
        FlextModelsContextScope,
        FlextModelsContextTokens,
        FlextModelsCqrs,
        FlextModelsDecorators,
        FlextModelsDispatcher,
        FlextModelsDomainEvent,
        FlextModelsEntity,
        FlextModelsErrors,
        FlextModelsExceptionParams,
        FlextModelsHandler,
        FlextModelsService,
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
_LAZY_IMPORTS = merge_lazy_imports(
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
    "FlextModelsErrors",
    "FlextModelsExceptionParams",
    "FlextModelsHandler",
    "FlextModelsService",
    "_context",
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
