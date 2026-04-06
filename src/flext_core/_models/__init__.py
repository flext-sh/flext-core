# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Models package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _t.TYPE_CHECKING:
    import flext_core._models._context as _flext_core__models__context

    _context = _flext_core__models__context
    import flext_core._models.base as _flext_core__models_base
    from flext_core._models._context import (
        FlextModelsContextData,
        FlextModelsContextExport,
        FlextModelsContextMetadata,
        FlextModelsContextProxyVar,
        FlextModelsContextScope,
        FlextModelsContextTokens,
    )

    base = _flext_core__models_base
    import flext_core._models.builder as _flext_core__models_builder
    from flext_core._models.base import FlextModelsBase

    builder = _flext_core__models_builder
    import flext_core._models.collections as _flext_core__models_collections
    from flext_core._models.builder import FlextModelsBuilder

    collections = _flext_core__models_collections
    import flext_core._models.container as _flext_core__models_container
    from flext_core._models.collections import FlextModelsCollections

    container = _flext_core__models_container
    import flext_core._models.containers as _flext_core__models_containers
    from flext_core._models.container import FlextModelsContainer

    containers = _flext_core__models_containers
    import flext_core._models.context as _flext_core__models_context
    from flext_core._models.containers import FlextModelsContainers

    context = _flext_core__models_context
    import flext_core._models.cqrs as _flext_core__models_cqrs
    from flext_core._models.context import FlextModelsContext

    cqrs = _flext_core__models_cqrs
    import flext_core._models.decorators as _flext_core__models_decorators
    from flext_core._models.cqrs import FlextModelsCqrs

    decorators = _flext_core__models_decorators
    import flext_core._models.dispatcher as _flext_core__models_dispatcher
    from flext_core._models.decorators import FlextModelsDecorators

    dispatcher = _flext_core__models_dispatcher
    import flext_core._models.domain_event as _flext_core__models_domain_event
    from flext_core._models.dispatcher import FlextModelsDispatcher

    domain_event = _flext_core__models_domain_event
    import flext_core._models.entity as _flext_core__models_entity
    from flext_core._models.domain_event import FlextModelsDomainEvent

    entity = _flext_core__models_entity
    import flext_core._models.errors as _flext_core__models_errors
    from flext_core._models.entity import FlextModelsEntity

    errors = _flext_core__models_errors
    import flext_core._models.exception_params as _flext_core__models_exception_params
    from flext_core._models.errors import FlextModelsErrors

    exception_params = _flext_core__models_exception_params
    import flext_core._models.generic as _flext_core__models_generic
    from flext_core._models.exception_params import FlextModelsExceptionParams

    generic = _flext_core__models_generic
    import flext_core._models.handler as _flext_core__models_handler
    from flext_core._models.generic import FlextGenericModels

    handler = _flext_core__models_handler
    import flext_core._models.namespace as _flext_core__models_namespace
    from flext_core._models.handler import FlextModelsHandler

    namespace = _flext_core__models_namespace
    import flext_core._models.registry as _flext_core__models_registry
    from flext_core._models.namespace import FlextModelsNamespace

    registry = _flext_core__models_registry
    import flext_core._models.service as _flext_core__models_service
    from flext_core._models.registry import FlextModelsRegistry

    service = _flext_core__models_service
    import flext_core._models.settings as _flext_core__models_settings
    from flext_core._models.service import FlextModelsService

    settings = _flext_core__models_settings
    from flext_core._models.settings import FlextModelsConfig
_LAZY_IMPORTS = merge_lazy_imports(
    ("flext_core._models._context",),
    {
        "FlextGenericModels": ("flext_core._models.generic", "FlextGenericModels"),
        "FlextModelsBase": ("flext_core._models.base", "FlextModelsBase"),
        "FlextModelsBuilder": ("flext_core._models.builder", "FlextModelsBuilder"),
        "FlextModelsCollections": (
            "flext_core._models.collections",
            "FlextModelsCollections",
        ),
        "FlextModelsConfig": ("flext_core._models.settings", "FlextModelsConfig"),
        "FlextModelsContainer": (
            "flext_core._models.container",
            "FlextModelsContainer",
        ),
        "FlextModelsContainers": (
            "flext_core._models.containers",
            "FlextModelsContainers",
        ),
        "FlextModelsContext": ("flext_core._models.context", "FlextModelsContext"),
        "FlextModelsCqrs": ("flext_core._models.cqrs", "FlextModelsCqrs"),
        "FlextModelsDecorators": (
            "flext_core._models.decorators",
            "FlextModelsDecorators",
        ),
        "FlextModelsDispatcher": (
            "flext_core._models.dispatcher",
            "FlextModelsDispatcher",
        ),
        "FlextModelsDomainEvent": (
            "flext_core._models.domain_event",
            "FlextModelsDomainEvent",
        ),
        "FlextModelsEntity": ("flext_core._models.entity", "FlextModelsEntity"),
        "FlextModelsErrors": ("flext_core._models.errors", "FlextModelsErrors"),
        "FlextModelsExceptionParams": (
            "flext_core._models.exception_params",
            "FlextModelsExceptionParams",
        ),
        "FlextModelsHandler": ("flext_core._models.handler", "FlextModelsHandler"),
        "FlextModelsNamespace": (
            "flext_core._models.namespace",
            "FlextModelsNamespace",
        ),
        "FlextModelsRegistry": ("flext_core._models.registry", "FlextModelsRegistry"),
        "FlextModelsService": ("flext_core._models.service", "FlextModelsService"),
        "_context": "flext_core._models._context",
        "base": "flext_core._models.base",
        "builder": "flext_core._models.builder",
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
        "namespace": "flext_core._models.namespace",
        "registry": "flext_core._models.registry",
        "service": "flext_core._models.service",
        "settings": "flext_core._models.settings",
    },
)
_ = _LAZY_IMPORTS.pop("cleanup_submodule_namespace", None)
_ = _LAZY_IMPORTS.pop("install_lazy_exports", None)
_ = _LAZY_IMPORTS.pop("lazy_getattr", None)
_ = _LAZY_IMPORTS.pop("merge_lazy_imports", None)
_ = _LAZY_IMPORTS.pop("output", None)
_ = _LAZY_IMPORTS.pop("output_reporting", None)

__all__ = [
    "FlextGenericModels",
    "FlextModelsBase",
    "FlextModelsBuilder",
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
    "FlextModelsNamespace",
    "FlextModelsRegistry",
    "FlextModelsService",
    "_context",
    "base",
    "builder",
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
    "namespace",
    "registry",
    "service",
    "settings",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
