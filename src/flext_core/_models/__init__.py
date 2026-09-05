# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _base_parts as _base_parts
    from . import _container_parts as _container_parts
    from . import _context as _context
    from . import _enforcement as _enforcement
    from . import _exception_params_parts as _exception_params_parts
    from ._context._data import FlextModelsContextData
    from ._context._export import FlextModelsContextExport
    from ._context._metadata import FlextModelsContextMetadata
    from ._context._proxy_var import FlextModelsContextProxyVar
    from ._context._scope import FlextModelsContextScope
    from ._context._tokens import FlextModelsContextTokens
    from ._enforcement._base import EnforcementModelBase, FlextModelsEnforcementBase
    from ._enforcement._catalog import FlextModelsEnforcementCatalog
    from ._enforcement._params import FlextModelsEnforcementParams
    from ._enforcement._sources import FlextModelsEnforcementSources
    from .base import FlextModelsBase
    from .builder import FlextModelsBuilder
    from .collections import FlextModelsCollections
    from .config import FlextModelsConfig
    from .container import FlextModelsContainer
    from .containers import FlextModelsContainers, mc
    from .context import FlextModelsContext
    from .cqrs import FlextModelsCqrs
    from .dispatcher import FlextModelsDispatcher
    from .domain_event import FlextModelsDomainEvent
    from .enforcement import FlextModelsEnforcement
    from .entity import FlextModelsEntity
    from .errors import FlextModelsErrors
    from .exception_params import FlextModelsExceptionParams
    from .handler import FlextModelsHandler
    from .namespace import FlextModelsNamespace
    from .project_metadata import FlextModelsProjectMetadata
    from .pydantic import FlextModelsPydantic
    from .registry import FlextModelsRegistry
    from .service import FlextModelsService
    from .settings import FlextModelsSettings
__all__: tuple[str, ...] = (
    "EnforcementModelBase",
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
    "FlextModelsDispatcher",
    "FlextModelsDomainEvent",
    "FlextModelsEnforcement",
    "FlextModelsEnforcementBase",
    "FlextModelsEnforcementCatalog",
    "FlextModelsEnforcementParams",
    "FlextModelsEnforcementSources",
    "FlextModelsEntity",
    "FlextModelsErrors",
    "FlextModelsExceptionParams",
    "FlextModelsHandler",
    "FlextModelsNamespace",
    "FlextModelsProjectMetadata",
    "FlextModelsPydantic",
    "FlextModelsRegistry",
    "FlextModelsService",
    "FlextModelsSettings",
    "_base_parts",
    "_container_parts",
    "_context",
    "_enforcement",
    "_exception_params_parts",
    "mc",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._base_parts": ("_base_parts",),
            "._container_parts": ("_container_parts",),
            "._context": ("_context",),
            "._context._data": ("FlextModelsContextData",),
            "._context._export": ("FlextModelsContextExport",),
            "._context._metadata": ("FlextModelsContextMetadata",),
            "._context._proxy_var": ("FlextModelsContextProxyVar",),
            "._context._scope": ("FlextModelsContextScope",),
            "._context._tokens": ("FlextModelsContextTokens",),
            "._enforcement": ("_enforcement",),
            "._enforcement._base": (
                "EnforcementModelBase",
                "FlextModelsEnforcementBase",
            ),
            "._enforcement._catalog": ("FlextModelsEnforcementCatalog",),
            "._enforcement._params": ("FlextModelsEnforcementParams",),
            "._enforcement._sources": ("FlextModelsEnforcementSources",),
            "._exception_params_parts": ("_exception_params_parts",),
            ".base": ("FlextModelsBase",),
            ".builder": ("FlextModelsBuilder",),
            ".collections": ("FlextModelsCollections",),
            ".config": ("FlextModelsConfig",),
            ".container": ("FlextModelsContainer",),
            ".containers": ("FlextModelsContainers", "mc"),
            ".context": ("FlextModelsContext",),
            ".cqrs": ("FlextModelsCqrs",),
            ".dispatcher": ("FlextModelsDispatcher",),
            ".domain_event": ("FlextModelsDomainEvent",),
            ".enforcement": ("FlextModelsEnforcement",),
            ".entity": ("FlextModelsEntity",),
            ".errors": ("FlextModelsErrors",),
            ".exception_params": ("FlextModelsExceptionParams",),
            ".handler": ("FlextModelsHandler",),
            ".namespace": ("FlextModelsNamespace",),
            ".project_metadata": ("FlextModelsProjectMetadata",),
            ".pydantic": ("FlextModelsPydantic",),
            ".registry": ("FlextModelsRegistry",),
            ".service": ("FlextModelsService",),
            ".settings": ("FlextModelsSettings",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
