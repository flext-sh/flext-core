# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models import (
        _base as _base,
        _container as _container,
        _context as _context,
        _cqrs as _cqrs,
        _enforcement as _enforcement,
        _exception_params as _exception_params,
        _project_metadata as _project_metadata,
    )
    from flext_core._models._context import (
        FlextModelsContextData,
        FlextModelsContextExport,
        FlextModelsContextMetadata,
        FlextModelsContextScope,
        FlextModelsContextTokens,
    )
    from flext_core._models._enforcement import (
        FlextModelsEnforcementBase,
        FlextModelsEnforcementCatalog,
        FlextModelsEnforcementInspection,
        FlextModelsEnforcementModelBase,
        FlextModelsEnforcementParams,
        FlextModelsEnforcementResolution,
        FlextModelsEnforcementSources,
    )
    from flext_core._models._project_metadata import (
        FlextModelsProjectMetadataAggregates,
        FlextModelsProjectMetadataContract,
        FlextModelsProjectMetadataDocument,
        FlextModelsPyprojectIngressContract,
    )
    from flext_core._models.base import FlextModelsBase
    from flext_core._models.builder import FlextModelsBuilder
    from flext_core._models.collection_models import FlextModelsCollections
    from flext_core._models.config import FlextModelsConfig
    from flext_core._models.container import FlextModelsContainer
    from flext_core._models.containers import FlextModelsContainers
    from flext_core._models.context import FlextModelsContext
    from flext_core._models.cqrs import FlextModelsCqrs
    from flext_core._models.domain_event import FlextModelsDomainEvent
    from flext_core._models.enforcement import FlextModelsEnforcement
    from flext_core._models.entity import FlextModelsEntity
    from flext_core._models.errors import FlextModelsErrors
    from flext_core._models.exception_params import FlextModelsExceptionParams
    from flext_core._models.flext_context import FlextContextState
    from flext_core._models.flext_mixins import FlextMixins
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.namespace import FlextModelsNamespace
    from flext_core._models.options import FlextModelsOptions
    from flext_core._models.project_metadata import FlextModelsProjectMetadata
    from flext_core._models.pydantic import FlextModelsPydantic
    from flext_core._models.registry import FlextModelsRegistry
    from flext_core._models.service import FlextModelsService
    from flext_core._models.settings import FlextModelsSettings


__all__: tuple[str, ...] = (
    "FlextContextState",
    "FlextMixins",
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
    "FlextModelsContextScope",
    "FlextModelsContextTokens",
    "FlextModelsCqrs",
    "FlextModelsDomainEvent",
    "FlextModelsEnforcement",
    "FlextModelsEnforcementBase",
    "FlextModelsEnforcementCatalog",
    "FlextModelsEnforcementInspection",
    "FlextModelsEnforcementModelBase",
    "FlextModelsEnforcementParams",
    "FlextModelsEnforcementResolution",
    "FlextModelsEnforcementSources",
    "FlextModelsEntity",
    "FlextModelsErrors",
    "FlextModelsExceptionParams",
    "FlextModelsHandler",
    "FlextModelsNamespace",
    "FlextModelsOptions",
    "FlextModelsProjectMetadata",
    "FlextModelsProjectMetadataAggregates",
    "FlextModelsProjectMetadataContract",
    "FlextModelsProjectMetadataDocument",
    "FlextModelsPydantic",
    "FlextModelsPyprojectIngressContract",
    "FlextModelsRegistry",
    "FlextModelsService",
    "FlextModelsSettings",
    "_base",
    "_container",
    "_context",
    "_cqrs",
    "_enforcement",
    "_exception_params",
    "_project_metadata",
)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextContextState": ".flext_context",
        "FlextMixins": ".flext_mixins",
        "FlextModelsBase": ".base",
        "FlextModelsBuilder": ".builder",
        "FlextModelsCollections": ".collection_models",
        "FlextModelsConfig": ".config",
        "FlextModelsContainer": ".container",
        "FlextModelsContainers": ".containers",
        "FlextModelsContext": ".context",
        "FlextModelsContextData": "._context._data",
        "FlextModelsContextExport": "._context._export",
        "FlextModelsContextMetadata": "._context._metadata",
        "FlextModelsContextScope": (
            "._context.__scope.flextmodelscontextscope_part_03"
        ),
        "FlextModelsContextTokens": "._context._tokens",
        "FlextModelsCqrs": ".cqrs",
        "FlextModelsDomainEvent": ".domain_event",
        "FlextModelsEnforcement": ".enforcement",
        "FlextModelsEnforcementBase": "._enforcement._base",
        "FlextModelsEnforcementCatalog": "._enforcement._catalog",
        "FlextModelsEnforcementInspection": "._enforcement._inspection",
        "FlextModelsEnforcementModelBase": "._enforcement._base",
        "FlextModelsEnforcementParams": "._enforcement._params",
        "FlextModelsEnforcementResolution": "._enforcement._resolution",
        "FlextModelsEnforcementSources": "._enforcement._sources",
        "FlextModelsEntity": ".entity",
        "FlextModelsErrors": ".errors",
        "FlextModelsExceptionParams": ".exception_params",
        "FlextModelsHandler": ".handler",
        "FlextModelsNamespace": ".namespace",
        "FlextModelsOptions": ".options",
        "FlextModelsProjectMetadata": ".project_metadata",
        "FlextModelsProjectMetadataAggregates": ("._project_metadata.aggregates"),
        "FlextModelsProjectMetadataContract": ("._project_metadata.contract"),
        "FlextModelsProjectMetadataDocument": ("._project_metadata.document"),
        "FlextModelsPydantic": ".pydantic",
        "FlextModelsPyprojectIngressContract": ("._project_metadata.pyproject_ingress"),
        "FlextModelsRegistry": ".registry",
        "FlextModelsService": ".service",
        "FlextModelsSettings": ".settings",
        "_base": "._base",
        "_container": "._container",
        "_context": "._context",
        "_cqrs": "._cqrs",
        "_enforcement": "._enforcement",
        "_exception_params": "._exception_params",
        "_project_metadata": "._project_metadata",
    }),
    public_exports=__all__,
)[0]
