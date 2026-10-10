# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models import (
        _base_parts,
        _container_parts,
        _context,
        _cqrs_parts,
        _enforcement,
        _exception_params_parts,
        _project_metadata_parts,
    )
    from flext_core._models._context.__scope_parts.flextmodelscontextscope_part_03 import (
        FlextModelsContextScope,
    )
    from flext_core._models._context._data import FlextModelsContextData
    from flext_core._models._context._export import FlextModelsContextExport
    from flext_core._models._context._metadata import FlextModelsContextMetadata
    from flext_core._models._context._proxy_var import FlextModelsContextProxyVar
    from flext_core._models._context._scope_ops import FlextContextScopeOps
    from flext_core._models._context._tokens import FlextModelsContextTokens
    from flext_core._models._enforcement._base import (
        FlextModelsEnforcementBase,
        FlextModelsEnforcementModelBase,
    )
    from flext_core._models._enforcement._catalog import FlextModelsEnforcementCatalog
    from flext_core._models._enforcement._inspection import (
        FlextModelsEnforcementInspection,
    )
    from flext_core._models._enforcement._params import FlextModelsEnforcementParams
    from flext_core._models._enforcement._resolution import (
        FlextModelsEnforcementResolution,
    )
    from flext_core._models._enforcement._sources import FlextModelsEnforcementSources
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_03 import (
        FlextModelsProjectMetadataFields,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
        FlextModelsProjectMetadataAggregates,
        FlextModelsProjectMetadataContract,
        FlextModelsProjectMetadataDocument,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_05 import (
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
    from flext_core._models.flext_context import FlextContext
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
    "FlextContext",
    "FlextContextScopeOps",
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
    "FlextModelsContextProxyVar",
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
    "FlextModelsProjectMetadataFields",
    "FlextModelsPydantic",
    "FlextModelsPyprojectIngressContract",
    "FlextModelsRegistry",
    "FlextModelsService",
    "FlextModelsSettings",
    "_base_parts",
    "_container_parts",
    "_context",
    "_cqrs_parts",
    "_enforcement",
    "_exception_params_parts",
    "_project_metadata_parts",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextContext": ".flext_context",
        "FlextContextScopeOps": "._context._scope_ops",
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
        "FlextModelsContextProxyVar": "._context._proxy_var",
        "FlextModelsContextScope": (
            "._context.__scope_parts.flextmodelscontextscope_part_03"
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
        "FlextModelsProjectMetadataAggregates": (
            "._project_metadata_parts.flextmodelsprojectmetadata_part_04"
        ),
        "FlextModelsProjectMetadataContract": (
            "._project_metadata_parts.flextmodelsprojectmetadata_part_04"
        ),
        "FlextModelsProjectMetadataDocument": (
            "._project_metadata_parts.flextmodelsprojectmetadata_part_04"
        ),
        "FlextModelsProjectMetadataFields": (
            "._project_metadata_parts.flextmodelsprojectmetadata_part_03"
        ),
        "FlextModelsPydantic": ".pydantic",
        "FlextModelsPyprojectIngressContract": (
            "._project_metadata_parts.flextmodelsprojectmetadata_part_05"
        ),
        "FlextModelsRegistry": ".registry",
        "FlextModelsService": ".service",
        "FlextModelsSettings": ".settings",
        "_base_parts": "._base_parts",
        "_container_parts": "._container_parts",
        "_context": "._context",
        "_cqrs_parts": "._cqrs_parts",
        "_enforcement": "._enforcement",
        "_exception_params_parts": "._exception_params_parts",
        "_project_metadata_parts": "._project_metadata_parts",
    }),
    public_exports=__all__,
)
