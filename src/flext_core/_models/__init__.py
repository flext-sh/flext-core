# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

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
    from flext_core._models._context._tokens import FlextModelsContextTokens
    from flext_core._models._enforcement._base import (
        EnforcementModelBase,
        FlextModelsEnforcementBase,
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
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_01 import (
        ProjectMetadataContract,
        PyprojectIngressContract,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_02 import (
        ProjectMetadataFields,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_03 import (
        ProjectMetadataAggregates,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
        ProjectMetadataDocument,
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
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.namespace import FlextModelsNamespace
    from flext_core._models.project_metadata import FlextModelsProjectMetadata
    from flext_core._models.pydantic import FlextModelsPydantic
    from flext_core._models.registry import FlextModelsRegistry
    from flext_core._models.service import FlextModelsService
    from flext_core._models.settings import FlextModelsSettings


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
    "FlextModelsDomainEvent",
    "FlextModelsEnforcement",
    "FlextModelsEnforcementBase",
    "FlextModelsEnforcementCatalog",
    "FlextModelsEnforcementInspection",
    "FlextModelsEnforcementParams",
    "FlextModelsEnforcementResolution",
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
    "ProjectMetadataAggregates",
    "ProjectMetadataContract",
    "ProjectMetadataDocument",
    "ProjectMetadataFields",
    "PyprojectIngressContract",
    "_base_parts",
    "_container_parts",
    "_context",
    "_cqrs_parts",
    "_enforcement",
    "_exception_params_parts",
    "_project_metadata_parts",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._base_parts": ("_base_parts",),
            "._container_parts": ("_container_parts",),
            "._context": ("_context",),
            "._context.__scope_parts.flextmodelscontextscope_part_03": (
                "FlextModelsContextScope",
            ),
            "._context._data": ("FlextModelsContextData",),
            "._context._export": ("FlextModelsContextExport",),
            "._context._metadata": ("FlextModelsContextMetadata",),
            "._context._proxy_var": ("FlextModelsContextProxyVar",),
            "._context._tokens": ("FlextModelsContextTokens",),
            "._cqrs_parts": ("_cqrs_parts",),
            "._enforcement": ("_enforcement",),
            "._enforcement._base": (
                "EnforcementModelBase",
                "FlextModelsEnforcementBase",
            ),
            "._enforcement._catalog": ("FlextModelsEnforcementCatalog",),
            "._enforcement._inspection": ("FlextModelsEnforcementInspection",),
            "._enforcement._params": ("FlextModelsEnforcementParams",),
            "._enforcement._resolution": ("FlextModelsEnforcementResolution",),
            "._enforcement._sources": ("FlextModelsEnforcementSources",),
            "._exception_params_parts": ("_exception_params_parts",),
            "._project_metadata_parts": ("_project_metadata_parts",),
            "._project_metadata_parts.flextmodelsprojectmetadata_part_01": (
                "ProjectMetadataContract",
                "PyprojectIngressContract",
            ),
            "._project_metadata_parts.flextmodelsprojectmetadata_part_02": (
                "ProjectMetadataFields",
            ),
            "._project_metadata_parts.flextmodelsprojectmetadata_part_03": (
                "ProjectMetadataAggregates",
            ),
            "._project_metadata_parts.flextmodelsprojectmetadata_part_04": (
                "ProjectMetadataDocument",
            ),
            ".base": ("FlextModelsBase",),
            ".builder": ("FlextModelsBuilder",),
            ".collection_models": ("FlextModelsCollections",),
            ".config": ("FlextModelsConfig",),
            ".container": ("FlextModelsContainer",),
            ".containers": ("FlextModelsContainers",),
            ".context": ("FlextModelsContext",),
            ".cqrs": ("FlextModelsCqrs",),
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
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
