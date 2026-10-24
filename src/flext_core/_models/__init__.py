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
        _base as _base,
        _container as _container,
        _context as _context,
        _cqrs as _cqrs,
        _enforcement as _enforcement,
        _exception_params as _exception_params,
        _project_metadata as _project_metadata,
    )
    from flext_core._models._base import (
        FlextModelsBasePart01,
        FlextModelsBasePart02,
        FlextModelsBasePart03,
    )
    from flext_core._models._container import (
        FlextModelsContainerPart01,
        FlextModelsContainerPart02,
        FlextModelsContainerPart03,
        FlextModelsContainerPart04,
    )
    from flext_core._models._context import (
        FlextModelsContextData,
        FlextModelsContextExport,
        FlextModelsContextMetadata,
        FlextModelsContextScope,
        FlextModelsContextScopePart01,
        FlextModelsContextScopePart02,
        FlextModelsContextScopePart03,
        FlextModelsContextTokens,
    )
    from flext_core._models._cqrs import FlextModelsCqrsPart01
    from flext_core._models._enforcement import (
        FlextModelsEnforcementBase,
        FlextModelsEnforcementCatalog,
        FlextModelsEnforcementInspection,
        FlextModelsEnforcementModelBase,
        FlextModelsEnforcementParams,
        FlextModelsEnforcementResolution,
        FlextModelsEnforcementSources,
    )
    from flext_core._models._exception_params import (
        FlextModelsExceptionParamsPart01,
        FlextModelsExceptionParamsPart02,
        FlextModelsExceptionParamsPart03,
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
    from flext_core._models.flext_context import FlextModelsContextState
    from flext_core._models.flext_mixins import FlextModelsServiceState
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.namespace import FlextModelsNamespace
    from flext_core._models.options import FlextModelsOptions
    from flext_core._models.project_metadata import FlextModelsProjectMetadata
    from flext_core._models.pydantic import FlextModelsPydantic
    from flext_core._models.registry import FlextModelsRegistry
    from flext_core._models.service import FlextModelsService
    from flext_core._models.settings import FlextModelsSettings


__all__: tuple[str, ...] = (
    "FlextModelsBase",
    "FlextModelsBasePart01",
    "FlextModelsBasePart02",
    "FlextModelsBasePart03",
    "FlextModelsBuilder",
    "FlextModelsCollections",
    "FlextModelsConfig",
    "FlextModelsContainer",
    "FlextModelsContainerPart01",
    "FlextModelsContainerPart02",
    "FlextModelsContainerPart03",
    "FlextModelsContainerPart04",
    "FlextModelsContainers",
    "FlextModelsContext",
    "FlextModelsContextData",
    "FlextModelsContextExport",
    "FlextModelsContextMetadata",
    "FlextModelsContextScope",
    "FlextModelsContextScopePart01",
    "FlextModelsContextScopePart02",
    "FlextModelsContextScopePart03",
    "FlextModelsContextState",
    "FlextModelsContextTokens",
    "FlextModelsCqrs",
    "FlextModelsCqrsPart01",
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
    "FlextModelsExceptionParamsPart01",
    "FlextModelsExceptionParamsPart02",
    "FlextModelsExceptionParamsPart03",
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
    "FlextModelsServiceState",
    "FlextModelsSettings",
    "_base",
    "_container",
    "_context",
    "_cqrs",
    "_enforcement",
    "_exception_params",
    "_project_metadata",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsBase": ".base",
        "FlextModelsBasePart01": "._base.flextmodelsbase_part_01",
        "FlextModelsBasePart02": "._base.flextmodelsbase_part_02",
        "FlextModelsBasePart03": "._base.flextmodelsbase_part_03",
        "FlextModelsBuilder": ".builder",
        "FlextModelsCollections": ".collection_models",
        "FlextModelsConfig": ".config",
        "FlextModelsContainer": ".container",
        "FlextModelsContainerPart01": "._container.flextmodelscontainer_part_01",
        "FlextModelsContainerPart02": "._container.flextmodelscontainer_part_02",
        "FlextModelsContainerPart03": "._container.flextmodelscontainer_part_03",
        "FlextModelsContainerPart04": "._container.flextmodelscontainer_part_04",
        "FlextModelsContainers": ".containers",
        "FlextModelsContext": ".context",
        "FlextModelsContextData": "._context._data",
        "FlextModelsContextExport": "._context._export",
        "FlextModelsContextMetadata": "._context._metadata",
        "FlextModelsContextScope": "._context._scope",
        "FlextModelsContextScopePart01": (
            "._context.__scope.flextmodelscontextscope_part_01"
        ),
        "FlextModelsContextScopePart02": (
            "._context.__scope.flextmodelscontextscope_part_02"
        ),
        "FlextModelsContextScopePart03": (
            "._context.__scope.flextmodelscontextscope_part_03"
        ),
        "FlextModelsContextState": ".flext_context",
        "FlextModelsContextTokens": "._context._tokens",
        "FlextModelsCqrs": ".cqrs",
        "FlextModelsCqrsPart01": "._cqrs.flextmodelscqrs_part_01",
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
        "FlextModelsExceptionParamsPart01": (
            "._exception_params.flextmodelsexceptionparams_part_01"
        ),
        "FlextModelsExceptionParamsPart02": (
            "._exception_params.flextmodelsexceptionparams_part_02"
        ),
        "FlextModelsExceptionParamsPart03": (
            "._exception_params.flextmodelsexceptionparams_part_03"
        ),
        "FlextModelsHandler": ".handler",
        "FlextModelsNamespace": ".namespace",
        "FlextModelsOptions": ".options",
        "FlextModelsProjectMetadata": ".project_metadata",
        "FlextModelsProjectMetadataAggregates": "._project_metadata.aggregates",
        "FlextModelsProjectMetadataContract": "._project_metadata.contract",
        "FlextModelsProjectMetadataDocument": "._project_metadata.document",
        "FlextModelsPydantic": ".pydantic",
        "FlextModelsPyprojectIngressContract": "._project_metadata.pyproject_ingress",
        "FlextModelsRegistry": ".registry",
        "FlextModelsService": ".service",
        "FlextModelsServiceState": ".flext_mixins",
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
)
