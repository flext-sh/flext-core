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
    from flext_core._models._base.flextmodelsbase_part_01 import FlextModelsBasePart01
    from flext_core._models._base.flextmodelsbase_part_02 import FlextModelsBasePart02
    from flext_core._models._base.flextmodelsbase_part_03 import FlextModelsBasePart03
    from flext_core._models._container.flextmodelscontainer_part_01 import (
        FlextModelsContainerPart01,
    )
    from flext_core._models._container.flextmodelscontainer_part_02 import (
        FlextModelsContainerPart02,
    )
    from flext_core._models._container.flextmodelscontainer_part_03 import (
        FlextModelsContainerPart03,
    )
    from flext_core._models._container.flextmodelscontainer_part_04 import (
        FlextModelsContainerPart04,
    )
    from flext_core._models._context.__scope.flextmodelscontextscope_part_01 import (
        FlextModelsContextScopePart01,
    )
    from flext_core._models._context.__scope.flextmodelscontextscope_part_02 import (
        FlextModelsContextScopePart02,
    )
    from flext_core._models._context.__scope.flextmodelscontextscope_part_03 import (
        FlextModelsContextScopePart03,
    )
    from flext_core._models._context._data import FlextModelsContextData
    from flext_core._models._context._export import FlextModelsContextExport
    from flext_core._models._context._metadata import FlextModelsContextMetadata
    from flext_core._models._context._scope import FlextModelsContextScope
    from flext_core._models._context._tokens import FlextModelsContextTokens
    from flext_core._models._cqrs.flextmodelscqrs_part_01 import FlextModelsCqrsPart01
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
    from flext_core._models._exception_params.base import FlextModelsExceptionParamsBase
    from flext_core._models._exception_params.operation_errors import (
        FlextModelsExceptionParamsOperationErrors,
    )
    from flext_core._models._exception_params.service_errors import (
        FlextModelsExceptionParamsServiceErrors,
    )
    from flext_core._models._project_metadata.aggregates import (
        FlextModelsProjectMetadataAggregates,
    )
    from flext_core._models._project_metadata.contract import (
        FlextModelsProjectMetadataContract,
    )
    from flext_core._models._project_metadata.document import (
        FlextModelsProjectMetadataDocument,
    )
    from flext_core._models._project_metadata.pyproject_ingress import (
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
    "FlextModelsExceptionParamsBase",
    "FlextModelsExceptionParamsOperationErrors",
    "FlextModelsExceptionParamsServiceErrors",
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
        "FlextModelsExceptionParamsBase": "._exception_params.base",
        "FlextModelsExceptionParamsOperationErrors": (
            "._exception_params.operation_errors"
        ),
        "FlextModelsExceptionParamsServiceErrors": "._exception_params.service_errors",
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
    }),
    public_exports=__all__,
)
