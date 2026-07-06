# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._base_parts.flextmodelsbase_part_03 import (
        FlextModelsBase as FlextModelsBase,
    )
    from flext_core._models._container_parts.flextmodelscontainer_part_04 import (
        FlextModelsContainer as FlextModelsContainer,
    )
    from flext_core._models._context.__scope_parts.flextmodelscontextscope_part_03 import (
        FlextModelsContextScope as FlextModelsContextScope,
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
    from flext_core._models._context._tokens import (
        FlextModelsContextTokens as FlextModelsContextTokens,
    )
    from flext_core._models._cqrs_parts.flextmodelscqrs_part_02 import (
        FlextModelsCqrs as FlextModelsCqrs,
    )
    from flext_core._models._enforcement._base import (
        EnforcementModelBase as EnforcementModelBase,
        FlextModelsEnforcementBase as FlextModelsEnforcementBase,
    )
    from flext_core._models._enforcement._catalog import (
        FlextModelsEnforcementCatalog as FlextModelsEnforcementCatalog,
    )
    from flext_core._models._enforcement._params import (
        FlextModelsEnforcementParams as FlextModelsEnforcementParams,
    )
    from flext_core._models._enforcement._sources import (
        FlextModelsEnforcementSources as FlextModelsEnforcementSources,
    )
    from flext_core._models._exception_params_parts.flextmodelsexceptionparams_part_03 import (
        FlextModelsExceptionParams as FlextModelsExceptionParams,
    )
    from flext_core._models._handler_parts.flextmodelshandler_part_02 import (
        FlextModelsHandler as FlextModelsHandler,
    )
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
        FlextModelsProjectMetadata as FlextModelsProjectMetadata,
    )
    from flext_core._models.builder import FlextModelsBuilder as FlextModelsBuilder
    from flext_core._models.collections import (
        FlextModelsCollections as FlextModelsCollections,
    )
    from flext_core._models.containers import (
        FlextModelsContainers as FlextModelsContainers,
        mc as mc,
    )
    from flext_core._models.context import FlextModelsContext as FlextModelsContext
    from flext_core._models.dispatcher import (
        FlextModelsDispatcher as FlextModelsDispatcher,
    )
    from flext_core._models.domain_event import (
        FlextModelsDomainEvent as FlextModelsDomainEvent,
    )
    from flext_core._models.enforcement import (
        FlextModelsEnforcement as FlextModelsEnforcement,
    )
    from flext_core._models.entity import FlextModelsEntity as FlextModelsEntity
    from flext_core._models.errors import FlextModelsErrors as FlextModelsErrors
    from flext_core._models.namespace import (
        FlextModelsNamespace as FlextModelsNamespace,
    )
    from flext_core._models.pydantic import FlextModelsPydantic as FlextModelsPydantic
    from flext_core._models.registry import FlextModelsRegistry as FlextModelsRegistry
    from flext_core._models.service import FlextModelsService as FlextModelsService
    from flext_core._models.settings import FlextModelsSettings as FlextModelsSettings
_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._base_parts": ("_base_parts",),
        "._base_parts.flextmodelsbase_part_03": ("FlextModelsBase",),
        "._container_parts": ("_container_parts",),
        "._container_parts.flextmodelscontainer_part_04": ("FlextModelsContainer",),
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
        "._cqrs_parts.flextmodelscqrs_part_02": ("FlextModelsCqrs",),
        "._enforcement": ("_enforcement",),
        "._enforcement._base": (
            "EnforcementModelBase",
            "FlextModelsEnforcementBase",
        ),
        "._enforcement._catalog": ("FlextModelsEnforcementCatalog",),
        "._enforcement._params": ("FlextModelsEnforcementParams",),
        "._enforcement._sources": ("FlextModelsEnforcementSources",),
        "._exception_params_parts": ("_exception_params_parts",),
        "._exception_params_parts.flextmodelsexceptionparams_part_03": (
            "FlextModelsExceptionParams",
        ),
        "._handler_parts": ("_handler_parts",),
        "._handler_parts.flextmodelshandler_part_02": ("FlextModelsHandler",),
        "._project_metadata_parts": ("_project_metadata_parts",),
        "._project_metadata_parts.flextmodelsprojectmetadata_part_04": (
            "FlextModelsProjectMetadata",
        ),
        ".builder": ("FlextModelsBuilder",),
        ".collections": ("FlextModelsCollections",),
        ".containers": (
            "FlextModelsContainers",
            "mc",
        ),
        ".context": ("FlextModelsContext",),
        ".dispatcher": ("FlextModelsDispatcher",),
        ".domain_event": ("FlextModelsDomainEvent",),
        ".enforcement": ("FlextModelsEnforcement",),
        ".entity": ("FlextModelsEntity",),
        ".errors": ("FlextModelsErrors",),
        ".namespace": ("FlextModelsNamespace",),
        ".pydantic": ("FlextModelsPydantic",),
        ".registry": ("FlextModelsRegistry",),
        ".service": ("FlextModelsService",),
        ".settings": ("FlextModelsSettings",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
