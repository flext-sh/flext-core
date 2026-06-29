# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._models._container_parts import (
        FlextModelsContainer as FlextModelsContainer,
    )
    from flext_core._models._context.__scope_parts import (
        FlextModelsContextScope as FlextModelsContextScope,
    )
    from flext_core._models._cqrs_parts import FlextModelsCqrs as FlextModelsCqrs
    from flext_core._models._exception_params_parts import (
        FlextModelsExceptionParams as FlextModelsExceptionParams,
    )
    from flext_core._models._handler_parts import (
        FlextModelsHandler as FlextModelsHandler,
    )
    from flext_core._models._project_metadata_parts import (
        FlextModelsProjectMetadata as FlextModelsProjectMetadata,
    )
    from flext_core._models.base import FlextModelsBase as FlextModelsBase
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
        "._container_parts.flextmodelscontainer_part_04": ("FlextModelsContainer",),
        "._context.__scope_parts.flextmodelscontextscope_part_03": (
            "FlextModelsContextScope",
        ),
        "._cqrs_parts.flextmodelscqrs_part_02": ("FlextModelsCqrs",),
        "._exception_params_parts.flextmodelsexceptionparams_part_03": (
            "FlextModelsExceptionParams",
        ),
        "._handler_parts.flextmodelshandler_part_02": ("FlextModelsHandler",),
        "._project_metadata_parts.flextmodelsprojectmetadata_part_04": (
            "FlextModelsProjectMetadata",
        ),
        ".base": ("FlextModelsBase",),
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
