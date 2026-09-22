# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import (
        _base_parts,
        _container_parts,
        _context,
        _cqrs_parts,
        _enforcement,
        _exception_params_parts,
        _project_metadata_parts,
    )
    from ._context._data import FlextModelsContextData
    from ._context._export import FlextModelsContextExport
    from ._context._metadata import FlextModelsContextMetadata
    from ._context._proxy_var import FlextModelsContextProxyVar
    from ._context._scope import FlextModelsContextScope
    from ._context._tokens import FlextModelsContextTokens
    from ._cqrs_parts.flextmodelscqrs_part_01 import CqrsPagination
    from ._enforcement._base import EnforcementModelBase, FlextModelsEnforcementBase
    from ._enforcement._catalog import FlextModelsEnforcementCatalog
    from ._enforcement._params import FlextModelsEnforcementParams
    from ._enforcement._sources import FlextModelsEnforcementSources
    from ._project_metadata_parts.flextmodelsprojectmetadata_part_01 import (
        ProjectMetadataContract,
        PyprojectIngressContract,
    )
    from ._project_metadata_parts.flextmodelsprojectmetadata_part_02 import (
        ProjectMetadataFields,
    )
    from ._project_metadata_parts.flextmodelsprojectmetadata_part_03 import (
        ProjectMetadataAggregates,
    )
    from ._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
        ProjectMetadataDocument,
    )
    from .base import FlextModelsBase
    from .builder import FlextModelsBuilder
    from .collection_models import FlextModelsCollections
    from .config import FlextModelsConfig
    from .container import FlextModelsContainer
    from .containers import FlextModelsContainers
    from .context import FlextModelsContext
    from .cqrs import FlextModelsCqrs
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
    "CqrsPagination",
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
            "._context._data": ("FlextModelsContextData",),
            "._context._export": ("FlextModelsContextExport",),
            "._context._metadata": ("FlextModelsContextMetadata",),
            "._context._proxy_var": ("FlextModelsContextProxyVar",),
            "._context._scope": ("FlextModelsContextScope",),
            "._context._tokens": ("FlextModelsContextTokens",),
            "._cqrs_parts": ("_cqrs_parts",),
            "._cqrs_parts.flextmodelscqrs_part_01": ("CqrsPagination",),
            "._enforcement": ("_enforcement",),
            "._enforcement._base": (
                "EnforcementModelBase",
                "FlextModelsEnforcementBase",
            ),
            "._enforcement._catalog": ("FlextModelsEnforcementCatalog",),
            "._enforcement._params": ("FlextModelsEnforcementParams",),
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
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
