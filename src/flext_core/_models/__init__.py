# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._base_parts": ("_base_parts",),
        "._container_parts": ("_container_parts",),
        "._context": ("_context",),
        "._context.__scope_parts.flextmodelscontextscope_part_03": (
            "FlextModelsContextScope",
        ),
        "._cqrs_parts": ("_cqrs_parts",),
        "._cqrs_parts.flextmodelscqrs_part_02": ("FlextModelsCqrs",),
        "._enforcement": ("_enforcement",),
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
        ".base": ("FlextModelsBase",),
        ".builder": ("FlextModelsBuilder",),
        ".collections": ("FlextModelsCollections",),
        ".container": ("FlextModelsContainer",),
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
