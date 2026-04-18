# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

_LAZY_IMPORTS = merge_lazy_imports(
    ("._context",),
    build_lazy_import_map(
        {
            "._context._data": ("FlextModelsContextData",),
            "._context._export": ("FlextModelsContextExport",),
            "._context._metadata": ("FlextModelsContextMetadata",),
            "._context._proxy_var": ("FlextModelsContextProxyVar",),
            "._context._scope": ("FlextModelsContextScope",),
            "._context._tokens": ("FlextModelsContextTokens",),
            ".base": ("FlextModelsBase",),
            ".builder": ("FlextModelsBuilder",),
            ".collections": ("FlextModelsCollections",),
            ".container": ("FlextModelsContainer",),
            ".containers": ("FlextModelsContainers",),
            ".context": ("FlextModelsContext",),
            ".cqrs": ("FlextModelsCqrs",),
            ".dispatcher": ("FlextModelsDispatcher",),
            ".domain_event": ("FlextModelsDomainEvent",),
            ".enforcement": ("FlextModelsEnforcement",),
            ".entity": ("FlextModelsEntity",),
            ".errors": ("FlextModelsErrors",),
            ".exception_params": ("FlextModelsExceptionParams",),
            ".generic": ("FlextGenericModels",),
            ".handler": ("FlextModelsHandler",),
            ".namespace": ("FlextModelsNamespace",),
            ".project_metadata": ("FlextModelsProjectMetadata",),
            ".pydantic": ("FlextModelsPydantic",),
            ".registry": ("FlextModelsRegistry",),
            ".service": ("FlextModelsService",),
            ".settings": ("FlextModelsSettings",),
        },
    ),
    exclude_names=(
        "cleanup_submodule_namespace",
        "install_lazy_exports",
        "lazy_getattr",
        "logger",
        "merge_lazy_imports",
        "output",
        "output_reporting",
    ),
    module_name=__name__,
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
