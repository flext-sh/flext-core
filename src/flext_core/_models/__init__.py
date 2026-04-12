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
            ".base": ("FlextModelsBase",),
            ".builder": ("FlextModelsBuilder",),
            ".collections": ("FlextModelsCollections",),
            ".container": ("FlextModelsContainer",),
            ".containers": ("FlextModelsContainers",),
            ".context": ("FlextModelsContext",),
            ".cqrs": ("FlextModelsCqrs",),
            ".decorators": ("FlextModelsDecorators",),
            ".dispatcher": ("FlextModelsDispatcher",),
            ".domain_event": ("FlextModelsDomainEvent",),
            ".entity": ("FlextModelsEntity",),
            ".errors": ("FlextModelsErrors",),
            ".exception_params": ("FlextModelsExceptionParams",),
            ".generic": ("FlextGenericModels",),
            ".handler": ("FlextModelsHandler",),
            ".namespace": ("FlextModelsNamespace",),
            ".pydantic": ("FlextModelsPydantic",),
            ".registry": ("FlextModelsRegistry",),
            ".service": ("FlextModelsService",),
            ".settings": ("FlextModelsSettings",),
        },
    ),
    exclude_names=(
        "FlextDispatcher",
        "FlextLogger",
        "FlextRegistry",
        "FlextRuntime",
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
