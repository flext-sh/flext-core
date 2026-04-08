# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import install_lazy_exports, merge_lazy_imports

_LAZY_IMPORTS = merge_lazy_imports(
    ("._context",),
    {
        "FlextGenericModels": ".generic",
        "FlextModelsBase": ".base",
        "FlextModelsBuilder": ".builder",
        "FlextModelsCollections": ".collections",
        "FlextModelsConfig": ".settings",
        "FlextModelsContainer": ".container",
        "FlextModelsContainers": ".containers",
        "FlextModelsContext": ".context",
        "FlextModelsCqrs": ".cqrs",
        "FlextModelsDecorators": ".decorators",
        "FlextModelsDispatcher": ".dispatcher",
        "FlextModelsDomainEvent": ".domain_event",
        "FlextModelsEntity": ".entity",
        "FlextModelsErrors": ".errors",
        "FlextModelsExceptionParams": ".exception_params",
        "FlextModelsHandler": ".handler",
        "FlextModelsNamespace": ".namespace",
        "FlextModelsRegistry": ".registry",
        "FlextModelsService": ".service",
    },
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
