# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Models package."""

from __future__ import annotations

from flext_core.lazy import install_lazy_exports, merge_lazy_imports

_LAZY_IMPORTS = merge_lazy_imports(
    ("flext_core._models._context",),
    {
        "FlextGenericModels": ("flext_core._models.generic", "FlextGenericModels"),
        "FlextModelsBase": ("flext_core._models.base", "FlextModelsBase"),
        "FlextModelsBuilder": ("flext_core._models.builder", "FlextModelsBuilder"),
        "FlextModelsCollections": (
            "flext_core._models.collections",
            "FlextModelsCollections",
        ),
        "FlextModelsConfig": ("flext_core._models.settings", "FlextModelsConfig"),
        "FlextModelsContainer": (
            "flext_core._models.container",
            "FlextModelsContainer",
        ),
        "FlextModelsContainers": (
            "flext_core._models.containers",
            "FlextModelsContainers",
        ),
        "FlextModelsContext": ("flext_core._models.context", "FlextModelsContext"),
        "FlextModelsCqrs": ("flext_core._models.cqrs", "FlextModelsCqrs"),
        "FlextModelsDecorators": (
            "flext_core._models.decorators",
            "FlextModelsDecorators",
        ),
        "FlextModelsDispatcher": (
            "flext_core._models.dispatcher",
            "FlextModelsDispatcher",
        ),
        "FlextModelsDomainEvent": (
            "flext_core._models.domain_event",
            "FlextModelsDomainEvent",
        ),
        "FlextModelsEntity": ("flext_core._models.entity", "FlextModelsEntity"),
        "FlextModelsErrors": ("flext_core._models.errors", "FlextModelsErrors"),
        "FlextModelsExceptionParams": (
            "flext_core._models.exception_params",
            "FlextModelsExceptionParams",
        ),
        "FlextModelsHandler": ("flext_core._models.handler", "FlextModelsHandler"),
        "FlextModelsNamespace": (
            "flext_core._models.namespace",
            "FlextModelsNamespace",
        ),
        "FlextModelsRegistry": ("flext_core._models.registry", "FlextModelsRegistry"),
        "FlextModelsService": ("flext_core._models.service", "FlextModelsService"),
    },
)
_ = _LAZY_IMPORTS.pop("cleanup_submodule_namespace", None)
_ = _LAZY_IMPORTS.pop("install_lazy_exports", None)
_ = _LAZY_IMPORTS.pop("lazy_getattr", None)
_ = _LAZY_IMPORTS.pop("logger", None)
_ = _LAZY_IMPORTS.pop("merge_lazy_imports", None)
_ = _LAZY_IMPORTS.pop("output", None)
_ = _LAZY_IMPORTS.pop("output_reporting", None)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
