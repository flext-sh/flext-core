# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_base": ("TestsFlextModelsBase",),
        ".test_cqrs": ("TestsFlextModelsCQRS",),
        ".test_enforcement_sources": ("TestsFlextModelsEnforcementSources",),
        ".test_entity": ("TestsFlextModelsEntity",),
        ".test_exception_params": ("test_exception_params",),
        ".test_exception_params_core": ("TestsFlextModelsExceptionParamsCore",),
        ".test_exception_params_operations": (
            "TestsFlextModelsExceptionParamsOperations",
        ),
        ".test_exception_params_resources": (
            "TestsFlextModelsExceptionParamsResources",
        ),
        "flext_tests": (
            "c",
            "d",
            "e",
            "h",
            "m",
            "p",
            "r",
            "s",
            "t",
            "td",
            "tf",
            "tk",
            "tm",
            "tv",
            "u",
            "x",
        ),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
