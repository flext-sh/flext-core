# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_base": ("TestsFlextCoreModelsBase",),
        ".test_cqrs": ("TestsFlextCoreModelsCQRS",),
        ".test_enforcement_sources": ("TestsFlextCoreModelsEnforcementSources",),
        ".test_entity": ("TestsFlextCoreModelsEntity",),
        ".test_exception_params": ("TestsFlextCoreModelsExceptionParams",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
