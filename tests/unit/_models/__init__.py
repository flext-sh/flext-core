# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_base": ("TestModelsBase",),
        ".test_cqrs": ("TestModelsCQRS",),
        ".test_entity": ("TestModelsEntity",),
        ".test_exception_params": ("TestFlextModelsExceptionParams",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
