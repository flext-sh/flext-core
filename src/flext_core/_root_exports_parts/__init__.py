# AUTO-GENERATED FILE — Regenerate with: make gen
"""Root Exports Parts package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".lazy_core": ("ROOT_LAZY_CORE",),
        ".lazy_facades": ("ROOT_LAZY_FACADES",),
        ".lazy_utilities": ("ROOT_LAZY_UTILITIES",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
