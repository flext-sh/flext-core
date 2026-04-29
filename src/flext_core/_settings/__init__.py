# AUTO-GENERATED FILE — Regenerate with: make gen
"""Settings package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".base": ("FlextSettingsBase",),
        ".context": ("FlextSettingsContext",),
        ".core": ("FlextSettingsCore",),
        ".database": ("FlextSettingsDatabase",),
        ".di": ("FlextSettingsDI",),
        ".dispatcher": ("FlextSettingsDispatcher",),
        ".infrastructure": ("FlextSettingsInfrastructure",),
        ".registry": ("FlextSettingsRegistry",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
