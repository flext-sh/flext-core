# AUTO-GENERATED FILE — Regenerate with: make gen
"""Settings package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._settings.base import FlextSettingsBase as FlextSettingsBase
    from flext_core._settings.context import (
        FlextSettingsContext as FlextSettingsContext,
    )
    from flext_core._settings.core import FlextSettingsCore as FlextSettingsCore
    from flext_core._settings.database import (
        FlextSettingsDatabase as FlextSettingsDatabase,
    )
    from flext_core._settings.di import FlextSettingsDI as FlextSettingsDI
    from flext_core._settings.dispatcher import (
        FlextSettingsDispatcher as FlextSettingsDispatcher,
    )
    from flext_core._settings.infrastructure import (
        FlextSettingsInfrastructure as FlextSettingsInfrastructure,
    )
    from flext_core._settings.registry import (
        FlextSettingsRegistry as FlextSettingsRegistry,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._base_parts": ("_base_parts",),
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


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
