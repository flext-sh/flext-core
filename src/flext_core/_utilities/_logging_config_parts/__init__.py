# AUTO-GENERATED FILE — Regenerate with: make gen
"""Logging Config Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._logging_config_parts.logging_config_part_03 import (
        FlextUtilitiesLoggingConfig as FlextUtilitiesLoggingConfig,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".logging_config_part_03": ("FlextUtilitiesLoggingConfig",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
