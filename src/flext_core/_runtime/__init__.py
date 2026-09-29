# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Runtime package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._base import FlextRuntimeBase
    from ._container import FlextRuntimeContainer
    from ._metadata import FlextRuntimeMetadata
    from ._metadata_validation import FlextRuntimeMetadataValidation


__all__: tuple[str, ...] = (
    "FlextRuntimeBase",
    "FlextRuntimeContainer",
    "FlextRuntimeMetadata",
    "FlextRuntimeMetadataValidation",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._base": ("FlextRuntimeBase",),
            "._container": ("FlextRuntimeContainer",),
            "._metadata": ("FlextRuntimeMetadata",),
            "._metadata_validation": ("FlextRuntimeMetadataValidation",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
