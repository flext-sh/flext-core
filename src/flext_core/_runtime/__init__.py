# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Runtime package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._runtime._base import FlextRuntimeBase
    from flext_core._runtime._container import FlextRuntimeContainer
    from flext_core._runtime._metadata import FlextRuntimeMetadata
    from flext_core._runtime._metadata_validation import FlextRuntimeMetadataValidation


__all__: tuple[str, ...] = (
    "FlextRuntimeBase",
    "FlextRuntimeContainer",
    "FlextRuntimeMetadata",
    "FlextRuntimeMetadataValidation",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextRuntimeBase": "._base",
        "FlextRuntimeContainer": "._container",
        "FlextRuntimeMetadata": "._metadata",
        "FlextRuntimeMetadataValidation": "._metadata_validation",
    }),
    public_exports=__all__,
)
