# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Typings package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._typings import _base_parts
    from flext_core._typings.annotateds import FlextTypesAnnotateds
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.config import FlextTypingConfig
    from flext_core._typings.containers import FlextTypingContainers
    from flext_core._typings.core import FlextTypesCore
    from flext_core._typings.lazy import FlextTypesLazy
    from flext_core._typings.project_metadata import FlextTypingProjectMetadata
    from flext_core._typings.pydantic import FlextTypesPydantic
    from flext_core._typings.services import FlextTypesServices
    from flext_core._typings.typeadapters import FlextTypesTypeAdapters


__all__: tuple[str, ...] = (
    "FlextTypesAnnotateds",
    "FlextTypesCore",
    "FlextTypesLazy",
    "FlextTypesPydantic",
    "FlextTypesServices",
    "FlextTypesTypeAdapters",
    "FlextTypingBase",
    "FlextTypingConfig",
    "FlextTypingContainers",
    "FlextTypingProjectMetadata",
    "_base_parts",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextTypesAnnotateds": ".annotateds",
        "FlextTypesCore": ".core",
        "FlextTypesLazy": ".lazy",
        "FlextTypesPydantic": ".pydantic",
        "FlextTypesServices": ".services",
        "FlextTypesTypeAdapters": ".typeadapters",
        "FlextTypingBase": ".base",
        "FlextTypingConfig": ".config",
        "FlextTypingContainers": ".containers",
        "FlextTypingProjectMetadata": ".project_metadata",
        "_base_parts": "._base_parts",
    }),
    public_exports=__all__,
)
