# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Typings package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._typings.annotateds import FlextTypesAnnotateds
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.config import FlextTypingConfig
    from flext_core._typings.core import FlextTypesCore
    from flext_core._typings.lazy import FlextTypesLazy
    from flext_core._typings.native_export import FlextTypingNativeExport
    from flext_core._typings.pydantic import FlextTypesPydantic
    from flext_core._typings.services import FlextTypesServices


__all__: tuple[str, ...] = (
    "FlextTypesAnnotateds",
    "FlextTypesCore",
    "FlextTypesLazy",
    "FlextTypesPydantic",
    "FlextTypesServices",
    "FlextTypingBase",
    "FlextTypingConfig",
    "FlextTypingNativeExport",
)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextTypesAnnotateds": ".annotateds",
        "FlextTypesCore": ".core",
        "FlextTypesLazy": ".lazy",
        "FlextTypesPydantic": ".pydantic",
        "FlextTypesServices": ".services",
        "FlextTypingBase": ".base",
        "FlextTypingConfig": ".config",
        "FlextTypingNativeExport": ".native_export",
    }),
    public_exports=__all__,
)[0]
