# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Context package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._context import __scope_parts as __scope_parts
    from flext_core._models._context.__scope_parts import FlextModelsContextScope
    from flext_core._models._context._data import FlextModelsContextData
    from flext_core._models._context._export import FlextModelsContextExport
    from flext_core._models._context._metadata import FlextModelsContextMetadata
    from flext_core._models._context._tokens import FlextModelsContextTokens


__all__: tuple[str, ...] = (
    "FlextModelsContextData",
    "FlextModelsContextExport",
    "FlextModelsContextMetadata",
    "FlextModelsContextScope",
    "FlextModelsContextTokens",
    "__scope_parts",
)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsContextData": "._data",
        "FlextModelsContextExport": "._export",
        "FlextModelsContextMetadata": "._metadata",
        "FlextModelsContextScope": ".__scope_parts.flextmodelscontextscope_part_03",
        "FlextModelsContextTokens": "._tokens",
        "__scope_parts": ".__scope_parts",
    }),
    public_exports=__all__,
)[0]
