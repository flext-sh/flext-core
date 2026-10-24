# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Context package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._context.__scope import (
        FlextModelsContextScopePart01,
        FlextModelsContextScopePart02,
        FlextModelsContextScopePart03,
    )
    from flext_core._models._context._data import FlextModelsContextData
    from flext_core._models._context._export import FlextModelsContextExport
    from flext_core._models._context._metadata import FlextModelsContextMetadata
    from flext_core._models._context._scope import FlextModelsContextScope
    from flext_core._models._context._tokens import FlextModelsContextTokens


__all__: tuple[str, ...] = (
    "FlextModelsContextData",
    "FlextModelsContextExport",
    "FlextModelsContextMetadata",
    "FlextModelsContextScope",
    "FlextModelsContextScopePart01",
    "FlextModelsContextScopePart02",
    "FlextModelsContextScopePart03",
    "FlextModelsContextTokens",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsContextData": "._data",
        "FlextModelsContextExport": "._export",
        "FlextModelsContextMetadata": "._metadata",
        "FlextModelsContextScope": "._scope",
        "FlextModelsContextScopePart01": ".__scope.flextmodelscontextscope_part_01",
        "FlextModelsContextScopePart02": ".__scope.flextmodelscontextscope_part_02",
        "FlextModelsContextScopePart03": ".__scope.flextmodelscontextscope_part_03",
        "FlextModelsContextTokens": "._tokens",
    }),
    public_exports=__all__,
)
