# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Context package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._context import __scope_parts
    from flext_core._models._context.__scope_parts.flextmodelscontextscope_part_03 import (
        FlextModelsContextScope,
    )
    from flext_core._models._context._data import FlextModelsContextData
    from flext_core._models._context._export import FlextModelsContextExport
    from flext_core._models._context._metadata import FlextModelsContextMetadata
    from flext_core._models._context._proxy_var import FlextModelsContextProxyVar
    from flext_core._models._context._tokens import FlextModelsContextTokens


__all__: tuple[str, ...] = (
    "FlextModelsContextData",
    "FlextModelsContextExport",
    "FlextModelsContextMetadata",
    "FlextModelsContextProxyVar",
    "FlextModelsContextScope",
    "FlextModelsContextTokens",
    "__scope_parts",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".__scope_parts": ("__scope_parts",),
            ".__scope_parts.flextmodelscontextscope_part_03": (
                "FlextModelsContextScope",
            ),
            "._data": ("FlextModelsContextData",),
            "._export": ("FlextModelsContextExport",),
            "._metadata": ("FlextModelsContextMetadata",),
            "._proxy_var": ("FlextModelsContextProxyVar",),
            "._tokens": ("FlextModelsContextTokens",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
