# AUTO-GENERATED FILE — Regenerate with: make gen
"""Context package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._context.__scope_parts.flextmodelscontextscope_part_03 import (
        FlextModelsContextScope,
    )
    from flext_core._models._context._data import FlextModelsContextData
    from flext_core._models._context._export import FlextModelsContextExport
    from flext_core._models._context._metadata import FlextModelsContextMetadata
    from flext_core._models._context._proxy_var import FlextModelsContextProxyVar
    from flext_core._models._context._tokens import FlextModelsContextTokens
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".__scope_parts": ("__scope_parts",),
        ".__scope_parts.flextmodelscontextscope_part_03": ("FlextModelsContextScope",),
        "._data": ("FlextModelsContextData",),
        "._export": ("FlextModelsContextExport",),
        "._metadata": ("FlextModelsContextMetadata",),
        "._proxy_var": ("FlextModelsContextProxyVar",),
        "._tokens": ("FlextModelsContextTokens",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
