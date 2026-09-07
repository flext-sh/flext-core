# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Context package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import __scope_parts as __scope_parts
    from ._data import FlextModelsContextData
    from ._export import FlextModelsContextExport
    from ._metadata import FlextModelsContextMetadata
    from ._proxy_var import FlextModelsContextProxyVar
    from ._scope import FlextModelsContextScope
    from ._tokens import FlextModelsContextTokens
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
            "._data": ("FlextModelsContextData",),
            "._export": ("FlextModelsContextExport",),
            "._metadata": ("FlextModelsContextMetadata",),
            "._proxy_var": ("FlextModelsContextProxyVar",),
            "._scope": ("FlextModelsContextScope",),
            "._tokens": ("FlextModelsContextTokens",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
