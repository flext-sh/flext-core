"""Private context model submodules.

Imported internally by `flext_core._models.context`. Not part of the public
surface; consumers should use `flext_core.m.Context.*` facades instead.

Submodules are loaded lazily to avoid eager circular imports during
``flext_core`` package initialization.
"""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._data": ("FlextModelsContextData",),
        "._export": ("FlextModelsContextExport",),
        "._metadata": ("FlextModelsContextMetadata",),
        "._proxy_var": ("FlextModelsContextProxyVar",),
        "._scope": ("FlextModelsContextScope",),
        "._tokens": ("FlextModelsContextTokens",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
