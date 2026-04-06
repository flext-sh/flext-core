# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Context package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import flext_core._models._context._data as _flext_core__models__context__data

    _data = _flext_core__models__context__data
    import flext_core._models._context._export as _flext_core__models__context__export
    from flext_core._models._context._data import FlextModelsContextData

    _export = _flext_core__models__context__export
    import flext_core._models._context._metadata as _flext_core__models__context__metadata
    from flext_core._models._context._export import FlextModelsContextExport

    _metadata = _flext_core__models__context__metadata
    import flext_core._models._context._proxy_var as _flext_core__models__context__proxy_var
    from flext_core._models._context._metadata import FlextModelsContextMetadata

    _proxy_var = _flext_core__models__context__proxy_var
    import flext_core._models._context._scope as _flext_core__models__context__scope
    from flext_core._models._context._proxy_var import FlextModelsContextProxyVar

    _scope = _flext_core__models__context__scope
    import flext_core._models._context._tokens as _flext_core__models__context__tokens
    from flext_core._models._context._scope import FlextModelsContextScope

    _tokens = _flext_core__models__context__tokens
    from flext_core._models._context._tokens import FlextModelsContextTokens
_LAZY_IMPORTS = {
    "FlextModelsContextData": (
        "flext_core._models._context._data",
        "FlextModelsContextData",
    ),
    "FlextModelsContextExport": (
        "flext_core._models._context._export",
        "FlextModelsContextExport",
    ),
    "FlextModelsContextMetadata": (
        "flext_core._models._context._metadata",
        "FlextModelsContextMetadata",
    ),
    "FlextModelsContextProxyVar": (
        "flext_core._models._context._proxy_var",
        "FlextModelsContextProxyVar",
    ),
    "FlextModelsContextScope": (
        "flext_core._models._context._scope",
        "FlextModelsContextScope",
    ),
    "FlextModelsContextTokens": (
        "flext_core._models._context._tokens",
        "FlextModelsContextTokens",
    ),
    "_data": "flext_core._models._context._data",
    "_export": "flext_core._models._context._export",
    "_metadata": "flext_core._models._context._metadata",
    "_proxy_var": "flext_core._models._context._proxy_var",
    "_scope": "flext_core._models._context._scope",
    "_tokens": "flext_core._models._context._tokens",
}

__all__ = [
    "FlextModelsContextData",
    "FlextModelsContextExport",
    "FlextModelsContextMetadata",
    "FlextModelsContextProxyVar",
    "FlextModelsContextScope",
    "FlextModelsContextTokens",
    "_data",
    "_export",
    "_metadata",
    "_proxy_var",
    "_scope",
    "_tokens",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
