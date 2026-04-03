# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Context package."""

from __future__ import annotations

import typing as _t

from flext_core._models._context._data import FlextModelsContextData
from flext_core._models._context._export import FlextModelsContextExport
from flext_core._models._context._metadata import FlextModelsContextMetadata
from flext_core._models._context._proxy_var import FlextModelsContextProxyVar
from flext_core._models._context._scope import FlextModelsContextScope
from flext_core._models._context._tokens import FlextModelsContextTokens
from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import flext_core._models._context._data as _flext_core__models__context__data

    _data = _flext_core__models__context__data
    import flext_core._models._context._export as _flext_core__models__context__export

    _export = _flext_core__models__context__export
    import flext_core._models._context._metadata as _flext_core__models__context__metadata

    _metadata = _flext_core__models__context__metadata
    import flext_core._models._context._proxy_var as _flext_core__models__context__proxy_var

    _proxy_var = _flext_core__models__context__proxy_var
    import flext_core._models._context._scope as _flext_core__models__context__scope

    _scope = _flext_core__models__context__scope
    import flext_core._models._context._tokens as _flext_core__models__context__tokens

    _tokens = _flext_core__models__context__tokens

    _ = (
        FlextModelsContextData,
        FlextModelsContextExport,
        FlextModelsContextMetadata,
        FlextModelsContextProxyVar,
        FlextModelsContextScope,
        FlextModelsContextTokens,
        _data,
        _export,
        _metadata,
        _proxy_var,
        _scope,
        _tokens,
    )
_LAZY_IMPORTS = {
    "FlextModelsContextData": "flext_core._models._context._data",
    "FlextModelsContextExport": "flext_core._models._context._export",
    "FlextModelsContextMetadata": "flext_core._models._context._metadata",
    "FlextModelsContextProxyVar": "flext_core._models._context._proxy_var",
    "FlextModelsContextScope": "flext_core._models._context._scope",
    "FlextModelsContextTokens": "flext_core._models._context._tokens",
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
