# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Context model sub-package — MRO-composed namespace modules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_core._models._context._data import *
    from flext_core._models._context._export import *
    from flext_core._models._context._metadata import *
    from flext_core._models._context._proxy_var import *
    from flext_core._models._context._scope import *
    from flext_core._models._context._tokens import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
